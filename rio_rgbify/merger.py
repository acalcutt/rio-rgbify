import sqlite3
import rasterio
import mercantile
from rasterio.warp import reproject, Resampling
import numpy as np
import io
from PIL import Image
import multiprocessing
from multiprocessing import Pool, Process, Queue
from pathlib import Path
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from typing import Optional, Tuple, List, Dict
from contextlib import contextmanager
from rio_rgbify.database import MBTilesDatabase
from rio_rgbify.image import ImageFormat, ImageEncoder
from queue import Queue

class EncodingType(Enum):
    MAPBOX = "mapbox"
    TERRARIUM = "terrarium"

@dataclass
class MBTilesSource:
    """Configuration for an MBTiles source file"""
    path: Path
    encoding: EncodingType
    height_adjustment: float = 0.0 # Added height adjustment
    base_val: float = -10000 # Add base val, with default of -10000 for mapbox
    interval: float = 0.1 # Add interval with default of 0.1 for mapbox
    mask_values: list = field(default_factory=lambda: [0.0])

    def __post_init__(self):
        if not self.path.exists():
            raise ValueError(f"Source file does not exist: {self.path}")


@dataclass
class TileData:
    """Container for decoded tile data"""
    data: np.ndarray
    meta: dict
    source_zoom: int


def process_tile_task(task_args: Tuple) -> Optional[Tuple[mercantile.Tile, bytes]]:
    """Standalone function for processing tiles that can be pickled"""
    tile, sources_config, output_path_str, output_encoding_str, resampling, output_image_format_str, \
    output_quantized_alpha, source_encodings_str, height_adjustments, base_vals, intervals, mask_values = task_args
    logging.info(f"process_tile_task: Starting for tile {tile.z}/{tile.x}/{tile.y}")
    try:
        # Reconstruct MBTilesSource objects
        sources = [
            MBTilesSource(
                path=Path(path),
                encoding=EncodingType(encoding),
                height_adjustment=height_adj,
                base_val=base_val,
                interval=interval,
                mask_values=mask_vals
            )
            for path, encoding, height_adj, base_val, interval, mask_vals in zip(
                sources_config,
                source_encodings_str,
                height_adjustments,
                base_vals,
                intervals,
                mask_values
            )
        ]

        # Create merger instance (without the write queue)
        merger = TerrainRGBMerger(
            sources=sources,
            output_path=Path(output_path_str),
            output_encoding=EncodingType(output_encoding_str),
            resampling=resampling,
            processes=1,
            default_tile_size=512,
            output_image_format=ImageFormat(output_image_format_str),
            output_quantized_alpha=output_quantized_alpha,
            min_zoom=tile.z,
            max_zoom=tile.z,
            bounds=None
        )

        # Process the tile
        source_conns = {
            source.path: sqlite3.connect(source.path)
            for source in sources
        }

        merged_data = merger.process_tile_data(tile, source_conns)

        # Clean up connections
        for conn in source_conns.values():
            conn.close()

        logging.info(f"process_tile_task: Finished for tile {tile.z}/{tile.x}/{tile.y}")
        return tile, merged_data

    except Exception as e:
        logging.error(f"process_tile_task: Error processing tile {tile.z}/{tile.x}/{tile.y}: {e}")
        return None

class TerrainRGBMerger:
    """
    A class to merge multiple Terrain RGB MBTiles files.
    """
    def __init__(
        self,
        sources: List[MBTilesSource],
        output_path: Path,
        output_encoding: EncodingType = EncodingType.MAPBOX,
        resampling: int = Resampling.lanczos,
        processes: Optional[int] = None,
        default_tile_size: int = 512,
        output_image_format: ImageFormat = ImageFormat.PNG,
        output_quantized_alpha: bool = False,
        min_zoom: int = 0,
        max_zoom: Optional[int] = None,
        bounds: Optional[List[float]] = None
    ):
        """
        Initializes the TerrainRGBMerger.

        Parameters
        ----------
        sources : List[MBTilesSource]
            A list of MBTiles source configurations.
        output_path : Path
            The path to the output MBTiles file.
        output_encoding : EncodingType, optional
            The encoding for the output tiles. Defaults to EncodingType.MAPBOX.
        resampling : int, optional
            The resampling method to use during tile merging. Defaults to Resampling.lanczos.
        processes : Optional[int], optional
            The number of processes to use for parallel processing. Defaults to multiprocessing.cpu_count().
        default_tile_size : int, optional
            The default tile size in pixels. Defaults to 512.
        output_image_format : ImageFormat, optional
            The output image format of the tiles. Defaults to ImageFormat.PNG
        output_quantized_alpha : bool, optional
            If set to true and using terrarium output encoding, the alpha channel will be populated with quantized data
        min_zoom : int, optional
            The minimum zoom level to process tiles, defaults to 0.
        max_zoom : Optional[int], optional
            The maximum zoom level to process tiles, if None, we use the maximum available, defaults to None.
        bounds : Optional[List[float]], optional
            The bounding box to limit the tiles being generated, defaults to None. If None, the bounds of the last source will be used.
        """
        print(f"__init__ called")
        self.sources = sources
        self.output_path = Path(output_path)
        self.output_encoding = output_encoding
        self.resampling = resampling
        self.processes = processes or multiprocessing.cpu_count()
        self.logger = logging.getLogger(__name__)
        self.default_tile_size = default_tile_size
        self.output_image_format = output_image_format
        self.output_quantized_alpha = output_quantized_alpha
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.bounds = bounds

    def _decode_tile(self, tile_data: bytes, tile: mercantile.Tile, encoding: EncodingType, source: MBTilesSource) -> Tuple[Optional[np.ndarray], dict]:
        if not isinstance(tile_data, bytes) or len(tile_data) == 0:
            raise ValueError("Invalid tile data")

        try:
            image = Image.open(io.BytesIO(tile_data))
            image = image.convert('RGB')
            image_png = io.BytesIO()
            image.save(image_png, format='PNG', bits=8)
            image_png.seek(0)

            with rasterio.open(image_png) as dataset:
                rgb = dataset.read(masked=False).astype(np.int32)

                if rgb.ndim != 3 or rgb.shape[0] != 3:
                    self.logger.error(f"Unexpected RGB shape in tile {tile.z}/{tile.x}/{tile.y}: {rgb.shape}")
                    return None, {}

                elevation = ImageEncoder._decode(rgb, source.base_val, source.interval, encoding.value)
                elevation = ImageEncoder._mask_elevation(elevation, source.mask_values)
                elevation += source.height_adjustment

                bounds = mercantile.bounds(tile)
                meta = dataset.meta.copy()
                meta.update({
                    'count': 1,
                    'dtype': rasterio.float32,
                    'driver': 'GTiff',
                    'crs': 'EPSG:3857',
                    'transform': rasterio.transform.from_bounds(
                        bounds.west, bounds.south, bounds.east, bounds.north,
                        meta['width'], meta['height']
                    )
                })
                return elevation, meta
        except Exception as e:
            self.logger.error(f"Failed to decode tile data for {tile.z}/{tile.x}/{tile.y}: {e}")
            return None, {}

    def _extract_tile(self, source: MBTilesSource, zoom: int, x: int, y: int, source_conns: Dict[Path, sqlite3.Connection]) -> Optional[TileData]:
        current_zoom = zoom
        current_x, current_y = x, y

        while current_zoom >= 0:
            conn = source_conns[source.path]
            cursor = conn.cursor()
            cursor.execute(
                "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                (current_zoom, current_x, current_y)
            )
            result = cursor.fetchone()

            if result is not None:
                try:
                    data_meta = self._decode_tile(result[0], mercantile.Tile(current_x, current_y, current_zoom), source.encoding, source)
                    if data_meta[0] is None:
                        return None
                    if data_meta[0].size == 0:
                       return None
                    return TileData(data_meta[0], data_meta[1], current_zoom)
                except Exception as e:
                    self.logger.error(f"Failed to decode tile for zoom {current_zoom}, x {current_x}, y {current_y}: {e}")
                    return None

            if current_zoom > 0:
                current_x //= 2
                current_y //= 2
            current_zoom -= 1

        return None

    def _merge_tiles(self, tile_datas: List[Optional[TileData]], target_tile: mercantile.Tile) -> Optional[np.ndarray]:
        if not any(tile_datas):
            return None

        result = None
        for i, tile_data in enumerate(tile_datas):
            if tile_data is not None:
                resampled_data = self._resample_if_needed(tile_data, target_tile)

                if result is None:
                    result = resampled_data
                else:
                    mask = ~np.isnan(resampled_data)
                    if np.any(mask):
                        if resampled_data.ndim == result.ndim:
                            result[mask] = resampled_data[mask]
                        elif resampled_data.ndim > result.ndim:
                            result[mask] = resampled_data[mask][..., :result.shape[-1]]
                        else:
                            result[mask] = resampled_data[mask][..., np.newaxis]

        return result

    def _resample_if_needed(self, tile_data: TileData, target_tile: mercantile.Tile) -> np.ndarray:
        if tile_data.source_zoom == target_tile.z:
            if tile_data.data.ndim == 3:
                return tile_data.data[0]
            else:
                return tile_data.data
        else:
            source_tile = mercantile.Tile(x=target_tile.x // (2**(target_tile.z - tile_data.source_zoom)),
                                            y=target_tile.y // (2**(target_tile.z - tile_data.source_zoom)),
                                            z=tile_data.source_zoom
                                            )
            source_bounds = mercantile.bounds(source_tile)

            x_offset = (target_tile.x % (2**(target_tile.z - tile_data.source_zoom)))
            y_offset = (target_tile.y % (2**(target_tile.z - tile_data.source_zoom)))

            sub_region_width = (source_bounds.east - source_bounds.west) / (2**(target_tile.z - tile_data.source_zoom))
            sub_region_height = (source_bounds.north - source_bounds.south) / (2**(target_tile.z - tile_data.source_zoom))

            sub_region_west = source_bounds.west + (x_offset * sub_region_width)
            sub_region_south = source_bounds.south + (y_offset * sub_region_height)
            sub_region_east = sub_region_west + sub_region_width
            sub_region_north = sub_region_south + sub_region_height

            tile_size = self.default_tile_size
            if tile_data.meta is not None and 'width' in tile_data.meta and 'height' in tile_data.meta:
                tile_size = tile_data.meta['width']

            sub_region_transform = rasterio.transform.from_bounds(sub_region_west, sub_region_south, sub_region_east, sub_region_north, tile_size, tile_size)

            with rasterio.io.MemoryFile() as memfile:
                with memfile.open(**tile_data.meta) as src:
                    dst_data = np.zeros((1, tile_size, tile_size), dtype=np.float32)
                    reproject(
                        source=tile_data.data,
                        destination=dst_data,
                        src_transform=tile_data.meta['transform'],
                        src_crs=tile_data.meta['crs'],
                        dst_transform=sub_region_transform,
                        dst_crs=tile_data.meta['crs'],
                        resampling=self.resampling
                    )

                    if dst_data.ndim == 3:
                        return dst_data[0]
                    else:
                        return dst_data

    def process_tile_data(self, tile: mercantile.Tile, source_conns: Dict[Path, sqlite3.Connection]) -> Optional[bytes]:
        try:
            tile_datas = [self._extract_tile(source, tile.z, tile.x, tile.y, source_conns) for source in self.sources]

            if not any(tile_datas):
                self.logger.debug(f"No data found for tile {tile.z}/{tile.x}/{tile.y}")
                return None

            merged_elevation = self._merge_tiles(tile_datas, tile)

            if merged_elevation is not None:
                rgb_data = ImageEncoder.data_to_rgb(
                    merged_elevation,
                    self.output_encoding,
                    0.1,
                    base_val=-10000,
                    quantized_alpha=self.output_quantized_alpha if self.output_encoding == EncodingType.TERRARIUM else False
                )
                image_bytes = ImageEncoder.save_rgb_to_bytes(rgb_data, self.output_image_format, self.default_tile_size)
                self.logger.info(f"Successfully processed tile data for {tile.z}/{tile.x}/{tile.y}")
                return image_bytes
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error processing tile data for {tile.z}/{tile.x}/{tile.y}: {e}")
            return None

    def _get_tiles_for_zoom(self, zoom: int) -> List[mercantile.Tile]:
        """Get list of tiles to process for a given zoom level

        Parameters
        ----------
        zoom : int
            The zoom level for which to get the tiles

        Returns
        -------
        List[mercantile.Tile]
            A list of mercantile.Tile objects for the given zoom level
        """
        print(f"_get_tiles_for_zoom called with zoom: {zoom}")
        self.logger.info(f"_get_tiles_for_zoom: Starting for zoom {zoom}")
        tiles = set()

        if self.bounds is not None:
            self.logger.info(f"_get_tiles_for_zoom: Using bounds: {self.bounds}")
            w,s,e,n = self.bounds
            for x, y in _tile_range(mercantile.tile(w, n, zoom), mercantile.tile(e, s, zoom)):
                tiles.add(mercantile.Tile(x=x, y=y, z=zoom))
        else:
            self.logger.info(f"_get_tiles_for_zoom: Getting tiles from the last source")
            source = self.sources[-1]
            self.logger.info(f"_get_tiles_for_zoom: Connecting to database: {source.path}")
            mbtiles_db = MBTilesDatabase(source.path)
            self.logger.info(f"_get_tiles_for_zoom: Connected to database")
            self.logger.info(f"_get_tiles_for_zoom: Getting distinct tiles for zoom {zoom}")
            rows = mbtiles_db.get_distinct_tiles(zoom)
            self.logger.info(f"_get_tiles_for_zoom: Received {len(rows)} rows from get_distinct_tiles")

            if not rows:
                self.logger.warning(f"_get_tiles_for_zoom: No tiles found for zoom level {zoom} in source {source.path}")
            else:
                #self.logger.debug(f"Rows fetched for zoom level : ")
                for row in rows:
                    self.logger.debug(f"_get_tiles_for_zoom: Processing row: {row}")
                    if isinstance(row, tuple) and len(row) == 2:
                        x, y = row
                        tiles.add(mercantile.Tile(x=x, y=y, z=zoom))
                    else:
                        self.logger.warning(f"_get_tiles_for_zoom: Skipping invalid row: {row}")

        self.logger.info(f"_get_tiles_for_zoom: Found {len(tiles)} tiles for zoom {zoom}")
        return list(tiles)

    def _writer_process(self, write_queue: Queue):
        """Writes to the db using a shared queue"""
        logging.info("Writer process started")
        with MBTilesDatabase(self.output_path) as db:
            while True:
                item = write_queue.get()
                if item is None:
                    logging.info("Writer process received None, exiting")
                    break
                tile, image_bytes = item
                db.insert_tile([tile.x, tile.y, tile.z], image_bytes)
                write_queue.task_done()
                logging.info(f"Writer process: wrote tile {tile.z}/{tile.x}/{tile.y}")
        logging.info("Writer process finished")

    def process_zoom_level(self, zoom: int):
        print(f"process_zoom_level called with zoom: {zoom}")
        self.logger.info(f"Processing zoom level {zoom}")

        tiles = self._get_tiles_for_zoom(zoom)
        self.logger.info(f"Found {len(tiles)} tiles to process for zoom {zoom}")

        write_queue = Queue()
        writer_process = multiprocessing.Process(target=self._writer_process, args=(write_queue,))
        writer_process.start()

        tasks = [
            (
                tile,
                [source.path for source in self.sources],
                str(self.output_path),
                self.output_encoding.value,
                self.resampling,
                self.output_image_format.value,
                self.output_quantized_alpha,
                [source.encoding.value for source in self.sources],
                [source.height_adjustment for source in self.sources],
                [source.base_val for source in self.sources],
                [source.interval for source in self.sources],
                [source.mask_values for source in self.sources],
            )
            for tile in tiles
        ]

        logging.info(f"Starting multiprocessing pool for zoom {zoom} with {self.processes} processes")
        with multiprocessing.Pool(self.processes) as pool:
            for result in pool.imap_unordered(process_tile_task, tasks, chunksize=1):
                if result:
                    tile, image_bytes = result
                    write_queue.put((tile, image_bytes))
        logging.info(f"Finished multiprocessing pool for zoom {zoom}")

        write_queue.put(None)
        write_queue.join()
        writer_process.join()

    def get_max_zoom_level(self) -> int:
        print("_get_max_zoom_level called")
        source = self.sources[-1]
        mbtiles_db = MBTilesDatabase(source.path)
        max_zoom = mbtiles_db.get_max_zoom_level()
        return max_zoom

    def process_all(self, min_zoom: int = 0):
        print(f"process_all called with min_zoom: {min_zoom}")
        max_zoom = self.max_zoom if self.max_zoom is not None else self.get_max_zoom_level()
        self.logger.info(f"Processing zoom levels {min_zoom} to {max_zoom}")

        for zoom in range(min_zoom, max_zoom + 1):
            self.process_zoom_level(zoom)

        self.logger.info("Completed processing all zoom levels")

def _tile_range(start: mercantile.Tile, stop: mercantile.Tile):
    for x in range(start.x, stop.x + 1):
        for y in range(start.y, stop.y + 1):
            yield x, y

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Example usage (replace with your actual setup)
    sources = [
        MBTilesSource(path=Path("/opt/swissALTI3D_2024_terrainrgb_z0-Z16.mbtiles"), encoding=EncodingType.MAPBOX)
        # Add more sources if needed
    ]
    merger = TerrainRGBMerger(sources=sources, output_path=Path("/opt/output.mbtiles"), processes=12)
    merger.process_all()
