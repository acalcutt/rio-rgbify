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
class ProcessTileArgs:
    """Arguments for parallel tile processing"""
    tile: mercantile.Tile
    sources: List[MBTilesSource]
    output_path: Path
    output_encoding: EncodingType
    resampling: int
    output_image_format: ImageFormat
    output_quantized_alpha: bool


@dataclass
class TileData:
    """Container for decoded tile data"""
    data: np.ndarray
    meta: dict
    source_zoom: int

@dataclass
class TileProcessingTask:
    """Container for all data needed to process a tile"""
    tile: mercantile.Tile
    sources: List[Path]  # Store paths instead of MBTilesSource objects
    output_path: Path
    output_encoding: str  # Store as string instead of EncodingType
    resampling: int
    output_image_format: str  # Store as string instead of ImageFormat
    output_quantized_alpha: bool
    source_encodings: List[str]  # Store encodings as strings
    height_adjustments: List[float]
    base_vals: List[float]
    intervals: List[float]
    mask_values: List[List[float]]

def process_tile_task(task_and_queue: Tuple[TileProcessingTask, Queue]) -> None:
    """Standalone function for processing tiles that can be pickled"""
    task, write_queue = task_and_queue
    
    try:
        # Reconstruct MBTilesSource objects
        sources = [
            MBTilesSource(
                path=path,
                encoding=EncodingType(encoding),
                height_adjustment=height_adj,
                base_val=base_val,
                interval=interval,
                mask_values=mask_vals
            )
            for path, encoding, height_adj, base_val, interval, mask_vals in zip(
                task.sources,
                task.source_encodings,
                task.height_adjustments,
                task.base_vals,
                task.intervals,
                task.mask_values
            )
        ]
        
        # Create merger instance
        merger = TerrainRGBMerger(
            sources=sources,
            output_path=task.output_path,
            output_encoding=EncodingType(task.output_encoding),
            resampling=task.resampling,
            processes=1,
            default_tile_size=512,
            output_image_format=ImageFormat(task.output_image_format),
            output_quantized_alpha=task.output_quantized_alpha,
            min_zoom=task.tile.z,
            max_zoom=task.tile.z,
            bounds=None
        )
        
        # Process the tile
        source_conns = {
            source.path: sqlite3.connect(source.path)
            for source in sources
        }
        
        merger.process_tile(task.tile, source_conns, write_queue)
        
        # Clean up connections
        for conn in source_conns.values():
            conn.close()
            
    except Exception as e:
        logging.error(f"Error processing tile {task.tile}: {e}")
        raise

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
        self.write_queue = Queue() # initialize the shared queue
    
    def _decode_tile(self, tile_data: bytes, tile: mercantile.Tile, encoding: EncodingType, source: MBTilesSource) -> Tuple[Optional[np.ndarray], dict]:
        """
        Decode tile data using specified encoding format

        Parameters
        ----------
        tile_data : bytes
            The raw tile data.
        tile : mercantile.Tile
            The mercantile tile object.
        encoding : EncodingType
            The encoding used for the tile.
        source : MBTilesSource
            The MBTiles source.

        Returns
        -------
        Tuple[Optional[np.ndarray], dict]
            A tuple containing the decoded elevation data and metadata, or None, None if decoding fails.
        """
        if not isinstance(tile_data, bytes) or len(tile_data) == 0:
            raise ValueError("Invalid tile data")
            
        try:
            # Log the size of the tile data to make sure it's non-empty
            #self.logger.debug(f"Tile data size for {tile.z}/{tile.x}/{tile.y}: {len(tile_data)} bytes")
            
            # Convert the image to a PNG using Pillow
            image = Image.open(io.BytesIO(tile_data))
            image = image.convert('RGB')  # Force to RGB
            image_png = io.BytesIO()
            image.save(image_png, format='PNG', bits=8)
            image_png.seek(0)
            
            with rasterio.open(image_png) as dataset:
                # Check if we can read data properly
                rgb = dataset.read(masked=False).astype(np.int32)
                #self.logger.debug(f"Decoded tile RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
                
                if rgb.ndim != 3 or rgb.shape[0] != 3:
                    self.logger.error(f"Unexpected RGB shape in tile {tile.z}/{tile.x}/{tile.y}: {rgb.shape}")
                    return None, {}

                elevation = ImageEncoder._decode(rgb, source.base_val, source.interval, encoding.value) # Use the static decode method from the encoder
                elevation = ImageEncoder._mask_elevation(elevation, source.mask_values)

                #Apply height adjustment
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
                
                #self.logger.debug(f"Decoded elevation: min={np.nanmin(elevation)}, max={np.nanmax(elevation)}")
                return elevation, meta
        except Exception as e:
            self.logger.error(f"Failed to decode tile data, returning None, None: {e}")
            return None, {}

    def _extract_tile(self, source: MBTilesSource, zoom: int, x: int, y: int, source_conns: Dict[Path, sqlite3.Connection]) -> Optional[TileData]:
        """Extract and decode a tile, with fallback to parent tiles
        
        Parameters
        ----------
        source : MBTilesSource
            The MBTiles source to use
        zoom : int
            The zoom level of the tile
        x : int
            The x index of the tile
        y : int
            The y index of the tile

        Returns
        -------
        Optional[TileData]
            TileData object or None if it cannot be extracted.
        """
        #print(f"_extract_tile called with source: {source}, zoom: {zoom}, x: {x}, y: {y}")
        current_zoom = zoom
        current_x, current_y = x, y
        
        while current_zoom >= 0:
          conn = source_conns[source.path] # get the database connection from the dictionary
          cursor = conn.cursor()
          cursor.execute(
                "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                (current_zoom, current_x, current_y)
            )
          result = cursor.fetchone()
            
          if result is not None:
              try:
                  data_meta = self._decode_tile(result[0], mercantile.Tile(current_x, current_y, current_zoom), source.encoding, source) #pass in source
                  #self.logger.debug(f"decoded data for {current_zoom}/{current_x}/{current_y}: data is {data_meta[0] is None}, meta is {data_meta[1] is None}")
                  if data_meta[0] is None:
                      return None
                  if data_meta[0].size == 0:
                    return None
                  return TileData(data_meta[0], data_meta[1], current_zoom)
              except Exception as e:
                  self.logger.error(f"Failed to decode tile {current_zoom}/{current_x}/{current_y}: {e}")
                  return None
            
          if current_zoom > 0:
                current_x //= 2
                current_y //= 2
          current_zoom -= 1
        
        return None

    def _merge_tiles(self, tile_datas: List[Optional[TileData]], target_tile: mercantile.Tile) -> Optional[np.ndarray]:
        """Merge tiles from multiple sources, handling upscaling and priorities
        
        Parameters
        ----------
        tile_datas : List[Optional[TileData]]
            A list of TileData objects
        target_tile : mercantile.Tile
            The mercantile tile object we are merging into
        
        Returns
        -------
        Optional[np.ndarray]
            The merged elevation array, or None if no valid tiles to merge.
        """
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
        """Resample tile data if source zoom differs from target
        
        Parameters
        ----------
        tile_data : TileData
            The tile data to resample
        target_tile : mercantile.Tile
            The mercantile tile object for the resampled data

        Returns
        -------
        np.ndarray
            The resampled elevation array.
        """
        #print(f"_resample_if_needed called with tile_data: {tile_data}, target_tile: {target_tile}")
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

    def process_tile(self, tile: mercantile.Tile, source_conns: Dict[Path, sqlite3.Connection], write_queue: Queue) -> None:
        """Process a single tile, merging data from multiple sources"""
        #print(f"process_tile called with tile: {tile}")
        try:
            # Extract tiles from all sources
            tile_datas = [self._extract_tile(source, tile.z, tile.x, tile.y, source_conns) for source in self.sources]

            if not any(tile_datas):
                self.logger.debug(f"No data found for tile {tile.z}/{tile.x}/{tile.y}")
                return

            # Merge the elevation data
            merged_elevation = self._merge_tiles(tile_datas, tile)
            
            if merged_elevation is not None:
                # Encode using output format and save
                rgb_data = ImageEncoder.data_to_rgb(merged_elevation, self.output_encoding, 0.1, base_val=-10000, quantized_alpha=self.output_quantized_alpha if self.output_encoding == EncodingType.TERRARIUM else False) # Use the encoder from the encoders.py file
                image_bytes = ImageEncoder.save_rgb_to_bytes(rgb_data, self.output_image_format, self.default_tile_size)
                
                write_queue.put((tile, image_bytes)) # Write to the queue
                self.logger.info(f"Successfully processed tile {tile.z}/{tile.x}/{tile.y}")
        except Exception as e:
            self.logger.error(f"Error processing tile {tile.z}/{tile.x}/{tile.y}: {e}")
            raise
        
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
        tiles = set()
        
        if self.bounds is not None:
          w,s,e,n = self.bounds
          for x, y in _tile_range(mercantile.tile(w, n, zoom), mercantile.tile(e, s, zoom)):
            tiles.add(mercantile.Tile(x=x, y=y, z=zoom))
        else:
            # Get tiles from the LAST source
            source = self.sources[-1]
            mbtiles_db = MBTilesDatabase(source.path)
            rows = mbtiles_db.get_distinct_tiles(zoom)
            
            if not rows:
                self.logger.warning(f"No tiles found for zoom level {zoom} in source {source.path}")
            else:
                #self.logger.debug(f"Rows fetched for zoom level {zoom}: {rows}")
                for row in rows:
                    if isinstance(row, tuple) and len(row) == 2:
                        x, y = row
                        tiles.add(mercantile.Tile(x=x, y=y, z=zoom))
                    else:
                        self.logger.warning(f"Skipping invalid row: {row}")
        
        return list(tiles)

    @staticmethod
    def _process_tile_with_queue(args_and_queue: tuple) -> None:
        """Static wrapper method for parallel tile processing that includes queue
        
        Parameters
        ----------
        args_and_queue : tuple
            Tuple containing (ProcessTileArgs, Queue)
        """
        args, write_queue = args_and_queue
        try:
            merger = TerrainRGBMerger(
                sources=args.sources,
                output_path=args.output_path,
                output_encoding=args.output_encoding,
                resampling=args.resampling,
                processes=1,
                default_tile_size=512,
                output_image_format=args.output_image_format,
                output_quantized_alpha=args.output_quantized_alpha,
                min_zoom=args.tile.z,
                max_zoom=args.tile.z,
                bounds=None
            )
            source_conns = {}
            for source in args.sources:
                source_conns[source.path] = sqlite3.connect(source.path)
            merger.process_tile(args.tile, source_conns, write_queue)
            for conn in source_conns.values():
                conn.close()
        except Exception as e:
            logging.error(f"Error processing tile {args.tile}: {e}")

    def process_zoom_level(self, zoom: int):
        """Process all tiles for a given zoom level in parallel"""
        print(f"process_zoom_level called with zoom: {zoom}")
        self.logger.info(f"Processing zoom level {zoom}")
        
        # Get list of tiles to process
        tiles = self._get_tiles_for_zoom(zoom)
        self.logger.info(f"Found {len(tiles)} tiles to process")
        
        # Create processing tasks with all necessary data
        processing_tasks = [
            TileProcessingTask(
                tile=tile,
                sources=[source.path for source in self.sources],
                output_path=self.output_path,
                output_encoding=self.output_encoding.value,
                resampling=self.resampling,
                output_image_format=self.output_image_format.value,
                output_quantized_alpha=self.output_quantized_alpha,
                source_encodings=[source.encoding.value for source in self.sources],
                height_adjustments=[source.height_adjustment for source in self.sources],
                base_vals=[source.base_val for source in self.sources],
                intervals=[source.interval for source in self.sources],
                mask_values=[source.mask_values for source in self.sources]
            )
            for tile in tiles
        ]
        
        # Pair tasks with queue
        tasks_with_queue = [(task, self.write_queue) for task in processing_tasks]
        
        # Create writer process
        writer_process = Process(target=self._writer_process, args=(self.write_queue,))
        writer_process.start()
        
        # Process tiles in parallel
        with Pool(self.processes) as pool:
            for _ in pool.imap_unordered(process_tile_task, tasks_with_queue, chunksize=1):
                pass
                
        # Clean up
        self.write_queue.put(None)
        self.write_queue.join()
        writer_process.join()

    def _writer_process(self, write_queue: Queue):
        """Writes to the db using a shared queue"""
        with MBTilesDatabase(self.output_path) as db:
            while True:
                item = write_queue.get()
                if item is None:
                    break
                tile, image_bytes = item
                db.insert_tile(tile = [tile.x, tile.y, tile.z], contents = image_bytes)
                write_queue.task_done()

    def process_zoom_level(self, zoom: int):
        """Process all tiles for a given zoom level in parallel
        
        Parameters
        ----------
        zoom : int
            The zoom level to process
        """
        print(f"process_zoom_level called with zoom: {zoom}")
        self.logger.info(f"Processing zoom level {zoom}")
        
        # Get list of tiles to process
        tiles = self._get_tiles_for_zoom(zoom)
        self.logger.info(f"Found {len(tiles)} tiles to process")
        
        # Prepare arguments for parallel processing
        process_args = [
            ProcessTileArgs(
                tile=tile,
                sources=self.sources,
                output_path=self.output_path,
                output_encoding=self.output_encoding,
                resampling=self.resampling,
                output_image_format = self.output_image_format,
                output_quantized_alpha=self.output_quantized_alpha,
            )
            for tile in tiles
        ]
        
        # Create the writer process
        writer_process = multiprocessing.Process(target=self._writer_process, args=(self.write_queue,))
        writer_process.start()

        # Process tiles in parallel - Fixed the imap_unordered call
        with multiprocessing.Pool(self.processes) as pool:
            for _ in pool.imap_unordered(
                lambda x: self._process_tile_wrapper(x, self.write_queue), 
                process_args,
                chunksize=1
            ):
                pass
        
        # Put the None value to stop the writer and wait for queue to empty
        self.write_queue.put(None)
        self.write_queue.join()
        writer_process.join()

    def get_max_zoom_level(self) -> int:
        """Get the maximum zoom level from the last source
        
        Returns
        -------
        int
            The maximum zoom level found in the last source.
        """
        print("_get_max_zoom_level called")
        source = self.sources[-1]
        mbtiles_db = MBTilesDatabase(source.path)
        max_zoom = mbtiles_db.get_max_zoom_level()
        return max_zoom

    def process_all(self, min_zoom: int = 0):
        """Process all zoom levels from min_zoom to max available
        
        Parameters
        ----------
        min_zoom : int, optional
            The minimum zoom level to start processing at. Defaults to 0
        """
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
