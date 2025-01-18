import sqlite3
import rasterio
import mercantile
from rasterio.warp import reproject, Resampling
import numpy as np
import io
from PIL import Image
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
import functools
from scipy.ndimage import gaussian_filter # Import gaussian filter
import time
import multiprocessing #Import the multiprocessing library

def retry(attempts, base_delay=1, max_delay=10):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(attempts):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    last_exception = e
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logging.warning(f"Database locked, retry attempt {attempt+1} after {delay} seconds...")
                    time.sleep(delay)

            if last_exception:
                logging.error(f"Failed after {attempts} attempts, raising last exception")
                raise last_exception
            return None
        return wrapper
    return decorator

class EncodingType(Enum):
    MAPBOX = "mapbox"
    TERRARIUM = "terrarium"

@dataclass
class RasterSource:
    """Configuration for an Raster source file"""
    path: Path
    height_adjustment: float = 0.0
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
    

class RasterRGBMerger:
    """
    A class to merge multiple Terrain RGB Raster files.
    """
    def __init__(self, sources, output_path, output_encoding=EncodingType.MAPBOX,
                 resampling=Resampling.lanczos, processes=None, default_tile_size=512,
                 output_image_format=ImageFormat.PNG, output_quantized_alpha=False,
                 min_zoom=0, max_zoom=None, bounds=None, gaussian_blur_sigma=0.2, base_val=-10000, interval=0.1): # Add gaussian_blur_sigma
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
        self.write_queue = Queue()
        self.gaussian_blur_sigma = gaussian_blur_sigma # Store the sigma for gaussian blur
        self.base_val = base_val # Store the base_val for mapbox output
        self.interval = interval # store the interval for mapbox output
        """
        Initializes the RasterRGBMerger.

        Parameters
        ----------
        sources : List[RasterSource]
            A list of Raster source configurations.
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
        gaussian_blur_sigma: float
            The sigma value to use for the gaussian blur filter, defaults to 0.2
        base_val: float, optional
            The base value for encoding mapbox output
        interval: float, optional
           The interval for encoding mapbox output
        """

    def _extract_tile(self, source: RasterSource, tile: mercantile.Tile, source_index: int, verbose=False) -> Optional[TileData]:
      """Extract and decode a tile from a Raster, with no fallback to parent tiles"""
      
      bounds = mercantile.bounds(tile)
      
      try:
        with rasterio.open(source.path) as src:
            
            window = rasterio.windows.from_bounds(*bounds, transform=src.transform)
            
            data = src.read(window=window, masked=True).astype(np.float32)
            
            if data.ndim == 3:
                data = data[0] #Only use the first band

            data = ImageEncoder._mask_elevation(data, source.mask_values)
            
            #Apply height adjustment
            data += source.height_adjustment
            
            meta = src.meta.copy()
            
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

            return TileData(data, meta, tile.z)

      except Exception as e:
          self.logger.error(f"Error reading tile at zoom {tile.z} from source at {source.path} : {e}")
          return None

    def _merge_tiles(self, tile_datas: List[Optional[TileData]], target_tile: mercantile.Tile) -> Optional[np.ndarray]:
        """Merge tiles from multiple sources, handling upscaling and priorities"""
        if not any(tile_datas):
            return None
        
        bounds = mercantile.bounds(target_tile)
        
        # Use the tile size of the first tile, or the default if no primary tile
        tile_size = self.default_tile_size
        if tile_datas[0] is not None and 'width' in tile_datas[0].meta and 'height' in tile_datas[0].meta:
            tile_size = tile_datas[0].meta['width']
            
        target_transform = rasterio.transform.from_bounds(
            bounds.west, bounds.south, bounds.east, bounds.north,
            tile_size, tile_size
        )
        
        result = None

        for i, tile_data in enumerate(tile_datas):
            if tile_data is not None:
                resampled_data = self._resample_if_needed(tile_data, target_tile, target_transform, tile_size)
                if result is None:
                  result = resampled_data
                else:
                  mask = ~np.isnan(resampled_data)
                  if np.any(mask):
                    result[mask] = resampled_data[mask]
        
        return result

    def _resample_if_needed(self, tile_data: TileData, target_tile: mercantile.Tile, target_transform, tile_size) -> np.ndarray:
        """Resample tile data if source zoom differs from target"""
        if tile_data.source_zoom != target_tile.z:
            zoom_diff = abs(target_tile.z - tile_data.source_zoom)
            
            #Scale the blur based on the zoom difference
            dynamic_sigma = self.gaussian_blur_sigma * (zoom_diff)
            source_tile = mercantile.Tile(x=target_tile.x // (2**(target_tile.z - tile_data.source_zoom)),
                                            y=target_tile.y // (2**(target_tile.z - tile_data.source_zoom)),
                                            z=tile_data.source_zoom
                                            )
            source_bounds = mercantile.bounds(source_tile)

            
            
            x_offset = (target_tile.x % (2**(target_tile.z - tile_data.source_zoom)))
            y_offset = (target_tile.y % (2**(target_tile.z - tile_data.source_zoom)))
            
            #Determine the sub region bounds.
            sub_region_width = (source_bounds.east - source_bounds.west) / (2**(target_tile.z - tile_data.source_zoom))
            sub_region_height = (source_bounds.north - source_bounds.south) / (2**(target_tile.z - tile_data.source_zoom))

            sub_region_west = source_bounds.west + (x_offset * sub_region_width)
            sub_region_south = source_bounds.south + (y_offset * sub_region_height)
            sub_region_east = sub_region_west + sub_region_width
            sub_region_north = sub_region_south + sub_region_height

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
                    # Apply Gaussian blur to destination data after reprojection
                    blurred_data = gaussian_filter(dst_data, sigma=dynamic_sigma)


                    if blurred_data.ndim == 3:
                        return blurred_data[0]
                    else:
                        return blurred_data
        if tile_data.data.ndim == 3:
            return tile_data.data[0]
        else:
            return tile_data.data
    
    def process_tile(self, tile: mercantile.Tile, source_conns: Dict[Path, sqlite3.Connection], write_queue: Queue, verbose:bool=False) -> None:
        """Process a single tile, merging data from multiple sources"""
        #print(f"process_tile called with tile: ")
        try:
            # Extract tiles from all sources
            self.logger.debug(f"Start process tile {tile.z}/{tile.x}/{tile.y}")
            tile_datas = [self._extract_tile(source, tile, i, verbose) for i, source in enumerate(self.sources)]
            self.logger.debug(f"tile datas: {len(tile_datas)}")

            if not any(tile_datas):
                self.logger.debug(f"No data found for tile {tile.z}/{tile.x}/{tile.y}")
                return

            # Merge the elevation data
            merged_elevation = self._merge_tiles(tile_datas, tile)
            
            if merged_elevation is None:
                self.logger.debug(f"No merged elevation for {tile.z}/{tile.x}/{tile.y}")
                return
            
            # Encode using output format and save
            rgb_data = ImageEncoder.data_to_rgb(
                merged_elevation,
                self.output_encoding,
                self.interval,
                base_val=self.base_val,
                quantized_alpha=self.output_quantized_alpha if self.output_encoding == EncodingType.TERRARIUM else False
            )
            image_bytes = ImageEncoder.save_rgb_to_bytes(rgb_data, self.output_image_format, self.default_tile_size)
            
            logging.debug(f"image_bytes {len(image_bytes)}")
            write_queue.put((tile, image_bytes))
            self.logger.info(f"Successfully processed tile {tile.z}/{tile.x}/{tile.y}")
        except Exception as e:
            self.logger.error(f"Error processing tile {tile.z}/{tile.x}/{tile.y}: {e}")
            raise
    
    def _get_tiles_for_zoom(self, zoom: int, verbose=False) -> List[mercantile.Tile]:
        """Get list of tiles to process for a given zoom level"""
        if verbose:
          print(f"_get_tiles_for_zoom called with zoom: {zoom}")
        tiles = set()
        
        if self.bounds is not None:
            w,s,e,n = self.bounds
            for x, y in _tile_range(mercantile.tile(w, n, zoom), mercantile.tile(e, s, zoom)):
                tiles.add(mercantile.Tile(x=x, y=y, z=zoom))
        else:
            # Get tiles from the LAST source
            source = self.sources[-1]
            with rasterio.open(source.path) as src:
                
                bounds = src.bounds
                
                w,s,e,n = bounds
                for x, y in _tile_range(mercantile.tile(w, n, zoom), mercantile.tile(e, s, zoom)):
                  tiles.add(mercantile.Tile(x=x, y=y, z=zoom))
        return list(tiles)

    def get_max_zoom_level(self) -> int:
        """Get the maximum zoom level from the last source"""
        with rasterio.open(self.sources[-1].path) as src:
            
            bounds = src.bounds
            
            max_zoom = mercantile.tile(bounds.west, bounds.north, self.min_zoom).z
            
            while True:
              test_zoom = max_zoom + 1
              test_tile = mercantile.tile(bounds.west, bounds.north, test_zoom)
              test_bounds = mercantile.bounds(test_tile)
              
              if test_bounds.west < bounds.west or test_bounds.south < bounds.south or test_bounds.east > bounds.east or test_bounds.north > bounds.north:
                break
              else:
                max_zoom = test_zoom
            
            return max_zoom

    def process_all(self, min_zoom: int = 0, verbose=False):
        """Process all zoom levels"""
        max_zoom = self.max_zoom if self.max_zoom is not None else self.get_max_zoom_level()
        self.logger.info(f"Processing zoom levels {min_zoom} to {max_zoom}")

        with MBTilesDatabase(self.output_path) as db:
            db.add_bounds_center_metadata(self.bounds, self.min_zoom, max_zoom, self.output_encoding.value, self.output_image_format.value, "Merged Raster")

            for zoom in range(min_zoom, max_zoom + 1):
                self.process_zoom_level(zoom, verbose)

        self.logger.info("Completed processing all zoom levels")

    def process_zoom_level(self, zoom: int, verbose:bool = False):
        """Process all tiles for a given zoom level in parallel"""
        self.logger.info(f"Processing zoom level {zoom}")
        
        # Get list of tiles to process
        tiles = self._get_tiles_for_zoom(zoom, verbose)
        self.logger.info(f"Found {len(tiles)} tiles to process at zoom {zoom}")

        # Create task tuples with all necessary data
        tasks = [
            (
                tile,
                [(s.path, s.height_adjustment, s.mask_values)
                 for s in self.sources],
                self.output_path,
                self.output_encoding.value,
                self.resampling,
                self.output_image_format.value,
                self.output_quantized_alpha,
                self.base_val,
                self.interval,
                verbose
            )
            for tile in tiles
        ]

        # Process tiles in parallel using the standalone function
        with multiprocessing.Pool(self.processes) as pool:
            for _ in pool.imap_unordered(
                process_tile_task,
                tasks,
                chunksize=1
            ):
                pass

@retry(attempts=5, base_delay=0.5, max_delay=5)
def process_tile_task(task_tuple: tuple) -> None:
    """Standalone function for processing tiles that can be pickled"""
    tile, source_configs, output_path, output_encoding, resampling, output_format, output_alpha, base_val, interval, verbose = task_tuple
    
    # Configure logging for each process
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.debug(f"process_tile_task started for tile {tile.z}/{tile.x}/{tile.y}")
    
    sources = []
    db = None
    try:
        # Reconstruct MBTilesSource objects and create connections
        for path, height_adj, mask_vals in source_configs:
            source = RasterSource(
                path=Path(path),
                height_adjustment=height_adj,
                mask_values=mask_vals
            )
            sources.append(source)

        # create instance
        merger_instance = RasterRGBMerger(sources, output_path, output_encoding=EncodingType(output_encoding), resampling=resampling, output_image_format=ImageFormat(output_format), output_quantized_alpha=output_alpha, base_val=base_val, interval=interval)
            
        # Open database connection for the entire task
        with MBTilesDatabase(output_path) as db:
            # Extract tiles from all sources
            tile_datas = []
            for i, source in enumerate(sources):
                tile_data = merger_instance._extract_tile(source, tile, i, verbose)
                tile_datas.append(tile_data)
            
            if not any(tile_datas):
                logging.debug(f"No tile data for {tile.z}/{tile.x}/{tile.y}")
                return

            # Merge the elevation data
            merged_elevation = merger_instance._merge_tiles(tile_datas, tile)
            
            if merged_elevation is None:
                logging.debug(f"No merged elevation {tile.z}/{tile.x}/{tile.y}")
                return
            
            # Encode using output format
            rgb_data = ImageEncoder.data_to_rgb(
                merged_elevation,
                output_encoding,
                interval,
                base_val=base_val,
                quantized_alpha=output_alpha
            )
            image_bytes = ImageEncoder.save_rgb_to_bytes(rgb_data, output_format)
            logging.debug(f"image_bytes {len(image_bytes)}")
            # Write to output database
            db.insert_tile_with_retry([tile.x, tile.y, tile.z], image_bytes)


    except Exception as e:
        logging.error(f"Error processing tile {tile.z}/{tile.x}/{tile.y}: {e}")
        raise

def _tile_range(start: mercantile.Tile, stop: mercantile.Tile):
    for x in range(start.x, stop.x + 1):
        for y in range(start.y, stop.y + 1):
            yield x, y
