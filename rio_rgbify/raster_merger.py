import rasterio
import mercantile
from rasterio.warp import reproject, Resampling, transform_bounds
import numpy as np
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from rio_rgbify.database import MBTilesDatabase
from rio_rgbify.image import ImageFormat, ImageEncoder
import functools
import multiprocessing
from scipy.ndimage import gaussian_filter # Import gaussian filter
import psutil


class EncodingType(Enum):
    MAPBOX = "mapbox"
    TERRARIUM = "terrarium"

@dataclass
class RasterSource:
    """Configuration for a raster source file"""
    path: Path
    height_adjustment: float = 0.0
    base_val: float = -10000 # Add base val, with default of -10000 for mapbox
    interval: float = 0.1 # Add interval with default of 0.1 for mapbox
    mask_values: list = field(default_factory=lambda: [0.0])

    def __post_init__(self):
        if not self.path.exists():
            raise ValueError(f"Source file does not exist: {self.path}")

@dataclass
class TileData:
    """Container for reprojected tile data"""
    data: np.ndarray
    meta: dict


class RasterRGBMerger:
    """
    A class to merge multiple GeoTiff rasters into an MBTiles file.
    """
    def __init__(self, sources, output_path, output_encoding=EncodingType.MAPBOX,
                 resampling=Resampling.lanczos, processes=None, default_tile_size=512,
                 output_image_format=ImageFormat.PNG, output_quantized_alpha=False,
                 min_zoom=0, max_zoom=None, bounds=None, gaussian_blur_sigma=0.2):
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
        self.gaussian_blur_sigma = gaussian_blur_sigma


    def _reproject_raster_tile(self, src_path, tile: mercantile.Tile, source: RasterSource) -> Optional[TileData]:
      
        """Reprojects a raster to a tile's bounds."""
        try:
          
            with rasterio.open(src_path) as src:
              
                bounds = mercantile.bounds(tile)
                dst_crs = "EPSG:3857"
                
                # Calculate the transform
                toaffine = rasterio.transform.from_bounds(bounds.west, bounds.south, bounds.east, bounds.north, 512, 512)
                out = np.empty((512, 512), dtype=src.meta["dtype"])

                reproject(
                    source=rasterio.band(src, 1),
                    destination=out,
                    dst_transform=toaffine,
                    dst_crs=dst_crs,
                    resampling=self.resampling
                  )
                # Apply mask if necessary
                masked_out = ImageEncoder._mask_elevation(out, source.mask_values)
                # Return TileData object
                return TileData(data=masked_out, meta={
                'crs': dst_crs,
                'transform': toaffine,
                'width': 512,
                'height': 512
                })
        except Exception as e:
            self.logger.error(f"Error reprojecting raster tile {tile.z}/{tile.x}/{tile.y} from {src_path}:")
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
                #Apply the height adjustment
                resampled_data += self.sources[i].height_adjustment
                if result is None:
                    result = resampled_data
                else:
                    mask = ~np.isnan(resampled_data)
                    if np.any(mask):
                        result[mask] = resampled_data[mask]

        return result

    def _resample_if_needed(self, tile_data: TileData, target_tile: mercantile.Tile, target_transform, tile_size) -> np.ndarray:
        """Resample tile data if source zoom differs from target"""
        source_bounds = mercantile.bounds(target_tile)
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**tile_data.meta) as src:

                dst_data = np.zeros((1, tile_size, tile_size), dtype=np.float32)
                reproject(
                    source=tile_data.data,
                    destination=dst_data,
                    src_transform=tile_data.meta['transform'],
                    src_crs=tile_data.meta['crs'],
                    dst_transform=target_transform,
                    dst_crs=tile_data.meta['crs'],
                    resampling=self.resampling
                  )
                # Apply Gaussian blur to destination data after reprojection
                blurred_data = gaussian_filter(dst_data, sigma=self.gaussian_blur_sigma)

                if blurred_data.ndim == 3:
                    return blurred_data[0]
                else:
                    return blurred_data
    

    def process_tile(self, tile: mercantile.Tile) -> Optional[bytes]:
        """Process a single tile, merging data from multiple sources"""
        try:
          
            self.logger.debug(f"Start process tile {tile.z}/{tile.x}/{tile.y}")
          
            tile_datas = [self._reproject_raster_tile(source.path, tile, source) for source in self.sources]

            if not any(tile_datas):
                self.logger.debug(f"No data found for tile {tile.z}/{tile.x}/{tile.y}")
                return None

            merged_elevation = self._merge_tiles(tile_datas, tile)
            if merged_elevation is None:
                self.logger.debug(f"No merged elevation for {tile.z}/{tile.x}/{tile.y}")
                return None
            # Encode using output format and save
            rgb_data = ImageEncoder.data_to_rgb(
                merged_elevation,
                self.output_encoding,
                0.1,
                base_val=-10000,
                quantized_alpha=self.output_quantized_alpha if self.output_encoding == EncodingType.TERRARIUM else False
            )
            image_bytes = ImageEncoder.save_rgb_to_bytes(rgb_data, self.output_image_format, self.default_tile_size)

            self.logger.info(f"Successfully processed tile {tile.z}/{tile.x}/{tile.y}")
            return (tile, image_bytes)
        except Exception as e:
            self.logger.error(f"Error processing tile {tile.z}/{tile.x}/{tile.y}: ")
            return None

    def _tile_range(self, min_tile, max_tile):
        """
        Given a min and max tile, return an iterator of
        all combinations of this tile range
        """
        min_x, min_y, _ = min_tile
        max_x, max_y, _ = max_tile

        return itertools.product(range(min_x, max_x + 1), range(min_y, max_y + 1))

    def _make_tiles(self, bbox, minz, maxz):
        """
        Given a bounding box, zoom range, and source crs,
        find all tiles that would intersect
        """
        w, s, e, n = bbox

        EPSILON = 1.0e-10

        w += EPSILON
        s += EPSILON
        e -= EPSILON
        n -= EPSILON
        

        for z in range(minz, maxz + 1):
            for x, y in self._tile_range(mercantile.tile(w, n, z), mercantile.tile(e, s, z)):
                yield [x, y, z]

    def _init_worker(self):
      """
      Set up each process to ignore SIGINT signals so they can be terminated correctly with ctrl+c
      """
      import signal
      signal.signal(signal.SIGINT, signal.SIG_IGN)

    def process_all(self, min_zoom: int = 0):
        """Process all zoom levels"""
        
        if self.bounds is None:
          source = self.sources[-1]
          with rasterio.open(source.path) as src:
            bbox = list(src.bounds)
            self.bounds = bbox
        
        max_zoom = self.max_zoom if self.max_zoom is not None else 18 #set a reasonable max zoom for raster data.
        self.logger.info(f"Processing zoom levels {min_zoom} to {max_zoom}")

        tiles = self._make_tiles(self.bounds, min_zoom, max_zoom)

        total_tiles = len(list(self._make_tiles(self.bounds, min_zoom, max_zoom)))
        
        # Smart process scaling - use fewer processes for fewer tiles
        if self.processes is None or self.processes <= 0:
          # Scale processes based on tile count and CPU count
          self.processes = multiprocessing.cpu_count() - 1  # Leave one CPU free
      
        # Ensure processes does not exceed tile count
        self.processes = min(total_tiles, self.processes)
        
        batch_size = max(1, total_tiles // (self.processes * 2))

        with MBTilesDatabase(self.output_path) as db:
            db.add_bounds_center_metadata(self.bounds, self.min_zoom, max_zoom, self.output_encoding.value, self.output_image_format.value, "Merged Terrain")
          
            ctx = multiprocessing.get_context("fork") #Set multiprocessing context so it works on mac

            with ctx.Pool(self.processes, initializer=self._init_worker) as pool:
                try:
                  total_processed = 0
                  for i, result in enumerate(pool.imap_unordered(self.process_tile, tiles, chunksize=batch_size), 1):
                    if result:
                      tile, image_bytes = result
                      db.insert_tile_with_retry([tile.x, tile.y, tile.z], image_bytes)
                      total_processed +=1
                      self.logger.info(f"Processed {i}/{total_tiles} tiles")
                    if i % batch_size == 0 or i == total_tiles:
                      db.conn.commit()
                      self.logger.info(f"Committed to database at tile {i}")

                  self.logger.info(f"Completed processing {total_processed} tiles")

                except KeyboardInterrupt:
                    self.logger.info("Caught KeyboardInterrupt, terminating workers")
                    pool.terminate()
                    raise
                except Exception as e:
                    self.logger.error(f"Error in processing: {str(e)}")
                    pool.terminate()
                    raise
                finally:
                    pool.close()
                    pool.join()
