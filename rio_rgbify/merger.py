import sqlite3
import rasterio
import mercantile
from rasterio.warp import reproject, Resampling
import numpy as np
import io
from PIL import Image
import multiprocessing
from pathlib import Path
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List
from contextlib import contextmanager
from datetime import datetime
from rio_rgbify.encoders import Encoder
from rio_rgbify.database import MBTilesDatabase
from rio_rgbify.image import  save_rgb_to_bytes, ImageFormat 

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

@dataclass
class ProcessTileArgs:
    """Arguments for parallel tile processing"""
    tile: mercantile.Tile
    sources: List[MBTilesSource]
    output_path: Path
    output_encoding: EncodingType
    resampling: int
    output_image_format: ImageFormat

@dataclass
class TileData:
    """Container for decoded tile data"""
    data: np.ndarray
    meta: dict
    source_zoom: int

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
        output_image_format: ImageFormat = ImageFormat.PNG
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

    @contextmanager
    def _db_connection(self, db_path: Path):
        """Context manager for database connections
        
        Parameters
        ----------
        db_path : Path
           The path to the database
        """
        print(f"_db_connection called with db_path: {db_path}")
        conn = sqlite3.connect(db_path)
        try:
            yield conn
        finally:
            conn.close()
    

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

                if encoding == EncodingType.MAPBOX: # Pass in the base and interval if we are using mapbox
                   elevation = Encoder._decode(rgb, source.base_val, source.interval, encoding.value) # Use the static decode method from the encoder
                elif encoding == EncodingType.TERRARIUM:
                   elevation = Encoder._decode(rgb, 0, 1, encoding.value) # Use the static decode method from the encoder
                else:
                  raise ValueError(f"Invalid encoding type: {encoding}")

                elevation = Encoder._mask_elevation(elevation)
                
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

    def _extract_tile(self, source: MBTilesSource, zoom: int, x: int, y: int) -> Optional[TileData]:
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
            with MBTilesDatabase._db_connection(source.path) as conn:
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
        #print(f"_merge_tiles called with tile_datas: {tile_datas}, target_tile: {target_tile}")
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
        """Resample tile data if source zoom differs from target
        
        Parameters
        ----------
        tile_data : TileData
            The tile data to resample
        target_tile : mercantile.Tile
            The mercantile tile object for the resampled data
        target_transform
            The transform for the target tile
        tile_size
            The size of the tile in pixels.

        Returns
        -------
        np.ndarray
            The resampled elevation array.
        """
        #print(f"_resample_if_needed called with tile_data: {tile_data}, target_tile: {target_tile}")
        if tile_data.source_zoom != target_tile.z:

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

              if dst_data.ndim == 3:
                  return dst_data[0]
              else:
                  return dst_data
        if tile_data.data.ndim == 3:
           return tile_data.data[0]
        else:
           return tile_data.data
            

    def process_tile(self, tile: mercantile.Tile) -> None:
        """Process a single tile, merging data from multiple sources.
        
        Parameters
        ----------
        tile : mercantile.Tile
            The tile to process.
        """
        #print(f"process_tile called with tile: {tile}")
        try:
            # Extract tiles from all sources
            tile_datas = [self._extract_tile(source, tile.z, tile.x, tile.y) for source in self.sources]

            if not any(tile_datas):
              self.logger.debug(f"No data found for tile {tile.z}/{tile.x}/{tile.y}")
              return

            # Merge the elevation data
            merged_elevation = self._merge_tiles(tile_datas, tile)
            
            if merged_elevation is not None:
                # Encode using output format and save
                rgb_data = Encoder.data_to_rgb(merged_elevation, self.output_encoding, 0.1, min_val=-10000, max_val=10000) # Use the encoder from the encoders.py file
                with MBTilesDatabase(self.output_path) as db:
                  image_bytes = save_rgb_to_bytes(rgb_data, self.output_image_format, self.default_tile_size)
                  db.insert_tile(tile = [tile.x, tile.y, tile.z], contents = image_bytes)
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
          
          # Get tiles from the LAST source
          source = self.sources[-1]
          with self._db_connection(source.path) as conn:
              cursor = conn.cursor()
              cursor.execute(
                  'SELECT DISTINCT tile_column, tile_row FROM tiles WHERE zoom_level = ?',
                  (zoom,)
              )
              rows = cursor.fetchall()
          
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
    def _process_tile_wrapper(args: ProcessTileArgs) -> None:
        """Static wrapper method for parallel tile processing
        
        Parameters
        ----------
        args : ProcessTileArgs
            The arguments for the processing of a tile
        """
        print(f"_process_tile_wrapper called with args: {args}")
        try:
            merger = TerrainRGBMerger(
                sources=args.sources,
                output_path=args.output_path,
                output_encoding=args.output_encoding,
                resampling=args.resampling,
                processes=1,
                default_tile_size=512,
                output_image_format=args.output_image_format
            )
            merger.process_tile(args.tile)
        except Exception as e:
            logging.error(f"Error processing tile {args.tile}: {e}")

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
                output_image_format = self.output_image_format
            )
            for tile in tiles
        ]
        
        # Process tiles in parallel
        with multiprocessing.Pool(self.processes) as pool:
            for _ in pool.imap_unordered(self._process_tile_wrapper, process_args):
                pass

    def get_max_zoom_level(self) -> int:
        """Get the maximum zoom level from both sources
        
        Returns
        -------
        int
            The maximum zoom level found in the sources.
        """
        print("_get_max_zoom_level called")
        max_zoom = 0
        for source in self.sources:
            with self._db_connection(source.path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(zoom_level) FROM tiles")
                result = cursor.fetchone()
                if result[0] is not None:
                    max_zoom = max(max_zoom, result[0])
        return max_zoom

    def process_all(self, min_zoom: int = 0):
        """Process all zoom levels from min_zoom to max available
        
        Parameters
        ----------
        min_zoom : int, optional
            The minimum zoom level to start processing at. Defaults to 0
        """
        print(f"process_all called with min_zoom: {min_zoom}")
        max_zoom = self.get_max_zoom_level()
        self.logger.info(f"Processing zoom levels {min_zoom} to {max_zoom}")

        for zoom in range(min_zoom, max_zoom + 1):
            self.process_zoom_level(zoom)

        self.logger.info("Completed processing all zoom levels")
