from __future__ import with_statement
from __future__ import division

import sys
import traceback
import itertools
import mercantile
import rasterio
import numpy as np
from multiprocessing import Pool, Queue, get_context, current_process, cpu_count
import os
from riomucho.single_process_pool import MockTub
from io import BytesIO
from PIL import Image
from rasterio import transform
from rasterio.warp import reproject, transform_bounds
from rasterio.enums import Resampling
from rio_rgbify.database import MBTilesDatabase
from rio_rgbify.image import ImageEncoder
import logging
import signal
import functools
import psutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_tile(inpath, format, encoding, interval, base_val, round_digits, resampling, quantized_alpha, tile):
    """Standalone tile processing function"""
    # Log the process ID and CPU core
    proc = psutil.Process()
    logging.info(f"Processing tile {tile} on CPU {proc.cpu_num()} (PID: {os.getpid()})")
    
    try:
        with rasterio.open(inpath) as src:
            x, y, z = tile

            bounds_3857 = [
                c for i in (
                    mercantile.xy(*mercantile.ul(x, y + 1, z)),
                    mercantile.xy(*mercantile.ul(x + 1, y, z)),
                )
                for c in i
            ]
            
            #Convert mercator bounds to source bounds
            bounds_src = transform_bounds("EPSG:3857", src.crs, *bounds_3857)

            toaffine = transform.from_bounds(*bounds_3857 + [512, 512])
            out = np.empty((512, 512), dtype=np.float64)
            
            # Calculate the window based on the transformed bounds
            window = rasterio.windows.from_bounds(*bounds_src, transform=src.transform)
            logging.info(f"Tile: {tile}, Window: {window}, Source Transform: {src.transform}")

            
            # If the window width or height is 0, return an empty array
            if window.width == 0 or window.height == 0:
               logging.info(f"Empty Window, skipping reprojection: {window}")
               out = np.empty((0,0), dtype=np.float64)
               
               out = ImageEncoder.data_to_rgb(
                  out, 
                  encoding, 
                  interval, 
                  base_val=base_val, 
                  round_digits=round_digits,
                  quantized_alpha=quantized_alpha
                )
                
               result = ImageEncoder.save_rgb_to_bytes(out, format)
               return tile, result
               
            # Read the source data using the window
            source_data = src.read(1, window=window, out_shape=(512,512), resampling=resampling)
            logging.info(f"Source Data shape: {source_data.shape} min:{np.min(source_data)} max:{np.max(source_data)}")
            
            reproject(
                source=source_data,  # Source data
                src_transform=src.transform, #Source Transform from file
                src_crs=src.crs, # Source CRS from file
                destination=out,
                dst_transform=toaffine,
                dst_crs="EPSG:3857",
                resampling=resampling,
                src_nodata=src.nodata,
                dst_nodata=np.nan
            )
            
            if src.nodata is not None:
                out[np.isnan(out)] = np.nan
                
            logging.info(f"out before data_to_rgb: {out}")
            out = ImageEncoder.data_to_rgb(
                out, 
                encoding, 
                interval, 
                base_val=base_val, 
                round_digits=round_digits,
                quantized_alpha=quantized_alpha
            )
            logging.info(f"out after data_to_rgb: {out}")
            
            result = ImageEncoder.save_rgb_to_bytes(out, format)
            return tile, result
            
    except Exception as e:
        logging.error(f"Error processing tile {tile}: {str(e)}")
        return None

class RGBTiler:
    """
    Takes continuous source data of an arbitrary bit depth and encodes it
    in parallel into RGB tiles in an MBTiles file. Provided with a context manager:
    ```
    with RGBTiler(inpath, outpath, min_z, max_x) as tiler:
        tiler.run(processes)
    ```

    Parameters
    -----------
    inpath: string
        filepath of the source file to read and encode
    outpath: string
        filepath of the output `mbtiles`
    min_z: int
        minimum zoom level to tile
    max_z: int
        maximum zoom level to tile

    Keyword Arguments
    ------------------
    baseval: float
        the base value of the RGB numbering system.
        (will be treated as zero for this encoding)
        Default=0
    interval: float
        the interval at which to encode
        Default=1
    round_digits: int
        Erased less significant digits
        Default=0
    encoding: str
        output tile encoding (mapbox or terrarium)
        Default=mapbox
    format: str
        output tile image format (png or webp)
        Default=png
    bounding_tile: list
        [x, y, z] of bounding tile; limits tiled output to this extent

    Returns
    --------
    None

    """

    def __init__(
        self,
        inpath,
        outpath,
        min_z,
        max_z,
        interval=1,
        base_val=0,
        round_digits=0,
        encoding="mapbox",
        format="webp",
        resampling=Resampling.nearest,
        quantized_alpha=True,
        bounding_tile=None,
    ):
        self.inpath = inpath
        self.outpath = outpath
        self.min_z = min_z
        self.max_z = max_z
        self.bounding_tile = bounding_tile
        self.encoding = encoding
        self.format = format
        self.interval = interval
        self.base_val = base_val
        self.round_digits = round_digits
        self.resampling = resampling
        self.quantized_alpha = quantized_alpha


    def _generate_tiles(self, bbox, src_crs):
      
        if self.bounding_tile is None:
            tiles = mercantile.tiles(
                *bbox, zooms=range(self.min_z, self.max_z + 1), truncate=False
            )
        else:
            constrained_bbox = list(mercantile.bounds(self.bounding_tile))
            tiles = mercantile.tiles(
                *constrained_bbox, zooms=range(self.min_z, self.max_z + 1), truncate=False
            )
        
        for tile in tiles:
            yield tile

    def _init_worker(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)


    def run(self, processes=None, batch_size=None):
        """Main processing loop with smart process scaling"""
        print(f"self.inpath {self.inpath}")
        with rasterio.open(self.inpath) as src:
            bbox = list(src.bounds)
            src_crs = src.crs
            print(f"src_crs1 {src_crs}")
            tiles = list(self._generate_tiles(bbox, src_crs))

        total_tiles = len(tiles)
        print(f"Total tiles to process: {total_tiles}")

        # Smart process scaling - use fewer processes for fewer tiles
        if processes is None or processes <= 0:
            # Scale processes based on tile count and CPU count
            processes = cpu_count() - 1  # Leave one CPU free
        
        # Ensure processes does not exceed tile count
        processes = min(total_tiles, processes)

        # Adjust batch size based on total tiles
        if batch_size is None:
            batch_size = max(1, total_tiles // (processes * 2))  # Ensure at least 1
        
        print(f"Running with {processes} processes and batch size of {batch_size}")

        # Multiprocessing implementation for all tiles
        ctx = get_context("fork")
        
        process_func = functools.partial(
            process_tile,
            self.inpath,
            self.format,
            self.encoding,
            self.interval,
            self.base_val,
            self.round_digits,
            self.resampling,
            self.quantized_alpha,
        )

        with self.db:
            self.db.add_metadata({
                "format": self.format,
                "name": "",
                "description": "",
                "version": "1",
                "type": "baselayer"
            })
            
            with ctx.Pool(processes, initializer=self._init_worker) as pool:
                try:
                    total_processed = 0
                    for i, result in enumerate(pool.imap_unordered(process_func, tiles, chunksize=batch_size), 1):
                        if result:
                            self.db.insert_tile(*result)
                            total_processed += 1
                            print(f"Processed {total_processed}/{total_tiles} tiles")
                            
                        if i % batch_size == 0 or i == total_tiles:  # Commit after each batch or at the end
                            self.db.conn.commit()
                            print("Committed to database")

                    print(f"Completed processing {total_processed} tiles")
                
                except KeyboardInterrupt:
                    print("Caught KeyboardInterrupt, terminating workers")
                    pool.terminate()
                    raise
                except Exception as e:
                    logging.error(f"Error in processing: {str(e)}")
                    pool.terminate()
                    raise
                finally:
                    pool.close()
                    pool.join()

    def __enter__(self):
        self.db = MBTilesDatabase(self.outpath)
        return self

    def __exit__(self, ext_t, ext_v, trace):
        if ext_t:
            traceback.print_exc()
        self.db.__exit__(ext_t, ext_v, trace)
