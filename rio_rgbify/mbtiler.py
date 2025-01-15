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

            bounds = [
                c for i in (
                    mercantile.xy(*mercantile.ul(x, y + 1, z)),
                    mercantile.xy(*mercantile.ul(x + 1, y, z)),
                )
                for c in i
            ]

            toaffine = transform.from_bounds(*bounds + [512, 512])
            out = np.empty((512, 512), dtype=src.meta["dtype"])
            
            # Calculate the window
            window = rasterio.windows.from_bounds(*bounds, transform=src.transform)
            
            # Read the source data using the window
            source_data = src.read(1, window=window, out_shape=(512,512), resampling=resampling)
            
            reproject(
                source=source_data,  # Source data
                src_transform=rasterio.transform.from_bounds(*bounds, width=512, height=512),
                src_crs=src.crs,
                destination=out,
                dst_transform=toaffine,
                dst_crs="EPSG:3857",
                resampling=resampling,
            )
            
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
        logging.info(f"Total tiles to process: {total_tiles}")

        # Smart process scaling - use fewer processes for fewer tiles
        if processes is None or processes <= 0:
            # Scale processes based on tile count and CPU count
            available_cpus = cpu_count() - 1  # Leave one CPU free
            processes = min(total_tiles, available_cpus, 4)  # Cap at 4 for small jobs
            if total_tiles < 4:
                processes = 1  # Use single process for very small jobs
        
        # Adjust batch size based on total tiles
        if batch_size is None:
            batch_size = max(1, total_tiles // (processes * 2))  # Ensure at least 1
        
        logging.info(f"Running with {processes} processes and batch size of {batch_size}")

        if processes == 1 or total_tiles == 1:
            logging.info("Using single process mode due to small number of tiles")
            with self.db:
                self.db.add_metadata({
                    "format": self.format,
                    "name": "",
                    "description": "",
                    "version": "1",
                    "type": "baselayer"
                })
                
                for tile in tiles:
                    result = process_tile(
                        self.inpath,
                        self.format,
                        self.encoding,
                        self.interval,
                        self.base_val,
                        self.round_digits,
                        self.resampling,
                        self.quantized_alpha,
                        tile
                    )
                    if result:
                        self.db.insert_tile(*result)
                        logging.info(f"Processed tile {tile}")
                self.db.conn.commit()
            return

        # Multiprocessing implementation for multiple tiles
        ctx = get_context("fork")
        
        process_func = functools.partial(
            process_tile,
            self.inpath,
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
                    for result in pool.imap_unordered(process_func, tiles, chunksize=batch_size):
                        if result:
                            self.db.insert_tile(*result)
                            total_processed += 1
                            logging.info(f"Processed {total_processed}/{total_tiles} tiles")
                            self.db.conn.commit()  # Commit after each tile for small jobs
                    
                    self.db.conn.commit()
                    logging.info(f"Completed processing {total_processed} tiles")
                
                except KeyboardInterrupt:
                    logging.info("Caught KeyboardInterrupt, terminating workers")
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
