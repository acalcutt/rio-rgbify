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

def process_tile(inpath, encoding, interval, base_val, round_digits, resampling, quantized_alpha, kwargs, tile):
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

            reproject(
                rasterio.band(src, 1),
                out,
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

            if encoding == "mapbox":
                writer_func = ImageEncoder.encode_as_png
            else:
                writer_func = ImageEncoder.encode_as_webp

            result = writer_func(out, kwargs.copy(), toaffine)
            return tile, result
            
    except Exception as e:
        logging.error(f"Error processing tile {tile}: {str(e)}")
        return None

class RGBTiler:
    def __init__(self, inpath, outpath, min_z, max_z, interval=1, base_val=0,
                 round_digits=0, encoding="mapbox", bounding_tile=None,
                 resampling="bilinear", quantized_alpha=False, **kwargs):
        self.inpath = inpath
        self.outpath = outpath
        self.min_z = min_z
        self.max_z = max_z
        self.bounding_tile = bounding_tile
        self.encoding = encoding
        self.interval = interval
        self.base_val = base_val
        self.round_digits = round_digits
        self.quantized_alpha = quantized_alpha
        
        if resampling not in ["nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average", "mode", "gauss"]:
            raise ValueError(f"{resampling} is not a supported resampling method!")
        
        self.resampling = Resampling[resampling.lower()]
        
        if kwargs.get("format", "png").lower() not in ["png", "webp"]:
            raise ValueError(f"{kwargs.get('format')} is not a supported filetype!")
            
        self.image_format = kwargs.get("format", "png").lower()
        
        self.kwargs = {
            "driver": "PNG",
            "dtype": "uint8",
            "height": 512,
            "width": 512,
            "count": 3,
            "crs": "EPSG:3857",
        }

    def _init_worker(self):
        """Initialize worker process"""
        # Don't restrict GDAL/rasterio threads per process
        os.environ['RASTERIO_NUM_THREADS'] = '1'
        # Ignore SIGINT in worker processes
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        proc = psutil.Process()
        logging.info(f"Initialized worker process {current_process().name} on CPU {proc.cpu_num()}")

    def _generate_tiles(self, bbox, src_crs):
        """Generate tile coordinates"""
        if self.bounding_tile is None:
            w, s, e, n = transform_bounds(*[src_crs, "EPSG:3857"] + bbox, densify_pts=21)
        else:
            w, s, e, n = mercantile.bounds(self.bounding_tile)
            w, s, e, n = transform_bounds("EPSG:4326", "EPSG:3857", w, s, e, n)

        EPSILON = 1.0e-10
        w += EPSILON
        s += EPSILON
        e -= EPSILON
        n -= EPSILON

        for z in range(self.min_z, self.max_z + 1):
            min_tile = mercantile.tile(w, n, z)
            max_tile = mercantile.tile(e, s, z)
            for x, y in itertools.product(
                range(min_tile.x, max_tile.x + 1),
                range(min_tile.y, max_tile.y + 1)
            ):
                yield [x, y, z]

    def run(self, processes=None, batch_size=None):
        """Main processing loop"""
        # Automatically determine optimal number of processes if not specified
        if processes is None or processes <= 0:
            processes = max(1, cpu_count() - 1)  # Leave one CPU free for system
        
        # Automatically determine batch size based on number of processes
        if batch_size is None:
            batch_size = processes * 4  # Rule of thumb: 4 tasks per process

        logging.info(f"Running with {processes} processes and batch size of {batch_size}")

        with rasterio.open(self.inpath) as src:
            bbox = list(src.bounds)
            src_crs = src.crs
            tiles = list(self._generate_tiles(bbox, src_crs))

        logging.info(f"Total tiles to process: {len(tiles)}")

        if processes == 1:
            with self.db:
                self.db.add_metadata({
                    "format": self.image_format,
                    "name": "",
                    "description": "",
                    "version": "1",
                    "type": "baselayer"
                })
                
                for tile in tiles:
                    result = process_tile(
                        self.inpath,
                        self.encoding,
                        self.interval,
                        self.base_val,
                        self.round_digits,
                        self.resampling,
                        self.quantized_alpha,
                        self.kwargs,
                        tile
                    )
                    if result:
                        self.db.insert_tile(*result)
                self.db.conn.commit()
            return

        # Multiprocessing implementation
        ctx = get_context("fork")
        
        # Create a partial function with all the constant arguments
        process_func = functools.partial(
            process_tile,
            self.inpath,
            self.encoding,
            self.interval,
            self.base_val,
            self.round_digits,
            self.resampling,
            self.quantized_alpha,
            self.kwargs
        )

        with self.db:
            self.db.add_metadata({
                "format": self.image_format,
                "name": "",
                "description": "",
                "version": "1",
                "type": "baselayer"
            })
            
            with ctx.Pool(processes, initializer=self._init_worker) as pool:
                try:
                    # Process tiles in batches using imap_unordered for better performance
                    total_processed = 0
                    for result in pool.imap_unordered(process_func, tiles, chunksize=batch_size):
                        if result:
                            self.db.insert_tile(*result)
                            total_processed += 1
                            
                            # Commit every batch_size tiles
                            if total_processed % batch_size == 0:
                                self.db.conn.commit()
                                logging.info(f"Processed {total_processed}/{len(tiles)} tiles")
                    
                    # Final commit
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
