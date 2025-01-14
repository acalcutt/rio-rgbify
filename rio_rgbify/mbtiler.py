from __future__ import with_statement
from __future__ import division

import sys
import traceback
import itertools
import mercantile
import rasterio
import numpy as np
from multiprocessing import Pool, Queue, get_context, current_process
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RGBTiler:
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
        bounding_tile=None,
        resampling="bilinear",
        quantized_alpha=False,
        **kwargs
    ):
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

        if not "format" in kwargs:
            self.writer_func = ImageEncoder.encode_as_png
            self.image_format = "png"
        elif kwargs["format"].lower() == "png":
            self.writer_func = ImageEncoder.encode_as_png
            self.image_format = "png"
        elif kwargs["format"].lower() == "webp":
            self.writer_func = ImageEncoder.encode_as_webp
            self.image_format = "webp"
        else:
            raise ValueError(f"{kwargs['format']} is not a supported filetype!")

        if resampling not in ["nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average", "mode", "gauss"]:
            raise ValueError(f"{resampling} is not a supported resampling method!")
        
        self.resampling = Resampling[resampling.lower()]
        
        self.kwargs = {
            "driver": "PNG",
            "dtype": "uint8",
            "height": 512,
            "width": 512,
            "count": 3,
            "crs": "EPSG:3857",
        }

    def _process_tile(self, tile_args):
        """Process a single tile with error handling"""
        tile, input_path = tile_args
        try:
            with rasterio.open(input_path) as src:
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
                    resampling=self.resampling,
                )

                out = ImageEncoder.data_to_rgb(
                    out, 
                    self.encoding, 
                    self.interval, 
                    base_val=self.base_val, 
                    round_digits=self.round_digits,
                    quantized_alpha=self.quantized_alpha
                )

                return tile, self.writer_func(out, self.kwargs.copy(), toaffine)
        except Exception as e:
            logging.error(f"Error processing tile {tile}: {str(e)}")
            return None

    def _init_worker(self):
        """Initialize worker process"""
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        logging.info(f"Initialized worker process {current_process().name}")

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

    def _chunk_tiles(self, tiles, chunk_size):
        """Chunk tiles into batches"""
        chunk = []
        for tile in tiles:
            chunk.append(tile)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def run(self, processes=4, batch_size=500):
        """Main processing loop with improved error handling and resource management"""
        if processes < 1:
            raise ValueError("Number of processes must be at least 1")

        with rasterio.open(self.inpath) as src:
            bbox = list(src.bounds)
            src_crs = src.crs

        if processes == 1:
            with rasterio.open(self.inpath) as src:
                with self.db:
                    self._init_db()
                    for tile in self._generate_tiles(bbox, src_crs):
                        result = self._process_tile((tile, self.inpath))
                        if result:
                            self.db.insert_tile(*result)
                    self.db.conn.commit()
            return

        # Multiprocessing implementation
        ctx = get_context("fork")
        os.environ['RASTERIO_NUM_THREADS'] = '1'

        with self.db:
            self._init_db()
            
            with ctx.Pool(processes, initializer=self._init_worker) as pool:
                try:
                    tiles = list(self._generate_tiles(bbox, src_crs))
                    for chunk in self._chunk_tiles(tiles, batch_size):
                        # Create tile_data without database connection
                        tile_data = [(tile, self.inpath) for tile in chunk]
                        
                        results = pool.map(self._process_tile, tile_data)
                        
                        # Process results and insert into database in main process
                        for result in filter(None, results):
                            self.db.insert_tile(*result)
                        
                        self.db.conn.commit()
                        logging.info(f"Processed batch of {len(chunk)} tiles")
                
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

    def _init_db(self):
        """Initialize database with metadata"""
        self.db.add_metadata({
            "format": self.image_format,
            "name": "",
            "description": "",
            "version": "1",
            "type": "baselayer"
        })

    def __enter__(self):
        self.db = MBTilesDatabase(self.outpath)
        return self

    def __exit__(self, ext_t, ext_v, trace):
        if ext_t:
            traceback.print_exc()
        self.db.__exit__(ext_t, ext_v, trace)
