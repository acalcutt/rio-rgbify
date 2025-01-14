from __future__ import with_statement
from __future__ import division

import sys
import traceback
import itertools
import mercantile
import rasterio
import numpy as np
from multiprocessing import Pool, Queue, Manager
from rasterio._io import virtual_file_to_buffer
from riomucho.single_process_pool import MockTub

from io import BytesIO
from PIL import Image

from rasterio import transform
from rasterio.warp import reproject, transform_bounds
from rasterio.enums import Resampling

from rio_rgbify.database import MBTilesDatabase
from rio_rgbify.image import ImageEncoder
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


buffer = bytes if sys.version_info > (3,) else buffer

work_func = None
global_args = None
src = None
worker_queue_holder = None

def _create_worker_queue():
    global worker_queue_holder
    worker_queue_holder = Manager().Queue()
    
def _main_worker(inpath, g_work_func, g_args):
    """
    Util for setting global vars w/ a Pool
    """
    global work_func
    global global_args
    global src
    work_func = g_work_func
    global_args = g_args

    src = rasterio.open(inpath)


def _tile_worker(tile):
    """
    For each tile, and given an open rasterio src, plus a`global_args` dictionary
    with attributes of `encoding`, `base_val`, `interval`, `round_digits` and a `writer_func`,
    warp a continous single band raster to a 512 x 512 mercator tile,
    then encode this tile into RGB.

    Parameters
    -----------
    tile: list
        [x, y, z] indices of tile

    Returns
    --------
    tile, buffer
        tuple with the input tile, and a bytearray with the data encoded into
        the format created in the `writer_func`
    """
    x, y, z = tile

    bounds = [
        c
        for i in (
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
        resampling=global_args["resampling"],
    )

    out = ImageEncoder.data_to_rgb(out, global_args["encoding"], global_args["interval"], base_val=global_args["base_val"], round_digits=global_args["round_digits"], quantized_alpha=global_args["quantized_alpha"])  # Use static Encoder method and pass in base_val
    global worker_queue_holder
    worker_queue_holder.put((tile, global_args["writer_func"](out, global_args["kwargs"].copy(), toaffine)))


def _tile_range(min_tile, max_tile):
    """
    Given a min and max tile, return an iterator of
    all combinations of this tile range

    Parameters
    -----------
    min_tile: list
        [x, y, z] of minimun tile
    max_tile:
        [x, y, z] of minimun tile

    Returns
    --------
    tiles: iterator
        iterator of [x, y, z] tiles
    """
    min_x, min_y, _ = min_tile
    max_x, max_y, _ = max_tile

    return itertools.product(range(min_x, max_x + 1), range(min_y, max_y + 1))


def _make_tiles(bbox, src_crs, minz, maxz):
    """
    Given a bounding box, zoom range, and source crs,
    find all tiles that would intersect

    Parameters
    -----------
    bbox: list
        [w, s, e, n] bounds in src_crs
    src_crs: str
        the source crs of the input bbox
    minz: int
        minumum zoom to find tiles for
    maxz: int
        maximum zoom to find tiles for

    Returns
    --------
    tiles: generator
        generator of [x, y, z] tiles that intersect
        the provided bounding box
    """
    w, s, e, n = transform_bounds(*[src_crs, "EPSG:3857"] + bbox, densify_pts=21)

    EPSILON = 1.0e-10

    w += EPSILON
    s += EPSILON
    e -= EPSILON
    n -= EPSILON

    for z in range(minz, maxz + 1):
        for x, y in _tile_range(mercantile.tile(w, n, z), mercantile.tile(e, s, z)):
            yield [x, y, z]

def _create_global_args(interval, base_val, round_digits, encoding, writer_func, resampling, quantized_alpha):
    return {
        "kwargs": {
            "driver": "PNG",
            "dtype": "uint8",
            "height": 512,
            "width": 512,
            "count": 3,
            "crs": "EPSG:3857",
        },
        "interval": interval,
        "round_digits": round_digits,
        "encoding": encoding,
        "writer_func": writer_func,
        "resampling": resampling,
        "base_val": base_val,
        "quantized_alpha": quantized_alpha,
    }


class RGBTiler:
    """
    Takes continous source data of an arbitrary bit depth and encodes it
    in parallel into RGB tiles in an MBTiles file. Provided with a context manager:
    ```
    with RGBTiler(inpath, outpath, min_z, max_x, **kwargs) as tiler:
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
    resampling: str
        Resampling method to use (nearest, bilinear, cubic, cubic_spline, lanczos, average, mode, gauss)
        Default=bilinear
    quantized_alpha: bool
        If True, adds the quantized elevation data to alpha channel if using terrarium encoding
        Default=False


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
        bounding_tile=None,
        resampling="bilinear",
        quantized_alpha=False,
        **kwargs
    ):
        self.run_function = _tile_worker
        self.inpath = inpath
        self.outpath = outpath
        self.min_z = min_z
        self.max_z = max_z
        self.bounding_tile = bounding_tile

        if not "format" in kwargs:
            writer_func = ImageEncoder.encode_as_png
            self.image_format = "png"
        elif kwargs["format"].lower() == "png":
            writer_func = ImageEncoder.encode_as_png
            self.image_format = "png"
        elif kwargs["format"].lower() == "webp":
            writer_func = ImageEncoder.encode_as_webp
            self.image_format = "webp"
        else:
            raise ValueError(
                " is not a supported filetype!".format(kwargs["format"])
            )

        if resampling not in ["nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average", "mode", "gauss"]:
            raise ValueError(
                " is not a supported resampling method!".format(resampling)
            )

        self.resampling = Resampling[resampling]
        # global kwargs not used if output is webp
        self.global_args = _create_global_args(interval, base_val, round_digits, encoding, writer_func, self.resampling, quantized_alpha)

    def __enter__(self):
        self.db = MBTilesDatabase(self.outpath)
        return self

    def __exit__(self, ext_t, ext_v, trace):
        if ext_t:
            traceback.print_exc()
        self.db.__exit__(ext_t, ext_v, trace)


    def run(self, processes=4, batch_size=500):
      """
      Warp, encode, and tile, processing in batches.
      """
      with rasterio.open(self.inpath) as src:
          bbox = list(src.bounds)
          src_crs = src.crs

      if processes == 1:
          self.pool = MockTub(
              _main_worker, (self.inpath, self.run_function, self.global_args)
          )
      else:
          self.pool = Pool(
              processes,
              initializer = _create_worker_queue,
              initargs = (_main_worker, (self.inpath, self.run_function, self.global_args)),
          )

      if self.bounding_tile is None:
          tiles = _make_tiles(bbox, src_crs, self.min_z, self.max_z)
      else:
          constrained_bbox = list(mercantile.bounds(self.bounding_tile))
          tiles = _make_tiles(constrained_bbox, "EPSG:4326", self.min_z, self.max_z)

      tile_batches = self._chunk_iterable(tiles, batch_size)

      with self.db:
         # populate metadata with required fields
          self.db.add_metadata({
              "format": self.image_format,
              "name": "",
              "description": "",
              "version": "1",
              "type": "baselayer"
          })

          for batch in tile_batches:
              logging.info(f"Processing batch of {len(batch)} tiles.")
              
              
              for _ in self.pool.imap_unordered(self.run_function, batch):
                  pass

              while not worker_queue_holder.empty():
                  tile, contents = worker_queue_holder.get()
                  self.db.insert_tile(tile, contents)

              self.db.conn.commit()

      self.pool.close()
      self.pool.join()
      return None

    def _chunk_iterable(self, iterable, chunk_size):
        """Helper to yield successive chunks from an iterable."""
        iterator = iter(iterable)
        while True:
            chunk = list(itertools.islice(iterator, chunk_size))
            if not chunk:
                break
            yield chunk
