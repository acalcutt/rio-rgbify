from io import BytesIO
from PIL import Image
import numpy as np
import rasterio
from rasterio._io import virtual_file_to_buffer


def encode_as_webp(data, profile=None, affine=None):
    """
    Uses BytesIO + PIL to encode a (3, height, width)
    array into a webp bytearray.

    Parameters
    -----------
    data: ndarray
        (3 x height x width) uint8 RGB array
    profile: None
        ignored
    affine: None
        ignored

    Returns
    --------
    contents: bytearray
        webp-encoded bytearray of the provided input data
    """
    with BytesIO() as f:
        im = Image.fromarray(np.rollaxis(data, 0, 3))
        im.save(f, format="webp", lossless=True)

        return f.getvalue()


def encode_as_png(data, profile, dst_transform):
    """
    Uses rasterio's virtual file system to encode a (3, height, width)
    array as a png-encoded bytearray.

    Parameters
    -----------
    data: ndarray
        (3 x height x width) uint8 RGB array
    profile: dictionary
        dictionary of kwargs for png writing
    affine: Affine
        affine transform for output tile

    Returns
    --------
    contents: bytearray
        png-encoded bytearray of the provided input data
    """
    profile["affine"] = dst_transform

    with rasterio.open("/vsimem/tileimg", "w", **profile) as dst:
        dst.write(data)

    contents = bytearray(virtual_file_to_buffer("/vsimem/tileimg"))

    return contents