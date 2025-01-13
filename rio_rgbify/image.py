from io import BytesIO
from PIL import Image
import numpy as np
import rasterio
from rasterio._io import virtual_file_to_buffer
from enum import Enum
import bisect
import logging

class ImageFormat(Enum):
    PNG = "png"
    WEBP = "webp"


class ImageEncoder:

    @staticmethod
    def data_to_rgb(data, encoding, interval, base_val=-10000, round_digits=0, quantized_alpha=False):
        """
        Given an arbitrary (rows x cols) ndarray,
        encode the data into uint8 RGB from an arbitrary
        base and interval

        Parameters
        ----------
        data: ndarray
            (rows x cols) ndarray of data to encode
        encoding: str
            output tile encoding (mapbox or terrarium)
        interval: float
            the interval at which to encode
        base_val: float
            the base value to apply when using mapbox. Default is -10000
        round_digits: int
            erased less significant digits
        quantized_alpha : bool, optional
            If True, adds the quantized elevation data to alpha channel if using terrarium encoding
            Default is False

        Returns
        --------
        ndarray: rgb data
            a uint8 (3 x rows x cols) or (4 x rows x cols) ndarray with the
            data encoded
        """
        print(f"data_to_rgb called with shape: {data.shape}, encoding: {encoding}, interval: {interval}, base_val: {base_val}, round_digits: {round_digits}, quantized_alpha: {quantized_alpha}")
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array")

        data = data.astype(np.float64)
        if(encoding == "terrarium"):
            data = np.clip(data, -32768, 32767)
            data += 32768
        else:
            data = data.copy()  # Create copy to avoid modifying input
            # CLAMP values before encoding for Mapbox encoding
            data = np.clip(data, base_val, 100000)
            data -= base_val   # Apply offset
            data /= interval
            
        data = np.nan_to_num(data, nan=0) # Replace nan with 0 before rounding
        data = np.around(data / 2**round_digits) * 2**round_digits

        rows, cols = data.shape
        if quantized_alpha and encoding == "terrarium":
            rgb = np.zeros((4, rows, cols), dtype=np.uint8)
            mapping_table = ImageEncoder._generate_mapping_table()
            
            # Vectorized alpha calculation
            alpha_values = np.zeros((rows, cols), dtype=np.uint8)
            for i, threshold in enumerate(mapping_table):
                alpha_values[data - 32768 > threshold] = 255 - i
            rgb[3] = np.clip(alpha_values, 0, 255)

            rgb[0] = data // 256
            rgb[1] = np.floor(data % 256)
            rgb[2] = np.floor((data - np.floor(data)) * 256)
            return rgb
        else:
            rgb = np.zeros((3, rows, cols), dtype=np.uint8)
            if(encoding == "terrarium"):
                rgb[0] = np.floor(data // 256)
                rgb[1] = np.floor(data % 256)
                rgb[2] = np.floor((data - np.floor(data)) * 256)
            else:
                rgb[0] = np.floor((data / (256 * 256)) % 256).astype(np.uint8)
                rgb[1] = np.floor((data / 256) % 256).astype(np.uint8)
                rgb[2] = np.floor(data % 256).astype(np.uint8)
            return rgb
    
    @staticmethod
    def _decode(data: np.ndarray, base: float, interval: float, encoding: str) -> np.ndarray:
        """
        Utility to decode RGB encoded data

        Parameters
        ----------
        data: np.ndarray
            RGB data to decode
        base: float
            Base value for mapbox encoding
        interval: float
            Interval value for mapbox encoding
        encoding: str
            Encoding type ('terrarium' or 'mapbox')

        Returns
        -------
        np.ndarray
            Decoded elevation data
        """
        data = data.astype(np.float64)
        if(encoding == "terrarium"):
            return (data[0] * 256 + data[1] + data[2] / 256) - 32768
        else:
            return base + (((data[0] * 256 * 256) + (data[1] * 256) + data[2]) * interval)

    @staticmethod
    def _mask_elevation(elevation: np.ndarray, mask_values: list = [0.0]) -> np.ndarray:
        """
        Mask specific elevation values with NaN

        Parameters
        ----------
        elevation: np.ndarray
            Elevation data array
        mask_values: list
            List of values to mask with NaN. Default is [0.0]

        Returns
        -------
        np.ndarray
            Masked elevation array
        """
        mask = np.zeros_like(elevation, dtype=bool)
        for mask_value in mask_values:
            mask = np.logical_or(mask, elevation == mask_value)
        return np.where(mask, np.nan, elevation)
    
    @staticmethod
    def _range_check(datarange):
        """
        Utility to check if data range is outside of precision for 3 digit base 256
        """
        maxrange = 256 ** 3
        return datarange > maxrange
    
    @staticmethod
    def _generate_mapping_table():
        """
        Generate elevation mapping table for alpha channel quantization
        """
        table = []
        for i in range(0, 11):
            table.append(-11000 + i * 1000)
        table.append(-100)
        table.append(-50)
        table.append(-20)
        table.append(-10)
        table.append(-1)
        for i in range(0, 150):
            table.append(20 * i)
        for i in range(0, 60):
            table.append(3000 + 50 * i)
        for i in range(0, 29):
            table.append(6000 + 100 * i)
        return table

    @staticmethod
    def encode_as_webp(data, profile=None, affine=None):
        """
        Uses BytesIO + PIL to encode a (3 or 4, height, width)
        array into a webp bytearray.

        Parameters
        ----------
        data: ndarray
            (3 or 4 x height x width) uint8 RGB array
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
            if data.ndim == 3:
                im = Image.fromarray(np.moveaxis(data, 0, 3))
            elif data.ndim == 4:
                im = Image.fromarray(np.moveaxis(data, 0, 3), mode='RGBA')
            else:
                raise ValueError("unexpected number of image dimensions")

            im.save(f, format="webp", lossless=True)
            return f.getvalue()

    @staticmethod
    def encode_as_png(data, profile, dst_transform):
        """
        Uses rasterio's virtual file system to encode a (3 or 4, height, width)
        array as a png-encoded bytearray.

        Parameters
        ----------
        data: ndarray
            (3 or 4 x height x width) uint8 RGB array
        profile: dictionary
            dictionary of kwargs for png writing
        dst_transform: Affine
            affine transform for output tile

        Returns
        --------
        contents: bytearray
            png-encoded bytearray of the provided input data
        """
        profile = profile.copy()  # Create copy to avoid modifying input
        profile["affine"] = dst_transform
        with rasterio.open("/vsimem/tileimg", "w", **profile) as dst:
            if data.ndim == 3:
                dst.write(data)
            elif data.ndim == 4:
                dst.write(np.moveaxis(data, 0, -1))
            else:
                raise ValueError(f"Unexpected number of dimensions {data.ndim}")
        contents = bytearray(virtual_file_to_buffer("/vsimem/tileimg"))
        return contents

    @staticmethod
    def save_rgb_to_bytes(rgb_data: np.ndarray, output_image_format: "ImageFormat", default_tile_size: int = 512) -> bytes:
        """
        Converts a numpy array to the specified image format

        Parameters
        ----------
        rgb_data: np.ndarray
            a (3 x height x width) or (4 x height x width) ndarray with the RGB values
        output_image_format: ImageFormat
            the format that the array should be encoded to
        default_tile_size: int
            The tile size of the image to create

        Returns
        -------
        bytes:
            bytes for a image encoded to the given output format
        """
        print(f"save_rgb_to_bytes called with rgb data shape {rgb_data.shape}")
        image_bytes = bytearray() # changed from BytesIO to bytearray
        if rgb_data.size > 0:
            try:
                if rgb_data.ndim == 3:
                    image = Image.fromarray(np.moveaxis(rgb_data, 0, -1).astype(np.uint8), 'RGB')
                elif rgb_data.ndim == 4:
                    image = Image.fromarray(np.moveaxis(rgb_data, 0, -1).astype(np.uint8), 'RGBA')
                else:
                  tile_size = default_tile_size
                  image = Image.fromarray(np.moveaxis(np.zeros((3,tile_size,tile_size),dtype=np.uint8), 0, -1), 'RGB')

                print(f"image created with shape: {np.array(image).shape}")
                
                if output_image_format == ImageFormat.PNG:
                    image.save(image_bytes, format='PNG')
                elif output_image_format == ImageFormat.WEBP:
                    image.save(image_bytes, format='WEBP', lossless=True)
            
                print(f"image_bytes size {len(image_bytes)}")
                
            except Exception as e:
                logging.error(f"Failed to encode image: {e}")
        else:
            print("rgb_data size is 0")
        return bytes(image_bytes) # return as bytestring
