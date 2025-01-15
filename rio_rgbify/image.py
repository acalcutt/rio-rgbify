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
        logging.debug(f"data_to_rgb called with shape: {data.shape}, encoding: {encoding}, interval: {interval}, base_val: {base_val}, round_digits: {round_digits}, quantized_alpha: {quantized_alpha}")
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array")

        data = data.astype(np.float64)
        if(encoding == "terrarium"):
            data = np.clip(data, -32768, 32767)
            data += 32768
        else:
            # CLAMP values before encoding for Mapbox encoding
            data = np.clip(data, base_val, 100000)
            data -= base_val   # Apply offset
            data /= interval
            
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
    def encode_as_webp(rgb, profile=None, affine=None):
        """Convert and RGB(A) numpy to bytes, force WebP encoding"""
        logging.debug(f"encode_as_webp called with rgb data shape {rgb.shape}")
        #Force the RGB to be uint8 for the webp
        rgb = rgb.astype(np.uint8)
        
        if rgb.shape[0] == 3:
            image = Image.fromarray(np.transpose(rgb, (1, 2, 0)), mode='RGB')
        elif rgb.shape[0] == 4:
            image = Image.fromarray(np.transpose(rgb, (1, 2, 0)), mode='RGBA')
        else:
            raise ValueError(f"Unsupported number of channels in RGB data: {rgb.shape[0]}")
        
        image_bytes = BytesIO()
        
        image.save(image_bytes, format='WEBP', lossless=True)
        image_bytes.seek(0)
        
        with rasterio.io.MemoryFile() as memfile:
            memfile.write(image_bytes.read())
            with memfile.open(driver="WEBP") as dataset:
                return memfile.read()
    
    @staticmethod
    def encode_as_png(rgb, kwargs, transform):
        """Convert and RGB(A) numpy to bytes, force PNG encoding"""
        logging.debug(f"encode_as_png called with rgb data shape {rgb.shape}")
        #Force the RGB to be uint8 for the png,
        rgb = rgb.astype(np.uint8)
        
        if rgb.shape[0] == 3:
            image = Image.fromarray(np.transpose(rgb, (1, 2, 0)), mode='RGB')
        elif rgb.shape[0] == 4:
            image = Image.fromarray(np.transpose(rgb, (1, 2, 0)), mode='RGBA')
        else:
            raise ValueError(f"Unsupported number of channels in RGB data: {rgb.shape[0]}")
        
        image_bytes = BytesIO()
        image.save(image_bytes, **kwargs)
        image_bytes.seek(0)
      
        with rasterio.io.MemoryFile() as memfile:
             memfile.write(image_bytes.read())
             with memfile.open(driver="PNG") as dataset:
                return memfile.read()

    @staticmethod
    def save_rgb_to_bytes(rgb_data: np.ndarray, output_image_format: str | ImageFormat, default_tile_size: int = 512) -> bytes:
        print(f"save_rgb_to_bytes called with rgb data shape {rgb_data.shape}")
        print(f"Requested format: {output_image_format}, type: {type(output_image_format)}")
        
        # Convert string to enum if needed
        if isinstance(output_image_format, str):
            try:
                output_image_format = ImageFormat(output_image_format.lower())
            except ValueError:
                print(f"Invalid format {output_image_format}, falling back to PNG")
                output_image_format = ImageFormat.PNG
        
        print(f"Using format: {output_image_format}")
        
        try:
            # Create image
            if rgb_data.ndim == 3:
                moved_data = np.moveaxis(rgb_data, 0, -1).astype(np.uint8)
                print(f"Moved data shape: {moved_data.shape}, dtype: {moved_data.dtype}")
                image = Image.fromarray(moved_data, 'RGB')
            elif rgb_data.ndim == 4:
                moved_data = np.moveaxis(rgb_data, 0, -1).astype(np.uint8)
                image = Image.fromarray(moved_data, 'RGBA')
            else:
                tile_size = default_tile_size
                image = Image.fromarray(np.moveaxis(np.zeros((3,tile_size,tile_size), dtype=np.uint8), 0, -1), 'RGB')
            
            print(f"Image created - size: {image.size}, mode: {image.mode}")
            
            with BytesIO() as f:
                if output_image_format == ImageFormat.WEBP:
                    print("Attempting to save as WebP")
                    image.save(f, format='WEBP', lossless=True)
                else:
                    print("Attempting to save as PNG")
                    image.save(f, format='PNG')
                
                f.seek(0)
                image_bytes = f.getvalue()
                print(f"Buffer size after save: {len(image_bytes)}")
                
                return bytes(image_bytes)
                
        except Exception as e:
            print(f"Failed to encode image: {str(e)}")
            raise