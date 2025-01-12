from __future__ import division
import numpy as np


class Encoder:
    @staticmethod
    def data_to_rgb(data, encoding, interval, base_val=-10000, round_digits=0):
        """
        Given an arbitrary (rows x cols) ndarray,
        encode the data into uint8 RGB from an arbitrary
        base and interval

        Parameters
        -----------
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

        Returns
        --------
        ndarray: rgb data
            a uint8 (3 x rows x cols) ndarray with the
            data encoded
        """
        data = data.astype(np.float64)
        if(encoding == "terrarium"):
            data = np.clip(data, -32768, 32767)
            data += 32768
        else:
            data -= base_val  # Apply offset
            data /= interval
            data = np.clip(data, base_val, 10000)

        data = np.around(data / 2**round_digits) * 2**round_digits

        rows, cols = data.shape
        rgb = np.zeros((3, rows, cols), dtype=np.uint8)

        if(encoding == "terrarium"):
            rgb[0] = data // 256
            rgb[1] = np.floor(data % 256)
            rgb[2] = np.floor((data - np.floor(data)) * 256)
        else:
            rgb[0] = ((((data // 256) // 256) / 256) - (((data // 256) // 256) // 256)) * 256
            rgb[1] = (((data // 256) / 256) - ((data // 256) // 256)) * 256
            rgb[2] = ((data / 256) - (data // 256)) * 256

        return rgb
    
    @staticmethod
    def _decode(data: np.ndarray, base: float, interval: float, encoding: str) -> np.ndarray:
        """
        Utility to decode RGB encoded data
        """
        data = data.astype(np.float64)
        if(encoding == "terrarium"):
            return (data[0] * 256 + data[1] + data[2] / 256) - 32768
        else:
            return base + (((data[0] * 256 * 256) + (data[1] * 256) + data[2]) * interval)

    @staticmethod
    def _mask_elevation(elevation: np.ndarray) -> np.ndarray:
        """
        Mask 0 and -1 elevation values with NaN
        """
        mask = np.logical_or(elevation == 0, elevation == -1)
        return np.where(mask, np.nan, elevation)
    

    @staticmethod
    def _range_check(datarange):
        """
        Utility to check if data range is outside of precision for 3 digit base 256
        """
        maxrange = 256 ** 3

        return datarange > maxrange
