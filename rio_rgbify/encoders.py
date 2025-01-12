from __future__ import division
import numpy as np


class Encoder:
    @staticmethod
    def data_to_rgb(data, encoding, interval, round_digits=0, min_val = -10000, max_val = 10000):
        """
        Given an arbitrary (rows x cols) ndarray,
        encode the data into uint8 RGB from an arbitrary
        base and interval

        Parameters
        -----------
        data: ndarray
            (rows x cols) ndarray of data to encode
        interval: float
            the interval at which to encode
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
            data += 32768
            data = np.clip(data, 0, 65535) # clip values so they don't cause issues with encoding
        else:
            data /= interval
            data = np.clip(data, min_val, max_val) # clip values so they don't cause issues with encoding

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
    def _decode(data, base, interval, encoding):
        """
        Utility to decode RGB encoded data
        """
        data = data.astype(np.float64)
        if(encoding == "terrarium"):
            return (data[0] * 256 + data[1] + data[2] / 256) - 32768
        else:
            return base + (((data[0] * 256 * 256) + (data[1] * 256) + data[2]) * interval)

    @staticmethod
    def _range_check(datarange):
        """
        Utility to check if data range is outside of precision for 3 digit base 256
        """
        maxrange = 256 ** 3

        return datarange > maxrange