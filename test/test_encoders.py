from __future__ import division
from rio_rgbify.encoders import Encoder
import numpy as np
import pytest


def test_encode_data_roundtrip():
    minrand, maxrand = np.sort(np.random.randint(-427, 8848, 2))

    testdata = np.round((np.sum(
        np.dstack(
            np.indices((512, 512),
            dtype=np.float64)),
        axis=2) / (511. + 511.)) * maxrand, 2) + minrand

    baseval = -1000
    interval = 0.1
    round_digits = 0
    encoding = "mapbox"

    rtripped = Encoder._decode(Encoder.data_to_rgb(testdata.copy(), encoding, interval, round_digits=round_digits), baseval, interval, encoding)

    assert np.all(testdata == rtripped)

def test_encode_failrange():
    testdata = np.zeros((2))

    testdata[1] = 256 ** 3 + 1

    with pytest.raises(ValueError):
        Encoder.data_to_rgb(testdata, "mapbox", 1, 0)


def test_catch_range():
    assert Encoder._range_check(256 ** 3 + 1)
    assert not Encoder._range_check(256 ** 3 - 1)
