import os
import json
import click
from click.testing import CliRunner

import numpy as np

import rasterio as rio
from rio_rgbify.scripts.cli import cli, rgbify, merge

from raster_tester.compare import affaux, upsample_array


in_elev_src = os.path.join(os.path.dirname(__file__), "fixtures", "elev.tif")
expected_src = os.path.join(os.path.dirname(__file__), "expected", "elev-rgb.tif")


def flex_compare(r1, r2, thresh=10):
    upsample = 4
    r1 = r1[::upsample]
    r2 = r2[::upsample]
    toAff, frAff = affaux(upsample)
    r1 = upsample_array(r1, upsample, frAff, toAff)
    r2 = upsample_array(r2, upsample, frAff, toAff)
    tdiff = np.abs(r1.astype(np.float64) - r2.astype(np.float64))

    click.echo(
        "{0} values exceed the threshold difference with a max variance of {1}".format(
            np.sum(tdiff > thresh), tdiff.max()
        ),
        err=True,
    )

    return not np.any(tdiff > thresh)


def test_cli_good_elev():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            rgbify,
            [in_elev_src, "rgb.tif", "--interval", 0.001, "--base-val", -100, "-j", 1],
        )

        assert result.exit_code == 0

        with rio.open("rgb.tif") as created:
            with rio.open(expected_src) as expected:
                carr = created.read()
                earr = expected.read()
                for a, b in zip(carr, earr):
                    assert flex_compare(a, b)


def test_cli_fail_elev():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            rgbify,
            [
                in_elev_src,
                "rgb.tif",
                "--interval",
                0.00000001,
                "--base-val",
                -100,
                "-j",
                1,
            ],
        )
        assert result.exit_code == 1
        assert result.exception


def test_mbtiler_webp():
    runner = CliRunner()
    with runner.isolated_filesystem():
        out_mbtiles_finer = "output-0-dot-1.mbtiles"
        result_finer = runner.invoke(
            rgbify,
            [
                in_elev_src,
                out_mbtiles_finer,
                "--interval",
                0.1,
                "--min-z",
                10,
                "--max-z",
                11,
                "--format",
                "webp",
                "-j",
                1,
            ],
        )
        assert result_finer.exit_code == 0

        out_mbtiles_coarser = "output-1.mbtiles"
        result_coarser = runner.invoke(
            rgbify,
            [
                in_elev_src,
                out_mbtiles_coarser,
                "--min-z",
                10,
                "--max-z",
                11,
                "--format",
                "webp",
                "-j",
                1,
            ],
        )
        assert result_coarser.exit_code == 0

        assert os.path.getsize(out_mbtiles_finer) > os.path.getsize(out_mbtiles_coarser)


def test_mbtiler_png():
    runner = CliRunner()
    with runner.isolated_filesystem():
        out_mbtiles_finer = "output-0-dot-1.mbtiles"
        result_finer = runner.invoke(
            rgbify,
            [
                in_elev_src,
                out_mbtiles_finer,
                "--interval",
                0.1,
                "--min-z",
                10,
                "--max-z",
                11,
                "--format",
                "png",
            ],
        )
        assert result_finer.exit_code == 0

        out_mbtiles_coarser = "output-1.mbtiles"
        result_coarser = runner.invoke(
            rgbify,
            [
                in_elev_src,
                out_mbtiles_coarser,
                "--min-z",
                10,
                "--max-z",
                11,
                "--format",
                "png",
                "-j",
                1,
            ],
        )
        assert result_coarser.exit_code == 0

        assert os.path.getsize(out_mbtiles_finer) > os.path.getsize(out_mbtiles_coarser)


def test_mbtiler_png_bounding_tile():
    runner = CliRunner()
    with runner.isolated_filesystem():
        out_mbtiles_not_limited = "output-not-limited.mbtiles"
        result_not_limited = runner.invoke(
            rgbify,
            [
                in_elev_src,
                out_mbtiles_not_limited,
                "--min-z",
                12,
                "--max-z",
                12,
                "--format",
                "png",
            ],
        )
        assert result_not_limited.exit_code == 0

        out_mbtiles_limited = "output-limited.mbtiles"
        result_limited = runner.invoke(
            rgbify,
            [
                in_elev_src,
                out_mbtiles_limited,
                "--min-z",
                12,
                "--max-z",
                12,
                "--format",
                "png",
                "--bounding-tile",
                "[654, 1582, 12]",
            ],
        )
        assert result_limited.exit_code == 0

        assert os.path.getsize(out_mbtiles_not_limited) > os.path.getsize(
            out_mbtiles_limited
        )

        result_badtile = runner.invoke(
            rgbify,
            [
                in_elev_src,
                out_mbtiles_limited,
                "--min-z",
                12,
                "--max-z",
                12,
                "--format",
                "png",
                "--bounding-tile",
                "654-1582-12",
            ],
        )
        assert result_badtile.exit_code == 1
        assert "is not valid" in str(result_badtile.exception)


def test_mbtiler_webp_badzoom():
    runner = CliRunner()
    with runner.isolated_filesystem():
        out_mbtiles = "output.mbtiles"
        result = runner.invoke(
            rgbify,
            [
                in_elev_src,
                out_mbtiles,
                "--min-z",
                10,
                "--max-z",
                9,
                "--format",
                "webp",
                "-j",
                1,
            ],
        )
        assert result.exit_code == 1
        assert result.exception


def test_mbtiler_webp_badboundingtile():
    runner = CliRunner()
    with runner.isolated_filesystem():
        out_mbtiles = "output.mbtiles"
        result = runner.invoke(
            rgbify,
            [
                in_elev_src,
                out_mbtiles,
                "--min-z",
                10,
                "--max-z",
                9,
                "--format",
                "webp",
                "--bounding-tile",
                "654, 1582, 12",
            ],
        )
        assert result.exit_code == 1
        assert result.exception


def test_mbtiler_webp_badboundingtile_values():
    runner = CliRunner()
    with runner.isolated_filesystem():
        out_mbtiles = "output.mbtiles"
        result = runner.invoke(
            rgbify,
            [
                in_elev_src,
                out_mbtiles,
                "--min-z",
                10,
                "--max-z",
                9,
                "--format",
                "webp",
                "--bounding-tile",
                "[654, 1582]",
            ],
        )
        assert result.exit_code == 1
        assert result.exception


def test_bad_input_format():
    runner = CliRunner()
    with runner.isolated_filesystem():
        out_mbtiles = "output.lol"
        result = runner.invoke(
            rgbify,
            [
                in_elev_src,
                out_mbtiles,
                "--min-z",
                10,
                "--max-z",
                9,
                "--format",
                "webp",
                "-j",
                1,
            ],
        )
        assert result.exit_code == 1
        assert result.exception


def test_merge_command():
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create a sample config file
        config_data = {
            "sources": [
                {"path": "test1.mbtiles", "encoding": "mapbox", "height_adjustment": 5},
                {"path": "test2.mbtiles", "encoding": "terrarium", "height_adjustment": -10}
                ],
            "output_path": "merged.mbtiles",
            "output_format": "webp",
            "output_encoding": "mapbox",
            "resampling": "bilinear",
            "output_quantized_alpha": True
        }
        
        with open("config.json", "w") as f:
            json.dump(config_data, f)

        # Create dummy mbtiles files
        open("test1.mbtiles", "w").close()
        open("test2.mbtiles", "w").close()

        result = runner.invoke(
            cli,
            [
                "merge",
                "--config",
                "config.json",
                "--output-quantized-alpha",
                "-j",
                "1"
            ]
        )
        assert result.exit_code == 0
        assert os.path.exists("merged.mbtiles")
        
def test_merge_command_no_quantized():
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create a sample config file
        config_data = {
            "sources": [
                {"path": "test1.mbtiles", "encoding": "mapbox", "height_adjustment": 5},
                {"path": "test2.mbtiles", "encoding": "terrarium", "height_adjustment": -10}
                ],
            "output_path": "merged.mbtiles",
            "output_format": "png",
            "output_encoding": "mapbox",
            "resampling": "bilinear"
        }
        
        with open("config.json", "w") as f:
            json.dump(config_data, f)

        # Create dummy mbtiles files
        open("test1.mbtiles", "w").close()
        open("test2.mbtiles", "w").close()

        result = runner.invoke(
            cli,
            [
                "merge",
                "--config",
                "config.json",
                "-j",
                 "1"
            ]
        )
        assert result.exit_code == 0
        assert os.path.exists("merged.mbtiles")


def test_mbtiler_resampling_cli():
  runner = CliRunner()
  with runner.isolated_filesystem():
    out_mbtiles = "output.mbtiles"
    result = runner.invoke(
        rgbify,
        [
            in_elev_src,
            out_mbtiles,
            "--min-z",
            10,
            "--max-z",
            11,
            "--format",
            "png",
            "--resampling",
            "nearest",
            "-j",
            1,
        ],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        rgbify,
        [
            in_elev_src,
            out_mbtiles,
            "--min-z",
            10,
            "--max-z",
            11,
            "--format",
            "png",
            "--resampling",
            "bilinear",
            "-j",
            1,
        ],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        rgbify,
        [
            in_elev_src,
            out_mbtiles,
            "--min-z",
            10,
            "--max-z",
            11,
            "--format",
            "png",
            "--resampling",
            "cubic",
            "-j",
            1,
        ],
    )
    assert result.exit_code == 0
    result = runner.invoke(
        rgbify,
        [
            in_elev_src,
            out_mbtiles,
            "--min-z",
            10,
            "--max-z",
            11,
            "--format",
            "png",
            "--resampling",
            "cubic_spline",
             "-j",
            1,
        ],
    )
    assert result.exit_code == 0
    result = runner.invoke(
        rgbify,
        [
            in_elev_src,
            out_mbtiles,
            "--min-z",
            10,
            "--max-z",
            11,
            "--format",
            "png",
            "--resampling",
            "lanczos",
            "-j",
             1,
        ],
    )
    assert result.exit_code == 0
    result = runner.invoke(
        rgbify,
        [
            in_elev_src,
            out_mbtiles,
            "--min-z",
            10,
            "--max-z",
            11,
            "--format",
            "png",
            "--resampling",
            "average",
            "-j",
            1,
        ],
    )
    assert result.exit_code == 0
    result = runner.invoke(
        rgbify,
        [
            in_elev_src,
            out_mbtiles,
            "--min-z",
            10,
            "--max-z",
            11,
            "--format",
            "png",
             "--resampling",
            "mode",
            "-j",
            1,
        ],
    )
    assert result.exit_code == 0
    result = runner.invoke(
        rgbify,
        [
            in_elev_src,
            out_mbtiles,
            "--min-z",
            10,
            "--max-z",
            11,
            "--format",
            "png",
             "--resampling",
            "gauss",
            "-j",
            1,
        ],
    )
    assert result.exit_code == 0


def test_mbtiler_quantized_alpha_cli():
    runner = CliRunner()
    with runner.isolated_filesystem():
        out_mbtiles = "output.mbtiles"
        result = runner.invoke(
            rgbify,
            [
                in_elev_src,
                out_mbtiles,
                "--min-z",
                10,
                "--max-z",
                11,
                "--format",
                "png",
                "--encoding",
                "terrarium",
                 "--quantized-alpha",
                "-j",
                1,
            ],
        )
        assert result.exit_code == 0
        
        result = runner.invoke(
            rgbify,
            [
                in_elev_src,
                out_mbtiles,
                "--min-z",
                10,
                "--max-z",
                11,
                "--format",
                "png",
                "--encoding",
                "terrarium",
                 "-j",
                 1,
            ],
        )
        assert result.exit_code == 0
        
def test_mbtiler_baseval_cli():
  runner = CliRunner()
  with runner.isolated_filesystem():
    out_mbtiles = "output.mbtiles"
    result = runner.invoke(
        rgbify,
        [
            in_elev_src,
            out_mbtiles,
            "--min-z",
            10,
            "--max-z",
            11,
            "--format",
            "png",
            "--encoding",
            "mapbox",
            "--base-val",
            "-500",
            "-j",
            1,
        ],
    )
    assert result.exit_code == 0
