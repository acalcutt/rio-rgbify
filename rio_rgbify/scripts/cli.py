"""rio_rgbify CLI."""

import click
import rasterio as rio
import numpy as np
from riomucho import RioMucho
import json
from rasterio.rio.options import creation_options
from rasterio.enums import Resampling
from pathlib import Path
from rio_rgbify.mbtiler import RGBTiler
from rio_rgbify.merger import TerrainRGBMerger, MBTilesSource, EncodingType, ImageFormat
from rio_rgbify.image import ImageEncoder
from typing import Dict
import sqlite3


def _rgb_worker(data, window, ij, g_args):
    return ImageEncoder.data_to_rgb(
        data[0][g_args["bidx"] - 1], g_args["encoding"], g_args["interval"], g_args["round_digits"], g_args["base_val"]
    )


@click.group(name="rgbify")
def cli():
    """Commands to generate rgb terrain tiles"""
    pass


@cli.command("rgbify")
@click.argument("src_path", type=click.Path(exists=True))
@click.argument("dst_path", type=click.Path(exists=False))
@click.option(
    "--base-val",
    "-b",
    type=float,
    default=0,
    help="The base value of which to base the output encoding on (Mapbox only) [DEFAULT=0]",
)
@click.option(
    "--interval",
    "-i",
    type=float,
    default=1,
    help="Describes the precision of the output, by incrementing interval (Mapbox only) [DEFAULT=1]",
)
@click.option(
    "--round-digits",
    "-r",
    type=int,
    default=0,
    help="Less significants encoded bits to be set to 0. Round the values, but have better images compression [DEFAULT=0]",
)
@click.option(
    "--encoding",
    "-e",
    type=click.Choice(["mapbox", "terrarium"]),
    default="mapbox",
    help="RGB encoding to use on the tiles",
)
@click.option("--bidx", type=int, default=1, help="Band to encode [DEFAULT=1]")
@click.option(
    "--max-z",
    type=int,
    default=None,
    help="Maximum zoom to tile",
)
@click.option(
    "--bounding-tile",
    type=str,
    default=None,
    help="Bounding tile '[, , ]' to limit output tiles",
)
@click.option(
    "--min-z",
    type=int,
    default=None,
    help="Minimum zoom to tile",
)
@click.option(
    "--format",
    type=click.Choice(["png", "webp"]),
    default="png",
    help="Output tile format",
)
@click.option(
    "--resampling",
    type=click.Choice(["nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average", "mode", "gauss"]),
    default="bilinear",
    help="Output tile resampling method",
)
@click.option(
    "--quantized-alpha",
    is_flag=True,
    default=False,
    help="If true, will add a quantized alpha channel to terrarium tiles (Terrarium Only)",
)
@click.option("--workers", "-j", type=int, default=4, help="Workers to run [DEFAULT=4]")
@click.option("--batch-size", type=int, default=None, help="Batch size for multiprocessing")
@click.option("--verbose", "-v", is_flag=True, default=False)
@click.pass_context
@creation_options
def rgbify(
    ctx,
    src_path,
    dst_path,
    base_val,
    interval,
    round_digits,
    encoding,
    bidx,
    max_z,
    min_z,
    bounding_tile,
    format,
    resampling,
    quantized_alpha,
    workers,
    batch_size,
    verbose,
    creation_options,
):
    """rio-rgbify cli."""
    if dst_path.split(".")[-1].lower() == "tif":
        with rio.open(src_path) as src:
            meta = src.profile.copy()

        meta.update(count=3, dtype=np.uint8)

        for c in creation_options:
            meta[c] = creation_options[c]

        gargs = {"interval": interval, "encoding": encoding, "base_val": base_val, "round_digits": round_digits, "bidx": bidx}

        with RioMucho(
            [src_path], dst_path, _rgb_worker, options=meta, global_args=gargs
        ) as rm:
            rm.run(workers)

    elif dst_path.split(".")[-1].lower() == "mbtiles":
        if min_z is None or max_z is None:
            raise ValueError("Zoom range must be provided for mbtile output")

        if max_z < min_z:
            raise ValueError(
                "Max zoom {0} must be greater than min zoom {1}".format(max_z, min_z)
            )

        if bounding_tile is not None:
            try:
                bounding_tile = json.loads(bounding_tile)
            except Exception:
                raise TypeError(
                    "Bounding tile of {0} is not valid".format(bounding_tile)
                )
        
        resampling_enum = getattr(Resampling, resampling)

        with RGBTiler(
            src_path,
            dst_path,
            min_z=min_z,
            max_z=max_z,
            interval=interval,
            base_val=base_val,
            round_digits=round_digits,
            encoding=encoding,
            resampling=resampling_enum,
            quantized_alpha=quantized_alpha,
            bounding_tile=bounding_tile
        ) as tiler:
            tiler.run(processes=workers, batch_size=batch_size)

    else:
        raise ValueError(
            "{} output filetype not supported".format(dst_path.split(".")[-1])
        )


@cli.command("merge")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to the JSON configuration file",
)
@click.option("--workers", "-j", type=int, default=4, help="Workers to run [DEFAULT=4]")
@click.option("--verbose", "-v", is_flag=True, default=False)
def merge(config, workers, verbose):
    """Merges multiple MBTiles files into one."""
    with open(config, "r") as f:
        config_data = json.load(f)

    sources = []
    source_connections = {}
    for source_data in config_data.get("sources", []):
        source = MBTilesSource(
            path=Path(source_data["path"]),
            encoding=EncodingType(source_data.get("encoding", "mapbox")),
            height_adjustment=float(source_data.get("height_adjustment", 0.0)),
            base_val=float(source_data.get("base_val",-10000)),
            interval=float(source_data.get("interval",0.1)),
            mask_values = config_data.get("mask_values",[0.0, -1.0])
        )
        source_connections[source.path] = sqlite3.connect(source.path)
        sources.append(source)
    
    output_path = Path(config_data.get("output_path", "output.mbtiles"))
    output_encoding = EncodingType(config_data.get("output_encoding", "mapbox"))
    output_format = ImageFormat(config_data.get("output_format", "png"))
    resampling_str = config_data.get("resampling","bilinear")
    if resampling_str.lower() not in ["nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average", "mode", "gauss"]:
      raise ValueError(f" is not a supported resampling method! {resampling_str}")
    resampling = Resampling[resampling_str]
    output_quantized_alpha = config_data.get("output_quantized_alpha", False)
    min_zoom = config_data.get("min_zoom", 0)
    max_zoom = config_data.get("max_zoom", None)
    bounds = config_data.get("bounds", None)
    
    if bounds is not None:
      try:
        bounds = [float(x) for x in bounds]
        if len(bounds) != 4:
          raise ValueError("Bounds must be a list of 4 floats in the order west, south, east, north")
      except Exception:
        raise TypeError(
          "Bounding box of  is not valid, must be a comma seperated list of 4 floats in the order west, south, east, north".format(bounds)
        )

    merger = TerrainRGBMerger(
        sources = sources,
        output_path = output_path,
        output_encoding = output_encoding,
        output_image_format = output_format,
        resampling = resampling,
        processes = workers,
        output_quantized_alpha = output_quantized_alpha,
        min_zoom = min_zoom,
        max_zoom = max_zoom,
        bounds = bounds,
    )
    
    try:
        merger.process_all(min_zoom=min_zoom if min_zoom is not None else 0)
    finally:
        for conn in source_connections.values():
            conn.close()
