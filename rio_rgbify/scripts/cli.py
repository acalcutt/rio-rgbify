import click
import logging
from pathlib import Path
import json
from rio_rgbify.mbtiler import RGBTiler
from rio_rgbify.merger import TerrainRGBMerger, MBTilesSource, EncodingType
from rio_rgbify.raster_merger import RasterRGBMerger, RasterSource
from rio_rgbify.image import ImageFormat
from rasterio.enums import Resampling

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@click.group(
    context_settings=dict(help_option_names=["-h", "--help"])
)
@click.version_option()
def main_group():
    """rio: Command line interface for raster processing"""
    pass


@main_group.command('rgbify', short_help="Create RGB encoded tiles from a raster file.")
@click.argument("src_path", type=click.Path(exists=True))
@click.argument("dst_path", type=click.Path(exists=False))
@click.option(
    "--base-val",
    "-b",
    type=float,
    default=0,
    help="The base value of which to base the output encoding on [DEFAULT=0]",
)
@click.option(
    "--interval",
    "-i",
    type=float,
    default=1,
    help="Describes the precision of the output, by incrementing interval [DEFAULT=1]",
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
    help="Maximum zoom to tile (.mbtiles output only)",
)
@click.option(
    "--bounding-tile",
    type=str,
    default=None,
    help="Bounding tile '[, , ]' to limit output tiles (.mbtiles output only)",
)
@click.option(
    "--min-z",
    type=int,
    default=None,
    help="Minimum zoom to tile (.mbtiles output only)",
)
@click.option(
    "--format",
    type=click.Choice(["png", "webp"]),
    default="png",
    help="Output tile format (.mbtiles output only)",
)
@click.option("--workers", "-j", type=int, default=4, help="Workers to run [DEFAULT=4]")
@click.option("--verbose", "-v", is_flag=True, default=False)
@click.option(
    "--batch-size", type=int, default=None,
    help="Number of tiles to process at a time in each process."
)
@click.option(
    "--resampling", type=click.Choice(["nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average", "mode", "gaussian"], case_sensitive=False), default="nearest",
    help="Resampling method"
)
# @click.pass_context
# @creation_options
def rgbify(
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
    workers,
    verbose,
    batch_size,
    resampling
):
    """rio-rgbify cli."""

    if min_z is None or max_z is None:
            raise ValueError("Zoom range must be provided for mbtile output")

    if max_z < min_z:
        raise ValueError(
            "Max zoom {0} must be greater than min zoom ".format(max_z, min_z)
        )

    if bounding_tile is not None:
        try:
            bounding_tile = json.loads(bounding_tile)
        except Exception:
            raise TypeError(
                "Bounding tile of {0} is not valid".format(bounding_tile)
            )
    
    resampling_enum = Resampling[resampling.lower()]
    

    with RGBTiler(
        src_path,
        dst_path,
        interval=interval,
        base_val=base_val,
        round_digits=round_digits,
        encoding=encoding,
        format=format,
        bounding_tile=bounding_tile,
        max_z=max_z,
        min_z=min_z,
         resampling=resampling_enum,
    ) as tiler:
        tiler.run(workers, batch_size = batch_size)

@main_group.command('merge', short_help='Merge multiple MBTiles or Raster files.')
@click.option(
    "--config", "-c", type=click.Path(exists=True),
    help="Configuration file"
)
@click.option(
    "--output-path", "-o", type=str, default="output.mbtiles",
    help="Path to the output mbtiles."
)
@click.option(
    "-j", "--workers", type=int, default=None,
    help="Number of processes to use for parallel execution."
)

@click.option(
    "-z", "--min-zoom", type=int, default=None,
    help="Minimum zoom level to generate."
)
def merge(config, output_path, workers, min_zoom):
    """Merge multiple MBTiles files."""
    try:
        with open(config) as f:
            config = json.load(f)

        sources = []
        output_type = config.get('output_type', 'mbtiles')

        if output_type.lower() != 'mbtiles' and output_type.lower() != 'raster':
            logging.error("Invalid output_type, please use `mbtiles` or `raster`")
            raise Exception(f"Invalid output_type: ")

        for source in config['sources']:
           
            source_type = source.get('source_type','mbtiles') # Default to mbtiles if source_type is not set
            if source_type.lower() != 'mbtiles' and source_type.lower() != 'raster':
                logging.error("Invalid source_type, please use `mbtiles` or `raster`")
                raise Exception(f"Invalid source_type: ")
            
            if source_type.lower() == 'mbtiles':
                sources.append(
                    MBTilesSource(
                        path=Path(source["path"]),
                        encoding=EncodingType(source.get("encoding", "mapbox").lower()),
                        height_adjustment=source.get("height_adjustment", 0.0),
                        base_val=source.get("base_val", -10000),
                        interval=source.get("interval", 0.1),
                        mask_values=source.get("mask_values", [0.0])
                    )
                )
            elif source_type.lower() == 'raster':
                sources.append(
                    RasterSource(
                        path=Path(source["path"]),
                        height_adjustment=source.get("height_adjustment", 0.0),
                        base_val=source.get("base_val", -10000),
                        interval=source.get("interval", 0.1),
                        mask_values=source.get("mask_values", [0.0])
                        )
                    )

        if output_type.lower() == 'mbtiles':
            merger = TerrainRGBMerger(
                sources,
                output_path=output_path,
                output_encoding=EncodingType(config.get('output_encoding', "mapbox").lower()),
                output_image_format=ImageFormat(config.get('output_format', 'webp').lower()),
                resampling=Resampling[config.get('resampling', 'lanczos').lower()],
                output_quantized_alpha=config.get('output_quantized_alpha', False),
                min_zoom=min_zoom if min_zoom is not None else config.get("min_zoom", 0),
                max_zoom=config.get("max_zoom", None),
                bounds=config.get("bounds", None),
                gaussian_blur_sigma=config.get("gaussian_blur_sigma", 0.2),
                processes=workers
            )
        elif output_type.lower() == 'raster':
            merger = RasterRGBMerger(
                sources,
                output_path=output_path,
                output_encoding=EncodingType(config.get('output_encoding', "mapbox").lower()),
                output_image_format=ImageFormat(config.get('output_format', 'webp').lower()),
                resampling=Resampling[config.get('resampling', 'lanczos').lower()],
                output_quantized_alpha=config.get('output_quantized_alpha', False),
                min_zoom=min_zoom if min_zoom is not None else config.get("min_zoom", 0),
                max_zoom=config.get("max_zoom", None),
                bounds=config.get("bounds", None),
                gaussian_blur_sigma=config.get("gaussian_blur_sigma", 0.2),
                processes=workers
            )


        merger.process_all(min_zoom=min_zoom if min_zoom is not None else 0)
    except Exception as e:
        logging.error(f"An error occured: {e}")

if __name__ == "__main__":
    main_group()
