import click
import logging
from pathlib import Path
import json
from rio_rgbify.mbtiler import RGBTiler
from rio_rgbify.merger import TerrainRGBMerger, MBTilesSource, EncodingType
from rio_rgbify.raster_merger import RasterRGBMerger, RasterSource
from rio_rgbify.image import ImageFormat
from rasterio.enums import Resampling
from typing import List


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@click.group(
    context_settings=dict(help_option_names=["-h", "--help"])
)
@click.version_option()
def main_group():
    """rio: Command line interface for raster processing"""
    pass

@main_group.command('rgbify', short_help="Create RGB encoded tiles from a raster file.")
@click.argument("inpath", type=click.Path(exists=True))
@click.argument("outpath", type=str)
@click.option(
    "-z", "--min-zoom", type=int, default=0,
    help="Minimum zoom level to generate."
)
@click.option(
    "-Z", "--max-zoom", type=int, default=None,
    help="Maximum zoom level to generate."
)
@click.option(
    "-b", "--bounding-tile", type=str, default=None,
    help="Limit the generated tiles to a bounding tile in the format `x,y,z`. For example `427,701,11`"
)
@click.option(
    "-j", "--workers", type=int, default=None,
    help="Number of processes to use for parallel execution."
)
@click.option(
    "--batch-size", type=int, default=None,
    help="Number of tiles to process at a time in each process."
)
@click.option(
    "--interval", type=float, default=1,
    help="The interval at which to encode"
)
@click.option(
    "--baseval", type=float, default=0,
    help="The base value of the RGB numbering system"
)
@click.option(
    "--round-digits", type=int, default=0,
    help="Erase less significant digits"
)
@click.option(
    "--encoding", type=click.Choice(["mapbox", "terrarium"], case_sensitive=False), default="mapbox",
    help="Output tile encoding"
)
@click.option(
    "--format", type=click.Choice(["png", "webp"], case_sensitive=False), default="webp",
    help="Output tile image format"
)
@click.option(
    "--resampling", type=click.Choice(["nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average", "mode", "gaussian"], case_sensitive=False), default="nearest",
    help="Resampling method"
)
@click.option(
    "--quantized-alpha", type=bool, default=True,
    help="If set to true and using terrarium output encoding, the alpha channel will be populated with quantized data"
)
def rgbify(inpath, outpath, min_zoom, max_zoom, bounding_tile, workers, batch_size, interval, baseval, round_digits, encoding, format, resampling, quantized_alpha):
    """Create RGB encoded tiles from a raster file."""
    try:
        if bounding_tile is not None:
            bounding_tile = [int(v) for v in bounding_tile.split(",")]
        with RGBTiler(
            inpath,
            outpath,
            min_zoom,
            max_zoom,
            interval=interval,
            base_val=baseval,
            round_digits=round_digits,
            encoding=encoding,
            format=format,
            resampling=resampling,
            quantized_alpha=quantized_alpha,
            bounding_tile=bounding_tile
        ) as tiler:
            tiler.run(processes=workers, batch_size=batch_size)
    except Exception as e:
        logging.error(f"An error occured: {e}")


@main_group.command('merge', short_help='Merge multiple MBTiles or Raster files.')
@click.option(
    "--config", "-c", type=click.Path(exists=True),
    help="Configuration file"
)
@click.option(
    "-j", "--workers", type=int, default=None,
    help="Number of processes to use for parallel execution."
)
@click.option(
    "--batch-size", type=int, default=None,
    help="Number of tiles to process at a time in each process."
)
@click.option(
    "-z", "--min-zoom", type=int, default=None,
    help="Minimum zoom level to generate."
)
def merge(config, workers, batch_size, min_zoom):
    """Merge multiple MBTiles files."""
    try:
      
      with open(config) as f:
        config = json.load(f)

      sources = []
      for source in config['sources']:
          
        source_type = source.get('source_type','mbtiles') # Default to mbtiles if source_type is not set
        if source_type.lower() != 'mbtiles' and source_type.lower() != 'raster':
            logging.error("Invalid source_type, please use `mbtiles` or `raster`")
            raise Exception(f"Invalid source_type: {source_type}")
        
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

      if source_type.lower() == 'mbtiles':
        merger = TerrainRGBMerger(
            sources,
            output_path=config.get('output_path', 'output.mbtiles'),
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
      elif source_type.lower() == 'raster':
        merger = RasterRGBMerger(
            sources,
            output_path=config.get('output_path', 'output.mbtiles'),
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
