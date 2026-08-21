# rio-rgbify

> **Fork:** This is a fork of [mapbox/rio-rgbify](https://github.com/mapbox/rio-rgbify), maintained by TechIdiots LLC under the same MIT License.

Encode arbitrary bit depth rasters in pseudo base-256 as RGB

## Installation

```
git clone https://github.com/acalcutt/rio-rgbify.git

cd rio-rgbify

pip install -e '.[test]'

```
## Required Packages on Ubuntu
To run `rio-rgbify` on Ubuntu, you will need to make sure you have the following installed:

*   `python3-dev`
*   `libspatialindex-dev`
*   `libgeos-dev`
*   `gdal-bin`
*   `python3-gdal`

You can install these using the following command:

```bash
sudo apt update
sudo apt install python3-dev libspatialindex-dev libgeos-dev gdal-bin python3-gdal
```

## CLI usage

`rio-rgbify` now has two subcommands `rgbify` and `merge`.

### `rgbify` Command

The `rgbify` command is used to encode a raster into RGB and output it as a GeoTIFF or an MBTiles file.

-   Input can be any raster readable by `rasterio`
-   Output can be a GeoTIFF or an MBTiles file, both created using tile-based processing.

```
Usage: rio rgbify [OPTIONS] SRC_PATH DST_PATH

  rio-rgbify cli.

Options:
  -b, --base-val FLOAT            The base value of which to base the output
                                  encoding on [DEFAULT=0]
  -i, --interval FLOAT            Describes the precision of the output, by
                                  incrementing interval [DEFAULT=1]
  -r, --round-digits INTEGER      Less significants encoded bits to be set to
                                  0. Round the values, but have better images
                                  compression [DEFAULT=0]
  -e, --encoding [mapbox|terrarium]
                                  RGB encoding to use on the tiles
  --bidx INTEGER                  Band to encode [DEFAULT=1]
  --max-z INTEGER                 Maximum zoom to tile (.mbtiles output only)
  --bounding-tile TEXT            Bounding tile '[, , ]' to limit output tiles
                                  (.mbtiles output only)
  --min-z INTEGER                 Minimum zoom to tile (.mbtiles output only)
  --format [png|webp]             Output tile format (.mbtiles output only)
  -j, --workers INTEGER           Workers to run [DEFAULT=4]
  -v, --verbose
  --batch-size INTEGER            Number of tiles to process at a time in each
                                  process.
  --resampling [nearest|bilinear|cubic|cubic_spline|lanczos|average|mode|gaussian]
                                  Resampling method
  --help                          Show this message and exit
```

### Mapbox TerrainRGB example

```
rio rgbify -e mapbox -b -10000 -i 0.1 --min-z 0 --max-z 8 -j 24 --format png SRC_PATH.vrt DST_PATH.mbtiles
```

### Mapzen Terrarium example

```
rio rgbify -e terrarium --min-z 0 --max-z 8 -j 24 --format png SRC_PATH.vrt DST_PATH.mbtiles
```

### `merge` Command

The `merge` command is used to merge multiple MBTiles or Raster files into one output MBTiles file. This is done by taking a JSON configuration file.

```
Usage: rio merge [OPTIONS]

Options:
 -c, --config PATH       Path to the JSON configuration file [required]
 -j, --workers INTEGER   Workers to run [DEFAULT=4]
 -z, --min-zoom INTEGER       Minimum zoom level to generate.
 -v, --verbose
 --help               Show this message and exit.
```

#### Configuration File

The `merge` command makes use of a json configuration file which should be passed in as the `--config` parameter. The JSON configuration file should follow the following structure:

```json
{
    "source_type": "mbtiles",
    "sources": [
        {
            "path": "/path/to/bathymetry.mbtiles",
            "encoding": "mapbox",
            "height_adjustment": -5.0
        },
        {
            "path": "/path/to/base_terrain.mbtiles",
            "encoding": "mapbox",
            "height_adjustment": 0.0,
            "base_val": -10000,
            "interval": 0.1,
             "mask_values": [-1,0]
        },
        {
            "path": "/path/to/secondary_terrain.mbtiles",
            "encoding": "terrarium",
            "height_adjustment": 10.0
        }
    ],
    "output_path": "/path/to/output.mbtiles",
    "output_encoding": "mapbox",
    "output_nodata": -9999,
    "output_format": "webp",
    "resampling": "bilinear",
    "sparse_tiles": true,
    "min_zoom": 2,
    "max_zoom": 10,
    "gaussian_blur_sigma": 0.2,
    "bounds": [-10,10,20,50],
    "bounds_source": 1
}
```

```json
{
    "source_type": "raster",
    "sources": [
        {
            "path": "/path/to/raster1.tif",
            "height_adjustment": -5.0,
            "base_val": -10000,
            "interval": 0.1,
            "mask_values": [0]
        },
        {
            "path": "/path/to/raster2.tif",
            "height_adjustment": 10.0,
            "mask_values": [-1,-32767]
        }
    ],
    "output_path": "/path/to/output.mbtiles",
    "output_nodata": -9999,
    "output_encoding": "terrarium",
    "output_format": "webp",
    "resampling": "bilinear",
    "sparse_tiles": true,
    "min_zoom": 2,
    "max_zoom": 10,
    "bounds": [-10,10,20,50],
    "bounds_source": 1
}
```

**Explanation:**

*   **`source_type` (Optional, Default: `mbtiles`):** This is a new key which tells the program whether the sources are `mbtiles` or `raster`.
*   **`sources` (Required):**
    *   A list of objects defining the input MBTiles or Raster files.
    *   **MBTiles Sources**:
        *   `path` (Required): The path to the MBTiles file.
        *   `encoding` (Optional, Default: `"mapbox"`): The encoding used for the MBTiles file (`"mapbox"` or `"terrarium"`).
        *   `height_adjustment` (Optional, Default: `0.0`): A floating-point value (in meters) to adjust the elevation of that particular input. Positive values raise the elevation, and negative values lower the elevation.
        *   `base_val` (Optional, Default: `-10000`): A floating-point value which will be the base value for mapbox encoded tiles, in meters.
        *    `interval` (Optional, Default: `0.1`): A floating-point value that represents the vertical distance between each level of encoded height.
        *   `mask_values` (Optional, Default `[0.0]`): A list of numbers representing the elevation values to mask.
     *   **Raster Sources**:
        *   `path` (Required): The path to the raster file.
        *   `height_adjustment` (Optional, Default: `0.0`): A floating-point value (in meters) to adjust the elevation of that particular input. Positive values raise the elevation, and negative values lower the elevation.
        *   `base_val` (Optional, Default: `-10000`): A floating-point value which will be the base value for mapbox encoded tiles, in meters.
        *   `interval` (Optional, Default: `0.1`): A floating-point value that represents the vertical distance between each level of encoded height.
        *   `mask_values` (Optional, Default `[0.0]`): A list of numbers representing the elevation values to mask.
*   `output_path` (Optional, Default: `"output.mbtiles"`): The output path for the merged MBTiles file.
*   `output_encoding` (Optional, Default: `"mapbox"`): The output encoding to use (`"mapbox"` or `"terrarium"`).
*   `output_nodata` (Optional, Default: `None`): The value to use for output nodata replacement.  If set, `NaN` values will be replaced with this value. If `None`, no nodata replacement is performed.
*   `output_format` (Optional, Default: `"png"`): The output image format (`"png"` or `"webp"`).
*   `resampling` (Optional, Default: `"bilinear"`): The method to use for resampling (`"nearest"`, `"bilinear"`, `"cubic"`, `"cubic_spline"`, `"lanczos"`, `"average"`, `"mode"`, or `"gauss"`).
*   `sparse_tiles` (Optional, Default: false): A boolean that determines whether to skip writing tiles that only contain upscaled data. If true, tiles consisting entirely of upscaled data will not be written to the output MBTiles file.
*   `min_zoom` (Optional, Default: `0`): The minimum zoom level to process.
*   `max_zoom` (Optional, Default: uses max from last file): The maximum zoom level to process.
*  `bounds` (Optional, Default: bounds of last file): A bounding box to limit the tiles being generated. Should be in the format: `[w,s,e,n]`. **Overrides `bounds_source` if set**.
*  `bounds_source` (Optional, Default: Uses last source): An integer which corresponds to the source in the `sources` array which should be used to generate bounds and tiles from. **The index starts at 0.**
* `gaussian_blur_sigma` (Optional, Default: `0.2`): A floating-point value that controls the base strength of the gaussian blur applied to source tiles during upscaling. The actual blur applied is scaled based on the zoom level difference between the source and the output tile.

  **Understanding Zoom-Level Dependent Blurring:**

   The `gaussian_blur_sigma` parameter no longer directly represents the amount of blur applied. Instead, it serves as a *base* value for the blur. The actual amount of blurring is now *dynamically* adjusted based on the zoom level difference between the source tile and the target tile:

    *   **Base Blur:** The `gaussian_blur_sigma` sets the starting point for how much blurring to apply.
    *   **Dynamic Adjustment:** When a tile is upscaled, the amount of blurring is scaled by the absolute difference between the source zoom level and the target zoom level. For example, if `gaussian_blur_sigma` is `0.2`, a tile that is upscaled by 2 zoom levels will have a sigma of 0.4 applied.
    *   **Adaptive Smoothing:** This means tiles that require significant upscaling receive more smoothing, reducing blockiness, while tiles that are closer to their target zoom receive less smoothing, preserving detail.
    *   **Linear Scaling**: The blurring is scaled linearly by the zoom difference. This value can be changed by multiplying by a different number, and will be considered in later versions.

    **Choosing a `gaussian_blur_sigma` Value:**

    As the `gaussian_blur_sigma` now acts as a base value, a good start is to aim for the ideal smoothing at a single zoom difference. For example if you want 0.4 smoothing when the zoom difference is 2, then use 0.2.
    Start with the default (`0.2`) and experiment, using higher values if the upscaling looks too "bumpy" or if you want more smoothing, and lower values if you think its too blurry.

The merge logic works by merging the input sources in order, applying the height adjustment as it merges. The last input source will be the base layer for tiles, and the bounds of this last file will be used if no bounds are passed in, unless a `bounds_source` parameter has been set.

## Merge Example

```
rio merge --config config.json -j 24
```
