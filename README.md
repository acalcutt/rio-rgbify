# rio-rgbify
Encode arbitrary bit depth rasters in pseudo base-256 as RGB

## Installation

```
git clone https://github.com/acalcutt/rio-rgbify.git

cd rio-rgbify

pip install -e '.[test]'

```

## CLI usage

`rio-rgbify` now has two subcommands `rgbify` and `merge`.

### `rgbify` Command

The `rgbify` command is used to encode a raster into RGB and output it as a GeoTIFF or an MBTiles file

- Input can be any raster readable by `rasterio`
- Output can be any raster format writable by `rasterio` OR
- To create tiles _directly_ from data (recommended), output to an `.mbtiles`

```
Usage: rio rgbify rgbify [OPTIONS] SRC_PATH DST_PATH

Options:
    -b, --base-val FLOAT         The base value of which to base the output
                                  encoding on [DEFAULT=0]
    -i, --interval FLOAT         Describes the precision of the output, by
                                  incrementing interval [DEFAULT=1]
    -r, --round-digits INTEGER    Less significants encoded bits to be set to
                                  0. Round the values, but have better images
                                  compression [DEFAULT=0]
    -e, --encoding [mapbox|terrarium]   RGB encoding to use on the tiles
    --bidx INTEGER              Band to encode [DEFAULT=1]
    --max-z INTEGER             Maximum zoom to tile (.mbtiles output only)
    --bounding-tile TEXT        Bounding tile '[{x}, {y}, {z}]' to limit
                                  output tiles (.mbtiles output only)
    --min-z INTEGER             Minimum zoom to tile (.mbtiles output only)
    --format [png|webp]         Output tile format (.mbtiles output only)
    -j, --workers INTEGER        Workers to run [DEFAULT=4]
    -v, --verbose
    --co, --profile NAME=VALUE   Driver specific creation options. See the
                                  documentation for the selected output driver
                                  for more information.
    --help                      Show this message and exit.
```
## Mapbox TerrainRGB example

```
rio rgbify -e mapbox -b -10000 -i 0.1 --min-z 0 --max-z 8 -j 24 --format png SRC_PATH.vrt DST_PATH.mbtiles
```

## Mapzen Terrarium example

```
rio rgbify -e terrarium --min-z 0 --max-z 8 -j 24 --format png SRC_PATH.vrt DST_PATH.mbtiles
```

### `merge` Command

The `merge` command is used to merge multiple MBTiles files into one output MBTiles file. This is done by taking a JSON configuration file.

```
Usage: rio rgbify merge [OPTIONS]

Options:
  -c, --config PATH  Path to the JSON configuration file  [required]
  -j, --workers INTEGER        Workers to run [DEFAULT=4]
    -v, --verbose
  --help            Show this message and exit.
```

#### Configuration File

The `merge` command makes use of a json configuration file which should be passed in as the `--config` parameter. The JSON configuration file should follow the following structure:

```json
{
  "sources": [
    {
      "path": "/path/to/bathymetry.mbtiles",
      "encoding": "mapbox",
      "height_adjustment": 0.0
    },
    {
      "path": "/path/to/base_terrain.mbtiles",
      "encoding": "terrarium",
      "height_adjustment": 10.0
    },
    {
      "path": "/path/to/secondary_terrain.mbtiles",
      "encoding": "mapbox",
      "height_adjustment": -5.0
    }
  ],
  "output_path": "/path/to/output.mbtiles",
   "output_encoding": "mapbox",
  "output_format": "webp",
  "resampling": "bilinear"
}
```

**Explanation:**

*   **`sources` (Required):**
    *   A list of objects defining the input MBTiles files.
    *   `path` (Required): The path to the MBTiles file.
    *   `encoding` (Optional, Default: `"mapbox"`): The encoding used for the MBTiles file (`"mapbox"` or `"terrarium"`).
    *   `height_adjustment` (Optional, Default: `0.0`): A floating-point value (in meters) to adjust the elevation of that particular input. Positive values raise the elevation, and negative values lower the elevation.
*   `output_path` (Optional, Default: `"output.mbtiles"`): The output path for the merged MBTiles file.
*   `output_encoding` (Optional, Default: `"mapbox"`): The output encoding to use (`"mapbox"` or `"terrarium"`).
*  `output_format` (Optional, Default: `"png"`): The output image format (`"png"` or `"webp"`).
*  `resampling` (Optional, Default: `"bilinear"`): The method to use for resampling (`"nearest"`, `"bilinear"`, `"cubic"`, `"cubic_spline"`, `"lanczos"`, `"average"`, `"mode"`, or `"gauss"`).

The merge logic works by merging the input sources in order, applying the height adjustment as it merges. The last input source will be the base layer for tiles.

## Merge Example
```
rio rgbify merge --config config.json -j 24
```
