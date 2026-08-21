"""
Download a small set of real terrain tiles from tiles.wifidb.net into
fixture MBTiles files for offline integration tests.

Run once (or whenever you want to refresh the fixtures):

    python test/download_fixtures.py

Produces:
    test/fixtures/gebco_sample.mbtiles   — GEBCO 2024 bathymetry, z0-z2
    test/fixtures/jaxa_sample.mbtiles    — JAXA AW3D30 2024 land, z0-z2
"""

import sys
import gzip
import time
import urllib.request
from pathlib import Path

# Ensure the package is importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from rio_rgbify.database import MBTilesDatabase

GEBCO_URL = "https://tiles.wifidb.net/data/ocean-rgb/{z}/{x}/{y}.webp"
JAXA_URL  = "https://tiles.wifidb.net/data/jaxa_terrainrgb_webp/{z}/{x}/{y}.webp"

# z=0: 1 tile; z=1: 4 tiles; z=2: 16 tiles — 21 tiles per source, all global coverage
MAX_ZOOM = 2

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def tile_coords(max_zoom: int):
    """Yield (z, x, y) for all XYZ tiles from z=0 to max_zoom (inclusive)."""
    for z in range(max_zoom + 1):
        n = 2 ** z
        for x in range(n):
            for y in range(n):
                yield z, x, y


def fetch(url: str, retries: int = 3, delay: float = 1.0) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rio-rgbify-test/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            # Decompress if the server returned gzip-encoded content
            if data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
            return data
        except Exception as exc:
            if attempt < retries - 1:
                print(f"  retry {attempt+1}/{retries-1}: {exc}")
                time.sleep(delay)
            else:
                raise


def download_source(url_template: str, out_path: Path, encoding: str, name: str):
    tiles = list(tile_coords(MAX_ZOOM))
    print(f"Downloading {len(tiles)} tiles -> {out_path.name}")
    out_path.unlink(missing_ok=True)
    with MBTilesDatabase(str(out_path)) as db:
        db.add_metadata({
            "name": name,
            "format": "webp",
            "encoding": encoding,
            "minzoom": "0",
            "maxzoom": str(MAX_ZOOM),
        })
        for z, x, y in tiles:
            url = url_template.format(z=z, x=x, y=y)
            try:
                data = fetch(url)
                db.insert_tile_with_retry([x, y, z], data)
                print(f"  OK z={z} x={x} y={y} ({len(data)} bytes)")
            except Exception as exc:
                print(f"  FAIL z={z} x={x} y={y}: {exc}")
    print(f"  Done: {out_path}")


if __name__ == "__main__":
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    download_source(GEBCO_URL, FIXTURES_DIR / "gebco_sample.mbtiles", "mapbox", "GEBCO 2024 TerrainRGB sample")
    download_source(JAXA_URL,  FIXTURES_DIR / "jaxa_sample.mbtiles",  "mapbox", "JAXA AW3D30 2024 TerrainRGB sample")
    print("\nFixtures written. Commit test/fixtures/gebco_sample.mbtiles and test/fixtures/jaxa_sample.mbtiles")
