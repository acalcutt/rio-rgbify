import sqlite3
import os
import math
from typing import List, Optional
from contextlib import contextmanager
from pathlib import Path

class MBTilesDatabase:
    def __init__(self, outpath: str):
        self.outpath = outpath
        self.conn = None
        self.cur = None

    def __enter__(self):
        # if os.path.exists(self.outpath):
        #     os.unlink(self.outpath)

        self.conn = sqlite3.connect(self.outpath)
        # Wall mode : Speedup by 10 the speed of writing in the database
        # self.conn.execute('pragma journal_mode=wal')
        self.cur = self.conn.cursor()
        
        # create the tiles table
        self.cur.execute(
            "CREATE TABLE IF NOT EXISTS tiles_shallow ("
            "TILES_COL_Z integer, "
            "TILES_COL_X integer, "
            "TILES_COL_Y integer, "
            "TILES_COL_DATA_ID text "
            ", primary key(TILES_COL_Z,TILES_COL_X,TILES_COL_Y) "
            ") without rowid;")
            
        self.cur.execute(
            "CREATE TABLE IF NOT EXISTS tiles_data ("
            "tile_data_id text primary key, "
            "tile_data blob "
            ");")
            
        self.cur.execute(
            "CREATE VIEW IF NOT EXISTS tiles AS "
            "select "
            "tiles_shallow.TILES_COL_Z as zoom_level, "
            "tiles_shallow.TILES_COL_X as tile_column, "
            "tiles_shallow.TILES_COL_Y as tile_row, "
            "tiles_data.tile_data as tile_data "
            "from tiles_shallow "
            "join tiles_data on tiles_shallow.TILES_COL_DATA_ID = tiles_data.tile_data_id;")

        self.cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS tiles_shallow_index on tiles_shallow (TILES_COL_Z, TILES_COL_X, TILES_COL_Y);")

        # create empty metadata
        self.cur.execute("CREATE TABLE IF NOT EXISTS metadata (name text, value text);")

        self.conn.commit()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.commit()
        # disable Wall mode
        # self.conn.execute('pragma journal_mode=DELETE')
        self.conn.close()

    def add_metadata(self, metadata: dict):
        """Adds metadata to the mbtiles db"""
        for key, value in metadata.items():
            self.cur.execute(
                "INSERT INTO metadata " "(name, value) " "VALUES (?, ?);",
                (key, value),
            )
        self.conn.commit()

    def fnv1a(self, buf: bytes) -> int:
        h = 14695981039346656037

        for b in buf:
            h ^= b
            h *= 1099511628211
            h &= 0xFFFFFFFFFFFFFFFF  # 64-bit mask

        return h

    def insert_tile(self, tile: List[int], contents: bytes):
        """Add tile to database with deduplication logic"""
        x, y, z = tile
        # mbtiles use inverse y indexing
        #tiley = int(math.pow(2, z)) - y - 1
        
        #create tile_id based on tile contents
        tileDataId = str(self.fnv1a(contents))
        
        # insert tile object
        self.cur.execute(
            "INSERT OR IGNORE INTO tiles_data "
            "(tile_data_id, tile_data) "
            "VALUES (?, ?);",
            (tileDataId, contents),
        )

        self.cur.execute(
            "INSERT INTO tiles_shallow "
            "(TILES_COL_Z, TILES_COL_X, TILES_COL_Y, TILES_COL_DATA_ID) "
            "VALUES (?, ?, ?, ?);",
            (z, x, tiley, tileDataId),
        )

    @contextmanager
    def db_connection(self):
        """Context manager for database connections"""
        print(f"db_connection called with outpath: {self.outpath}")
        conn = sqlite3.connect(self.outpath)
        try:
            yield conn
        finally:
            conn.close()

    def get_tile_data(self, zoom: int, x: int, y: int) -> Optional[bytes]:
        """Retrieves tile data from the database based on zoom, x, y"""
        with self.db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                (zoom, x, y)
            )
            result = cursor.fetchone()
            if result:
                return result[0]
            return None
        
    def get_max_zoom_level(self) -> int:
        """Get the maximum zoom level from the tiles table."""
        with self.db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(zoom_level) FROM tiles")
            result = cursor.fetchone()
            if result and result[0] is not None:
                return result[0]
            return 0
    
    def get_distinct_tiles(self, zoom: int) -> List[tuple[int, int]]:
        """Get the distinct tile_column and tile_row for a given zoom level."""
        with self.db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT DISTINCT tile_column, tile_row FROM tiles WHERE zoom_level = ?',
                (zoom,)
            )
            rows = cursor.fetchall()
            return rows
