"""
SQLite-backed database for storing song embeddings.

Each row holds:
  - id        INTEGER PRIMARY KEY
  - title     TEXT     human-readable song title
  - artist    TEXT     artist/band name (optional)
  - file_path TEXT     original file path (informational only)
  - embedding BLOB     raw bytes of a float32 numpy array
  - added_at  TEXT     ISO-8601 timestamp
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS songs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    title     TEXT    NOT NULL,
    artist    TEXT    NOT NULL DEFAULT '',
    file_path TEXT    NOT NULL DEFAULT '',
    embedding BLOB    NOT NULL,
    added_at  TEXT    NOT NULL
);
"""


class EmbeddingDatabase:
    """Persistent store for song embeddings."""

    def __init__(self, db_path: str = "songs.db") -> None:
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_song(
        self,
        embedding: np.ndarray,
        title: str,
        artist: str = "",
        file_path: str = "",
    ) -> int:
        """Insert a song embedding and return the new row id."""
        blob = embedding.astype(np.float32).tobytes()
        added_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO songs (title, artist, file_path, embedding, added_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (title, artist, file_path, blob, added_at),
            )
            return cursor.lastrowid

    def remove_song(self, song_id: int) -> bool:
        """Delete a song by id.  Returns True if a row was deleted."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM songs WHERE id = ?", (song_id,))
            return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_all_songs(self) -> list[dict]:
        """Return metadata (without the embedding blob) for all songs."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, artist, file_path, added_at FROM songs"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_embeddings(self) -> list[dict]:
        """Return id, title, artist, and the decoded numpy embedding for every song."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, artist, file_path, embedding FROM songs"
            ).fetchall()
        result = []
        for r in rows:
            vec = np.frombuffer(r["embedding"], dtype=np.float32).copy()
            result.append(
                {
                    "id": r["id"],
                    "title": r["title"],
                    "artist": r["artist"],
                    "file_path": r["file_path"],
                    "embedding": vec,
                }
            )
        return result

    def song_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
