"""
Core detection logic for the fake-song-detector.

The ``SongDetector`` class ties together embedding extraction and database lookup.
It answers the question: "Is this new song too similar to any song I already know?"

Similarity metric
-----------------
Because every embedding is L2-normalised to unit length, the cosine similarity
between two vectors equals their dot product:

    similarity = dot(a, b)   ∈ [-1, 1]

A value of 1.0 means the two songs are essentially identical; 0.0 means
orthogonal (no similarity).  The default ``threshold`` of **0.90** flags a song
as a potential copy if it shares ≥ 90 % of its "audio fingerprint" with an
existing song.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .database import EmbeddingDatabase
from .embeddings import extract_embedding, embedding_from_array

# Default cosine similarity threshold above which a match is reported.
DEFAULT_THRESHOLD = 0.90


@dataclass
class Match:
    """A single match returned by ``SongDetector.check``."""

    song_id: int
    title: str
    artist: str
    similarity: float  # cosine similarity in [0, 1]
    threshold: float = DEFAULT_THRESHOLD

    @property
    def is_copy(self) -> bool:
        """True when similarity is at or above the threshold used for this match."""
        return self.similarity >= self.threshold

    def as_dict(self) -> dict:
        return {
            "song_id": self.song_id,
            "title": self.title,
            "artist": self.artist,
            "similarity": round(float(self.similarity), 4),
            "is_copy": self.is_copy,
        }


@dataclass
class DetectionResult:
    """Full result of checking one song against the database."""

    query_title: str
    query_artist: str
    threshold: float
    matches: list[Match] = field(default_factory=list)

    @property
    def is_copied(self) -> bool:
        """True if any match is at or above the threshold."""
        return any(m.similarity >= self.threshold for m in self.matches)

    @property
    def top_match(self) -> Optional[Match]:
        if not self.matches:
            return None
        return max(self.matches, key=lambda m: m.similarity)

    def as_dict(self) -> dict:
        return {
            "query_title": self.query_title,
            "query_artist": self.query_artist,
            "threshold": self.threshold,
            "is_copied": self.is_copied,
            "top_match": self.top_match.as_dict() if self.top_match else None,
            "all_matches": [m.as_dict() for m in self.matches],
        }


class SongDetector:
    """High-level API for adding songs and checking for plagiarism."""

    def __init__(
        self,
        db_path: str = "songs.db",
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.db = EmbeddingDatabase(db_path)
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Adding songs to the reference database
    # ------------------------------------------------------------------

    def add_song_file(
        self,
        audio_path: str,
        title: str,
        artist: str = "",
    ) -> int:
        """Extract an embedding from *audio_path* and store it in the DB.

        Returns the new database row id.
        """
        embedding = extract_embedding(audio_path)
        return self.db.add_song(embedding, title=title, artist=artist, file_path=audio_path)

    def add_song_array(
        self,
        y: np.ndarray,
        sr: int,
        title: str,
        artist: str = "",
    ) -> int:
        """Add a song from an in-memory audio array (useful for testing)."""
        embedding = embedding_from_array(y, sr)
        return self.db.add_song(embedding, title=title, artist=artist)

    # ------------------------------------------------------------------
    # Checking a new song against the database
    # ------------------------------------------------------------------

    def check_file(
        self,
        audio_path: str,
        query_title: str = "Unknown",
        query_artist: str = "",
        top_n: int = 5,
    ) -> DetectionResult:
        """Check an audio file against all songs in the database.

        Returns a :class:`DetectionResult` sorted by descending similarity.
        """
        embedding = extract_embedding(audio_path)
        return self._compare(embedding, query_title, query_artist, top_n)

    def check_array(
        self,
        y: np.ndarray,
        sr: int,
        query_title: str = "Unknown",
        query_artist: str = "",
        top_n: int = 5,
    ) -> DetectionResult:
        """Check an in-memory audio array against the database."""
        embedding = embedding_from_array(y, sr)
        return self._compare(embedding, query_title, query_artist, top_n)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compare(
        self,
        query_embedding: np.ndarray,
        query_title: str,
        query_artist: str,
        top_n: int,
    ) -> DetectionResult:
        """Compare *query_embedding* against every embedding in the DB."""
        all_songs = self.db.get_all_embeddings()

        matches: list[Match] = []
        for song in all_songs:
            sim = float(np.dot(query_embedding, song["embedding"]))
            # Clamp to [0, 1] – negative cosine similarity means the songs are
            # unrelated; we treat those as 0 % similar.
            sim = max(0.0, sim)
            matches.append(
                Match(
                    song_id=song["id"],
                    title=song["title"],
                    artist=song["artist"],
                    similarity=sim,
                    threshold=self.threshold,
                )
            )

        # Sort by descending similarity and keep top N.
        matches.sort(key=lambda m: m.similarity, reverse=True)
        matches = matches[:top_n]

        return DetectionResult(
            query_title=query_title,
            query_artist=query_artist,
            threshold=self.threshold,
            matches=matches,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def list_songs(self) -> list[dict]:
        """Return metadata for all songs in the reference database."""
        return self.db.get_all_songs()

    def remove_song(self, song_id: int) -> bool:
        """Remove a song from the database by id."""
        return self.db.remove_song(song_id)
