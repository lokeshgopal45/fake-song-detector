"""Fake Song Detector – detect whether a new song is copied from existing music."""

from .detector import SongDetector
from .database import EmbeddingDatabase
from .embeddings import extract_embedding

__all__ = ["SongDetector", "EmbeddingDatabase", "extract_embedding"]
