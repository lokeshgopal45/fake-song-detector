"""
Unit tests for the fake-song-detector package.

Tests are designed to run without any real audio files by synthesising
simple sine-wave signals with numpy.  This keeps the test suite fast and
avoids any external dependencies (no downloading models or sample files).
"""

import tempfile
import os

import numpy as np
import pytest
import soundfile as sf

from song_detector.embeddings import extract_embedding, embedding_from_array
from song_detector.database import EmbeddingDatabase
from song_detector.detector import SongDetector, DetectionResult, Match

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 22050  # sample rate used throughout tests


def _sine_wave(freq: float = 440.0, duration: float = 3.0, sr: int = SR) -> np.ndarray:
    """Generate a pure sine-wave audio array."""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noise(duration: float = 3.0, sr: int = SR, seed: int = 42) -> np.ndarray:
    """Generate white-noise audio array."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(duration * sr)).astype(np.float32) * 0.3


def _write_wav(y: np.ndarray, sr: int = SR) -> str:
    """Write an audio array to a temporary WAV file and return its path."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    sf.write(path, y, sr)
    return path


# ---------------------------------------------------------------------------
# embeddings.py
# ---------------------------------------------------------------------------


class TestEmbeddingFromArray:
    def test_returns_ndarray(self):
        y = _sine_wave()
        emb = embedding_from_array(y, SR)
        assert isinstance(emb, np.ndarray)

    def test_unit_length(self):
        """Embedding should be L2-normalised to (approximately) unit length."""
        y = _sine_wave()
        emb = embedding_from_array(y, SR)
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5

    def test_float32_dtype(self):
        y = _sine_wave()
        emb = embedding_from_array(y, SR)
        assert emb.dtype == np.float32

    def test_fixed_length(self):
        """All embeddings must have the same dimensionality."""
        emb1 = embedding_from_array(_sine_wave(440), SR)
        emb2 = embedding_from_array(_sine_wave(880), SR)
        emb3 = embedding_from_array(_noise(), SR)
        assert emb1.shape == emb2.shape == emb3.shape

    def test_different_sounds_differ(self):
        """Two very different sounds should have low cosine similarity."""
        emb_sine = embedding_from_array(_sine_wave(440), SR)
        emb_noise = embedding_from_array(_noise(), SR)
        similarity = float(np.dot(emb_sine, emb_noise))
        # We don't require a specific threshold here – just that they are not identical
        assert similarity < 0.999

    def test_same_sound_identical(self):
        """The same audio array should produce identical embeddings."""
        y = _sine_wave(440)
        emb1 = embedding_from_array(y, SR)
        emb2 = embedding_from_array(y, SR)
        np.testing.assert_array_equal(emb1, emb2)

    def test_similar_sounds_high_similarity(self):
        """Two very similar sounds (same frequency, tiny amplitude difference)
        should have a very high cosine similarity."""
        y1 = _sine_wave(440)
        y2 = y1 * 0.99  # almost identical
        emb1 = embedding_from_array(y1, SR)
        emb2 = embedding_from_array(y2, SR)
        similarity = float(np.dot(emb1, emb2))
        assert similarity > 0.99

    def test_empty_audio_raises(self):
        with pytest.raises(ValueError, match="no usable audio"):
            embedding_from_array(np.zeros(100, dtype=np.float32), SR)


class TestExtractEmbeddingFromFile:
    def test_wav_file(self):
        y = _sine_wave()
        path = _write_wav(y)
        try:
            emb = extract_embedding(path)
            assert isinstance(emb, np.ndarray)
            assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        with pytest.raises(Exception):
            extract_embedding("/nonexistent/path/song.wav")


# ---------------------------------------------------------------------------
# database.py
# ---------------------------------------------------------------------------


class TestEmbeddingDatabase:
    def _make_db(self) -> EmbeddingDatabase:
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(path)  # let EmbeddingDatabase create it
        return EmbeddingDatabase(path)

    def test_initially_empty(self):
        db = self._make_db()
        assert db.song_count() == 0
        assert db.get_all_songs() == []

    def test_add_and_count(self):
        db = self._make_db()
        emb = embedding_from_array(_sine_wave(), SR)
        db.add_song(emb, title="Test Song", artist="Test Artist")
        assert db.song_count() == 1

    def test_add_returns_id(self):
        db = self._make_db()
        emb = embedding_from_array(_sine_wave(), SR)
        song_id = db.add_song(emb, title="Song A")
        assert isinstance(song_id, int)
        assert song_id >= 1

    def test_sequential_ids(self):
        db = self._make_db()
        emb = embedding_from_array(_sine_wave(), SR)
        id1 = db.add_song(emb, title="Song 1")
        id2 = db.add_song(emb, title="Song 2")
        assert id2 > id1

    def test_get_all_songs_metadata(self):
        db = self._make_db()
        emb = embedding_from_array(_sine_wave(), SR)
        db.add_song(emb, title="My Song", artist="My Artist")
        songs = db.get_all_songs()
        assert len(songs) == 1
        assert songs[0]["title"] == "My Song"
        assert songs[0]["artist"] == "My Artist"

    def test_get_all_embeddings_roundtrip(self):
        """Embedding stored and retrieved must be numerically identical."""
        db = self._make_db()
        emb = embedding_from_array(_sine_wave(), SR)
        db.add_song(emb, title="Song")
        retrieved = db.get_all_embeddings()
        assert len(retrieved) == 1
        np.testing.assert_array_almost_equal(emb, retrieved[0]["embedding"], decimal=6)

    def test_remove_song(self):
        db = self._make_db()
        emb = embedding_from_array(_sine_wave(), SR)
        song_id = db.add_song(emb, title="Temp")
        assert db.remove_song(song_id) is True
        assert db.song_count() == 0

    def test_remove_nonexistent_returns_false(self):
        db = self._make_db()
        assert db.remove_song(9999) is False

    def test_multiple_songs(self):
        db = self._make_db()
        for i in range(5):
            emb = embedding_from_array(_sine_wave(220 + i * 110), SR)
            db.add_song(emb, title=f"Song {i}")
        assert db.song_count() == 5
        assert len(db.get_all_embeddings()) == 5


# ---------------------------------------------------------------------------
# detector.py
# ---------------------------------------------------------------------------


class TestSongDetector:
    def _make_detector(self, threshold: float = 0.90) -> SongDetector:
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(path)
        return SongDetector(db_path=path, threshold=threshold)

    # -- add_song_array -------------------------------------------------------

    def test_add_song_returns_id(self):
        det = self._make_detector()
        y = _sine_wave()
        song_id = det.add_song_array(y, SR, title="Song A")
        assert isinstance(song_id, int)

    def test_list_songs(self):
        det = self._make_detector()
        det.add_song_array(_sine_wave(440), SR, title="A")
        det.add_song_array(_sine_wave(880), SR, title="B")
        songs = det.list_songs()
        assert len(songs) == 2
        titles = {s["title"] for s in songs}
        assert titles == {"A", "B"}

    # -- check_array ----------------------------------------------------------

    def test_empty_db_returns_no_matches(self):
        det = self._make_detector()
        y = _sine_wave()
        result = det.check_array(y, SR, query_title="New Song")
        assert isinstance(result, DetectionResult)
        assert result.matches == []
        assert result.is_copied is False

    def test_exact_copy_detected(self):
        """Uploading the exact same audio that was added should be flagged."""
        det = self._make_detector(threshold=0.90)
        y = _sine_wave(440)
        det.add_song_array(y, SR, title="Original")
        result = det.check_array(y, SR, query_title="Copy")
        assert result.is_copied is True
        assert result.top_match is not None
        assert result.top_match.similarity > 0.99

    def test_very_similar_copy_detected(self):
        """A near-identical song (tiny amplitude change) should still be flagged."""
        det = self._make_detector(threshold=0.90)
        y_orig = _sine_wave(440)
        y_copy = y_orig * 0.98
        det.add_song_array(y_orig, SR, title="Original")
        result = det.check_array(y_copy, SR, query_title="Copy")
        assert result.is_copied is True

    def test_different_song_not_flagged(self):
        """A clearly different song should not be flagged as a copy."""
        det = self._make_detector(threshold=0.90)
        det.add_song_array(_sine_wave(440), SR, title="Original A")
        result = det.check_array(_noise(), SR, query_title="Noise Song")
        # Noise is very different from a pure sine – should not exceed the threshold.
        assert result.is_copied is False

    def test_result_sorted_by_similarity(self):
        """Matches should be sorted in descending order of similarity."""
        det = self._make_detector()
        y_base = _sine_wave(440)
        det.add_song_array(y_base, SR, title="Original")
        det.add_song_array(_sine_wave(1000), SR, title="Different")
        result = det.check_array(y_base, SR, query_title="Query")
        assert len(result.matches) == 2
        assert result.matches[0].similarity >= result.matches[1].similarity

    def test_top_n_respected(self):
        """Only top_n results should be returned."""
        det = self._make_detector()
        for i in range(10):
            det.add_song_array(_sine_wave(200 + i * 50), SR, title=f"Song {i}")
        result = det.check_array(_sine_wave(440), SR, top_n=3)
        assert len(result.matches) <= 3

    def test_result_as_dict(self):
        """DetectionResult.as_dict() should return a JSON-serialisable dict."""
        det = self._make_detector()
        y = _sine_wave()
        det.add_song_array(y, SR, title="Song")
        result = det.check_array(y, SR, query_title="Test")
        d = result.as_dict()
        assert "is_copied" in d
        assert "top_match" in d
        assert "all_matches" in d
        assert isinstance(d["all_matches"], list)

    def test_remove_song(self):
        det = self._make_detector()
        song_id = det.add_song_array(_sine_wave(), SR, title="To Remove")
        assert det.remove_song(song_id) is True
        assert det.list_songs() == []

    def test_remove_nonexistent(self):
        det = self._make_detector()
        assert det.remove_song(9999) is False

    # -- check_file -----------------------------------------------------------

    def test_check_file_exact_copy(self):
        det = self._make_detector(threshold=0.90)
        y = _sine_wave(440)
        path = _write_wav(y)
        try:
            det.add_song_array(y, SR, title="Original")
            result = det.check_file(path, query_title="Copy")
            assert result.is_copied is True
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Match dataclass
# ---------------------------------------------------------------------------


class TestMatch:
    def test_is_copy_true(self):
        m = Match(song_id=1, title="X", artist="Y", similarity=0.95)
        assert m.is_copy is True

    def test_is_copy_false(self):
        m = Match(song_id=1, title="X", artist="Y", similarity=0.80)
        assert m.is_copy is False

    def test_as_dict_keys(self):
        m = Match(song_id=1, title="X", artist="Y", similarity=0.95)
        d = m.as_dict()
        assert set(d.keys()) == {"song_id", "title", "artist", "similarity", "is_copy"}
