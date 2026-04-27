"""
Audio embedding extraction for the fake-song-detector.

Uses a combination of librosa audio features to produce a compact, fixed-length
embedding vector that captures the timbral, harmonic, and rhythmic content of a song.

Feature groups
--------------
* MFCCs (40 coefficients × mean + std = 80 values) – timbral texture
* Chroma STFT (12 bins × mean + std = 24 values) – harmonic/pitch content
* Spectral Contrast (7 bands × mean + std = 14 values) – spectral peaks vs valleys
* Mel Spectrogram statistics (128 bands × mean + std = 256 values) – overall energy distribution
* Zero Crossing Rate (mean + std = 2 values) – noisiness / percussiveness
* RMS Energy (mean + std = 2 values) – loudness dynamics

Total embedding dimension: 378 (L2-normalised to unit length)
"""

import librosa
import numpy as np
from typing import Optional

# Number of samples to analyse per song (30 s @ 22 050 Hz = 661 500 samples).
# Trimming to a fixed length makes embeddings comparable regardless of song duration.
_SAMPLE_RATE = 22050
_MAX_DURATION_SECONDS = 30
_MAX_SAMPLES = _SAMPLE_RATE * _MAX_DURATION_SECONDS


def extract_embedding(
    audio_path: str,
    sr: int = _SAMPLE_RATE,
    max_samples: int = _MAX_SAMPLES,
    offset: float = 0.0,
) -> np.ndarray:
    """Load an audio file and return a normalised embedding vector.

    Parameters
    ----------
    audio_path:
        Path to the audio file (WAV, MP3, FLAC, OGG, …).
    sr:
        Target sample rate.  All audio is resampled to this rate.
    max_samples:
        Maximum number of samples to analyse (the middle ``max_samples`` samples
        are taken so that silence at the start/end of a track does not dominate).
    offset:
        Start reading the file at this many seconds from the beginning.

    Returns
    -------
    numpy.ndarray
        1-D float32 array of length 378, L2-normalised to unit length.
    """
    y, _ = librosa.load(audio_path, sr=sr, offset=offset, mono=True)

    # Trim leading/trailing silence so we don't compare silence against music.
    y, _ = librosa.effects.trim(y, top_db=20)

    # If the track is longer than max_samples, take the centre segment.
    if len(y) > max_samples:
        start = (len(y) - max_samples) // 2
        y = y[start : start + max_samples]

    if len(y) == 0:
        raise ValueError(f"Audio file '{audio_path}' contains no usable audio.")

    return _compute_features(y, sr)


def _compute_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Compute the feature vector from a raw audio array."""
    if len(y) == 0 or np.max(np.abs(y)) == 0:
        raise ValueError("Audio array contains no usable audio (empty or silent).")

    parts: list[np.ndarray] = []

    # --- MFCCs ---------------------------------------------------------------
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    parts.append(np.mean(mfcc, axis=1))
    parts.append(np.std(mfcc, axis=1))

    # --- Chroma STFT ---------------------------------------------------------
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    parts.append(np.mean(chroma, axis=1))
    parts.append(np.std(chroma, axis=1))

    # --- Spectral Contrast ---------------------------------------------------
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    parts.append(np.mean(spec_contrast, axis=1))
    parts.append(np.std(spec_contrast, axis=1))

    # --- Mel Spectrogram -----------------------------------------------------
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    parts.append(np.mean(mel_db, axis=1))
    parts.append(np.std(mel_db, axis=1))

    # --- Zero Crossing Rate --------------------------------------------------
    zcr = librosa.feature.zero_crossing_rate(y)
    parts.append(np.array([np.mean(zcr), np.std(zcr)]))

    # --- RMS Energy ----------------------------------------------------------
    rms = librosa.feature.rms(y=y)
    parts.append(np.array([np.mean(rms), np.std(rms)]))

    embedding = np.concatenate(parts).astype(np.float32)

    # L2 normalise to unit length so cosine similarity == dot product.
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def embedding_from_array(y: np.ndarray, sr: int = _SAMPLE_RATE) -> np.ndarray:
    """Compute an embedding directly from a pre-loaded audio array.

    Useful for testing without writing files to disk.
    """
    return _compute_features(y, sr)
