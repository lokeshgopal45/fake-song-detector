# 🎵 Fake Song Detector

> Upload any song and find out whether it sounds copied from music in your reference library.

![Fake Song Detector UI](https://github.com/user-attachments/assets/cc5a5d0c-0415-4e75-82c3-16315994b946)

---

## How It Works

Each audio file is converted into a compact **embedding vector** — a numerical fingerprint that captures:

| Feature | What it captures |
|---------|-----------------|
| MFCCs (40 coefficients) | Timbral texture |
| Chroma STFT (12 bins) | Harmonic / pitch content |
| Spectral Contrast (7 bands) | Spectral peaks vs valleys |
| Mel Spectrogram (128 bands) | Overall energy distribution |
| Zero Crossing Rate | Noisiness / percussiveness |
| RMS Energy | Loudness dynamics |

All features are concatenated into a **378-dimensional, L2-normalised unit vector**.  
Similarity between two songs is the **cosine similarity** (dot product) of their embedding vectors.  
A score ≥ **90 %** flags the song as a potential copy (configurable).

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the web app

```bash
python app.py
```

Open <http://localhost:5000> in your browser.

**Web UI features:**
- 🔍 **Check a Song** — upload an audio file and see similarity scores against all reference songs.
- ➕ **Add to Reference Database** — register a known song so future uploads can be compared against it.
- 📚 **Reference Database** — browse and remove stored songs.

---

## Command-Line Interface

```bash
# Add a reference song
python cli.py add path/to/song.wav --title "Bohemian Rhapsody" --artist "Queen"

# Check whether a new song is copied (top-5 matches)
python cli.py check path/to/my_song.mp3 --title "My New Song"

# List all songs in the database
python cli.py list

# Remove a song by database id
python cli.py remove 3

# Output result as JSON
python cli.py check song.wav --json
```

Global options:
| Option | Default | Description |
|--------|---------|-------------|
| `--db PATH` | `songs.db` | SQLite database file |
| `--threshold FLOAT` | `0.90` | Cosine similarity threshold for flagging copies |

---

## JSON / REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/songs` | List all reference songs |
| `POST` | `/api/add` | Add a song (`audio_file`, `title`, `artist` fields) |
| `POST` | `/api/check` | Check a song (`audio_file`, `title`, `artist` fields) |

Example with curl:

```bash
# Add a reference song
curl -X POST http://localhost:5000/api/add \
  -F "audio_file=@bohemian.wav" \
  -F "title=Bohemian Rhapsody" \
  -F "artist=Queen"

# Check a new song
curl -X POST http://localhost:5000/api/check \
  -F "audio_file=@new_song.mp3" \
  -F "title=My Song"
```

---

## Supported Audio Formats

WAV · MP3 · FLAC · OGG · AIFF · M4A

---

## Project Structure

```
fake-song-detector/
├── app.py                    # Flask web application
├── cli.py                    # Command-line interface
├── requirements.txt
├── song_detector/
│   ├── __init__.py
│   ├── embeddings.py         # Audio feature extraction
│   ├── database.py           # SQLite embedding store
│   └── detector.py           # Similarity comparison logic
├── templates/
│   └── index.html            # Web UI
└── tests/
    └── test_detector.py      # Unit tests (34 tests)
```

---

## Running Tests

```bash
pip install pytest soundfile
python -m pytest tests/ -v
```

---

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DB_PATH` | `songs.db` | Path to the SQLite database |
| `SIMILARITY_THRESHOLD` | `0.90` | Copy detection threshold (0–1) |
| `SECRET_KEY` | `change-me-in-production` | Flask session secret |
| `PORT` | `5000` | Web server port |
