"""
Fake Song Detector – Flask web application.

Routes
------
GET  /               Landing page with upload form and database listing
POST /check          Upload a song to check for plagiarism
POST /add            Upload a song to add to the reference database
POST /remove/<id>    Remove a song from the reference database by id
GET  /api/songs      JSON list of all songs in the DB
"""

import logging
import os
import tempfile
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
    flash,
)
from werkzeug.utils import secure_filename

from song_detector import SongDetector

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production")

DB_PATH = os.environ.get("DB_PATH", "songs.db")
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.90"))

ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "aiff", "m4a"}

logger = logging.getLogger(__name__)

detector = SongDetector(db_path=DB_PATH, threshold=SIMILARITY_THRESHOLD)


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# HTML routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    songs = detector.list_songs()
    return render_template("index.html", songs=songs, threshold=SIMILARITY_THRESHOLD)


@app.route("/check", methods=["POST"])
def check_song():
    file = request.files.get("audio_file")
    if not file or file.filename == "":
        flash("Please select an audio file to check.", "warning")
        return redirect(url_for("index"))

    if not _allowed_file(file.filename):
        flash(f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}", "danger")
        return redirect(url_for("index"))

    query_title = request.form.get("title", "Unknown Song").strip() or "Unknown Song"
    query_artist = request.form.get("artist", "").strip()

    suffix = Path(secure_filename(file.filename)).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        result = detector.check_file(
            tmp_path,
            query_title=query_title,
            query_artist=query_artist,
        )
    except Exception as exc:
        flash(f"Error processing audio: {exc}", "danger")
        return redirect(url_for("index"))
    finally:
        os.unlink(tmp_path)

    songs = detector.list_songs()
    return render_template(
        "index.html",
        songs=songs,
        threshold=SIMILARITY_THRESHOLD,
        result=result.as_dict(),
    )


@app.route("/add", methods=["POST"])
def add_song():
    file = request.files.get("audio_file")
    if not file or file.filename == "":
        flash("Please select an audio file to add.", "warning")
        return redirect(url_for("index"))

    if not _allowed_file(file.filename):
        flash(f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}", "danger")
        return redirect(url_for("index"))

    title = request.form.get("title", "").strip()
    artist = request.form.get("artist", "").strip()
    if not title:
        flash("Please provide a song title.", "warning")
        return redirect(url_for("index"))

    suffix = Path(secure_filename(file.filename)).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        song_id = detector.add_song_file(tmp_path, title=title, artist=artist)
        flash(f'Song "{title}" added to the database (id={song_id}).', "success")
    except Exception as exc:
        flash(f"Error adding song: {exc}", "danger")
    finally:
        os.unlink(tmp_path)

    return redirect(url_for("index"))


@app.route("/remove/<int:song_id>", methods=["POST"])
def remove_song(song_id: int):
    removed = detector.remove_song(song_id)
    if removed:
        flash(f"Song #{song_id} removed from the database.", "success")
    else:
        flash(f"Song #{song_id} not found.", "warning")
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# JSON API routes
# ---------------------------------------------------------------------------


@app.get("/api/songs")
def api_list_songs():
    return jsonify(detector.list_songs())


@app.post("/api/check")
def api_check_song():
    """JSON endpoint for programmatic access.

    Expects a multipart/form-data POST with:
      - audio_file: the audio file
      - title (optional): song title
      - artist (optional): artist name
    """
    file = request.files.get("audio_file")
    if not file or file.filename == "":
        return jsonify({"error": "No audio_file provided"}), 400

    if not _allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    query_title = request.form.get("title", "Unknown Song").strip() or "Unknown Song"
    query_artist = request.form.get("artist", "").strip()

    suffix = Path(secure_filename(file.filename)).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        result = detector.check_file(tmp_path, query_title=query_title, query_artist=query_artist)
        return jsonify(result.as_dict())
    except Exception as exc:
        logger.exception("Error processing audio in /api/check")
        return jsonify({"error": "Failed to process audio file."}), 500
    finally:
        os.unlink(tmp_path)


@app.post("/api/add")
def api_add_song():
    """JSON endpoint to add a song to the reference database."""
    file = request.files.get("audio_file")
    if not file or file.filename == "":
        return jsonify({"error": "No audio_file provided"}), 400

    if not _allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    title = request.form.get("title", "").strip()
    artist = request.form.get("artist", "").strip()
    if not title:
        return jsonify({"error": "A 'title' field is required"}), 400

    suffix = Path(secure_filename(file.filename)).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        song_id = detector.add_song_file(tmp_path, title=title, artist=artist)
        return jsonify({"song_id": song_id, "title": title, "artist": artist}), 201
    except Exception as exc:
        logger.exception("Error adding song in /api/add")
        return jsonify({"error": "Failed to process audio file."}), 500
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, port=port)
