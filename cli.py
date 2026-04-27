#!/usr/bin/env python3
"""
Fake Song Detector – command-line interface.

Usage examples
--------------
# Add a reference song
python cli.py add song.wav --title "Bohemian Rhapsody" --artist "Queen"

# Check whether a new song is copied
python cli.py check my_song.mp3 --title "My New Song"

# List all songs in the database
python cli.py list

# Remove a song by id
python cli.py remove 3
"""

import argparse
import json
import sys

from song_detector import SongDetector

DEFAULT_DB = "songs.db"
DEFAULT_THRESHOLD = 0.90


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fake-song-detector",
        description="Detect whether a song has been copied from existing music.",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        metavar="PATH",
        help=f"Path to the SQLite database (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        metavar="FLOAT",
        help=f"Cosine similarity threshold for flagging copies (default: {DEFAULT_THRESHOLD})",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # -- add ------------------------------------------------------------------
    add_p = sub.add_parser("add", help="Add a reference song to the database.")
    add_p.add_argument("audio_file", help="Path to the audio file.")
    add_p.add_argument("--title", required=True, help="Song title.")
    add_p.add_argument("--artist", default="", help="Artist name (optional).")

    # -- check ----------------------------------------------------------------
    check_p = sub.add_parser("check", help="Check a song against the database.")
    check_p.add_argument("audio_file", help="Path to the audio file to check.")
    check_p.add_argument("--title", default="Unknown Song", help="Song title (informational).")
    check_p.add_argument("--artist", default="", help="Artist name (informational).")
    check_p.add_argument(
        "--top", type=int, default=5, metavar="N",
        help="Number of closest matches to display (default: 5).",
    )
    check_p.add_argument(
        "--json", dest="as_json", action="store_true",
        help="Output result as JSON.",
    )

    # -- list -----------------------------------------------------------------
    sub.add_parser("list", help="List all songs in the reference database.")

    # -- remove ---------------------------------------------------------------
    rm_p = sub.add_parser("remove", help="Remove a song from the database by id.")
    rm_p.add_argument("song_id", type=int, help="Database id of the song to remove.")

    return parser


def cmd_add(detector: SongDetector, args: argparse.Namespace) -> int:
    print(f"Extracting embedding from '{args.audio_file}' …")
    song_id = detector.add_song_file(args.audio_file, title=args.title, artist=args.artist)
    print(f"✅  Added '{args.title}' (id={song_id}) to the database.")
    return 0


def cmd_check(detector: SongDetector, args: argparse.Namespace) -> int:
    if detector.db.song_count() == 0:
        print("⚠️  The reference database is empty. Add some songs first with the 'add' command.")
        return 1

    print(f"Analysing '{args.audio_file}' …")
    result = detector.check_file(
        args.audio_file,
        query_title=args.title,
        query_artist=args.artist,
        top_n=args.top,
    )

    if args.as_json:
        print(json.dumps(result.as_dict(), indent=2))
        return 0

    print()
    print(f"  Song   : {result.query_title}" + (f" – {result.query_artist}" if result.query_artist else ""))
    print(f"  Threshold : {result.threshold * 100:.0f}%")
    print()

    if not result.matches:
        print("  No matches found in the database.")
        return 0

    header = f"  {'#':<4} {'Title':<30} {'Artist':<20} {'Similarity':>10}  {'Flag':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, m in enumerate(result.matches, 1):
        flag = "🚨 COPY" if m.similarity >= result.threshold else "✅ OK  "
        print(f"  {i:<4} {m.title[:30]:<30} {(m.artist or '—')[:20]:<20} {m.similarity * 100:>9.1f}%  {flag}")

    print()
    if result.is_copied:
        top = result.top_match
        print(f"🚨  VERDICT: Potential copy of '{top.title}' ({top.similarity * 100:.1f}% similar).")
    else:
        top = result.top_match
        if top:
            print(f"✅  VERDICT: Sounds original (closest match: '{top.title}' at {top.similarity * 100:.1f}%).")
        else:
            print("✅  VERDICT: No matches — database is empty.")

    return 0


def cmd_list(detector: SongDetector, args: argparse.Namespace) -> int:
    songs = detector.list_songs()
    if not songs:
        print("The reference database is empty.")
        return 0

    header = f"  {'ID':<6} {'Title':<35} {'Artist':<25} {'Added':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s in songs:
        print(f"  {s['id']:<6} {s['title'][:35]:<35} {(s['artist'] or '—')[:25]:<25} {s['added_at'][:10]:>10}")
    print(f"\n  Total: {len(songs)} song(s)")
    return 0


def cmd_remove(detector: SongDetector, args: argparse.Namespace) -> int:
    if detector.remove_song(args.song_id):
        print(f"✅  Removed song #{args.song_id} from the database.")
        return 0
    print(f"⚠️  Song #{args.song_id} not found in the database.")
    return 1


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    detector = SongDetector(db_path=args.db, threshold=args.threshold)

    dispatch = {
        "add": cmd_add,
        "check": cmd_check,
        "list": cmd_list,
        "remove": cmd_remove,
    }

    try:
        exit_code = dispatch[args.command](detector, args)
    except FileNotFoundError as exc:
        print(f"❌  File not found: {exc}", file=sys.stderr)
        exit_code = 1
    except Exception as exc:
        print(f"❌  Error: {exc}", file=sys.stderr)
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
