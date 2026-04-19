"""Replay viewer routes."""
import json
from pathlib import Path

from flask import Blueprint, render_template, jsonify, current_app

bp = Blueprint("replay", __name__, url_prefix="/replay")


def _load_game_records(log_path: Path) -> list[dict]:
    """Load game_log.jsonl as a list, skipping blank lines written by the logger.

    `rl.env._append_game_log_line` appends each record followed by `\\n\\n`,
    so the file interleaves data and blank lines. We index only non-empty
    stripped lines so `game_idx` stays stable across the listing and the
    per-game API.
    """
    if not log_path.exists():
        return []
    records: list[dict] = []
    with open(log_path, encoding="utf-8") as f:
        for idx, raw in enumerate(f):
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec["game_idx"] = len(records)
            records.append(rec)
    return records


@bp.route("/")
def replay_list():
    """List recent games from game_log.jsonl."""
    data_dir = current_app.config["DATA_DIR"]
    log_path = data_dir / "game_log.jsonl"

    all_records = _load_game_records(log_path)
    games = list(reversed(all_records[-50:]))
    return render_template("replay.html", games=games)


@bp.route("/<int:game_idx>")
def replay_game(game_idx: int):
    return render_template("replay.html", game_idx=game_idx, games=[])


@bp.route("/api/<int:game_idx>")
def replay_api(game_idx: int):
    """Return game data for a specific game index."""
    data_dir = current_app.config["DATA_DIR"]
    log_path = data_dir / "game_log.jsonl"

    records = _load_game_records(log_path)
    if not records:
        return jsonify({"error": "No game log"}), 404

    if game_idx < 0 or game_idx >= len(records):
        return jsonify({"error": "Game not found"}), 404

    return jsonify(records[game_idx])
