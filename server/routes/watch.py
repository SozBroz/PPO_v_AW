"""Live game watcher routes."""
import json
import subprocess
import sys
from pathlib import Path
from flask import Blueprint, render_template, jsonify, current_app, request

from rl.paths import WATCH_STATE_PATH

bp = Blueprint("watch", __name__)


@bp.route("/")
@bp.route("/watch")
def watch():
    return render_template("watch.html")


@bp.route("/api/watch/state")
def watch_state():
    """Return current watch_state.json for live polling."""
    if not WATCH_STATE_PATH.exists():
        return jsonify({
            "status": "no_game",
            "message": "No active game. Run: python train.py --watch-only"
        })

    with open(WATCH_STATE_PATH) as f:
        state = json.load(f)
    return jsonify(state)


@bp.route("/api/watch/start", methods=["POST"])
def start_watch_game():
    """Spawn a watch game in background and return."""
    data = request.get_json() or {}
    map_id = data.get("map_id")
    co_p0 = data.get("co_p0", 1)
    co_p1 = data.get("co_p1", 7)

    root = current_app.config["ROOT"]
    cmd = [sys.executable, str(root / "train.py"), "--watch-only"]
    if map_id:
        cmd += ["--map-id", str(map_id)]
    cmd += ["--co-p0", str(co_p0), "--co-p1", str(co_p1)]

    subprocess.Popen(cmd, cwd=str(root))
    return jsonify({"status": "started", "cmd": " ".join(cmd)})
