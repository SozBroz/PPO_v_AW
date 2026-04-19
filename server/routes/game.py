"""Human vs bot game routes."""
from __future__ import annotations

from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request

from server import play_human

bp = Blueprint("game", __name__, url_prefix="/play")


def _checkpoint_dir() -> Path:
    return Path(current_app.config["CHECKPOINT_DIR"])


@bp.route("/")
def play():
    return render_template("play.html")


@bp.route("/api/new", methods=["POST"])
def api_new():
    data = request.get_json(silent=True) or {}
    payload, err = play_human.new_session(
        map_id=data.get("map_id"),
        tier=data.get("tier"),
        human_co_id=data.get("human_co_id", data.get("co_id")),
        bot_co_id=data.get("bot_co_id"),
        checkpoint_dir=_checkpoint_dir(),
    )
    code = 200 if err is None and payload.get("ok") else 503
    return jsonify(payload), code


@bp.route("/api/state/<session_id>", methods=["GET"])
def api_state(session_id: str):
    payload, err = play_human.get_session_state(session_id)
    if err == "Unknown session_id":
        return jsonify(payload), 404
    return jsonify(payload)


@bp.route("/api/step", methods=["POST"])
def api_step():
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    if not sid:
        return jsonify({"ok": False, "error": "session_id required"}), 400
    payload, err = play_human.apply_human_step(sid, data, _checkpoint_dir())
    code = 200 if payload.get("ok") else 400
    return jsonify(payload), code


@bp.route("/api/cancel_selection", methods=["POST"])
def api_cancel_selection():
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    if not sid:
        return jsonify({"ok": False, "error": "session_id required"}), 400
    payload, err = play_human.cancel_selection(sid)
    if err == "Unknown session_id":
        return jsonify(payload), 404
    code = 200 if payload.get("ok") else 400
    return jsonify(payload), code


# Legacy stub paths (redirect clients to /api/step)
@bp.route("/api/move", methods=["POST"])
def human_move():
    return jsonify({"error": "Use POST /play/api/step with JSON body per play API"}), 410
