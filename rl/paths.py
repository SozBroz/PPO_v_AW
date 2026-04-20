"""Repository paths for training and runtime logs (TensorBoard, JSONL, watch state).

All paths under ``REPO_ROOT / "logs"`` so a clean training restart can wipe one tree.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"

GAME_LOG_PATH = LOGS_DIR / "game_log.jsonl"
SLOW_GAMES_LOG_PATH = LOGS_DIR / "slow_games.jsonl"
HUMAN_DEMOS_PATH = LOGS_DIR / "human_demos.jsonl"
WATCH_STATE_PATH = LOGS_DIR / "watch_state.json"


def ensure_logs_dir() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
