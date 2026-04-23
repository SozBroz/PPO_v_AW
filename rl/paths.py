"""Repository paths for training and runtime logs (TensorBoard, JSONL, watch state).

All paths under ``REPO_ROOT / "logs"`` so a clean training restart can wipe one tree.

Fleet convention (see :mod:`rl.fleet_logs`): the writer on each machine ALWAYS
lands in ``REPO_ROOT/logs/`` exactly. The Main mirror at ``logs/logs/`` and any
future ``logs/<machine_id>/`` subtrees are read-only conventions managed via
``rl.fleet_logs``; the writer must never accidentally land in one of them
(e.g. if the process is launched from a wrong CWD that resolves ``..`` weirdly).
The startup assert below catches that class of accident at import time.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"

GAME_LOG_PATH = LOGS_DIR / "game_log.jsonl"
SLOW_GAMES_LOG_PATH = LOGS_DIR / "slow_games.jsonl"
HUMAN_DEMOS_PATH = LOGS_DIR / "human_demos.jsonl"
WATCH_STATE_PATH = LOGS_DIR / "watch_state.json"


def _assert_logs_dir_is_writer_root(logs_dir: Path, repo_root: Path) -> None:
    """Guard the writer path: ``logs_dir`` must resolve to ``repo_root/logs`` exactly.

    Refactored out of module-level so tests can call it with synthetic paths.
    Module top-level invokes it with the real :data:`LOGS_DIR` / :data:`REPO_ROOT`.
    """
    expected = (repo_root / "logs").resolve()
    actual = logs_dir.resolve()
    if actual != expected:
        raise RuntimeError(
            f"[paths] LOGS_DIR must be REPO_ROOT/logs exactly (got {logs_dir}). "
            f"Process likely launched from wrong CWD."
        )


_assert_logs_dir_is_writer_root(LOGS_DIR, REPO_ROOT)


def ensure_logs_dir() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
