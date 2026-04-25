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

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"

# ---------------------------------------------------------------------------
# AWBW Replay Player (optional gitignored clone — improved Desktop build)
# ---------------------------------------------------------------------------
REPLAY_PLAYER_THIRD_PARTY_DIR = REPO_ROOT / "third_party" / "AWBW-Replay-Player"
"""Root of the local AWBW Replay Player fork (sources + ``dotnet build``)."""

REPLAY_PLAYER_DESKTOP_BIN = REPLAY_PLAYER_THIRD_PARTY_DIR / "AWBWApp.Desktop" / "bin"
"""Directory containing ``Release`` / ``Debug`` and ``net*`` output folders."""

REPLAY_PLAYER_EXE_ENV = "AWBW_REPLAY_PLAYER_EXE"
"""Environment variable name; when set to an exe path, that build wins over ``REPLAY_PLAYER_DESKTOP_BIN``."""


def resolve_awbw_replay_player_exe(repo_root: Path | None = None) -> Path | None:
    """Return ``AWBW Replay Player.exe`` if found.

    Resolution order:

    1. ``os.environ[REPLAY_PLAYER_EXE_ENV]`` when it points to an existing file.
    2. Under ``<repo_root>/third_party/AWBW-Replay-Player/AWBWApp.Desktop/bin``,
       prefer ``Release`` then ``Debug``, known ``net*`` TFMs, then any ``net*`` subdir.
    """
    root = REPO_ROOT if repo_root is None else Path(repo_root).resolve()
    env = os.environ.get(REPLAY_PLAYER_EXE_ENV, "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p.resolve()
    base = root / "third_party" / "AWBW-Replay-Player" / "AWBWApp.Desktop" / "bin"
    for cfg in ("Release", "Debug"):
        for tfm in ("net8.0", "net7.0", "net6.0"):
            cand = base / cfg / tfm / "AWBW Replay Player.exe"
            if cand.is_file():
                return cand.resolve()
    for cfg in ("Release", "Debug"):
        d = base / cfg
        if not d.is_dir():
            continue
        for sub in sorted(d.iterdir(), key=lambda x: x.name, reverse=True):
            if sub.is_dir() and sub.name.startswith("net"):
                exe = sub / "AWBW Replay Player.exe"
                if exe.is_file():
                    return exe.resolve()
    return None

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
