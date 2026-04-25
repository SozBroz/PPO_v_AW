"""
Pickle I/O for **live** engine snapshots used by :class:`rl.env.AWBWEnv` workers.

The training main process (or a refresh script) writes ``{games_id}.pkl``;
Subproc workers read, :func:`copy.deepcopy` the :class:`engine.game.GameState`,
and do **not** re-walk ``load_replay`` on every ``reset()``.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from engine.game import GameState

LIVE_SNAPSHOT_VERSION = 1


def write_live_snapshot(
    path: Path | str,
    state: GameState,
    *,
    games_id: int,
    learner_seat: int,
    awbw_to_engine: dict[int, int] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "v": LIVE_SNAPSHOT_VERSION,
        "games_id": int(games_id),
        "learner_seat": int(learner_seat) & 1,
        "state": state,
        "awbw_to_engine": dict(awbw_to_engine) if awbw_to_engine else {},
    }
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_live_snapshot_dict(path: Path | str) -> dict[str, Any]:
    with open(path, "rb") as f:
        d = pickle.load(f)
    if not isinstance(d, dict) or "state" not in d:
        raise ValueError(f"unrecognized live snapshot format: {path!r}")
    return d
