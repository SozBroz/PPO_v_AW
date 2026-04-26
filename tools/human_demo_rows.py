"""
Build ``human_demos.jsonl`` row dicts matching ``server.play_human._append_human_demo``.

Used by offline replay ingest (trace JSON or oracle zip) without Flask.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

from engine.action import Action, ActionStage
from engine.game import GameState, make_initial_state
from engine.map_loader import load_map

from rl.encoder import N_SCALARS, N_SPATIAL_CHANNELS, encode_state
from rl.env import _action_label, _action_to_flat, _get_action_mask
from tools.export_awbw_replay_actions import _trace_to_action

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_POOL = _REPO / "data" / "gl_map_pool.json"
_DEFAULT_MAPS = _REPO / "data" / "maps"


def build_demo_row_dict(
    session_id: str,
    state: GameState,
    action: Action,
    map_id: Optional[int],
    tier: Optional[str],
    **extra: Any,
) -> dict[str, Any]:
    """Same fields as ``server.play_human._append_human_demo`` (pre-step state)."""
    spatial, scalars = encode_state(state)
    mask = _get_action_mask(state)
    row: dict[str, Any] = {
        "encoder_version": [int(N_SPATIAL_CHANNELS), int(N_SCALARS)],
        "spatial": spatial.tolist(),
        "scalars": scalars.tolist(),
        "action_mask": mask.tolist(),
        "action_idx": int(_action_to_flat(action, state)),
        "action_stage": state.action_stage.name,
        "action_label": _action_label(action),
        "active_player": int(state.active_player),
        "map_id": map_id,
        "tier": tier,
        "session_id": session_id,
    }
    if extra:
        row.update(extra)
    return row


def iter_demo_rows_from_trace_record(
    record: dict[str, Any],
    *,
    map_pool: Path | None = None,
    maps_dir: Path | None = None,
    session_prefix: str = "trace",
    include_move_stage: bool = False,
    seats: tuple[int, ...] = (0,),
    max_turn: int | None = None,
    opening_only: bool = False,
    source_game_id: int | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Replay ``full_trace`` and yield demo rows for each engine step where
    ``active_player`` is in ``seats`` (default ``(0,)`` = learner / P0 only).

    ``max_turn`` — if set, stop before applying any trace entry with
    ``entry[\"turn\"] > max_turn`` (AWBW calendar day in exported traces).

    ``opening_only`` — set ``opening_segment`` on every emitted row.
    """
    pool = map_pool or _DEFAULT_POOL
    mdir = maps_dir or _DEFAULT_MAPS
    map_data = load_map(record["map_id"], pool, mdir)
    co0 = int(record["co0"])
    co1 = int(record["co1"])
    st = make_initial_state(
        map_data,
        co0,
        co1,
        starting_funds=0,
        tier_name=record.get("tier", "T2"),
    )
    mid = int(record["map_id"])
    # Stable id for build_opening_book grouping (one session per source trace, not per row).
    book_session_id = f"{session_prefix}:m{mid}"
    for i, entry in enumerate(record["full_trace"]):
        t = int(entry.get("turn", 0) or 0)
        if max_turn is not None and t > max_turn:
            break
        act = _trace_to_action(entry)
        ap = int(st.active_player)
        if ap in seats:
            if include_move_stage or st.action_stage != ActionStage.MOVE:
                ex: dict[str, Any] = {
                    "awbw_turn": t,
                    "trace_index": i,
                    "demo_seat": ap,
                    "co0": co0,
                    "co1": co1,
                    "book_session_id": book_session_id,
                }
                if source_game_id is not None:
                    ex["source_game_id"] = int(source_game_id)
                if opening_only:
                    ex["opening_segment"] = True
                yield build_demo_row_dict(
                    book_session_id,
                    st,
                    act,
                    mid,
                    str(record.get("tier", "")),
                    **ex,
                )
        st.step(act)


def infer_training_meta_from_awbw_zip(
    zip_path: Path,
    *,
    map_pool: Path | None = None,
) -> dict[str, Any]:
    """Read first PHP snapshot from a site/oracle zip; return map_id, co0, co1, tier."""
    from tools.diff_replay_zips import load_replay

    pool_path = map_pool or _DEFAULT_POOL
    with open(pool_path, encoding="utf-8") as f:
        pool: list[dict] = __import__("json").load(f)
    snaps = load_replay(zip_path)
    g = snaps[0]
    mid = int(g.get("maps_id") or 0)
    pl = g.get("players") or {}
    p0 = pl.get(0) if isinstance(pl, dict) else None
    p1 = pl.get(1) if isinstance(pl, dict) else None
    if p0 is None or p1 is None:
        raise ValueError(f"zip {zip_path}: missing players[0/1] in first snapshot")
    co0 = int(p0.get("co_id") or 1)
    co1 = int(p1.get("co_id") or 1)
    tier = "T3"
    for m in pool:
        if int(m.get("map_id", 0) or 0) == mid:
            t = m.get("tier")
            if t:
                tier = str(t)
            break
    return {"map_id": mid, "co0": co0, "co1": co1, "tier": tier}


def collect_demo_rows_from_oracle_zip(
    zip_path: Path,
    *,
    map_pool: Path | None = None,
    maps_dir: Path | None = None,
    map_id: int,
    co0: int,
    co1: int,
    tier_name: str,
    session_prefix: str = "oracle_zip",
    include_move_stage: bool = False,
    seats: tuple[int, ...] = (0,),
    max_calendar_turn: int | None = None,
    opening_only: bool = False,
    source_game_id: int | None = None,
) -> list[dict[str, Any]]:
    """
    Full oracle ``p:`` replay with a hook so rows match per-engine-step logging
    for the requested ``seats`` (default P0 only).
    """
    pool = map_pool or _DEFAULT_POOL
    mdir = maps_dir or _DEFAULT_MAPS
    out: list[dict[str, Any]] = []
    n = 0
    book_session_id = f"{session_prefix}:zip{zip_path.stem}:m{int(map_id)}"

    def before(st: GameState, act: Action) -> None:
        nonlocal n
        if int(st.active_player) not in seats:
            return
        if not include_move_stage and st.action_stage == ActionStage.MOVE:
            return
        cal = int(getattr(st, "turn", 0) or 0)
        if max_calendar_turn is not None and cal > int(max_calendar_turn):
            return
        ex: dict[str, Any] = {
            "calendar_turn": cal,
            "trace_index": n,
            "demo_seat": int(st.active_player),
            "co0": int(co0),
            "co1": int(co1),
            "book_session_id": book_session_id,
        }
        if source_game_id is not None:
            ex["source_game_id"] = int(source_game_id)
        if opening_only:
            ex["opening_segment"] = True
        out.append(
            build_demo_row_dict(
                book_session_id,
                st,
                act,
                map_id,
                tier_name,
                **ex,
            )
        )
        n += 1

    from tools.oracle_zip_replay import replay_oracle_zip

    replay_oracle_zip(
        zip_path,
        map_pool=pool,
        maps_dir=mdir,
        map_id=map_id,
        co0=co0,
        co1=co1,
        tier_name=tier_name,
        before_engine_step=before,
    )
    return out


def write_demo_rows_jsonl(
    rows: Iterator[dict[str, Any]], out_path: Path, *, append: bool = False
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    mode = "a" if append else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            n += 1
    return n
