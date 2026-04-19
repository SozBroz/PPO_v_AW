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
) -> dict[str, Any]:
    """Same fields as ``server.play_human._append_human_demo`` (pre-step state)."""
    spatial, scalars = encode_state(state)
    mask = _get_action_mask(state)
    return {
        "encoder_version": [int(N_SPATIAL_CHANNELS), int(N_SCALARS)],
        "spatial": spatial.tolist(),
        "scalars": scalars.tolist(),
        "action_mask": mask.tolist(),
        "action_idx": int(_action_to_flat(action)),
        "action_stage": state.action_stage.name,
        "action_label": _action_label(action),
        "active_player": int(state.active_player),
        "map_id": map_id,
        "tier": tier,
        "session_id": session_id,
    }


def iter_demo_rows_from_trace_record(
    record: dict[str, Any],
    *,
    map_pool: Path | None = None,
    maps_dir: Path | None = None,
    session_prefix: str = "trace",
    include_move_stage: bool = False,
) -> Iterator[dict[str, Any]]:
    """
    Replay ``full_trace`` and yield one demo row per engine step when
    ``active_player == 0`` (same contract as training / play UI).
    """
    pool = map_pool or _DEFAULT_POOL
    mdir = maps_dir or _DEFAULT_MAPS
    map_data = load_map(record["map_id"], pool, mdir)
    st = make_initial_state(
        map_data,
        record["co0"],
        record["co1"],
        starting_funds=0,
        tier_name=record.get("tier", "T2"),
    )
    sid = f"{session_prefix}:{record.get('map_id', '')}"
    for i, entry in enumerate(record["full_trace"]):
        act = _trace_to_action(entry)
        if st.active_player == 0:
            if include_move_stage or st.action_stage != ActionStage.MOVE:
                yield build_demo_row_dict(
                    f"{sid}:{i}",
                    st,
                    act,
                    int(record["map_id"]),
                    str(record.get("tier", "")),
                )
        st.step(act)


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
) -> list[dict[str, Any]]:
    """
    Full oracle ``p:`` replay with a hook so rows match per-engine-step logging
    for player 0 (same semantics as trace ingest).
    """
    pool = map_pool or _DEFAULT_POOL
    mdir = maps_dir or _DEFAULT_MAPS
    out: list[dict[str, Any]] = []
    n = 0

    def before(st: GameState, act: Action) -> None:
        nonlocal n
        if int(st.active_player) != 0:
            return
        if not include_move_stage and st.action_stage == ActionStage.MOVE:
            return
        out.append(
            build_demo_row_dict(
                f"{session_prefix}:{zip_path.stem}:{n}",
                st,
                act,
                map_id,
                tier_name,
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


def write_demo_rows_jsonl(rows: Iterator[dict[str, Any]], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            n += 1
    return n
