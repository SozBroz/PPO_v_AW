"""
Durable RHEA self-play game corpus: build, append, and replay JSONL rows.

Each row stores a ``full_trace`` action log plus luck determinism fields so the
engine can reproduce the game bit-exact. Flat ``action_indices`` are derived at
write time for BC/PPO ingestion (opening-book compatible).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from engine.game import GameState, make_initial_state
from engine.map_loader import load_map
from rl.env import _action_to_flat
from tools.export_awbw_replay_actions import _trace_to_action

CORPUS_SCHEMA_VERSION = "1.0"
DEFAULT_CORPUS_PATH = Path("data/corpus/rhea_games.jsonl")
DEFAULT_MAP_POOL = Path("data/gl_map_pool.json")
DEFAULT_MAPS_DIR = Path("data/maps")

_corpus_write_warned = False


def corpus_enabled() -> bool:
    v = (os.environ.get("AWBW_GAME_CORPUS", "1") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def corpus_path() -> Path:
    raw = (os.environ.get("AWBW_GAME_CORPUS_PATH") or "").strip()
    return Path(raw) if raw else DEFAULT_CORPUS_PATH


def _serialize_rng_state(rng: random.Random) -> list[Any]:
    version, state, gauss = rng.getstate()
    return [int(version), [int(x) for x in state], gauss]


def _deserialize_rng_state(data: list[Any]) -> random.Random:
    rng = random.Random()
    rng.setstate((int(data[0]), tuple(int(x) for x in data[1]), data[2]))
    return rng


def _luck_fields(
    *,
    luck_seed: int | None,
    luck_rng_state: list[Any] | None = None,
    state: GameState | None = None,
) -> dict[str, Any]:
    if luck_seed is not None:
        return {"luck_seed": int(luck_seed)}
    if luck_rng_state is not None:
        return {"luck_rng_state": luck_rng_state}
    if state is not None:
        return {"luck_rng_state": _serialize_rng_state(state.luck_rng)}
    raise ValueError("luck_seed or luck_rng_state required")


def _checkpoint_meta() -> dict[str, Any] | None:
    candidates: list[Path] = []
    env_path = (os.environ.get("AWBW_VALUE_CHECKPOINT_PATH") or "").strip()
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path("checkpoints/value_rhea_latest.pt"))
    for p in candidates:
        try:
            if p.is_file():
                st = p.stat()
                return {
                    "path": str(p.resolve()),
                    "mtime": float(st.st_mtime),
                    "size": int(st.st_size),
                }
        except OSError:
            continue
    return None


def _search_config_fingerprint() -> str | None:
    for rel in ("checkpoints/hparams_parallel.json", "runs/rhea_value/hparams_parallel.json"):
        p = Path(rel)
        try:
            if p.is_file():
                return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        except OSError:
            continue
    return None


def _make_initial_for_row(
    row: dict[str, Any],
    *,
    map_pool: Path,
    maps_dir: Path,
) -> GameState:
    map_data = load_map(int(row["map_id"]), map_pool, maps_dir)
    mk: dict[str, Any] = {
        "starting_funds": 0,
        "tier_name": str(row.get("tier") or "T2"),
    }
    if row.get("luck_seed") is not None:
        mk["luck_seed"] = int(row["luck_seed"])
    if row.get("replay_first_mover") is not None:
        mk["replay_first_mover"] = int(row["replay_first_mover"])
    if row.get("max_days") is not None:
        md = int(row["max_days"])
        mk["max_days"] = md
        mk["max_turns"] = md
    st = make_initial_state(
        map_data,
        int(row["p0_co_id"]),
        int(row["p1_co_id"]),
        **mk,
    )
    if row.get("luck_rng_state") is not None:
        st.luck_rng = _deserialize_rng_state(row["luck_rng_state"])
    if row.get("p0_cop_activation_disabled"):
        st.co_states[0].cop_activation_disabled = True
    if row.get("p1_cop_activation_disabled"):
        st.co_states[1].cop_activation_disabled = True
    return st


def _compute_action_indices(
    full_trace: list[dict[str, Any]],
    *,
    map_id: int,
    p0_co_id: int,
    p1_co_id: int,
    tier: str,
    luck_seed: int | None,
    luck_rng_state: list[Any] | None,
    replay_first_mover: int | None,
    max_days: int | None,
    p0_cop_activation_disabled: bool,
    p1_cop_activation_disabled: bool,
    map_pool: Path,
    maps_dir: Path,
) -> list[int]:
    row = {
        "map_id": map_id,
        "p0_co_id": p0_co_id,
        "p1_co_id": p1_co_id,
        "tier": tier,
        "luck_seed": luck_seed,
        "luck_rng_state": luck_rng_state,
        "replay_first_mover": replay_first_mover,
        "max_days": max_days,
        "p0_cop_activation_disabled": p0_cop_activation_disabled,
        "p1_cop_activation_disabled": p1_cop_activation_disabled,
    }
    st = _make_initial_for_row(row, map_pool=map_pool, maps_dir=maps_dir)
    indices: list[int] = []
    for entry in full_trace:
        act = _trace_to_action(entry)
        indices.append(int(_action_to_flat(act, st)))
        st.step(act, oracle_mode=True)
    return indices


def build_corpus_row(
    state: GameState,
    *,
    map_id: int,
    map_name: str,
    tier: str,
    p0_co_id: int,
    p1_co_id: int,
    luck_seed: int | None = None,
    luck_rng_state: list[Any] | None = None,
    replay_first_mover: int | None = None,
    max_days: int | None = None,
    p0_cop_activation_disabled: bool = False,
    p1_cop_activation_disabled: bool = False,
    opening_book_id_p0: str | None = None,
    opening_book_id_p1: str | None = None,
    machine_id: str | None = None,
    timestamp: float | None = None,
    truncated: bool = False,
    truncation_reason: str | None = None,
    map_pool: Path | None = None,
    maps_dir: Path | None = None,
) -> dict[str, Any]:
    """Build one corpus JSON object from a terminal ``GameState``."""
    pool = map_pool or DEFAULT_MAP_POOL
    mdir = maps_dir or DEFAULT_MAPS_DIR
    full_trace = [dict(e) for e in state.full_trace]
    luck = _luck_fields(
        luck_seed=luck_seed,
        luck_rng_state=luck_rng_state,
        state=state,
    )
    ts = float(timestamp if timestamp is not None else time.time())
    ts_iso = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    action_indices = _compute_action_indices(
        full_trace,
        map_id=int(map_id),
        p0_co_id=int(p0_co_id),
        p1_co_id=int(p1_co_id),
        tier=str(tier),
        luck_seed=luck.get("luck_seed"),
        luck_rng_state=luck.get("luck_rng_state"),
        replay_first_mover=replay_first_mover,
        max_days=max_days,
        p0_cop_activation_disabled=bool(p0_cop_activation_disabled),
        p1_cop_activation_disabled=bool(p1_cop_activation_disabled),
        map_pool=pool,
        maps_dir=mdir,
    )
    row: dict[str, Any] = {
        "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "map_id": int(map_id),
        "map_name": str(map_name),
        "tier": str(tier),
        "p0_co_id": int(p0_co_id),
        "p1_co_id": int(p1_co_id),
        "full_trace": full_trace,
        "action_indices": action_indices,
        "n_actions": len(full_trace),
        "replay_first_mover": replay_first_mover,
        "winner": state.winner,
        "win_condition": state.win_reason,
        "days": int(state.turn),
        "turns": int(state.turn),
        "truncated": bool(truncated),
        "truncation_reason": truncation_reason,
        "machine_id": machine_id,
        "timestamp": ts,
        "timestamp_iso": ts_iso,
        "opening_book_id_p0": opening_book_id_p0,
        "opening_book_id_p1": opening_book_id_p1,
        "p0_cop_activation_disabled": bool(p0_cop_activation_disabled),
        "p1_cop_activation_disabled": bool(p1_cop_activation_disabled),
    }
    if max_days is not None:
        row["max_days"] = int(max_days)
    row.update(luck)
    ckpt = _checkpoint_meta()
    if ckpt is not None:
        row["value_net_checkpoint"] = ckpt
    fp = _search_config_fingerprint()
    if fp is not None:
        row["search_config_fingerprint"] = fp
    return row


def build_corpus_row_from_env(env: Any) -> dict[str, Any]:
    """Build a corpus row from a finished :class:`rl.env.AWBWEnv` episode."""
    if env.state is None:
        raise ValueError("env.state is None")
    info = getattr(env, "_episode_info", {}) or {}
    book_log = getattr(env, "_opening_book_log", {}) or {}
    return build_corpus_row(
        env.state,
        map_id=int(info.get("map_id") or env.state.map_data.map_id),
        map_name=str(info.get("map_name") or env.state.map_data.name),
        tier=str(info.get("tier") or env.state.tier_name or "T2"),
        p0_co_id=int(info.get("p0_co") or env.state.co_states[0].co_id),
        p1_co_id=int(info.get("p1_co") or env.state.co_states[1].co_id),
        luck_seed=info.get("luck_seed"),
        luck_rng_state=info.get("luck_rng_state"),
        replay_first_mover=getattr(env, "_opening_player", None),
        max_days=info.get("max_days"),
        p0_cop_activation_disabled=bool(info.get("p0_cop_activation_disabled")),
        p1_cop_activation_disabled=bool(info.get("p1_cop_activation_disabled")),
        opening_book_id_p0=book_log.get("opening_book_id_p0"),
        opening_book_id_p1=book_log.get("opening_book_id_p1"),
        machine_id=os.environ.get("AWBW_MACHINE_ID"),
        truncated=bool(getattr(env, "_log_episode_truncated", False)),
        truncation_reason=getattr(env, "_log_episode_truncation_reason", None),
    )


def append_corpus_row(row: dict[str, Any], path: Path | None = None) -> None:
    """Append one JSONL line (raises on I/O failure)."""
    out = path or corpus_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, separators=(",", ":")) + "\n")


def write_env_corpus_row(env: Any, *, path: Path | None = None) -> None:
    """Build and append a corpus row; log a warning once on failure."""
    global _corpus_write_warned
    if not corpus_enabled():
        return
    try:
        row = build_corpus_row_from_env(env)
        append_corpus_row(row, path=path)
    except Exception as exc:
        if not _corpus_write_warned:
            warnings.warn(f"AWBW game corpus write failed: {exc!r}", stacklevel=2)
            _corpus_write_warned = True


def iter_corpus_states(
    row: dict[str, Any],
    *,
    map_pool: Path | None = None,
    maps_dir: Path | None = None,
) -> Iterator[tuple[GameState, int, int]]:
    """Yield ``(state_before, flat_action_idx, acting_seat)`` for each trace step.

    The yielded ``GameState`` is mutated in place on the next iteration; encode or
    copy before advancing if you need to retain prior positions.
    """
    pool = map_pool or DEFAULT_MAP_POOL
    mdir = maps_dir or DEFAULT_MAPS_DIR
    st = _make_initial_for_row(row, map_pool=pool, maps_dir=mdir)
    trace = row.get("full_trace") or []
    indices = row.get("action_indices") or []
    if len(indices) != len(trace):
        raise ValueError(
            f"action_indices length {len(indices)} != full_trace length {len(trace)}"
        )
    for entry, flat_idx in zip(trace, indices):
        acting_seat = int(st.active_player)
        yield st, int(flat_idx), acting_seat
        act = _trace_to_action(entry)
        st.step(act, oracle_mode=True)


def replay_corpus_row(
    row: dict[str, Any],
    *,
    map_pool: Path | None = None,
    maps_dir: Path | None = None,
) -> GameState:
    """Replay ``full_trace`` and return the terminal ``GameState``."""
    pool = map_pool or DEFAULT_MAP_POOL
    mdir = maps_dir or DEFAULT_MAPS_DIR
    st = _make_initial_for_row(row, map_pool=pool, maps_dir=mdir)
    for entry in row.get("full_trace") or []:
        act = _trace_to_action(entry)
        st.step(act, oracle_mode=True)
    return st


def state_fingerprint(state: GameState) -> dict[str, Any]:
    """Compact end-state summary for replay validation tests."""
    units: list[tuple[Any, ...]] = []
    for player in (0, 1):
        for u in state.units[player]:
            if u.is_alive:
                units.append(
                    (
                        int(player),
                        int(u.unit_id),
                        tuple(u.pos),
                        int(u.hp),
                        u.unit_type.name,
                    )
                )
    return {
        "winner": state.winner,
        "win_condition": state.win_reason,
        "days": int(state.turn),
        "turns": int(state.turn),
        "funds": [int(state.funds[0]), int(state.funds[1])],
        "alive_unit_count": [
            sum(1 for u in state.units[0] if u.is_alive),
            sum(1 for u in state.units[1] if u.is_alive),
        ],
        "property_count": [
            int(state.count_properties(0)),
            int(state.count_properties(1)),
        ],
        "units": sorted(units),
    }
