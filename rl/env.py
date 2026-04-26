"""
Gymnasium environment for AWBW self-play.

The environment wraps the AWBW game engine and exposes:
  - observation_space: Dict {'spatial': Box, 'scalars': Box}
  - action_space: Discrete(ACTION_SPACE_SIZE)
  - action_masks(): bool array for MaskablePPO compatibility

By default the trained agent controls engine seat 0; the other seat is stepped
automatically (checkpoint opponent or random). With ``AWBW_SEAT_BALANCE=1`` the
learner seat is sampled 50/50 each episode (ego-centric obs + learner-frame Φ).

``AWBW_VECENV_OBS_COPY`` (default on): return C-contiguous observation copies from
``_get_obs`` so ``SubprocVecEnv`` workers do not pickle arrays that alias reused
buffers; helps Windows multiprocessing when ``n_envs`` is large. Set ``0`` to
restore zero-copy returns (slightly faster single-process / tests).
"""
from rl import _win_triton_warnings

_win_triton_warnings.apply()

import collections
import copy
import json
import os
import random
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from threading import Lock

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Keep the env's default reward mode and engine-side capture-shaping gate aligned.
# ``engine.game`` reads this at import time to suppress legacy capture bonuses.
os.environ.setdefault("AWBW_REWARD_SHAPING", "phi")

from engine.game import GameState, make_initial_state, MAX_TURNS
from engine.map_loader import MapData, load_map
from engine.action import Action, ActionStage, ActionType, get_legal_actions
from engine.unit import UnitType, UNIT_STATS
from engine.terrain import get_terrain
from engine.combat import damage_range
from engine.belief import BeliefState

from rl.encoder import encode_state, GRID_SIZE, N_SPATIAL_CHANNELS, N_SCALARS
from rl.network import ACTION_SPACE_SIZE
from rl.paths import GAME_LOG_PATH, SLOW_GAMES_LOG_PATH
from rl.log_timestamp import log_now_iso, log_timestamp_iso
from rl.heuristic_termination import (
    SPIRIT_BROKEN_REASON,
    army_value_for_player,
    config_from_env,
    run_calendar_day,
    DEFAULT_DISAGREEMENT_LOG,
)
from server.write_watch_state import board_dict

ROOT = Path(__file__).parent.parent
POOL_PATH = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"

# region agent log
_AGENT_DEBUG_LOG_PATH = ROOT / "debug-a6d5a1.log"
_AGENT_DEBUG_SESSION_ID = "a6d5a1"


def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    try:
        payload = {
            "sessionId": _AGENT_DEBUG_SESSION_ID,
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": {**data, "pid": os.getpid()},
            "timestamp": int(time.time() * 1000),
            "timestamp_iso": log_now_iso(),
        }
        with open(_AGENT_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass
# endregion

# Session game counter: set by training (SelfPlayTrainer) so all SubprocVecEnv workers share one sequence.
SESSION_GAME_COUNTER_DB_ENV = "AWBW_SESSION_GAME_COUNTER_DB"

# When set to "1", each finished game in game_log.jsonl carries a `frames` array with one
# board snapshot per engine step (P0 + opponent substeps). Disabled by default because the
# payload grows roughly O(turns * actions_per_turn) per record.
LOG_REPLAY_FRAMES_ENV = "AWBW_LOG_REPLAY_FRAMES"

# Slow-game threshold (wall seconds). Episodes exceeding this get a compact red-flag
# line appended to logs/slow_games.jsonl alongside the normal game_log row. Override
# via AWBW_SLOW_GAME_WALL_S env var. Zero/negative disables.
SLOW_GAME_WALL_S_ENV = "AWBW_SLOW_GAME_WALL_S"

# Optional per-P0-step stall penalty (subtracted from reward while episode continues).
TIME_COST_ENV = "AWBW_TIME_COST"
# Optional fixed penalty when an episode truncates without an engine terminal result.
TRUNCATION_PENALTY_ENV = "AWBW_TRUNCATION_PENALTY"
# End-turn "hoarding" penalty: subtract when learner ends turn with bank above threshold.
HOARD_FUNDS_THRESHOLD_ENV = "AWBW_HOARD_FUNDS_THRESHOLD"
HOARD_PENALTY_ENV = "AWBW_HOARD_PENALTY"
# Terminal shaping: ``AWBW_INCOME_TERM_COEF`` × (income_props_p0 − p1) / cap_limit when episode ends.
INCOME_TERM_COEF_ENV = "AWBW_INCOME_TERM_COEF"
# When ``1``, zero all BUILD action-mask entries except ``INFANTRY`` (narrow bootstrap).
BUILD_MASK_INFANTRY_ONLY_ENV = "AWBW_BUILD_MASK_INFANTRY_ONLY"
# Probability that the learner's chosen action is overridden by the same
# capture-greedy heuristic that bootstraps the opponent. DAGGER-lite teacher
# mixing — gives P0 the same scaffold P1 has had silently. Read once per
# AWBWEnv instance (workers spawn with this fixed); restart with a lower
# value to decay. See plan p0-capture-architecture-fix Tier 1.
LEARNER_GREEDY_MIX_ENV = "AWBW_LEARNER_GREEDY_MIX"

# When ``1``, each ``AWBWEnv`` records wall time per ``step()`` in a small ring
# buffer for FPS / straggler diagnostics (SubprocVecEnv). Default off — zero deque
# allocation and no extra timers in the hot path.
TRACK_PER_WORKER_TIMES_ENV = "AWBW_TRACK_PER_WORKER_TIMES"
FPS_DIAG_ENV = "AWBW_FPS_DIAG"
MACHINE_ID_ENV = "AWBW_MACHINE_ID"
# Default on: ``_get_obs`` returns C-contiguous copies so ``SubprocVecEnv`` IPC
# pickling never aliases buffers that are reused on the next step (helps Windows
# pipe / multiprocessing stability when ``n_envs`` is high). Set ``0`` to opt out.
VECENV_OBS_COPY_ENV = "AWBW_VECENV_OBS_COPY"


def effective_track_per_worker_times() -> bool:
    """
    Whether to sample per-``step()`` wall times in each env (Subproc workers).

    ``AWBW_TRACK_PER_WORKER_TIMES=0/false/...`` forces off; ``=1`` forces on.
    When unset, default **on** if ``AWBW_FPS_DIAG`` or a non-empty
    ``AWBW_MACHINE_ID`` is set (fleet / ``train.py --fps-diag`` / ``--machine-id``).
    """
    raw = (os.environ.get(TRACK_PER_WORKER_TIMES_ENV) or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    fps = (os.environ.get(FPS_DIAG_ENV) or "").strip().lower()
    if fps in ("1", "true", "yes", "on"):
        return True
    if (os.environ.get(MACHINE_ID_ENV) or "").strip():
        return True
    return False

# Phase 1b: reuse env-owned numpy buffers for ``encode_state`` and ``_get_action_mask``
# in the hot path. Set to ``0`` to use fresh allocations (parity / A–B). Default on.
# Read once per :class:`AWBWEnv` instance (set before spawn in training).
PREALLOCATED_BUFFERS_ENV = "AWBW_PREALLOCATED_BUFFERS"

# Reward shaping mode (plan rl_capture-combat_recalibration).
#   "phi" (default)   — potential-based: per-step reward gets
#                       Φ(s_after) − Φ(s_before) in the learner frame, plus
#                       a one-time kill bonus (see PHI_ENEMY_KILL_BONUS_FRAC,
#                       _phi_enemy_kill_one_time_bonus).
#                       On engine day-cap resolution (``win_reason``): replace the
#                       usual sparse ±1/0 with scaled outcomes — max_turns_tie or
#                       (legacy) max_turns_draw → −0.1; max_turns_tiebreak win → +0.5;
#                       tiebreak loss with
#                       ≥2 property deficit → −0.5 (else −1). See
#                       _apply_phi_sparse_terminal_replacement.
#   "level"           — legacy property + unit-value differential (me − enemy).
# When mode is "phi", AWBW_PHI_PROFILE picks defaults for α,β,κ if the
# per-coefficient env vars are unset. Explicit AWBW_PHI_ALPHA / _BETA / _KAPPA
# still override. Default profile is "balanced"; "capture" skews toward κ.
REWARD_SHAPING_ENV = "AWBW_REWARD_SHAPING"
# Seat-balanced rollouts: ``1`` / ``true`` → random learner seat {0,1} each reset.
SEAT_BALANCE_ENV = "AWBW_SEAT_BALANCE"
# Force learner seat when seat balance is off (0 or 1).
LEARNER_SEAT_ENV = "AWBW_LEARNER_SEAT"
PHI_PROFILE_ENV = "AWBW_PHI_PROFILE"
PHI_ALPHA_ENV = "AWBW_PHI_ALPHA"   # value-coin coefficient
PHI_BETA_ENV  = "AWBW_PHI_BETA"    # property-count coefficient
PHI_KAPPA_ENV = "AWBW_PHI_KAPPA"   # contested-cap coefficient
# (α, β, κ) when in phi mode and a coefficient env is unset:
PHI_PROFILE_DEFAULTS: dict[str, tuple[float, float, float]] = {
    "balanced": (2e-5, 0.05, 0.05),
    "capture": (2e-5, 0.02, 0.25),
}
# Φ: one-time bonus for removing an enemy unit on the learner’s engine step,
# in the same value units as the army line (``α × cost × hp/100``), scaled
# with ``_phi_alpha`` so it tracks profile/env overrides.
PHI_ENEMY_KILL_BONUS_FRAC = 0.1

# In-process: threads must not interleave JSONL lines. Cross-process: use SQLite (see _append_game_log_line).
_log_lock = Lock()
# When SESSION_GAME_COUNTER_DB_ENV is unset, count completed games in this process only (tests, ad-hoc env use).
_local_session_game_count = 0


def _synthetic_env_cap_property_tiebreak(p0_props: int, p1_props: int) -> tuple[int, str]:
    """
    P0 vs P1 ``count_properties`` margin — same rule as engine calendar max-turns
    (``engine.game.GameState`` end of ``_end_turn`` when ``turn > max_turns``):
    |Δ| ≤ 1 → draw; else the leader wins. Used for ``game_log`` only when the
    episode is env-truncated (``max_env_steps`` / ``max_p1_microsteps``) and the
    engine never set ``winner`` / ``win_reason``.

    Return value is (engine_seat_winner, win_reason) with -1 for draw. Reasons
    ``env_step_cap_*`` are only for env truncation (``max_env_steps`` / ``max_p1_microsteps``),
    distinct from calendar ``max_turns_tie`` / ``max_turns_tiebreak``.
    """
    d = int(p0_props) - int(p1_props)
    if abs(d) <= 1:
        return -1, "env_step_cap_tie"
    if d >= 2:
        return 0, "env_step_cap_tiebreak"
    return 1, "env_step_cap_tiebreak"


def _resolve_opponent_critic_model(opp: object) -> Any:
    """
    Policy module for spirit + heuristic value diag (``predict_values`` on the
    checkpoint). Wrappers such as :class:`rl.opening_book.OpeningBookCheckpointOpponent`
    delegate P1 actions to an inner :class:`_CheckpointOpponent`; expose the same
    ``_model`` the calendar heuristics expect.
    """
    m = getattr(opp, "_model", None)
    if m is not None:
        return m
    inner = getattr(opp, "_inner", None)
    if inner is not None:
        return getattr(inner, "_model", None)
    return None


def _append_game_log_line(record: dict) -> None:
    """
    Assign monotonic ``game_id`` and append one JSONL record.

    When ``SESSION_GAME_COUNTER_DB_ENV`` points at the session SQLite file (set by
    ``SelfPlayTrainer.train()``), ``BEGIN IMMEDIATE`` locks that DB until commit so
    **game_id allocation and the full blob write** run as one critical section across
    all SubprocVecEnv workers (threading.Lock alone is not enough across processes).

    Without that env var, a single threading.Lock serializes id + write in this process.
    """
    GAME_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    db_path = os.environ.get(SESSION_GAME_COUNTER_DB_ENV)
    # Phase 10/11 prereq (game_log schema ≥1.7): stamp every row with the writer's
    # machine identity. Read at write time so a single dev box without the env
    # var emits None and the orchestrator's per-machine slicing degrades
    # cleanly. Held at the writer boundary (not plumbed through log_record)
    # so the two code paths above can't drift.
    machine_id = os.environ.get("AWBW_MACHINE_ID")

    if db_path:
        conn = sqlite3.connect(db_path, timeout=120.0)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS session_seq ("
                "singleton INTEGER PRIMARY KEY CHECK (singleton = 1), n INTEGER NOT NULL)"
            )
            conn.execute("INSERT OR IGNORE INTO session_seq (singleton, n) VALUES (1, 0)")
            conn.execute("UPDATE session_seq SET n = n + 1 WHERE singleton = 1")
            row = conn.execute("SELECT n FROM session_seq WHERE singleton = 1").fetchone()
            game_id = int(row[0])
            full = {"game_id": game_id, "machine_id": machine_id, **record}
            line = json.dumps(full) + "\n\n"
            with open(GAME_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        return

    global _local_session_game_count
    with _log_lock:
        _local_session_game_count += 1
        full = {"game_id": _local_session_game_count, "machine_id": machine_id, **record}
        line = json.dumps(full) + "\n\n"
        with open(GAME_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)

# Encoding constants — kept consistent regardless of actual map dimensions
_ENC_W = 30
_ATTACK_OFFSET = _ENC_W * _ENC_W          # 900
_CAPTURE_IDX = _ATTACK_OFFSET * 2         # 1800
_WAIT_IDX = _CAPTURE_IDX + 1              # 1801
_LOAD_IDX = _CAPTURE_IDX + 2              # 1802
_JOIN_IDX = _CAPTURE_IDX + 3              # 1803  (same-type merge; before UNLOAD block)
_DIVE_HIDE_IDX = _CAPTURE_IDX + 4         # 1804  (Sub dive / Stealth hide)
# UNLOAD: 4 cardinal drop directions × 2 cargo slots = 8 slots.
# Index = _UNLOAD_OFFSET + slot_idx * 4 + dir (0=N, 1=S, 2=W, 3=E).
_UNLOAD_OFFSET = _CAPTURE_IDX + 10        # 1810
_UNLOAD_DIRS   = ((-1, 0), (1, 0), (0, -1), (0, 1))
# MOVE-stage SELECT_UNIT encodes destination tile only (1818..2717). See
# docs/restart_arch/move_encoding_redesign.md
_MOVE_OFFSET = _UNLOAD_OFFSET + 8         # 1818
_BUILD_OFFSET = 10_000
# REPAIR (Black Boat): one index per target tile; kept below _BUILD_OFFSET.
_REPAIR_OFFSET = 3500
# Must match UnitType enum cardinality (27 types).
_N_UNIT_TYPES = len(UnitType)


def _action_to_flat(action: Action, state: Optional[GameState] = None) -> int:
    """Encode an Action to a flat integer index.

    MOVE-stage ``SELECT_UNIT`` encodes ``move_pos`` at ``_MOVE_OFFSET + r*30+c``;
    SELECT-stage encodes the unit tile at ``3 + r*30+c``. When ``state`` is
    omitted, ``move_pos is not None`` implies MOVE encoding (for well-formed
    engine actions); otherwise SELECT encoding.
    """
    at = action.action_type

    if at == ActionType.END_TURN:
        return 0
    if at == ActionType.ACTIVATE_COP:
        return 1
    if at == ActionType.ACTIVATE_SCOP:
        return 2

    if at == ActionType.SELECT_UNIT:
        if state is not None:
            if state.action_stage == ActionStage.SELECT:
                r, c = action.unit_pos
                return 3 + r * _ENC_W + c
            if state.action_stage == ActionStage.MOVE:
                if action.move_pos is None:
                    return 0
                r, c = action.move_pos
                return _MOVE_OFFSET + r * _ENC_W + c
            return 0
        if action.move_pos is not None:
            r, c = action.move_pos
            return _MOVE_OFFSET + r * _ENC_W + c
        r, c = action.unit_pos
        return 3 + r * _ENC_W + c

    if at == ActionType.ATTACK:
        r, c = action.target_pos
        return _ATTACK_OFFSET + r * _ENC_W + c

    if at == ActionType.CAPTURE:
        return _CAPTURE_IDX

    if at == ActionType.WAIT:
        return _WAIT_IDX

    if at == ActionType.DIVE_HIDE:
        return _DIVE_HIDE_IDX

    if at == ActionType.LOAD:
        return _LOAD_IDX

    if at == ActionType.JOIN:
        return _JOIN_IDX

    if at == ActionType.UNLOAD:
        # Resolve the slot from the active state's selected transport so the
        # encoding stays stable across cargo permutations within one turn.
        if action.move_pos is None or action.target_pos is None:
            return 0
        dr = action.target_pos[0] - action.move_pos[0]
        dc = action.target_pos[1] - action.move_pos[1]
        try:
            direction = _UNLOAD_DIRS.index((dr, dc))
        except ValueError:
            return 0
        # Slot index is encoded as 0 by default; the env decoder picks the
        # first cargo whose drop matches direction + (optional) unit_type.
        slot = 0
        if action.unit_type is not None:
            slot = int(action.unit_type) & 1   # collapse to 0/1 just to vary the index
        return _UNLOAD_OFFSET + slot * 4 + direction

    if at == ActionType.BUILD:
        # One index per (factory tile, unit type); required for direct factory builds.
        if action.move_pos is None or action.unit_type is None:
            return 0
        br, bc = action.move_pos
        return (
            _BUILD_OFFSET
            + (br * _ENC_W + bc) * _N_UNIT_TYPES
            + int(action.unit_type)
        )

    if at == ActionType.REPAIR:
        if action.target_pos is None:
            return 0
        tr, tc = action.target_pos
        return _REPAIR_OFFSET + tr * _ENC_W + tc

    # Fallback (should not happen with a well-formed engine)
    return 0


def _flat_to_action(
    flat_idx: int,
    state: Optional[GameState],
    legal: list[Action] | None = None,
) -> Optional[Action]:
    """
    Decode a flat integer back to a legal Action for the current state.
    Returns None if the index does not correspond to any legal action.
    """
    if legal is None:
        legal = get_legal_actions(state)

    if (
        state is not None
        and state.action_stage == ActionStage.MOVE
        and _MOVE_OFFSET <= flat_idx < _MOVE_OFFSET + _ENC_W * _ENC_W
    ):
        r = (flat_idx - _MOVE_OFFSET) // _ENC_W
        c = (flat_idx - _MOVE_OFFSET) % _ENC_W
        u = state.selected_unit
        if u is not None:
            for a in legal:
                if (
                    a.action_type == ActionType.SELECT_UNIT
                    and a.move_pos == (r, c)
                    and a.unit_pos == u.pos
                ):
                    return a
        return None

    for a in legal:
        if _action_to_flat(a, state) == flat_idx:
            return a
    return None


def _action_label(action: Optional[Action]) -> Optional[dict]:
    """Compact JSON-safe description of an Action for replay frames."""
    if action is None:
        return None
    return {
        "type":       action.action_type.name,
        "unit_pos":   list(action.unit_pos)   if action.unit_pos   else None,
        "move_pos":   list(action.move_pos)   if action.move_pos   else None,
        "target_pos": list(action.target_pos) if action.target_pos else None,
        "unit_type":  action.unit_type.name   if action.unit_type is not None else None,
    }


def _get_action_mask(
    state: GameState,
    out: np.ndarray | None = None,
    legal: list[Action] | None = None,
) -> np.ndarray:
    """Return a bool mask over [0, ACTION_SPACE_SIZE) indicating legal actions.

    When ``out`` is a pre-allocated array of shape (ACTION_SPACE_SIZE,) and dtype bool,
    fills it in place and returns it (hot path). When ``out`` is None, allocates a
    fresh array (tools, server, legacy callers).
    """
    if out is None:
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    else:
        mask = out
        mask.fill(False)
    if legal is None:
        legal = get_legal_actions(state)
    for action in legal:
        idx = _action_to_flat(action, state)
        if 0 <= idx < ACTION_SPACE_SIZE:
            mask[idx] = True
    return mask


def _strip_non_infantry_builds(
    mask: np.ndarray,
    state: GameState,
    legal: list[Action] | None = None,
) -> None:
    """In-place: clear BUILD entries except ``INFANTRY`` (bootstrap curriculum)."""
    if legal is None:
        legal = get_legal_actions(state)
    for action in legal:
        if action.action_type == ActionType.BUILD and action.unit_type != UnitType.INFANTRY:
            idx = _action_to_flat(action, state)
            if 0 <= idx < ACTION_SPACE_SIZE:
                mask[idx] = False


def sample_training_matchup(
    sample_map_pool: list[dict],
    *,
    co_p0: int | None = None,
    co_p1: int | None = None,
    tier_name: str | None = None,
    curriculum_broad_prob: float = 0.0,
    rng: random.Random | None = None,
) -> tuple[int, str, int, int, str]:
    """
    One sample of ``(map_id, tier_name, p0_co, p1_co, map_name)``.

    Mirrors :meth:`AWBWEnv._sample_config` (same distribution as training
    for the given curriculum knobs). When ``rng`` is ``None``, uses the
    global ``random`` module like the env does on each ``reset``.
    """
    def _choice(seq: Sequence[Any]) -> Any:
        if rng is None:
            return random.choice(seq)
        return rng.choice(seq)

    def _randf() -> float:
        if rng is None:
            return random.random()
        return rng.random()

    def _full_random() -> tuple[int, str, int, int, str]:
        meta = _choice(sample_map_pool)
        enabled = [t for t in meta["tiers"] if t.get("enabled") and t.get("co_ids")]
        tier = _choice(enabled) if enabled else meta["tiers"][0]
        co_ids: list[int] = tier["co_ids"]
        p0_co = _choice(co_ids)
        p1_co = _choice(co_ids)
        return (
            meta["map_id"],
            tier["tier_name"],
            p0_co,
            p1_co,
            str(meta.get("name", "")),
        )

    def _pick_tier_for_fixed_cos(meta: dict) -> dict:
        enabled = [t for t in meta["tiers"] if t.get("enabled") and t.get("co_ids")]
        need = [c for c in (co_p0, co_p1) if c is not None]
        if not need:
            return _choice(enabled) if enabled else meta["tiers"][0]
        candidates = [t for t in enabled if all(c in t["co_ids"] for c in need)]
        if not candidates:
            raise ValueError(
                f"Map {meta.get('name', meta['map_id'])}: no enabled tier contains "
                f"CO id(s) {need}"
            )
        return _choice(candidates)

    if curriculum_broad_prob > 0.0 and _randf() < curriculum_broad_prob:
        return _full_random()

    meta = _choice(sample_map_pool)

    if tier_name is not None:
        tier = next(
            (t for t in meta["tiers"] if t.get("tier_name") == tier_name),
            None,
        )
        if tier is None or not tier.get("enabled") or not tier.get("co_ids"):
            raise ValueError(
                f"Map {meta.get('name', meta['map_id'])}: no enabled tier "
                f"{tier_name!r}"
            )
    elif co_p0 is not None or co_p1 is not None:
        tier = _pick_tier_for_fixed_cos(meta)
    else:
        enabled = [t for t in meta["tiers"] if t.get("enabled") and t.get("co_ids")]
        tier = _choice(enabled) if enabled else meta["tiers"][0]

    co_ids: list[int] = tier["co_ids"]
    tname = tier["tier_name"]

    if co_p0 is not None:
        if co_p0 not in co_ids:
            raise ValueError(
                f"CO {co_p0} not in tier {tname} for map "
                f"{meta.get('name', meta['map_id'])}"
            )
        p0_co = co_p0
    else:
        p0_co = _choice(co_ids)

    if co_p1 is not None:
        if co_p1 not in co_ids:
            raise ValueError(
                f"CO {co_p1} not in tier {tname} for map "
                f"{meta.get('name', meta['map_id'])}"
            )
        p1_co = co_p1
    else:
        p1_co = _choice(co_ids)

    return (
        meta["map_id"],
        tname,
        p0_co,
        p1_co,
        str(meta.get("name", "")),
    )


class AWBWEnv(gym.Env):
    """
    AWBW Gymnasium environment for single-agent (vs opponent) training.

    The environment always presents the perspective of player 0.
    After each player-0 action, if player 1's turn begins the environment
    automatically runs the opponent policy (or random) until it is player
    0's turn again.

    Parameters
    ----------
    map_pool:
        List of map metadata dicts (loaded from gl_map_pool.json). Sampled
        uniformly each episode. If None, loaded from POOL_PATH.
    opponent_policy:
        Callable(obs: dict, mask: np.ndarray) -> int  or None.
        When None, the opponent plays uniformly random legal actions.
    render_mode:
        "ansi" to print the board after each step, None to suppress.
    co_p0, co_p1:
        If set, fix that player's CO id for each episode (must appear in the
        sampled tier's ``co_ids``). Used with ``tier_name`` for narrow curriculum.
    tier_name:
        If set, use this tier (e.g. ``\"T3\"``) for the sampled map; raises if
        the map has no enabled tier with that name.
    curriculum_broad_prob:
        Each episode, with this probability ignore fixed CO/tier and sample the
        full random matchup (0 = always use fixed settings when provided).
    curriculum_tag:
        Optional label appended to game_log records for slicing runs by stage.
    max_env_steps:
        If set, end the episode with ``truncated=True`` once this many P0
        ``step`` calls have completed (independent of engine calendar /
        ``max_turns``). Useful for playoff / eval so games cannot run unbounded.
    max_turns:
        Engine day-count tiebreak threshold (passed to ``make_initial_state``).
        Defaults to ``MAX_TURNS`` from ``engine.game`` when ``None``.
    max_p1_microsteps:
        Cap on engine ``step`` calls while auto-playing player 1 in one P0
        ``env.step`` (prevents infinite loops if the opponent never hands back).
        If ``None`` and ``max_env_steps`` is set, defaults to
        ``max(500, max_env_steps * 30)``.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        map_pool: list[dict] | None = None,
        opponent_policy: Callable | None = None,
        render_mode: str | None = None,
        log_replay_frames: bool | None = None,
        co_p0: int | None = None,
        co_p1: int | None = None,
        tier_name: str | None = None,
        curriculum_broad_prob: float = 0.0,
        curriculum_tag: str | None = None,
        max_env_steps: int | None = None,
        max_p1_microsteps: int | None = None,
        max_turns: int | None = None,
        live_snapshot_path: str | Path | None = None,
        live_games_id: int | None = None,
        live_fallback_curriculum: bool = True,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.opponent_policy = opponent_policy
        self.co_p0 = co_p0
        self.co_p1 = co_p1
        self.tier_name = tier_name
        self.curriculum_broad_prob = float(curriculum_broad_prob)
        self.curriculum_tag = curriculum_tag
        self._max_env_steps: int | None = int(max_env_steps) if max_env_steps is not None else None
        if max_p1_microsteps is not None:
            self._max_p1_microsteps_cap: int | None = int(max_p1_microsteps)
        elif self._max_env_steps is not None:
            self._max_p1_microsteps_cap = max(500, self._max_env_steps * 30)
        else:
            self._max_p1_microsteps_cap = None
        self._max_turns: int | None = int(max_turns) if max_turns is not None else None
        # Explicit kwarg wins over env var; env var defaults off.
        if log_replay_frames is None:
            log_replay_frames = os.environ.get(LOG_REPLAY_FRAMES_ENV, "0") == "1"
        self.log_replay_frames: bool = bool(log_replay_frames)
        self._replay_frames: list[dict] = []

        if map_pool is None:
            with open(POOL_PATH) as f:
                map_pool = json.load(f)
        self.map_pool: list[dict] = map_pool
        # Global League "Std" maps only (see gl_map_pool.json "type"). Fog/HF stay in JSON for later.
        _std = [m for m in map_pool if m.get("type") == "std"]
        self._sample_map_pool: list[dict] = _std if _std else map_pool

        self._map_cache: dict[int, MapData] = {}

        # Gymnasium spaces
        self.observation_space = spaces.Dict(
            {
                "spatial": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS),
                    dtype=np.float32,
                ),
                "scalars": spaces.Box(
                    low=-1.0,
                    high=10.0,
                    shape=(N_SCALARS,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # Phase 1b (FPS campaign): reuse numpy buffers for mask + obs to cut allocator churn.
        # Golden tests: tests/test_env_buffer_reuse_golden.py, tests/test_phase1b_golden_buffers.py
        self._action_mask_buf = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        self._spatial_obs_buf = np.zeros(
            (GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS), dtype=np.float32
        )
        self._scalars_obs_buf = np.zeros((N_SCALARS,), dtype=np.float32)
        
        # --- Memory optimization additions ---
        # Buffer pool for reusable numpy arrays
        self._buffer_pool: deque[np.ndarray] = collections.deque(maxlen=10)
        
        # Pinned memory buffers for GPU transfer (also sets ``_reusable_tensors``).
        self._pinned_buffers: dict[str, np.ndarray] = {}
        self._init_pinned_memory()

        # Memory profile tracking
        self._memory_profile: dict[str, list] = {
            "allocations": [],
            "deallocations": []
        }
        
        # Read AWBW_PREALLOCATED_BUFFERS env var at init; defaults to True for performance.
        # Set to "0" to disable and test with fresh allocations (used by golden A<->B tests).
        self._use_preallocated_buffers: bool = os.environ.get(PREALLOCATED_BUFFERS_ENV, "1") == "1"

        # Phase 4: env-scoped legal-action cache. Populated lazily by
        # self._get_legal(); invalidated after every state.step in
        # _engine_step_with_belief and on reset(). Kills the 3x
        # get_legal_actions per P0 step (mask + strip + decode) and
        # the 2x per P1 policy microstep (mask + decode).
        self._legal_cache: list[Action] | None = None

        self.state: Optional[GameState] = None
        self._episode_info: dict[str, Any] = {}
        self._diag_lines_this_ep: int = 0
        self._episode_map_is_std: bool = True
        self._spirit_broken_kind: str | None = None
        self._spirit_debug_events: int = 0
        self._p1_truncated_mid_turn: bool = False
        # Filled each ``reset()``; used for ego-centric obs + learner-frame rewards.
        self._learner_seat: int = 0
        self._enemy_seat: int = 1
        # Incremented every ``reset()`` so opponent policies can detect new episodes.
        self._episode_id: int = 0

        # Tier 1: read the teacher-mix probability ONCE at construction so
        # SubprocVecEnv workers (each a separate process) inherit the value
        # set in train.py before spawn. Restart the run with a different
        # value to "decay" the mix — there is no online mutation hook.
        try:
            self._learner_greedy_mix = float(
                os.environ.get(LEARNER_GREEDY_MIX_ENV, "0") or 0.0
            )
        except ValueError:
            self._learner_greedy_mix = 0.0
        self._learner_greedy_mix = max(0.0, min(1.0, self._learner_greedy_mix))

        # Hoarding penalty: END_TURN with unspent funds above threshold (read once at spawn).
        try:
            self._hoard_funds_threshold = int(
                float(os.environ.get(HOARD_FUNDS_THRESHOLD_ENV, "25000") or 25000)
            )
        except ValueError:
            self._hoard_funds_threshold = 25_000
        try:
            self._hoard_penalty = float(os.environ.get(HOARD_PENALTY_ENV, "0") or 0.0)
        except ValueError:
            self._hoard_penalty = 0.0
        if self._hoard_penalty < 0.0:
            self._hoard_penalty = 0.0

        # Reward shaping mode + Φ coefficients — read once per env instance
        # so SubprocVecEnv workers inherit a stable value at spawn. Restart
        # the run to change. See plan rl_capture-combat_recalibration.
        mode = (os.environ.get(REWARD_SHAPING_ENV, "phi") or "phi").strip().lower()
        self._reward_shaping_mode: str = mode if mode in ("level", "phi") else "phi"

        prof_raw = (os.environ.get(PHI_PROFILE_ENV, "balanced") or "balanced").strip().lower()
        if prof_raw not in PHI_PROFILE_DEFAULTS:
            prof_raw = "balanced"
        self._phi_profile: str = prof_raw
        p_alpha, p_beta, p_kappa = PHI_PROFILE_DEFAULTS[prof_raw]

        def _read_float(env_name: str, default: float) -> float:
            try:
                return float(os.environ.get(env_name, "") or default)
            except ValueError:
                return default

        self._phi_alpha: float = _read_float(PHI_ALPHA_ENV, p_alpha)
        self._phi_beta: float = _read_float(PHI_BETA_ENV, p_beta)
        self._phi_kappa: float = _read_float(PHI_KAPPA_ENV, p_kappa)

        self._step_times: collections.deque[float] | None = (
            collections.deque(maxlen=100) if effective_track_per_worker_times() else None
        )
        # Live PPO: optional pickle path written by the training process / refresh
        # script; workers copy.deepcopy on each reset and skip curriculum / replay walk.
        self._live_snapshot_path: str | None = (
            str(live_snapshot_path) if live_snapshot_path else None
        )
        self._live_games_id: int | None = (
            int(live_games_id) if live_games_id is not None else None
        )
        self._live_fallback_curriculum: bool = bool(live_fallback_curriculum)

    def get_step_time_stats(self) -> dict[str, float]:
        """Return percentile summary of recent ``step()`` wall times, or ``{}`` if tracking is off.

        See :func:`effective_track_per_worker_times` (explicit ``=1``/``=0`` or
        defaults when ``AWBW_FPS_DIAG`` / ``AWBW_MACHINE_ID`` is set). Subproc
        workers inherit env at spawn.
        """
        if self._step_times is None:
            return {}
        if not self._step_times:
            return {
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "max": 0.0,
                "count": 0.0,
            }
        arr = np.fromiter(self._step_times, dtype=np.float64, count=len(self._step_times))
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()),
            "count": float(arr.size),
        }

    def reload_opponent_pool(self) -> Optional[int]:
        """Phase 10c: refresh the underlying opponent's checkpoint pool.

        Called via SubprocVecEnv.env_method between rollouts. Returns the
        new candidate count, or None if the opponent does not support
        refresh (e.g. random-policy opponents in tests).

        SB3's :meth:`~stable_baselines3.common.vec_env.VecEnv.env_method`
        uses :meth:`gymnasium.Wrapper.get_wrapper_attr`, so this method
        is reachable on the unwrapped env without attaching it to
        :class:`sb3_contrib.common.wrappers.ActionMasker`.
        """
        fn = getattr(self.opponent_policy, "reload_pool", None)
        if fn is None:
            return None
        try:
            return int(fn())
        except Exception:
            return None

    def _get_legal(self) -> list[Action]:
        """Return cached legal actions for self.state; populate on first call.

        Phase 4: cache invalidates after every state.step (see
        _engine_step_with_belief) and on reset(). Safe to call multiple
        times between steps — always returns the same list reference.
        """
        if self._legal_cache is None:
            self._legal_cache = get_legal_actions(self.state)
        return self._legal_cache

    def _invalidate_legal_cache(self) -> None:
        self._legal_cache = None

    # ── Map helpers ───────────────────────────────────────────────────────────

    def _load_map(self, map_id: int) -> MapData:
        if map_id not in self._map_cache:
            self._map_cache[map_id] = load_map(map_id, POOL_PATH, MAPS_DIR)
        return self._map_cache[map_id]
        
    # --- Memory Management Methods ---
    def _init_pinned_memory(self) -> None:
        """Create pinned memory buffers for GPU transfers."""
        # Create pinned buffers for observation tensors
        self._pinned_buffers = {
            "spatial": self._allocate_pinned_buffer((GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS), np.float32),
            "scalars": self._allocate_pinned_buffer((N_SCALARS,), np.float32),
            "action_mask": self._allocate_pinned_buffer((ACTION_SPACE_SIZE,), np.bool_)
        }
        
        # Initialize reusable tensors with pinned buffers
        self._reusable_tensors = {
            "spatial": self._pinned_buffers["spatial"],
            "scalars": self._pinned_buffers["scalars"],
            "action_mask": self._pinned_buffers["action_mask"]
        }
        
    def _allocate_pinned_buffer(self, shape: tuple, dtype: np.dtype) -> np.ndarray:
        """Allocate a pinned memory buffer for GPU transfers."""
        # Calculate size in bytes
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        
        # Allocate using ctypes to get page-locked memory
        import ctypes
        buffer = (ctypes.c_byte * size)()
        
        # Create numpy array view of the buffer
        arr = np.frombuffer(buffer, dtype=dtype).reshape(shape)
        arr.flags.writeable = True
        
        # Track memory allocation
        if hasattr(self, '_memory_profile'):
            self._memory_profile["allocations"].append({
                "shape": shape,
                "dtype": str(dtype),
                "size": size,
                "pinned": True
            })
        
        return arr
        
    def _get_buffer(self, shape: tuple, dtype: np.dtype) -> np.ndarray:
        """Get a buffer from the pool or create a new one."""
        # Try to find a matching buffer in the pool
        for buf in self._buffer_pool:
            if buf.shape == shape and buf.dtype == dtype:
                self._buffer_pool.remove(buf)
                return buf
                
        # Allocate new buffer if none found
        return np.zeros(shape, dtype=dtype)
        
    def _return_buffer(self, buf: np.ndarray) -> None:
        """Return a buffer to the pool for reuse."""
        # Clear the buffer before returning
        if np.issubdtype(buf.dtype, np.floating):
            buf.fill(0.0)
        elif np.issubdtype(buf.dtype, np.integer):
            buf.fill(0)
        elif np.issubdtype(buf.dtype, np.bool_):
            buf.fill(False)
            
        # Add to pool if there's space
        if len(self._buffer_pool) < self.BUFFER_POOL_SIZE:
            self._buffer_pool.append(buf)
        
    def _track_memory_allocation(self, size: int) -> None:
        """Track memory allocation for profiling."""
        if hasattr(self, '_memory_profile'):
            self._memory_profile["allocations"].append({
                "time": time.time(),
                "size": size,
                "stack": traceback.format_stack(limit=5) if 'traceback' in globals() else []
            })

    def _sample_config(self) -> tuple[int, str, int, int, str]:
        """
        Return (map_id, tier_name, p0_co_id, p1_co_id, map_name).

        With ``curriculum_broad_prob > 0``, sometimes delegates to full random
        sampling for mixture training. Fixed ``tier_name`` / ``co_p0`` / ``co_p1``
        implement narrow curriculum when broad sampling is not taken.

        Live / ladder workers (``live_snapshot_path`` set) never use broad: every
        reset that falls through to curriculum sampling must respect the pinned
        matchup so we do not replace ladder replays with random GL draws.
        """
        broad = (
            0.0
            if self._live_snapshot_path
            else float(self.curriculum_broad_prob)
        )
        return sample_training_matchup(
            self._sample_map_pool,
            co_p0=self.co_p0,
            co_p1=self.co_p1,
            tier_name=self.tier_name,
            curriculum_broad_prob=broad,
            rng=None,
        )

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self._episode_id = int(getattr(self, "_episode_id", 0)) + 1
        self._opening_book_log: dict[str, Any] = {}

        # Zero reused buffers before any encode / mask use (unused map cells must not
        # carry stale floats from a prior episode or env instance path).
        self._spatial_obs_buf.fill(0.0)
        self._scalars_obs_buf.fill(0.0)
        self._action_mask_buf.fill(False)

        from rl.live_snapshot import load_live_snapshot_dict  # local: optional dep in workers

        self.state = None
        from_live: bool = False
        lpath = self._live_snapshot_path
        if lpath and Path(lpath).is_file():
            try:
                raw = load_live_snapshot_dict(lpath)
                st = copy.deepcopy(raw["state"])
                # Subproc live workers set AWBW_LEARNER_SEAT from --live-learner-seats; prefer
                # that over a stale ``learner_seat`` in the pickle (e.g. export used 0 for all).
                _envls = (os.environ.get(LEARNER_SEAT_ENV) or "").strip()
                if _envls in ("0", "1"):
                    self._learner_seat = int(_envls) & 1
                elif raw.get("learner_seat") is not None:
                    self._learner_seat = int(raw["learner_seat"]) & 1
                else:
                    try:
                        self._learner_seat = int(os.environ.get(LEARNER_SEAT_ENV, "0"))
                    except ValueError:
                        self._learner_seat = 0
                if int(st.active_player) != int(self._learner_seat):
                    if not self._live_fallback_curriculum:
                        raise RuntimeError(
                            f"Live snapshot {lpath!s}: active_player={st.active_player} "
                            f"but learner_seat={self._learner_seat}. Refresh the pickle when it "
                            "is your turn, or enable live_fallback_curriculum."
                        )
                else:
                    self.state = st
                    from_live = True
            except (OSError, ValueError, KeyError) as exc:
                if not self._live_fallback_curriculum:
                    raise
                print(
                    f"[AWBWEnv] live snapshot load failed ({exc!r}); using curriculum."
                )
        if self._live_snapshot_path and not Path(self._live_snapshot_path).is_file():
            if not self._live_fallback_curriculum:
                raise FileNotFoundError(
                    f"Live snapshot path set but file missing: {self._live_snapshot_path}"
                )

        if self.state is None:
            if options is not None and options.get("map_id") is not None:
                map_id = int(options["map_id"])
                narrow = [m for m in self._sample_map_pool if m.get("map_id") == map_id]
                if not narrow:
                    raise ValueError(
                        f"map_id {map_id} not found in env map pool (sample_map_pool)"
                    )
                map_id, tier_name, p0_co, p1_co, map_name = sample_training_matchup(
                    narrow,
                    co_p0=self.co_p0,
                    co_p1=self.co_p1,
                    tier_name=self.tier_name,
                    curriculum_broad_prob=0.0,
                    rng=None,
                )
            else:
                map_id, tier_name, p0_co, p1_co, map_name = self._sample_config()
            map_data = self._load_map(map_id)

            _mk: dict = dict(starting_funds=0, tier_name=tier_name)
            if self._max_turns is not None:
                _mk["max_turns"] = self._max_turns
            if seed is not None:
                _mk["luck_seed"] = int(seed)
            rfm = getattr(map_data, "replay_first_mover", None)
            if rfm is not None:
                _mk["replay_first_mover"] = int(rfm)
            self.state = make_initial_state(map_data, p0_co, p1_co, **_mk)
        self._invalidate_legal_cache()
        # Who opens (engine seat 0 or 1) per make_initial_state / snapshot.
        self._opening_player = int(self.state.active_player)

        if from_live and self.state is not None:
            self._enemy_seat = 1 - self._learner_seat
        else:
            _bal = (os.environ.get(SEAT_BALANCE_ENV, "") or "").strip().lower()
            if _bal in ("1", "true", "yes", "on"):
                self._learner_seat = int(self.np_random.integers(0, 2))
            else:
                try:
                    self._learner_seat = int(os.environ.get(LEARNER_SEAT_ENV, "0"))
                except ValueError:
                    self._learner_seat = 0
            if self._learner_seat not in (0, 1):
                self._learner_seat = 0
            self._enemy_seat = 1 - self._learner_seat

        if from_live and self.state is not None:
            self._episode_info = {
                "map_id": int(self.state.map_data.map_id),
                "map_name": str(self.state.map_data.name),
                "tier": str(self.state.tier_name),
                "p0_co": int(self.state.co_states[0].co_id),
                "p1_co": int(self.state.co_states[1].co_id),
                "learner_seat": self._learner_seat,
                "episode_started_at": time.time(),
                "live": True,
            }
            if self._live_games_id is not None:
                self._episode_info["games_id"] = int(self._live_games_id)
        else:
            self._episode_info = {
                "map_id": map_id,
                "map_name": map_name,
                "tier": tier_name,
                "p0_co": p0_co,
                "p1_co": p1_co,
                "learner_seat": self._learner_seat,
                "episode_started_at": time.time(),  # Track episode start time
            }
        if self.curriculum_tag:
            self._episode_info["curriculum_tag"] = self.curriculum_tag

        self._diag_lines_this_ep = 0
        self._spirit_broken_kind = None
        self._spirit_debug_events = 0
        _mid = int(self._episode_info.get("map_id") or 0)
        _meta = next((m for m in self.map_pool if m.get("map_id") == _mid), {})
        self._episode_map_is_std = str(_meta.get("type", "")).lower() == "std"
        if self.state is not None:
            self.state.spirit_map_is_std = bool(self._episode_map_is_std)

        # Per-episode diagnostic counters (see _log_finished_game).
        # These are O(1) per step and help flag degenerate episodes —
        # especially mask/decode divergence (invalid_action_count) and
        # P1-loop heaviness (max_p1_microsteps).
        self._p0_env_steps: int = 0
        self._invalid_action_count: int = 0
        self._max_p1_microsteps: int = 0
        self._p1_truncated_mid_turn = False
        # Episode-end log metadata (set in step() immediately before _log_finished_game).
        self._log_episode_truncated: bool = False
        self._log_episode_truncation_reason: str | None = None
        # Step-cap tie-break: learner property lead (me − enemy); logged when >=2.
        self._log_tie_breaker_property_count: int | None = None
        # Phase 0a.2 (FPS campaign): per-episode wall-time split between the
        # P0 step path and the P1 microstep loop. Bounded by perf_counter()
        # at episode boundary; never read during the hot path. Surfaced as
        # wall_p0_s / wall_p1_s in game_log.jsonl.
        self._wall_p0_s: float = 0.0
        self._wall_p1_s: float = 0.0
        # Tier 1 (plan p0-capture-architecture-fix): per-episode count of
        # learner actions that were overridden by the capture-greedy teacher.
        # Surfaced in game_log.jsonl so we can verify the teacher is firing.
        self._learner_teacher_overrides: int = 0
        # Snapshot opponent reload count at episode start so we can
        # report per-episode reloads in the log record.
        self._opponent_reloads_at_start: int = int(
            getattr(self.opponent_policy, "reload_count", 0) or 0
        )
        self._first_learner_capture_step: int | None = None

        # HP belief overlays — one per seat. Seeded with the initial board so
        # predeployed units start at their visible bucket (not exact HP) for the
        # opposing observer. Engine keeps the exact 0-100 integer; these mirror
        # what a human AWBW player actually sees. See docs/hp_belief.md.
        self._beliefs: dict[int, BeliefState] = {
            0: BeliefState(observer=0),
            1: BeliefState(observer=1),
        }
        for b in self._beliefs.values():
            b.seed_from_state(self.state)

        # If the non-learner opens, autoplay that seat until the learner's clock.
        # Phase 0a.2: count opening autoplay as wall_p1_s (enemy-side wall bucket).
        # Live snapshots already reflect the site's clock after load_replay; do not
        # simulate the opponent to "catch up" (would desync from the real game).
        if (
            not from_live
            and not self.state.done
            and int(self.state.active_player) != self._learner_seat
        ):
            _t_open = time.perf_counter()
            if self.opponent_policy is not None:
                self._run_policy_opponent(0.0)
            else:
                self._run_random_opponent(0.0)
            self._wall_p1_s += time.perf_counter() - _t_open

        # Reset replay buffer and record the starting position.
        self._replay_frames = []
        self._capture_frame(action=None)

        return self._get_obs(observer=self._learner_seat), dict(self._episode_info)

    def step(
        self, action_idx: int
    ) -> tuple[dict, float, bool, bool, dict]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        _track_step_wall = self._step_times is not None
        if _track_step_wall:
            _t_wall_track = time.perf_counter()

        # Phase 0a.2: per-step wall accounting. _t_step_start brackets the entire
        # step; _p1_delta is added to _wall_p1_s and subtracted from the P0 share.
        # perf_counter() calls are at episode/section boundaries only, so they
        # cannot bias hot-path timing.
        _t_step_start = time.perf_counter()
        _p1_delta = 0.0

        self._p0_env_steps += 1

        # Tier 1 (plan p0-capture-architecture-fix): with probability
        # `learner_greedy_mix`, override the policy-sampled action with the
        # capture-greedy teacher used by the cold opponent. DAGGER-lite —
        # gives P0 the same scaffold P1 has had silently. The original
        # action_idx is not recorded back to the rollout buffer; PPO will
        # think the policy chose the teacher action. Bias is bounded by the
        # mask (only legal actions are sampled) and is the well-known cost
        # of behavior-policy mixing; decay the mix to 0 in a follow-up run
        # for the final on-policy phase.
        if self._learner_greedy_mix > 0.0 and random.random() < self._learner_greedy_mix:
            from rl.self_play import pick_capture_greedy_flat
            _mout = self._action_mask_buf if self._use_preallocated_buffers else None
            mask = _get_action_mask(self.state, out=_mout, legal=self._get_legal())
            action_idx = pick_capture_greedy_flat(self.state, mask)
            self._learner_teacher_overrides += 1

        # ── Decode & apply learner action (must be learner's clock) ───────────
        action = _flat_to_action(action_idx, self.state, legal=self._get_legal())
        if action is None:
            self._invalid_action_count += 1
            obs = self._get_obs(observer=self._learner_seat)
            # Phase 0a.2: invalid-action early return is still P0 work; account it.
            self._wall_p0_s += time.perf_counter() - _t_step_start
            if _track_step_wall:
                self._step_times.append(time.perf_counter() - _t_wall_track)
            return obs, -0.1, False, False, {"invalid_action": True}

        # Φ-shaping snapshot (plan rl_capture-combat_recalibration). Bracketed
        # around P0 action AND opponent micro-steps so a chip → opponent kills
        # capturer → cp resets sequence is captured as a single ΔΦ on this step.
        if self._reward_shaping_mode == "phi":
            phi_before = self._compute_phi(self.state)
            en_s = int(self._enemy_seat)
            pre_enemy_alive: dict[int, tuple[UnitType, int]] = {
                u.unit_id: (u.unit_type, u.hp)
                for u in self.state.units[en_s]
                if u.is_alive
            }
        else:
            phi_before = 0.0
            pre_enemy_alive = None

        acting = int(self.state.active_player)
        self.state, reward, done = self._engine_step_with_belief(action)
        reward = self._signed_engine_reward(reward, acting)
        if self.state is not None and self.state.done:
            reward = self._apply_phi_sparse_terminal_replacement(reward, acting)
        if pre_enemy_alive is not None and self.state is not None:
            reward += self._phi_enemy_kill_one_time_bonus(pre_enemy_alive)
        self._capture_frame(action=action)

        if (
            action.action_type == ActionType.CAPTURE
            and self._first_learner_capture_step is None
        ):
            self._first_learner_capture_step = self._p0_env_steps

        # Dense shaping (legacy "level" mode): me − enemy in learner frame.
        if self._reward_shaping_mode == "level" and not done:
            me = self._learner_seat
            en = self._enemy_seat
            p_me = self.state.count_properties(me)
            p_en = self.state.count_properties(en)
            diff = p_me - p_en
            if diff >= 0:
                reward += diff * 0.005
            else:
                reward += diff * 0.001

            v_me = sum(
                UNIT_STATS[u.unit_type].cost * u.hp / 100
                for u in self.state.units[me]
                if u.is_alive
            )
            v_en = sum(
                UNIT_STATS[u.unit_type].cost * u.hp / 100
                for u in self.state.units[en]
                if u.is_alive
            )
            reward += (v_me - v_en) * 2e-6

        # ── Auto-step non-learner seat ─────────────────────────────────────────
        if not done and int(self.state.active_player) != self._learner_seat:
            _t_p1 = time.perf_counter()
            if self.opponent_policy is not None:
                reward = self._run_policy_opponent(reward)
            else:
                reward = self._run_random_opponent(reward)
            _p1_delta = time.perf_counter() - _t_p1

        terminated = bool(self.state.done)
        p1_cap_trunc = bool(self._p1_truncated_mid_turn)
        self._p1_truncated_mid_turn = False
        env_step_trunc = (
            self._max_env_steps is not None
            and self._p0_env_steps >= self._max_env_steps
            and not self.state.done
        )
        truncated = p1_cap_trunc or env_step_trunc
        truncation_reason: str | None = None
        if truncated:
            if env_step_trunc:
                truncation_reason = "max_env_steps"
            elif p1_cap_trunc:
                truncation_reason = "max_p1_microsteps"

        # Apply Φ-delta AFTER opponent loop so the snapshot brackets the full
        # P0-action-to-next-P0-decision transition. Engine terminals and forced
        # truncations both use Φ:=0; otherwise a max-step slog can bank material /
        # property potential without actually converting the game.
        if self._reward_shaping_mode == "phi":
            phi_after = 0.0 if (terminated or truncated) else self._compute_phi(self.state)
            reward += phi_after - phi_before

        if truncated and not terminated:
            raw_tp = os.environ.get(TRUNCATION_PENALTY_ENV, "0").strip()
            if raw_tp:
                try:
                    reward -= float(raw_tp)
                except ValueError:
                    pass

        if (
            self._hoard_penalty > 0.0
            and self.state is not None
            and action.action_type == ActionType.END_TURN
            and int(self.state.funds[acting]) > int(self._hoard_funds_threshold)
        ):
            reward -= float(self._hoard_penalty)

        # Partial win at the P0 step cap: engine never emits ±1 without a terminal, so
        # credit half a win (+0.5 vs +1.0) when we hit max_env_steps with a solid
        # property lead in the learner frame.
        if (
            truncated
            and not terminated
            and truncation_reason == "max_env_steps"
            and self.state is not None
        ):
            me = int(self._learner_seat)
            en = int(self._enemy_seat)
            prop_lead = self.state.count_properties(me) - self.state.count_properties(
                en
            )
            if prop_lead >= 2:
                self._log_tie_breaker_property_count = int(prop_lead)
                reward += 0.5

        obs = self._get_obs(observer=self._learner_seat)
        info = {
            **self._episode_info,
            "turn": self.state.turn,
            "winner": self.state.winner,
            "truncated": truncated,
        }
        if (self.state.win_reason or "") == SPIRIT_BROKEN_REASON:
            info["spirit_broken"] = True
            _sk = getattr(self.state.spirit, "spirit_broken_kind", None)
            if _sk is not None:
                info["spirit_broken_kind"] = _sk

        if terminated:
            raw_inc = os.environ.get(INCOME_TERM_COEF_ENV, "0").strip()
            if raw_inc:
                try:
                    coef = float(raw_inc)
                    if coef != 0.0:
                        cap_lim = max(1, self.state.map_data.cap_limit)
                        inc_me = self.state.count_income_properties(self._learner_seat)
                        inc_en = self.state.count_income_properties(self._enemy_seat)
                        reward += coef * (inc_me - inc_en) / float(cap_lim)
                except ValueError:
                    pass

        if not self.state.done and not truncated:
            raw_tc = os.environ.get(TIME_COST_ENV, "0").strip()
            if raw_tc:
                try:
                    reward -= float(raw_tc)
                except ValueError:
                    pass

        if (
            self.state is not None
            and self.state.done
            and (self.state.win_reason or "") == SPIRIT_BROKEN_REASON
        ):
            w = int(self.state.winner) if self.state.winner is not None else -1
            if w in (0, 1):
                r_sparse = 1.0 if w == int(self._learner_seat) else -1.0
                if self._reward_shaping_mode == "phi":
                    reward = r_sparse - phi_before
                else:
                    reward = r_sparse

        # Phase 0a.2: finalize per-step wall split BEFORE logging so the
        # finished-game record reflects this step's contribution.
        _step_total = time.perf_counter() - _t_step_start
        self._wall_p1_s += _p1_delta
        self._wall_p0_s += max(0.0, _step_total - _p1_delta)

        # Log finished game (Phase A requirement) — natural end or forced truncation.
        if terminated or truncated:
            self._log_episode_truncated = truncated
            self._log_episode_truncation_reason = truncation_reason
            self._log_finished_game()

        if self.render_mode == "ansi":
            print(self.state.render_ascii())

        if _track_step_wall:
            self._step_times.append(time.perf_counter() - _t_wall_track)

        return obs, float(reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return valid-action bool mask. Required by MaskablePPO wrappers."""
        if self.state is None:
            if self._use_preallocated_buffers:
                self._action_mask_buf.fill(False)
                return self._action_mask_buf
            return np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        legal = self._get_legal()
        _mout = self._action_mask_buf if self._use_preallocated_buffers else None
        mask = _get_action_mask(self.state, out=_mout, legal=legal)
        flag = os.environ.get(BUILD_MASK_INFANTRY_ONLY_ENV, "").strip().lower()
        if flag in ("1", "true", "yes", "on"):
            _strip_non_infantry_builds(mask, self.state, legal=legal)
        return mask

    def render(self) -> str | None:
        if self.render_mode == "ansi" and self.state is not None:
            return self.state.render_ascii()
        return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _signed_engine_reward(self, r_engine: float, acting_seat: int) -> float:
        """Map engine reward (acting player's perspective) into learner coordinates."""
        if int(acting_seat) == int(self._learner_seat):
            return float(r_engine)
        return float(-r_engine)

    def _learner_frame_terminal_outcome(self, acting_seat: int) -> float:
        """Sparse win/lose/draw from a terminal step, learner frame; excludes capture shaping.

        Matches ``_check_win_conditions`` in acting-player coordinates, then
        :meth:`_signed_engine_reward` so it can be split from
        ``reward = signed(sparse + capture) = signed(sparse) + signed(capture)``.
        """
        st = self.state
        if st is None or not st.done or st.winner is None:
            return 0.0
        wi = int(st.winner)
        if wi == -1:
            s = 0.0
        else:
            s = 1.0 if wi == int(acting_seat) else -1.0
        return self._signed_engine_reward(s, acting_seat)

    def _apply_phi_sparse_terminal_replacement(
        self, reward: float, acting_seat: int
    ) -> float:
        """Φ mode: replace engine sparse terminal ±1.0/0, not stack on it.

        * ``max_turns_tie`` or (legacy) ``max_turns_draw`` → −0.1 (replaces 0.0)
        * ``max_turns_tiebreak`` win → +0.5 (replaces +1.0)
        * ``max_turns_tiebreak`` loss with **≥2** property deficit in learner
          frame (enemy has ≥2 more properties) → −0.5 (replaces −1.0);
          smaller deficit → keep −1.0

        Captured as ``rest = reward - ls`` and recombined (preserves per-step
        capture shaping in ``rest``). Non-day-cap terminations and ``level``
        mode pass through.
        """
        if self._reward_shaping_mode != "phi" or self.state is None or not self.state.done:
            return float(reward)
        wr = self.state.win_reason
        if wr not in ("max_turns_draw", "max_turns_tie", "max_turns_tiebreak"):
            return float(reward)
        ls = self._learner_frame_terminal_outcome(acting_seat)
        rest = float(reward) - ls
        if wr in ("max_turns_draw", "max_turns_tie"):
            return -0.1 + rest
        w = self.state.winner
        if w is None or int(w) == -1:
            return float(reward)
        me = int(self._learner_seat)
        en = int(self._enemy_seat)
        if int(w) == me:
            l_new = 0.5
        else:
            p_en = int(self.state.count_properties(en))
            p_me = int(self.state.count_properties(me))
            if p_en - p_me >= 2:
                l_new = -0.5
            else:
                l_new = -1.0
        return l_new + rest

    def _compute_phi(self, state: GameState) -> float:
        """Potential Φ(s) in the **learner** frame (me = ``_learner_seat``)."""
        me = int(self._learner_seat)
        en = int(self._enemy_seat)

        v_me = sum(
            UNIT_STATS[u.unit_type].cost * u.hp / 100
            for u in state.units[me]
            if u.is_alive
        )
        v_en = sum(
            UNIT_STATS[u.unit_type].cost * u.hp / 100
            for u in state.units[en]
            if u.is_alive
        )

        p_me = state.count_properties(me)
        p_en = state.count_properties(en)

        cap_me = 0.0
        cap_en = 0.0
        for prop in state.properties:
            cp = prop.capture_points
            if cp >= 20:
                continue
            chip = 1.0 - cp / 20.0
            owner = prop.owner
            if owner != me:
                cap_me += chip
            if owner != en:
                cap_en += chip

        return (
            self._phi_alpha * (v_me - v_en)
            + self._phi_beta * (p_me - p_en)
            + self._phi_kappa * (cap_me - cap_en)
        )

    def _phi_enemy_kill_one_time_bonus(
        self, pre_enemy_alive: dict[int, tuple[UnitType, int]]
    ) -> float:
        """Extra reward in Φ mode when enemy units are removed on the learner’s step.

        Per removed enemy (by ``unit_id`` from a pre-step snapshot), pays
        ``PHI_ENEMY_KILL_BONUS_FRAC × _phi_alpha × (cost × hp/100)`` using
        the unit’s pre-step ``hp`` — same per-unit *value* scale as
        :meth:`_compute_phi` army terms. Independent of the Φ potential
        difference (so it explicitly nudges toward lethal play).
        """
        if not pre_enemy_alive or self.state is None:
            return 0.0
        en = int(self._enemy_seat)
        post_ids = {u.unit_id for u in self.state.units[en] if u.is_alive}
        total = 0.0
        frac = float(PHI_ENEMY_KILL_BONUS_FRAC) * float(self._phi_alpha)
        for uid, (ut, hi) in pre_enemy_alive.items():
            if uid in post_ids:
                continue
            v = float(UNIT_STATS[ut].cost) * (float(hi) / 100.0)
            total += frac * v
        return float(total)

    def _get_obs(self, observer: int | None = None) -> dict:
        """Render observation from ``observer``'s perspective, honouring the
        HP belief overlay so enemy units leak only bucket + formula-narrowed
        interval information — never exact HP.
        
        When ``observer`` is omitted, use the current learner seat (after ``reset``).
        """
        if observer is None:
            observer = int(getattr(self, "_learner_seat", 0))
        belief = self._beliefs.get(observer) if hasattr(self, "_beliefs") else None
        
        # Use reusable tensors from buffer pool to avoid allocations
        spatial_buf = self._get_buffer((GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS), np.float32)
        scalars_buf = self._get_buffer((N_SCALARS,), np.float32)
        
        # Encode state into reusable buffers
        encode_state(
            self.state,
            observer=observer,
            belief=belief,
            out_spatial=spatial_buf,
            out_scalars=scalars_buf,
        )

        # Track memory usage
        self._track_memory_allocation(spatial_buf.nbytes + scalars_buf.nbytes)

        raw_copy = (os.environ.get(VECENV_OBS_COPY_ENV) or "1").strip().lower()
        if raw_copy not in ("0", "false", "no", "off"):
            return {
                "spatial": np.array(spatial_buf, copy=True, order="C"),
                "scalars": np.array(scalars_buf, copy=True, order="C"),
            }
        return {
            "spatial": spatial_buf,
            "scalars": scalars_buf,
        }

    # -- Belief bookkeeping ----------------------------------------------------
    def _belief_early_exit_enabled(self) -> bool:
        """When false (``AWBW_BELIEF_EARLY_EXIT_FULL=0``), every step runs the
        snapshot/diff/sync_own_units path — for A/B tests and parity checks."""
        v = os.environ.get("AWBW_BELIEF_EARLY_EXIT_FULL", "1").strip().lower()
        return v not in ("0", "false", "no", "off")

    def _snapshot_units(self) -> dict[int, dict]:
        """Per-unit snapshot keyed by ``unit_id`` before an engine step.

        Captures the minimal fields needed to diff the post-step state and
        emit belief updates: hp, pos, player. Unit objects themselves are
        mutated in place by the engine, so we copy the primitives.
        """
        snap: dict[int, dict] = {}
        for p in (0, 1):
            for u in self.state.units[p]:
                if not u.is_alive:
                    continue
                snap[u.unit_id] = {
                    "hp":     u.hp,
                    "pos":    u.pos,
                    "player": u.player,
                }
        return snap

    def _engine_step_with_belief(
        self, action: Action
    ) -> tuple[GameState, float, bool]:
        """Wrap ``state.step(action)`` and update both beliefs from the diff.

        Combat path (ATTACK): compute ``damage_range`` on pre-step state so
        the interval reflects what an observer could deduce from the bar
        change + the formula. Apply forward-damage range to defender; if
        attacker also took HP (counter), apply counter range using the
        post-forward defender as the counter-attacker.

        Non-combat paths (repair / day-start heal / CO-power HP shifts):
        diff HP per surviving unit; positive delta → ``on_heal``, negative
        delta → ``on_damage`` with ``(delta, delta)`` as the formula range.
        The observer can see the exact delta from bucket change + prior
        belief, so this is the tightest honest update.

        Always finishes with ``sync_own_units`` so own-unit beliefs are
        authoritative on the engine's exact HP.
        """
        turn_before = self.state.turn
        beliefs = list(self._beliefs.values())

        # Phase 5: belief diff early-exit. SELECT_UNIT in SELECT or MOVE stages
        # only advances state.action_stage and sets selected_*_pos -- no unit
        # list, HP, or position mutation. In ACTION stage, ``GameState.step``
        # has no branch for SELECT_UNIT (only SELECT/MOVE are handled) so the
        # step is a legal-mask no-op if ever emitted — same skip applies.
        # Skip snapshot/diff/enemy-event overhead; still sync_own_units above.
        # Invalidate the legal cache because the mask can change. ATTACK and all
        # real ACTION terminators take the full path below.
        _pre_stage = self.state.action_stage
        if (
            self._belief_early_exit_enabled()
            and action.action_type == ActionType.SELECT_UNIT
            and _pre_stage
            in (ActionStage.SELECT, ActionStage.MOVE, ActionStage.ACTION)
        ):
            # Fast path: no unit mutation from this action — skip snapshot/diff
            # and enemy belief events. Still sync own units (cheap): they must
            # stay hp_min=hp_max=engine HP every step (see belief parity tests).
            self.state, reward, done = self.state.step(action)
            self._invalidate_legal_cache()
            for b in beliefs:
                b.sync_own_units(self.state)
            self._maybe_spirit_calendar(turn_before)
            return self.state, reward, done

        # Pre-step snapshot + optional attack range.
        pre = self._snapshot_units()
        fwd_range: Optional[tuple[int, int]] = None
        attacker_id: Optional[int] = None
        defender_id: Optional[int] = None

        if action.action_type == ActionType.ATTACK and action.target_pos is not None:
            att = self.state.get_unit_at(*action.unit_pos) if action.unit_pos else None
            dfd = self.state.get_unit_at(*action.target_pos)
            if att is not None and dfd is not None:
                attacker_id = att.unit_id
                defender_id = dfd.unit_id
                move_r, move_c = (action.move_pos if action.move_pos else action.unit_pos)
                tr, tc = action.target_pos
                att_terrain = get_terrain(self.state.map_data.terrain[move_r][move_c])
                def_terrain = get_terrain(self.state.map_data.terrain[tr][tc])
                fwd_range = damage_range(
                    att, dfd,
                    att_terrain, def_terrain,
                    self.state.co_states[att.player],
                    self.state.co_states[dfd.player],
                )

        # Execute
        self.state, reward, done = self.state.step(action)
        self._invalidate_legal_cache()
        self._maybe_spirit_calendar(turn_before)

        post_by_id: dict[int, Any] = {}
        for p in (0, 1):
            for u in self.state.units[p]:
                if u.is_alive:
                    post_by_id[u.unit_id] = u

        # Kills (unit dropped from alive set).
        for uid in pre:
            if uid not in post_by_id:
                for b in beliefs:
                    b.on_unit_killed(uid)

        # New units (built / deployed / CO-power spawned).
        for uid, u in post_by_id.items():
            if uid not in pre:
                for b in beliefs:
                    b.on_unit_built(u)

        # Attack-specific: forward damage range → defender; counter range → attacker.
        if fwd_range is not None and defender_id is not None and attacker_id is not None:
            dfd_post = post_by_id.get(defender_id)
            att_post = post_by_id.get(attacker_id)
            if dfd_post is not None:
                for b in beliefs:
                    b.on_damage(dfd_post, fwd_range[0], fwd_range[1])
            pre_att_hp = pre[attacker_id]["hp"]
            if (
                att_post is not None
                and dfd_post is not None
                and att_post.hp < pre_att_hp
            ):
                # Counter-attack: post-forward defender rolls against attacker.
                att_terrain_ctr = get_terrain(
                    self.state.map_data.terrain[att_post.pos[0]][att_post.pos[1]]
                )
                def_terrain_ctr = get_terrain(
                    self.state.map_data.terrain[dfd_post.pos[0]][dfd_post.pos[1]]
                )
                ctr_range = damage_range(
                    dfd_post, att_post,
                    def_terrain_ctr, att_terrain_ctr,
                    self.state.co_states[dfd_post.player],
                    self.state.co_states[att_post.player],
                )
                if ctr_range is not None:
                    for b in beliefs:
                        b.on_damage(att_post, ctr_range[0], ctr_range[1])

        # Non-attack HP changes (repair, BB heal, day-start heal, CO power).
        # Skip attacker/defender here to avoid double-applying the combat path.
        for uid, pre_u in pre.items():
            post_u = post_by_id.get(uid)
            if post_u is None:
                continue
            if uid == attacker_id or uid == defender_id:
                continue
            delta = post_u.hp - pre_u["hp"]
            if delta > 0:
                for b in beliefs:
                    b.on_heal(post_u, delta, delta)
            elif delta < 0:
                for b in beliefs:
                    b.on_damage(post_u, -delta, -delta)

        # Own units: authoritative exact HP.
        for b in beliefs:
            b.sync_own_units(self.state)

        return self.state, reward, done

    def _maybe_spirit_calendar(self, turn_before: int) -> None:
        """Once per new calendar day (P0 to move): optional value-head JSONL diag only.

        Spirit **termination** runs in the engine on every ``END_TURN``; see
        ``engine/spirit_pressure.maybe_spirit_after_end_turn``.
        """
        from pathlib import Path as _P

        from rl import heuristic_termination as _ht

        st = self.state
        if st is None:
            return
        if st.done:
            if self._spirit_debug_events < 4:
                self._spirit_debug_events += 1
                # region agent log
                _agent_debug_log(
                    "H10",
                    "rl/env.py:AWBWEnv._maybe_spirit_calendar",
                    "spirit calendar skipped because engine already ended the game",
                    {
                        "episode_id": int(self._episode_id),
                        "turn": int(st.turn),
                        "map_id": self._episode_info.get("map_id"),
                        "winner": st.winner,
                        "win_reason": st.win_reason,
                        "income": [
                            int(st.count_income_properties(0)),
                            int(st.count_income_properties(1)),
                        ],
                        "alive_units": [
                            sum(1 for u in st.units[0] if u.is_alive),
                            sum(1 for u in st.units[1] if u.is_alive),
                        ],
                        "army_value": [
                            float(army_value_for_player(st, 0)),
                            float(army_value_for_player(st, 1)),
                        ],
                    },
                )
                # endregion
            return
        if st.turn <= turn_before or int(st.active_player) != 0:
            return
        if not _ht.diag_enabled_from_env():
            return
        model = _resolve_opponent_critic_model(self.opponent_policy)
        if model is None:
            if self._spirit_debug_events < 4:
                self._spirit_debug_events += 1
                # region agent log
                _agent_debug_log(
                    "H4",
                    "rl/env.py:AWBWEnv._maybe_spirit_calendar",
                    "heuristic value diag skipped because opponent critic model is missing",
                    {
                        "episode_id": int(self._episode_id),
                        "turn": int(st.turn),
                        "map_id": self._episode_info.get("map_id"),
                        "opponent_type": type(self.opponent_policy).__name__,
                        "opponent_mode": (
                            self.opponent_policy.mode()
                            if hasattr(self.opponent_policy, "mode")
                            else None
                        ),
                    },
                )
                # endregion
            return
        cfg = config_from_env()
        tier = str(self._episode_info.get("tier") or st.tier_name or "")
        tier_ok = not cfg.allowed_tiers or tier in cfg.allowed_tiers
        if self._spirit_debug_events < 4:
            self._spirit_debug_events += 1
            # region agent log
            _agent_debug_log(
                "H3,H4,H5",
                "rl/env.py:AWBWEnv._maybe_spirit_calendar",
                "spirit calendar gate reached before heuristic evaluation",
                {
                    "episode_id": int(self._episode_id),
                    "turn": int(st.turn),
                    "turn_before": int(turn_before),
                    "map_id": self._episode_info.get("map_id"),
                    "tier": tier,
                    "tier_ok": bool(tier_ok),
                    "is_std_map": bool(self._episode_map_is_std),
                    "spirit_enabled": bool(_ht.spirit_enabled_from_env()),
                    "model_present": model is not None,
                    "income": [
                        int(st.count_income_properties(0)),
                        int(st.count_income_properties(1)),
                    ],
                    "alive_units": [
                        sum(1 for u in st.units[0] if u.is_alive),
                        sum(1 for u in st.units[1] if u.is_alive),
                    ],
                    "army_value": [
                        float(army_value_for_player(st, 0)),
                        float(army_value_for_player(st, 1)),
                    ],
                    "streaks": {
                        "pressure": list(st.spirit.pressure_streak),
                        "resign": list(st.spirit.resign_streak),
                    },
                },
            )
            # endregion

        def _enc(s, observer: int):
            sp, sc = encode_state(s, observer=observer, belief=None)
            return {"spatial": sp, "scalars": sc}

        p = str(os.environ.get("AWBW_HEURISTIC_DIAG_LOG", "") or DEFAULT_DISAGREEMENT_LOG)
        _kind, nlines = run_calendar_day(
            st,
            model,
            cfg,
            _enc,
            is_std_map=bool(self._episode_map_is_std),
            map_tier_ok=tier_ok,
            episode_id=int(self._episode_id),
            map_id=self._episode_info.get("map_id"),
            learner_seat=int(self._learner_seat),
            log_path=_P(p),
            diag_line_budget=self._diag_lines_this_ep,
        )
        self._diag_lines_this_ep += int(nlines)
        if self._spirit_debug_events < 6:
            self._spirit_debug_events += 1
            # region agent log
            _agent_debug_log(
                "H5",
                "rl/env.py:AWBWEnv._maybe_spirit_calendar",
                "heuristic value diag result",
                {
                    "episode_id": int(self._episode_id),
                    "turn": int(st.turn),
                    "map_id": self._episode_info.get("map_id"),
                    "kind": _kind,
                    "winner": st.winner,
                    "win_reason": st.win_reason,
                    "diag_lines": int(nlines),
                    "streaks": {
                        "pressure": list(st.spirit.pressure_streak),
                        "resign": list(st.spirit.resign_streak),
                    },
                },
            )
            # endregion

    def _run_random_opponent(self, accumulated_reward: float) -> float:
        """Run the non-learner seat using uniform-random legal actions."""
        microsteps = 0
        cap = self._max_p1_microsteps_cap
        enemy = int(self._enemy_seat)
        while not self.state.done and int(self.state.active_player) == enemy:
            if cap is not None and microsteps >= cap:
                self._p1_truncated_mid_turn = True
                break
            legal = self._get_legal()
            if not legal:
                break
            action = random.choice(legal)
            acting = int(self.state.active_player)
            self.state, r_opp, done_opp = self._engine_step_with_belief(action)
            accumulated_reward += self._signed_engine_reward(r_opp, acting)
            self._capture_frame(action=action)
            microsteps += 1
        if microsteps > self._max_p1_microsteps:
            self._max_p1_microsteps = microsteps
        return accumulated_reward

    def _run_policy_opponent(self, accumulated_reward: float) -> float:
        """Run the non-learner seat using the provided opponent policy callable."""
        microsteps = 0
        cap = self._max_p1_microsteps_cap
        enemy = int(self._enemy_seat)
        while not self.state.done and int(self.state.active_player) == enemy:
            if cap is not None and microsteps >= cap:
                self._p1_truncated_mid_turn = True
                break
            needs_obs_fn = getattr(self.opponent_policy, "needs_observation", None)
            needs_obs = True if needs_obs_fn is None else bool(needs_obs_fn())
            obs = self._get_obs(observer=enemy) if needs_obs else None
            legal = self._get_legal()
            _mout = self._action_mask_buf if self._use_preallocated_buffers else None
            mask = _get_action_mask(self.state, out=_mout, legal=legal)
            try:
                opp_idx = int(self.opponent_policy(obs, mask))
            except Exception:
                opp_idx = -1

            action = _flat_to_action(opp_idx, self.state, legal=legal)
            if action is None:
                if not legal:
                    break
                action = random.choice(legal)

            acting = int(self.state.active_player)
            self.state, r_opp, done_opp = self._engine_step_with_belief(action)
            accumulated_reward += self._signed_engine_reward(r_opp, acting)
            self._capture_frame(action=action)
            microsteps += 1
        if microsteps > self._max_p1_microsteps:
            self._max_p1_microsteps = microsteps
        return accumulated_reward

    def _capture_frame(self, action: Optional[Action]) -> None:
        """Append a board snapshot to the replay buffer if frame logging is enabled.

        Frames omit the static `terrain` grid; it is stored once at the log-record
        root under `board` and merged back in by the viewer. The `action` field is
        a compact label derived from the `Action` that produced this state (or
        `None` for the initial frame captured at `reset`).
        """
        if not self.log_replay_frames or self.state is None:
            return
        self._replay_frames.append({
            "turn":          self.state.turn,
            "active_player": self.state.active_player,
            "action":        _action_label(action),
            "funds":         list(self.state.funds),
            "gold_spent":    list(self.state.gold_spent),
            "board":         board_dict(self.state, include_terrain=False),
        })

    def _log_finished_game(self):
        """
        Log finished game to logs/game_log.jsonl (Phase A requirement).

        Appends one line via ``_append_game_log_line`` so ``game_id`` and the full
        JSON blob are written under a single lock (threading, or SQLite
        ``BEGIN IMMEDIATE`` across SubprocVecEnv worker processes).
        """
        if self.state is None:
            return
        
        timestamp = time.time()
        timestamp_iso = log_timestamp_iso(timestamp)

        started_at = self._episode_info.get("episode_started_at") or timestamp
        episode_wall_s = max(0.0, timestamp - started_at)
        n_actions = len(self.state.game_log)
        p0_env_steps = getattr(self, "_p0_env_steps", 0)
        invalid_action_count = getattr(self, "_invalid_action_count", 0)
        max_p1_microsteps = getattr(self, "_max_p1_microsteps", 0)
        approx_engine_actions_per_p0_step = (
            n_actions / p0_env_steps if p0_env_steps > 0 else None
        )
        reloads_now = int(getattr(self.opponent_policy, "reload_count", 0) or 0)
        opponent_checkpoint_reload_count = max(
            0, reloads_now - getattr(self, "_opponent_reloads_at_start", 0)
        )

        # Phase 0a.2: per-episode P0/P1 wall split. Sums of perf_counter()
        # deltas captured around the step() body and the opponent loop.
        wall_p0_s = float(getattr(self, "_wall_p0_s", 0.0))
        wall_p1_s = float(getattr(self, "_wall_p1_s", 0.0))

        # Phase 0a.3: worker RSS at episode end. Lazy-imported so a stale
        # worker without psutil installed degrades to None instead of crashing
        # training. Cheap (~us) and only at episode boundary.
        worker_rss_mb: float | None
        try:
            import psutil  # type: ignore[import]
            worker_rss_mb = float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            worker_rss_mb = None

        # Phase 11e (schema 1.7): fraction of P0 unit positions sitting on
        # terrain with defense_stars >= 2 at episode end. Required signal for
        # the MCTS health gate. Empty unit list yields 0.0 (no divide-by-zero).
        # `defense_stars` lives on TerrainInfo as `defense` (engine/terrain.py:90);
        # there is no helper named `get_defense_stars` in this repo despite the
        # hint in the plan — we read the field directly via get_terrain().
        p0_units = self.state.units.get(0, [])
        defended_count = 0
        total_count = 0
        for u in p0_units:
            if not u.is_alive:
                continue
            r, c = u.pos
            tid = self.state.map_data.terrain[r][c]
            if get_terrain(tid).defense >= 2:
                defended_count += 1
            total_count += 1
        terrain_usage_p0 = defended_count / max(total_count, 1)

        # Outcome for the row: engine terminal, or synthetic property tiebreak when
        # env caps ended the episode before ``state.done`` (no engine winner).
        log_winner = self.state.winner
        log_win_reason = self.state.win_reason
        if (
            getattr(self, "_log_episode_truncated", False)
            and self.state.winner is None
            and getattr(self, "_log_episode_truncation_reason", None)
            in ("max_env_steps", "max_p1_microsteps")
        ):
            log_winner, log_win_reason = _synthetic_env_cap_property_tiebreak(
                self.state.count_properties(0),
                self.state.count_properties(1),
            )

        # Build comprehensive log record per LOGGING_PLAN.md Phase A (game_id added in _append_game_log_line)
        log_record = {
            # High-signal outcome fields first
            "property_count": [
                self.state.count_properties(0),
                self.state.count_properties(1),
            ],
            "income_property_count": [
                self.state.count_income_properties(0),
                self.state.count_income_properties(1),
            ],
            "first_learner_capture_step": getattr(
                self, "_first_learner_capture_step", None
            ),
            "first_p0_capture_p0_step": (
                getattr(self, "_first_learner_capture_step", None)
                if int(getattr(self, "_learner_seat", 0)) == 0
                else None
            ),
            "captures_completed_p0": sum(
                1
                for e in self.state.game_log
                if e.get("type") == "capture" and e.get("player") == 0
            ),
            "captures_completed_p1": sum(
                1
                for e in self.state.game_log
                if e.get("type") == "capture" and e.get("player") == 1
            ),
            "infantry_builds_p0": sum(
                1
                for e in self.state.game_log
                if e.get("type") == "build"
                and e.get("player") == 0
                and str(e.get("unit", "")).upper() == "INFANTRY"
            ),
            "turns": self.state.turn,
            "win_condition": log_win_reason,
            "losses_hp": self.state.losses_hp.copy(),

            # Outcome & matchup (CO names human-readable; IDs kept for analysis tools)
            "winner": log_winner,
            "p0_co": self.state.co_states[0].name,
            "p1_co": self.state.co_states[1].name,
            "p0_co_id": self._episode_info.get("p0_co"),
            "p1_co_id": self._episode_info.get("p1_co"),

            # Where it was played
            "map_name": self._episode_info.get("map_name") or self.state.map_data.name,
            "map_id": self._episode_info.get("map_id"),
            "tier": self._episode_info.get("tier"),

            # Economy
            "funds_end": self.state.funds.copy(),
            "gold_spent": self.state.gold_spent.copy(),

            # End state — army (same cost×hp/100 as Φ / spirit)
            "alive_unit_count": [
                sum(1 for u in self.state.units[0] if u.is_alive),
                sum(1 for u in self.state.units[1] if u.is_alive),
            ],
            "army_value": [
                float(army_value_for_player(self.state, 0)),
                float(army_value_for_player(self.state, 1)),
            ],

            # Losses
            "losses_units": self.state.losses_units.copy(),

            # Length & scale
            "n_actions": n_actions,

            # Training context
            "learner_seat": int(getattr(self, "_learner_seat", 0)),
            "agent_plays": int(getattr(self, "_learner_seat", 0)),
            "reward_mode": getattr(self, "_reward_shaping_mode", "phi"),
            "arch_version": (os.environ.get("AWBW_ARCH_VERSION", "wave2") or "wave2").strip(),
            "opponent_sampler": (
                "pfsp"
                if (os.environ.get("AWBW_PFSP", "") or "").strip().lower()
                in ("1", "true", "yes", "on")
                else "uniform"
            ),
            "opening_player": getattr(self, "_opening_player", None),
            "opponent_type": (
                self.opponent_policy.mode()
                if hasattr(self.opponent_policy, "mode")
                else ("policy" if self.opponent_policy is not None else "random")
            ),

            # Diagnostics — lightweight per-episode counters used to flag
            # degenerate / slow games. See LOGGING_PLAN and plan file.
            "episode_wall_s": episode_wall_s,
            "p0_env_steps": p0_env_steps,
            "invalid_action_count": invalid_action_count,
            "max_p1_microsteps": max_p1_microsteps,
            "approx_engine_actions_per_p0_step": approx_engine_actions_per_p0_step,
            "opponent_checkpoint_reload_count": opponent_checkpoint_reload_count,

            # Phase 0a (FPS campaign): per-episode wall split between the
            # P0 step path and the P1 microstep loop, plus worker RSS at
            # episode end. Used to size Phase 1a / Phase 6 ROI before any
            # hot-path code change. See .cursor/plans/train.py_fps_campaign_*.
            "wall_p0_s": wall_p0_s,
            "wall_p1_s": wall_p1_s,
            "worker_rss_mb": worker_rss_mb,

            # Phase 11e (FPS campaign / Phase 11d MCTS health gate): fraction
            # of P0 unit positions on defense>=2 terrain at episode end.
            # See definition above.
            "terrain_usage_p0": terrain_usage_p0,

            # Tier 1 (plan p0-capture-architecture-fix): visibility into
            # teacher-mix so we can verify it is firing and slice metrics by mix value.
            "learner_greedy_mix": float(getattr(self, "_learner_greedy_mix", 0.0)),
            "learner_teacher_overrides": int(getattr(self, "_learner_teacher_overrides", 0)),
            # Deprecated: env-side END_TURN gate removed; engine/action.py:_get_select_actions enforces the rule. Field retained for log schema continuity.
            "end_turn_gate_active": False,

            # Timestamps
            "timestamp": timestamp,
            "timestamp_iso": timestamp_iso,
            "episode_started_at": self._episode_info.get("episode_started_at"),

            # Schema version for future compatibility
            # 1.6: Phase 0a (FPS campaign) — added wall_p0_s / wall_p1_s / worker_rss_mb.
            # 1.7: Phase 10/11 prereq — machine_id (writer-stamped) + terrain_usage_p0.
            # 1.8: terminated / truncated / truncation_reason (forced episode caps).
            # 1.9: restart bundle — learner_seat, reward_mode, arch_version, opponent_sampler;
            #      agent_plays now mirrors learner_seat (was always 0).
            # 1.10: tie_breaker_property_count — learner property lead when step-cap partial win.
            # 1.11: winner / win_condition filled from property tiebreak when truncated
            #       and engine left winner unset (env_step_cap_* reasons).
            # 1.13: alive_unit_count, army_value at episode end.
            "terminated": bool(self.state.done),
            "truncated": bool(getattr(self, "_log_episode_truncated", False)),
            "truncation_reason": getattr(self, "_log_episode_truncation_reason", None),
            "tie_breaker_property_count": getattr(
                self, "_log_tie_breaker_property_count", None
            ),
            "log_schema_version": "1.13",
        }
        sk = getattr(self.state.spirit, "spirit_broken_kind", None)
        if sk is None:
            sk = getattr(self, "_spirit_broken_kind", None)
        if sk is not None:
            log_record["spirit_broken_kind"] = sk
        log_record.update(getattr(self, "_opening_book_log", {}))
        if self.curriculum_tag:
            log_record["curriculum_tag"] = self.curriculum_tag
        # region agent log
        try:
            from rl import heuristic_termination as _ht

            _model = _resolve_opponent_critic_model(self.opponent_policy)
            _cfg = config_from_env()
            _m = _ht.income_props_and_counts(self.state)
            _d_prop, _d_count, _p0v, _p1v = _ht.material_margins(_m, _cfg.value_margin)
            _p0 = _p1 = _r0 = _r1 = 0.0
            if _model is not None:
                def _debug_enc(s, observer: int):
                    sp, sc = encode_state(s, observer=observer, belief=None)
                    return {"spatial": sp, "scalars": sc}

                _p0, _p1, _r0, _r1 = _ht._predict_p_win_both(self.state, _model, _debug_enc, cfg=_cfg)
            _agent_debug_log(
                "H5,H9,H10",
                "rl/env.py:AWBWEnv._log_finished_game",
                "spirit final-state predicate evaluation",
                {
                    "episode_id": int(self._episode_id),
                    "map_id": log_record.get("map_id"),
                    "turns": log_record.get("turns"),
                    "win_condition": log_record.get("win_condition"),
                    "terminated": bool(self.state.done),
                    "truncated": bool(getattr(self, "_log_episode_truncated", False)),
                    "model_present": _model is not None,
                    "is_std_map": bool(self._episode_map_is_std),
                    "material": _m,
                    "d_prop": int(_d_prop),
                    "d_count": int(_d_count),
                    "p0_value_lead": bool(_p0v),
                    "p1_value_lead": bool(_p1v),
                    "p0_model_win": float(_p0),
                    "p1_model_win": float(_p1),
                    "v0_raw": float(_r0),
                    "v1_raw": float(_r1),
                    "snowball_holds": {
                        "p0": bool(_ht.snowball_holds(_m, 0, _p0, _cfg)),
                        "p1": bool(_ht.snowball_holds(_m, 1, _p1, _cfg)),
                    },
                    "resign_crush_holds": {
                        "p0": bool(_ht.resign_crush_holds(_m, 0, _p0, _cfg)),
                        "p1": bool(_ht.resign_crush_holds(_m, 1, _p1, _cfg)),
                    },
                    "thresholds": {
                        "p_snowball": float(_cfg.p_snowball),
                        "p_trailer_resign_max": float(_cfg.p_trailer_resign_max),
                        "value_margin": float(_cfg.value_margin),
                    },
                },
            )
        except Exception as exc:
            _agent_debug_log(
                "H9",
                "rl/env.py:AWBWEnv._log_finished_game",
                "spirit final-state predicate evaluation failed",
                {"error": repr(exc), "map_id": log_record.get("map_id")},
            )
        # endregion
        # region agent log
        _agent_debug_log(
            "H2,H3,H4,H5,H6,H7,H8",
            "rl/env.py:AWBWEnv._log_finished_game",
            "finished game summary before game_log write",
            {
                "episode_id": int(self._episode_id),
                "map_id": log_record.get("map_id"),
                "map_name": log_record.get("map_name"),
                "tier": log_record.get("tier"),
                "winner": log_record.get("winner"),
                "win_condition": log_record.get("win_condition"),
                "turns": log_record.get("turns"),
                "learner_seat": log_record.get("learner_seat"),
                "opening_player": log_record.get("opening_player"),
                "opponent_type": log_record.get("opponent_type"),
                "opening_book": {
                    "id_p0": log_record.get("opening_book_id_p0"),
                    "used_p0": log_record.get("opening_book_used_p0"),
                    "actions_p0": log_record.get("opening_book_actions_p0"),
                    "desync_p0": log_record.get("opening_book_desync_p0"),
                    "fallback_p0": log_record.get("opening_book_fallback_reason_p0"),
                    "episode_enabled_p0": log_record.get(
                        "opening_book_episode_enabled_p0"
                    ),
                    "suggest_calls_p0": log_record.get(
                        "opening_book_suggest_calls_p0"
                    ),
                    "id_p1": log_record.get("opening_book_id_p1"),
                    "used_p1": log_record.get("opening_book_used_p1"),
                    "actions_p1": log_record.get("opening_book_actions_p1"),
                    "desync_p1": log_record.get("opening_book_desync_p1"),
                    "fallback_p1": log_record.get("opening_book_fallback_reason_p1"),
                    "episode_enabled_p1": log_record.get(
                        "opening_book_episode_enabled_p1"
                    ),
                    "suggest_calls_p1": log_record.get(
                        "opening_book_suggest_calls_p1"
                    ),
                },
                "spirit": {
                    "kind": log_record.get("spirit_broken_kind"),
                    "env": os.environ.get("AWBW_SPIRIT_BROKEN"),
                    "debug_events": int(getattr(self, "_spirit_debug_events", 0)),
                },
                "material": {
                    "income_property_count": log_record.get("income_property_count"),
                    "alive_unit_count": log_record.get("alive_unit_count"),
                    "army_value": log_record.get("army_value"),
                },
            },
        )
        # endregion

        # Per-step replay data — optional, gated by log_replay_frames.
        # `board` holds the static terrain + dimensions once; each entry in
        # `frames` then carries only the dynamic `units`/`properties` payload
        # (see `_capture_frame`). The replay.js viewer merges the two.
        if self.log_replay_frames and self._replay_frames:
            log_record["board"] = {
                "height":  self.state.map_data.height,
                "width":   self.state.map_data.width,
                "terrain": self.state.map_data.terrain,
            }
            log_record["frames"] = self._replay_frames

        _append_game_log_line(log_record)

        # Slow / degenerate-episode alert: append a compact row to
        # logs/slow_games.jsonl when this game looks abnormal. Cheap
        # (one extra file write on rare events) and easy to tail.
        try:
            threshold = float(os.environ.get(SLOW_GAME_WALL_S_ENV, "60") or 60)
        except ValueError:
            threshold = 60.0
        is_slow_wall = threshold > 0 and episode_wall_s >= threshold
        has_invalids = invalid_action_count > 0
        if is_slow_wall or has_invalids:
            alert = {
                "timestamp_iso": timestamp_iso,
                "map_id": self._episode_info.get("map_id"),
                "tier": self._episode_info.get("tier"),
                "p0_co_id": self._episode_info.get("p0_co"),
                "p1_co_id": self._episode_info.get("p1_co"),
                "turns": self.state.turn,
                "n_actions": n_actions,
                "p0_env_steps": p0_env_steps,
                "invalid_action_count": invalid_action_count,
                "max_p1_microsteps": max_p1_microsteps,
                "approx_engine_actions_per_p0_step": approx_engine_actions_per_p0_step,
                "opponent_checkpoint_reload_count": opponent_checkpoint_reload_count,
                "episode_wall_s": episode_wall_s,
                "winner": log_winner,
                "win_condition": log_win_reason,
                "opponent_type": log_record["opponent_type"],
                "reasons": [
                    *(["slow_wall"] if is_slow_wall else []),
                    *(["invalid_actions"] if has_invalids else []),
                ],
            }
            try:
                SLOW_GAMES_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                with _log_lock:
                    with open(SLOW_GAMES_LOG_PATH, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(alert) + "\n")
            except Exception:
                # Diagnostics must never crash training.
                pass
