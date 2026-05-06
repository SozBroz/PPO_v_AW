"""
Gymnasium environment for AWBW self-play.

The environment wraps the AWBW game engine and exposes:
  - observation_space: Dict with ``spatial``, ``scalars``, ``candidate_features``, ``candidate_mask``
    (padded candidate table; action index is the row).
  - action_space: Discrete(MAX_CANDIDATES). Legacy 35k flat indices remain for opening-book /
    opponent helpers via internal flat masks only. Joint opening books bypass AWBW_CAPTURE_MOVE_GATE
    on the learner clock and force a single candidate row so greedy mix cannot override the line.
  - action_masks(): bool array over candidate rows (MaskablePPO)

By default the trained agent controls engine seat 0; the other seat is stepped
automatically (checkpoint opponent or random). With ``AWBW_SEAT_BALANCE=1`` the
learner seat is sampled 50/50 each episode (ego-centric obs + learner-frame Φ).
With ``AWBW_EGOCENTRIC_EPISODE_PROB`` in (0,1], seat is randomized that fraction
of episodes instead of using ``AWBW_LEARNER_SEAT`` (ignored when seat balance is on;
live snapshots keep CLI/env-pinned seats).

``AWBW_VECENV_OBS_COPY`` (default on): return C-contiguous observation copies from
``_get_obs`` so ``SubprocVecEnv`` workers do not pickle arrays that alias reused
buffers; helps Windows multiprocessing when ``n_envs`` is large. Set ``0`` to
restore zero-copy returns (slightly faster single-process / tests).
"""
from rl import _win_triton_warnings

_win_triton_warnings.apply()

import collections
import contextlib
import math
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
from engine.combat import damage_range, _colin_atk_rider, _kindle_atk_rider
from engine.belief import BeliefState

from rl.encoder import encode_state, GRID_SIZE, N_SPATIAL_CHANNELS, N_SCALARS
from rl.network import ACTION_SPACE_SIZE
from rl.candidate_actions import (
    MAX_CANDIDATES,
    CANDIDATE_FEATURE_DIM,
    CandidateAction,
    candidate_arrays,
)
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

@contextlib.contextmanager
def _without_capture_move_gate() -> Any:
    """Disable AWBW_CAPTURE_MOVE_GATE for the block (opening-book legals vs RL gate)."""
    key = "AWBW_CAPTURE_MOVE_GATE"
    prev = os.environ.get(key)
    try:
        os.environ.pop(key, None)
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


ROOT = Path(__file__).parent.parent
POOL_PATH = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


def coerce_map_id_filter(raw: int | Sequence[int] | None) -> list[int] | None:
    """``None`` = full GL pool; singleton or sequence → deduped ordered id list."""
    if raw is None:
        return None
    if isinstance(raw, int):
        n = int(raw)
        if n < 0:
            raise ValueError(f"map_id must be non-negative, got {n}")
        return [n]
    out: list[int] = []
    for x in raw:
        n = int(x)
        if n < 0:
            raise ValueError(f"map_id must be non-negative, got {n}")
        if n not in out:
            out.append(n)
    if not out:
        raise ValueError("map id list must be non-empty")
    return out


def coerce_co_selection(raw: int | Sequence[int] | None) -> list[int] | None:
    """``None`` = sample CO from tier each episode; else uniform choice from list per reset."""
    if raw is None:
        return None
    if isinstance(raw, int):
        return [int(raw)]
    out = [int(x) for x in raw]
    if not out:
        raise ValueError("CO list must be non-empty")
    return list(dict.fromkeys(out))


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

# Calendar milestones (``GameState.turn`` — 1-indexed day counter) where we snapshot
# ``neutral_income_properties`` into game_log rows (see AWBWEnv._log_finished_game).
GAME_LOG_NEUTRAL_INCOME_SNAPSHOT_DAYS = frozenset((7, 9, 11, 13, 15))

# When set to "1", each finished game in game_log.jsonl carries a `frames` array with one
# board snapshot per engine step (P0 + opponent substeps). Disabled by default because the
# payload grows roughly O(turns * actions_per_turn) per record.
LOG_REPLAY_FRAMES_ENV = "AWBW_LOG_REPLAY_FRAMES"
# Per seat, each curriculum episode: probability COP activation is disabled (SCOP
# unchanged). Live snapshots skip this. Override with ``cop_disable_per_seat_p`` kwarg.
COP_DISABLE_PER_SEAT_P_ENV = "AWBW_COP_DISABLE_PER_SEAT_P"

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
# End-turn "building punishment": subtract when learner ends turn with owned bases but didn't build.
BUILD_PUNISHMENT_ENV = "AWBW_BUILD_PUNISHMENT"
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
# Episode-level seat mixture (read once per env): with probability p each ``reset()``,
# sample learner seat {0,1}; else use ``AWBW_LEARNER_SEAT`` / default 0. Live snapshots
# ignore this (pinned seat). ``AWBW_SEAT_BALANCE`` still forces per-episode random seat.
EGOCENTRIC_EPISODE_PROB_ENV = "AWBW_EGOCENTRIC_EPISODE_PROB"
# Deprecated property-loss punishment env key. Training strips it and the env no
# longer reads it; property loss is already represented by Φ property/income deltas.
PHI_ENEMY_PROPERTY_CAPTURE_PENALTY_ENV = "AWBW_PHI_ENEMY_PROPERTY_CAPTURE_PENALTY"

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
# Experimental candidate-action policy interface: fixed padded legal-intent table.


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
#                       usual sparse ±1/0 with scaled outcomes — max_days_tie or
#                       max_days_draw → −0.1; max_days_tiebreak win → +0.5;
#                       tiebreak loss with
#                       ≥1 property deficit → −0.5 (else −1). See
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
PHI_KAPPA_ENV = "AWBW_PHI_KAPPA"   # capture-progress coefficient
PHI_GAMMA_ENV = "AWBW_PHI_GAMMA"   # income-saturation coefficient

# Optional contextual capture weighting for Φ. Default-off so old checkpoints /
# launch scripts keep exact reward semantics until the flag is enabled. When on,
# the κ capture-progress term remains potential-based, but progress on tactically
# important properties is weighted higher: contested neutrals, enemy income
# properties, production buildings, and HQs.
PHI_CONTEXTUAL_CAPTURE_ENV = "AWBW_PHI_CONTEXTUAL_CAPTURE"
PHI_CONTESTED_NEUTRAL_CAPTURE_MULT_ENV = "AWBW_PHI_CONTESTED_NEUTRAL_CAPTURE_MULT"
PHI_ENEMY_PROPERTY_CAPTURE_MULT_ENV = "AWBW_PHI_ENEMY_PROPERTY_CAPTURE_MULT"
PHI_PRODUCTION_CAPTURE_MULT_ENV = "AWBW_PHI_PRODUCTION_CAPTURE_MULT"
PHI_HQ_CAPTURE_MULT_ENV = "AWBW_PHI_HQ_CAPTURE_MULT"
PHI_CAPTURE_CONTEXT_RADIUS_ENV = "AWBW_PHI_CAPTURE_CONTEXT_RADIUS"
# Optional component-specific day/turn phase weighting inside capture Φ.
# This is not a global reward discount: safe neutral expansion gets opening
# urgency and late falloff, contested neutral gets mild falloff, and enemy /
# production / HQ capture progress does not fall off.
PHI_CAPTURE_PHASE_WEIGHTING_ENV = "AWBW_PHI_CAPTURE_PHASE_WEIGHTING"
PHI_SAFE_NEUTRAL_OPENING_MULT_ENV = "AWBW_PHI_SAFE_NEUTRAL_OPENING_MULT"
PHI_SAFE_NEUTRAL_EARLY_MID_MULT_ENV = "AWBW_PHI_SAFE_NEUTRAL_EARLY_MID_MULT"
PHI_SAFE_NEUTRAL_MID_MULT_ENV = "AWBW_PHI_SAFE_NEUTRAL_MID_MULT"
PHI_SAFE_NEUTRAL_LATE_MULT_ENV = "AWBW_PHI_SAFE_NEUTRAL_LATE_MULT"
PHI_SAFE_NEUTRAL_ENDGAME_MULT_ENV = "AWBW_PHI_SAFE_NEUTRAL_ENDGAME_MULT"
PHI_CONTESTED_NEUTRAL_OPENING_MULT_ENV = "AWBW_PHI_CONTESTED_NEUTRAL_OPENING_MULT"
PHI_CONTESTED_NEUTRAL_MID_MULT_ENV = "AWBW_PHI_CONTESTED_NEUTRAL_MID_MULT"
PHI_CONTESTED_NEUTRAL_LATE_MULT_ENV = "AWBW_PHI_CONTESTED_NEUTRAL_LATE_MULT"
PHI_CAPTURE_OPENING_END_DAY_ENV = "AWBW_PHI_CAPTURE_OPENING_END_DAY"
PHI_CAPTURE_EARLY_MID_END_DAY_ENV = "AWBW_PHI_CAPTURE_EARLY_MID_END_DAY"
PHI_CAPTURE_MID_END_DAY_ENV = "AWBW_PHI_CAPTURE_MID_END_DAY"
PHI_CAPTURE_LATE_END_DAY_ENV = "AWBW_PHI_CAPTURE_LATE_END_DAY"
PAIRWISE_ZERO_SUM_REWARD_ENV = "AWBW_PAIRWISE_ZERO_SUM_REWARD"
# (α, β, κ, γ) when in phi mode and a coefficient env is unset:
#   α (alpha, 2e-5): army value coefficient — unit cost × hp/100, scaled by 2e-5
#   β (beta,  0.05): property count coefficient — +0.05 per owned property
#   κ (kappa, 0.05): contested-cap coefficient — partial capture progress toward enemy tiles (cp < 20)
#   γ (gamma, 0.20): income-saturation coefficient — log-scale bonus for income property lead
PHI_PROFILE_DEFAULTS: dict[str, tuple[float, float, float, float]] = {
    "balanced": (2e-5, 0.05, 0.05, 0.05),
    "capture": (2e-5, 0.02, 0.25, 0.20),
}
# Φ: one-time bonus for removing an enemy unit on the learner’s engine step,
# in the same value units as the army line (α × cost × hp/100), scaled
# with ``_phi_alpha`` so it tracks profile/env overrides.
# Halved vs historical 0.3 (phi reshape): explicit combat kill chip only, not α.
PHI_ENEMY_KILL_BONUS_FRAC = 0.07
# Φ: CO power usage (per acting player; standard step maps to learner frame via
# ``_signed_engine_reward``). Attack bonus uses pre-step ``state`` so only
# attacks *after* COP/SCOP activation on that turn qualify (``cop_active`` /
# ``scop_active`` on the pre-action state).
# Base Φ bonuses before meter scaling (see ``PHI_VON_BOLT_SCOP_REF_THRESHOLD``).
PHI_COP_ACTIVATION_BONUS = 0.001
PHI_SCOP_ACTIVATION_BONUS = 0.003
# Legacy flat attack-under-power chip (replaced by stats-advantage shaping).
PHI_POWER_TURN_ATTACK_BONUS = 0.0001
# Von Bolt (co_id 30): 10★ SCOP, no COP — ``data/co_data.json``. Activation
# bonuses multiply by (this power's ``_cop_threshold`` / ``_scop_threshold``)
# ÷ this value so cheap meters (Adder) earn less than expensive ones (Hawke)
# on each COP/SCOP tap. Uses AWBW first-segment formula ``stars × 9000`` as
# the reference bar (same unit as ``COState._scop_threshold`` at power_uses=0
# for a 10★ SCOP).
PHI_VON_BOLT_SCOP_STARS = 10
PHI_VON_BOLT_SCOP_REF_THRESHOLD = float(PHI_VON_BOLT_SCOP_STARS * 9000)

# Engine calendar day-cap outcomes (:meth:`GameState._end_turn`). Canonical ``max_days_*``;
# ``max_turns_*`` kept for log/replay compatibility.
PHI_DAY_CAP_DRAWLIKE = frozenset(
    {"max_days_draw", "max_days_tie", "max_turns_draw", "max_turns_tie"}
)
PHI_DAY_CAP_TIEBREAK = frozenset({"max_days_tiebreak", "max_turns_tiebreak"})
PHI_DAY_CAP_REASONS = PHI_DAY_CAP_DRAWLIKE | PHI_DAY_CAP_TIEBREAK

# Designed Desires curriculum: per-seat owned property counts (includes comm
# towers) must reach (p0_target, p1_target) by calendar entry to this day.
PHI_DESIGNED_DESIRES_MAP_ID: int = 171596
PHI_DESIGNED_DESIRES_CHECKPOINTS: dict[int, tuple[int, int]] = {
    6: (7, 8),
    7: (9, 10),
    8: (12, 11),
    9: (14, 16),
    12: (17, 17),
} #SPECIFIC TO DESIGNED DESIRES


def _env_truthy(name: str) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    return raw in ("1", "true", "yes", "on")

# In-process: threads must not interleave JSONL lines. Cross-process: use SQLite (see _append_game_log_line).
_log_lock = Lock()
# When SESSION_GAME_COUNTER_DB_ENV is unset, count completed games in this process only (tests, ad-hoc env use).
_local_session_game_count = 0


def _synthetic_env_cap_property_tiebreak(p0_props: int, p1_props: int) -> tuple[int, str]:
    """
    P0 vs P1 ``count_properties`` — same rule as engine calendar max-turns
    (``engine.game.GameState`` end of ``_end_turn`` when ``turn > max_turns``):
    strictly more properties wins; equal counts draw. Used for ``game_log`` only
    when the episode is env-truncated (``max_env_steps`` / ``max_p1_microsteps``)
    and the engine never set ``winner`` / ``win_reason``.

    Return value is (engine_seat_winner, win_reason) with -1 for draw. Reasons
    ``env_step_cap_*`` are only for env truncation (``max_env_steps`` / ``max_p1_microsteps``),
    distinct from calendar ``max_days_tie`` / ``max_days_tiebreak`` (and legacy ``max_turns_*``).
    """
    d = int(p0_props) - int(p1_props)
    if d > 0:
        return 0, "env_step_cap_tiebreak"
    if d < 0:
        return 1, "env_step_cap_tiebreak"
    return -1, "env_step_cap_tie"


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


def _tier_order_num(tier_name: str) -> int:
    """Numeric ordering for GL tier rows (TL < T0 < T1 < …). Unknown names → -2."""
    if not str(tier_name).startswith("T"):
        return -2
    if tier_name == "TL":
        return -1
    rest = tier_name[1:]
    if rest.isdigit():
        return int(rest)
    return -2


def _resolve_named_tier_row(meta: dict, tier_name: str) -> dict:
    """
    Resolve a pinned tier name to a roster dict usable for curriculum sampling.

    Training follows operator intent: use the named tier row's ``co_ids`` even when
    the row is ``enabled: false`` on a map (GL ladder flags differ from sim roster).
    If that row has no IDs, merge IDs from enabled tiers at or above this tier's rank.
    """
    tiers = meta.get("tiers") or []
    row = next((t for t in tiers if t.get("tier_name") == tier_name), None)
    if row is None:
        raise ValueError(
            f"Map {meta.get('name', meta['map_id'])}: unknown tier {tier_name!r}"
        )
    raw_ids = list(row.get("co_ids") or [])
    if raw_ids:
        return row
    need = _tier_order_num(tier_name)
    enabled = [t for t in tiers if t.get("enabled") and t.get("co_ids")]
    at_or_above = [
        t for t in enabled if _tier_order_num(str(t.get("tier_name", ""))) >= need
    ]
    pool_src = at_or_above if at_or_above else enabled
    merged: list[int] = []
    seen: set[int] = set()
    for t in sorted(pool_src, key=lambda x: _tier_order_num(str(x.get("tier_name", "")))):
        for cid in t.get("co_ids") or []:
            ci = int(cid)
            if ci not in seen:
                seen.add(ci)
                merged.append(ci)
    if not merged:
        raise ValueError(
            f"Map {meta.get('name', meta['map_id'])}: tier {tier_name!r} has no CO roster "
            "and no fallback tiers provided CO ids"
        )
    out = dict(row)
    out["co_ids"] = merged
    return out


def _is_co_allowed_in_tier(meta: dict, tier_name: str, co_id: int) -> bool:
    """
    Check if a CO is allowed in a tier based on hierarchy.
    Lower tiers (numerically lower) can use COs from higher tiers.
    Based on request: T2 can use T2, T3, T4; T3 can use T3, T4; T4 can use T4.
    Tier order: TL, T0, T1, T2, T3, T4, T5 (T2 < T3 < T4)
    """
    # Parse tier number from tier_name (e.g., "T2" -> 2, "TL" -> -1, "T0" -> 0)
    if tier_name.startswith("T"):
        try:
            if tier_name[1:].isdigit():
                tier_num = int(tier_name[1:])
            elif tier_name == "TL":
                tier_num = -1  # TL is lowest
            else:
                tier_num = -2  # Unknown tier
        except ValueError:
            tier_num = -2
    else:
        tier_num = -2
    
    # Check all tiers in the map
    for tier in meta.get("tiers", []):
        tname = tier.get("tier_name", "")
        if tname.startswith("T"):
            try:
                if tname[1:].isdigit():
                    t_num = int(tname[1:])
                elif tname == "TL":
                    t_num = -1
                else:
                    t_num = -2
            except ValueError:
                t_num = -2
        else:
            t_num = -2
        
        # CO is allowed if it's in this tier AND this tier number >= requested tier number
        # (higher or equal tier number means it's a higher or equal tier)
        # Example: For T2 (tier_num=2), allow tiers with t_num >= 2 (T2, T3, T4, T5)
        if co_id in tier.get("co_ids", []) and t_num >= tier_num:
            return True
    
    return False


def sample_training_matchup(
    sample_map_pool: list[dict],
    *,
    co_p0: int | Sequence[int] | None = None,
    co_p1: int | Sequence[int] | None = None,
    tier_name: str | None = None,
    curriculum_broad_prob: float = 0.0,
    rng: random.Random | None = None,
) -> tuple[int, str, int, int, str]:
    """
    One sample of ``(map_id, tier_name, p0_co, p1_co, map_name)``.

    Mirrors :meth:`AWBWEnv._sample_config` (same distribution as training
    for the given curriculum knobs). When ``rng`` is ``None``, uses the
    global ``random`` module like the env does on each ``reset``.

    ``co_p0`` / ``co_p1`` may be a single CO id or a non-empty sequence; each
    reset draws uniformly from that set (after tier selection). ``None`` means
    draw uniformly from the resolved tier roster for that seat.

    Explicit CO ids are never rejected for hierarchy vs the pinned tier — the tier
    controls roster sampling for unset seats only; disabled tier rows still supply
    rosters when named explicitly.
    """
    c0 = coerce_co_selection(co_p0)
    c1 = coerce_co_selection(co_p1)

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
        need: list[int] = []
        if c0 is not None:
            need.extend(c0)
        if c1 is not None:
            need.extend(c1)
        need = list(dict.fromkeys(need))
        if not need:
            return _choice(enabled) if enabled else meta["tiers"][0]
        candidates = [t for t in enabled if all(c in t["co_ids"] for c in need)]
        if not candidates:
            # CO ids are explicit for this reset; use any enabled tier for metadata.
            candidates = enabled
        if not candidates:
            raise ValueError(
                f"Map {meta.get('name', meta['map_id'])}: no enabled tier contains "
                f"CO id(s) {need} and map has no enabled tiers"
            )
        return _choice(candidates)

    if curriculum_broad_prob > 0.0 and _randf() < curriculum_broad_prob:
        return _full_random()

    meta = _choice(sample_map_pool)

    if tier_name is not None:
        tier = _resolve_named_tier_row(meta, tier_name)
    elif c0 is not None or c1 is not None:
        tier = _pick_tier_for_fixed_cos(meta)
    else:
        enabled = [t for t in meta["tiers"] if t.get("enabled") and t.get("co_ids")]
        tier = _choice(enabled) if enabled else meta["tiers"][0]

    co_ids: list[int] = tier["co_ids"]
    tname = tier["tier_name"]

    if c0 is not None:
        p0_co = int(_choice(c0))
    else:
        p0_co = int(_choice(co_ids))

    if c1 is not None:
        p1_co = int(_choice(c1))
    else:
        p1_co = int(_choice(co_ids))

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
        If set, each episode samples that seat's CO uniformly from this list (or
        a fixed singleton). Explicit ids are not filtered against tier hierarchy.
        ``None`` means sample uniformly from the pinned tier roster (if ``tier_name``
        is set) or from the tier picked for this episode. Also accepts a bare ``int``
        for backward compatibility (coerced to a one-element list).
    tier_name:
        If set, use this tier name's roster on the sampled map (even when that GL row
        is ``enabled: false``); raises only if the tier name is unknown on that map.
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
    cop_disable_per_seat_p:
        Each curriculum ``reset``, independently per seat that has a COP,
        COP activation is disabled with this probability in ``[0, 1]`` (SCOP
        unchanged). ``None`` reads ``AWBW_COP_DISABLE_PER_SEAT_P``. Ignored when
        loading a live snapshot (site parity).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        map_pool: list[dict] | None = None,
        opponent_policy: Callable | None = None,
        render_mode: str | None = None,
        log_replay_frames: bool | None = None,
        co_p0: int | Sequence[int] | None = None,
        co_p1: int | Sequence[int] | None = None,
        tier_name: str | None = None,
        curriculum_broad_prob: float = 0.0,
        curriculum_tag: str | None = None,
        max_env_steps: int | None = None,
        max_p1_microsteps: int | None = None,
        max_turns: int | None = None,
        live_snapshot_path: str | Path | None = None,
        live_games_id: int | None = None,
        live_fallback_curriculum: bool = True,
        opening_book_path: str | Path | None = None,
        opening_book_seats: str = "both",
        opening_book_prob: float = 0.0,
        opening_book_strict_co: bool = False,
        opening_book_max_day: int | None = None,
        opening_book_seed: int = 0,
        opening_book_force_mask_for_learner: bool = True,
        opening_book_strike_release: bool = False,
        cop_disable_per_seat_p: float | None = None,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.opponent_policy = opponent_policy
        self.co_p0 = coerce_co_selection(co_p0)
        self.co_p1 = coerce_co_selection(co_p1)
        self.tier_name = tier_name
        self.curriculum_broad_prob = float(curriculum_broad_prob)
        self.curriculum_tag = curriculum_tag
        self._opening_book_path = str(opening_book_path) if opening_book_path else None
        self._opening_book_seats = str(opening_book_seats or "both")
        self._opening_book_prob = max(0.0, min(1.0, float(opening_book_prob)))
        self._opening_book_strict_co = bool(opening_book_strict_co)
        self._opening_book_max_day = (
            int(opening_book_max_day) if opening_book_max_day is not None else None
        )
        self._opening_book_seed = int(opening_book_seed)
        self._opening_book_force_mask_for_learner = bool(opening_book_force_mask_for_learner)
        self._opening_book_strike_release = bool(opening_book_strike_release)
        self._opening_book_manager: Any | None = None
        if cop_disable_per_seat_p is None:
            _cdp_raw = (os.environ.get(COP_DISABLE_PER_SEAT_P_ENV) or "").strip()
            try:
                self._cop_disable_per_seat_p = (
                    max(0.0, min(1.0, float(_cdp_raw))) if _cdp_raw else 0.0
                )
            except ValueError:
                self._cop_disable_per_seat_p = 0.0
        else:
            self._cop_disable_per_seat_p = max(
                0.0, min(1.0, float(cop_disable_per_seat_p))
            )
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

        # Gymnasium spaces. Candidate-action mode is the live action contract:
        # action index == padded candidate row. Legacy 35k flat remains for helpers.
        self._candidate_actions_enabled = True
        obs_spaces: dict[str, Any] = {
            "spatial": spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS),
                dtype=np.float32,
            ),
            "scalars": spaces.Box(
                low=-1.0,
                high=10.0,
                shape=(N_SCALARS,),
                dtype=np.float32,
            ),
            "candidate_features": spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(MAX_CANDIDATES, CANDIDATE_FEATURE_DIM),
                dtype=np.float32,
            ),
            "candidate_mask": spaces.Box(
                low=0,
                high=1,
                shape=(MAX_CANDIDATES,),
                dtype=np.int8,
            ),
        }
        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Discrete(MAX_CANDIDATES)

        # Phase 1b (FPS campaign): reuse numpy buffers for mask + obs to cut allocator churn.
        # Golden tests: tests/test_env_buffer_reuse_golden.py, tests/test_phase1b_golden_buffers.py
        self._action_mask_buf = np.zeros(self.action_space.n, dtype=bool)
        # Legacy flat-action scratch buffer (opening books / flat helpers).
        self._flat_action_mask_buf = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
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

        if self._opening_book_path:
            try:
                from rl.opening_book import TwoSidedOpeningBookManager

                self._opening_book_manager = TwoSidedOpeningBookManager(
                    self._opening_book_path,
                    seats=self._opening_book_seats,
                    prob=self._opening_book_prob,
                    strict_co=self._opening_book_strict_co,
                    max_day=self._opening_book_max_day,
                    seed=self._opening_book_seed,
                    strike_release=self._opening_book_strike_release,
                )
            except Exception as exc:
                print(
                    f"[AWBWEnv] opening book disabled: failed to load "
                    f"{self._opening_book_path!r}: {exc!r}"
                )
                self._opening_book_manager = None

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

        try:
            self._egocentric_episode_prob = float(
                os.environ.get(EGOCENTRIC_EPISODE_PROB_ENV, "0") or 0.0
            )
        except ValueError:
            self._egocentric_episode_prob = 0.0
        self._egocentric_episode_prob = max(0.0, min(1.0, self._egocentric_episode_prob))

        self._phi_enemy_property_capture_penalty = 0.0

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

        # Building punishment: END_TURN with owned bases but no build (read once at spawn).
        try:
            self._build_punishment = float(os.environ.get(BUILD_PUNISHMENT_ENV, "0") or 0.0)
        except ValueError:
            self._build_punishment = 0.0
        if self._build_punishment < 0.0:
            self._build_punishment = 0.0

        # Reward shaping mode + Φ coefficients — read once per env instance
        # so SubprocVecEnv workers inherit a stable value at spawn. Restart
        # the run to change. See plan rl_capture-combat_recalibration.
        mode = (os.environ.get(REWARD_SHAPING_ENV, "phi") or "phi").strip().lower()
        self._reward_shaping_mode: str = mode if mode in ("level", "phi") else "phi"

        prof_raw = (os.environ.get(PHI_PROFILE_ENV, "balanced") or "balanced").strip().lower()
        if prof_raw not in PHI_PROFILE_DEFAULTS:
            prof_raw = "balanced"
        self._phi_profile: str = prof_raw
        p_alpha, p_beta, p_kappa, p_gamma = PHI_PROFILE_DEFAULTS[prof_raw]

        def _read_float(env_name: str, default: float) -> float:
            try:
                return float(os.environ.get(env_name, "") or default)
            except ValueError:
                return default

        self._phi_alpha: float = _read_float(PHI_ALPHA_ENV, p_alpha) * 0.5
        self._phi_beta: float = _read_float(PHI_BETA_ENV, p_beta)
        self._phi_kappa: float = _read_float(PHI_KAPPA_ENV, p_kappa)
        self._phi_gamma: float = _read_float(PHI_GAMMA_ENV, p_gamma)
        self._phi_contextual_capture: bool = _env_truthy(PHI_CONTEXTUAL_CAPTURE_ENV)
        self._phi_contested_neutral_capture_mult: float = max(
            1.0, _read_float(PHI_CONTESTED_NEUTRAL_CAPTURE_MULT_ENV, 1.75)
        )
        self._phi_enemy_property_capture_mult: float = max(
            1.0, _read_float(PHI_ENEMY_PROPERTY_CAPTURE_MULT_ENV, 2.0)
        )
        self._phi_production_capture_mult: float = max(
            1.0, _read_float(PHI_PRODUCTION_CAPTURE_MULT_ENV, 3.0)
        )
        self._phi_hq_capture_mult: float = max(
            1.0, _read_float(PHI_HQ_CAPTURE_MULT_ENV, 5.0)
        )
        try:
            self._phi_capture_context_radius: int = max(
                0, int(float(os.environ.get(PHI_CAPTURE_CONTEXT_RADIUS_ENV, "3") or 3))
            )
        except ValueError:
            self._phi_capture_context_radius = 3

        # Optional phase weighting inside the capture-progress potential. This is
        # deliberately component-specific: safe neutral expansion gets opening
        # urgency and late falloff; contested neutral gets mild falloff;
        # enemy properties / production / HQ stay at 1.0 so late conversion is
        # never discounted away.
        self._phi_capture_phase_weighting: bool = _env_truthy(PHI_CAPTURE_PHASE_WEIGHTING_ENV)
        self._phi_safe_neutral_opening_mult: float = max(
            0.0, _read_float(PHI_SAFE_NEUTRAL_OPENING_MULT_ENV, 1.30)
        )
        self._phi_safe_neutral_early_mid_mult: float = max(
            0.0, _read_float(PHI_SAFE_NEUTRAL_EARLY_MID_MULT_ENV, 1.15)
        )
        self._phi_safe_neutral_mid_mult: float = max(
            0.0, _read_float(PHI_SAFE_NEUTRAL_MID_MULT_ENV, 1.00)
        )
        self._phi_safe_neutral_late_mult: float = max(
            0.0, _read_float(PHI_SAFE_NEUTRAL_LATE_MULT_ENV, 0.75)
        )
        self._phi_safe_neutral_endgame_mult: float = max(
            0.0, _read_float(PHI_SAFE_NEUTRAL_ENDGAME_MULT_ENV, 0.50)
        )
        self._phi_contested_neutral_opening_mult: float = max(
            0.0, _read_float(PHI_CONTESTED_NEUTRAL_OPENING_MULT_ENV, 1.25)
        )
        self._phi_contested_neutral_mid_mult: float = max(
            0.0, _read_float(PHI_CONTESTED_NEUTRAL_MID_MULT_ENV, 1.00)
        )
        self._phi_contested_neutral_late_mult: float = max(
            0.0, _read_float(PHI_CONTESTED_NEUTRAL_LATE_MULT_ENV, 0.90)
        )

        def _read_int(env_name: str, default: int) -> int:
            try:
                return int(float(os.environ.get(env_name, "") or default))
            except ValueError:
                return default

        self._phi_capture_opening_end_day: int = max(0, _read_int(PHI_CAPTURE_OPENING_END_DAY_ENV, 5))
        self._phi_capture_early_mid_end_day: int = max(
            self._phi_capture_opening_end_day,
            _read_int(PHI_CAPTURE_EARLY_MID_END_DAY_ENV, 8),
        )
        self._phi_capture_mid_end_day: int = max(
            self._phi_capture_early_mid_end_day,
            _read_int(PHI_CAPTURE_MID_END_DAY_ENV, 12),
        )
        self._phi_capture_late_end_day: int = max(
            self._phi_capture_mid_end_day,
            _read_int(PHI_CAPTURE_LATE_END_DAY_ENV, 18),
        )
        # Opt-in only: standard learner-frame ``step()`` can expose and return a
        # pairwise-centered competitive reward. The active-seat
        # ``step_active_seat_once()`` path is deliberately left unchanged so
        # ego-centric / dual-gradient self-play keeps its existing contract.
        self._pairwise_zero_sum_reward: bool = _env_truthy(PAIRWISE_ZERO_SUM_REWARD_ENV)

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
        # Async IMPALA dual-gradient: set each episode (``mirror`` vs ``hist``).
        self._async_rollout_mode: str | None = None

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

    def set_async_rollout_mode(self, mode: str | None) -> None:
        """Tag the current episode for ``game_log.jsonl`` (async dual-gradient only).

        Use ``mirror`` (both seats, shared policy) or ``hist`` (vs historical checkpoint).
        Cleared at the start of each ``reset()``.
        """
        if mode is not None and mode not in ("mirror", "hist"):
            raise ValueError(
                f"async_rollout_mode must be 'mirror', 'hist', or None; got {mode!r}"
            )
        self._async_rollout_mode = mode

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

    def _peek_learner_book_flat_safe_ungated(self) -> int | None:
        """Next booked flat index if legal under MOVE gate *off* (captures gate/book clash)."""
        mgr = getattr(self, "_opening_book_manager", None)
        if (
            mgr is None
            or self.state is None
            or int(self.state.active_player) != int(self._learner_seat)
        ):
            return None
        _mout = self._flat_action_mask_buf if self._use_preallocated_buffers else None
        with _without_capture_move_gate():
            legal = get_legal_actions(self.state)
            m = _get_action_mask(self.state, out=_mout, legal=legal)
        return mgr.peek_book_candidate_flat_safe(
            seat=int(self._learner_seat),
            calendar_turn=int(getattr(self.state, "turn", 0) or 0),
            action_mask=m,
        )

    def _build_candidate_policy_tensors(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """Padded candidate features/mask/list; joint opening book bypasses capture MOVE gate."""
        if self.state is None:
            zf = np.zeros((MAX_CANDIDATES, CANDIDATE_FEATURE_DIM), dtype=np.float32)
            zm = np.zeros((MAX_CANDIDATES,), dtype=bool)
            return zf, zm, []
        use_ungated = (
            int(self.state.active_player) == int(self._learner_seat)
            and self._opening_book_force_mask_for_learner
        )
        expected_flat: int | None = None
        if use_ungated:
            expected_flat = self._peek_learner_book_flat_safe_ungated()
        if use_ungated and expected_flat is not None:
            with _without_capture_move_gate():
                feats, mask, cands = candidate_arrays(self.state, max_candidates=MAX_CANDIDATES)
        else:
            feats, mask, cands = candidate_arrays(self.state, max_candidates=MAX_CANDIDATES)
        if use_ungated and expected_flat is not None:
            hit = -1
            exp = int(expected_flat)
            for i in range(min(len(cands), MAX_CANDIDATES)):
                if not mask[i]:
                    continue
                if _action_to_flat(cands[i].terminal_action, self.state) == exp:
                    hit = i
                    break
            if hit >= 0:
                nmask = np.zeros_like(mask)
                nmask[hit] = True
                mask = nmask
            else:
                # Book line expected a flat that did not appear in ungated enumeration
                # (truncation, _action_to_flat drift, etc.). Full ungated mask vs gated
                # step gate caused IllegalActionError; fall back to gated candidates.
                feats, mask, cands = candidate_arrays(
                    self.state, max_candidates=MAX_CANDIDATES
                )
        return feats, mask, cands

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
        self._async_rollout_mode = None
        self._opening_book_log: dict[str, Any] = {}

        # Zero reused buffers before any encode / mask use (unused map cells must not
        # carry stale floats from a prior episode or env instance path).
        self._spatial_obs_buf.fill(0.0)
        self._scalars_obs_buf.fill(0.0)
        self._action_mask_buf.fill(False)
        self._flat_action_mask_buf.fill(False)

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
                    # Live play must match site power rules; never carry curriculum COP-disable.
                    self.state.co_states[0].cop_activation_disabled = False
                    self.state.co_states[1].cop_activation_disabled = False
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
                _mk["max_days"] = self._max_turns
                _mk["max_turns"] = self._max_turns
            if seed is not None:
                _mk["luck_seed"] = int(seed)
            rfm = getattr(map_data, "replay_first_mover", None)
            if rfm is not None:
                _mk["replay_first_mover"] = int(rfm)
            self.state = make_initial_state(map_data, p0_co, p1_co, **_mk)
        p0_cop_activation_disabled = False
        p1_cop_activation_disabled = False
        if (
            self.state is not None
            and not from_live
            and self._cop_disable_per_seat_p > 0.0
        ):
            _pcd = float(self._cop_disable_per_seat_p)
            for _seat in (0, 1):
                _co = self.state.co_states[_seat]
                if _co.cop_stars is None or _co._data.get("cop") is None:
                    continue
                if float(self.np_random.random()) < _pcd:
                    _co.cop_activation_disabled = True
                    if _seat == 0:
                        p0_cop_activation_disabled = True
                    else:
                        p1_cop_activation_disabled = True
        self._invalidate_legal_cache()
        # Who opens (engine seat 0 or 1) per make_initial_state / snapshot.
        self._opening_player = int(self.state.active_player)

        if from_live and self.state is not None:
            self._enemy_seat = 1 - self._learner_seat
        else:
            _bal = (os.environ.get(SEAT_BALANCE_ENV, "") or "").strip().lower()
            if _bal in ("1", "true", "yes", "on"):
                self._learner_seat = int(self.np_random.integers(0, 2))
            elif self._egocentric_episode_prob > 0.0 and float(
                self.np_random.random()
            ) < self._egocentric_episode_prob:
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
                "p0_cop_activation_disabled": p0_cop_activation_disabled,
                "p1_cop_activation_disabled": p1_cop_activation_disabled,
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
                "p0_cop_activation_disabled": p0_cop_activation_disabled,
                "p1_cop_activation_disabled": p1_cop_activation_disabled,
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
        # Step-cap tie-break: learner property lead (me − enemy); logged when >=1.
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
        # Deprecated Φ property-loss penalty counter retained for log schema
        # continuity; the penalty itself is disabled. 
        self._phi_enemy_property_captures_ep: int = 0
        # Episode COP/SCOP activation counts by engine seat (for game_log.jsonl).
        self._episode_cop_by_seat: list[int] = [0, 0]
        self._episode_scop_by_seat: list[int] = [0, 0]
        # Episode reward component accumulators (for game_log breakdown).
        # Each tracks cumulative reward received by seat [p0, p1].
        self._episode_reward_army_cumulative: list[float] = [0.0, 0.0]
        self._episode_reward_property_cumulative: list[float] = [0.0, 0.0]
        self._episode_reward_capture_cumulative: list[float] = [0.0, 0.0]
        self._episode_reward_income_cumulative: list[float] = [0.0, 0.0]
        self._episode_reward_kill_bonus_cumulative: list[float] = [0.0, 0.0]
        # SCOP/COP tap shaping (COP Φ bonus is 0) vs strike stats-advantage chip.
        self._episode_reward_power_activation_cumulative: list[float] = [0.0, 0.0]
        self._episode_reward_attack_stats_cumulative: list[float] = [0.0, 0.0]
        self._episode_reward_power_cumulative: list[float] = [0.0, 0.0]
        self._episode_reward_sparse_cumulative: list[float] = [0.0, 0.0]
        self._episode_reward_designed_desires_cumulative: list[float] = [0.0, 0.0]
        self._episode_reward_capture_interrupt_cumulative: list[float] = [0.0, 0.0]
        self._episode_reward_enemy_property_loss_cumulative: list[float] = [0.0, 0.0]
        self._episode_reward_phi_potential_delta_cumulative: list[float] = [0.0, 0.0]
        self._episode_reward_build_punishment_cumulative: list[float] = [0.0, 0.0]
        # Snapshot opponent reload count at episode start so we can
        # report per-episode reloads in the log record.
        self._opponent_reloads_at_start: int = int(
            getattr(self.opponent_policy, "reload_count", 0) or 0
        )
        self._first_learner_capture_step: int | None = None

        # Neutral income property count at specific calendar days (turn == day).
        self._neutral_income_snapshot_by_day: dict[int, int] = {}

        self._begin_opening_book_episode()

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

        self._maybe_record_neutral_income_snapshot_days()

        self._designed_desires_checkpoint_misses = 0
        self._dd_prev_calendar_turn_at_step_end = int(
            getattr(self.state, "turn", 0) or 0
        )

        return self._get_obs(observer=self._learner_seat), dict(self._episode_info)

    def _decode_candidate_action(
        self, action_idx: int
    ) -> CandidateAction | None:
        """Map padded candidate-mode row index to a :class:`CandidateAction`."""
        if self.state is None:
            return None
        feats, mask, cands = self._build_candidate_policy_tensors()
        self._candidate_features_cache = feats
        self._candidate_mask_cache = mask
        self._candidate_list_cache = cands
        i = int(action_idx)
        if i < 0 or i >= MAX_CANDIDATES or not bool(mask[i]):
            return None
        return cands[i]

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
        #
        # Opening-book invariant: never apply this teacher while the joint book
        # still expects the learner's next flat — doing so executes a move other
        # than the committed book index (cursor advances from policy-vs-book
        # commit logic vs execution mismatch), drifting state until the book
        # line hits ``action_not_legal`` mid-opening (often before the intended
        # calendar horizon).
        skip_capture_teacher = self._peek_learner_book_flat_safe_ungated() is not None

        if (
            not skip_capture_teacher
            and self._learner_greedy_mix > 0.0
            and random.random() < self._learner_greedy_mix
        ):
            from rl.self_play import pick_capture_greedy_candidate_idx

            _m = self.action_masks()
            cands = getattr(self, "_candidate_list_cache", None) or []
            action_idx = pick_capture_greedy_candidate_idx(self.state, _m, cands)
            self._learner_teacher_overrides += 1

        # Env-level opening book for the learner must be PPO-consistent: the
        # action mask is forced to the book action in action_masks(), so here
        # we advance the cursor based on the **executed** flat index after any
        # capture-greedy override above (skipped while a book line is active).
        # Opening-book flat-action commits use the terminal engine action's flat id.

        # ── Decode & apply learner action (must be learner's clock) ───────────
        candidate = self._decode_candidate_action(int(action_idx))
        action = candidate.terminal_action if candidate is not None else None
        if action is None:
            self._invalid_action_count += 1
            obs = self._get_obs(observer=self._learner_seat)
            reward = 0
            info = {
                "invalid_action": True,
                "reward_components": {},
                "reward": float(reward),
                "reward_contract": self._reward_contract_info(
                    competitive_learner=0.0,
                    common_learner=0.0,
                    seat_local_learner=float(reward),
                    final_reward=float(reward),
                ),
                "property_pressure": self._property_pressure_snapshot(),
            }
            self._wall_p0_s += time.perf_counter() - _t_step_start
            if _track_step_wall:
                self._step_times.append(time.perf_counter() - _t_wall_track)
            return obs, float(reward), False, False, info

        self._maybe_commit_learner_opening_book_action(
            _action_to_flat(action, self.state)
        )

        # Φ-shaping snapshot (plan rl_capture-combat_recalibration). Bracketed
        # around P0 action AND opponent micro-steps so a chip → opponent kills
        # capturer → cp resets sequence is captured as a single ΔΦ on this step.
        if self._reward_shaping_mode == "phi":
            phi_before = self._compute_phi(self.state)
            # Store Φ component breakdown for later delta calculation
            self._phi_before_components = self._compute_phi_components_for_seat(self.state, int(self._learner_seat))
            # Snapshot learner's capture progress before learner action (for interrupt detection)
            pre_action_capture_progress = self._get_learner_capture_progress()
            en_s = int(self._enemy_seat)
            pre_enemy_alive: dict[int, tuple[UnitType, int]] = {
                u.unit_id: (u.unit_type, u.hp)
                for u in self.state.units[en_s]
                if u.is_alive
            }
            if self._phi_enemy_property_capture_penalty > 0.0:
                phi_prop_pre_cells = self._phi_learner_non_hq_property_cells(self.state)
            else:
                phi_prop_pre_cells = None
        else:
            phi_before = 0.0
            pre_enemy_alive = None
            pre_action_capture_progress = None
            phi_prop_pre_cells = None

        acting = int(self.state.active_player)
        assert candidate is not None
        reward = 0.0
        sparse_total = 0.0
        done = False
        rc = {"learner_engine_signed_sparse_capture": 0.0}
        build_action_taken = False
        # Opening-book candidate lists are built with capture MOVE gate off; engine
        # step uses the same legality so masks and STEP-GATE agree.
        _step_capture_ctx: Any = (
            _without_capture_move_gate()
            if skip_capture_teacher
            else contextlib.nullcontext()
        )
        with _step_capture_ctx:
            for sub in (candidate.first, candidate.second):
                if sub is None:
                    continue
                # Track if any BUILD action was taken this turn
                if sub.action_type == ActionType.BUILD:
                    build_action_taken = True
                phi_power_b, phi_power_b_act, phi_power_rc, _phi_pa_mag, _phi_as_mag = (
                    self._phi_power_activation_and_attack_bonus(
                        sub, acting, self.state, learner_frame=True
                    )
                )
                self.state, r_engine, done = self._engine_step_with_belief(sub)
                r_signed = self._signed_engine_reward(r_engine, acting)
                sparse_total += r_signed
                reward += r_signed
                if phi_power_b != 0.0:
                    reward += phi_power_b
                    rc.update(phi_power_rc)
                    if hasattr(self, "_episode_reward_power_cumulative"):
                        self._episode_reward_power_activation_cumulative[
                            acting
                        ] += float(_phi_pa_mag)
                        self._episode_reward_attack_stats_cumulative[acting] += float(
                            _phi_as_mag
                        )
                        self._episode_reward_power_cumulative[acting] += float(
                            _phi_pa_mag + _phi_as_mag
                        )
                self._phi_after_step_record_power_activations(sub, acting)
                if done:
                    break
        rc["learner_engine_signed_sparse_capture"] = float(sparse_total)
        action = candidate.terminal_action
        if self.state is not None and self.state.done:
            rb = reward
            reward = self._apply_phi_sparse_terminal_replacement(reward, acting)
            rc["phi_max_turn_sparse_terminal_adjust"] = float(reward - rb)
            # Φ day-cap replacement only (not engine HQ ±1). Antisymmetric in
            # (P0,P1) so both seats see the opposite adjustment.
            if hasattr(self, "_episode_reward_sparse_cumulative"):
                learner = int(self._learner_seat)
                enemy = 1 - learner
                dphi = float(reward - rb)
                self._episode_reward_sparse_cumulative[learner] += dphi
                self._episode_reward_sparse_cumulative[enemy] -= dphi
        kb = 0.0
        if pre_enemy_alive is not None and self.state is not None:
            kb = float(self._phi_enemy_kill_one_time_bonus(pre_enemy_alive))
            reward += kb
            rc["phi_enemy_kill_one_time_bonus"] = kb
            # Track kill bonus for learner seat
            if hasattr(self, "_episode_reward_kill_bonus_cumulative"):
                learner = int(self._learner_seat)
                self._episode_reward_kill_bonus_cumulative[learner] += kb
        self._capture_frame(action=action)

        if self._first_learner_capture_step is None and (
            action.action_type == ActionType.CAPTURE
        ):
            self._first_learner_capture_step = self._p0_env_steps

        # Dense shaping (legacy "level" mode): me − enemy in learner frame.
        if self._reward_shaping_mode == "level" and not done:
            me = self._learner_seat
            en = self._enemy_seat
            p_me = self.state.count_properties(me)
            p_en = self.state.count_properties(en)
            diff = p_me - p_en
            lv_prop = (
                diff * 0.005 if diff >= 0 else diff * 0.001
            )
            reward += lv_prop
            rc["level_dense_property_margin"] = float(lv_prop)

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
            lv_army = float((v_me - v_en) * 1e-6)
            reward += lv_army
            rc["level_dense_army_value_diff"] = lv_army

        # ── Auto-step non-learner seat ─────────────────────────────────────────
        _opp_eng = 0.0
        if not done and int(self.state.active_player) != self._learner_seat:
            _t_p1 = time.perf_counter()
            if self.opponent_policy is not None:
                reward, _opp_eng = self._run_policy_opponent(reward)
            else:
                reward, _opp_eng = self._run_random_opponent(reward)
            _p1_delta = time.perf_counter() - _t_p1
        rc["opponent_engine_signed_microsteps"] = float(_opp_eng)

        dd_pen_v = 0.0
        if self.state is not None:
            cur_dd = int(self.state.turn)
            prev_dd = int(self._dd_prev_calendar_turn_at_step_end)
            if self._designed_desires_shaping_active() and cur_dd > prev_dd:
                dd_pen_v = self._designed_desires_checkpoint_penalty_sum_for_turn_bump(
                    prev_dd, cur_dd, int(self._learner_seat)
                )
            self._dd_prev_calendar_turn_at_step_end = cur_dd
        if dd_pen_v != 0.0:
            reward += dd_pen_v
            if hasattr(self, "_episode_reward_designed_desires_cumulative"):
                self._episode_reward_designed_desires_cumulative[
                    int(self._learner_seat)
                ] += float(dd_pen_v)
        rc["phi_designed_desires_checkpoint_penalty"] = float(dd_pen_v)

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
            pd = phi_after - phi_before
            rc["phi_potential_delta"] = float(pd)
            reward += pd
            if hasattr(self, "_episode_reward_phi_potential_delta_cumulative"):
                self._episode_reward_phi_potential_delta_cumulative[
                    int(self._learner_seat)
                ] += float(pd)
            
            # Track Φ component deltas for learner seat.
            # Unlike the training reward (which zeros Φ at termination so
            # the agent doesn't bank useless potential), the component
            # breakdown *must* use the actual state Φ so accumulated deltas
            # converge to phi_final (not zero).  Otherwise the final step
            # delta 0 - phi_before_components wipes every accumulator clean.
            if hasattr(self, "_episode_reward_army_cumulative"):
                learner = int(self._learner_seat)
                phi_before_components = getattr(self, "_phi_before_components", None)
                if phi_before_components is not None:
                    phi_after_components = self._compute_phi_components_for_seat(
                        self.state, learner
                    )
                    d_army = phi_after_components["army"] - phi_before_components["army"]
                    d_prop = phi_after_components["property"] - phi_before_components["property"]
                    d_cap = phi_after_components["capture"] - phi_before_components["capture"]
                    d_inc = phi_after_components["income"] - phi_before_components["income"]
                    enemy = 1 - learner
                    # Learner-frame ΔΦ components; opponent seat moves the
                    # negated component delta (antisymmetric split across seats).
                    self._episode_reward_army_cumulative[learner] += d_army
                    self._episode_reward_army_cumulative[enemy] -= d_army
                    self._episode_reward_property_cumulative[learner] += d_prop
                    self._episode_reward_property_cumulative[enemy] -= d_prop
                    self._episode_reward_capture_cumulative[learner] += d_cap
                    self._episode_reward_capture_cumulative[enemy] -= d_cap
                    self._episode_reward_income_cumulative[learner] += d_inc
                    self._episode_reward_income_cumulative[enemy] -= d_inc
            # Explicit penalty for learner capture progress that was lost (interrupted/reset).
            # If learner started a capture (pre→mid progress increased) but opponent killed the
            # capturer or it got reset, apply -κ × lost_progress penalty.
            kappa_loss = 0.0
            if pre_action_capture_progress is not None:
                post_progress = self._get_learner_capture_progress() if not (terminated or truncated) else 0.0
                # Only penalize if learner made progress from their action that got lost
                if post_progress < pre_action_capture_progress:
                    lost = pre_action_capture_progress - post_progress
                    kappa_loss = float(-(self._phi_kappa * lost))
                    reward -= self._phi_kappa * lost
            rc["phi_capture_interrupt_penalty"] = float(kappa_loss)
            if (
                kappa_loss != 0.0
                and hasattr(self, "_episode_reward_capture_interrupt_cumulative")
            ):
                self._episode_reward_capture_interrupt_cumulative[
                    int(self._learner_seat)
                ] += float(kappa_loss)

        phi_prop_loss_n = 0
        phi_prop_pen = 0.0
        if (
            phi_prop_pre_cells is not None
            and self._phi_enemy_property_capture_penalty > 0.0
            and self.state is not None
        ):
            phi_prop_loss_n = self._phi_count_learner_props_lost_to_enemy(
                self.state, phi_prop_pre_cells, self._enemy_seat
            )
            if phi_prop_loss_n:
                phi_prop_pen = -(
                    float(self._phi_enemy_property_capture_penalty)
                    * float(phi_prop_loss_n)
                )
                reward += phi_prop_pen
                self._phi_enemy_property_captures_ep += int(phi_prop_loss_n)
        if (
            phi_prop_pen != 0.0
            and hasattr(self, "_episode_reward_enemy_property_loss_cumulative")
        ):
            self._episode_reward_enemy_property_loss_cumulative[
                int(self._learner_seat)
            ] += float(phi_prop_pen)
        rc["phi_enemy_property_loss_penalty"] = float(phi_prop_pen)

        trunc_pen_v = 0.0
        if truncated and not terminated:
            raw_tp = os.environ.get(TRUNCATION_PENALTY_ENV, "0").strip()
            if raw_tp:
                try:
                    trunc_pen_v = -float(raw_tp)
                    reward -= float(raw_tp)
                except ValueError:
                    pass
        rc["truncation_penalty_awbw_trunc"] = trunc_pen_v

        hoard_pen_v = 0.0
        if (
            self._hoard_penalty > 0.0
            and self.state is not None
            and action.action_type == ActionType.END_TURN
            and int(self.state.funds[acting]) > int(self._hoard_funds_threshold)
        ):
            hoard_pen_v = -float(self._hoard_penalty)
            reward -= float(self._hoard_penalty)
        rc["end_turn_hoard_penalty"] = hoard_pen_v

        # Building punishment: END_TURN with owned bases but no build
        build_punish_v = 0.0
        if (
            self._build_punishment > 0.0
            and self.state is not None
            and action.action_type == ActionType.END_TURN
            and not build_action_taken
            and self._player_has_bases(self.state, acting)
        ):
            # Punishment equal to being down 30k unit value to opponent (increased from 6k)
            # Since phi_alpha is the army value coefficient, being down 30k is -30000 * phi_alpha
            punishment = -30000.0 * self._phi_alpha
            build_punish_v = punishment
            reward += punishment
        rc["end_turn_build_punishment"] = build_punish_v
        if build_punish_v != 0.0 and hasattr(self, "_episode_reward_build_punishment_cumulative"):
            self._episode_reward_build_punishment_cumulative[acting] += float(build_punish_v)

        # Partial win at the P0 step cap: engine never emits ±1 without a terminal, so
        # credit half a win (+0.5 vs +1.0) when we hit max_env_steps with a
        # property lead in the learner frame (same single-property margin as calendar tiebreak).
        pb_tiebreak = 0.0
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
            if prop_lead >= 1:
                self._log_tie_breaker_property_count = int(prop_lead)
                pb_tiebreak = 0.5
                reward += 0.5
        rc["partial_win_property_lead_bonus_half"] = pb_tiebreak

        if self._reward_shaping_mode != "phi":
            rc.setdefault("phi_potential_delta", 0.0)
            rc.setdefault("phi_capture_interrupt_penalty", 0.0)
        rc.setdefault("phi_designed_desires_checkpoint_penalty", 0.0)
        rc.setdefault("phi_max_turn_sparse_terminal_adjust", 0.0)
        rc.setdefault("level_dense_property_margin", 0.0)
        rc.setdefault("level_dense_army_value_diff", 0.0)

        obs = self._get_obs(observer=self._learner_seat)
        info = {
            **self._episode_info,
            "turn": self.state.turn,
            "winner": self.state.winner,
            "truncated": truncated,
            "phi_enemy_property_captures": int(phi_prop_loss_n),
        }
        if (self.state.win_reason or "") == SPIRIT_BROKEN_REASON:
            info["spirit_broken"] = True
            _sk = getattr(self.state.spirit, "spirit_broken_kind", None)
            if _sk is not None:
                info["spirit_broken_kind"] = _sk

        inc_adj = 0.0
        if terminated:
            raw_inc = os.environ.get(INCOME_TERM_COEF_ENV, "0").strip()
            if raw_inc:
                try:
                    coef = float(raw_inc)
                    if coef != 0.0:
                        cap_lim = max(1, self.state.map_data.cap_limit)
                        inc_me = self.state.count_income_properties(self._learner_seat)
                        inc_en = self.state.count_income_properties(self._enemy_seat)
                        inc_adj = coef * (inc_me - inc_en) / float(cap_lim)
                        reward += inc_adj
                except ValueError:
                    inc_adj = 0.0
        rc["income_terminal_coefficient_adj"] = float(inc_adj)

        time_cost_adj = 0.0
        if not self.state.done and not truncated:
            raw_tc = os.environ.get(TIME_COST_ENV, "0").strip()
            if raw_tc:
                try:
                    time_cost_adj = -float(raw_tc)
                    reward -= float(raw_tc)
                except ValueError:
                    time_cost_adj = 0.0
        rc["per_step_time_cost"] = float(time_cost_adj)

        _pre_spirit = reward
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
        rc["spirit_broken_sparse_substitution_delta"] = float(reward - _pre_spirit)

        draw_common_v = 0.0
        if (
            self.state is not None
            and self.state.done
            and int(self.state.winner if self.state.winner is not None else -2) == -1
        ):
            draw_common_v = min(
                0.0, float(rc.get("phi_max_turn_sparse_terminal_adjust", 0.0))
            )
        common_reward_v = float(time_cost_adj + trunc_pen_v + draw_common_v)
        seat_local_reward_v = float(hoard_pen_v + dd_pen_v)
        competitive_reward_v = float(reward - common_reward_v - seat_local_reward_v)
        reward_contract = self._reward_contract_info(
            competitive_learner=competitive_reward_v,
            common_learner=common_reward_v,
            seat_local_learner=seat_local_reward_v,
            final_reward=float(reward),
        )
        if self._pairwise_zero_sum_reward:
            reward = float(reward_contract["training_reward"])

        _rc_sum = sum(rc.values())
        rc["_component_sum_gap"] = float(reward - _rc_sum)
        info["reward_components"] = rc
        info["reward_contract"] = reward_contract
        info["property_pressure"] = self._property_pressure_snapshot()
        if self._reward_shaping_mode == "phi" and self.state is not None:
            info["phi_diagnostics"] = self._phi_diagnostics_for_seat(
                self.state, int(self._learner_seat)
            )
        info["reward"] = float(reward)

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

    def _reward_contract_info(
        self,
        *,
        competitive_learner: float,
        common_learner: float,
        seat_local_learner: float,
        final_reward: float,
    ) -> dict[str, Any]:
        """Inspectable reward contract for learner-frame ``step()`` rows.

        Competitive reward is represented as a two-seat zero-sum pair. Common
        penalties intentionally hit both POVs, while seat-local discipline only
        hits the learner row that caused it. This keeps ego-centric consumers
        explicit without changing ``step_active_seat_once()``.
        """
        learner = int(self._learner_seat)
        enemy = 1 - learner
        competitive = [0.0, 0.0]
        competitive[learner] = float(competitive_learner)
        competitive[enemy] = -float(competitive_learner)

        common = [float(common_learner), float(common_learner)]
        seat_local = [0.0, 0.0]
        seat_local[learner] = float(seat_local_learner)
        final_by_seat = [
            float(competitive[i] + common[i] + seat_local[i])
            for i in (0, 1)
        ]
        return {
            "version": "pairwise_zero_sum_v1",
            "pairwise_zero_sum_enabled": bool(self._pairwise_zero_sum_reward),
            "learner_seat": learner,
            "enemy_seat": enemy,
            "competitive_by_seat": competitive,
            "common_by_seat": common,
            "seat_local_by_seat": seat_local,
            "final_by_seat": final_by_seat,
            "training_reward": float(final_by_seat[learner]),
            "legacy_reward": float(final_reward),
            "competitive_antisymmetry_gap": float(sum(competitive)),
        }

    def _maybe_record_neutral_income_snapshot_days(self) -> None:
        """First time ``state.turn`` hits a milestone day, stash neutral-income count."""

        if self.state is None:
            return
        cur = int(getattr(self.state, "turn", 0) or 0)
        if cur not in GAME_LOG_NEUTRAL_INCOME_SNAPSHOT_DAYS:
            return
        dct = self._neutral_income_snapshot_by_day
        if cur in dct:
            return
        snap = self._property_pressure_snapshot()
        ni = snap.get("neutral_income_properties")
        if ni is None:
            return
        dct[cur] = int(ni)

    def _property_pressure_snapshot(self) -> dict[str, Any]:
        """Compact opening/property-pressure diagnostics for reward audits."""
        st = self.state
        if st is None:
            return {}

        def is_income_prop(prop: Any) -> bool:
            return (
                prop.owner is not None
                and not bool(getattr(prop, "is_hq", False))
                and not bool(getattr(prop, "is_comm_tower", False))
                and not bool(getattr(prop, "is_lab", False))
            )

        income_owned = [0, 0]
        props_owned = [0, 0]
        contested = [0.0, 0.0]
        contested_income = [0.0, 0.0]
        neutral_income = 0

        for prop in st.properties:
            owner = prop.owner
            if owner in (0, 1):
                props_owned[int(owner)] += 1
                if is_income_prop(prop):
                    income_owned[int(owner)] += 1
            elif (
                owner is None
                and not bool(getattr(prop, "is_hq", False))
                and not bool(getattr(prop, "is_comm_tower", False))
                and not bool(getattr(prop, "is_lab", False))
            ):
                neutral_income += 1

            cp = int(getattr(prop, "capture_points", 20) or 20)
            if cp < 20:
                chip = float(1.0 - cp / 20.0)
                for seat in (0, 1):
                    if owner != seat:
                        contested[seat] += chip
                        if owner is not None and is_income_prop(prop):
                            contested_income[seat] += chip

        return {
            "turn": int(getattr(st, "turn", 0) or 0),
            "learner_seat": int(self._learner_seat),
            "income_owned_by_seat": income_owned,
            "properties_owned_by_seat": props_owned,
            "neutral_income_properties": int(neutral_income),
            "contested_capture_chips_by_seat": contested,
            "contested_income_chips_by_seat": contested_income,
        }

    def action_masks(self) -> np.ndarray:
        """Return valid-action bool mask. Required by MaskablePPO wrappers."""
        if self.state is None:
            if self._use_preallocated_buffers:
                self._action_mask_buf.fill(False)
                return self._action_mask_buf
            return np.zeros(self.action_space.n, dtype=bool)

        feats, mask, cands = self._build_candidate_policy_tensors()
        self._candidate_features_cache = feats
        self._candidate_mask_cache = mask
        self._candidate_list_cache = cands
        if self._use_preallocated_buffers:
            self._action_mask_buf.fill(False)
            self._action_mask_buf[: mask.shape[0]] = mask
            return self._action_mask_buf
        return mask

    def active_seat_observation(self) -> dict:
        """Observation from the current engine active player's perspective.

        Async dual-gradient self-play uses this to train both seats with one
        shared policy.  Standard ``step()`` keeps using learner-frame
        observations and opponent autoplay.
        """
        if self.state is None:
            raise RuntimeError("Call reset() before active_seat_observation().")
        return self._get_obs(observer=int(self.state.active_player))

    def active_seat_action_mask(self) -> np.ndarray:
        """Legal-action mask for the current engine active player."""
        if self.state is None:
            if self._use_preallocated_buffers:
                self._action_mask_buf.fill(False)
                return self._action_mask_buf
            return np.zeros(MAX_CANDIDATES, dtype=bool)
        if int(self.state.active_player) == int(self._learner_seat):
            feats, mask, cands = self._build_candidate_policy_tensors()
        else:
            feats, mask, cands = candidate_arrays(self.state, max_candidates=MAX_CANDIDATES)
        self._candidate_features_cache = feats
        self._candidate_mask_cache = mask
        self._candidate_list_cache = cands
        if self._use_preallocated_buffers:
            self._action_mask_buf.fill(False)
            self._action_mask_buf[: mask.shape[0]] = mask
            return self._action_mask_buf
        return mask

    def step_active_seat_once(
        self, action_idx: int
    ) -> tuple[dict, float, bool, bool, dict]:
        """Apply one action for ``state.active_player`` and return that seat's reward.

        This is intentionally lower level than :meth:`step`: it does not autoplay
        the other seat back to a fixed learner clock.  It is for async
        dual-gradient self-play where every engine decision becomes one training
        row for the shared policy.
        """
        if self.state is None:
            raise RuntimeError("Call reset() before step_active_seat_once().")

        self._p0_env_steps += 1
        acting = int(self.state.active_player)
        other = 1 - acting
        candidate = self._decode_candidate_action(int(action_idx))
        action = candidate.terminal_action if candidate is not None else None
        if action is None:
            obs = self._get_obs(observer=acting)
            return obs, -0.1, False, False, {
                "invalid_action": True,
                "seat": acting,
                "reward_components": {},
            }

        phi_before_components_dg: dict[str, float] | None = None
        if self._reward_shaping_mode == "phi":
            phi_before = self._compute_phi_for_seat(self.state, acting)
            phi_before_components_dg = self._compute_phi_components_for_seat(
                self.state, acting
            )
        else:
            phi_before = 0.0
        pre_capture_progress = (
            self._capture_progress_for_seat(self.state, acting)
            if self._reward_shaping_mode == "phi"
            else None
        )
        pre_enemy_alive = (
            {
                u.unit_id: (u.unit_type, u.hp)
                for u in self.state.units[other]
                if u.is_alive
            }
            if self._reward_shaping_mode == "phi"
            else None
        )
        pre_prop_cells = (
            self._seat_non_hq_property_cells(self.state, acting)
            if (
                self._reward_shaping_mode == "phi"
                and self._phi_enemy_property_capture_penalty > 0.0
            )
            else None
        )

        assert candidate is not None
        reward = 0.0
        done_chain = False
        rc_actor: dict[str, float] = {
            "acting_engine_sparse_plus_capture_shaping": 0.0,
        }
        build_action_taken = False
        book_steers_capture = self._peek_learner_book_flat_safe_ungated() is not None
        _dg_step_capture_ctx: Any = (
            _without_capture_move_gate()
            if book_steers_capture
            else contextlib.nullcontext()
        )
        with _dg_step_capture_ctx:
            for sub in (candidate.first, candidate.second):
                if sub is None:
                    continue
                # Track if any BUILD action was taken this turn
                if sub.action_type == ActionType.BUILD:
                    build_action_taken = True
                phi_power_b, phi_power_b_act, phi_power_rc, _phi_pa_mag_dg, _phi_as_mag_dg = (
                    self._phi_power_activation_and_attack_bonus(
                        sub, acting, self.state, learner_frame=False
                    )
                )
                self.state, r_engine, done_chain = self._engine_step_with_belief(sub)
                r_signed = float(r_engine)
                rc_actor["acting_engine_sparse_plus_capture_shaping"] += r_signed
                reward += r_signed
                if phi_power_b != 0.0:
                    reward += phi_power_b
                    rc_actor.update(phi_power_rc)
                    if hasattr(self, "_episode_reward_power_cumulative"):
                        self._episode_reward_power_activation_cumulative[acting] += float(
                            _phi_pa_mag_dg
                        )
                        self._episode_reward_attack_stats_cumulative[acting] += float(
                            _phi_as_mag_dg
                        )
                        self._episode_reward_power_cumulative[acting] += float(
                            _phi_pa_mag_dg + _phi_as_mag_dg
                        )
                self._phi_after_step_record_power_activations(sub, acting)
                if done_chain:
                    break
        action = candidate.terminal_action
        rc = rc_actor
        if self.state is not None and self.state.done:
            rb = reward
            reward = self._apply_phi_sparse_terminal_replacement_for_seat(
                reward, acting
            )
            rc["phi_max_turn_sparse_terminal_adjust"] = float(reward - rb)
            if hasattr(self, "_episode_reward_sparse_cumulative"):
                oth = 1 - int(acting)
                dphi = float(reward - rb)
                self._episode_reward_sparse_cumulative[acting] += dphi
                self._episode_reward_sparse_cumulative[oth] -= dphi
        else:
            rc.setdefault("phi_max_turn_sparse_terminal_adjust", 0.0)
        kb = 0.0
        if pre_enemy_alive is not None and self.state is not None:
            kb = float(
                self._phi_enemy_kill_one_time_bonus_for_seat(pre_enemy_alive, acting)
            )
            reward += kb
        rc["phi_enemy_kill_one_time_bonus"] = kb
        if kb != 0.0 and hasattr(self, "_episode_reward_kill_bonus_cumulative"):
            self._episode_reward_kill_bonus_cumulative[acting] += kb

        self._capture_frame(action=action)

        terminated = bool(self.state.done)
        truncated = (
            self._max_env_steps is not None
            and self._p0_env_steps >= self._max_env_steps
            and not terminated
        )
        truncation_reason = "max_env_steps" if truncated else None
        if self._reward_shaping_mode == "phi":
            phi_after = 0.0 if (terminated or truncated) else self._compute_phi_for_seat(
                self.state, acting
            )
            pd = phi_after - phi_before
            reward += pd
            rc["phi_potential_delta"] = float(pd)
            if hasattr(self, "_episode_reward_phi_potential_delta_cumulative"):
                self._episode_reward_phi_potential_delta_cumulative[acting] += float(
                    pd
                )
            kappa_loss = 0.0
            if pre_capture_progress is not None:
                post_progress = (
                    0.0
                    if (terminated or truncated)
                    else self._capture_progress_for_seat(self.state, acting)
                )
                if post_progress < pre_capture_progress:
                    lost = pre_capture_progress - post_progress
                    kappa_loss = float(-(self._phi_kappa * lost))
                    reward -= self._phi_kappa * lost
            rc["phi_capture_interrupt_penalty"] = float(kappa_loss)
            if (
                kappa_loss != 0.0
                and hasattr(self, "_episode_reward_capture_interrupt_cumulative")
            ):
                self._episode_reward_capture_interrupt_cumulative[acting] += float(
                    kappa_loss
                )
            # Φ-component episode totals for ``phi_reward_breakdown`` (mirror / DG path).
            if hasattr(self, "_episode_reward_army_cumulative") and (
                phi_before_components_dg is not None
            ):
                phi_after_components_dg = self._compute_phi_components_for_seat(
                    self.state, acting
                )
                d_a = (
                    phi_after_components_dg["army"]
                    - phi_before_components_dg["army"]
                )
                d_p = (
                    phi_after_components_dg["property"]
                    - phi_before_components_dg["property"]
                )
                d_c = (
                    phi_after_components_dg["capture"]
                    - phi_before_components_dg["capture"]
                )
                d_i = (
                    phi_after_components_dg["income"]
                    - phi_before_components_dg["income"]
                )
                oth = 1 - acting
                self._episode_reward_army_cumulative[acting] += d_a
                self._episode_reward_army_cumulative[oth] -= d_a
                self._episode_reward_property_cumulative[acting] += d_p
                self._episode_reward_property_cumulative[oth] -= d_p
                self._episode_reward_capture_cumulative[acting] += d_c
                self._episode_reward_capture_cumulative[oth] -= d_c
                self._episode_reward_income_cumulative[acting] += d_i
                self._episode_reward_income_cumulative[oth] -= d_i
        else:
            rc.setdefault("phi_potential_delta", 0.0)
            rc.setdefault("phi_capture_interrupt_penalty", 0.0)

        phi_prop_loss_n = 0
        phi_prop_pen = 0.0
        if (
            pre_prop_cells is not None
            and self._phi_enemy_property_capture_penalty > 0.0
            and self.state is not None
        ):
            phi_prop_loss_n = self._count_props_lost_to_seat(
                self.state, pre_prop_cells, other
            )
            if phi_prop_loss_n:
                phi_prop_pen = -(
                    float(self._phi_enemy_property_capture_penalty)
                    * float(phi_prop_loss_n)
                )
                reward += phi_prop_pen
                self._phi_enemy_property_captures_ep += int(phi_prop_loss_n)
        if (
            phi_prop_pen != 0.0
            and hasattr(self, "_episode_reward_enemy_property_loss_cumulative")
        ):
            self._episode_reward_enemy_property_loss_cumulative[acting] += float(
                phi_prop_pen
            )
        rc["phi_enemy_property_loss_penalty"] = float(phi_prop_pen)

        dd_pen_v = 0.0
        if self.state is not None:
            cur_dd = int(self.state.turn)
            prev_dd = int(self._dd_prev_calendar_turn_at_step_end)
            if self._designed_desires_shaping_active() and cur_dd > prev_dd:
                dd_pen_v = self._designed_desires_checkpoint_penalty_sum_for_turn_bump(
                    prev_dd, cur_dd, int(acting)
                )
            self._dd_prev_calendar_turn_at_step_end = cur_dd
        if dd_pen_v != 0.0:
            reward += dd_pen_v
            if hasattr(self, "_episode_reward_designed_desires_cumulative"):
                self._episode_reward_designed_desires_cumulative[acting] += float(
                    dd_pen_v
                )
        rc["phi_designed_desires_checkpoint_penalty"] = float(dd_pen_v)

        trunc_pen_v = 0.0
        if truncated and not terminated:
            raw_tp = os.environ.get(TRUNCATION_PENALTY_ENV, "0").strip()
            if raw_tp:
                try:
                    trunc_pen_v = -float(raw_tp)
                    reward -= float(raw_tp)
                except ValueError:
                    pass
        rc["truncation_penalty_awbw_trunc"] = float(trunc_pen_v)

        # Building punishment: END_TURN with owned bases but no build
        build_punish_v = 0.0
        if (
            self._build_punishment > 0.0
            and self.state is not None
            and action is not None
            and action.action_type == ActionType.END_TURN
            and not build_action_taken
            and self._player_has_bases(self.state, acting)
        ):
            # Punishment equal to being down 30k unit value to opponent (increased from 6k)
            # Since phi_alpha is the army value coefficient, being down 30k is -30000 * phi_alpha
            punishment = -30000.0 * self._phi_alpha
            build_punish_v = punishment
            reward += punishment
        rc["end_turn_build_punishment"] = build_punish_v
        if build_punish_v != 0.0 and hasattr(self, "_episode_reward_build_punishment_cumulative"):
            self._episode_reward_build_punishment_cumulative[acting] += float(build_punish_v)

        _rc_sum = sum(rc.values())
        rc["_component_sum_gap"] = float(reward - _rc_sum)

        next_observer = int(self.state.active_player) if not terminated else acting
        obs = self._get_obs(observer=next_observer)
        info = {
            **self._episode_info,
            "turn": self.state.turn,
            "winner": self.state.winner,
            "truncated": truncated,
            "truncation_reason": truncation_reason,
            "seat": acting,
            "next_active_seat": int(self.state.active_player),
            "dual_gradient_self_play": True,
            "phi_enemy_property_captures": int(phi_prop_loss_n),
            "reward_components": rc,
            "reward": float(reward),
        }
        if self._reward_shaping_mode == "phi" and self.state is not None:
            info["phi_diagnostics"] = self._phi_diagnostics_for_seat(
                self.state, acting
            )
        if terminated or truncated:
            self._log_episode_truncated = bool(truncated)
            self._log_episode_truncation_reason = truncation_reason
            self._log_finished_game()
        return obs, float(reward), terminated, truncated, info

    def _begin_opening_book_episode(self) -> None:
        mgr = getattr(self, "_opening_book_manager", None)
        if mgr is None or self.state is None:
            return
        try:
            co_ids = [
                int(self.state.co_states[0].co_id),
                int(self.state.co_states[1].co_id),
            ]
            mgr.on_episode_start(
                episode_id=int(self._episode_id),
                map_id=int(self.state.map_data.map_id),
                co_ids=co_ids,
            )
            self._sync_opening_book_log()
        except Exception as exc:
            self._opening_book_log["opening_book_init_error"] = repr(exc)

    def _sync_opening_book_log(self) -> None:
        mgr = getattr(self, "_opening_book_manager", None)
        if mgr is not None:
            self._opening_book_log.update(mgr.log_fields())

    def _maybe_force_learner_opening_book_mask(self, mask: np.ndarray) -> np.ndarray:
        """Flat-mask forcing is obsolete; learner books use :meth:`_build_candidate_policy_tensors`."""
        return mask

    def _maybe_commit_learner_opening_book_action(self, action_idx: int) -> None:
        if (
            not self._opening_book_force_mask_for_learner
            or self.state is None
            or int(self.state.active_player) != int(self._learner_seat)
        ):
            return
        mgr = getattr(self, "_opening_book_manager", None)
        if mgr is None:
            return
        # Validate against the current legal mask before committing.  If the
        # mask was not forced or the model did not return the forced action,
        # commit_flat marks the selected book line desynced instead of silently
        # replacing the learner action.
        _mout = self._flat_action_mask_buf if self._use_preallocated_buffers else None
        mask = _get_action_mask(self.state, out=_mout, legal=self._get_legal())
        expected = mgr.peek_flat(
            seat=int(self._learner_seat),
            calendar_turn=int(getattr(self.state, "turn", 0) or 0),
            action_mask=mask,
        )
        if expected is not None:
            mgr.commit_flat(seat=int(self._learner_seat), action_idx=int(action_idx))
            self._sync_opening_book_log()

    def _suggest_opening_book_for_active(self, mask: np.ndarray) -> int | None:
        if self.state is None:
            return None
        mgr = getattr(self, "_opening_book_manager", None)
        if mgr is None:
            return None
        
        # Check for strike release before suggesting a book action
        active_seat = int(self.state.active_player)
        ctl = mgr.controllers.get(active_seat)
        if ctl is not None and hasattr(ctl, 'check_strike_release'):
            if ctl.check_strike_release(self.state):
                # Book was released due to strike range
                self._sync_opening_book_log()
                return None
        
        a = mgr.suggest_flat(
            seat=active_seat,
            calendar_turn=int(getattr(self.state, "turn", 0) or 0),
            action_mask=mask,
        )
        self._sync_opening_book_log()
        return int(a) if a is not None else None

    def render(self) -> str | None:
        if self.render_mode == "ansi" and self.state is not None:
            return self.state.render_ascii()
        return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _designed_desires_shaping_active(self) -> bool:
        if self._reward_shaping_mode != "phi":
            return False
        try:
            mid = int(self._episode_info.get("map_id") or 0)
        except (TypeError, ValueError):
            return False
        return mid == PHI_DESIGNED_DESIRES_MAP_ID

    def _designed_desires_checkpoint_penalty_sum_for_turn_bump(
        self, prev_turn: int, new_turn: int, seat: int
    ) -> float:
        """Accumulate miss penalties when ``GameState.turn`` advances past checkpoints."""
        total = 0.0
        st = self.state
        if st is None:
            return total
        seat_i = int(seat) & 1
        for day in range(int(prev_turn) + 1, int(new_turn) + 1):
            tup = PHI_DESIGNED_DESIRES_CHECKPOINTS.get(day)
            if tup is None:
                continue
            need = int(tup[seat_i])
            if int(st.count_properties(seat_i)) >= need:
                continue
            self._designed_desires_checkpoint_misses += 1
            if self._designed_desires_checkpoint_misses == 1:
                total -= 0.1
            else:
                total -= 0.01
        return float(total)

    def _phi_strike_av_bonus(self, unit: Any, co: Any, terrain: Any) -> int:
        """Attack value (AWBW %% points) for Φ stats chip; mirrors ``calculate_damage`` riders."""
        av = int(co.total_atk_for_unit(unit.unit_type))
        if co.co_id == 16 and (co.scop_active or co.cop_active):
            av += int(terrain.defense) * 10
        av += int(_kindle_atk_rider(co, terrain))
        av += int(_colin_atk_rider(co))
        return av

    def _phi_strike_def_bonus_portion(
        self, unit: Any, co: Any, terrain: Any, foe_type: UnitType
    ) -> float:
        """Defense side of stats advantage: CO DEF − 100 plus 10 × terrain stars (surface)."""
        dv = int(co.total_def_for_unit_against(unit.unit_type, foe_type))
        dtr = 0 if unit.is_submerged else int(terrain.defense)
        return float(dv - 100 + dtr * 10)

    def _phi_attack_stats_advantage_mult(
        self, state: GameState, action: Action
    ) -> float:
        if action.action_type != ActionType.ATTACK or action.target_pos is None:
            return 0.0
        att = state.get_unit_at(*action.unit_pos) if action.unit_pos else None
        dfd = state.get_unit_at(*action.target_pos)
        if att is None or dfd is None or not att.is_alive or not dfd.is_alive:
            return 0.0
        move_r, move_c = (action.move_pos if action.move_pos else action.unit_pos)
        tr, tc = action.target_pos
        att_terrain = get_terrain(state.map_data.terrain[move_r][move_c])
        def_terrain = get_terrain(state.map_data.terrain[tr][tc])
        att_co = state.co_states[att.player]
        dfd_co = state.co_states[dfd.player]
        att_av = self._phi_strike_av_bonus(att, att_co, att_terrain)
        def_hypo_av = self._phi_strike_av_bonus(dfd, dfd_co, def_terrain)
        atk_term = 0.9 * ((att_av - 100) - (def_hypo_av - 100))
        def_bonus_def = self._phi_strike_def_bonus_portion(
            dfd, dfd_co, def_terrain, att.unit_type
        )
        def_bonus_att = self._phi_strike_def_bonus_portion(
            att, att_co, att_terrain, dfd.unit_type
        )
        return float(atk_term + (def_bonus_def - def_bonus_att))

    def _signed_engine_reward(self, r_engine: float, acting_seat: int) -> float:
        """Map engine reward (acting player's perspective) into learner coordinates."""
        if int(acting_seat) == int(self._learner_seat):
            return float(r_engine)
        return float(-r_engine)

    def _phi_power_activation_and_attack_bonus(
        self,
        action: Action,
        acting: int,
        state: GameState,
        *,
        learner_frame: bool,
    ) -> tuple[float, float, dict[str, float], float, float]:
        """Φ-mode COP/SCOP tap bonuses and attack stats-advantage micro-bonus.

        ``state`` must be the **pre-step** game state. ``learner_frame``: when
        True (``step()`` / opponent autoplay), return bonus signed into learner
        coordinates; when False (``step_active_seat_once``), return acting-seat
        frame.

        COP activation shaping is disabled; SCOP bonus is 5%% of the nominal
        value (95%% reduction). ATTACK adds ``0.001 × max(0, mult)`` where
        ``mult`` compares strike AV/DV ingredients (see
        :meth:`_phi_attack_stats_advantage_mult`).

        Returns
        ``(bonus_for_reward, bonus_acting_seat, rc, power_activation_mag, attack_stats_mag)``
        where the last two are **unsigned** acting-seat magnitudes for
        ``phi_reward_breakdown`` (``power_activation_mag`` is SCOP taps only here;
        ``attack_stats_mag`` is the strike chip). ``bonus_acting_seat`` equals their
        sum.
        """
        if self._reward_shaping_mode != "phi":
            return 0.0, 0.0, {}, 0.0, 0.0
        a = int(acting)
        co = state.co_states[a]
        b_act = 0.0
        key: str | None = None
        rc_extra: dict[str, float] = {}
        pow_act_mag = 0.0
        atk_stats_mag = 0.0
        vb_ref = float(PHI_VON_BOLT_SCOP_REF_THRESHOLD)
        if action.action_type == ActionType.ACTIVATE_COP:
            return 0.0, 0.0, {}, 0.0, 0.0
        if action.action_type == ActionType.ACTIVATE_SCOP:
            scale = float(co._scop_threshold) / vb_ref
            b_act = float(PHI_SCOP_ACTIVATION_BONUS) * scale * 0.05
            pow_act_mag = float(b_act)
            key = "phi_scop_activation_bonus"
            rc_extra["phi_power_bonus_meter_scale_vs_vb_scop"] = float(scale)
        elif action.action_type == ActionType.ATTACK:
            mult = self._phi_attack_stats_advantage_mult(state, action)
            rc_extra["phi_attack_stats_advantage_mult"] = float(mult)
            if mult <= 0.0:
                return 0.0, 0.0, rc_extra, 0.0, 0.0
            b_act = 0.001 * float(mult)
            atk_stats_mag = float(b_act)
            key = "phi_attack_stats_advantage_bonus"
        if b_act == 0.0 or key is None:
            return 0.0, 0.0, (rc_extra if rc_extra else {}), 0.0, 0.0
        if learner_frame:
            b = float(self._signed_engine_reward(b_act, a))
        else:
            b = b_act
        out: dict[str, float] = {key: float(b)}
        out.update(rc_extra)
        return b, float(b_act), out, float(pow_act_mag), float(atk_stats_mag)

    def _phi_after_step_record_power_activations(
        self, action: Action, acting: int
    ) -> None:
        """Count COP/SCOP uses by seat for game_log (all reward modes)."""
        a = int(acting)
        if action.action_type == ActionType.ACTIVATE_COP:
            self._episode_cop_by_seat[a] += 1
        elif action.action_type == ActionType.ACTIVATE_SCOP:
            self._episode_scop_by_seat[a] += 1

    def _seat_frame_terminal_outcome(self, observer_seat: int) -> float:
        """Sparse terminal outcome from ``observer_seat`` coordinates."""
        st = self.state
        if st is None or not st.done or st.winner is None:
            return 0.0
        wi = int(st.winner)
        if wi == -1:
            return 0.0
        return 1.0 if wi == int(observer_seat) else -1.0

    @staticmethod
    def _engine_terminal_sparse_pair_for_log_winner(
        winner: int | None,
    ) -> tuple[float, float]:
        """Canonical engine-style terminal sparse ±1 for (P0, P1) from row ``winner``.

        Uses the same ``winner`` encoding as game_log (0 / 1 / -1 draw / None unknown).
        Truncation tiebreak rows should set ``log_winner`` before calling.
        """
        if winner is None:
            return (0.0, 0.0)
        try:
            w = int(winner)
        except (TypeError, ValueError):
            return (0.0, 0.0)
        if w == -1:
            return (0.0, 0.0)
        if w == 0:
            return (1.0, -1.0)
        if w == 1:
            return (-1.0, 1.0)
        return (0.0, 0.0)

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

    def _apply_phi_sparse_terminal_replacement_for_seat(
        self, reward: float, observer_seat: int
    ) -> float:
        """Φ mode terminal replacement in an arbitrary seat frame."""
        if self._reward_shaping_mode != "phi" or self.state is None or not self.state.done:
            return float(reward)
        wr = self.state.win_reason
        if wr not in PHI_DAY_CAP_REASONS:
            return float(reward)
        ls = self._seat_frame_terminal_outcome(observer_seat)
        rest = float(reward) - ls
        if wr in PHI_DAY_CAP_DRAWLIKE:
            return -0.1 + rest
        w = self.state.winner
        if w is None or int(w) == -1:
            return float(reward)
        me = int(observer_seat)
        en = 1 - me
        if int(w) == me:
            l_new = 0.5
        else:
            p_en = int(self.state.count_properties(en))
            p_me = int(self.state.count_properties(me))
            if p_en - p_me >= 1:
                l_new = -0.5
            else:
                l_new = -1.0
        return l_new + rest

    def _apply_phi_sparse_terminal_replacement(
        self, reward: float, acting_seat: int
    ) -> float:
        """Φ mode: replace engine sparse terminal ±1.0/0, not stack on it.

        * ``max_days_tie`` / legacy ``max_turns_tie`` or ``max_days_draw`` /
          ``max_turns_draw`` → −0.1 (replaces 0.0)
        * ``max_days_tiebreak`` / ``max_turns_tiebreak`` win → +0.5 (replaces +1.0)
        * ``max_days_tiebreak`` / ``max_turns_tiebreak`` loss with **≥1** property deficit in learner
          frame (enemy has more properties) → −0.5 (replaces −1.0);
          otherwise → keep −1.0

        Captured as ``rest = reward - ls`` and recombined (preserves per-step
        capture shaping in ``rest``). Non-day-cap terminations and ``level``
        mode pass through.
        """
        if self._reward_shaping_mode != "phi" or self.state is None or not self.state.done:
            return float(reward)
        wr = self.state.win_reason
        if wr not in PHI_DAY_CAP_REASONS:
            return float(reward)
        ls = self._learner_frame_terminal_outcome(acting_seat)
        rest = float(reward) - ls
        if wr in PHI_DAY_CAP_DRAWLIKE:
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
            if p_en - p_me >= 1:
                l_new = -0.5
            else:
                l_new = -1.0
        return l_new + rest

    def _income_saturation(self, state: GameState, me: int, en: int) -> float:
        """Superlinear income-property lead bonus in learner frame.
        
        Grows as log(1 + max(0, inc_lead))^2 to avoid saturation while still
        rewarding map saturation. Sign-aware: positive when ahead, negative when behind.
        Only applied from day 8 onward to avoid poisoning early-game exploration.
        """
        if int(state.turn) < 8:
            return 0.0
        inc_me = state.count_income_properties(me)
        inc_en = state.count_income_properties(en)
        lead = inc_me - inc_en
        if lead == 0:
            return 0.0
        return math.log(1.0 + abs(lead)) ** 2 * (1.0 if lead > 0 else -1.0)

    def _compute_phi(self, state: GameState) -> float:
        """Potential Φ(s) in the **learner** frame (me = ``_learner_seat``).

        Φ = α × army_value_diff + β × property_diff + κ × contested_cap_diff + γ × income_saturation

        Terms:
        - army_value: Σ unit.cost × hp/100 for alive units
        - property_diff: owned properties (directly owned, not neutral)
        - contested_cap: partial capture progress (chip = 1 - cp/20) on enemy-owned properties
          WARNING: κ rewards *progress toward* capture, not completion — this can incentivize
          repeatedly starting captures without finishing them. Consider setting κ=0 or using
          completed-capture-only logic.
        - income_saturation: log-scale bonus for income property lead after day 8
        """
        return self._compute_phi_for_seat(state, int(self._learner_seat))

    def _compute_phi_for_seat(self, state: GameState, observer_seat: int) -> float:
        """Potential Φ(s) from ``observer_seat`` coordinates."""
        me = int(observer_seat)
        en = 1 - me

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

        cap_me = self._capture_progress_for_seat(state, me)
        cap_en = self._capture_progress_for_seat(state, en)

        return (
            self._phi_alpha * (v_me - v_en)
            + self._phi_beta * (p_me - p_en)
            + self._phi_kappa * (cap_me - cap_en)
            + self._phi_gamma * self._income_saturation(state, me, en)
        )

    def _compute_phi_components_for_seat(self, state: GameState, observer_seat: int) -> dict[str, float]:
        """Break down Φ(s) into its components from ``observer_seat`` coordinates."""
        me = int(observer_seat)
        en = 1 - me

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

        cap_me = self._capture_progress_for_seat(state, me)
        cap_en = self._capture_progress_for_seat(state, en)

        return {
            "army": self._phi_alpha * (v_me - v_en),
            "property": self._phi_beta * (p_me - p_en),
            "capture": self._phi_kappa * (cap_me - cap_en),
            "income": self._phi_gamma * self._income_saturation(state, me, en),
        }

    def _get_learner_capture_progress(self) -> float:
        """Compute learner's partial capture progress (chip sum) toward enemy properties.

        This is the κ-contribution from learner's perspective: sum of (1 - cp/20) for
        all properties that learner is capturing but doesn't yet own. Used to detect
        when captures get interrupted/reset so we can apply explicit negative reward.
        """
        if self.state is None:
            return 0.0
        return self._capture_progress_for_seat(self.state, int(self._learner_seat))

    def _phi_is_income_property(self, prop: Any) -> bool:
        """Whether a property contributes normal income/saturation pressure."""
        return (
            not bool(getattr(prop, "is_hq", False))
            and not bool(getattr(prop, "is_comm_tower", False))
            and not bool(getattr(prop, "is_lab", False))
        )

    def _phi_enemy_near_property(
        self, state: GameState, observer_seat: int, prop: Any
    ) -> bool:
        """Cheap contested-neutral proxy: enemy unit within Manhattan radius."""
        radius = int(getattr(self, "_phi_capture_context_radius", 3))
        if radius <= 0:
            return False
        enemy = 1 - int(observer_seat)
        pr, pc = int(prop.row), int(prop.col)
        for u in state.units[enemy]:
            if not u.is_alive:
                continue
            ur, uc = int(u.pos[0]), int(u.pos[1])
            if abs(ur - pr) + abs(uc - pc) <= radius:
                return True
        return False

    def _phi_capture_day(self, state: GameState) -> int:
        """Return the strategic day/turn index used for capture phase weighting."""
        try:
            return int(getattr(state, "turn", 0) or 0)
        except Exception:
            return 0

    def _phi_safe_neutral_phase_multiplier(self, state: GameState) -> float:
        """Phase multiplier for safe neutral expansion capture progress.

        This intentionally falls off by strategic day/turn rather than by
        micro-step count. The goal is to value early safe-city saturation without
        teaching the agent that late enemy-property conversion is unimportant.
        """
        if not bool(getattr(self, "_phi_capture_phase_weighting", False)):
            return 1.0
        day = self._phi_capture_day(state)
        if day <= int(self._phi_capture_opening_end_day):
            return float(self._phi_safe_neutral_opening_mult)
        if day <= int(self._phi_capture_early_mid_end_day):
            return float(self._phi_safe_neutral_early_mid_mult)
        if day <= int(self._phi_capture_mid_end_day):
            return float(self._phi_safe_neutral_mid_mult)
        if day <= int(self._phi_capture_late_end_day):
            return float(self._phi_safe_neutral_late_mult)
        return float(self._phi_safe_neutral_endgame_mult)

    def _phi_contested_neutral_phase_multiplier(self, state: GameState) -> float:
        """Mild phase multiplier for contested neutral capture progress."""
        if not bool(getattr(self, "_phi_capture_phase_weighting", False)):
            return 1.0
        day = self._phi_capture_day(state)
        if day <= int(self._phi_capture_early_mid_end_day):
            return float(self._phi_contested_neutral_opening_mult)
        if day <= int(self._phi_capture_late_end_day):
            return float(self._phi_contested_neutral_mid_mult)
        return float(self._phi_contested_neutral_late_mult)

    def _phi_property_capture_multiplier(
        self, state: GameState, prop: Any, observer_seat: int
    ) -> float:
        """Multiplier for partial capture progress in contextual-capture mode.

        The multiplier is component-specific. Safe neutral expansion can receive
        turn/phase weighting, contested neutrals receive mild turn weighting, and
        enemy/production/HQ properties keep their contextual value with no
        late-game falloff.
        """
        me = int(observer_seat)
        owner = getattr(prop, "owner", None)

        if bool(getattr(prop, "is_hq", False)):
            return float(self._phi_hq_capture_mult) if bool(getattr(self, "_phi_contextual_capture", False)) else 1.0
        if (
            bool(getattr(prop, "is_base", False))
            or bool(getattr(prop, "is_airport", False))
            or bool(getattr(prop, "is_port", False))
        ):
            return float(self._phi_production_capture_mult) if bool(getattr(self, "_phi_contextual_capture", False)) else 1.0
        if owner is None:
            contested = self._phi_enemy_near_property(state, me, prop)
            base = 1.0
            if bool(getattr(self, "_phi_contextual_capture", False)) and contested:
                base = float(self._phi_contested_neutral_capture_mult)
            if contested:
                return base * self._phi_contested_neutral_phase_multiplier(state)
            return base * self._phi_safe_neutral_phase_multiplier(state)
        if owner != me:
            return float(self._phi_enemy_property_capture_mult) if bool(getattr(self, "_phi_contextual_capture", False)) else 1.0
        return 1.0

    def _capture_progress_for_seat(self, state: GameState, observer_seat: int) -> float:
        """Compute weighted partial capture progress for ``observer_seat``."""
        me = int(observer_seat)
        cap_progress = 0.0
        for prop in state.properties:
            cp = prop.capture_points
            if cp >= 20:
                continue
            # Only count progress toward properties we don't own yet.
            if prop.owner != me:
                chip = 1.0 - cp / 20.0
                cap_progress += chip * self._phi_property_capture_multiplier(
                    state, prop, me
                )
        return cap_progress

    def _capture_progress_breakdown_for_seat(
        self, state: GameState, observer_seat: int
    ) -> dict[str, float]:
        """Diagnostic-only capture-progress breakdown.

        Do not put these values in ``reward_components``; that dict is summed to
        audit reward accounting. This is for ``info["phi_diagnostics"]`` and
        offline dashboards.
        """
        me = int(observer_seat)
        out = {
            "safe_neutral": 0.0,
            "contested_neutral": 0.0,
            "enemy_property": 0.0,
            "production_property": 0.0,
            "hq": 0.0,
            "weighted_total": 0.0,
        }
        for prop in state.properties:
            cp = prop.capture_points
            if cp >= 20 or prop.owner == me:
                continue
            chip = float(1.0 - cp / 20.0)
            mult = self._phi_property_capture_multiplier(state, prop, me)
            weighted = chip * mult
            out["weighted_total"] += weighted
            if bool(getattr(prop, "is_hq", False)):
                out["hq"] += weighted
            elif (
                bool(getattr(prop, "is_base", False))
                or bool(getattr(prop, "is_airport", False))
                or bool(getattr(prop, "is_port", False))
            ):
                out["production_property"] += weighted
            elif getattr(prop, "owner", None) is None:
                if self._phi_enemy_near_property(state, me, prop):
                    out["contested_neutral"] += weighted
                else:
                    out["safe_neutral"] += weighted
            else:
                out["enemy_property"] += weighted
        return out

    def _phi_diagnostics_for_seat(
        self, state: GameState, observer_seat: int
    ) -> dict[str, Any]:
        """Compact reward-shaping diagnostics from one seat's perspective."""
        me = int(observer_seat)
        en = 1 - me
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
        return {
            "seat": me,
            "phi_contextual_capture": bool(self._phi_contextual_capture),
            "army_value_diff": float(v_me - v_en),
            "property_diff": int(state.count_properties(me) - state.count_properties(en)),
            "income_property_diff": int(
                state.count_income_properties(me) - state.count_income_properties(en)
            ),
            "income_saturation": float(self._income_saturation(state, me, en)),
            "capture_progress_me": self._capture_progress_breakdown_for_seat(state, me),
            "capture_progress_enemy": self._capture_progress_breakdown_for_seat(state, en),
        }

    def _phi_learner_non_hq_property_cells(self, state: GameState) -> frozenset[tuple[int, int]]:
        """Board cells of capturable properties owned by the learner, excluding HQ.

        HQ is omitted so terminal ``hq_capture`` is not double-penalized alongside
        sparse / Φ terminal outcomes.
        """
        return self._seat_non_hq_property_cells(state, int(self._learner_seat))

    def _seat_non_hq_property_cells(
        self, state: GameState, observer_seat: int
    ) -> frozenset[tuple[int, int]]:
        """Board cells of non-HQ properties owned by ``observer_seat``."""
        me = int(observer_seat)
        return frozenset(
            (int(p.row), int(p.col))
            for p in state.properties
            if p.owner == me and not p.is_hq
        )

    def _phi_count_learner_props_lost_to_enemy(
        self,
        state: GameState,
        pre_cells: frozenset[tuple[int, int]],
        enemy_seat: int,
    ) -> int:
        """Count properties that were learner-owned in ``pre_cells`` and are now ``enemy_seat``."""
        en = int(enemy_seat)
        return sum(
            1
            for p in state.properties
            if (int(p.row), int(p.col)) in pre_cells and p.owner == en
        )

    def _count_props_lost_to_seat(
        self,
        state: GameState,
        pre_cells: frozenset[tuple[int, int]],
        new_owner_seat: int,
    ) -> int:
        """Count properties from ``pre_cells`` now owned by ``new_owner_seat``."""
        return sum(
            1
            for p in state.properties
            if (int(p.row), int(p.col)) in pre_cells and p.owner == int(new_owner_seat)
        )

    def _player_has_bases(self, state: GameState, seat: int) -> bool:
        """Check if player has any bases (factories, airports, ports) they own."""
        for prop in state.properties:
            if prop.owner == seat and (
                bool(getattr(prop, "is_base", False)) or
                bool(getattr(prop, "is_airport", False)) or
                bool(getattr(prop, "is_port", False))
            ):
                return True
        return False

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

    def _phi_enemy_kill_one_time_bonus_for_seat(
        self, pre_enemy_alive: dict[int, tuple[UnitType, int]], observer_seat: int
    ) -> float:
        """Extra Φ reward for kills by ``observer_seat``."""
        if not pre_enemy_alive or self.state is None:
            return 0.0
        enemy = 1 - int(observer_seat)
        post_ids = {u.unit_id for u in self.state.units[enemy] if u.is_alive}
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
        copy_obs = raw_copy not in ("0", "false", "no", "off")
        obs: dict[str, Any] = {
            "spatial": np.array(spatial_buf, copy=True, order="C") if copy_obs else spatial_buf,
            "scalars": np.array(scalars_buf, copy=True, order="C") if copy_obs else scalars_buf,
        }
        if self.state is None:
            cand_features = np.zeros((MAX_CANDIDATES, CANDIDATE_FEATURE_DIM), dtype=np.float32)
            cand_mask = np.zeros((MAX_CANDIDATES,), dtype=np.int8)
            cands = []
        else:
            obs_learner = int(observer) == int(self._learner_seat)
            active_learner = int(self.state.active_player) == int(self._learner_seat)
            if obs_learner and active_learner:
                cand_features, cand_mask_bool, cands = self._build_candidate_policy_tensors()
            else:
                cand_features, cand_mask_bool, cands = candidate_arrays(
                    self.state, max_candidates=MAX_CANDIDATES
                )
            cand_mask = cand_mask_bool.astype(np.int8, copy=False)
        self._candidate_features_cache = cand_features
        self._candidate_mask_cache = cand_mask.astype(bool, copy=False)
        self._candidate_list_cache = cands
        obs["candidate_features"] = (
            np.array(cand_features, copy=True, order="C") if copy_obs else cand_features
        )
        obs["candidate_mask"] = (
            np.array(cand_mask, copy=True, order="C") if copy_obs else cand_mask
        )
        return obs

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
            self._maybe_record_neutral_income_snapshot_days()
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

        self._maybe_record_neutral_income_snapshot_days()
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

    def _run_random_opponent(self, accumulated_reward: float) -> tuple[float, float]:
        """Run the non-learner seat — opening book (if configured) then uniform-random legal.

        Mirrors :meth:`_run_policy_opponent`'s book path so ``opponent_policy=None``
        (“cold random”) still plays book lines when :class:`~rl.opening_book.TwoSidedOpeningBookManager`
        is loaded; same as checkpoint opponent when the inner policy would mask-sample.

        Returns ``(accumulated_reward, opponent_engine_signed_sum)``: the second
        value is only the summed signed learner-frame engine rewards accumulated
        during opponent micro-steps (AUDIT_FLASK reward breakdown).
        """
        opp_engine_signed = 0.0
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
            _mout = self._flat_action_mask_buf if self._use_preallocated_buffers else None
            mask = _get_action_mask(self.state, out=_mout, legal=legal)
            book_idx = self._suggest_opening_book_for_active(mask)
            if book_idx is not None:
                opp_idx = int(book_idx)
            else:
                idxs = np.flatnonzero(mask)
                opp_idx = (
                    int(idxs[int(self.np_random.integers(0, len(idxs)))])
                    if idxs.size
                    else -1
                )
            action = _flat_to_action(opp_idx, self.state, legal=legal)
            if action is None:
                action = random.choice(legal)
            acting = int(self.state.active_player)
            phi_pb, phi_pb_act, _, phi_pa_m, phi_as_m = (
                self._phi_power_activation_and_attack_bonus(
                    action, acting, self.state, learner_frame=True
                )
            )
            self.state, r_opp, done_opp = self._engine_step_with_belief(action)
            dr = self._signed_engine_reward(r_opp, acting) + phi_pb
            opp_engine_signed += dr
            accumulated_reward += dr
            if phi_pb_act != 0.0 and hasattr(self, "_episode_reward_power_cumulative"):
                self._episode_reward_power_activation_cumulative[acting] += float(
                    phi_pa_m
                )
                self._episode_reward_attack_stats_cumulative[acting] += float(
                    phi_as_m
                )
                self._episode_reward_power_cumulative[acting] += float(
                    phi_pa_m + phi_as_m
                )
            self._phi_after_step_record_power_activations(action, acting)
            self._capture_frame(action=action)
            microsteps += 1
        if microsteps > self._max_p1_microsteps:
            self._max_p1_microsteps = microsteps
        return accumulated_reward, opp_engine_signed

    def _run_policy_opponent(self, accumulated_reward: float) -> tuple[float, float]:
        """Run the non-learner seat using the provided opponent policy callable.

        Returns ``(accumulated_reward, opponent_engine_signed_sum)`` —
        same convention as :meth:`_run_random_opponent`.
        """
        opp_engine_signed = 0.0
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
            _mout = self._flat_action_mask_buf if self._use_preallocated_buffers else None
            mask = _get_action_mask(self.state, out=_mout, legal=legal)
            book_idx = self._suggest_opening_book_for_active(mask)
            if book_idx is not None:
                opp_idx = int(book_idx)
            else:
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
            phi_pb, phi_pb_act, _, phi_pa_m, phi_as_m = (
                self._phi_power_activation_and_attack_bonus(
                    action, acting, self.state, learner_frame=True
                )
            )
            self.state, r_opp, done_opp = self._engine_step_with_belief(action)
            dr = self._signed_engine_reward(r_opp, acting) + phi_pb
            opp_engine_signed += dr
            accumulated_reward += dr
            if phi_pb_act != 0.0 and hasattr(self, "_episode_reward_power_cumulative"):
                self._episode_reward_power_activation_cumulative[acting] += float(
                    phi_pa_m
                )
                self._episode_reward_attack_stats_cumulative[acting] += float(
                    phi_as_m
                )
                self._episode_reward_power_cumulative[acting] += float(
                    phi_pa_m + phi_as_m
                )
            self._phi_after_step_record_power_activations(action, acting)
            self._capture_frame(action=action)
            microsteps += 1
        if microsteps > self._max_p1_microsteps:
            self._max_p1_microsteps = microsteps
        return accumulated_reward, opp_engine_signed

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

        eng_sparse_p0, eng_sparse_p1 = self._engine_terminal_sparse_pair_for_log_winner(
            log_winner
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
                if e.get("type") == "capture"
                and e.get("player") == 0
                and (cp := e.get("cp_remaining")) is not None
                and (cp == 0 or cp == 20)
            ),
            "captures_completed_p1": sum(
                1
                for e in self.state.game_log
                if e.get("type") == "capture"
                and e.get("player") == 1
                and (cp := e.get("cp_remaining")) is not None
                and (cp == 0 or cp == 20)
            ),
            "infantry_builds_p0": sum(
                1
                for e in self.state.game_log
                if e.get("type") == "build"
                and e.get("player") == 0
                and str(e.get("unit", "")).upper() == "INFANTRY"
            ),
            "turns": self.state.turn,
            "days": self.state.turn,
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
            "pairwise_zero_sum_reward": bool(
                getattr(self, "_pairwise_zero_sum_reward", False)
            ),
            # Canonical ±1/0 terminal outcome from ``log_winner`` (engine or
            # env step-cap synthetic tiebreak). Not Φ-shaped — see ``phi_reward_breakdown``.
            "engine_terminal_sparse_by_seat": [eng_sparse_p0, eng_sparse_p1],
            # Φ reward breakdown — cumulative per **engine seat** (P0 / P1).
            # Φ components use antisymmetric updates each learner step; power
            # bonuses credit the activating seat (including opponent micro-steps).
            # ``power_bonus`` = ``power_activation`` + ``attack_stats_advantage`` (backward compatible).
            # ``sparse_terminal`` / ``phi_day_cap_terminal_replacement``: only the
            # **Φ day-cap replacement** delta (antisymmetric), not HQ ±1.
            # ``engine_sparse_terminal``: copy of ``engine_terminal_sparse_by_seat`` for this seat.
            "phi_reward_breakdown": {
                "p0": {
                    "army": float(getattr(self, "_episode_reward_army_cumulative", [0.0, 0.0])[0]),
                    "property": float(getattr(self, "_episode_reward_property_cumulative", [0.0, 0.0])[0]),
                    "capture": float(getattr(self, "_episode_reward_capture_cumulative", [0.0, 0.0])[0]),
                    "income": float(getattr(self, "_episode_reward_income_cumulative", [0.0, 0.0])[0]),
                    "phi_potential_delta": float(getattr(self, "_episode_reward_phi_potential_delta_cumulative", [0.0, 0.0])[0]),
                    "kill_bonus": float(getattr(self, "_episode_reward_kill_bonus_cumulative", [0.0, 0.0])[0]),
                    "power_activation": float(getattr(self, "_episode_reward_power_activation_cumulative", [0.0, 0.0])[0]),
                    "attack_stats_advantage": float(getattr(self, "_episode_reward_attack_stats_cumulative", [0.0, 0.0])[0]),
                    "power_bonus": float(getattr(self, "_episode_reward_power_cumulative", [0.0, 0.0])[0]),
                    "designed_desires_checkpoint": float(getattr(self, "_episode_reward_designed_desires_cumulative", [0.0, 0.0])[0]),
                    "capture_interrupt": float(getattr(self, "_episode_reward_capture_interrupt_cumulative", [0.0, 0.0])[0]),
                    "enemy_property_loss": float(getattr(self, "_episode_reward_enemy_property_loss_cumulative", [0.0, 0.0])[0]),
                    "build_punishment": float(getattr(self, "_episode_reward_build_punishment_cumulative", [0.0, 0.0])[0]),
                    "sparse_terminal": float(getattr(self, "_episode_reward_sparse_cumulative", [0.0, 0.0])[0]),
                    "phi_day_cap_terminal_replacement": float(
                        getattr(self, "_episode_reward_sparse_cumulative", [0.0, 0.0])[0]
                    ),
                    "engine_sparse_terminal": float(eng_sparse_p0),
                },
                "p1": {
                    "army": float(getattr(self, "_episode_reward_army_cumulative", [0.0, 0.0])[1]),
                    "property": float(getattr(self, "_episode_reward_property_cumulative", [0.0, 0.0])[1]),
                    "capture": float(getattr(self, "_episode_reward_capture_cumulative", [0.0, 0.0])[1]),
                    "income": float(getattr(self, "_episode_reward_income_cumulative", [0.0, 0.0])[1]),
                    "phi_potential_delta": float(getattr(self, "_episode_reward_phi_potential_delta_cumulative", [0.0, 0.0])[1]),
                    "kill_bonus": float(getattr(self, "_episode_reward_kill_bonus_cumulative", [0.0, 0.0])[1]),
                    "power_activation": float(getattr(self, "_episode_reward_power_activation_cumulative", [0.0, 0.0])[1]),
                    "attack_stats_advantage": float(getattr(self, "_episode_reward_attack_stats_cumulative", [0.0, 0.0])[1]),
                    "power_bonus": float(getattr(self, "_episode_reward_power_cumulative", [0.0, 0.0])[1]),
                    "designed_desires_checkpoint": float(getattr(self, "_episode_reward_designed_desires_cumulative", [0.0, 0.0])[1]),
                    "capture_interrupt": float(getattr(self, "_episode_reward_capture_interrupt_cumulative", [0.0, 0.0])[1]),
                    "enemy_property_loss": float(getattr(self, "_episode_reward_enemy_property_loss_cumulative", [0.0, 0.0])[1]),
                    "build_punishment": float(getattr(self, "_episode_reward_build_punishment_cumulative", [0.0, 0.0])[1]),
                    "sparse_terminal": float(getattr(self, "_episode_reward_sparse_cumulative", [0.0, 0.0])[1]),
                    "phi_day_cap_terminal_replacement": float(
                        getattr(self, "_episode_reward_sparse_cumulative", [0.0, 0.0])[1]
                    ),
                    "engine_sparse_terminal": float(eng_sparse_p1),
                },
            },
            # Final Φ potential values at game end
            "phi_final": {
                "p0": float(self._compute_phi_for_seat(self.state, 0) if self.state is not None else 0.0),
                "p1": float(self._compute_phi_for_seat(self.state, 1) if self.state is not None else 0.0),
            },
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
            "property_pressure_end": self._property_pressure_snapshot(),
            "neutral_income_remaining_by_day_7": self._neutral_income_snapshot_by_day.get(
                7
            ),
            "neutral_income_remaining_by_day_9": self._neutral_income_snapshot_by_day.get(
                9
            ),
            "neutral_income_remaining_by_day_11": self._neutral_income_snapshot_by_day.get(
                11
            ),
            "neutral_income_remaining_by_day_13": self._neutral_income_snapshot_by_day.get(
                13
            ),
            "neutral_income_remaining_by_day_15": self._neutral_income_snapshot_by_day.get(
                15
            ),

            # Tier 1 (plan p0-capture-architecture-fix): visibility into
            # teacher-mix so we can verify it is firing and slice metrics by mix value.
            "learner_greedy_mix": float(getattr(self, "_learner_greedy_mix", 0.0)),
            "learner_teacher_overrides": int(getattr(self, "_learner_teacher_overrides", 0)),
            "phi_enemy_property_captures": int(
                getattr(self, "_phi_enemy_property_captures_ep", 0)
            ),
            # Φ COP/SCOP activation counts per engine seat (episode totals).
            "cop_activations_p0": int(getattr(self, "_episode_cop_by_seat", [0, 0])[0]),
            "scop_activations_p0": int(
                getattr(self, "_episode_scop_by_seat", [0, 0])[0]
            ),
            "cop_activations_p1": int(getattr(self, "_episode_cop_by_seat", [0, 0])[1]),
            "scop_activations_p1": int(
                getattr(self, "_episode_scop_by_seat", [0, 0])[1]
            ),
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
            # 1.10: tie_breaker_property_count — learner property lead when step-cap partial win (≥1).
            # 1.11: winner / win_condition filled from property tiebreak when truncated
            #       and engine left winner unset (env_step_cap_* reasons).
            # 1.13: alive_unit_count, army_value at episode end.
            # 1.14: phi_enemy_property_captures — episode sum of learner→enemy property flips (Φ penalty).
            # 1.15: neutral_income_remaining_by_day_{7,9,11,13,15} — neutral income tile counts
            #       at first engine step where ``turn`` equals each milestone (see GAME_LOG_NEUTRAL_INCOME_SNAPSHOT_DAYS).
            # 1.16: async_rollout_mode — async dual-gradient episodes only: mirror self-play vs hist checkpoint.
            # 1.17: cop_activations / scop_activations per seat (Φ power-use logging).
            # 1.18: engine_terminal_sparse_by_seat — canonical ±1/0 winner sparse;
            #       phi_reward_breakdown.*.engine_sparse_terminal / phi_day_cap_terminal_replacement.
            "terminated": bool(self.state.done),
            "truncated": bool(getattr(self, "_log_episode_truncated", False)),
            "truncation_reason": getattr(self, "_log_episode_truncation_reason", None),
            "tie_breaker_property_count": getattr(
                self, "_log_tie_breaker_property_count", None
            ),
            # 1.16: async_rollout_mode — dual-gradient mirror vs historical checkpoint episode.
            "log_schema_version": "1.18",
        }
        sk = getattr(self.state.spirit, "spirit_broken_kind", None)
        if sk is None:
            sk = getattr(self, "_spirit_broken_kind", None)
        if sk is not None:
            log_record["spirit_broken_kind"] = sk
        arm = getattr(self, "_async_rollout_mode", None)
        if arm is not None:
            log_record["async_rollout_mode"] = arm
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
                    "days": log_record.get("days"),
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
                "days": log_record.get("days"),
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
                "days": self.state.turn,
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
