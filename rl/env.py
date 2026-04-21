"""
Gymnasium environment for AWBW self-play.

The environment wraps the AWBW game engine and exposes:
  - observation_space: Dict {'spatial': Box, 'scalars': Box}
  - action_space: Discrete(ACTION_SPACE_SIZE)
  - action_masks(): bool array for MaskablePPO compatibility

The agent always controls player 0 (red seat: first mover on symmetric starts).
Player 1 (blue seat) is stepped automatically using either a provided opponent
policy or a random fallback.
"""
import json
import os
import random
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from threading import Lock

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from engine.game import GameState, make_initial_state, MAX_TURNS
from engine.map_loader import MapData, load_map
from engine.action import Action, ActionType, get_legal_actions
from engine.unit import UnitType, UNIT_STATS
from engine.terrain import get_terrain
from engine.combat import damage_range
from engine.belief import BeliefState

from rl.encoder import encode_state, N_SPATIAL_CHANNELS, N_SCALARS, GRID_SIZE
from rl.network import ACTION_SPACE_SIZE
from rl.paths import GAME_LOG_PATH, SLOW_GAMES_LOG_PATH
from server.write_watch_state import board_dict

ROOT = Path(__file__).parent.parent
POOL_PATH = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"

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

# In-process: threads must not interleave JSONL lines. Cross-process: use SQLite (see _append_game_log_line).
_log_lock = Lock()
# When SESSION_GAME_COUNTER_DB_ENV is unset, count completed games in this process only (tests, ad-hoc env use).
_local_session_game_count = 0


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
            full = {"game_id": game_id, **record}
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
        full = {"game_id": _local_session_game_count, **record}
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
_BUILD_OFFSET = 10_000
# REPAIR (Black Boat): one index per target tile; kept below _BUILD_OFFSET.
_REPAIR_OFFSET = 3500
# Must match UnitType enum cardinality (27 types).
_N_UNIT_TYPES = len(UnitType)


def _action_to_flat(action: Action) -> int:
    """Encode an Action to a flat integer index."""
    at = action.action_type

    if at == ActionType.END_TURN:
        return 0
    if at == ActionType.ACTIVATE_COP:
        return 1
    if at == ActionType.ACTIVATE_SCOP:
        return 2

    if at == ActionType.SELECT_UNIT:
        # SELECT: unit tile; MOVE stage also uses SELECT_UNIT with move_pos set (engine).
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


def _flat_to_action(flat_idx: int, state: GameState) -> Optional[Action]:
    """
    Decode a flat integer back to a legal Action for the current state.
    Returns None if the index does not correspond to any legal action.
    """
    legal = get_legal_actions(state)
    for a in legal:
        if _action_to_flat(a) == flat_idx:
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


def _get_action_mask(state: GameState) -> np.ndarray:
    """Return a bool mask over [0, ACTION_SPACE_SIZE) indicating legal actions."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    for action in get_legal_actions(state):
        idx = _action_to_flat(action)
        if 0 <= idx < ACTION_SPACE_SIZE:
            mask[idx] = True
    return mask


def _strip_non_infantry_builds(mask: np.ndarray, state: GameState) -> None:
    """In-place: clear BUILD entries except ``INFANTRY`` (bootstrap curriculum)."""
    for action in get_legal_actions(state):
        if action.action_type == ActionType.BUILD and action.unit_type != UnitType.INFANTRY:
            idx = _action_to_flat(action)
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
        ``step`` calls have completed (independent of engine ``MAX_TURNS`` /
        calendar days). Useful for playoff / eval so games cannot run unbounded.
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

        self.state: Optional[GameState] = None
        self._episode_info: dict[str, Any] = {}
        self._p1_truncated_mid_turn: bool = False

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

    # ── Map helpers ───────────────────────────────────────────────────────────

    def _load_map(self, map_id: int) -> MapData:
        if map_id not in self._map_cache:
            self._map_cache[map_id] = load_map(map_id, POOL_PATH, MAPS_DIR)
        return self._map_cache[map_id]

    def _sample_config(self) -> tuple[int, str, int, int, str]:
        """
        Return (map_id, tier_name, p0_co_id, p1_co_id, map_name).

        With ``curriculum_broad_prob > 0``, sometimes delegates to full random
        sampling for mixture training. Fixed ``tier_name`` / ``co_p0`` / ``co_p1``
        implement narrow curriculum when broad sampling is not taken.
        """
        return sample_training_matchup(
            self._sample_map_pool,
            co_p0=self.co_p0,
            co_p1=self.co_p1,
            tier_name=self.tier_name,
            curriculum_broad_prob=self.curriculum_broad_prob,
            rng=None,
        )

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)

        map_id, tier_name, p0_co, p1_co, map_name = self._sample_config()
        map_data = self._load_map(map_id)

        self.state = make_initial_state(
            map_data,
            p0_co,
            p1_co,
            starting_funds=0,
            tier_name=tier_name,
        )
        # Who opens (engine seat 0 or 1) per make_initial_state predeploy rule; see engine/game.py.
        self._opening_player = int(self.state.active_player)

        self._episode_info = {
            "map_id": map_id,
            "map_name": map_name,
            "tier": tier_name,
            "p0_co": p0_co,
            "p1_co": p1_co,
            "episode_started_at": time.time(),  # Track episode start time
        }
        if self.curriculum_tag:
            self._episode_info["curriculum_tag"] = self.curriculum_tag

        # Per-episode diagnostic counters (see _log_finished_game).
        # These are O(1) per step and help flag degenerate episodes —
        # especially mask/decode divergence (invalid_action_count) and
        # P1-loop heaviness (max_p1_microsteps).
        self._p0_env_steps: int = 0
        self._invalid_action_count: int = 0
        self._max_p1_microsteps: int = 0
        self._p1_truncated_mid_turn = False
        # Tier 1 (plan p0-capture-architecture-fix): per-episode count of
        # learner actions that were overridden by the capture-greedy teacher.
        # Surfaced in game_log.jsonl so we can verify the teacher is firing.
        self._learner_teacher_overrides: int = 0
        # Snapshot opponent reload count at episode start so we can
        # report per-episode reloads in the log record.
        self._opponent_reloads_at_start: int = int(
            getattr(self.opponent_policy, "reload_count", 0) or 0
        )
        self._first_p0_capture_step: int | None = None

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

        # If P1 opens, run the opponent before the learner's first step — same contract as
        # server/play_human.new_session: observations must always be on P0's clock.
        # (After _max_p1_microsteps init — _run_random_opponent updates that counter.)
        if not self.state.done and self.state.active_player == 1:
            if self.opponent_policy is not None:
                self._run_policy_opponent(0.0)
            else:
                self._run_random_opponent(0.0)

        # Reset replay buffer and record the starting position.
        self._replay_frames = []
        self._capture_frame(action=None)

        return self._get_obs(), dict(self._episode_info)

    def step(
        self, action_idx: int
    ) -> tuple[dict, float, bool, bool, dict]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

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
            mask = _get_action_mask(self.state)
            action_idx = pick_capture_greedy_flat(self.state, mask)
            self._learner_teacher_overrides += 1

        # ── Decode & apply player-0 action ────────────────────────────────────
        action = _flat_to_action(action_idx, self.state)
        if action is None:
            self._invalid_action_count += 1
            obs = self._get_obs()
            return obs, -0.1, False, False, {"invalid_action": True}

        self.state, reward, done = self._engine_step_with_belief(action)
        self._capture_frame(action=action)

        if action.action_type == ActionType.CAPTURE and self._first_p0_capture_step is None:
            self._first_p0_capture_step = self._p0_env_steps

        # Dense shaping: property advantage + unit value differential.
        # Coefficients are small relative to terminal ±1.0 but large enough to
        # provide a meaningful learning signal across the long horizon.
        # Asymmetric penalty (plan p0-capture-architecture-fix): the prior
        # symmetric (p0 - p1) * 0.005 manufactured a "punished for existing"
        # gradient when the opponent bootstrapped with capture-greedy and the
        # learner had no teacher. Penalty is now 5x softer than reward to
        # break the dead-curriculum trap without removing the diff signal.
        if not done:
            p0_props = self.state.count_properties(0)
            p1_props = self.state.count_properties(1)
            diff = p0_props - p1_props
            if diff >= 0:
                reward += diff * 0.005
            else:
                reward += diff * 0.001

            p0_val = sum(
                UNIT_STATS[u.unit_type].cost * u.hp / 100
                for u in self.state.units[0]
                if u.is_alive
            )
            p1_val = sum(
                UNIT_STATS[u.unit_type].cost * u.hp / 100
                for u in self.state.units[1]
                if u.is_alive
            )
            reward += (p0_val - p1_val) * 2e-6

        # ── Auto-step opponent ────────────────────────────────────────────────
        if not done and self.state.active_player == 1:
            if self.opponent_policy is not None:
                reward = self._run_policy_opponent(reward)
            else:
                reward = self._run_random_opponent(reward)

        obs = self._get_obs()
        terminated = bool(self.state.done)
        truncated = bool(self._p1_truncated_mid_turn)
        self._p1_truncated_mid_turn = False
        if (
            self._max_env_steps is not None
            and self._p0_env_steps >= self._max_env_steps
            and not self.state.done
        ):
            truncated = True
        info = {
            **self._episode_info,
            "turn": self.state.turn,
            "winner": self.state.winner,
            "truncated": truncated,
        }

        if terminated:
            raw_inc = os.environ.get(INCOME_TERM_COEF_ENV, "0").strip()
            if raw_inc:
                try:
                    coef = float(raw_inc)
                    if coef != 0.0:
                        cap_lim = max(1, self.state.map_data.cap_limit)
                        inc0 = self.state.count_income_properties(0)
                        inc1 = self.state.count_income_properties(1)
                        reward += coef * (inc0 - inc1) / float(cap_lim)
                except ValueError:
                    pass

        if not self.state.done and not truncated:
            raw_tc = os.environ.get(TIME_COST_ENV, "0").strip()
            if raw_tc:
                try:
                    reward -= float(raw_tc)
                except ValueError:
                    pass

        # Log finished game (Phase A requirement)
        if terminated:
            self._log_finished_game()

        if self.render_mode == "ansi":
            print(self.state.render_ascii())

        return obs, float(reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return valid-action bool mask. Required by MaskablePPO wrappers."""
        if self.state is None:
            return np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask = _get_action_mask(self.state)
        flag = os.environ.get(BUILD_MASK_INFANTRY_ONLY_ENV, "").strip().lower()
        if flag in ("1", "true", "yes", "on"):
            _strip_non_infantry_builds(mask, self.state)
        return mask

    def render(self) -> str | None:
        if self.render_mode == "ansi" and self.state is not None:
            return self.state.render_ascii()
        return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_obs(self, observer: int = 0) -> dict:
        """Render observation from ``observer``'s perspective, honouring the
        HP belief overlay so enemy units leak only bucket + formula-narrowed
        interval information — never exact HP.
        """
        belief = self._beliefs.get(observer) if hasattr(self, "_beliefs") else None
        spatial, scalars = encode_state(
            self.state, observer=observer, belief=belief,
        )
        return {"spatial": spatial, "scalars": scalars}

    # -- Belief bookkeeping ----------------------------------------------------
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
        beliefs = list(self._beliefs.values())

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

    def _run_random_opponent(self, accumulated_reward: float) -> float:
        """Run player 1's turn using uniform-random legal actions."""
        microsteps = 0
        cap = self._max_p1_microsteps_cap
        while not self.state.done and self.state.active_player == 1:
            if cap is not None and microsteps >= cap:
                self._p1_truncated_mid_turn = True
                break
            legal = get_legal_actions(self.state)
            if not legal:
                break
            action = random.choice(legal)
            self.state, r_opp, done_opp = self._engine_step_with_belief(action)
            if done_opp:
                # Engine reward is from the acting player (P1); training is P0-only.
                accumulated_reward -= r_opp
            self._capture_frame(action=action)
            microsteps += 1
        if microsteps > self._max_p1_microsteps:
            self._max_p1_microsteps = microsteps
        return accumulated_reward

    def _run_policy_opponent(self, accumulated_reward: float) -> float:
        """Run player 1's turn using the provided opponent policy callable."""
        microsteps = 0
        cap = self._max_p1_microsteps_cap
        while not self.state.done and self.state.active_player == 1:
            if cap is not None and microsteps >= cap:
                self._p1_truncated_mid_turn = True
                break
            # Opponent sees the board from P1's seat, with P1's belief overlay —
            # not P0's. Before the HP belief work this leaked exact P0 HP into
            # the opponent model on the blue seat.
            obs = self._get_obs(observer=1)
            mask = _get_action_mask(self.state)
            try:
                opp_idx = int(self.opponent_policy(obs, mask))
            except Exception:
                opp_idx = -1

            action = _flat_to_action(opp_idx, self.state)
            if action is None:
                # Policy returned illegal index — fall back to random
                legal = get_legal_actions(self.state)
                if not legal:
                    break
                action = random.choice(legal)

            self.state, r_opp, done_opp = self._engine_step_with_belief(action)
            if done_opp:
                # Engine reward is from the acting player (P1); training is P0-only.
                accumulated_reward -= r_opp
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
        timestamp_iso = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

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
            "first_p0_capture_p0_step": getattr(self, "_first_p0_capture_step", None),
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
            "win_condition": self.state.win_reason,
            "losses_hp": self.state.losses_hp.copy(),

            # Outcome & matchup (CO names human-readable; IDs kept for analysis tools)
            "winner": self.state.winner,
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

            # Losses
            "losses_units": self.state.losses_units.copy(),

            # Length & scale
            "n_actions": n_actions,

            # Training context
            "agent_plays": 0,  # Agent always controls player 0
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
            "log_schema_version": "1.5",
        }
        if self.curriculum_tag:
            log_record["curriculum_tag"] = self.curriculum_tag

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
                "winner": self.state.winner,
                "win_condition": self.state.win_reason,
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
