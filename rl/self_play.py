"""
Self-play training loop for AWBW DRL agent.

Usage
-----
# Start/resume training (runs until Ctrl+C by default):
    python -m rl.self_play

# Watch a single random game without training:
    python -m rl.self_play watch [map_id]

Training loop
-------------
- MaskablePPO (sb3_contrib) against rotating historical checkpoints
- Logs every completed game to logs/game_log.jsonl
- Saves a checkpoint every `save_every` env steps
- Rotating opponent pool: keeps the last `checkpoint_pool_size` checkpoints (in-memory tail).
- On-disk `checkpoint_*.zip` retention: `checkpoint_zip_cap` (default 100) prunes oldest (by mtime).
"""
from __future__ import annotations

import atexit
import json
import os
import random
import tempfile
import time
import weakref
from pathlib import Path
from typing import Any, Callable, Optional

from engine.action import Action, ActionStage, ActionType, get_legal_actions
from engine.game import GameState
from engine.unit import UnitType

from rl.env import SESSION_GAME_COUNTER_DB_ENV
from rl.paths import GAME_LOG_PATH, LOGS_DIR

import numpy as np

ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
POOL_PATH = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_map_pool() -> list[dict]:
    with open(POOL_PATH) as f:
        return json.load(f)


def log_game(
    *,
    map_id: int,
    tier: str,
    p0_co: int,
    p1_co: int,
    winner: int,
    turns: int,
    funds_end: list[int],
    n_actions: int,
    opening_player: int | None = None,
) -> None:
    """Append a game result record to logs/game_log.jsonl."""
    GAME_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "map_id": map_id,
        "tier": tier,
        "p0_co": p0_co,
        "p1_co": p1_co,
        "winner": winner,
        "turns": turns,
        "funds_end": funds_end,
        "n_actions": n_actions,
        "timestamp": time.time(),
        "log_schema_version": "1.5",
        "opening_player": opening_player,
    }
    with open(GAME_LOG_PATH, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def update_elo(
    ratings: dict[str, float],
    winner_key: str,
    loser_key: str,
    k: float = 32.0,
) -> dict[str, float]:
    """In-place ELO update; returns the same dict for convenience."""
    ra = ratings.setdefault(winner_key, 1200.0)
    rb = ratings.setdefault(loser_key, 1200.0)
    ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
    ratings[winner_key] = ra + k * (1.0 - ea)
    ratings[loser_key] = rb + k * (0.0 - (1.0 - ea))
    return ratings


# ── Trainer ───────────────────────────────────────────────────────────────────

def _mask_fn(env: "AWBWEnv") -> np.ndarray:  # type: ignore[name-defined]
    """Module-level mask function — picklable for SubprocVecEnv workers."""
    return env.action_masks()


def _atomic_model_save(model, dest_no_ext: str | os.PathLike) -> None:
    """
    Save an SB3 model to ``<dest_no_ext>.zip`` atomically.

    SB3's ``model.save(path)`` writes ``path + ".zip"``. Over SMB / shared
    mounts, an opponent process (`_CheckpointOpponent`) can race a partial
    write and load a truncated zip. We write to ``<dest>.tmp.zip`` first, then
    ``os.replace`` it onto the final filename — atomic on the same volume.
    """
    dest = Path(dest_no_ext)
    final_zip = dest.with_suffix(".zip")
    tmp_no_ext = dest.with_name(dest.name + ".tmp")
    tmp_zip = tmp_no_ext.with_suffix(".zip")
    try:
        if tmp_zip.exists():
            try:
                tmp_zip.unlink()
            except OSError:
                pass
        model.save(str(tmp_no_ext))
        os.replace(str(tmp_zip), str(final_zip))
    finally:
        # Defensive cleanup: if save raised mid-write leave nothing dangling.
        if tmp_zip.exists():
            try:
                tmp_zip.unlink()
            except OSError:
                pass


def _nearest_neutral_income_dist(state: GameState, pos: tuple[int, int]) -> int:
    r, c = pos
    best = 10_000
    for prop in state.properties:
        if prop.owner is not None or prop.is_comm_tower or prop.is_lab:
            continue
        d = abs(prop.row - r) + abs(prop.col - c)
        best = min(best, d)
    return best


def _greedy_action_score(state: GameState, a: Action) -> float:
    """Higher = better for capture-focused bootstrapping."""
    at = a.action_type
    if at == ActionType.CAPTURE:
        return 1_000.0
    if at == ActionType.BUILD:
        if a.unit_type == UnitType.INFANTRY:
            return 800.0
        if a.unit_type == UnitType.MECH:
            return 750.0
        return 600.0
    if at == ActionType.ATTACK:
        return 500.0
    if at in (ActionType.ACTIVATE_SCOP, ActionType.ACTIVATE_COP):
        return 400.0
    if at == ActionType.UNLOAD or at == ActionType.LOAD or at == ActionType.JOIN:
        return 120.0
    if at == ActionType.REPAIR:
        return 110.0
    if at in (ActionType.WAIT, ActionType.DIVE_HIDE):
        return 80.0
    if at == ActionType.SELECT_UNIT:
        if state.action_stage == ActionStage.SELECT:
            u = state.get_unit_at(*a.unit_pos) if a.unit_pos else None
            if u is not None and u.player == state.active_player:
                if u.unit_type in (UnitType.INFANTRY, UnitType.MECH):
                    d = _nearest_neutral_income_dist(state, a.unit_pos)
                    return 300.0 - float(min(d, 200))
            return 100.0
        if state.action_stage == ActionStage.MOVE and a.move_pos is not None:
            d = _nearest_neutral_income_dist(state, a.move_pos)
            return 250.0 - float(min(d, 199))
        return 120.0
    if at == ActionType.END_TURN:
        return 10.0
    return 50.0


def pick_capture_greedy_flat(state: GameState, mask: np.ndarray) -> int:
    """Pick a masked flat action with capture-first heuristics (P1 cold-start)."""
    from rl.env import _action_to_flat
    from rl.network import ACTION_SPACE_SIZE

    legal = get_legal_actions(state)
    best_score = -1e18
    best: list[int] = []
    for act in legal:
        idx = _action_to_flat(act)
        if not (0 <= idx < ACTION_SPACE_SIZE) or not mask[idx]:
            continue
        sc = _greedy_action_score(state, act)
        if sc > best_score:
            best_score = sc
            best = [idx]
        elif sc == best_score:
            best.append(idx)
    if not best:
        legal_idx = np.where(mask)[0]
        return int(np.random.choice(legal_idx)) if len(legal_idx) > 0 else 0
    return int(random.choice(best))


_VALID_COLD_OPPONENTS = ("random", "greedy_capture", "end_turn")


def _passive_cold_action(state: GameState, mask: np.ndarray) -> int | None:
    """Pick an action that keeps the active player passive this microstep.

    Implements the "true punching bag" semantics for ``cold_opponent='end_turn'``
    when the engine refuses END_TURN (mandatory unit activation). The walk is:

    - At SELECT: pick any unmoved unit via SELECT_UNIT (forced; END_TURN was
      already filtered by the caller via mask[0]==False).
    - At MOVE  : pick the destination equal to the selected unit's current
      tile (a 0-cost stay-in-place move; always reachable).
    - At ACTION: pick WAIT (flat ``_WAIT_IDX``) — ends the action chain
      without attacking, capturing, or building.

    Returns ``None`` if no defensible passive action could be chosen, in
    which case the caller falls back to a random legal action.
    """
    from rl.env import _action_to_flat, _WAIT_IDX
    from rl.network import ACTION_SPACE_SIZE

    stage = state.action_stage
    if stage == ActionStage.SELECT:
        # Some unit must move; pick the first SELECT_UNIT in legal order.
        for act in get_legal_actions(state):
            if act.action_type == ActionType.SELECT_UNIT and act.move_pos is None:
                idx = _action_to_flat(act)
                if 0 <= idx < ACTION_SPACE_SIZE and bool(mask[idx]):
                    return idx
        return None

    if stage == ActionStage.MOVE:
        unit = state.selected_unit
        if unit is None:
            return None
        # Stay-in-place move: SELECT_UNIT with move_pos = unit.pos.
        for act in get_legal_actions(state):
            if (
                act.action_type == ActionType.SELECT_UNIT
                and act.move_pos == unit.pos
            ):
                idx = _action_to_flat(act)
                if 0 <= idx < ACTION_SPACE_SIZE and bool(mask[idx]):
                    return idx
        # Stay-in-place not in legal moves (rare); fall through to None
        # so caller picks random legal MOVE.
        return None

    if stage == ActionStage.ACTION:
        # Prefer WAIT to terminate without attacking/capturing/building.
        if 0 <= _WAIT_IDX < ACTION_SPACE_SIZE and bool(mask[_WAIT_IDX]):
            return int(_WAIT_IDX)
        return None

    return None


class _CheckpointOpponent:
    """
    Picklable opponent policy that loads random historical checkpoints.

    Lazily loads a checkpoint on first call and refreshes every
    `refresh_every` calls so the opponent pool rotates naturally as
    new checkpoints are written by the trainer.

    Cold-start behaviour (no checkpoints yet) is selectable via
    ``cold_opponent``:

    - ``"random"`` (default, post fix): uniform random legal action. The
      weakest reasonable opponent — gives the learner a chance to discover
      capture before facing a teacher.
    - ``"greedy_capture"``: pre-fix legacy default; uses
      :func:`pick_capture_greedy_flat`. Strong, asymmetric — manufactures
      a property-skew gradient if the learner has no matching teacher.
    - ``"end_turn"``: punching-bag opponent; picks END_TURN whenever
      legal so P0 has the entire map to itself. Used for the smoke gate
      in plan p0-capture-architecture-fix.

    With ``pool_from_fleet``, also samples fleet-wide ``checkpoint_*.zip`` under
    ``<fleet_opponent_root>`` (top-level plus ``pool/*/``). For auxiliary pool
    trainers, pass ``fleet_opponent_root`` = shared ``checkpoints/`` (e.g.
    ``Z:\\checkpoints``) so opponents include main line + every pool export.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        refresh_every: int = 500,
        opponent_mix: float = 0.0,
        pool_from_fleet: bool = False,
        cold_opponent: str = "random",
        fleet_opponent_root: Optional[str] = None,
    ) -> None:
        self._dir = checkpoint_dir
        self._fleet_opponent_root = fleet_opponent_root
        self._refresh_every = refresh_every
        self._opponent_mix = max(0.0, min(1.0, float(opponent_mix)))
        self._pool_from_fleet = bool(pool_from_fleet)
        cold = str(cold_opponent or "random").strip().lower()
        if cold not in _VALID_COLD_OPPONENTS:
            raise ValueError(
                f"cold_opponent must be one of {_VALID_COLD_OPPONENTS}; got {cold_opponent!r}"
            )
        self._cold_opponent = cold
        self._model = None
        self._n_calls = 0
        # Exposed so AWBWEnv can count per-episode reloads. Incremented
        # only on a successful checkpoint load (not random fallback).
        self.reload_count: int = 0
        self._env_ref: Optional[Callable[[], Any]] = None

    def attach_env(self, env: object) -> None:
        """Weak reference so greedy fallback can read ``env.state``."""
        self._env_ref = weakref.ref(env)

    def _load_random(self) -> None:
        import glob as _glob

        ckpts = sorted(_glob.glob(os.path.join(self._dir, "checkpoint_*.zip")))
        if self._pool_from_fleet:
            from rl.fleet_env import iter_fleet_opponent_checkpoint_zips

            root = self._fleet_opponent_root or self._dir
            ckpts = sorted(set(ckpts + iter_fleet_opponent_checkpoint_zips(Path(root))))
        if not ckpts:
            self._model = None
            return
        path = random.choice(ckpts)
        try:
            from rl.ckpt_compat import load_maskable_ppo_compat

            # Load with minimal buffer dims so _setup_model() allocates a tiny
            # rollout buffer (~KB) rather than replicating the learner's full
            # n_steps × n_envs × obs_shape tensor (~GB). Policy weights are
            # preserved; only inference (predict) is used from this model.
            self._model = load_maskable_ppo_compat(
                path,
                device="cpu",
                verbose=0,
                n_envs=1,
                n_steps=1,
                batch_size=1,
            )
            self.reload_count += 1
        except MemoryError as exc:
            print(f"[opponent] OOM loading {path}: {exc} — using random (retry next refresh)")
            self._model = None
        except Exception as exc:
            print(f"[opponent] Could not load {path}: {exc} — using random")
            self._model = None

    def mode(self) -> str:
        """Return the opponent mode label for game_log records."""
        if self._model is None:
            return f"cold_{self._cold_opponent}"
        if self._opponent_mix > 0.0:
            return "mixed"
        return "checkpoint"

    def _cold_action(self, mask: np.ndarray) -> int:
        """Pick an action under the configured cold-opponent policy."""
        if self._cold_opponent == "end_turn":
            # END_TURN flat=0 is only legal once every friendly unit has
            # moved this turn (engine forces unit activity; see
            # _get_select_actions). When END_TURN is illegal, walk every
            # unit through SELECT -> stay-in-place MOVE -> WAIT so the
            # turn ends without doing anything substantive — the true
            # punching-bag semantics.
            if len(mask) > 0 and bool(mask[0]):
                return 0
            env = self._env_ref() if self._env_ref is not None else None
            st: GameState | None = getattr(env, "state", None) if env is not None else None
            if st is not None:
                idx = _passive_cold_action(st, mask)
                if idx is not None:
                    return idx
            legal = np.where(mask)[0]
            return int(np.random.choice(legal)) if len(legal) > 0 else 0
        if self._cold_opponent == "greedy_capture":
            env = self._env_ref() if self._env_ref is not None else None
            st: GameState | None = getattr(env, "state", None) if env is not None else None
            if st is not None:
                return pick_capture_greedy_flat(st, mask)
            legal = np.where(mask)[0]
            return int(np.random.choice(legal)) if len(legal) > 0 else 0
        # default: random legal action
        legal = np.where(mask)[0]
        return int(np.random.choice(legal)) if len(legal) > 0 else 0

    def __call__(self, obs: dict, mask: np.ndarray) -> int:
        if self._n_calls % self._refresh_every == 0:
            self._load_random()
        self._n_calls += 1

        # When opponent_mix is set, sometimes substitute the *cold opponent*
        # (capture-greedy historically; now configurable) for the loaded
        # checkpoint to inject a teacher signal mid-training.
        if self._model is not None and self._opponent_mix > 0.0:
            if random.random() < self._opponent_mix:
                return self._cold_action(mask)

        if self._model is None:
            return self._cold_action(mask)

        action, _ = self._model.predict(obs, action_masks=mask, deterministic=False)
        return int(action)


def _make_env_factory(
    map_pool: list[dict],
    checkpoint_dir: str,
    co_p0: int | None = None,
    co_p1: int | None = None,
    tier_name: str | None = None,
    curriculum_broad_prob: float = 0.0,
    curriculum_tag: str | None = None,
    opponent_mix: float = 0.0,
    pool_from_fleet: bool = False,
    cold_opponent: str = "random",
    fleet_opponent_root: str | None = None,
) -> Callable:
    """Return a picklable env factory for SubprocVecEnv."""
    def _init():
        from rl.env import AWBWEnv
        from sb3_contrib.common.wrappers import ActionMasker  # type: ignore[import]
        opponent = _CheckpointOpponent(
            checkpoint_dir,
            opponent_mix=opponent_mix,
            pool_from_fleet=pool_from_fleet,
            cold_opponent=cold_opponent,
            fleet_opponent_root=fleet_opponent_root,
        )
        env = AWBWEnv(
            map_pool=map_pool,
            opponent_policy=opponent,
            render_mode=None,
            co_p0=co_p0,
            co_p1=co_p1,
            tier_name=tier_name,
            curriculum_broad_prob=curriculum_broad_prob,
            curriculum_tag=curriculum_tag,
        )
        opponent.attach_env(env)
        return ActionMasker(env, _mask_fn)
    return _init


class _EpisodeDiagnosticsCallback:
    """
    Lightweight SB3 callback that surfaces episode-length signals to
    TensorBoard so you can tell at a glance whether slowdowns are
    coming from env simulation (episodes-per-rollout dropping, env
    steps per episode spiking) vs GPU/learning work.

    Scalars published on rollout end:
      - diag/episodes_per_rollout
      - diag/ep_len_mean
      - diag/ep_len_max
      - diag/ep_len_min

    Built as a class only if sb3 is importable at runtime; the factory
    function :func:`_build_diagnostics_callback` handles the import so
    this module stays importable in environments without sb3.
    """


def _build_diagnostics_callback():
    """Return a `BaseCallback` instance or None if sb3 is unavailable."""
    try:
        from stable_baselines3.common.callbacks import BaseCallback  # type: ignore[import]
    except ImportError:
        return None

    class _Cb(BaseCallback):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            # Per-env running step count (reset on `done`). Sized lazily
            # on first _on_step since we don't know n_envs at __init__.
            self._cur_lens: list[int] = []
            self._finished_lens: list[int] = []
            # Phase 0a.1 (FPS campaign): env-collect vs PPO-update wall split.
            # SB3 calls _on_rollout_start before each rollout, _on_rollout_end
            # after the rollout buffer is full, then runs the PPO update before
            # the next _on_rollout_start. So:
            #   env_collect_s = rollout_end_t - rollout_start_t  (this rollout)
            #   ppo_update_s  = rollout_start_t - prev_rollout_end_t  (prev update)
            # The first rollout has no preceding update; ppo_update_s is omitted.
            self._t_rollout_start: float | None = None
            self._t_prev_rollout_end: float | None = None
            self._last_ppo_update_s: float | None = None
            # Per-env step count for env_steps_per_s_collect; sized lazily.
            self._steps_in_rollout: int = 0

        def _on_step(self) -> bool:
            dones = self.locals.get("dones")
            if dones is None:
                return True
            n_envs = len(dones)
            if len(self._cur_lens) != n_envs:
                self._cur_lens = [0] * n_envs
            self._steps_in_rollout += n_envs
            for i in range(n_envs):
                self._cur_lens[i] += 1
                if dones[i]:
                    self._finished_lens.append(self._cur_lens[i])
                    self._cur_lens[i] = 0
            return True

        def _on_rollout_start(self) -> None:
            now = time.perf_counter()
            # If we have a previous rollout end, the gap until now is the PPO
            # update wall for that rollout. Stash it so _on_rollout_end can
            # bundle env_collect + ppo_update into a single TB flush.
            if self._t_prev_rollout_end is not None:
                self._last_ppo_update_s = max(0.0, now - self._t_prev_rollout_end)
            self._t_rollout_start = now
            self._steps_in_rollout = 0

        def _on_rollout_end(self) -> None:
            now = time.perf_counter()
            n = len(self._finished_lens)
            self.logger.record("diag/episodes_per_rollout", n)
            if n:
                self.logger.record(
                    "diag/ep_len_mean", sum(self._finished_lens) / n
                )
                self.logger.record("diag/ep_len_max", max(self._finished_lens))
                self.logger.record("diag/ep_len_min", min(self._finished_lens))
            self._finished_lens.clear()

            # Phase 0a.1: env-collect wall + steps/s for this rollout, plus
            # the PPO-update wall from the prior rollout (derived in
            # _on_rollout_start). env_steps_per_s_total uses (this rollout's
            # collect + previous rollout's update) — that pair represents one
            # full env+learn cycle, which is the steady-state FPS the user
            # actually feels.
            if self._t_rollout_start is not None:
                env_collect_s = max(0.0, now - self._t_rollout_start)
                self.logger.record("diag/env_collect_s", env_collect_s)
                if env_collect_s > 0 and self._steps_in_rollout > 0:
                    self.logger.record(
                        "diag/env_steps_per_s_collect",
                        self._steps_in_rollout / env_collect_s,
                    )
                if self._last_ppo_update_s is not None:
                    self.logger.record("diag/ppo_update_s", self._last_ppo_update_s)
                    cycle_s = env_collect_s + self._last_ppo_update_s
                    if cycle_s > 0 and self._steps_in_rollout > 0:
                        self.logger.record(
                            "diag/env_steps_per_s_total",
                            self._steps_in_rollout / cycle_s,
                        )
            self._t_prev_rollout_end = now
            self._last_ppo_update_s = None

    return _Cb()


class SelfPlayTrainer:
    """
    Manages the PPO self-play training loop with checkpoint rotation.

    Parameters
    ----------
    total_timesteps : int | None
        Total env steps across all workers. ``None`` runs until KeyboardInterrupt (Ctrl+C).
    n_envs : int
        Number of parallel SubprocVecEnv workers (default 4).
        Each worker is a separate Python process that runs env simulation and
        opponent inference on CPU.  More workers raise throughput but cost
        ~2-3 GB host RAM each.

        Step-sync note: SubprocVecEnv is synchronous — every call to
        ``VecEnv.step()`` waits for the slowest worker before returning.
        Within a rollout individual episodes reset independently (no waiting
        for every env to finish a game), but per-timestep progress is still
        gated by the laggard.  True per-worker async rollouts would require a
        custom ``VecEnv`` + rollout collector and are not supported here.
    device : str
        Torch device for the **learner only** — "cuda", "cpu", or "auto".
        Opponent inference always runs on CPU with a minimal rollout buffer
        (``n_envs=1, n_steps=1``) so it does not compete for VRAM or
        allocate a multi-GB NumPy array.  The learner's VRAM footprint scales
        with ``n_steps * n_envs * obs_shape``; reduce ``batch_size`` first
        if VRAM is tight, then ``n_steps``.
    save_every : int
        Save checkpoint every N total steps.
    checkpoint_pool_size : int
        Max historical checkpoints held as opponent candidates.
    map_id : int | None
        If set, restrict training to this single map.
    co_p0, co_p1, tier_name
        Optional fixed COs and tier for narrow curriculum (see ``AWBWEnv``).
    curriculum_broad_prob
        Per-episode probability of sampling full random matchups (mixture).
    curriculum_tag
        Stored in ``game_log.jsonl`` for slicing runs.
    batch_size : int
        PPO minibatch size; must be ``<= n_steps * n_envs``.
        Raising toward the rollout cap (``n_steps * n_envs``) gives more
        stable gradient estimates with little VRAM cost beyond the rollout
        buffer already allocated.
    opponent_mix
        When a checkpoint opponent is loaded, probability of using capture-greedy
        for that ``predict`` call instead (0 = always checkpoint when available).
    ent_coef
        PPO entropy coefficient (fresh runs only; resumed ``latest.zip`` keeps saved hparams).
    n_steps : int
        PPO rollout length per parallel env (``n_steps × n_envs`` env steps per
        rollout before each policy update).
    checkpoint_dir : Path | str | None
        Checkpoint directory (default ``<repo>/checkpoints``).
    pool_from_fleet : bool
        Also sample opponent checkpoints from the fleet root (see ``fleet_opponent_root``).
    fleet_opponent_root : Path | str | None
        Shared ``checkpoints/`` directory for fleet-wide opponent zips when using
        ``pool_from_fleet`` (defaults to ``checkpoint_dir`` on main; aux pool
        trainers should pass the shared root, e.g. ``Z:/checkpoints``).
    checkpoint_zip_cap : int
        Maximum on-disk ``checkpoint_*.zip`` files under ``checkpoint_dir``;
        oldest snapshots (by mtime) deleted after each save. ``0`` disables pruning.
    load_promoted : bool
        Prefer ``checkpoint_dir/promoted/best.zip`` over ``latest.zip`` when newer.
    bc_init : Path | str | None
        Warm-start zip for fresh runs (no resume path); ignored when resuming.
    """

    def __init__(
        self,
        total_timesteps: Optional[int] = None,
        n_envs: int = 4,
        n_steps: int = 512,
        batch_size: int = 256,
        device: str = "auto",
        save_every: int = 50_000,
        checkpoint_pool_size: int = 5,
        map_id: Optional[int] = None,
        co_p0: Optional[int] = None,
        co_p1: Optional[int] = None,
        tier_name: Optional[str] = None,
        curriculum_broad_prob: float = 0.0,
        curriculum_tag: Optional[str] = None,
        opponent_mix: float = 0.0,
        ent_coef: float = 0.05,
        checkpoint_dir: Optional[Path | str] = None,
        pool_from_fleet: bool = False,
        fleet_opponent_root: Optional[Path | str] = None,
        checkpoint_zip_cap: int = 100,
        load_promoted: bool = False,
        bc_init: Optional[Path | str] = None,
        cold_opponent: str = "random",
    ) -> None:
        roll = n_steps * n_envs
        if batch_size > roll:
            raise ValueError(
                f"batch_size ({batch_size}) must be <= n_steps * n_envs ({roll})"
            )
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.save_every = save_every
        self.checkpoint_pool_size = checkpoint_pool_size
        self.map_id_filter = map_id
        self.co_p0 = co_p0
        self.co_p1 = co_p1
        self.tier_name = tier_name
        self.curriculum_broad_prob = curriculum_broad_prob
        self.curriculum_tag = curriculum_tag
        self.opponent_mix = float(opponent_mix)
        self.ent_coef = float(ent_coef)
        self.checkpoint_dir = (
            Path(checkpoint_dir).resolve() if checkpoint_dir is not None else CHECKPOINT_DIR.resolve()
        )
        self.pool_from_fleet = bool(pool_from_fleet)
        self.fleet_opponent_root: Optional[str] = (
            str(Path(fleet_opponent_root).resolve()) if fleet_opponent_root else None
        )
        self.checkpoint_zip_cap = int(checkpoint_zip_cap)
        self.load_promoted = bool(load_promoted)
        self.bc_init = Path(bc_init).resolve() if bc_init else None
        cold = str(cold_opponent or "random").strip().lower()
        if cold not in _VALID_COLD_OPPONENTS:
            raise ValueError(
                f"cold_opponent must be one of {_VALID_COLD_OPPONENTS}; got {cold_opponent!r}"
            )
        self.cold_opponent = cold

        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.map_pool = load_map_pool()
        if self.map_id_filter is not None:
            self.map_pool = [m for m in self.map_pool if m["map_id"] == self.map_id_filter]
            if not self.map_pool:
                raise ValueError(f"No maps found with map_id={self.map_id_filter}")

        from rl.fleet_env import prune_checkpoint_zip_snapshots, sorted_checkpoint_zip_paths

        if self.checkpoint_zip_cap > 0:
            pruned = prune_checkpoint_zip_snapshots(self.checkpoint_dir, self.checkpoint_zip_cap)
            if pruned:
                print(
                    f"[self_play] Pruned {pruned} old checkpoint_*.zip "
                    f"(cap={self.checkpoint_zip_cap})"
                )
        self.checkpoints = sorted_checkpoint_zip_paths(self.checkpoint_dir)
        self.elo_ratings: dict[str, float] = {"latest": 1200.0}

    # ── Opponent helpers ──────────────────────────────────────────────────────

    def _make_opponent_policy(self) -> Optional[Callable]:
        """
        Return a policy callable from a random historical checkpoint, or None
        for random opponent. Only used in single-env (watch) mode.

        SubprocVecEnv workers build their opponent independently via
        :class:`_CheckpointOpponent` in :func:`_make_env_factory`, which
        uses the **rotating checkpoint pool** whenever
        ``checkpoints/checkpoint_*.zip`` exist, and **capture-greedy**
        heuristics when none exist yet.
        """
        if self.pool_from_fleet:
            from rl.fleet_env import iter_fleet_opponent_checkpoint_zips, sorted_checkpoint_zip_paths

            root = Path(self.fleet_opponent_root) if self.fleet_opponent_root else self.checkpoint_dir
            local = {str(p) for p in sorted_checkpoint_zip_paths(self.checkpoint_dir)}
            fleet = set(iter_fleet_opponent_checkpoint_zips(root))
            candidates = [Path(p) for p in sorted(local | fleet)]
        else:
            candidates = list(self.checkpoints)
        if not candidates:
            return None

        ckpt_path = random.choice(candidates)
        try:
            from rl.ckpt_compat import load_maskable_ppo_compat

            model = load_maskable_ppo_compat(ckpt_path, device=self.device)

            def _policy(obs: dict, mask: np.ndarray) -> int:
                action, _ = model.predict(obs, action_masks=mask, deterministic=False)
                return int(action)

            return _policy
        except Exception as exc:
            print(f"[self_play] Could not load checkpoint {ckpt_path}: {exc} — using random")
            return None

    def _build_vec_env(self):
        """Build SubprocVecEnv (n_envs > 1) or a single masked env."""
        from sb3_contrib.common.wrappers import ActionMasker  # type: ignore[import]

        env_kw = dict(
            co_p0=self.co_p0,
            co_p1=self.co_p1,
            tier_name=self.tier_name,
            curriculum_broad_prob=self.curriculum_broad_prob,
            curriculum_tag=self.curriculum_tag,
            opponent_mix=self.opponent_mix,
            pool_from_fleet=self.pool_from_fleet,
            cold_opponent=self.cold_opponent,
            fleet_opponent_root=self.fleet_opponent_root,
        )
        if self.n_envs > 1:
            from stable_baselines3.common.vec_env import SubprocVecEnv  # type: ignore[import]
            factories = [
                _make_env_factory(self.map_pool, str(self.checkpoint_dir), **env_kw)
                for _ in range(self.n_envs)
            ]
            vec_env = SubprocVecEnv(factories, start_method="spawn")
            print(f"[self_play] SubprocVecEnv: {self.n_envs} workers (spawn)")
            return vec_env
        else:
            from rl.env import AWBWEnv

            opponent = _CheckpointOpponent(
                str(self.checkpoint_dir),
                opponent_mix=self.opponent_mix,
                pool_from_fleet=self.pool_from_fleet,
                cold_opponent=self.cold_opponent,
                fleet_opponent_root=self.fleet_opponent_root,
            )
            env = AWBWEnv(
                map_pool=self.map_pool,
                opponent_policy=opponent,
                render_mode=None,
                co_p0=self.co_p0,
                co_p1=self.co_p1,
                tier_name=self.tier_name,
                curriculum_broad_prob=self.curriculum_broad_prob,
                curriculum_tag=self.curriculum_tag,
            )
            opponent.attach_env(env)
            return ActionMasker(env, _mask_fn)

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> None:
        try:
            from sb3_contrib import MaskablePPO  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sb3-contrib is required. Install: pip install sb3-contrib"
            ) from exc

        steps_msg = (
            f"{self.total_timesteps:,}"
            if self.total_timesteps is not None
            else "unlimited"
        )
        n_std = sum(1 for m in self.map_pool if m.get("type") == "std")
        cur_bits: list[str] = []
        if self.co_p0 is not None or self.co_p1 is not None:
            cur_bits.append(f"co_p0={self.co_p0} co_p1={self.co_p1}")
        if self.tier_name is not None:
            cur_bits.append(f"tier={self.tier_name!r}")
        if self.curriculum_broad_prob > 0.0:
            cur_bits.append(f"broad_prob={self.curriculum_broad_prob}")
        if self.curriculum_tag:
            cur_bits.append(f"tag={self.curriculum_tag!r}")
        if self.opponent_mix > 0.0:
            cur_bits.append(f"opponent_mix={self.opponent_mix}")
        cur_bits.append(f"ent_coef={self.ent_coef}")
        if self.pool_from_fleet:
            cur_bits.append("pool_from_fleet=1")
            if self.fleet_opponent_root:
                fro = Path(self.fleet_opponent_root)
                if fro.resolve() != self.checkpoint_dir.resolve():
                    cur_bits.append(f"fleet_opponent_root={fro}")
        if self.checkpoint_zip_cap > 0:
            cur_bits.append(f"checkpoint_zip_cap={self.checkpoint_zip_cap}")
        if self.load_promoted:
            cur_bits.append("load_promoted=1")
        if self.bc_init is not None:
            cur_bits.append(f"bc_init={self.bc_init}")
        cur_bits.append(f"cold_opponent={self.cold_opponent}")
        cur_msg = (" | " + " ".join(cur_bits)) if cur_bits else ""
        print(
            f"[self_play] Starting | steps={steps_msg} | "
            f"n_envs={self.n_envs} | n_steps={self.n_steps} "
            f"(rollout {self.n_steps * self.n_envs:,} env steps) | "
            f"batch_size={self.batch_size} | device={self.device} | "
            f"maps={len(self.map_pool)} (Std sampling: {n_std}){cur_msg}"
        )

        # Session DB under repo `.tmp/` so scratch stays on the same drive as the checkout.
        session_tmp = ROOT / ".tmp"
        session_tmp.mkdir(parents=True, exist_ok=True)
        fd, session_counter_db = tempfile.mkstemp(
            prefix="awbw_session_games_",
            suffix=".sqlite",
            dir=str(session_tmp),
        )
        os.close(fd)
        os.environ[SESSION_GAME_COUNTER_DB_ENV] = session_counter_db
        atexit.register(lambda p=session_counter_db: Path(p).unlink(missing_ok=True))

        vec_env = self._build_vec_env()

        # ── PPO hyperparameters (batch_size / n_steps tunable via train.py) ───
        # ent_coef=0.05 drives aggressive early exploration (decay manually later)
        # gamma=0.99 slightly lower than 0.995 to propagate win signal faster

        from rl.network import AWBWFeaturesExtractor  # type: ignore[import]
        policy_kwargs = dict(
            features_extractor_class=AWBWFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[],  # AWBWFeaturesExtractor already includes 2 FC layers
        )

        latest_path = self.checkpoint_dir / "latest.zip"
        promoted_path = self.checkpoint_dir / "promoted" / "best.zip"
        resume_path = latest_path
        if self.load_promoted and promoted_path.is_file():
            if not latest_path.is_file():
                resume_path = promoted_path
                print(f"[self_play] Resuming from {resume_path} (no latest.zip; --load-promoted)")
            elif promoted_path.stat().st_mtime > latest_path.stat().st_mtime:
                resume_path = promoted_path
                print(
                    f"[self_play] Resuming from {resume_path} "
                    f"(newer than latest.zip; --load-promoted)"
                )

        if resume_path.exists():
            print(f"[self_play] Resuming from {resume_path}")
            from rl.ckpt_compat import load_maskable_ppo_compat

            model = load_maskable_ppo_compat(
                resume_path,
                env=vec_env,
                device=self.device,
                custom_objects={"n_steps": self.n_steps},
            )
            # Checkpoints from another machine (e.g. Main D:\) embed tensorboard_log in the zip;
            # always write TensorBoard under this repo's logs/.
            model.tensorboard_log = str(LOGS_DIR)
        elif self.bc_init is not None and self.bc_init.is_file():
            print(f"[self_play] Fresh run: warm-start from {self.bc_init}")
            from rl.ckpt_compat import load_maskable_ppo_compat

            model = load_maskable_ppo_compat(
                self.bc_init,
                env=vec_env,
                device=self.device,
                custom_objects={
                    "n_steps": self.n_steps,
                    "batch_size": self.batch_size,
                },
            )
            model.tensorboard_log = str(LOGS_DIR)
        else:
            model = MaskablePPO(
                "MultiInputPolicy",
                vec_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                device=self.device,
                learning_rate=3e-4,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=self.ent_coef,
                clip_range=0.2,
                vf_coef=0.5,
                max_grad_norm=0.5,
                normalize_advantage=True,
                tensorboard_log=str(LOGS_DIR),
            )

        from rl.fleet_env import (
            new_checkpoint_stem_utc,
            prune_checkpoint_zip_snapshots,
            sorted_checkpoint_zip_paths,
        )

        steps_done = 0
        t_start = time.time()
        cap = self.total_timesteps

        # Built once and reused across `model.learn` chunks so the
        # per-env running counters survive between rollouts.
        diag_callback = _build_diagnostics_callback()

        try:
            while cap is None or steps_done < cap:
                if cap is None:
                    chunk = self.save_every
                else:
                    chunk = min(self.save_every, cap - steps_done)
                elapsed = time.time() - t_start
                rate = steps_done / elapsed if elapsed > 0 else 0
                if cap is None:
                    eta_str = "—"
                else:
                    rem = cap - steps_done
                    eta = rem / rate if rate > 0 else float("inf")
                    eta_str = f"{eta / 60:.1f}min"
                print(
                    f"\n[self_play] Steps {steps_done:,}->{steps_done + chunk:,} | "
                    f"{rate:,.0f} steps/s | ETA {eta_str}"
                )

                model.learn(
                    total_timesteps=chunk,
                    reset_num_timesteps=False,
                    callback=diag_callback,
                )
                steps_done += chunk

                ckpt_stem = new_checkpoint_stem_utc()
                _atomic_model_save(model, self.checkpoint_dir / ckpt_stem)
                _atomic_model_save(model, self.checkpoint_dir / "latest")
                print(f"[self_play] Saved {ckpt_stem}.zip")

                if self.checkpoint_zip_cap > 0:
                    pruned = prune_checkpoint_zip_snapshots(
                        self.checkpoint_dir, self.checkpoint_zip_cap
                    )
                    if pruned:
                        print(
                            f"[self_play] Pruned {pruned} checkpoint_*.zip "
                            f"(cap={self.checkpoint_zip_cap})"
                        )
                tail = self.checkpoint_pool_size if self.checkpoint_pool_size > 0 else None
                all_ck = sorted_checkpoint_zip_paths(self.checkpoint_dir)
                self.checkpoints = all_ck[-tail:] if tail else all_ck
        except KeyboardInterrupt:
            print("\n[self_play] Stopped by user (KeyboardInterrupt). Saving latest checkpoint…")
            _atomic_model_save(model, self.checkpoint_dir / "latest")
            print(f"[self_play] Saved -> {self.checkpoint_dir / 'latest.zip'}")

        total_elapsed = time.time() - t_start
        steps_note = f"{steps_done:,}" if cap is None else f"{self.total_timesteps:,}"
        print(
            f"\n[self_play] Done. {steps_note} steps in "
            f"{total_elapsed/60:.1f}min | Final -> {self.checkpoint_dir / 'latest.zip'}"
        )


# ── Watch helper ──────────────────────────────────────────────────────────────

def watch_game(
    map_id: Optional[int] = None,
    co_p0: int = 1,
    co_p1: int = 7,
    delay: float = 0.05,
) -> None:
    """
    Play a single game with random policies, printing the board every 10 steps.
    Useful for smoke-testing the engine without running full training.
    """
    import time as _time

    from engine.map_loader import load_map
    from engine.game import make_initial_state
    from engine.action import get_legal_actions

    with open(POOL_PATH) as f:
        pool: list[dict] = json.load(f)

    if map_id is None:
        meta = random.choice(pool)
        map_id = meta["map_id"]
    else:
        try:
            meta = next(m for m in pool if m["map_id"] == map_id)
        except StopIteration:
            raise ValueError(f"map_id={map_id} not found in pool.")

    enabled = [t for t in meta["tiers"] if t.get("enabled") and t.get("co_ids")]
    tier = random.choice(enabled) if enabled else meta["tiers"][0]
    tier_name: str = tier["tier_name"]

    print(f"\n[watch] Map: {meta.get('name', map_id)} ({map_id}) | Tier: {tier_name}")
    print(f"[watch] P0: CO#{co_p0}  P1: CO#{co_p1}")

    map_data = load_map(map_id, POOL_PATH, MAPS_DIR)
    state = make_initial_state(map_data, co_p0, co_p1, starting_funds=0, tier_name=tier_name)
    opening_player = int(state.active_player)

    step_count = 0
    while not state.done and step_count < 10_000:
        if step_count % 10 == 0:
            print(f"\n--- Step {step_count} | Turn {state.turn} | P{state.active_player} ---")
            print(state.render_ascii())
            _time.sleep(delay)

        legal = get_legal_actions(state)
        if not legal:
            print("[watch] No legal actions — stopping.")
            break
        action = random.choice(legal)
        state, _, _ = state.step(action)
        step_count += 1

    print(f"\n[watch] Game over — winner: P{state.winner} | turns: {state.turn} | steps: {step_count}")
    print(state.render_ascii())

    log_game(
        map_id=map_id,
        tier=tier_name,
        p0_co=co_p0,
        p1_co=co_p1,
        winner=state.winner if state.winner is not None else -1,
        turns=state.turn,
        funds_end=list(state.funds),
        n_actions=len(state.game_log),
        opening_player=opening_player,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    if args and args[0] == "watch":
        mid = int(args[1]) if len(args) > 1 else None
        watch_game(map_id=mid)
    else:
        trainer = SelfPlayTrainer(
            total_timesteps=None,
            save_every=10_000,
        )
        trainer.train()
