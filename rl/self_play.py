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
- Rotating opponent pool: keeps the last `checkpoint_pool_size` checkpoints
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
    if at == ActionType.WAIT:
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


class _CheckpointOpponent:
    """
    Picklable opponent policy that loads random historical checkpoints.

    Lazily loads a checkpoint on first call and refreshes every
    `refresh_every` calls so the opponent pool rotates naturally as
    new checkpoints are written by the trainer.      Falls back to :func:`pick_capture_greedy_flat` when no checkpoints exist yet
    (instead of uniform random) so early training sees property contests.

    With ``pool_from_fleet``, also samples ``checkpoint_*.zip`` under
    ``<checkpoint_dir>/pool/*/`` (divergent aux trainers writing into
    per-machine pool dirs).
    """

    def __init__(
        self,
        checkpoint_dir: str,
        refresh_every: int = 500,
        opponent_mix: float = 0.0,
        pool_from_fleet: bool = False,
    ) -> None:
        self._dir = checkpoint_dir
        self._refresh_every = refresh_every
        self._opponent_mix = max(0.0, min(1.0, float(opponent_mix)))
        self._pool_from_fleet = bool(pool_from_fleet)
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
            from rl.fleet_env import iter_pool_checkpoint_zips

            ckpts = sorted(set(ckpts + iter_pool_checkpoint_zips(Path(self._dir))))
        if not ckpts:
            self._model = None
            return
        path = random.choice(ckpts)
        try:
            from sb3_contrib import MaskablePPO as _PPO  # type: ignore[import]
            self._model = _PPO.load(path, device="cpu")
            self.reload_count += 1
        except Exception as exc:
            print(f"[opponent] Could not load {path}: {exc} — using random")
            self._model = None

    def mode(self) -> str:
        """Return ``checkpoint``, ``greedy``, or ``mixed`` for game logs."""
        if self._model is None:
            return "greedy_capture"
        if self._opponent_mix > 0.0:
            return "mixed"
        return "checkpoint"

    def __call__(self, obs: dict, mask: np.ndarray) -> int:
        if self._n_calls % self._refresh_every == 0:
            self._load_random()
        self._n_calls += 1

        use_greedy = self._model is None
        if self._model is not None and self._opponent_mix > 0.0:
            if random.random() < self._opponent_mix:
                use_greedy = True

        if use_greedy:
            env = self._env_ref() if self._env_ref is not None else None
            st: GameState | None = getattr(env, "state", None) if env is not None else None
            if st is not None:
                return pick_capture_greedy_flat(st, mask)

        if self._model is None:
            legal = np.where(mask)[0]
            return int(np.random.choice(legal)) if len(legal) > 0 else 0

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
) -> Callable:
    """Return a picklable env factory for SubprocVecEnv."""
    def _init():
        from rl.env import AWBWEnv
        from sb3_contrib.common.wrappers import ActionMasker  # type: ignore[import]
        opponent = _CheckpointOpponent(
            checkpoint_dir,
            opponent_mix=opponent_mix,
            pool_from_fleet=pool_from_fleet,
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

        def _on_step(self) -> bool:
            dones = self.locals.get("dones")
            if dones is None:
                return True
            n_envs = len(dones)
            if len(self._cur_lens) != n_envs:
                self._cur_lens = [0] * n_envs
            for i in range(n_envs):
                self._cur_lens[i] += 1
                if dones[i]:
                    self._finished_lens.append(self._cur_lens[i])
                    self._cur_lens[i] = 0
            return True

        def _on_rollout_end(self) -> None:
            n = len(self._finished_lens)
            self.logger.record("diag/episodes_per_rollout", n)
            if n:
                self.logger.record(
                    "diag/ep_len_mean", sum(self._finished_lens) / n
                )
                self.logger.record("diag/ep_len_max", max(self._finished_lens))
                self.logger.record("diag/ep_len_min", min(self._finished_lens))
            self._finished_lens.clear()

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
    device : str
        Torch device — "cuda", "cpu", or "auto".
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
        Also sample opponent checkpoints from ``checkpoint_dir/pool/*/checkpoint_*.zip``.
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
        load_promoted: bool = False,
        bc_init: Optional[Path | str] = None,
    ) -> None:
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.n_steps = n_steps
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
        self.load_promoted = bool(load_promoted)
        self.bc_init = Path(bc_init).resolve() if bc_init else None

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

        self.checkpoints: list[Path] = sorted(self.checkpoint_dir.glob("checkpoint_*.zip"))
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
        if not self.checkpoints:
            return None

        ckpt_path = random.choice(self.checkpoints)
        try:
            from sb3_contrib import MaskablePPO  # type: ignore[import]
            model = MaskablePPO.load(ckpt_path, device=self.device)

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
            env = AWBWEnv(map_pool=self.map_pool, render_mode=None, **env_kw)
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
        if self.load_promoted:
            cur_bits.append("load_promoted=1")
        if self.bc_init is not None:
            cur_bits.append(f"bc_init={self.bc_init}")
        cur_msg = (" | " + " ".join(cur_bits)) if cur_bits else ""
        print(
            f"[self_play] Starting | steps={steps_msg} | "
            f"n_envs={self.n_envs} | n_steps={self.n_steps} "
            f"(rollout {self.n_steps * self.n_envs:,} env steps) | device={self.device} | "
            f"maps={len(self.map_pool)} (Std sampling: {n_std}){cur_msg}"
        )

        # One shared monotonic game_id sequence for all parallel workers this run (spawn inherits env).
        fd, session_counter_db = tempfile.mkstemp(prefix="awbw_session_games_", suffix=".sqlite")
        os.close(fd)
        os.environ[SESSION_GAME_COUNTER_DB_ENV] = session_counter_db
        atexit.register(lambda p=session_counter_db: Path(p).unlink(missing_ok=True))

        vec_env = self._build_vec_env()

        # ── Hyperparameters tuned for i7-13700F + RTX 4070 @ 33% budget ──────
        # n_steps=512 per env → 512 × n_envs total steps per rollout
        #   (reduced from 2048 to cut rollout-buffer RAM ~4×; more frequent updates)
        # batch_size=256 keeps GPU utilisation high without excess VRAM
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
            model = MaskablePPO.load(
                resume_path,
                env=vec_env,
                device=self.device,
                custom_objects={"n_steps": self.n_steps},
            )
        elif self.bc_init is not None and self.bc_init.is_file():
            print(f"[self_play] Fresh run: warm-start from {self.bc_init}")
            model = MaskablePPO.load(
                self.bc_init,
                env=vec_env,
                device=self.device,
                custom_objects={"n_steps": self.n_steps},
            )
        else:
            model = MaskablePPO(
                "MultiInputPolicy",
                vec_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                device=self.device,
                learning_rate=3e-4,
                n_steps=self.n_steps,
                batch_size=256,
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

        steps_done = 0
        ckpt_idx = len(self.checkpoints)
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

                ckpt_name = f"checkpoint_{ckpt_idx:04d}"
                ckpt_path = self.checkpoint_dir / ckpt_name
                model.save(str(ckpt_path))
                model.save(str(self.checkpoint_dir / "latest"))
                print(f"[self_play] Saved {ckpt_name}.zip")

                resolved = Path(str(ckpt_path) + ".zip")
                self.checkpoints.append(resolved)
                if len(self.checkpoints) > self.checkpoint_pool_size:
                    self.checkpoints.pop(0)

                ckpt_idx += 1
        except KeyboardInterrupt:
            print("\n[self_play] Stopped by user (KeyboardInterrupt). Saving latest checkpoint…")
            model.save(str(self.checkpoint_dir / "latest"))
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
