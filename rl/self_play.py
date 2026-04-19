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
- Logs every completed game to data/game_log.jsonl
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
from pathlib import Path
from typing import Callable, Optional

from rl.env import SESSION_GAME_COUNTER_DB_ENV

import numpy as np

ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
GAME_LOG_PATH = ROOT / "data" / "game_log.jsonl"
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
    """Append a game result record to data/game_log.jsonl."""
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


class _CheckpointOpponent:
    """
    Picklable opponent policy that loads random historical checkpoints.

    Lazily loads a checkpoint on first call and refreshes every
    `refresh_every` calls so the opponent pool rotates naturally as
    new checkpoints are written by the trainer.  Falls back to uniform-
    random legal actions when no checkpoints exist yet.
    """

    def __init__(self, checkpoint_dir: str, refresh_every: int = 500) -> None:
        self._dir = checkpoint_dir
        self._refresh_every = refresh_every
        self._model = None
        self._n_calls = 0
        # Exposed so AWBWEnv can count per-episode reloads. Incremented
        # only on a successful checkpoint load (not random fallback).
        self.reload_count: int = 0

    def _load_random(self) -> None:
        import glob as _glob
        ckpts = sorted(_glob.glob(os.path.join(self._dir, "checkpoint_*.zip")))
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
        """Return 'checkpoint' if a model is loaded, 'random' if falling back."""
        return "checkpoint" if self._model is not None else "random"

    def __call__(self, obs: dict, mask: np.ndarray) -> int:
        if self._n_calls % self._refresh_every == 0:
            self._load_random()
        self._n_calls += 1
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
) -> Callable:
    """Return a picklable env factory for SubprocVecEnv."""
    def _init():
        from rl.env import AWBWEnv
        from sb3_contrib.common.wrappers import ActionMasker  # type: ignore[import]
        opponent = _CheckpointOpponent(checkpoint_dir)
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
    n_steps : int
        PPO rollout length per parallel env (``n_steps × n_envs`` env steps per
        rollout before each policy update).
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

        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        self.map_pool = load_map_pool()
        if self.map_id_filter is not None:
            self.map_pool = [m for m in self.map_pool if m["map_id"] == self.map_id_filter]
            if not self.map_pool:
                raise ValueError(f"No maps found with map_id={self.map_id_filter}")

        self.checkpoints: list[Path] = sorted(CHECKPOINT_DIR.glob("checkpoint_*.zip"))
        self.elo_ratings: dict[str, float] = {"latest": 1200.0}

    # ── Opponent helpers ──────────────────────────────────────────────────────

    def _make_opponent_policy(self) -> Optional[Callable]:
        """
        Return a policy callable from a random historical checkpoint, or None
        for random opponent. Only used in single-env (watch) mode.

        SubprocVecEnv workers build their opponent independently via
        :class:`_CheckpointOpponent` in :func:`_make_env_factory`, which
        defaults to the **rotating checkpoint pool** (not pure random)
        whenever `checkpoints/checkpoint_*.zip` exist. It falls back to
        random only when no checkpoints are present yet.
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
        )
        if self.n_envs > 1:
            from stable_baselines3.common.vec_env import SubprocVecEnv  # type: ignore[import]
            factories = [
                _make_env_factory(self.map_pool, str(CHECKPOINT_DIR), **env_kw)
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

        latest_path = CHECKPOINT_DIR / "latest.zip"
        if latest_path.exists():
            print(f"[self_play] Resuming from {latest_path}")
            model = MaskablePPO.load(
                latest_path,
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
                ent_coef=0.05,
                clip_range=0.2,
                vf_coef=0.5,
                max_grad_norm=0.5,
                normalize_advantage=True,
                tensorboard_log=str(ROOT / "logs"),
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
                ckpt_path = CHECKPOINT_DIR / ckpt_name
                model.save(str(ckpt_path))
                model.save(str(CHECKPOINT_DIR / "latest"))
                print(f"[self_play] Saved {ckpt_name}.zip")

                resolved = Path(str(ckpt_path) + ".zip")
                self.checkpoints.append(resolved)
                if len(self.checkpoints) > self.checkpoint_pool_size:
                    self.checkpoints.pop(0)

                ckpt_idx += 1
        except KeyboardInterrupt:
            print("\n[self_play] Stopped by user (KeyboardInterrupt). Saving latest checkpoint…")
            model.save(str(CHECKPOINT_DIR / "latest"))
            print(f"[self_play] Saved -> {CHECKPOINT_DIR / 'latest.zip'}")

        total_elapsed = time.time() - t_start
        steps_note = f"{steps_done:,}" if cap is None else f"{self.total_timesteps:,}"
        print(
            f"\n[self_play] Done. {steps_note} steps in "
            f"{total_elapsed/60:.1f}min | Final -> {CHECKPOINT_DIR / 'latest.zip'}"
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
