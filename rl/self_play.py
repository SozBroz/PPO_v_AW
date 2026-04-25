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
import gc
import json
import os
import random
import sys
import tempfile
import time
import weakref
from pathlib import Path
from typing import Any, Callable, Optional

from engine.action import Action, ActionStage, ActionType, get_legal_actions
from engine.game import GameState
from engine.unit import UnitType

from rl.env import SESSION_GAME_COUNTER_DB_ENV, effective_track_per_worker_times
from rl.paths import GAME_LOG_PATH, LOGS_DIR
from rl.train_reconfig_log import append_train_reconfig_line

WATCH_LOG_PATH = LOGS_DIR / "watch_log.jsonl"
FPS_DIAG_PATH = LOGS_DIR / "fps_diag.jsonl"
_OPPONENT_CUDA_CAP_WARNED = False

# Max Subproc worker indices [0, N) that may use CUDA for checkpoint opponents (VRAM / contention).
OPPONENT_CUDA_WORKERS_MAX = 4


def _default_async_learner_transitions_cap() -> int:
    """When async and ``async_learner_batch`` is omitted, cap transitions per learner update (VRAM)."""
    raw = (os.environ.get("AWBW_ASYNC_LEARNER_TRANSITIONS_CAP") or "2048").strip()
    try:
        v = int(raw, 10)
    except ValueError:
        v = 2048
    return max(64, v)


def _snap_async_learner_batch_to_unroll(batch: int, unroll: int, roll: int) -> int:
    """Floor ``batch`` to a multiple of ``unroll`` in ``[unroll, roll]`` (IMPALA segment alignment)."""
    ub = max(4, int(unroll))
    bv = max(ub, min(int(batch), int(roll)))
    segs = max(1, bv // ub)
    return min(int(roll), segs * ub)


def _env_flag_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def gpu_opponent_pool_enabled() -> bool:
    """True when workers should share a global CUDA inference cap (semaphore), not fixed indices."""
    return _env_flag_truthy("AWBW_GPU_OPPONENT_POOL") and _env_flag_truthy(
        "AWBW_ALLOW_CUDA_OPPONENT"
    )


def gpu_opponent_pool_permits() -> int:
    """Concurrent GPU opponent forwards allowed process-wide (semaphore value)."""
    raw = (os.environ.get("AWBW_GPU_OPPONENT_POOL_SIZE") or "").strip()
    if raw.isdigit():
        v = int(raw)
        return max(1, min(v, 32))
    return int(OPPONENT_CUDA_WORKERS_MAX)


def _install_subproc_worker_excepthook() -> None:
    """Ensure SubprocVecEnv workers print a traceback before exit (Windows ``spawn``)."""

    def _hook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        import traceback

        print(
            "[awbw_subproc_worker] Unhandled exception in env worker:",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
        sys.stderr.flush()

    sys.excepthook = _hook


def _wrap_subproc_vec_env_worker_ipc(vec_env: Any, *, n_envs: int) -> None:
    """
    SB3 raises EOFError when a child dies; replace with a message that points to
    common causes (RAM, CUDA) and any ``[awbw_subproc_worker]`` traceback above.
    """
    hints = (
        "SubprocVecEnv: a worker process exited (IPC pipe closed). "
        "Each worker loads Torch and a checkpoint opponent — high --n-envs can exhaust RAM "
        "(OS may kill a child with no Python traceback). Also check CUDA OOM and any "
        "[awbw_subproc_worker] traceback above. "
        f"This run uses n_envs={int(n_envs)}."
    )
    if sys.platform == "win32":
        hints += (
            " On Windows + pc-b, proposed_args often cap n_envs≈4; overriding with much larger "
            "values frequently triggers this failure."
        )

    _orig_step_wait = vec_env.step_wait
    _orig_reset = vec_env.reset

    def step_wait() -> Any:
        try:
            return _orig_step_wait()
        except (EOFError, BrokenPipeError) as e:
            raise RuntimeError(hints) from e

    def reset(*args: Any, **kwargs: Any) -> Any:
        try:
            return _orig_reset(*args, **kwargs)
        except (EOFError, BrokenPipeError) as e:
            raise RuntimeError(hints) from e

    vec_env.step_wait = step_wait  # type: ignore[method-assign]
    vec_env.reset = reset  # type: ignore[method-assign]

    _orig_env_method = getattr(vec_env, "env_method", None)
    if _orig_env_method is not None:
        def env_method(*args: Any, **kwargs: Any) -> Any:
            try:
                return _orig_env_method(*args, **kwargs)
            except (EOFError, BrokenPipeError) as e:
                raise RuntimeError(hints) from e

        vec_env.env_method = env_method  # type: ignore[method-assign]


def _policy_torch_compile_skip_note(exc: BaseException) -> str:
    s = str(exc)
    if sys.platform == "win32" and "Failed to find C compiler" in s:
        return (
            s
            + " (Windows: install Visual Studio Build Tools with the C++ workload so Triton "
            "can compile its CUDA driver; ensure CUDA tookits match PyTorch.)"
        )
    return s


def _windows_torch_compile_opt_in() -> bool:
    """
    Triton on Windows still JIT-compiles a small CUDA driver on first use and needs cl.exe.
    Default off so training works without Visual Studio Build Tools; set AWBW_TORCH_COMPILE=1
    when the toolchain is installed and you want Inductor speedups.
    """
    if sys.platform != "win32":
        return True
    v = os.environ.get("AWBW_TORCH_COMPILE", "").strip().lower()
    return v in ("1", "true", "yes", "on")
# Rotate before append when the log grows large (mirrors ad-hoc JSONL hygiene elsewhere).
_FPS_DIAG_MAX_BYTES = 32 * 1024 * 1024
_WORKER_RSS_RESAMPLE_S = 5.0

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
    """Append a watch-tool game record to ``logs/watch_log.jsonl``.

    Standalone ``watch_game`` writer — kept on the legacy schema 1.5 shape (no
    ``game_id``). Production training writes through
    :func:`rl.env._log_finished_game` to ``game_log.jsonl`` on schema >= 1.6;
    this writer is intentionally segregated so a misaimed watch session cannot
    pollute the orchestrator's parser.
    """
    # Writes to logs/watch_log.jsonl (separate from production game_log.jsonl per Phase 10/11 logging prereqs).
    WATCH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
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
    with open(WATCH_LOG_PATH, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def _append_fps_diag_line(record: dict) -> None:
    """Append one JSON object line to ``logs/fps_diag.jsonl`` (size-capped rotation)."""
    FPS_DIAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FPS_DIAG_PATH.is_file() and FPS_DIAG_PATH.stat().st_size > _FPS_DIAG_MAX_BYTES:
        rotated = FPS_DIAG_PATH.with_name(FPS_DIAG_PATH.name + ".1")
        if rotated.is_file():
            rotated.unlink()
        FPS_DIAG_PATH.rename(rotated)
    with open(FPS_DIAG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _sum_python_children_rss_mb() -> float:
    """Sum RSS (MiB) for this process's descendants whose executable looks like Python."""
    try:
        import psutil  # type: ignore[import]

        main = psutil.Process()
        total_b = 0
        for p in main.children(recursive=True):
            try:
                name = (p.name() or "").lower()
                if "python" not in name:
                    continue
                total_b += int(p.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return float(total_b) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _main_proc_rss_mb() -> float:
    try:
        import psutil  # type: ignore[import]

        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _system_ram_used_pct() -> float:
    try:
        import psutil  # type: ignore[import]

        return float(psutil.virtual_memory().percent)
    except Exception:
        return 0.0


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


def _pfsp_pick_checkpoint(paths: list[str]) -> str:
    """Weighted random checkpoint: concave ``(1-w)*w`` on optional win-rates JSON.

    Enable with ``AWBW_PFSP=1``. Optional ``AWBW_PFSP_STATS`` path to JSON mapping
    zip path / basename / resolved path → win rate in ``[0,1]``. Missing entries
    default to 0.5 (uniform-ish weights).
    """
    if not paths:
        raise ValueError("PFSP pick on empty checkpoint list")
    flag = (os.environ.get("AWBW_PFSP", "") or "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return random.choice(paths)
    stats_path = (os.environ.get("AWBW_PFSP_STATS", "") or "").strip()
    if not stats_path or not os.path.isfile(stats_path):
        return random.choice(paths)
    try:
        with open(stats_path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return random.choice(paths)
    weights: list[float] = []
    for p in paths:
        w = 0.5
        for k in (p, os.path.basename(p), str(Path(p).resolve())):
            if isinstance(raw, dict) and k in raw:
                try:
                    w = float(raw[k])
                except (TypeError, ValueError):
                    w = 0.5
                break
        w = min(0.99, max(0.01, w))
        weights.append((1.0 - w) * w)
    s = sum(weights)
    if s <= 0:
        return random.choice(paths)
    norm = [x / s for x in weights]
    return random.choices(paths, weights=norm, k=1)[0]


# ── Trainer ───────────────────────────────────────────────────────────────────

def _mask_fn(env: "AWBWEnv") -> np.ndarray:  # type: ignore[name-defined]
    """Module-level mask function — picklable for SubprocVecEnv workers."""
    return env.action_masks()


def _atomic_model_save(model, dest_no_ext: str | os.PathLike) -> None:
    """
    Save an SB3 model to ``<dest_no_ext>.zip`` atomically.

    Stable-Baselines3 does **not** simply do ``str(path) + ".zip"``.  Its
    ``open_path`` helper only appends ``.zip`` when :func:`pathlib.Path`
    reports an **empty** ``suffix``.  A base path ending in ``.tmp`` has
    ``suffix == ".tmp"``, so the model is written to a file literally named
    ``<name>.tmp`` (a zip bitstream) — *not* ``<name>.tmp.zip``.  That broke
    expected ``.tmp.zip`` / ``os.replace`` logic on every OS.

    We use a temp base ``<name>_saving`` (no extra ``.`` in the filename) so
    ``suffix == ""`` and SB3 creates ``<name>_saving.zip`` predictably, then
    we ``os.replace`` onto ``<name>.zip``.  Over SMB / shared mounts, an
    opponent process can still race a partial read; this keeps a single
    final rename.
    """
    dest = Path(dest_no_ext)
    final_zip = dest.parent / f"{dest.name}.zip"
    tmp_base = dest.parent / f"{dest.name}_saving"  # Path.suffix must be ""
    tmp_zip = dest.parent / f"{dest.name}_saving.zip"
    try:
        if tmp_zip.exists():
            try:
                tmp_zip.unlink()
            except OSError:
                pass
        model.save(str(tmp_base))
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
        idx = _action_to_flat(act, state)
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


_VALID_COLD_OPPONENTS = ("random", "greedy_capture", "greedy_mix", "end_turn")


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
                idx = _action_to_flat(act, state)
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
                idx = _action_to_flat(act, state)
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
    - ``"greedy_mix"``: each microstep, 50% capture-greedy / 50% uniform
      random legal — curriculum stage_b+ ``--cold-opponent``; softens pure
      teacher without going fully random.
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
        inference_device: str = "cpu",
        gpu_infer_semaphore: Any = None,
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
        d = str(inference_device or "cpu").strip().lower()
        if d in ("gpu", "cuda"):
            self._inference_device = "cuda"
        else:
            self._inference_device = "cpu"
        # Process-shared cap on concurrent CUDA opponent forwards (SubprocVecEnv workers).
        self._gpu_infer_sem = gpu_infer_semaphore
        self._model = None
        self._n_calls = 0
        # Exposed so AWBWEnv can count per-episode reloads. Incremented
        # only on a successful checkpoint load (not random fallback).
        self.reload_count: int = 0
        self._env_ref: Optional[Callable[[], Any]] = None
        # Phase 10c: set by ``reload_pool``; consumed by ``_load_random`` (None = glob each time)
        self._pool_candidates: Optional[list[str]] = None

    def attach_env(self, env: object) -> None:
        """Weak reference so greedy fallback can read ``env.state``."""
        self._env_ref = weakref.ref(env)

    def _load_random(self) -> None:
        import glob as _glob

        if self._pool_candidates is not None:
            ckpts = list(self._pool_candidates)
        else:
            ckpts = sorted(_glob.glob(os.path.join(self._dir, "checkpoint_*.zip")))
            if self._pool_from_fleet:
                from rl.fleet_env import iter_fleet_opponent_checkpoint_zips

                root = self._fleet_opponent_root or self._dir
                ckpts = sorted(set(ckpts + iter_fleet_opponent_checkpoint_zips(Path(root))))
        if not ckpts:
            self._model = None
            return
        path = _pfsp_pick_checkpoint(ckpts)
        try:
            from rl.ckpt_compat import load_maskable_ppo_compat

            import torch

            dev = self._inference_device
            if dev == "cuda" and not torch.cuda.is_available():
                dev = "cpu"

            # Load with minimal buffer dims so _setup_model() allocates a tiny
            # rollout buffer (~KB) rather than replicating the learner's full
            # n_steps × n_envs × obs_shape tensor (~GB). Policy weights are
            # preserved; only inference (predict) is used from this model.
            self._model = load_maskable_ppo_compat(
                path,
                device=dev,
                verbose=0,
                n_envs=1,
                n_steps=1,
                batch_size=1,
            )

            if self._model is not None:
                try:
                    # Dynamic quantization is CPU-only and breaks .cuda() pool inference.
                    if dev == "cpu" and self._gpu_infer_sem is None:
                        from torch.quantization import quantize_dynamic

                        self._model.policy = quantize_dynamic(
                            self._model.policy,
                            {torch.nn.Linear},
                            dtype=torch.qint8,
                        )
                    fp16 = os.environ.get("AWBW_FP16_INFERENCE", "").lower() in (
                        "1",
                        "true",
                    )
                    if fp16:
                        self._model.policy = self._model.policy.half()
                except Exception as e:
                    print(f"[opponent] Quantization/FP16 setup failed: {e}")
            
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

    def needs_observation(self) -> bool:
        """Phase 1a: tells the env whether this opponent will consume an
        observation on the *next* call.

        Returns True when a checkpoint model is loaded and the opponent
        will route through ``self._model.predict(obs, ...)``. Returns
        False during cold-start (model is None) when ``__call__`` will
        fall through to ``_cold_action(mask)``, which never reads ``obs``.

        Conservative: when ``opponent_mix > 0`` and a model is loaded, we
        return True even though some fraction of calls will substitute the
        cold path. The expected encode work is still nonzero, and we must
        not predict the rng outcome of the substitution. The cold-only
        regime (model is None) is where the real win lives — the dominant
        training state today per Phase 0 baseline (16x P1/P0 wall ratio).
        """
        return self._model is not None

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
        if self._cold_opponent == "greedy_mix":
            if random.random() < 0.5:
                env = self._env_ref() if self._env_ref is not None else None
                st = getattr(env, "state", None) if env is not None else None
                if st is not None:
                    return pick_capture_greedy_flat(st, mask)
            legal = np.where(mask)[0]
            return int(np.random.choice(legal)) if len(legal) > 0 else 0
        # default: random legal action
        legal = np.where(mask)[0]
        return int(np.random.choice(legal)) if len(legal) > 0 else 0

    def reload_pool(self, zip_paths: Optional[list[str]] = None) -> int:
        """Phase 10c: refresh the opponent pool view without process restart.

        By default re-globs the same sources ``_load_random`` already uses
        (local checkpoint dir + fleet pool roots when ``pool_from_fleet``).
        If ``zip_paths`` is given, treat it as the new candidate set
        (used by tests).

        Returns the new candidate count. Does NOT reload the model — the
        next `_load_random` call will pick from the refreshed set on the
        next ``__call__`` boundary that crosses ``_refresh_every``.

        If the currently loaded ``self._model`` came from a zip that is
        no longer in the candidate set, the next ``_load_random`` picks a
        different zip; we do not force-evict mid-call (would race rollout
        collection).
        """
        import glob as _glob

        if zip_paths is not None:
            self._pool_candidates = list(zip_paths)
            return len(self._pool_candidates)
        ckpts = sorted(_glob.glob(os.path.join(self._dir, "checkpoint_*.zip")))
        if self._pool_from_fleet:
            from rl.fleet_env import iter_fleet_opponent_checkpoint_zips

            root = self._fleet_opponent_root or self._dir
            ckpts = sorted(set(ckpts + iter_fleet_opponent_checkpoint_zips(Path(root))))
        self._pool_candidates = ckpts
        return len(ckpts)

    def _predict_checkpoint_policy_pool(self, obs: dict, mask: np.ndarray, *, use_cuda: bool) -> int:
        """Run ``predict`` on CPU or CUDA; always leave policy on CPU after (for VRAM sharing)."""
        import torch

        m = self._model
        assert m is not None
        fp16 = os.environ.get("AWBW_FP16_INFERENCE", "").lower() in ("1", "true")
        want_cuda = bool(use_cuda and torch.cuda.is_available())
        pol = m.policy
        if want_cuda:
            pol.cuda()
        else:
            pol.cpu()
        try:
            if fp16 and want_cuda:
                with torch.cuda.amp.autocast():
                    action, _ = m.predict(obs, action_masks=mask, deterministic=False)
            elif fp16 and not want_cuda:
                try:
                    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                        action, _ = m.predict(
                            obs, action_masks=mask, deterministic=False
                        )
                except Exception:
                    action, _ = m.predict(obs, action_masks=mask, deterministic=False)
            else:
                action, _ = m.predict(obs, action_masks=mask, deterministic=False)
            return int(action)
        finally:
            pol.cpu()

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

        if self._gpu_infer_sem is not None:
            import torch

            acquired = False
            if torch.cuda.is_available():
                acquired = bool(self._gpu_infer_sem.acquire(block=False))
            try:
                return self._predict_checkpoint_policy_pool(
                    obs, mask, use_cuda=acquired
                )
            finally:
                if acquired:
                    self._gpu_infer_sem.release()

        import torch

        fp16 = os.environ.get("AWBW_FP16_INFERENCE", "").lower() in ("1", "true")
        dev = self._inference_device
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        try:
            p_dev = next(self._model.policy.parameters()).device
        except (StopIteration, AttributeError):
            p_dev = torch.device(dev)

        if fp16 and p_dev.type == "cuda":
            with torch.cuda.amp.autocast():
                action, _ = self._model.predict(
                    obs, action_masks=mask, deterministic=False
                )
        elif fp16 and p_dev.type == "cpu":
            try:
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    action, _ = self._model.predict(
                        obs, action_masks=mask, deterministic=False
                    )
            except Exception:
                action, _ = self._model.predict(
                    obs, action_masks=mask, deterministic=False
                )
        else:
            action, _ = self._model.predict(obs, action_masks=mask, deterministic=False)

        return int(action)


def _apply_learner_cuda_perf_opts(device: str) -> None:
    """
    Optional CUDA matmul / cuDNN toggles for the **learner** process (main only).

    - ``AWBW_CUDA_TF32``: when unset or truthy, enable TF32 for matmul/cudnn on CUDA.
      Set ``0`` / ``false`` to disable.
    - ``AWBW_CUDNN_BENCHMARK=1``: enable ``cudnn.benchmark`` (can help static shapes;
      try off first if you see regressions).
    """
    if device != "cuda":
        return
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    raw_tf32 = (os.environ.get("AWBW_CUDA_TF32") or "").strip().lower()
    if raw_tf32 in ("0", "false", "no", "off"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    bench = (os.environ.get("AWBW_CUDNN_BENCHMARK") or "").strip().lower()
    if bench in ("1", "true", "yes", "on"):
        torch.backends.cudnn.benchmark = True


def _subproc_worker_thread_env(worker_index: int) -> None:
    """
    Configure BLAS / torch thread counts for one SubprocVecEnv worker.

    Default (no ``AWBW_N_LEAN_WORKERS``): same as legacy — optional global
    ``AWBW_WORKER_OMP_THREADS``, else OMP/MKL defaults of 1 via setdefault.

    With ``AWBW_N_LEAN_WORKERS=K``: workers ``0..K-1`` use 1 thread; workers
    ``K..`` use ``AWBW_CPU_WORKER_THREADS``, else ``AWBW_WORKER_OMP_THREADS``,
    else 4. Aim for ``(n_envs - K) * fat_threads <=`` spare physical cores.
    """
    import os as _os
    import torch as _torch

    lean_raw = (_os.environ.get("AWBW_N_LEAN_WORKERS") or "").strip()
    n_lean = int(lean_raw) if lean_raw.isdigit() else 0
    n_lean = min(max(n_lean, 0), 256)

    def _apply_explicit_threads(wt: int) -> None:
        wt = min(max(int(wt), 1), 64)
        _os.environ["OMP_NUM_THREADS"] = str(wt)
        _os.environ["MKL_NUM_THREADS"] = str(wt)
        _os.environ["OPENBLAS_NUM_THREADS"] = str(wt)
        _os.environ["NUMEXPR_NUM_THREADS"] = str(wt)
        try:
            _torch.set_num_threads(int(wt))
        except Exception:
            pass

    if n_lean <= 0:
        wt_raw = (_os.environ.get("AWBW_WORKER_OMP_THREADS") or "").strip()
        if wt_raw.isdigit() and int(wt_raw) >= 1:
            _apply_explicit_threads(int(wt_raw))
        else:
            _os.environ.setdefault("OMP_NUM_THREADS", "1")
            _os.environ.setdefault("MKL_NUM_THREADS", "1")
            _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            _os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
            try:
                _torch.set_num_threads(1)
            except Exception:
                pass
        return

    if worker_index < n_lean:
        _apply_explicit_threads(1)
        return

    fat_raw = (_os.environ.get("AWBW_CPU_WORKER_THREADS") or "").strip()
    if not fat_raw.isdigit() or int(fat_raw) < 1:
        fat_raw = (_os.environ.get("AWBW_WORKER_OMP_THREADS") or "").strip()
    if fat_raw.isdigit() and int(fat_raw) >= 1:
        _apply_explicit_threads(int(fat_raw))
    else:
        _apply_explicit_threads(4)


def _opponent_inference_device_for_worker(worker_index: int) -> str:
    """
    Return ``\"cuda\"`` or ``\"cpu\"`` for **initial** checkpoint load (``_load_random``).

    When ``AWBW_GPU_OPPONENT_POOL=1``, always ``\"cpu\"`` (weights stay CPU; ``__call__``
    moves policy to CUDA only while a process-wide semaphore permit is held).

    Otherwise: ``AWBW_ALLOW_CUDA_OPPONENT`` + ``AWBW_OPPONENT_CUDA_WORKERS`` fixed-index routing.
    """
    if gpu_opponent_pool_enabled():
        return "cpu"
    global _OPPONENT_CUDA_CAP_WARNED
    gate = (os.environ.get("AWBW_ALLOW_CUDA_OPPONENT") or "").strip().lower()
    if gate not in ("1", "true", "yes", "on"):
        return "cpu"
    raw_n = (os.environ.get("AWBW_OPPONENT_CUDA_WORKERS") or "").strip()
    n_gpu = int(raw_n) if raw_n.isdigit() else 0
    if n_gpu <= 0:
        return "cpu"
    cap = int(OPPONENT_CUDA_WORKERS_MAX)
    if n_gpu > cap:
        if not _OPPONENT_CUDA_CAP_WARNED:
            print(
                f"[self_play] AWBW_OPPONENT_CUDA_WORKERS>{cap} is unsupported; capping at {cap} "
                "(VRAM / single-GPU contention)."
            )
            _OPPONENT_CUDA_CAP_WARNED = True
        n_gpu = cap
    if worker_index >= n_gpu:
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    return "cuda"


def _resolve_live_snapshot_pkl_path(live_snapshot_dir: Path, games_id: int) -> str:
    """
    Prefer ``<dir>/{games_id}.pkl`` (default live training layout).  Else use
    ``<dir>/{games_id}/engine_snapshot.pkl`` (e.g. ``replays/amarinner_my_games/`` from
    ``tools/amarriner_export_my_games_replays.py``).
    """
    root = Path(live_snapshot_dir)
    flat = root / f"{int(games_id)}.pkl"
    if flat.is_file():
        return str(flat)
    nested = root / str(int(games_id)) / "engine_snapshot.pkl"
    if nested.is_file():
        return str(nested)
    return str(flat)


class _PicklableEnvFactory:
    """
    Zero-arg env builder stored as a class so instances are picklable on Windows
    ``spawn`` (``multiprocessing.Process`` / async actors). Nested ``def`` factories
    are not picklable: ``Can't get local object '_make_env_factory.<locals>._init'``.
    """

    __slots__ = (
        "map_pool",
        "checkpoint_dir",
        "co_p0",
        "co_p1",
        "tier_name",
        "curriculum_broad_prob",
        "curriculum_tag",
        "opponent_mix",
        "pool_from_fleet",
        "cold_opponent",
        "fleet_opponent_root",
        "max_env_steps",
        "max_p1_microsteps",
        "live_snapshot_path",
        "live_games_id",
        "live_learner_seat",
        "live_fallback_curriculum",
        "worker_index",
        "gpu_infer_semaphore",
        "opponent_force_cpu",
    )

    def __init__(
        self,
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
        max_env_steps: int | None = None,
        max_p1_microsteps: int | None = None,
        live_snapshot_path: str | None = None,
        live_games_id: int | None = None,
        live_learner_seat: int = 0,
        live_fallback_curriculum: bool = True,
        worker_index: int = 0,
        gpu_infer_semaphore: Any = None,
        opponent_force_cpu: bool = False,
    ) -> None:
        self.map_pool = map_pool
        self.checkpoint_dir = checkpoint_dir
        self.co_p0 = co_p0
        self.co_p1 = co_p1
        self.tier_name = tier_name
        self.curriculum_broad_prob = curriculum_broad_prob
        self.curriculum_tag = curriculum_tag
        self.opponent_mix = opponent_mix
        self.pool_from_fleet = pool_from_fleet
        self.cold_opponent = cold_opponent
        self.fleet_opponent_root = fleet_opponent_root
        self.max_env_steps = max_env_steps
        self.max_p1_microsteps = max_p1_microsteps
        self.live_snapshot_path = live_snapshot_path
        self.live_games_id = live_games_id
        self.live_learner_seat = live_learner_seat
        self.live_fallback_curriculum = live_fallback_curriculum
        self.worker_index = worker_index
        self.gpu_infer_semaphore = gpu_infer_semaphore
        self.opponent_force_cpu = bool(opponent_force_cpu)

    def __call__(self) -> Any:
        _install_subproc_worker_excepthook()
        _subproc_worker_thread_env(int(self.worker_index))

        from rl.env import AWBWEnv, LEARNER_SEAT_ENV
        from sb3_contrib.common.wrappers import ActionMasker  # type: ignore[import]

        if self.live_snapshot_path:
            os.environ[LEARNER_SEAT_ENV] = str(int(self.live_learner_seat) & 1)
        tag = self.curriculum_tag
        if self.live_games_id is not None and tag is None:
            tag = f"live-gid-{int(self.live_games_id)}"
        opp_dev = (
            "cpu"
            if self.opponent_force_cpu
            else _opponent_inference_device_for_worker(int(self.worker_index))
        )
        opponent = _CheckpointOpponent(
            self.checkpoint_dir,
            opponent_mix=self.opponent_mix,
            pool_from_fleet=self.pool_from_fleet,
            cold_opponent=self.cold_opponent,
            fleet_opponent_root=self.fleet_opponent_root,
            inference_device=opp_dev,
            gpu_infer_semaphore=self.gpu_infer_semaphore,
        )
        env = AWBWEnv(
            map_pool=self.map_pool,
            opponent_policy=opponent,
            render_mode=None,
            co_p0=self.co_p0,
            co_p1=self.co_p1,
            tier_name=self.tier_name,
            curriculum_broad_prob=self.curriculum_broad_prob,
            curriculum_tag=tag,
            max_env_steps=self.max_env_steps,
            max_p1_microsteps=self.max_p1_microsteps,
            live_snapshot_path=self.live_snapshot_path,
            live_games_id=self.live_games_id,
            live_fallback_curriculum=self.live_fallback_curriculum,
        )
        opponent.attach_env(env)
        return ActionMasker(env, _mask_fn)


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
    max_env_steps: int | None = None,
    max_p1_microsteps: int | None = None,
    live_snapshot_path: str | None = None,
    live_games_id: int | None = None,
    live_learner_seat: int = 0,
    live_fallback_curriculum: bool = True,
    worker_index: int = 0,
    gpu_infer_semaphore: Any = None,
    opponent_force_cpu: bool = False,
) -> Callable[[], Any]:
    """Return a picklable zero-arg env factory (SubprocVecEnv, async IMPALA actors)."""
    return _PicklableEnvFactory(
        map_pool,
        checkpoint_dir,
        co_p0=co_p0,
        co_p1=co_p1,
        tier_name=tier_name,
        curriculum_broad_prob=curriculum_broad_prob,
        curriculum_tag=curriculum_tag,
        opponent_mix=opponent_mix,
        pool_from_fleet=pool_from_fleet,
        cold_opponent=cold_opponent,
        fleet_opponent_root=fleet_opponent_root,
        max_env_steps=max_env_steps,
        max_p1_microsteps=max_p1_microsteps,
        live_snapshot_path=live_snapshot_path,
        live_games_id=live_games_id,
        live_learner_seat=live_learner_seat,
        live_fallback_curriculum=live_fallback_curriculum,
        worker_index=worker_index,
        gpu_infer_semaphore=gpu_infer_semaphore,
        opponent_force_cpu=opponent_force_cpu,
    )


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
            # Phase 6b (FPS / iter-5 cliff): baseline RSS and elapsed — set once
            # across the first ``learn()`` chunk only (``_on_training_start`` fires
            # every chunk in ``SelfPlayTrainer``).
            self._diag_perf_t0: float | None = None
            self._diag_initial_rss_mb: float | None = None
            self._diag_rollout_seq: int = 0
            self._diag_last_worker_scan_mono: float = 0.0
            self._diag_cached_sum_worker_rss_mb: float = 0.0

        def _on_training_start(self) -> None:
            if self._diag_perf_t0 is None:
                self._diag_perf_t0 = time.perf_counter()
            if self._diag_initial_rss_mb is None:
                self._diag_initial_rss_mb = _main_proc_rss_mb()

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
            env_collect_s = 0.0
            env_steps_per_s_collect = 0.0
            env_steps_per_s_total = 0.0
            ppo_update_s_out: float | None = None
            if self._t_rollout_start is not None:
                env_collect_s = max(0.0, now - self._t_rollout_start)
                self.logger.record("diag/env_collect_s", env_collect_s)
                if env_collect_s > 0 and self._steps_in_rollout > 0:
                    env_steps_per_s_collect = self._steps_in_rollout / env_collect_s
                    self.logger.record(
                        "diag/env_steps_per_s_collect",
                        env_steps_per_s_collect,
                    )
                if self._last_ppo_update_s is not None:
                    ppo_update_s_out = self._last_ppo_update_s
                    self.logger.record("diag/ppo_update_s", self._last_ppo_update_s)
                    cycle_s = env_collect_s + self._last_ppo_update_s
                    if cycle_s > 0 and self._steps_in_rollout > 0:
                        env_steps_per_s_total = self._steps_in_rollout / cycle_s
                        self.logger.record(
                            "diag/env_steps_per_s_total",
                            env_steps_per_s_total,
                        )

            # ── Phase 6b: memory + per-worker step skew (rate-limited where needed) ──
            main_rss = _main_proc_rss_mb()
            base = self._diag_initial_rss_mb
            delta_mb = main_rss - base if base is not None else 0.0
            self.logger.record("diag/main_proc_rss_mb", main_rss)
            self.logger.record("diag/main_proc_rss_delta_mb", delta_mb)

            sys_pct = _system_ram_used_pct()
            self.logger.record("diag/system_ram_used_pct", sys_pct)

            if now - self._diag_last_worker_scan_mono >= _WORKER_RSS_RESAMPLE_S:
                self._diag_cached_sum_worker_rss_mb = _sum_python_children_rss_mb()
                self._diag_last_worker_scan_mono = now
            sum_worker = self._diag_cached_sum_worker_rss_mb
            self.logger.record("diag/sum_worker_rss_mb", sum_worker)

            track_steps = effective_track_per_worker_times()
            p99_max = 0.0
            p99_min = 0.0
            mean_p99 = 0.0
            if track_steps and self.training_env is not None:
                try:
                    raw_list = self.training_env.env_method("get_step_time_stats")
                except Exception:
                    raw_list = []
                p99_vals: list[float] = []
                for row in raw_list or []:
                    if not isinstance(row, dict):
                        continue
                    if int(row.get("count", 0)) <= 0:
                        continue
                    p99_vals.append(float(row.get("p99", 0.0)))
                if p99_vals:
                    p99_max = max(p99_vals)
                    p99_min = min(p99_vals)
                    mean_p99 = sum(p99_vals) / len(p99_vals)
            self.logger.record("diag/per_worker_step_time_s_p99", mean_p99)
            self.logger.record(
                "diag/worker_step_time_p99_max_across_envs", p99_max
            )
            self.logger.record(
                "diag/worker_step_time_p99_min_across_envs", p99_min
            )

            self._diag_rollout_seq += 1
            t_elapsed = (
                now - self._diag_perf_t0
                if self._diag_perf_t0 is not None
                else 0.0
            )
            n_envs_tb = 0
            if self.training_env is not None:
                try:
                    n_envs_tb = int(self.training_env.num_envs)
                except Exception:
                    n_envs_tb = 0

            json_row = {
                "schema_version": "1.0",
                "iteration": int(self._diag_rollout_seq),
                "total_timesteps": int(getattr(self, "num_timesteps", 0) or 0),
                "time_elapsed_s": float(t_elapsed),
                "env_collect_s": float(env_collect_s),
                "ppo_update_s": float(ppo_update_s_out)
                if ppo_update_s_out is not None
                else None,
                "env_steps_per_s_collect": float(env_steps_per_s_collect),
                "env_steps_per_s_total": float(env_steps_per_s_total),
                "main_proc_rss_mb": float(main_rss),
                "main_proc_rss_delta_mb": float(delta_mb),
                "sum_worker_rss_mb": float(sum_worker),
                "system_ram_used_pct": float(sys_pct),
                "worker_step_time_p99_max_s": float(p99_max),
                "worker_step_time_p99_min_s": float(p99_min),
                "n_envs": int(n_envs_tb),
                "machine_id": os.environ.get("AWBW_MACHINE_ID"),
            }
            try:
                _append_fps_diag_line(json_row)
            except Exception:
                pass

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
        (by default) opponent inference on CPU.         Optional CUDA checkpoint opponents: ``AWBW_ALLOW_CUDA_OPPONENT=1`` plus either
        ``AWBW_GPU_OPPONENT_POOL=1`` (semaphore, any worker; size ``AWBW_GPU_OPPONENT_POOL_SIZE``)
        or legacy fixed indices ``AWBW_OPPONENT_CUDA_WORKERS``. More workers raise
        throughput but cost ~2-3 GB host RAM each.

        Step-sync note: the default ``SubprocVecEnv`` waits for the slowest worker
        each ``VecEnv.step()``. Set ``AWBW_ASYNC_VEC=1`` to try
        ``gymnasium.vector.AsyncVectorEnv`` (experimental with MaskablePPO; falls
        back to SubprocVecEnv if construction fails).

        CPU thread caps: ``AWBW_N_LEAN_WORKERS`` / ``AWBW_CPU_WORKER_THREADS`` (legacy split) or
        uniform ``AWBW_WORKER_OMP_THREADS`` when using the GPU opponent pool bootstrap.
    device : str
        Torch device for the **learner** — "cuda", "cpu", or "auto".
        Opponent inference defaults to CPU (minimal rollout buffer
        ``n_envs=1, n_steps=1``). CUDA opponents duplicate policy VRAM per
        worker. The learner's VRAM footprint scales with
        ``n_steps * n_envs * obs_shape``; reduce ``batch_size`` first
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
        Ignored when ``checkpoint_curate`` is True (K/M/D curator cap instead).
    checkpoint_curate : bool
        Use :func:`rl.fleet_env.prune_checkpoint_zip_curated` instead of FIFO.
    curator_k_newest, curator_m_top_winrate, curator_d_diversity : int
        Curator pool geometry (see ``prune_checkpoint_zip_curated``).
    curator_min_age_minutes : float
        Never delete zips newer than this age (minutes).
    verdicts_root : Path | None
        Parent ``fleet/`` directory with per-machine ``*/eval/*.json`` verdicts;
        ``None`` makes the curator fall back to FIFO until set.
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
        checkpoint_curate: bool = False,
        curator_k_newest: int = 8,
        curator_m_top_winrate: int = 12,
        curator_d_diversity: int = 4,
        curator_min_age_minutes: float = 5.0,
        verdicts_root: Optional[Path | str] = None,
        load_promoted: bool = False,
        bc_init: Optional[Path | str] = None,
        cold_opponent: str = "random",
        local_checkpoint_mirror: Optional[Path | str] = None,
        publisher_queue_max: int = 4,
        publisher_drain_timeout_s: float = 60.0,
        fleet_cfg: Optional["FleetConfig"] = None,  # type: ignore[name-defined]  # noqa: F821
        opponent_refresh_rollouts: int = 4,
        hot_reload_enabled: bool = False,
        hot_reload_min_steps_done: int = 0,
        mcts_mode: str = "off",
        mcts_sims: int = 16,
        mcts_c_puct: float = 1.5,
        mcts_dirichlet_alpha: float = 0.3,
        mcts_dirichlet_epsilon: float = 0.25,
        mcts_temperature: float = 1.0,
        mcts_min_depth: int = 4,
        mcts_root_plans: int = 8,
        mcts_max_plan_actions: int = 256,
        max_env_steps: int | None = 10000,
        max_p1_microsteps: int | None = 4000,
        live_games_id: list[int] | None = None,
        live_learner_seats: list[int] | None = None,
        live_snapshot_dir: Path | str | None = None,
        training_backend: str = "sync",
        async_unroll_length: int | None = None,
        async_learner_batch: int | None = None,
        async_queue_max: int = 64,
        async_gpu_opponent_permits_subtract: int = 2,
        async_learner_forward_chunk: int | None = None,
        async_clip_rho: float = 1.0,
        async_clip_pg_rho: float = 1.0,
        async_gamma: float = 0.99,
        async_learning_rate: float = 3e-4,
        async_vf_coef: float = 0.5,
        async_max_grad_norm: float = 0.5,
        async_weight_save_every: int = 1,
        async_log_rho_floor: float = -20.0,
    ) -> None:
        tb = str(training_backend or "sync").strip().lower()
        if tb not in ("sync", "async"):
            raise ValueError("training_backend must be 'sync' or 'async'")
        self.training_backend = tb

        roll = n_steps * n_envs
        if self.training_backend == "sync" and batch_size > roll:
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
        self.checkpoint_curate = bool(checkpoint_curate)
        self.curator_k_newest = int(curator_k_newest)
        self.curator_m_top_winrate = int(curator_m_top_winrate)
        self.curator_d_diversity = int(curator_d_diversity)
        self.curator_min_age_minutes = float(curator_min_age_minutes)
        self.verdicts_root = (
            Path(verdicts_root).resolve() if verdicts_root is not None else None
        )
        self.load_promoted = bool(load_promoted)
        self.bc_init = Path(bc_init).resolve() if bc_init else None
        cold = str(cold_opponent or "random").strip().lower()
        if cold not in _VALID_COLD_OPPONENTS:
            raise ValueError(
                f"cold_opponent must be one of {_VALID_COLD_OPPONENTS}; got {cold_opponent!r}"
            )
        self.cold_opponent = cold
        self.fleet_cfg = fleet_cfg
        self.opponent_refresh_rollouts = max(0, int(opponent_refresh_rollouts))
        self.hot_reload_enabled = bool(hot_reload_enabled)
        self.hot_reload_min_steps_done = int(hot_reload_min_steps_done)
        _mm = str(mcts_mode or "off").strip().lower()
        if _mm not in ("off", "eval_only"):
            raise ValueError("mcts_mode must be 'off' or 'eval_only'")
        self.mcts_mode = _mm
        self.mcts_sims = int(mcts_sims)
        self.mcts_c_puct = float(mcts_c_puct)
        self.mcts_dirichlet_alpha = float(mcts_dirichlet_alpha)
        self.mcts_dirichlet_epsilon = float(mcts_dirichlet_epsilon)
        self.mcts_temperature = float(mcts_temperature)
        self.mcts_min_depth = int(mcts_min_depth)
        self.mcts_root_plans = int(mcts_root_plans)
        self.mcts_max_plan_actions = int(mcts_max_plan_actions)
        def _cap_or_none(val: int | None) -> int | None:
            if val is None:
                return None
            iv = int(val)
            return None if iv <= 0 else iv

        self.max_env_steps = _cap_or_none(max_env_steps)
        self.max_p1_microsteps = _cap_or_none(max_p1_microsteps)
        self.live_games_id: list[int] = [int(x) for x in (live_games_id or [])]
        n_live = len(self.live_games_id)
        if n_live and n_envs < n_live:
            raise ValueError(
                f"n_envs ({n_envs}) must be >= number of live games_id ({n_live})"
            )
        self.live_learner_seats: list[int] | None = None
        if n_live:
            if live_learner_seats is not None:
                ls = [int(x) & 1 for x in live_learner_seats]
                if len(ls) != n_live:
                    raise ValueError(
                        "live_learner_seats must have the same length as live_games_id"
                    )
                self.live_learner_seats = ls
            else:
                self.live_learner_seats = [0] * n_live
        self.live_snapshot_dir: Path = (
            Path(live_snapshot_dir).resolve()
            if live_snapshot_dir
            else (ROOT / ".tmp" / "awbw_live_snapshot")
        )
        _unroll = int(async_unroll_length) if async_unroll_length is not None else int(n_steps)
        self.async_unroll_length = max(4, _unroll)
        self.async_learner_batch_explicit = async_learner_batch is not None
        if async_learner_batch is not None:
            self.async_learner_batch = int(async_learner_batch)
        elif self.training_backend == "async":
            # Default ``roll`` is often too large for one MaskablePPO forward on ~12GB GPUs.
            # Also cap at ``n_envs * unroll`` so the learner tends to consume one wave of actor
            # chunks per update (reduces queue backpressure / actors blocked on ``queue.put``).
            _cap = _default_async_learner_transitions_cap()
            _one_wave = min(roll, self.n_envs * self.async_unroll_length)
            self.async_learner_batch = min(
                roll, max(self.async_unroll_length, _cap), _one_wave
            )
        else:
            self.async_learner_batch = roll
        self.async_learner_batch = max(self.async_unroll_length, self.async_learner_batch)
        self.async_learner_batch = _snap_async_learner_batch_to_unroll(
            self.async_learner_batch, self.async_unroll_length, roll
        )
        self.async_queue_max = max(2, int(async_queue_max))
        self.async_gpu_opponent_permits_subtract = max(
            0, int(async_gpu_opponent_permits_subtract)
        )
        self.async_learner_forward_chunk = (
            int(async_learner_forward_chunk)
            if async_learner_forward_chunk is not None
            else None
        )
        if (
            self.async_learner_forward_chunk is not None
            and self.async_learner_forward_chunk <= 0
        ):
            self.async_learner_forward_chunk = None
        self.async_clip_rho = float(async_clip_rho)
        self.async_clip_pg_rho = float(async_clip_pg_rho)
        self.async_gamma = float(async_gamma)
        self.async_learning_rate = float(async_learning_rate)
        self.async_vf_coef = float(async_vf_coef)
        self.async_max_grad_norm = float(async_max_grad_norm)
        self.async_weight_save_every = max(1, int(async_weight_save_every))
        self.async_log_rho_floor = float(async_log_rho_floor)
        self._gpu_pool_manager: Any | None = None
        self._rollout_index: int = 0
        self._applied_reload_requests: set[tuple[str, str | int | None]] = set()
        self.local_checkpoint_mirror = (
            Path(local_checkpoint_mirror).resolve() if local_checkpoint_mirror is not None else None
        )
        self.publisher_queue_max = int(publisher_queue_max)
        self.publisher_drain_timeout_s = float(publisher_drain_timeout_s)
        # One-time warn for permissive auxiliary heartbeat path; see _write_trainer_status.
        self._heartbeat_machine_id_warned = False

        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._publisher = None  # type: Optional["CheckpointPublisher"]
        if self.local_checkpoint_mirror is not None:
            from rl.checkpoint_publisher import CheckpointPublisher

            self.local_checkpoint_mirror.mkdir(parents=True, exist_ok=True)
            self._publisher = CheckpointPublisher(
                local_mirror_dir=self.local_checkpoint_mirror,
                shared_dir=self.checkpoint_dir,
                queue_max=publisher_queue_max,
                drain_timeout_s=publisher_drain_timeout_s,
            )

        self.map_pool = load_map_pool()
        if self.map_id_filter is not None:
            self.map_pool = [m for m in self.map_pool if m["map_id"] == self.map_id_filter]
            if not self.map_pool:
                raise ValueError(f"No maps found with map_id={self.map_id_filter}")

        from rl.fleet_env import (
            prune_checkpoint_zip_curated,
            prune_checkpoint_zip_snapshots,
            sorted_checkpoint_zip_paths,
        )

        if self.checkpoint_curate:
            summary = prune_checkpoint_zip_curated(
                self.checkpoint_dir,
                k_newest=self.curator_k_newest,
                m_top_winrate=self.curator_m_top_winrate,
                d_diversity=self.curator_d_diversity,
                verdicts_root=self.verdicts_root,
                min_age_minutes=self.curator_min_age_minutes,
                dry_run=False,
            )
            if summary["removed"]:
                print(
                    f"[self_play] Curated pool: kept {summary['kept_total']}, "
                    f"removed {len(summary['removed'])} "
                    f"(fallback={summary['fallback_used']})"
                )
        elif self.checkpoint_zip_cap > 0:
            pruned = prune_checkpoint_zip_snapshots(self.checkpoint_dir, self.checkpoint_zip_cap)
            if pruned:
                print(
                    f"[self_play] Pruned {pruned} old checkpoint_*.zip "
                    f"(cap={self.checkpoint_zip_cap})"
                )
        self.checkpoints = sorted_checkpoint_zip_paths(self.checkpoint_dir)
        self.elo_ratings: dict[str, float] = {"latest": 1200.0}

    def analyze_precision_tradeoffs(self) -> None:
        """Stub for precision tradeoff analysis (placeholder)."""
        pass

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
            max_env_steps=self.max_env_steps,
            max_p1_microsteps=self.max_p1_microsteps,
        )
        if self.n_envs > 1:
            from stable_baselines3.common.vec_env import SubprocVecEnv  # type: ignore[import]

            gpu_sem = None
            if gpu_opponent_pool_enabled():
                if self._gpu_pool_manager is None:
                    from multiprocessing import Manager  # noqa: PLC0415

                    self._gpu_pool_manager = Manager()
                gpu_sem = self._gpu_pool_manager.BoundedSemaphore(
                    gpu_opponent_pool_permits()
                )
                print(
                    f"[self_play] GPU opponent semaphore (manager proxy): {gpu_opponent_pool_permits()} "
                    "concurrent CUDA checkpoint forwards (pool)"
                )

            n_live = len(self.live_games_id)
            self.live_snapshot_dir.mkdir(parents=True, exist_ok=True)
            seats = self.live_learner_seats or [0] * n_live
            factories: list[Callable[[], Any]] = []
            for i in range(self.n_envs):
                if i < n_live:
                    gid = int(self.live_games_id[i])
                    spath = _resolve_live_snapshot_pkl_path(self.live_snapshot_dir, gid)
                    print(
                        f"[self_play] live env {i}: games_id={gid} snapshot={spath} "
                        f"learner_seat={(seats[i] if i < len(seats) else 0)}"
                    )
                    factories.append(
                        _make_env_factory(
                            self.map_pool,
                            str(self.checkpoint_dir),
                            live_snapshot_path=spath,
                            live_games_id=gid,
                            live_learner_seat=seats[i],
                            worker_index=i,
                            gpu_infer_semaphore=gpu_sem,
                            **env_kw,
                        )
                    )
                else:
                    factories.append(
                        _make_env_factory(
                            self.map_pool,
                            str(self.checkpoint_dir),
                            worker_index=i,
                            gpu_infer_semaphore=gpu_sem,
                            **env_kw,
                        )
                    )
            use_async = (os.environ.get("AWBW_ASYNC_VEC", "") or "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            vec_env = None
            if use_async:
                try:
                    from gymnasium.vector import AsyncVectorEnv  # type: ignore[import]

                    # NOTE: AsyncVectorEnv is incompatible with SB3's MaskablePPO.set_env()
                    # which tries to wrap the env and fails. Fall back to SubprocVecEnv.
                    # Kept here for future reference if SB3 support improves.
                    vec_env = AsyncVectorEnv(factories, shared_memory=True)
                    print(f"[self_play] AsyncVectorEnv: {self.n_envs} workers (shared_memory=True) - NOTE: incompatible with SB3, falling back")
                except Exception as exc:
                    print(f"[self_play] AsyncVectorEnv failed ({exc}); using SubprocVecEnv")
            if vec_env is None:
                vec_env = SubprocVecEnv(factories, start_method="spawn")
                _wrap_subproc_vec_env_worker_ipc(vec_env, n_envs=self.n_envs)
                print(f"[self_play] SubprocVecEnv: {self.n_envs} workers (spawn)")
            return vec_env
        else:
            from rl.env import AWBWEnv, LEARNER_SEAT_ENV

            n_live = len(self.live_games_id)
            if n_live > 1:
                raise ValueError("with n_envs=1, pass at most one --live-games-id")
            self.live_snapshot_dir.mkdir(parents=True, exist_ok=True)
            lpath = (
                _resolve_live_snapshot_pkl_path(
                    self.live_snapshot_dir, int(self.live_games_id[0])
                )
                if n_live
                else None
            )
            lgid = int(self.live_games_id[0]) if n_live else None
            lseat = (self.live_learner_seats or [0])[0] if n_live else 0
            if n_live:
                os.environ[LEARNER_SEAT_ENV] = str(int(lseat) & 1)
            ctag = self.curriculum_tag
            if n_live and ctag is None:
                ctag = f"live-gid-{lgid}"
            opp_dev = _opponent_inference_device_for_worker(0)
            opponent = _CheckpointOpponent(
                str(self.checkpoint_dir),
                opponent_mix=self.opponent_mix,
                pool_from_fleet=self.pool_from_fleet,
                cold_opponent=self.cold_opponent,
                fleet_opponent_root=self.fleet_opponent_root,
                inference_device=opp_dev,
            )
            env = AWBWEnv(
                map_pool=self.map_pool,
                opponent_policy=opponent,
                render_mode=None,
                co_p0=self.co_p0,
                co_p1=self.co_p1,
                tier_name=self.tier_name,
                curriculum_broad_prob=self.curriculum_broad_prob,
                curriculum_tag=ctag,
                max_env_steps=self.max_env_steps,
                max_p1_microsteps=self.max_p1_microsteps,
                live_snapshot_path=lpath,
                live_games_id=lgid,
            )
            opponent.attach_env(env)
            return ActionMasker(env, _mask_fn)

    # ── Fleet heartbeat ───────────────────────────────────────────────────────

    def _write_trainer_status(
        self,
        *,
        steps_done: int,
        rate: float,
    ) -> None:
        """Write ``fleet/<id>/status.json`` heartbeat for orchestrator stuck-worker detection.

        Called once per outer training cycle after the freshly-published
        ``latest.zip``. Permissive: missing fleet_cfg or aux machine_id is a
        no-op (with a single one-time warn for the aux case). OSError on the
        write itself is swallowed and logged — Samba shares hiccup, and we
        will not crash a multi-day training run on one failed status write.
        """
        cfg = self.fleet_cfg
        if cfg is None:
            return

        from rl.fleet_env import write_status_json

        if cfg.role == "auxiliary":
            if not cfg.machine_id:
                if not self._heartbeat_machine_id_warned:
                    print(
                        "[self_play] heartbeat skipped: auxiliary role without "
                        "AWBW_MACHINE_ID; orchestrator stuck-worker detection "
                        "will not see this trainer."
                    )
                    self._heartbeat_machine_id_warned = True
                return
            if cfg.shared_root is None:
                return
            status_path = Path(cfg.shared_root) / "fleet" / cfg.machine_id / "status.json"
        else:
            machine_dir = cfg.machine_id or "main"
            status_path = Path(cfg.repo_root) / "fleet" / machine_dir / "status.json"

        extra = {
            "steps_done": int(steps_done),
            "n_envs": int(self.n_envs),
            "save_every": int(self.save_every),
            "checkpoint_dir": str(self.checkpoint_dir),
            "rate_steps_per_s": float(rate),
        }
        try:
            write_status_json(
                status_path,
                role=cfg.role,
                machine_id=cfg.machine_id,
                task="train",
                current_target=str(self.checkpoint_dir / "latest.zip"),
                extra=extra,
            )
        except OSError as exc:
            print(f"[self_play] heartbeat write failed ({status_path}): {exc}")

    def _maybe_handle_rollout_boundary(
        self,
        model: Any,
        vec_env: Any,
        steps_done: int,
    ) -> tuple[Any, Any]:
        """Phase 10c + 10d: between-rollout fleet hooks.

        Safe to call unconditionally — does nothing when both features are
        off. Called from the main training loop AFTER the heartbeat write
        and BEFORE the prune block.

        Opponent pool refresh uses ``model.env`` (the SB3 :class:`VecEnv`),
        not the pre-wrap ``_vec_env`` — when ``n_envs==1`` the local ref is
        a raw :class:`gymnasium.Wrapper` without ``env_method``.

        Returns ``(model, vec_env)`` so in-process reconfig can swap the
        learner and the vectorized environment.
        """
        self._rollout_index += 1

        if self.opponent_refresh_rollouts > 0 and (
            self._rollout_index % self.opponent_refresh_rollouts
        ) == 0:
            vec = getattr(model, "env", None)
            try:
                if vec is not None and hasattr(vec, "env_method"):
                    results = vec.env_method("reload_opponent_pool")
                else:
                    results = []
                n_refreshed = sum(1 for r in results if r is not None)
                print(
                    f"[self_play] Opponent pool refreshed across {n_refreshed} env(s)"
                )
            except Exception as exc:
                print(f"[self_play] Opponent refresh failed: {exc}")

        if self.hot_reload_enabled and self.fleet_cfg is not None:
            self._maybe_apply_reload_request(model, steps_done)

        return self._maybe_apply_train_reconfig_request(model, vec_env, steps_done)

    def _maybe_apply_reload_request(self, model: Any, steps_done: int) -> None:
        """Phase 10d: check for and apply a fleet reload request.

        Reads ``<shared>/fleet/<machine_id>/reload_request.json`` and, if
        valid, calls ``set_parameters`` on the learner, then renames the
        request to ``reload_request.applied.<unix_ts>.json`` (atomic where
        the host OS allows).
        """
        cfg = self.fleet_cfg
        if cfg is None or cfg.shared_root is None:
            return

        machine_id = (
            cfg.machine_id
            if cfg.machine_id
            else os.environ.get("AWBW_MACHINE_ID")
        )
        if not machine_id:
            return

        fleet_dir = Path(cfg.shared_root) / "fleet" / str(machine_id)
        request_path = fleet_dir / "reload_request.json"
        if not request_path.is_file():
            return
        try:
            with open(request_path, encoding="utf-8") as f:
                req = json.load(f)
        except OSError as exc:
            print(f"[hot-reload] Could not read {request_path}: {exc}")
            return
        except json.JSONDecodeError as exc:
            print(f"[hot-reload] Could not read {request_path}: {exc}")
            return

        target_zip = req.get("target_zip")
        min_steps = int(req.get("min_steps_done", 0) or 0)
        issued_at = req.get("issued_at")

        if not target_zip or not Path(str(target_zip)).is_file():
            print(f"[hot-reload] Skipping: target_zip missing/unreadable: {target_zip!r}")
            return
        if steps_done < min_steps:
            return
        if steps_done < self.hot_reload_min_steps_done:
            return

        key = (str(target_zip), issued_at)
        if key in self._applied_reload_requests:
            return

        try:
            model.set_parameters(
                str(target_zip), exact_match=True, device=self.device
            )
        except Exception as exc:
            print(f"[hot-reload] set_parameters failed: {exc}")
            return

        self._applied_reload_requests.add(key)
        print(
            f"[hot-reload] Applied weights from {target_zip} "
            f"(reason={req.get('reason', '?')}, steps_done={steps_done})"
        )

        ts = int(time.time())
        ack_path = fleet_dir / f"reload_request.applied.{ts}.json"
        try:
            os.replace(str(request_path), str(ack_path))
        except OSError as exc:
            print(f"[hot-reload] Ack rename failed (non-fatal): {exc}")

    def _maybe_apply_train_reconfig_request(
        self, model: Any, vec_env: Any, steps_done: int
    ) -> tuple[Any, Any]:
        """In-process PPO/VecEnv geometry swap (``train_reconfig_request.json``)."""
        cfg = self.fleet_cfg
        if cfg is None or not getattr(cfg, "shared_root", None):
            return model, vec_env
        machine_id = (
            cfg.machine_id
            if getattr(cfg, "machine_id", None)
            else os.environ.get("AWBW_MACHINE_ID")
        )
        if not machine_id:
            return model, vec_env

        fleet_dir = Path(cfg.shared_root) / "fleet" / str(machine_id)
        request_path = fleet_dir / "train_reconfig_request.json"
        if not request_path.is_file():
            return model, vec_env

        t0 = time.time()
        try:
            with open(request_path, encoding="utf-8") as f:
                req = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[train_reconfig] Could not read {request_path}: {exc}")
            return model, vec_env

        if not isinstance(req, dict):
            return model, vec_env
        request_id = str(req.get("request_id", "") or "")

        def _fail(msg: str) -> tuple[Any, Any]:
            print(f"[train_reconfig] Failed: {msg}")
            failed = fleet_dir / f"train_reconfig_request.failed.{int(time.time())}.json"
            try:
                payload = {
                    "request_id": request_id,
                    "error": msg,
                    "steps_done": steps_done,
                }
                failed.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                try:
                    request_path.unlink(missing_ok=True)
                except OSError:
                    pass
            except OSError as exc2:
                print(f"[train_reconfig] failed ack write: {exc2}")
            try:
                append_train_reconfig_line(
                    Path(cfg.shared_root),
                    {
                        "event": "soft_reconfig",
                        "source": "trainer",
                        "machine_id": str(machine_id),
                        "request_id": request_id,
                        "outcome": "failed",
                        "message": msg,
                    },
                )
            except OSError:
                pass
            return model, vec_env

        raw = req.get("args")
        if not isinstance(raw, dict):
            return _fail("args must be a dict")

        def _pi(key: str) -> int:
            v = raw.get(key)
            if v is None:
                raise KeyError(key)
            return int(v)

        try:
            if "--n-envs" in raw:
                self.n_envs = _pi("--n-envs")
            if "--n-steps" in raw:
                self.n_steps = _pi("--n-steps")
            if "--batch-size" in raw:
                self.batch_size = _pi("--batch-size")
        except (KeyError, TypeError, ValueError) as exc:
            return _fail(f"int parse: {exc}")

        n_e, n_s, b_s = self.n_envs, self.n_steps, self.batch_size
        if n_e < 1 or n_s < 1 or b_s < 1:
            return _fail("n_envs, n_steps, batch_size must be positive")
        if b_s > n_s * n_e:
            return _fail(f"batch_size {b_s} > n_steps*n_envs ({n_s * n_e})")

        from rl.ckpt_compat import load_maskable_ppo_compat

        new_vec = self._build_vec_env()
        try:
            if new_vec.observation_space != vec_env.observation_space:
                new_vec.close()
                return _fail("observation_space mismatch after reconfig (hard restart required)")
        except Exception as exc:  # noqa: BLE001
            try:
                new_vec.close()
            except OSError:
                pass
            return _fail(f"obs check: {exc}")

        _fd, _tp = tempfile.mkstemp(suffix=".zip", prefix="awbw_reconfig_")
        os.close(_fd)
        tmp_path = Path(_tp)
        try:
            model.save(str(tmp_path))
        except Exception as exc:  # noqa: BLE001
            new_vec.close()
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return _fail(f"model.save: {exc}")

        try:
            new_model = load_maskable_ppo_compat(
                tmp_path,
                env=new_vec,
                device=self.device,
                custom_objects={"n_steps": n_s, "batch_size": b_s},
            )
            new_model.tensorboard_log = str(LOGS_DIR)
        except Exception as exc:  # noqa: BLE001
            new_vec.close()
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return _fail(f"load: {exc}")

        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

        try:
            vec_env.close()
        except OSError as exc:
            print(f"[train_reconfig] old vec_env.close: {exc}")

        del model
        gc.collect()

        ts = int(time.time())
        ack_path = fleet_dir / f"train_reconfig_request.applied.{ts}.json"
        try:
            ack_body = {**req, "ack_at": ts, "steps_done": steps_done}
            ack_path.write_text(json.dumps(ack_body, indent=2), encoding="utf-8")
            request_path.unlink(missing_ok=True)
        except OSError as exc:
            print(f"[train_reconfig] applied ack rename: {exc}")

        dt_ms = int((time.time() - t0) * 1000)
        print(
            f"[train_reconfig] Applied request_id={request_id} "
            f"n_envs={n_e} n_steps={n_s} batch_size={b_s} ({dt_ms}ms)"
        )
        try:
            append_train_reconfig_line(
                Path(cfg.shared_root),
                {
                    "event": "soft_reconfig",
                    "source": "trainer",
                    "machine_id": str(machine_id),
                    "request_id": request_id,
                    "outcome": "applied",
                    "soft_reconfig_ms": dt_ms,
                    "steps_done": steps_done,
                },
            )
        except OSError:
            pass
        return new_model, new_vec

    # ── Main training loop ────────────────────────────────────────────────────

    def _save_checkpoint_with_publish(
        self,
        model,
        stem: str,
        *,
        also_publish_as_latest: bool = False,
    ) -> Path:
        """Phase 10a: route saves through the publisher when enabled.

        Default path (publisher None): preserves pre-Phase-10a semantics —
        direct ``_atomic_model_save`` to the shared checkpoint_dir. No
        behavior change.

        Publisher path: write to local mirror, enqueue async copy to
        shared. Returns the LOCAL path so callers can stat it for size
        if needed.
        """
        if self._publisher is None:
            target = self.checkpoint_dir / stem
            _atomic_model_save(model, target)
            if also_publish_as_latest:
                _atomic_model_save(model, self.checkpoint_dir / "latest")
            return self.checkpoint_dir / f"{stem}.zip"

        local_zip = self._publisher.save_and_publish(model, stem)
        if also_publish_as_latest:
            latest_local = self._publisher.save_and_publish(model, "latest")
            del latest_local
        return local_zip

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
        if self.training_backend == "async":
            print(
                f"[self_play] Starting | backend=async (IMPALA+V-trace) | steps={steps_msg} | "
                f"actors={self.n_envs} | unroll={self.async_unroll_length} "
                f"| learner_batch>={self.async_learner_batch} | queue_max={self.async_queue_max} | "
                f"device={self.device} | maps={len(self.map_pool)} (Std sampling: {n_std}){cur_msg}"
            )
        else:
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

        if self.training_backend == "async":
            from rl.async_impala import run_impala_training

            run_impala_training(self)
            return

        vec_env = self._build_vec_env()
        _apply_learner_cuda_perf_opts(self.device)

        lean_raw = (os.environ.get("AWBW_N_LEAN_WORKERS") or "").strip()
        if lean_raw.isdigit() and int(lean_raw) > 0:
            print(
                f"[self_play] AWBW_N_LEAN_WORKERS={lean_raw}: first {lean_raw} workers "
                "use 1 BLAS/torch thread; others use AWBW_CPU_WORKER_THREADS / "
                "AWBW_WORKER_OMP_THREADS / default 4."
            )
        if (os.environ.get("AWBW_ALLOW_CUDA_OPPONENT") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            if gpu_opponent_pool_enabled():
                psz = (os.environ.get("AWBW_GPU_OPPONENT_POOL_SIZE") or "").strip()
                print(
                    f"[self_play] CUDA opponents: semaphore pool size "
                    f"{psz or str(OPPONENT_CUDA_WORKERS_MAX)} (non-blocking GPU or CPU infer)."
                )
            else:
                ocn = (os.environ.get("AWBW_OPPONENT_CUDA_WORKERS") or "").strip()
                print(
                    f"[self_play] CUDA opponents enabled: AWBW_OPPONENT_CUDA_WORKERS="
                    f"{ocn or '0'} (first N worker indices; cap {OPPONENT_CUDA_WORKERS_MAX})."
                )

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

            # Pass env=None to avoid MaskablePPO.load wrapping AsyncVectorEnv (incompatible).
            # We set env afterwards via set_env() after loading completes.
            model = load_maskable_ppo_compat(
                resume_path,
                env=None,  # Load without env, set it after to avoid AsyncVectorEnv compat issue
                device=self.device,
                custom_objects={"n_steps": self.n_steps, "n_envs": self.n_envs},
            )
            model.set_env(vec_env)
            # Checkpoints from another machine (e.g. Main D:\) embed tensorboard_log in the zip;
            # always write TensorBoard under this repo's logs/.
            model.tensorboard_log = str(LOGS_DIR)
        elif self.bc_init is not None and self.bc_init.is_file():
            print(f"[self_play] Fresh run: warm-start from {self.bc_init}")
            from rl.ckpt_compat import load_maskable_ppo_compat

            model = load_maskable_ppo_compat(
                self.bc_init,
                env=None,  # Load without env, set it after
                device=self.device,
                custom_objects={
                    "n_steps": self.n_steps,
                    "n_envs": self.n_envs,
                    "batch_size": self.batch_size,
                },
            )
            model.set_env(vec_env)
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

        # Tier 1a: torch.compile for 20-40% inference speedup
        # reduce-overhead targets GPU kernel launch overhead in repeated small-batch inference
        # Needs a Triton build Inductor accepts (see torch.utils._triton.has_triton). On
        # Windows use `triton-windows` per requirements.txt; Linux CUDA wheels usually
        # include a compatible Triton.
        import torch
        from torch.cuda.amp import autocast as amp_autocast, GradScaler
        from torch.utils._triton import has_triton as _inductor_triton_ok

        if hasattr(torch, "compile") and self.device != "cpu":
            if not _windows_torch_compile_opt_in():
                print(
                    "[self_play] torch.compile skipped: Windows default off (Triton needs MSVC "
                    "on first Inductor run). Set AWBW_TORCH_COMPILE=1 and install VS Build Tools "
                    "with C++ to enable, or run training on Linux."
                )
            elif not _inductor_triton_ok():
                print(
                    "[self_play] torch.compile skipped: Triton/Inductor not usable for this install "
                    "(Windows: `pip install triton-windows` pinned to your PyTorch version, see "
                    "requirements.txt; Linux: use official PyTorch+CUDA wheels). Incompatible "
                    "triton-windows vs torch also fails this check."
                )
            else:
                try:
                    model.policy = torch.compile(
                        model.policy,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                    print("[self_play] torch.compile applied to policy (mode=reduce-overhead)")
                except Exception as e:
                    print(
                        "[self_play] torch.compile skipped: "
                        f"{_policy_torch_compile_skip_note(e)}"
                    )

        from rl.fleet_env import (
            new_checkpoint_stem_utc,
            prune_checkpoint_zip_curated,
            prune_checkpoint_zip_snapshots,
            sorted_checkpoint_zip_paths,
        )

        steps_done = 0
        t_start = time.time()
        cap = self.total_timesteps

        # Built once and reused across `model.learn` chunks so the
        # per-env running counters survive between rollouts.
        diag_callback = _build_diagnostics_callback()

        # Mixed precision training
        scaler = GradScaler()

        try:
            while cap is None or steps_done < cap:
                if cap is None:
                    chunk = self.save_every
                else:
                    chunk = min(self.save_every, cap - steps_done)
                elapsed = time.time() - t_start
                rate = steps_done / elapsed if elapsed > 0 else 0
                if cap is None:
                    eta_str = "-"
                else:
                    rem = cap - steps_done
                    eta = rem / rate if rate > 0 else float("inf")
                    eta_str = f"{eta / 60:.1f}min"
                print(
                    f"\n[self_play] Steps {steps_done:,}->{steps_done + chunk:,} | "
                    f"{rate:,.0f} steps/s | ETA {eta_str}"
                )
                
                # Wrap the learn call with autocast for mixed precision
                with amp_autocast():
                    model.learn(
                        total_timesteps=chunk,
                        reset_num_timesteps=False,
                        callback=diag_callback,
                    )
                    steps_done += chunk

                    ckpt_stem = new_checkpoint_stem_utc()
                    self._save_checkpoint_with_publish(
                        model, ckpt_stem, also_publish_as_latest=True
                    )
                    print(f"[self_play] Saved {ckpt_stem}.zip")

                    self._write_trainer_status(steps_done=steps_done, rate=rate)

                    model, vec_env = self._maybe_handle_rollout_boundary(
                        model, vec_env, steps_done
                    )

                    if self.checkpoint_curate:
                        summary = prune_checkpoint_zip_curated(
                            self.checkpoint_dir,
                            k_newest=self.curator_k_newest,
                            m_top_winrate=self.curator_m_top_winrate,
                            d_diversity=self.curator_d_diversity,
                            verdicts_root=self.verdicts_root,
                            min_age_minutes=self.curator_min_age_minutes,
                            dry_run=False,
                        )
                        if summary["removed"]:
                            print(
                                f"[self_play] Curated pool: kept {summary['kept_total']}, "
                                f"removed {len(summary['removed'])} "
                                f"(fallback={summary['fallback_used']})"
                            )
                    elif self.checkpoint_zip_cap > 0:
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
                saved = self._save_checkpoint_with_publish(
                    model, "latest", also_publish_as_latest=False
                )
                print(f"[self_play] Saved -> {saved}")
        finally:
            mgr = self._gpu_pool_manager
            if mgr is not None:
                try:
                    mgr.shutdown()
                except Exception:
                    pass
                self._gpu_pool_manager = None
            if self._publisher is not None:
                drained = self._publisher.drain(timeout_s=self.publisher_drain_timeout_s)
                print(
                    f"[self_play] checkpoint publisher drained {drained} pending publishes; "
                    f"{self._publisher.queue_depth} still pending."
                )
                self._publisher.close()

        total_elapsed = time.time() - t_start
        steps_note = f"{steps_done:,}" if cap is None else f"{self.total_timesteps:,}"
        final_path = self.checkpoint_dir / "latest.zip"
        if self._publisher is not None and self.local_checkpoint_mirror is not None:
            final_path = self.local_checkpoint_mirror / "latest.zip"
        print(
            f"\n[self_play] Done. {steps_note} steps in "
            f"{total_elapsed/60:.1f}min | Final -> {final_path}"
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
    _mks: dict = {"starting_funds": 0, "tier_name": tier_name}
    rfm = getattr(map_data, "replay_first_mover", None)
    if rfm is not None:
        _mks["replay_first_mover"] = int(rfm)
    state = make_initial_state(map_data, co_p0, co_p1, **_mks)
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
