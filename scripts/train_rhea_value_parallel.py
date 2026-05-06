from __future__ import annotations

"""Parallel RHEA value training.

This is the `--n-envs` version of the RHEA/value learner.

Architecture:
    N actor processes:
        AWBWEnv -> RHEA full-turn search -> transition queue

    Main process:
        transition queue -> replay buffer -> TD value learner -> checkpoint saves

This is not PPO VecEnv. The actors are independent RHEA self-play workers.
They periodically refresh their value net from the learner checkpoint if it
exists, but stale actor values are acceptable for the first parallel collector.
"""

import argparse
import copy
import dataclasses
import json
import multiprocessing as mp
import os
import queue
import random
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Cython auto-recompile: if any .pyx is newer than its compiled .pyd/.so,
# rebuild the Cython extensions before importing rl.* modules.
# ---------------------------------------------------------------------------
def _maybe_recompile_cython() -> None:
    """Rebuild Cython extensions if any .pyx source is newer than the binary."""
    import subprocess
    from pathlib import Path as _Path
    import sys as _sys

    project_root = _Path(__file__).resolve().parents[1]
    setup_script = project_root / "setup_cython.py"
    if not setup_script.exists():
        return

    pyx_dirs = [project_root / "rl", project_root / "engine"]
    pyx_files = []
    for d in pyx_dirs:
        if d.exists():
            pyx_files.extend(d.glob("*.pyx"))

    if not pyx_files:
        return

    if _sys.platform.startswith("win"):
        ext_suffix = ".pyd"
    else:
        ext_suffix = ".so"

    needs_rebuild = False
    for pyx in pyx_files:
        compiled = pyx.with_suffix(ext_suffix)
        if not compiled.exists():
            needs_rebuild = True
            break
        if pyx.stat().st_mtime > compiled.stat().st_mtime:
            needs_rebuild = True
            break

    if needs_rebuild:
        print("Cython sources changed; rebuilding extensions...", flush=True)
        try:
            result = subprocess.run(
                [_sys.executable, str(setup_script), "build_ext", "--inplace"],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                print(
                    f"Cython rebuild failed (rc={result.returncode}):\n"
                    f"{result.stdout}\n{result.stderr}",
                    file=_sys.stderr,
                    flush=True,
                )
            else:
                print("Cython rebuild complete.", flush=True)
        except Exception as exc:
            print(f"Cython rebuild error: {exc}", file=_sys.stderr, flush=True)


# Run the check before importing rl.* (which may import the .pyd files)
# Commented out to avoid rebuild race conditions during multi-process training:
# _maybe_recompile_cython()

import numpy as np
import torch

from rl.encoder import encode_state, GRID_SIZE, N_SPATIAL_CHANNELS, N_SCALARS
from rl.env import AWBWEnv
from rl.rhea import RheaConfig, RheaPlanner
from rl.rhea_fitness import RheaFitness
from rl.rhea_replay import RheaReplayBuffer, RheaTransition
from rl.rhea_value_learner import RheaValueLearner, RheaValueLearnerConfig
from rl.value_net import AWBWValueNet, load_value_checkpoint


def _parse_co_list(s: str) -> list[int]:
    return [int(x) for x in str(s).split(",") if str(x).strip()]


def _encode(state, observer_seat: int) -> tuple[np.ndarray, np.ndarray]:
    spatial = np.zeros((GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS), dtype=np.float32)
    scalars = np.zeros((N_SCALARS,), dtype=np.float32)
    encode_state(
        state,
        observer=int(observer_seat),
        belief=None,
        out_spatial=spatial,
        out_scalars=scalars,
    )
    return spatial, scalars


def _encode_into(
    state,
    observer_seat: int,
    spatial_buf: np.ndarray,
    scalars_buf: np.ndarray,
) -> None:
    """Encode state into pre-allocated buffers (avoids per-call allocation)."""
    encode_state(
        state,
        observer=int(observer_seat),
        belief=None,
        out_spatial=spatial_buf,
        out_scalars=scalars_buf,
    )


def _make_env(args: argparse.Namespace) -> AWBWEnv:
    co_p0 = _parse_co_list(args.co_p0)
    co_p1 = _parse_co_list(args.co_p1)
    
    # Load map pool and filter to the specified map_id
    pool_path = Path(__file__).parent.parent / "data" / "gl_map_pool.json"
    with open(pool_path, encoding="utf-8") as f:
        full_pool = json.load(f)
    
    # Filter to the specific map_id
    map_pool = [m for m in full_pool if m.get("map_id") == args.map_id]
    if not map_pool:
        raise ValueError(f"Map ID {args.map_id} not found in pool")
    
    # max_days parameter is passed as max_turns to AWBWEnv
    # (they are aliases for the same calendar day cap)
    max_turns = args.max_days if args.max_days else None

    # Opening book integration
    opening_book_path = str(args.opening_book_path) if args.opening_book_path else None
    opening_book_prob = max(0.0, min(1.0, float(args.opening_book_prob)))
    opening_book_strike_release = bool(args.opening_book_strike_release)

    try:
        return AWBWEnv(
            map_pool=map_pool,
            co_p0=co_p0,
            co_p1=co_p1,
            max_turns=max_turns,
            opening_book_path=opening_book_path,
            opening_book_seats="both",
            opening_book_prob=opening_book_prob,
            opening_book_strict_co=False,
            opening_book_strike_release=opening_book_strike_release,
        )
    except TypeError:
        return AWBWEnv(
            map_pool=map_pool,
            co_p0=co_p0[0],
            co_p1=co_p1[0],
            max_turns=max_turns,
            opening_book_path=opening_book_path,
            opening_book_seats="both",
            opening_book_prob=opening_book_prob,
            opening_book_strict_co=False,
            opening_book_strike_release=opening_book_strike_release,
        )


def _save_checkpoint(path: Path, model: AWBWValueNet, learner_cfg: RheaValueLearnerConfig, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "learner_cfg": dataclasses.asdict(learner_cfg),
            "step": step,
        },
        tmp,
    )
    os.replace(tmp, path)

def _timestamp_str() -> str:
    """Return a compact timestamp string for checkpoint naming."""
    return time.strftime("%Y%m%d_%H%M%S")



def _save_weight_delta(path: Path, model: AWBWValueNet, base_state_dict: dict, step: int) -> None:
    """Save weight deltas (current - base) for federated averaging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    current = model.state_dict()
    delta = {k: v - base_state_dict[k] for k, v in current.items()}
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "delta": delta,
            "step": step,
            "timestamp": time.time(),
        },
        tmp,
    )
    os.replace(tmp, path)


def _load_weight_deltas(remote_dir: Path) -> list[tuple[int, dict[str, torch.Tensor], Path]]:
    """Load all pending weight deltas from remote_dir/fleet/*/weights/."""
    import glob

    pattern = str(remote_dir / "fleet" / "*" / "weights" / "*.pt")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)

    results = []
    for fpath in files:
        f = Path(fpath)
        try:
            ckpt = torch.load(f, map_location="cpu")
            step = int(ckpt["step"])
            delta = ckpt["delta"]
            results.append((step, delta, f))
        except Exception as e:
            print(json.dumps({
                "event": "weight_delta_read_error",
                "file": str(f),
                "error": str(e),
            }), flush=True)
    return results
    """Return a compact timestamp string for checkpoint naming."""
    import time
    return time.strftime("%Y%m%d_%H%M%S")


def _load_value_pt_into_model(path: Path, model: AWBWValueNet, device: str, verbose: bool = False) -> bool:
    if not path.exists():
        return False
    try:
        ckpt = torch.load(path, map_location=device)
        # Support both 'state_dict' and 'model_state_dict' keys for consistency
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        else:
            # If neither key exists, try using the entire checkpoint
            sd = ckpt
        model.load_state_dict(sd, strict=False)
        model.to(device)
        model.eval()
        return True
    except Exception as exc:
        if verbose:
            print(json.dumps({"event": "actor_refresh_failed", "path": str(path), "error": repr(exc)}), flush=True)
        return False


def _transition_to_payload(t: RheaTransition) -> dict[str, Any]:
    # Numpy arrays are picklable; using a dict avoids class-version issues across
    # long-running actor processes during rapid iteration.
    return {
        "spatial_before": t.spatial_before,
        "scalars_before": t.scalars_before,
        "reward_turn": float(t.reward_turn),
        "spatial_after": t.spatial_after,
        "scalars_after": t.scalars_after,
        "done": bool(t.done),
        "winner": t.winner,
        "acting_seat": int(t.acting_seat),
        "day": int(t.day),
        "phi_delta": float(t.phi_delta),
        "value_after_at_search_time": float(t.value_after_at_search_time),
        "search_score": float(t.search_score),
    }


def _compute_gradients_for_transitions(
    model: AWBWValueNet,
    transitions: list[RheaTransition],
    device: str,
    cfg: RheaValueLearnerConfig,
) -> dict[str, list[float]] | None:
    """Compute gradients for a batch of transitions using the current model.
    
    Returns a dict mapping parameter names to gradient tensors (as lists for JSON serialization).
    Returns None if no trainable parameters or transitions.
    """
    if not transitions:
        return None
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        return None
    
    # Convert transitions to batch tensors
    spatial_before = torch.as_tensor(
        np.stack([t.spatial_before for t in transitions]),
        dtype=torch.float32,
        device=device
    )
    scalars_before = torch.as_tensor(
        np.stack([t.scalars_before for t in transitions]),
        dtype=torch.float32,
        device=device
    )
    spatial_after = torch.as_tensor(
        np.stack([t.spatial_after for t in transitions]),
        dtype=torch.float32,
        device=device
    )
    scalars_after = torch.as_tensor(
        np.stack([t.scalars_after for t in transitions]),
        dtype=torch.float32,
        device=device
    )
    done = torch.as_tensor(
        [bool(t.done) for t in transitions],
        dtype=torch.float32,
        device=device
    )
    winner = torch.as_tensor(
        [int(t.winner) if t.winner is not None else -1 for t in transitions],
        dtype=torch.int64,
        device=device
    )
    acting_seat = torch.as_tensor(
        [int(t.acting_seat) for t in transitions],
        dtype=torch.int64,
        device=device
    )
    
    # Compute loss (same as RheaValueLearner.train_one_batch)
    from torch.nn import functional as F
    
    pred_logits = model(spatial_before, scalars_before)
    
    with torch.no_grad():
        next_logits = model(spatial_after, scalars_after)
        # Detach to avoid computing gradients through target
        next_logits = next_logits.detach()
        
        win_target = torch.where(
            winner == -1,
            torch.tensor(0.5, device=device).expand_as(winner),
            (winner == acting_seat).float(),
        )
        
        next_win_prob = torch.sigmoid(next_logits)
        immediate_win = win_target * done
        gamma = cfg.gamma_turn if cfg.gamma_turn is not None else 0.99
        td_target_win = immediate_win + gamma * next_win_prob * (1.0 - done)
        
        if cfg.target_clip is not None:
            c = float(cfg.target_clip)
            td_target_win = torch.clamp(td_target_win, 0.0, 1.0)
    
    loss = F.binary_cross_entropy_with_logits(pred_logits, td_target_win)
    
    # Compute gradients
    model.zero_grad(set_to_none=True)
    loss.backward()
    
    # Collect gradients (only trainable parameters)
    # Preserve shape — flattening breaks assignment to conv weights later
    grads = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[name] = p.grad.detach().cpu().tolist()
    
    return grads


def _write_gradients_to_shared(
    actor_id: int,
    grads: dict[str, list[float]],
    step_num: int,
    shared_root: str = "Z:",
) -> str | None:
    """Write gradient deltas to shared filesystem for main to aggregate.
    
    Returns the path where gradients were written, or None on failure.
    """
    import json
    import tempfile
    
    try:
        grad_dir = Path(shared_root) / "fleet" / f"actor-{actor_id}" / "gradients"
        grad_dir.mkdir(parents=True, exist_ok=True)
        
        # Use atomic write: write to temp, then rename
        grad_data = {
            "actor_id": actor_id,
            "step": step_num,
            "timestamp": time.time(),
            "gradients": grads,
        }
        
        tmp_path = grad_dir / f"grad_{step_num}_{int(time.time())}.tmp"
        final_path = grad_dir / f"grad_{step_num}_{int(time.time())}.json"
        
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(grad_data, f)
        
        os.replace(str(tmp_path), str(final_path))
        return str(final_path)
    
    except Exception as e:
        print(json.dumps({
            "event": "gradient_write_error",
            "actor_id": actor_id,
            "error": str(e),
        }), flush=True)
        return None


def _apply_gradients_to_model(
    model: AWBWValueNet,
    grads_dict: dict[str, torch.Tensor],
    opt: torch.optim.Optimizer,
    clip_norm: float = 1.0,
) -> float:
    """Apply aggregated gradients to model.
    
    Returns gradient norm.
    """
    # First, set gradients on model
    for name, p in model.named_parameters():
        if p.requires_grad and name in grads_dict:
            p.grad = grads_dict[name].to(p.device)
    
    # Clip and step
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        clip_norm,
    )
    opt.step()
    opt.zero_grad(set_to_none=True)
    
    return float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm)


def _poll_gradients_from_shared(
    shared_root: str = "Z:",
    last_poll_time: dict[str, float] | None = None,
) -> tuple[list[tuple[int, dict[str, torch.Tensor], float]], dict[str, float]]:
    """Poll shared filesystem for gradient files from actors.
    
    Returns:
        List of tuples: (actor_id, gradients_dict, timestamp)
        Updated last_poll_time dict
    """
    import json
    import glob
    
    if last_poll_time is None:
        last_poll_time = {}
    
    shared_path = Path(shared_root)
    if not shared_path.exists():
        return [], last_poll_time
    
    # Pattern: Z:/fleet/*/gradients/*.json
    pattern = str(shared_path / "fleet" / "*" / "gradients" / "*.json")
    grad_files = glob.glob(pattern)
    
    results = []
    for fpath in grad_files:
        f = Path(fpath)
        try:
            mtime = f.stat().st_mtime
            if fpath in last_poll_time and last_poll_time[fpath] >= mtime:
                continue

            with open(f, "r", encoding="utf-8") as fh:
                grad_data = json.load(fh)
            
            actor_id = grad_data["actor_id"]
            timestamp = grad_data.get("timestamp", mtime)
            
            # Convert lists back to tensors
            grads_dict = {}
            for name, grad_list in grad_data["gradients"].items():
                grads_dict[name] = torch.tensor(grad_list)
            
            results.append((actor_id, grads_dict, timestamp))
            last_poll_time[fpath] = mtime
            
            # Mark as consumed by renaming
            done_path = f.with_suffix(".json.done")
            os.rename(str(f), str(done_path))
            
        except Exception as e:
            print(json.dumps({
                "event": "gradient_read_error",
                "file": str(f),
                "error": str(e),
            }), flush=True)
    
    return results, last_poll_time


def _maybe_disable_cop_for_seat(co_state, disable_prob: float = 0.10) -> bool:
    """Randomly disable COP for a seat at game start (10% default).
    
    Returns True if COP was disabled for this seat.
    Only applies if the CO has a COP (cop_stars is not None and has cop data).
    """
    if disable_prob <= 0.0:
        return False
    if co_state.cop_stars is None or co_state._data.get("cop") is None:
        return False
    if random.random() < disable_prob:
        co_state.cop_activation_disabled = True
        return True
    return False


def _payload_to_transition(p: dict[str, Any]) -> RheaTransition:
    return RheaTransition(
        spatial_before=p["spatial_before"],
        scalars_before=p["scalars_before"],
        reward_turn=float(p["reward_turn"]),
        spatial_after=p["spatial_after"],
        scalars_after=p["scalars_after"],
        done=bool(p["done"]),
        winner=p.get("winner"),
        acting_seat=int(p["acting_seat"]),
        day=int(p["day"]),
        phi_delta=float(p["phi_delta"]),
        value_after_at_search_time=float(p["value_after_at_search_time"]),
        search_score=float(p["search_score"]),
    )


def _poll_remote_transitions(
    remote_dir: str | Path,
    replay: RheaReplayBuffer,
    last_poll_mtime: dict[str, float] | None = None,
) -> tuple[int, dict[str, float]]:
    """Poll remote transition files and ingest them into the replay buffer.

    Reads both plain .jsonl and compressed .jsonl.gz files from
    fleet/*/transitions/ and flat transitions/ directories.

    Args:
        remote_dir: Root directory containing fleet/*/transitions/ subdirectories.
        replay: The replay buffer to add transitions to.
        last_poll_mtime: Optional dict mapping file paths to last modification time
                            to avoid re-processing the same files.

    Returns:
        (num_ingested, updated_last_poll_mtime)
    """
    import gzip

    if last_poll_mtime is None:
        last_poll_mtime = {}

    remote_path = Path(remote_dir)
    if not remote_path.exists():
        return 0, last_poll_mtime

    # Glob all transition files (both plain and compressed)
    import glob

    # Pattern: fleet/*/transitions/*.jsonl and *.jsonl.gz
    pattern1 = str(remote_path / "fleet" / "*" / "transitions" / "*.jsonl")
    pattern2 = str(remote_path / "fleet" / "*" / "transitions" / "*.jsonl.gz")
    transition_files = glob.glob(pattern1) + glob.glob(pattern2)

    # Also check flat transitions dir if it exists
    flat_pattern1 = str(remote_path / "transitions" / "*.jsonl")
    flat_pattern2 = str(remote_path / "transitions" / "*.jsonl.gz")
    transition_files.extend(glob.glob(flat_pattern1))
    transition_files.extend(glob.glob(flat_pattern2))

    total_ingested = 0
    files_to_process = []

    for fpath in transition_files:
        f = Path(fpath)
        try:
            mtime = f.stat().st_mtime
        except OSError:
            continue
        # Skip files we've already processed (by mtime)
        if fpath in last_poll_mtime and last_poll_mtime[fpath] >= mtime:
            continue
        files_to_process.append((f, mtime))

    for f, mtime in files_to_process:
        try:
            # Handle both plain and compressed files
            if f.suffix == ".gz":
                with gzip.open(f, "rt", encoding="utf-8") as fh:
                    lines = [line.strip() for line in fh.readlines() if line.strip()]
            else:
                with open(f, "r", encoding="utf-8") as fh:
                    lines = [line.strip() for line in fh.readlines() if line.strip()]

            transitions = []
            for line in lines:
                try:
                    payload = json.loads(line)
                    t = _payload_to_transition(payload)
                    transitions.append(t)
                except (json.JSONDecodeError, KeyError) as e:
                    print(json.dumps({
                        "event": "remote_transition_parse_error",
                        "file": str(f),
                        "error": str(e),
                    }), flush=True)

            if transitions:
                added = replay.add_batch(transitions)
                total_ingested += added
                last_poll_mtime[str(f)] = mtime

                print(json.dumps({
                    "event": "remote_transitions_ingested",
                    "file": str(f),
                    "count": len(transitions),
                    "added": added,
                    "replay_size": len(replay),
                }), flush=True)

            # Mark file as consumed by renaming to .done
            try:
                if f.suffix == ".gz":
                    done_path = f.with_suffix(".jsonl.gz.done")
                else:
                    done_path = f.with_suffix(".jsonl.done")
                os.rename(str(f), str(done_path))
            except OSError:
                # If rename fails, just continue
                pass

        except Exception as e:
            print(json.dumps({
                "event": "remote_transition_file_error",
                "file": str(f),
                "error": str(e),
            }), flush=True)

    return total_ingested, last_poll_mtime


def _actor_loop(
    actor_id: int,
    args: argparse.Namespace,
    out_q: mp.Queue,
    stop_event: mp.Event,
) -> None:
    try:
        seed = int(args.seed) + 1009 * int(actor_id)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Avoid each actor grabbing a full CPU threadpool.
        if args.actor_torch_threads > 0:
            torch.set_num_threads(int(args.actor_torch_threads))

        env = _make_env(args)

        # Device assignment:
        #   --n-envs N          total actor processes
        #   --gpu-actors K     first K actor ranks evaluate their value net on CUDA
        #   remaining actors use --actor-device, which defaults to CPU
        #
        # RHEA itself is still CPU-heavy. GPU actors only accelerate value-head calls
        # during fitness evaluation. Start with K=1 or K=2 and watch VRAM.
        if int(actor_id) < int(args.gpu_actors):
            actor_device = str(args.actor_gpu_device)
        else:
            actor_device = str(args.actor_device)

        print(
            json.dumps(
                {
                    "event": "actor_start",
                    "actor_id": int(actor_id),
                    "actor_device": actor_device,
                    "gpu_actors": int(args.gpu_actors),
                }
            ),
            flush=True,
        )

        value_model = None
        hist_value_model = None
        latest_path = Path("checkpoints") / "value_rhea_latest.pt"
        
        # Try to load checkpoint, but don't fail if it doesn't exist yet
        # The actor will refresh from latest.pt once the learner creates it
        try:
            value_model = load_value_checkpoint(args.checkpoint, device=actor_device)
            print(json.dumps({
                "event": "actor_checkpoint_loaded",
                "actor_id": actor_id,
                "checkpoint": str(args.checkpoint),
            }), flush=True)
        except Exception as e:
            print(json.dumps({
                "event": "actor_checkpoint_missing",
                "actor_id": actor_id,
                "checkpoint": str(args.checkpoint),
                "error": str(e),
                "message": "Will retry on first refresh"
            }), flush=True)
            # Create a fresh model as fallback
            from rl.value_net import AWBWValueNet
            value_model = AWBWValueNet().to(actor_device)
        
        last_refresh = 0.0

        # Load historical checkpoint for hist-prob games
        hist_checkpoint_path = args.hist_checkpoint_path
        if args.dual_gradient_self_play and args.dual_gradient_hist_prob > 0:
            if not hist_checkpoint_path:
                # Auto-discover: use the oldest saved transition checkpoint as historical
                try:
                    ckpt_dir = Path("checkpoints")
                    hist_candidates = sorted(
                        ckpt_dir.glob("value_rhea_transition_*.pt"),
                        key=lambda p: p.stat().st_mtime
                    )
                    if len(hist_candidates) >= 2:
                        # Use a checkpoint from at least 2 saves ago (not the most recent)
                        hist_checkpoint_path = str(hist_candidates[0])  # Oldest
                        print(json.dumps({
                            "event": "hist_checkpoint_auto_discovered",
                            "actor_id": actor_id,
                            "hist_checkpoint_path": hist_checkpoint_path,
                        }), flush=True)
                except Exception:
                    pass
            
            if hist_checkpoint_path and Path(hist_checkpoint_path).exists():
                try:
                    hist_value_model = load_value_checkpoint(hist_checkpoint_path, device=actor_device)
                    print(json.dumps({
                        "event": "hist_checkpoint_loaded",
                        "actor_id": actor_id,
                        "hist_checkpoint_path": hist_checkpoint_path,
                    }), flush=True)
                except Exception as e:
                    print(json.dumps({
                        "event": "hist_checkpoint_load_failed",
                        "actor_id": actor_id,
                        "hist_checkpoint_path": hist_checkpoint_path,
                        "error": str(e),
                    }), flush=True)

        fitness = RheaFitness(
            env_template=env,
            value_model=value_model,
            device=actor_device,
            reward_weight=args.reward_weight,
            value_weight=args.value_weight,
        )

        # Pre-allocate encode buffers (reused across turns to avoid per-call allocation)
        spatial_buf = np.empty((GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS), dtype=np.float32)
        scalars_buf = np.empty((N_SCALARS,), dtype=np.float32)

        # Create RheaPlanner once (reused across turns and games)
        try:
            planner = RheaPlanner(
                fitness,
                RheaConfig(
                    population=args.rhea_population,
                    generations=args.rhea_generations,
                    elite=args.rhea_elite,
                    mutation_rate=args.rhea_mutation_rate,
                    top_k_per_state=args.rhea_top_k_per_state,
                    max_actions_per_turn=args.rhea_max_actions_per_turn,
                    reward_weight=args.reward_weight,
                    value_weight=args.value_weight,
                    seed=seed,
                    use_tactical_beam=args.rhea_use_tactical_beam,
                    tactial_beam_max_width=args.rhea_tactical_beam_max_width,
                    tactial_beam_max_depth=args.rhea_tactical_beam_max_depth,
                    tactial_beam_max_expand=args.rhea_tactical_beam_max_expand,
                ),
                dynamic_budget=args.rhea_autotune,
                complexity_metrics=None,
            )
        except Exception as e:
            print(json.dumps({
                "event": "planner_creation_failed",
                "actor_id": actor_id,
                "error": repr(e),
            }), flush=True)
            raise

        games_done = 0
        transitions_sent = 0

        # For historical checkpoint games, we need to track which model to use
        # hist_value_model may be None if loading failed; in that case, force hist mode off
        local_hist_prob = 0.0
        if args.dual_gradient_self_play and args.dual_gradient_hist_prob > 0:
            if hist_value_model is not None:
                local_hist_prob = args.dual_gradient_hist_prob
            else:
                print(json.dumps({
                    "event": "warning",
                    "actor_id": actor_id,
                    "message": "dual_gradient_hist_prob > 0 but hist checkpoint not loaded; disabling hist mode",
                }), flush=True)

        # Gradient pushing state (A3C-style)
        push_gradients = bool(getattr(args, "push_gradients", False))
        local_transitions: list[RheaTransition] = []
        gradient_step = 0
        learner_cfg_for_grads = RheaValueLearnerConfig(
            gamma_turn=args.gamma_turn,
            target_clip=args.target_clip,
        )

        while not stop_event.is_set():
            try:
                # Decide if this game uses historical checkpoint opponent
                use_hist_checkpoint = random.random() < local_hist_prob
                
                # Periodically refresh the actor's value model from the learner
                now = time.time()
                if args.actor_refresh_seconds > 0 and now - last_refresh >= args.actor_refresh_seconds:
                    if _load_value_pt_into_model(latest_path, value_model, actor_device, verbose=True):
                        last_refresh = now

                env.reset()
                game_turns = 0

                # 10% chance to disable COP for each seat at game start (forces SCOP learning)
                cop_disable_p = getattr(args, "cop_disable_per_seat_p", 0.10)
                for seat in (0, 1):
                    _maybe_disable_cop_for_seat(env.state.co_states[seat], cop_disable_p)

                # Set async rollout mode for logging if dual-gradient is enabled
                if args.dual_gradient_self_play:
                    try:
                        env.set_async_rollout_mode("hist" if use_hist_checkpoint else "mirror")
                    except AttributeError:
                        pass

                while env.state is not None and env.state.winner is None and not stop_event.is_set():
                    try:
                        state = env.state
                        acting = int(state.active_player)
                        day = int(getattr(state, "turn", getattr(state, "day", 0)))

                        # Encode before state using pre-allocated buffers
                        _encode_into(state, acting, spatial_buf, scalars_buf)
                        before_spatial = spatial_buf.copy()
                        before_scalars = scalars_buf.copy()
                        phi_before = fitness.phi(state, acting)

                        # Compute complexity metrics for dynamic budgeting if enabled
                        complexity_metrics = None
                        if args.rhea_autotune:
                            try:
                                complexity_metrics = RheaPlanner.compute_complexity_metrics(state, acting)
                            except Exception as e:
                                print(json.dumps({
                                    "event": "complexity_metrics_error",
                                    "actor_id": actor_id,
                                    "error": str(e)
                                }), flush=True)
                                complexity_metrics = None

                        # Update planner's dynamic budget (planner was created once before the loop)
                        planner.dynamic_budget = args.rhea_autotune
                        planner.complexity_metrics = complexity_metrics

                        result = planner.choose_full_turn(state)

                        for action in result.actions:
                            if env.state is None or env.state.winner is not None:
                                break
                            if int(env.state.active_player) != acting:
                                break
                            env.state.step(action)

                        after = env.state
                        if after is None:
                            break

                        _encode_into(after, acting, spatial_buf, scalars_buf)
                        after_spatial = spatial_buf.copy()
                        after_scalars = scalars_buf.copy()
                        phi_after = fitness.phi(after, acting)
                        reward_turn = float(phi_after - phi_before)
                        done = bool(after.winner is not None)

                        t = RheaTransition(
                            spatial_before=before_spatial,
                            scalars_before=before_scalars,
                            reward_turn=reward_turn,
                            spatial_after=after_spatial,
                            scalars_after=after_scalars,
                            done=done,
                            winner=after.winner,
                            acting_seat=acting,
                            day=day,
                            phi_delta=float(result.breakdown.phi_delta),
                            value_after_at_search_time=float(result.breakdown.value),
                            search_score=float(result.score),
                        )

                        # If pushing gradients, accumulate transitions locally
                        if push_gradients:
                            local_transitions.append(t)
                            # Compute and push gradients when batch is ready
                            if len(local_transitions) >= args.gradient_batch_size:
                                grads = _compute_gradients_for_transitions(
                                    value_model,
                                    local_transitions,
                                    actor_device,
                                    learner_cfg_for_grads,
                                )
                                if grads:
                                    gradient_step += 1
                                    grad_path = _write_gradients_to_shared(
                                        actor_id,
                                        grads,
                                        gradient_step,
                                        shared_root=args.gradient_shared_root,
                                    )
                                    if grad_path:
                                        print(json.dumps({
                                            "event": "gradients_pushed",
                                            "actor_id": actor_id,
                                            "step": gradient_step,
                                            "path": grad_path,
                                            "num_transitions": len(local_transitions),
                                        }), flush=True)
                                local_transitions = []
                        else:
                            # Original behavior: send transition to main process
                            out_q.put(
                                {
                                    "type": "transition",
                                    "actor_id": actor_id,
                                    "transition": _transition_to_payload(t),
                                    "log": {
                                        "day": day,
                                        "acting": acting,
                                        "reward_turn": reward_turn,
                                        "search_score": float(result.score),
                                        "phi_delta_search": float(result.breakdown.phi_delta),
                                        "value_after": float(result.breakdown.value),
                                        "initial_best_score": result.initial_best_score,
                                        "evolved_gain": result.evolved_gain,
                                        "illegal_genes": int(result.illegal_genes),
                                        "actions": len(result.actions),
                                        "use_hist_checkpoint": use_hist_checkpoint,
                                    },
                                }
                            )

                        transitions_sent += 1
                        game_turns += 1

                        if day > args.max_days + 1:
                            break

                    except Exception as e:
                        import traceback
                        print(json.dumps({
                            "event": "turn_error",
                            "actor_id": actor_id,
                            "error": repr(e),
                            "game_turns": game_turns,
                            "traceback": traceback.format_exc(),
                        }), flush=True)
                        break  # Exit the inner while loop for this game

                games_done += 1
                out_q.put(
                    {
                        "type": "game_done",
                        "actor_id": actor_id,
                        "winner": None if env.state is None else env.state.winner,
                        "turns": game_turns,
                        "games_done": games_done,
                        "transitions_sent": transitions_sent,
                        "use_hist_checkpoint": use_hist_checkpoint,
                        "dual_gradient_self_play": args.dual_gradient_self_play,
                    }
                )
            except Exception as exc:
                print(json.dumps({
                    "event": "actor_exception",
                    "actor_id": actor_id,
                    "error": repr(exc),
                    "games_done": games_done,
                    "transitions_sent": transitions_sent,
                }), flush=True)
                out_q.put({
                    "type": "actor_dead",
                    "actor_id": actor_id,
                    "error": repr(exc),
                })
                raise  # Re-raise to stop this actor

    except Exception as exc:
        print(json.dumps({
            "event": "actor_fatal",
            "actor_id": actor_id,
            "error": repr(exc),
        }), flush=True)
        out_q.put({
            "type": "actor_dead",
            "actor_id": actor_id,
            "error": repr(exc),
        })
        return


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    # Env / checkpoint.
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--map-id", type=int, default=171596)
    ap.add_argument("--co-p0", type=str, default="14,8,28,7")
    ap.add_argument("--co-p1", type=str, default="14,8,28,7")
    ap.add_argument("--max-days", type=int, default=30)
    ap.add_argument("--device", type=str, default="cuda", help="learner device")
    ap.add_argument("--actor-device", type=str, default="cpu", help="actor value-net device; cpu is safest for n-envs")
    ap.add_argument("--actor-gpu-device", type=str, default="cuda", help="device used by the first --gpu-actors actor processes")
    ap.add_argument("--gpu-actors", type=int, default=0, help="number of actor processes that should run value evaluation on --actor-gpu-device")
    ap.add_argument("--actor-torch-threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--queue-size", type=int, default=2048)
    ap.add_argument("--actor-refresh-seconds", type=float, default=120.0)
    ap.add_argument("--verbose", action="store_true", help="Print diagnostic logs to stdout")

    # COP disable (forces SCOP learning)
    ap.add_argument("--cop-disable-per-seat-p", type=float, default=0.10,
                      help="Probability (0-1) to disable COP for each seat at game start (default 0.10 = 10%%)")

    # Opening book (Designed Desires)
    ap.add_argument("--opening-book-path", type=Path, default=None,
                      help="Path to opening book JSONL (e.g., data/designed_desires_opening_book.jsonl)")
    ap.add_argument("--opening-book-prob", type=float, default=1.0,
                      help="Probability (0-1) to use opening book vs RHEA from day 1 (default 1.0 = always)")
    ap.add_argument("--opening-book-strike-release", action="store_true",
                      help="Release opening book if a unit moves into enemy strike range")

    # RHEA search.
    ap.add_argument("--rhea-autotune", action="store_true",
        help="Enable dynamic RHEA budget auto-tuning based on game state complexity")
    ap.add_argument("--rhea-population", type=int, default=32)
    ap.add_argument("--rhea-generations", type=int, default=5)
    ap.add_argument("--rhea-elite", type=int, default=4)
    ap.add_argument("--rhea-mutation-rate", type=float, default=0.20)
    ap.add_argument("--rhea-top-k-per-state", type=int, default=24)
    ap.add_argument("--rhea-max-actions-per-turn", type=int, default=128)
    ap.add_argument("--reward-weight", type=float, default=0.90)
    ap.add_argument("--value-weight", type=float, default=0.10)
    # Tactical beam.
    ap.add_argument("--rhea-use-tactical-beam", action="store_true")
    ap.add_argument("--rhea-tactical-beam-max-width", type=int, default=48)
    ap.add_argument("--rhea-tactical-beam-max-depth", type=int, default=14)
    ap.add_argument("--rhea-tactical-beam-max-expand", type=int, default=24)

    # Value learner.
    ap.add_argument("--value-lr", type=float, default=1.0e-4)
    ap.add_argument("--value-batch-size", type=int, default=128)
    ap.add_argument("--replay-size", type=int, default=50_000)
    ap.add_argument("--min-replay-before-train", type=int, default=1_000)
    ap.add_argument("--updates-per-turn", type=int, default=1)
    ap.add_argument("--gamma-turn", type=float, default=0.99)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--target-update-interval", type=int, default=1_000)
    ap.add_argument("--target-tau", type=float, default=None)
    ap.add_argument("--target-clip", type=float, default=5.0)

    ap.add_argument(
        "--no-learner",
        action="store_true",
        help="Actors only — do not train a value net. Transitions are still produced and (on the orchestrator) polled from remote dirs.",
    )

    # Freeze schedule.
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--unfreeze-last-resblocks", type=int, default=0)

    # Phi capture phase weighting.
    ap.add_argument(
        "--phi-capture-phase-weighting",
        action="store_true",
        help=(
            "Enable component-specific day/turn phase weighting inside capture Φ. "
            "Safe neutral expansion gets early urgency and late falloff; contested "
            "neutrals get mild falloff; enemy/production/HQ capture progress does not fall off."
        ),
    )
    ap.add_argument(
        "--phi-safe-neutral-opening-mult",
        type=float,
        default=None,
        help="Safe neutral capture-progress phase multiplier through opening end day (default 1.30).",
    )
    ap.add_argument(
        "--phi-safe-neutral-early-mid-mult",
        type=float,
        default=None,
        help="Safe neutral capture-progress phase multiplier through early-mid end day (default 1.15).",
    )
    ap.add_argument(
        "--phi-safe-neutral-mid-mult",
        type=float,
        default=None,
        help="Safe neutral capture-progress phase multiplier through mid end day (default 1.00).",
    )
    ap.add_argument(
        "--phi-safe-neutral-late-mult",
        type=float,
        default=None,
        help="Safe neutral capture-progress phase multiplier through late end day (default 0.75).",
    )
    ap.add_argument(
        "--phi-safe-neutral-endgame-mult",
        type=float,
        default=None,
        help="Safe neutral capture-progress phase multiplier after late end day (default 0.50).",
    )
    ap.add_argument(
        "--phi-contested-neutral-opening-mult",
        type=float,
        default=None,
        help="Contested neutral capture-progress phase multiplier through early-mid end day (default 1.25).",
    )
    ap.add_argument(
        "--phi-contested-neutral-mid-mult",
        type=float,
        default=None,
        help="Contested neutral capture-progress phase multiplier through late end day (default 1.00).",
    )
    ap.add_argument(
        "--phi-contested-neutral-late-mult",
        type=float,
        default=None,
        help="Contested neutral capture-progress phase multiplier after late end day (default 0.90).",
    )
    ap.add_argument(
        "--phi-capture-opening-end-day",
        type=int,
        default=None,
        help="Day/turn boundary for safe-neutral opening phase weighting (default 5).",
    )
    ap.add_argument(
        "--phi-capture-early-mid-end-day",
        type=int,
        default=None,
        help="Day/turn boundary for safe-neutral early-mid and contested opening phase weighting (default 8).",
    )
    ap.add_argument(
        "--phi-capture-mid-end-day",
        type=int,
        default=None,
        help="Day/turn boundary for safe-neutral mid phase weighting (default 12).",
    )
    ap.add_argument(
        "--phi-capture-late-end-day",
        type=int,
        default=None,
        help="Day/turn boundary for late capture phase weighting (default 18).",
    )

    # Dual-gradient self-play and zero-sum.
    ap.add_argument(
        "--dual-gradient-self-play",
        action="store_true",
        help=(
            "Both engine seats sample from the shared policy and each "
            "active-seat decision is recorded as a policy-gradient row with "
            "seat-relative zero-sum Phi/reward signals."
        ),
    )
    ap.add_argument(
        "--dual-gradient-hist-prob",
        type=float,
        default=0.0,
        help=(
            "Only with --dual-gradient-self-play: probability each episode uses a "
            "historical checkpoint as the opponent instead of symmetric mirror self-play "
            "from synced weights. Set to 0.2 for '~80%% mirror / 20%% vs archive'."
        ),
    )
    ap.add_argument(
        "--pairwise-zero-sum-reward",
        action="store_true",
        help=(
            "Opt in to the learner-frame pairwise reward contract for AWBWEnv.step(): "
            "competitive reward is exposed as a zero-sum seat pair."
        ),
    )
    ap.add_argument(
        "--hist-checkpoint-path",
        type=str,
        default=None,
        help="Path to historical checkpoint for --dual-gradient-hist-prob games",
    )

    # Run control.
    ap.add_argument("--total-transitions", type=int, default=100_000)
    ap.add_argument("--save-every-transitions", type=int, default=500)  # Save more frequently
    ap.add_argument("--log-every-transitions", type=int, default=100)

    # Distributed gradient pushing (A3C-style)
    ap.add_argument("--push-gradients", action="store_true",
                      help="Enable actors to compute gradients locally and push to shared filesystem for main to aggregate")
    ap.add_argument("--gradient-batch-size", type=int, default=32,
                      help="Number of transitions to accumulate before computing and pushing gradients (default: 32)")
    ap.add_argument("--gradient-shared-root", type=str, default="Z:",
                      help="Shared filesystem root for gradient exchange (default: Z:)")
    ap.add_argument("--gradient-poll-interval", type=float, default=30.0,
                      help="Seconds between main polling for gradient files (default: 30)")

    # Remote transition polling (multi-machine)
    ap.add_argument("--remote-transition-dir", type=str, default=None,
                      help="Directory to poll for remote transition files (default: <shared-root>/fleet/*/transitions/)")
    ap.add_argument("--poll-remote-transitions-interval", type=float, default=60.0,
                      help="Seconds between polling remote transition directories (default: 60)")
    # Output directory is hardcoded to checkpoints/
    # Game logs will be written to logs/games_log.jsonl

    return ap


def _setup_env_vars(args: argparse.Namespace) -> None:
    """Set environment variables for phi capture phase weighting and other features."""
    if bool(getattr(args, "phi_capture_phase_weighting", False)):
        os.environ["AWBW_PHI_CAPTURE_PHASE_WEIGHTING"] = "1"
    else:
        os.environ.pop("AWBW_PHI_CAPTURE_PHASE_WEIGHTING", None)

    for attr, env_name in (
        ("phi_safe_neutral_opening_mult", "AWBW_PHI_SAFE_NEUTRAL_OPENING_MULT"),
        ("phi_safe_neutral_early_mid_mult", "AWBW_PHI_SAFE_NEUTRAL_EARLY_MID_MULT"),
        ("phi_safe_neutral_mid_mult", "AWBW_PHI_SAFE_NEUTRAL_MID_MULT"),
        ("phi_safe_neutral_late_mult", "AWBW_PHI_SAFE_NEUTRAL_LATE_MULT"),
        ("phi_safe_neutral_endgame_mult", "AWBW_PHI_SAFE_NEUTRAL_ENDGAME_MULT"),
        ("phi_contested_neutral_opening_mult", "AWBW_PHI_CONTESTED_NEUTRAL_OPENING_MULT"),
        ("phi_contested_neutral_mid_mult", "AWBW_PHI_CONTESTED_NEUTRAL_MID_MULT"),
        ("phi_contested_neutral_late_mult", "AWBW_PHI_CONTESTED_NEUTRAL_LATE_MULT"),
        ("phi_capture_opening_end_day", "AWBW_PHI_CAPTURE_OPENING_END_DAY"),
        ("phi_capture_early_mid_end_day", "AWBW_PHI_CAPTURE_EARLY_MID_END_DAY"),
        ("phi_capture_mid_end_day", "AWBW_PHI_CAPTURE_MID_END_DAY"),
        ("phi_capture_late_end_day", "AWBW_PHI_CAPTURE_LATE_END_DAY"),
    ):
        value = getattr(args, attr, None)
        if value is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = str(value)

    if bool(getattr(args, "dual_gradient_self_play", False)):
        os.environ["AWBW_DUAL_GRADIENT_SELF_PLAY"] = "1"
    else:
        os.environ.pop("AWBW_DUAL_GRADIENT_SELF_PLAY", None)

    if bool(getattr(args, "pairwise_zero_sum_reward", False)):
        os.environ["AWBW_PAIRWISE_ZERO_SUM_REWARD"] = "1"
    else:
        os.environ.pop("AWBW_PAIRWISE_ZERO_SUM_REWARD", None)


def main() -> None:
    args = build_arg_parser().parse_args()
    _setup_env_vars(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if int(args.gpu_actors) < 0:
        raise ValueError("--gpu-actors must be >= 0")
    if int(args.gpu_actors) > int(args.n_envs):
        raise ValueError("--gpu-actors cannot exceed --n-envs")
    if int(args.gpu_actors) > 0 and not str(args.actor_gpu_device).startswith("cuda"):
        if args.verbose:
            print(json.dumps({"event": "warning", "message": "--gpu-actors > 0 but --actor-gpu-device is not cuda*"}), flush=True)

    # Hardcoded output directory
    output_dir = Path("checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hparams_parallel.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    
    # Game log file
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    game_log_path = logs_dir / "games_log.jsonl"

    online = load_value_checkpoint(args.checkpoint, device=args.device)
    target = copy.deepcopy(online)
    replay = RheaReplayBuffer(args.replay_size, seed=args.seed)

    learner = None
    learner_cfg = None
    if not args.no_learner:
        learner_cfg = RheaValueLearnerConfig(
            value_lr=args.value_lr,
            value_batch_size=args.value_batch_size,
            replay_buffer_size=args.replay_size,
            min_replay_before_train=args.min_replay_before_train,
            updates_per_real_turn=args.updates_per_turn,
            gamma_turn=args.gamma_turn,
            gradient_clip_norm=args.grad_clip,
            weight_decay=args.weight_decay,
            target_update_interval=args.target_update_interval,
            target_tau=args.target_tau,
            target_clip=args.target_clip,
            freeze_encoder=args.freeze_encoder,
            unfreeze_last_resblocks=args.unfreeze_last_resblocks,
        )
        learner = RheaValueLearner(online, target, replay, learner_cfg, device=args.device)

    # Save an initial refresh checkpoint so actors can load learner-format .pt.
    # This must be done BEFORE starting actors so they have something to load.
    latest_path = output_dir / "value_rhea_latest.pt"
    _save_checkpoint(latest_path, online, learner_cfg if learner else None, 0)
    print(json.dumps({"event": "initial_checkpoint_saved", "path": str(latest_path)}), flush=True)

    # Gradient aggregation state (for push-gradients mode)
    gradient_poll_interval = float(getattr(args, "gradient_poll_interval", 30.0))
    last_gradient_poll_time = 0.0
    gradient_poll_mtime: dict[str, float] = {}
    gradient_step = 0
    optimizer_for_gradients = None
    
    if args.push_gradients and learner is not None:
        # Create optimizer for applying remote gradients
        params = [p for p in online.parameters() if p.requires_grad]
        optimizer_for_gradients = torch.optim.AdamW(
            params,
            lr=learner_cfg.value_lr,
            weight_decay=learner_cfg.weight_decay,
        )
        print(json.dumps({
            "event": "gradient_aggregation_ready",
            "gradient_poll_interval": gradient_poll_interval,
            "gradient_shared_root": args.gradient_shared_root,
        }), flush=True)

    ctx = mp.get_context("spawn")
    out_q: mp.Queue = ctx.Queue(maxsize=int(args.queue_size))
    stop_event: mp.Event = ctx.Event()
    procs: list[mp.Process] = []

    for actor_id in range(int(args.n_envs)):
        p = ctx.Process(target=_actor_loop, args=(actor_id, args, out_q, stop_event), daemon=True)
        p.start()
        procs.append(p)

    transitions = 0
    games_done = 0
    last_log: dict[str, Any] = {}
    actor_alive = [True] * int(args.n_envs)
    last_transition_time = time.time()
    last_heartbeat = time.time()
    heartbeat_interval = 60.0  # seconds between heartbeat logs
    start_time = time.time()

    # Remote transition polling
    last_poll_time = 0.0
    poll_interval = float(args.poll_remote_transitions_interval)
    last_poll_mtime: dict[str, float] = {}
    remote_transitions_ingested = 0

    try:
        while transitions < int(args.total_transitions):
            try:
                msg = out_q.get(timeout=30.0)
            except queue.Empty:
                # Check actor health
                alive_count = sum(actor_alive)
                elapsed = time.time() - last_transition_time
                timeout_log_entry = {
                    "event": "queue_timeout",
                    "transitions": transitions,
                    "replay": len(replay),
                    "actors_alive": alive_count,
                    "seconds_since_last_transition": round(elapsed, 1),
                }
                print(json.dumps(timeout_log_entry), flush=True)
                # Write to game log file
                with open(game_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(timeout_log_entry) + "\n")

                # If all actors are dead, abort
                if alive_count == 0:
                    print(json.dumps({"event": "all_actors_dead", "transitions": transitions}), flush=True)
                    break

                # Poll remote transition files (traditional mode)
                now = time.time()
                if poll_interval > 0 and now - last_poll_time >= poll_interval:
                    try:
                        remote_dir = args.remote_transition_dir
                        if not remote_dir:
                            # Default: use the same directory as checkpoints (shared root)
                            remote_dir = str(latest_path.parent.parent)
                        ingested, last_poll_mtime = _poll_remote_transitions(
                            remote_dir, replay, last_poll_mtime
                        )
                        remote_transitions_ingested += ingested
                        if ingested > 0:
                            print(json.dumps({
                                "event": "remote_poll_complete",
                                "ingested": ingested,
                                "total_remote_ingested": remote_transitions_ingested,
                                "replay_size": len(replay),
                            }), flush=True)
                        last_poll_time = now
                    except Exception as e:
                        print(json.dumps({
                            "event": "remote_poll_error",
                            "error": str(e),
                        }), flush=True)

                # Poll for gradient files (push-gradients mode)
                if args.push_gradients and optimizer_for_gradients is not None:
                    now = time.time()
                    if now - last_gradient_poll_time >= gradient_poll_interval:
                        try:
                            gradient_results, gradient_poll_mtime = _poll_gradients_from_shared(
                                shared_root=args.gradient_shared_root,
                                last_poll_time=gradient_poll_mtime,
                            )
                            
                            if gradient_results:
                                # Aggregate gradients from all actors
                                aggregated_grads: dict[str, torch.Tensor] = {}
                                total_actors = 0
                                
                                for actor_id, grads_dict, timestamp in gradient_results:
                                    total_actors += 1
                                    for name, grad_tensor in grads_dict.items():
                                        if name not in aggregated_grads:
                                            aggregated_grads[name] = grad_tensor.clone()
                                        else:
                                            aggregated_grads[name] += grad_tensor
                                
                                # Average the gradients
                                if total_actors > 0:
                                    for name in aggregated_grads:
                                        aggregated_grads[name] /= total_actors
                                    
                                    # Apply aggregated gradients
                                    grad_norm = _apply_gradients_to_model(
                                        online,
                                        aggregated_grads,
                                        optimizer_for_gradients,
                                        clip_norm=args.grad_clip,
                                    )
                                    
                                    gradient_step += 1
                                    
                                    # Update target network if needed
                                    if learner is not None:
                                        learner.num_updates = gradient_step
                                        learner._maybe_update_target()
                                    
                                    print(json.dumps({
                                        "event": "gradients_applied",
                                        "step": gradient_step,
                                        "actors_contributed": total_actors,
                                        "grad_norm": grad_norm,
                                        "transitions": transitions,
                                        "replay_size": len(replay),
                                    }), flush=True)
                                    
                                    # Save checkpoint periodically
                                    if gradient_step % 10 == 0:
                                        _save_checkpoint(latest_path, online, learner_cfg, gradient_step)
                        
                        except Exception as e:
                            print(json.dumps({
                                "event": "gradient_poll_error",
                                "error": str(e),
                            }), flush=True)
                        
                        last_gradient_poll_time = now

                # Periodic heartbeat
                now = time.time()
                if now - last_heartbeat >= heartbeat_interval:
                    hb = {
                        "event": "heartbeat",
                        "transitions": transitions,
                        "games_done": games_done,
                        "replay": len(replay),
                        "actors_alive": alive_count,
                        "remote_ingested": remote_transitions_ingested,
                        "uptime_minutes": round((now - start_time) / 60.0, 1),
                    }
                    if last_log:
                        hb["last_value_loss"] = last_log.get("value_loss")
                        hb["last_v_pred_mean"] = last_log.get("v_pred_mean")
                    print(json.dumps(hb), flush=True)
                    with open(game_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(hb) + "\n")
                    last_heartbeat = now
                continue

            mtype = msg.get("type")
            if mtype == "transition":
                last_transition_time = time.time()
                t = _payload_to_transition(msg["transition"])
                replay.add(t)
                transitions += 1

                train_logs = []
                if learner and not args.push_gradients:
                    train_logs = learner.maybe_train_after_turn()
                if train_logs:
                    last_log = train_logs[-1]

                if args.log_every_transitions > 0 and transitions % args.log_every_transitions == 0:
                    log = dict(msg.get("log", {}))
                    log_entry = {
                        "event": "transition",
                        "transitions": transitions,
                        "actor_id": msg.get("actor_id"),
                        "replay": len(replay),
                        "games_done": games_done,
                        **log,
                        **last_log,
                    }
                    print(json.dumps(log_entry, sort_keys=True), flush=True)
                    # Write to game log file
                    with open(game_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_entry) + "\n")

                if args.save_every_transitions > 0 and transitions % args.save_every_transitions == 0:
                    if learner:
                        _save_checkpoint(latest_path, online, learner_cfg, transitions)
                    _save_checkpoint(output_dir / f"value_rhea_{_timestamp_str()}.pt", online, learner_cfg if learner else None, transitions)

            elif mtype == "game_done":
                games_done += 1
                game_log_entry = {"event": "game_done", **msg, "total_games_done": games_done}
                print(json.dumps(game_log_entry), flush=True)
                # Write to game log file
                with open(game_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(game_log_entry) + "\n")

            elif mtype == "actor_dead":
                actor_id = int(msg.get("actor_id", -1))
                if 0 <= actor_id < len(actor_alive):
                    actor_alive[actor_id] = False
                print(json.dumps({
                    "event": "actor_dead",
                    "actor_id": actor_id,
                    "error": msg.get("error", "unknown"),
                    "actors_alive": sum(actor_alive),
                }), flush=True)
                with open(game_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"event": "actor_dead", **msg, "actors_alive": sum(actor_alive)}) + "\n")

            else:
                unknown_log_entry = {"event": "unknown_actor_msg", "msg": msg}
                print(json.dumps(unknown_log_entry), flush=True)
                # Write to game log file
                with open(game_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(unknown_log_entry) + "\n")

    finally:
        stop_event.set()
        if learner:
            _save_checkpoint(latest_path, online, learner_cfg, transitions)
        for p in procs:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()


if __name__ == "__main__":
    main()
