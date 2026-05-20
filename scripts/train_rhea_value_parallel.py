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

Gradient flow (--push-gradients):
    Remote workers (--machine-id != learner): actors write gradients locally; a background
    thread SCPs them to the learner host (--learner-gradient-dir). Another thread SCPs
    value_rhea_latest.pt (and a historical archive) down into --local-checkpoint-dir.
    Learner (--machine-id learner): main process polls --learner-gradient-dir on local disk,
    aggregates gradients, saves value_rhea_latest.pt plus timestamped backups. Local actors
    write gradients directly into the inbox and read checkpoints locally — no SCP.
"""

import argparse
import atexit
import copy
import dataclasses
import json
import multiprocessing as mp
import os
import queue
import random
import tempfile
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

# SSH/SCP for gradient transfer to workhorse1
try:
    import paramiko
    import scp
    _HAVE_PARAMIKO = True
except ImportError:
    _HAVE_PARAMIKO = False

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
from rl.env import AWBWEnv, SESSION_GAME_COUNTER_DB_ENV
from rl.rhea import RheaConfig, RheaPlanner, replay_rhea_actions
from rl.rhea_fitness import RheaFitness
from rl.rhea_replay import RheaReplayBuffer, RheaTransition
from rl.rhea_value_learner import RheaValueLearner, RheaValueLearnerConfig
from rl.value_net import AWBWValueNet, load_value_checkpoint

# ── Helper functions for gradient file naming and SSH transfer ──────────

def _is_learner_host(machine_id: str) -> bool:
    """True when this process is the central gradient aggregator."""
    return str(machine_id).strip().lower() == "learner"


def _is_remote_worker(machine_id: str, push_gradients: bool) -> bool:
    """True for --push-gradients actors on a non-learner machine."""
    return bool(push_gradients) and not _is_learner_host(machine_id)


def _est_timestamp() -> str:
    """Return human-readable timestamp in EST timezone."""
    est = timezone(timedelta(hours=-5))
    now = datetime.now(est)
    return now.strftime("%Y-%m-%d_%I-%M-%S_%p_EST")


def _gradient_filename(machine_id: str, actor_id: int) -> str:
    """Generate gradient filename: <machine-id>_<actor-id>_<EST-timestamp>.json"""
    return f"{machine_id}_{actor_id}_{_est_timestamp()}.json"


def _ssh_scp_put(local_path: str, remote_path: str, hostname: str = "192.168.0.160", username: str = "sshuser") -> bool:
    """SCP a file to remote host via paramiko. Uses atomic write (tmp + rename) on remote.
    Returns True on success."""
    if not _HAVE_PARAMIKO:
        print(json.dumps({"event": "ssh_scp_error", "error": "paramiko or scp not installed"}), flush=True)
        return False
    try:
        import paramiko
        import scp
        # Normalize remote path (use forward slashes for remote command)
        remote_path = remote_path.replace("\\", "/")
        remote_tmp = remote_path + ".tmp"
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname, username=username)
        # Create remote directory if it doesn't exist (Windows)
        remote_path_win = remote_path.replace("/", "\\")
        remote_dir = str(__import__('pathlib').PureWindowsPath(remote_path_win).parent)
        mkdir_cmd = f'if not exist "{remote_dir}" mkdir "{remote_dir}"'
        client.exec_command(mkdir_cmd)
        with scp.SCPClient(client.get_transport()) as scp_client:
            # Atomic write: SCP to .tmp, then rename on remote
            scp_client.put(local_path, remote_tmp)
            # Use Windows 'move' command (both machines are Windows)
            # Need backslashes for Windows move command
            remote_path_win = remote_path.replace("/", "\\")
            remote_tmp_win = remote_tmp.replace("/", "\\")
            rename_cmd = f'move /Y "{remote_tmp_win}" "{remote_path_win}"'
            stdin, stdout, stderr = client.exec_command(rename_cmd)
            err = stderr.read().decode(errors="replace").strip()
            if err:
                print(json.dumps({"event": "ssh_scp_rename_error", "error": err, "remote": remote_path, "cmd": rename_cmd}), flush=True)
                client.close()
                return False
        client.close()
        return True
    except Exception as e:
        print(json.dumps({"event": "ssh_scp_put_error", "error": str(e), "local": local_path, "remote": remote_path}), flush=True)
        return False


def _ssh_scp_get(remote_path: str, local_path: str, hostname: str = "192.168.0.160", username: str = "sshuser") -> bool:
    """SCP a file from remote host via paramiko. Uses atomic write (tmp + rename) locally.
    Retries on Windows file-lock errors (WinError 32). Returns True on success."""
    if not _HAVE_PARAMIKO:
        print(json.dumps({"event": "ssh_scp_error", "error": "paramiko or scp not installed"}), flush=True)
        return False
    import uuid
    local = Path(local_path)
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(local.parent / f".{local.name}.{uuid.uuid4().hex}.download")
    try:
        import paramiko
        import scp
        remote_path = remote_path.replace("\\", "/")
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname, username=username)
        with scp.SCPClient(client.get_transport()) as scp_client:
            scp_client.get(remote_path, tmp_path)
        client.close()
        for attempt in range(20):
            try:
                os.replace(tmp_path, str(local))
                return True
            except OSError as e:
                if getattr(e, "winerror", None) == 32 and attempt < 19:
                    print(json.dumps({
                        "event": "ssh_scp_get_retry",
                        "error": str(e),
                        "remote": remote_path,
                        "local": local_path,
                        "attempt": attempt + 1,
                    }), flush=True)
                    time.sleep(0.25 * (attempt + 1))
                else:
                    raise
        return False
    except Exception as e:
        print(json.dumps({"event": "ssh_scp_get_error", "error": str(e), "remote": remote_path, "local": local_path}), flush=True)
        return False
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def _ssh_list_files(remote_dir: str, hostname: str = "192.168.0.160", username: str = "sshuser") -> list[str]:
    """List files in remote directory via SSH. Returns list of filenames."""
    if not _HAVE_PARAMIKO:
        return []
    try:
        import paramiko
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname, username=username)
        remote_dir_win = remote_dir.replace("/", "\\")
        stdin, stdout, stderr = client.exec_command(f'dir /b "{remote_dir_win}" 2>NUL')
        files = [line.strip() for line in stdout.readlines() if line.strip()]
        client.close()
        return files
    except Exception as e:
        print(json.dumps({"event": "ssh_list_error", "error": str(e)}), flush=True)
        return []


def _ssh_delete_file(remote_path: str, hostname: str = "192.168.0.160", username: str = "sshuser") -> bool:
    """Delete a file on remote host via SSH. Returns True on success."""
    if not _HAVE_PARAMIKO:
        return False
    try:
        import paramiko
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname, username=username)
        stdin, stdout, stderr = client.exec_command(f'del "{remote_path}" /q')
        client.close()
        return True
    except Exception as e:
        print(json.dumps({"event": "ssh_delete_error", "error": str(e)}), flush=True)
        return False


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


def _save_checkpoint(path: Path, model: AWBWValueNet, learner_cfg: RheaValueLearnerConfig | None, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "step": step,
    }
    if learner_cfg is not None:
        payload["learner_cfg"] = dataclasses.asdict(learner_cfg)
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _maybe_save_learner_checkpoints(
    *,
    latest_path: Path,
    output_dir: Path,
    model: AWBWValueNet,
    learner_cfg: RheaValueLearnerConfig | None,
    step: int,
    save_every: int,
    grad_norm: float | None,
) -> None:
    """Persist latest (every 10 steps) and timestamped archive on save-every boundary."""
    if grad_norm is None:
        return
    if isinstance(grad_norm, float) and torch.isnan(torch.tensor(grad_norm)):
        return
    if step % 10 == 0:
        _save_checkpoint(latest_path, model, learner_cfg, step)
    if save_every > 0 and step % save_every == 0:
        _save_checkpoint(latest_path, model, learner_cfg, step)
        archive = output_dir / f"value_rhea_{_timestamp_str()}.pt"
        _save_checkpoint(archive, model, learner_cfg, step)
        print(json.dumps({
            "event": "checkpoint_archive_saved",
            "step": step,
            "path": str(archive),
        }), flush=True)


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
    # Important: ensure ALL parameters that require grad have valid gradients
    for name, p in model.named_parameters():
        if p.requires_grad:
            if name in grads_dict:
                p.grad = grads_dict[name].to(p.device)
            else:
                # Parameter not in grads_dict - this shouldn't happen, but if it does,
                # set gradient to zero to avoid NaN
                p.grad = torch.zeros_like(p)
    
    # Clip and step
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        clip_norm,
    )
    
    # Check for NaN gradients before applying
    nan_grads = [name for name, p in model.named_parameters() if p.grad is not None and torch.isnan(p.grad).any()]
    if nan_grads:
        print(json.dumps({"event": "skip_nan_apply", "nan_grads_count": len(nan_grads)}), flush=True)
        opt.zero_grad(set_to_none=True)
        return None  # Signal that update was skipped
    
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


def _sync_local_gradients_to_shared_worker(
    local_dir: str,
    machine_id: str,
    actor_id: int,
    stop_event: mp.synchronize.Event,
    sync_interval: float = 30.0,
    hostname: str = "192.168.0.160",
    username: str = "sshuser",
    remote_grad_dir: str = "D:/awbw/data/gradients",
) -> None:
    """Background thread that SCPs local gradient files to 192.168.0.160:D:/awbw/data/gradients/."""
    import json
    import glob
    import os

    local_path = Path(local_dir)
    if not local_path.exists():
        return

    while not stop_event.is_set():
        try:
            pattern = str(local_path / "*.json")
            local_files = [f for f in glob.glob(pattern) if not f.endswith(".tmp")]

            for lf in local_files:
                lf_path = Path(lf)
                try:
                    # SCP to workhorse1
                    remote_path = f"{remote_grad_dir}\\{lf_path.name}"
                    if _ssh_scp_put(str(lf_path), remote_path, hostname, username):
                        lf_path.unlink()  # Delete local copy after successful transfer
                        print(json.dumps({
                            "event": "gradient_scp_to_learner",
                            "actor_id": actor_id,
                            "machine_id": machine_id,
                            "file": lf_path.name,
                            "remote": remote_path,
                        }), flush=True)
                    else:
                        print(json.dumps({
                            "event": "gradient_scp_failed",
                            "actor_id": actor_id,
                            "file": lf_path.name,
                        }), flush=True)
                except Exception as e:
                    print(json.dumps({
                        "event": "gradient_sync_error",
                        "actor_id": actor_id,
                        "file": str(lf_path),
                        "error": str(e),
                    }), flush=True)
        except Exception as e:
            print(json.dumps({
                "event": "gradient_sync_worker_error",
                "actor_id": actor_id,
                "error": str(e),
            }), flush=True)

        for _ in range(int(sync_interval * 10)):
            if stop_event.is_set():
                break
            time.sleep(0.1)


def _sync_checkpoint_from_shared_worker(
    machine_id: str,
    local_dir: str,
    stop_event: mp.synchronize.Event,
    sync_interval: float = 60.0,
    hostname: str = "192.168.0.160",
    username: str = "sshuser",
    remote_checkpoint_dir: str = "D:/awbw/checkpoints",
) -> None:
    """Single machine-level thread: SCP value_rhea_latest.pt from learner into local_dir."""
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    local_checkpoint = local_path / "value_rhea_latest.pt"
    local_new = local_path / "value_rhea_latest.pt.new"
    remote_checkpoint = remote_checkpoint_dir.replace("\\", "/") + "/value_rhea_latest.pt"
    last_applied_remote_mtime = 0.0

    while not stop_event.is_set():
        try:
            remote_files = _ssh_list_files(remote_checkpoint_dir, hostname, username)
            if "value_rhea_latest.pt" not in remote_files:
                for _ in range(int(sync_interval * 10)):
                    if stop_event.is_set():
                        break
                    time.sleep(0.1)
                continue

            if not _ssh_scp_get(remote_checkpoint, str(local_new), hostname, username):
                for _ in range(int(sync_interval * 10)):
                    if stop_event.is_set():
                        break
                    time.sleep(0.1)
                continue

            new_mtime = local_new.stat().st_mtime
            if new_mtime <= last_applied_remote_mtime and local_checkpoint.exists():
                local_new.unlink(missing_ok=True)
                for _ in range(int(sync_interval * 10)):
                    if stop_event.is_set():
                        break
                    time.sleep(0.1)
                continue

            for attempt in range(20):
                try:
                    os.replace(str(local_new), str(local_checkpoint))
                    last_applied_remote_mtime = new_mtime
                    print(json.dumps({
                        "event": "checkpoint_synced_from_learner",
                        "machine_id": machine_id,
                        "remote": remote_checkpoint,
                        "local": str(local_checkpoint),
                    }), flush=True)
                    break
                except OSError as e:
                    if getattr(e, "winerror", None) == 32 and attempt < 19:
                        time.sleep(0.25 * (attempt + 1))
                    else:
                        raise
        except Exception as e:
            print(json.dumps({
                "event": "checkpoint_sync_worker_error",
                "machine_id": machine_id,
                "error": str(e),
            }), flush=True)

        for _ in range(int(sync_interval * 10)):
            if stop_event.is_set():
                break
            time.sleep(0.1)


def _sync_hist_checkpoint_from_learner_worker(
    machine_id: str,
    local_dir: str,
    stop_event: mp.synchronize.Event,
    sync_interval: float = 120.0,
    hostname: str = "192.168.0.160",
    username: str = "sshuser",
    remote_checkpoint_dir: str = "D:/awbw/checkpoints",
) -> None:
    """SCP the oldest timestamped value_rhea_*.pt archive from learner for hist self-play."""
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    local_hist = local_path / "value_rhea_hist.pt"
    remote_dir_norm = remote_checkpoint_dir.replace("\\", "/")
    last_synced_name = ""

    while not stop_event.is_set():
        try:
            remote_files = _ssh_list_files(remote_checkpoint_dir, hostname, username)
            hist_names = sorted(
                f for f in remote_files
                if f.startswith("value_rhea_")
                and f.endswith(".pt")
                and "latest" not in f.lower()
            )
            if not hist_names:
                for _ in range(int(sync_interval * 10)):
                    if stop_event.is_set():
                        break
                    time.sleep(0.1)
                continue

            oldest_name = hist_names[0]
            if oldest_name == last_synced_name and local_hist.exists():
                for _ in range(int(sync_interval * 10)):
                    if stop_event.is_set():
                        break
                    time.sleep(0.1)
                continue

            remote_hist = f"{remote_dir_norm}/{oldest_name}"
            local_new = local_path / "value_rhea_hist.pt.new"
            if not _ssh_scp_get(remote_hist, str(local_new), hostname, username):
                for _ in range(int(sync_interval * 10)):
                    if stop_event.is_set():
                        break
                    time.sleep(0.1)
                continue

            for attempt in range(20):
                try:
                    os.replace(str(local_new), str(local_hist))
                    last_synced_name = oldest_name
                    print(json.dumps({
                        "event": "hist_checkpoint_synced_from_learner",
                        "machine_id": machine_id,
                        "remote": remote_hist,
                        "local": str(local_hist),
                    }), flush=True)
                    break
                except OSError as e:
                    if getattr(e, "winerror", None) == 32 and attempt < 19:
                        time.sleep(0.25 * (attempt + 1))
                    else:
                        raise
        except Exception as e:
            print(json.dumps({
                "event": "hist_checkpoint_sync_worker_error",
                "machine_id": machine_id,
                "error": str(e),
            }), flush=True)

        for _ in range(int(sync_interval * 10)):
            if stop_event.is_set():
                break
            time.sleep(0.1)


def _poll_gradients_for_learner(
    gradient_dir: str = "D:/awbw/data/gradients",
    last_poll_mtime: dict[str, float] | None = None,
) -> tuple[list[tuple[str, dict[str, torch.Tensor], float]], dict[str, float]]:
    """Poll local gradient directory for gradient files.
    
    For learner (machine-id == "learner") running on workhorse1.
    Reads gradient files directly from D:/awbw/data/gradients/.
    Also checks D:/awbw/.tmp/vessel/ for worker gradients and moves them.
    Returns: (results, updated_last_poll_mtime)
    """
    import json
    import glob
    
    if last_poll_mtime is None:
        last_poll_mtime = {}
    
    # First, move any files from vessel directory to gradient_dir
    vessel_dir = Path("d:/awbw/.tmp/vessel")
    if vessel_dir.exists():
        try:
            for vf in vessel_dir.glob("workstation_*.json"):
                try:
                    with open(vf, "r", encoding="utf-8") as f:
                        grad_data = json.load(f)
                    actor_id = grad_data.get("actor_id", 0)
                    target_dir = Path(gradient_dir) / f"actor-{actor_id}"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_path = target_dir / vf.name
                    vf.rename(target_path)
                    print(json.dumps({
                        "event": "vessel_moved",
                        "file": str(vf.name),
                        "actor_id": actor_id,
                        "to": str(target_path),
                    }), flush=True)
                except Exception as e:
                    print(json.dumps({
                        "event": "vessel_error",
                        "file": str(vf),
                        "error": str(e),
                    }), flush=True)
        except Exception as e:
            print(json.dumps({
                "event": "vessel_poll_error",
                "error": str(e),
            }), flush=True)
    
    grad_dir = Path(gradient_dir)
    if not grad_dir.exists():
        return [], last_poll_mtime
    
    # Recursively find all .json files in subdirectories (skip .tmp)
    pattern = str(grad_dir / "**/*.json")
    grad_files = [f for f in glob.glob(pattern, recursive=True) if not f.endswith(".tmp")]
    
    results = []
    for fpath in grad_files:
        f = Path(fpath)
        try:
            mtime = f.stat().st_mtime
            if fpath in last_poll_mtime and last_poll_mtime[fpath] >= mtime:
                continue
            
            with open(f, "r", encoding="utf-8") as fh:
                grad_data = json.load(fh)
            
            machine_id = grad_data.get("machine_id", "unknown")
            actor_id = grad_data["actor_id"]
            timestamp = grad_data.get("timestamp", mtime)
            
            # Convert lists back to tensors
            grads_dict = {}
            for name, grad_list in grad_data["gradients"].items():
                grads_dict[name] = torch.tensor(grad_list)
            
            results.append((fpath, machine_id, actor_id, grads_dict, timestamp))
            last_poll_mtime[fpath] = mtime
            
        except Exception as e:
            print(json.dumps({
                "event": "gradient_read_error",
                "file": str(f),
                "error": str(e),
            }), flush=True)
    
    return results, last_poll_mtime


def _write_gradients_locally(
    actor_id: int,
    grads: dict,
    step_num: int,
    local_dir: str,
    machine_id: str = "actor",
) -> str | None:
    """Write gradient deltas to local filesystem.
    
    Filename: <machine-id>_<actor-id>_<EST-timestamp>.json
    """
    import json

    try:
        grad_dir = Path(local_dir)
        grad_dir.mkdir(parents=True, exist_ok=True)

        grad_data = {
            "actor_id": actor_id,
            "machine_id": machine_id,
            "step": step_num,
            "timestamp": time.time(),
            "gradients": grads,
        }

        filename = _gradient_filename(machine_id, actor_id)
        final_path = grad_dir / filename
        tmp_path = grad_dir / (filename + ".tmp")

        # Write to temp file, then atomically rename to final path.
        # This prevents corrupt files if the process is killed mid-write.
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(grad_data, f)
        os.replace(str(tmp_path), str(final_path))

        return str(final_path)

    except Exception as e:
        print(json.dumps({
            "event": "local_gradient_write_error",
            "actor_id": actor_id,
            "error": str(e),
        }), flush=True)
        return None


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
        
        # Determine where to read checkpoints from:
        # Checkpoints always land under D:/awbw/checkpoints/ (project root).
        # Gradients remain on D:/data/ (separate disk).
        _actor_project_root = Path(__file__).resolve().parents[1]
        latest_path = _actor_project_root / "checkpoints" / "value_rhea_latest.pt"
        
        # Load checkpoint - use clean checkpoint, never the potentially corrupted latest.pt
        # This prevents NaN gradient propagation from corrupted models.
        actor_checkpoint = args.checkpoint
        # If using latest.pt, try to use a clean historical checkpoint instead
        checkpoint_path = Path(actor_checkpoint)
        if checkpoint_path.name.lower() == "value_rhea_latest.pt":
            # Look for clean historical checkpoints (not latest)
            import glob
            # Search in the same directory where we expect latest.pt to be
            search_dir = latest_path.parent
            clean_candidates = sorted(
                [p for p in search_dir.glob("value_rhea_*.pt") 
                 if "latest" not in p.name.lower()],
                key=lambda p: p.stat().st_mtime,
                reverse=True  # Newest first
            )
            if clean_candidates:
                actor_checkpoint = str(clean_candidates[0])
                print(json.dumps({
                    "event": "actor_using_clean_checkpoint",
                    "actor_id": actor_id,
                    "original": str(checkpoint_path),
                    "using": actor_checkpoint,
                }), flush=True)
        
        try:
            value_model = load_value_checkpoint(actor_checkpoint, device=actor_device)
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

        # Gradient pushing state (A3C-style) — set paths before hist discovery / sync threads.
        push_gradients = bool(getattr(args, "push_gradients", False))
        on_learner = _is_learner_host(getattr(args, "machine_id", "actor"))
        learner_grad_dir = getattr(args, "learner_gradient_dir", "D:/awbw/data/gradients")
        learner_ckpt_dir = getattr(args, "learner_checkpoint_dir", "D:/awbw/checkpoints")
        learner_host = getattr(args, "learner_host", "192.168.0.160")
        learner_user = getattr(args, "learner_user", "sshuser")

        local_grad_dir = args.local_gradient_dir
        local_ckpt_dir = args.local_checkpoint_dir

        if push_gradients:
            if on_learner:
                local_grad_dir = local_grad_dir or f"{learner_grad_dir}/actor-{actor_id}"
                latest_path = Path(learner_ckpt_dir) / "value_rhea_latest.pt"
                local_ckpt_dir = None
                print(json.dumps({
                    "event": "actor_learner_local_paths",
                    "actor_id": actor_id,
                    "grad_dir": local_grad_dir,
                    "latest_path": str(latest_path),
                }), flush=True)
            else:
                local_grad_dir = local_grad_dir or f"D:/awbw/data/gradients/actor-{actor_id}"
                local_ckpt_dir = local_ckpt_dir or "D:/awbw/checkpoints/local_checkpoints"
                latest_path = Path(local_ckpt_dir) / "value_rhea_latest.pt"
                print(json.dumps({
                    "event": "actor_using_local_checkpoint_dir",
                    "actor_id": actor_id,
                    "local_ckpt_dir": local_ckpt_dir,
                    "latest_path": str(latest_path),
                }), flush=True)

        # Load historical checkpoint for hist-prob games (after local_ckpt_dir is known).
        hist_checkpoint_path = args.hist_checkpoint_path
        if args.dual_gradient_self_play and args.dual_gradient_hist_prob > 0:
            if not hist_checkpoint_path:
                try:
                    hist_search_dirs: list[Path] = []
                    if push_gradients and local_ckpt_dir:
                        hist_search_dirs.append(Path(local_ckpt_dir))
                    hist_search_dirs.append(_actor_project_root / "checkpoints")
                    hist_fixed = (
                        Path(local_ckpt_dir) / "value_rhea_hist.pt"
                        if push_gradients and local_ckpt_dir
                        else None
                    )
                    if hist_fixed is not None and hist_fixed.exists():
                        hist_checkpoint_path = str(hist_fixed)
                    else:
                        for ckpt_dir in hist_search_dirs:
                            if not ckpt_dir.exists():
                                continue
                            hist_candidates = sorted(
                                [
                                    p for p in ckpt_dir.glob("value_rhea_*.pt")
                                    if p.name not in (
                                        "value_rhea_latest.pt",
                                        "value_rhea_latest_backup.pt",
                                        "value_rhea_hist.pt",
                                    )
                                ],
                                key=lambda p: p.stat().st_mtime,
                            )
                            if hist_candidates:
                                hist_checkpoint_path = str(hist_candidates[0])
                                break
                    if hist_checkpoint_path:
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
                    reward_weight=args.reward_weight,
                    value_weight=args.value_weight,
                    build_value_weight=args.build_value_weight,
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

        # Gradient pushing accumulators (paths configured above).
        local_transitions: list[RheaTransition] = []
        gradient_step = 0
        learner_cfg_for_grads = RheaValueLearnerConfig(
            gamma_turn=args.gamma_turn,
            target_clip=args.target_clip,
        )

        sync_stop_event = mp.Event()

        # Remote workers: per-actor gradient SCP only (checkpoint/hist sync runs in main).
        if push_gradients and local_grad_dir and not on_learner:
            grad_sync_thread = threading.Thread(
                target=_sync_local_gradients_to_shared_worker,
                args=(
                    local_grad_dir,
                    args.machine_id,
                    actor_id,
                    sync_stop_event,
                    args.gradient_sync_interval,
                    learner_host,
                    learner_user,
                    learner_grad_dir,
                ),
                daemon=True,
                name=f"grad-sync-{actor_id}",
            )
            grad_sync_thread.start()
            print(json.dumps({
                "event": "gradient_sync_thread_started",
                "actor_id": actor_id,
                "machine_id": args.machine_id,
                "local_dir": local_grad_dir,
                "learner_host": learner_host,
            }), flush=True)

        last_hist_mtime = 0.0
        last_ckpt_mtime = 0.0
        if hist_value_model is not None and hist_checkpoint_path:
            try:
                last_hist_mtime = Path(hist_checkpoint_path).stat().st_mtime
            except OSError:
                last_hist_mtime = 0.0
        if push_gradients and local_ckpt_dir and latest_path.exists():
            try:
                last_ckpt_mtime = latest_path.stat().st_mtime
            except OSError:
                last_ckpt_mtime = 0.0

        while not stop_event.is_set():
            try:
                # Shared checkpoint updated by main-process sync thread; reload on mtime change.
                if push_gradients and local_ckpt_dir and not on_learner:
                    shared_latest = Path(local_ckpt_dir) / "value_rhea_latest.pt"
                    if shared_latest.exists():
                        try:
                            ckpt_mtime = shared_latest.stat().st_mtime
                            if ckpt_mtime > last_ckpt_mtime:
                                if _load_value_pt_into_model(
                                    shared_latest, value_model, actor_device, verbose=True,
                                ):
                                    last_ckpt_mtime = ckpt_mtime
                                    last_refresh = time.time()
                                    print(json.dumps({
                                        "event": "actor_checkpoint_reloaded",
                                        "actor_id": actor_id,
                                        "path": str(shared_latest),
                                    }), flush=True)
                        except OSError as ckpt_exc:
                            print(json.dumps({
                                "event": "actor_checkpoint_reload_error",
                                "actor_id": actor_id,
                                "error": repr(ckpt_exc),
                            }), flush=True)

                # Reload historical opponent when main-process sync delivers a new archive.
                if (
                    push_gradients
                    and local_ckpt_dir
                    and args.dual_gradient_self_play
                    and args.dual_gradient_hist_prob > 0
                ):
                    hist_path = Path(local_ckpt_dir) / "value_rhea_hist.pt"
                    if hist_path.exists():
                        try:
                            hist_mtime = hist_path.stat().st_mtime
                            if hist_mtime > last_hist_mtime:
                                hist_value_model = load_value_checkpoint(
                                    str(hist_path), device=actor_device,
                                )
                                last_hist_mtime = hist_mtime
                                local_hist_prob = args.dual_gradient_hist_prob
                                print(json.dumps({
                                    "event": "hist_checkpoint_reloaded",
                                    "actor_id": actor_id,
                                    "path": str(hist_path),
                                }), flush=True)
                        except Exception as hist_exc:
                            print(json.dumps({
                                "event": "hist_checkpoint_reload_error",
                                "actor_id": actor_id,
                                "error": repr(hist_exc),
                            }), flush=True)

                # Decide if this game uses historical checkpoint opponent
                use_hist_checkpoint = random.random() < local_hist_prob
                
                # Timer-based refresh for learner-local actors; remote workers use mtime above.
                now = time.time()
                if (
                    args.actor_refresh_seconds > 0
                    and now - last_refresh >= args.actor_refresh_seconds
                    and not (push_gradients and local_ckpt_dir and not on_learner)
                ):
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


                # Track abnormal termination (e.g. IllegalActionError from RHEA
                # producing an illegal action).  When set, game_done will carry
                # abnormal_termination=True so downstream analysis can separate
                # clean engine wins from buggy early exits.
                _abnormal_exit_error = None

                while env.state is not None and env.state.winner is None and not stop_event.is_set():
                    try:
                        state = env.state
                        acting = int(state.active_player)
                        day = int(getattr(state, "turn", getattr(state, "day", 0)))
                        if game_turns <= 20:
                            print(json.dumps({"event": "debug_turn", "day": day, "max_days": args.max_days, "winner": state.winner, "turn": state.turn, "game_turns": game_turns}), flush=True)

                        # Swap value model if using historical checkpoint opponent
                        if use_hist_checkpoint and hist_value_model is not None:
                            # Use hist_value_model for opponent (seat 1 - enemy), current model for self (seat 0)
                            fitness.value_model = hist_value_model if acting == 1 else value_model

                        # Encode before state using pre-allocated buffers
                        _encode_into(state, acting, spatial_buf, scalars_buf)
                        before_spatial = spatial_buf.copy()
                        before_scalars = scalars_buf.copy()
                        state_before = env.state
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

                        # Replay actions on real environment.
                        applied_actions, skipped_actions = replay_rhea_actions(env.state, result.actions, acting)

                        if skipped_actions > 0:
                            print(json.dumps({
                                "event": "actions_skipped_during_replay",
                                "actor_id": actor_id,
                                "game_turns": game_turns,
                                "day": day,
                                "skipped": skipped_actions,
                                "applied": applied_actions,
                            }), flush=True)
                            _abnormal_exit_error = _abnormal_exit_error or f"actions_skipped:{skipped_actions}"

                        after = env.state
                        if after is None:
                            print(json.dumps({"event": "state_none_after_turn", "game_turns": game_turns, "day": day, "acting": acting}), flush=True)
                            break

                        # If no actions were applied and we're still on the
                        # same player, the turn is stuck -- break to avoid
                        # an infinite loop of zero-progress "transitions".
                        if applied_actions == 0 and after is not None and int(after.active_player) == acting:
                            print(json.dumps({
                                "event": "turn_stuck_no_progress",
                                "actor_id": actor_id,
                                "game_turns": game_turns,
                                "day": day,
                                "skipped": skipped_actions,
                                "action_stage": after.action_stage.name,
                            }), flush=True)
                            _abnormal_exit_error = _abnormal_exit_error or "turn_stuck_no_progress"
                            break

                        _encode_into(after, acting, spatial_buf, scalars_buf)
                        after_spatial = spatial_buf.copy()
                        after_scalars = scalars_buf.copy()
                        phi_after = fitness.phi(after, acting)
                        reward_turn = float(phi_after - phi_before)
                        done = bool(after.winner is not None)

                        try:
                            env.record_rhea_turn(
                                acting,
                                state_before=state_before,
                                phi_before=phi_before,
                                phi_after=phi_after,
                                terminal_done=done,
                            )
                        except Exception as log_turn_exc:
                            print(json.dumps({
                                "event": "rhea_turn_log_error",
                                "actor_id": actor_id,
                                "error": repr(log_turn_exc),
                            }), flush=True)

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
                                    # Write gradients locally (background thread SCPs to workhorse1)
                                    if local_grad_dir:
                                        grad_path = _write_gradients_locally(
                                            actor_id,
                                            grads,
                                            gradient_step,
                                            local_grad_dir,
                                            machine_id=args.machine_id,
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
                                        "actions_planned": len(result.actions),
                                        "actions_applied": applied_actions,
                                        "actions_skipped": skipped_actions,
                                        "use_hist_checkpoint": use_hist_checkpoint,
                                    },
                                }
                            )

                        transitions_sent += 1
                        game_turns += 1

                        if day > args.max_days:
                            print(json.dumps({"event": "day_limit_break", "day": day, "max_days": args.max_days, "game_turns": game_turns, "winner": after.winner, "engine_turn": after.turn}), flush=True)
                            break

                        # Diagnostic: if we're on turn 12 with no winner, log it
                        if game_turns == 12 and after.winner is None:
                            print(json.dumps({"event": "turn12_no_winner", "day": day, "max_days": args.max_days, "engine_turn": after.turn, "env_done": after.done}), flush=True)

                    except Exception as e:
                        import traceback
                        print(json.dumps({
                            "event": "turn_error",
                            "actor_id": actor_id,
                            "error": repr(e),
                            "game_turns": game_turns,
                            "traceback": traceback.format_exc(),
                        }), flush=True)
                        _abnormal_exit_error = repr(e)
                        break  # Exit the inner while loop for this game

                games_done += 1
                truncated = bool(
                    env.state is not None
                    and env.state.winner is None
                )
                truncation_reason = "max_env_steps" if truncated else None
                async_mode = None
                if args.dual_gradient_self_play:
                    async_mode = "hist" if use_hist_checkpoint else "mirror"
                try:
                    env.finalize_rhea_episode(
                        truncated=truncated,
                        truncation_reason=truncation_reason,
                        async_rollout_mode=async_mode,
                    )
                except Exception as log_game_exc:
                    print(json.dumps({
                        "event": "rhea_game_log_error",
                        "actor_id": actor_id,
                        "error": repr(log_game_exc),
                    }), flush=True)

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
                        "abnormal_termination": _abnormal_exit_error is not None,
                        "termination_error": _abnormal_exit_error,
                        "env_state_none": env.state is None,
                    }
                )
            except Exception as exc:
                # Signal background threads to stop
                if 'sync_stop_event' in dir():
                    sync_stop_event.set()
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
        # Signal background threads to stop
        if 'sync_stop_event' in dir():
            sync_stop_event.set()
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

    # Machine identity
    ap.add_argument("--machine-id", type=str, default="actor",
                      help="Machine identifier (e.g. 'pc-b', 'workstation', 'learner'). "
                           "Use 'learner' on the central training host; all other ids are remote workers.")
    ap.add_argument("--learner-host", type=str, default="192.168.0.160",
                      help="Learner SSH host for remote worker SCP (default: workhorse1)")
    ap.add_argument("--learner-user", type=str, default="sshuser",
                      help="Learner SSH username for remote worker SCP")
    ap.add_argument("--learner-gradient-dir", type=str, default="D:/awbw/data/gradients",
                      help="Directory on learner where gradient JSON files are collected")
    ap.add_argument("--learner-checkpoint-dir", type=str, default="D:/awbw/checkpoints",
                      help="Directory on learner where value_rhea_latest.pt is published")

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
    ap.add_argument("--rhea-population", type=int, default=64)
    ap.add_argument("--rhea-generations", type=int, default=10)
    ap.add_argument("--rhea-elite", type=int, default=8)
    ap.add_argument("--rhea-mutation-rate", type=float, default=0.20)
    ap.add_argument("--rhea-top-k-per-state", type=int, default=48)
    ap.add_argument("--reward-weight", type=float, default=0.90)
    ap.add_argument("--value-weight", type=float, default=0.10)
    # Tactical beam.
    ap.add_argument("--rhea-use-tactical-beam", action="store_true")
    ap.add_argument("--rhea-tactical-beam-max-width", type=int, default=48)
    ap.add_argument("--rhea-tactical-beam-max-depth", type=int, default=14)
    ap.add_argument("--rhea-tactical-beam-max-expand", type=int, default=24)

    # Build punishment.
    ap.add_argument("--build-punishment", type=float, default=0.0,
                      help="AWBW_BUILD_PUNISHMENT: non-zero penalises ending turn with owned bases but no build")
    ap.add_argument("--build-value-weight", type=float, default=1.0,
                      help="Independent reward weight for build army value (extruded from phi)")

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
                      help="Enable actors to compute gradients locally. Remote workers SCP to "
                           "--learner-host; learner (--machine-id learner) polls --learner-gradient-dir.")
    ap.add_argument("--gradient-batch-size", type=int, default=32,
                      help="Number of transitions to accumulate before computing and pushing gradients (default: 32)")
    ap.add_argument("--gradient-poll-interval", type=float, default=30.0,
                      help="Seconds between learner polling for gradient files (default: 30)")

    # Local gradient / checkpoint optimization (avoid Samba writes)
    ap.add_argument("--local-gradient-dir", type=str, default=None,
                      help="Local directory for gradient writes (actors write here, background thread SCPs to workhorse1). "
                           "Default: on Windows aux (Z: mounted) -> D:/awbw/checkpoints/local_gradients/actor-{id} "
                           "(or C:/Users/sshuser/AWBW if exists), else -> checkpoints/local_gradients/actor-{id}")
    ap.add_argument("--gradient-sync-interval", type=float, default=30.0,
                      help="Seconds between SCPing local gradients to workhorse1 (default: 30)")
    ap.add_argument("--local-checkpoint-dir", type=str, default=None,
                      help="Local directory for checkpoints (background thread SCPs from workhorse1 to here). "
                           "Default: on Windows aux (Z: mounted) -> D:/awbw/checkpoints/local_checkpoints "
                           "(or C:/Users/sshuser/AWBW if exists), else -> checkpoints/local_checkpoints")
    ap.add_argument("--checkpoint-sync-interval", type=float, default=60.0,
                      help="Seconds between SCPing checkpoints from workhorse1 to local (default: 60)")

    # Remote transition polling (multi-machine)
    ap.add_argument("--remote-transition-dir", type=str, default=None,
                      help="Directory to poll for remote transition files (default: <shared-root>/fleet/*/transitions/)")
    ap.add_argument("--poll-remote-transitions-interval", type=float, default=60.0,
                      help="Seconds between polling remote transition directories (default: 60)")
    # Game logs: full schema rows go to logs/game_log.jsonl via env.finalize_rhea_episode.
    # Training heartbeats still go to logs/games_log.jsonl.

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

    mid = getattr(args, "machine_id", None)
    if mid is not None and str(mid).strip() and str(mid).strip().lower() != "actor":
        os.environ["AWBW_MACHINE_ID"] = str(mid).strip()
    else:
        os.environ.pop("AWBW_MACHINE_ID", None)

    bp = getattr(args, "build_punishment", None)
    if bp is not None and float(bp) > 0.0:
        os.environ["AWBW_BUILD_PUNISHMENT"] = str(bp)
    else:
        os.environ.pop("AWBW_BUILD_PUNISHMENT", None)


def main() -> None:
    args = build_arg_parser().parse_args()
    _setup_env_vars(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.push_gradients and not _is_learner_host(args.machine_id) and not args.no_learner:
        print(json.dumps({
            "event": "auto_no_learner",
            "machine_id": args.machine_id,
            "message": "Remote worker with --push-gradients: main process will not train (actors only).",
        }), flush=True)
        args.no_learner = True

    is_remote_worker = _is_remote_worker(args.machine_id, args.push_gradients) and args.no_learner

    if int(args.gpu_actors) < 0:
        raise ValueError("--gpu-actors must be >= 0")
    if int(args.gpu_actors) > int(args.n_envs):
        raise ValueError("--gpu-actors cannot exceed --n-envs")
    if int(args.gpu_actors) > 0 and not str(args.actor_gpu_device).startswith("cuda"):
        if args.verbose:
            print(json.dumps({"event": "warning", "message": "--gpu-actors > 0 but --actor-gpu-device is not cuda*"}), flush=True)

    # Resolve project root: scripts/../ = D:/awbw/
    _project_root = Path(__file__).resolve().parents[1]

    # Checkpoints always land under project checkpoints/ unless this is the learner host.
    output_dir = Path(args.learner_checkpoint_dir) if _is_learner_host(args.machine_id) else (_project_root / "checkpoints")
    latest_path = output_dir / "value_rhea_latest.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    if _is_learner_host(args.machine_id):
        Path(args.learner_gradient_dir).mkdir(parents=True, exist_ok=True)
    # Convert args to JSON-serializable dict (handle Path objects)
    hparams = vars(args).copy()
    for k, v in hparams.items():
        if isinstance(v, Path):
            hparams[k] = str(v)
    (output_dir / "hparams_parallel.json").write_text(json.dumps(hparams, indent=2), encoding="utf-8")
    
    # Game log file
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    game_log_path = logs_dir / "games_log.jsonl"

    online: AWBWValueNet | None
    target: AWBWValueNet | None
    replay: RheaReplayBuffer
    learner = None
    learner_cfg = None

    if is_remote_worker:
        online = None
        target = None
        replay = RheaReplayBuffer(args.replay_size, seed=args.seed)
        print(json.dumps({
            "event": "remote_worker_main",
            "machine_id": args.machine_id,
            "learner_host": args.learner_host,
        }), flush=True)
    else:
        online = load_value_checkpoint(args.checkpoint, device=args.device)
        target = copy.deepcopy(online)
        replay = RheaReplayBuffer(args.replay_size, seed=args.seed)

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
    # Remote workers read checkpoints via SCP; learner/local hosts seed latest here.
    latest_path.parent.mkdir(parents=True, exist_ok=True)

    if not is_remote_worker:
        import shutil
        clean_checkpoint = Path(args.checkpoint) if Path(args.checkpoint).exists() else None
        if clean_checkpoint and clean_checkpoint.exists():
            abs_src = clean_checkpoint.resolve()
            abs_dst = latest_path.resolve()
            if abs_src == abs_dst:
                print(json.dumps({
                    "event": "initial_checkpoint_same_file",
                    "path": str(abs_src),
                }), flush=True)
            else:
                shutil.copy(abs_src, abs_dst)
                print(json.dumps({
                    "event": "initial_checkpoint_copied",
                    "from": str(abs_src),
                    "to": str(abs_dst),
                }), flush=True)
        elif online is not None:
            _save_checkpoint(latest_path, online, learner_cfg, 0)
        print(json.dumps({"event": "initial_checkpoint_ready", "path": str(latest_path)}), flush=True)

    # Gradient aggregation state (for push-gradients mode)
    gradient_poll_interval = float(getattr(args, "gradient_poll_interval", 30.0))
    last_gradient_poll_time = 0.0
    gradient_poll_mtime: dict[str, float] = {}
    gradient_step = 0
    optimizer_for_gradients = None
    
    if args.push_gradients and learner is not None and online is not None:
        # Create optimizer for applying remote gradients (learner host only)
        params = [p for p in online.parameters() if p.requires_grad]
        optimizer_for_gradients = torch.optim.AdamW(
            params,
            lr=learner_cfg.value_lr,
            weight_decay=learner_cfg.weight_decay,
        )
        print(json.dumps({
            "event": "gradient_aggregation_ready",
            "gradient_poll_interval": gradient_poll_interval,
            "machine_id": args.machine_id,
            "gradient_dir": args.learner_gradient_dir,
        }), flush=True)

    session_tmp = _project_root / ".tmp"
    session_tmp.mkdir(parents=True, exist_ok=True)
    fd, session_counter_db = tempfile.mkstemp(
        prefix="awbw_rhea_session_games_",
        suffix=".sqlite",
        dir=str(session_tmp),
    )
    os.close(fd)
    os.environ[SESSION_GAME_COUNTER_DB_ENV] = session_counter_db
    atexit.register(lambda p=session_counter_db: Path(p).unlink(missing_ok=True))

    ctx = mp.get_context("spawn")
    out_q: mp.Queue = ctx.Queue(maxsize=int(args.queue_size))
    stop_event: mp.Event = ctx.Event()
    procs: list[mp.Process] = []

    for actor_id in range(int(args.n_envs)):
        p = ctx.Process(target=_actor_loop, args=(actor_id, args, out_q, stop_event), daemon=True)
        p.start()
        procs.append(p)

    worker_sync_stop = threading.Event()
    worker_sync_threads: list[threading.Thread] = []
    if is_remote_worker and args.push_gradients:
        worker_local_ckpt_dir = args.local_checkpoint_dir or "D:/awbw/checkpoints/local_checkpoints"
        Path(worker_local_ckpt_dir).mkdir(parents=True, exist_ok=True)
        worker_sync_threads.append(threading.Thread(
            target=_sync_checkpoint_from_shared_worker,
            args=(
                args.machine_id,
                worker_local_ckpt_dir,
                worker_sync_stop,
                args.checkpoint_sync_interval,
                args.learner_host,
                args.learner_user,
                args.learner_checkpoint_dir,
            ),
            daemon=True,
            name="main-ckpt-sync",
        ))
        worker_sync_threads.append(threading.Thread(
            target=_sync_hist_checkpoint_from_learner_worker,
            args=(
                args.machine_id,
                worker_local_ckpt_dir,
                worker_sync_stop,
                max(args.checkpoint_sync_interval * 2, 120.0),
                args.learner_host,
                args.learner_user,
                args.learner_checkpoint_dir,
            ),
            daemon=True,
            name="main-hist-sync",
        ))
        for t in worker_sync_threads:
            t.start()
        print(json.dumps({
            "event": "worker_checkpoint_sync_started",
            "machine_id": args.machine_id,
            "local_ckpt_dir": worker_local_ckpt_dir,
            "learner_host": args.learner_host,
        }), flush=True)

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
        while (
            is_remote_worker
            or (gradient_step if args.push_gradients else transitions) < int(args.total_transitions)
        ):
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

                # Poll for gradient files (push-gradients mode, learner host only)
                if (
                    args.push_gradients
                    and optimizer_for_gradients is not None
                    and online is not None
                    and _is_learner_host(args.machine_id)
                ):
                    now = time.time()
                    if now - last_gradient_poll_time >= gradient_poll_interval:
                        try:
                            gradient_results, gradient_poll_mtime = _poll_gradients_for_learner(
                                gradient_dir=args.learner_gradient_dir,
                                last_poll_mtime=gradient_poll_mtime,
                            )
                            
                            if gradient_results:
                                # Aggregate gradients from all actors
                                aggregated_grads: dict[str, torch.Tensor] = {}
                                total_actors = 0
                                file_paths_to_delete = []
                                machine_gradient_counts: dict[str, int] = {}
                                gradient_contributors: list[dict[str, Any]] = []

                                for item in gradient_results:
                                    if len(item) == 3:  # (actor_id, grads_dict, timestamp)
                                        actor_id, grads_dict, timestamp = item
                                        machine_id = "unknown"
                                    else:  # (file_path, machine_id, actor_id, grads_dict, timestamp)
                                        file_path, machine_id, actor_id, grads_dict, timestamp = item
                                        file_paths_to_delete.append(file_path)

                                    mid = str(machine_id)
                                    machine_gradient_counts[mid] = machine_gradient_counts.get(mid, 0) + 1
                                    gradient_contributors.append(
                                        {"machine_id": mid, "actor_id": int(actor_id)},
                                    )

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
                                
                                    # Adaptive learning rate: higher LR when more gradients
                                    adaptive_lr = learner_cfg.value_lr * min(2.0, 1.0 + (total_actors / 100.0))
                                    optimizer_for_gradients.param_groups[0]["lr"] = adaptive_lr
                                    
                                    # Check for NaN/Inf gradients before applying
                                    nan_grads = [name for name, g in aggregated_grads.items() if torch.isnan(g).any()]
                                    inf_grads = [name for name, g in aggregated_grads.items() if torch.isinf(g).any()]
                                    if nan_grads or inf_grads:
                                        print(json.dumps({
                                            "event": "skip_nan_grads",
                                            "nan_count": len(nan_grads),
                                            "inf_count": len(inf_grads),
                                            "first_few": (nan_grads + inf_grads)[:5],
                                        }), flush=True)
                                        # Delete ALL gradient files to prevent re-processing
                                        all_files_to_delete = set()
                                        for item in gradient_results:
                                            if len(item) == 5:
                                                all_files_to_delete.add(item[0])  # file_path
                                        if _is_learner_host(args.machine_id):
                                            for fp in all_files_to_delete:
                                                try:
                                                    Path(fp).unlink()
                                                    print(json.dumps({
                                                        "event": "gradient_deleted",
                                                        "file": str(fp),
                                                        "reason": "NaN/Inf grads",
                                                    }), flush=True)
                                                except Exception:
                                                    pass
                                        continue
                                    
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
                                    
                                    non_learner_in_batch = sum(
                                        n for mid, n in machine_gradient_counts.items()
                                        if mid.lower() != "learner"
                                    )
                                    print(json.dumps({
                                        "event": "gradients_applied",
                                        "step": gradient_step,
                                        "actors_contributed": total_actors,
                                        "grad_norm": grad_norm,
                                        "adaptive_lr": adaptive_lr,
                                        "transitions": transitions,
                                        "replay_size": len(replay),
                                        "machines_in_batch": machine_gradient_counts,
                                        "contributors": gradient_contributors,
                                        "non_learner_files_in_batch": non_learner_in_batch,
                                    }), flush=True)
                                    
                                    # Delete gradient files after training (learner only)
                                    if _is_learner_host(args.machine_id):
                                        for fp in file_paths_to_delete:
                                            try:
                                                Path(fp).unlink()
                                            except Exception as e:
                                                print(json.dumps({
                                                    "event": "gradient_delete_error",
                                                    "file": str(fp),
                                                    "error": str(e),
                                                }), flush=True)
                                    
                                    _maybe_save_learner_checkpoints(
                                        latest_path=latest_path,
                                        output_dir=output_dir,
                                        model=online,
                                        learner_cfg=learner_cfg,
                                        step=gradient_step,
                                        save_every=args.save_every_transitions,
                                        grad_norm=grad_norm,
                                    )
                        
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
                        "transitions": (gradient_step if args.push_gradients else transitions),
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
                        "transitions": (gradient_step if args.push_gradients else transitions),
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
        worker_sync_stop.set()
        if learner and online is not None:
            step = gradient_step if args.push_gradients else transitions
            _save_checkpoint(latest_path, online, learner_cfg, step)
        for p in procs:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()


if __name__ == "__main__":
    main()

