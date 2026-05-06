"""
Remote RHEA actor for multi-machine value training.

Runs on any machine with access to the shared Samba mount (Z:\\ or /mnt/awbw).
Reads the latest value net from checkpoints/value_rhea_latest.pt, runs RHEA
self-play, and writes COMPRESSED transition batches to a LOCAL staging dir,
then BACKGROUND-SYNCS them to the shared Samba mount.

This avoids the Samba slowness by:
1. Writing batches to fast local disk (not Samba)
2. Compressing batches with gzip (much faster over network)
3. Using a background thread to async sync to Samba

Usage (on any machine):
    python -m scripts.rhea_remote_actor \
        --shared-root Z:\\ \
        --machine-id workhorse2 \
        --checkpoint Z:/checkpoints/value_rhea_latest.pt \
        --local-staging-dir C:\\temp\\rhea_transitions \
        --transition-batch-size 500 \
        --n-envs 8
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.encoder import GRID_SIZE, N_SPATIAL_CHANNELS, N_SCALARS, encode_state
from rl.env import AWBWEnv
from rl.rhea import RheaConfig, RheaPlanner
from rl.rhea_fitness import RheaFitness
from rl.rhea_replay import RheaTransition
from rl.value_net import AWBWValueNet, load_value_checkpoint


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
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print("Cython rebuild complete.", flush=True)
        except Exception as exc:
            print(f"Cython rebuild error: {exc}", file=sys.stderr, flush=True)


# Run the check before importing rl.* (which may import the .pyd files)
_maybe_recompile_cython()


from rl.encoder import GRID_SIZE, N_SPATIAL_CHANNELS, N_SCALARS, encode_state
from rl.env import AWBWEnv
from rl.rhea import RheaConfig, RheaPlanner
from rl.rhea_fitness import RheaFitness
from rl.rhea_replay import RheaTransition
from rl.value_net import AWBWValueNet, load_value_checkpoint


def _parse_co_list(s: str) -> list[int]:
    return [int(x) for x in str(s).split(",") if str(x).strip()]


def _make_env(args: argparse.Namespace) -> AWBWEnv:
    co_p0 = _parse_co_list(args.co_p0)
    co_p1 = _parse_co_list(args.co_p1)

    pool_path = PROJECT_ROOT / "data" / "gl_map_pool.json"
    with open(pool_path, encoding="utf-8") as f:
        full_pool = json.load(f)

    map_pool = [m for m in full_pool if m.get("map_id") == args.map_id]
    if not map_pool:
        raise ValueError(f"Map ID {args.map_id} not found in pool")

    max_turns = args.max_days if args.max_days else None

    try:
        return AWBWEnv(
            map_pool=map_pool,
            co_p0=co_p0,
            co_p1=co_p1,
            max_turns=max_turns,
        )
    except TypeError:
        return AWBWEnv(
            map_pool=map_pool,
            co_p0=co_p0[0],
            co_p1=co_p1[0],
            max_turns=max_turns,
        )


def _setup_env_vars(args: argparse.Namespace) -> None:
    """Set environment variables for phi capture phase weighting and other features."""
    env_var_map = {
        "phi_capture_phase_weighting": ("AWBW_PHI_CAPTURE_PHASE_WEIGHTING", "1"),
        "phi_safe_neutral_opening_mult": ("AWBW_PHI_SAFE_NEUTRAL_OPENING_MULT", str(args.phi_safe_neutral_opening_mult) if args.phi_safe_neutral_opening_mult is not None else None),
        "phi_safe_neutral_early_mid_mult": ("AWBW_PHI_SAFE_NEUTRAL_EARLY_MID_MULT", str(args.phi_safe_neutral_early_mid_mult) if args.phi_safe_neutral_early_mid_mult is not None else None),
        "phi_safe_neutral_mid_mult": ("AWBW_PHI_SAFE_NEUTRAL_MID_MULT", str(args.phi_safe_neutral_mid_mult) if args.phi_safe_neutral_mid_mult is not None else None),
        "phi_safe_neutral_late_mult": ("AWBW_PHI_SAFE_NEUTRAL_LATE_MULT", str(args.phi_safe_neutral_late_mult) if args.phi_safe_neutral_late_mult is not None else None),
        "phi_safe_neutral_endgame_mult": ("AWBW_PHI_SAFE_NEUTRAL_ENDGAME_MULT", str(args.phi_safe_neutral_endgame_mult) if args.phi_safe_neutral_endgame_mult is not None else None),
        "phi_contested_neutral_opening_mult": ("AWBW_PHI_CONTESTED_NEUTRAL_OPENING_MULT", str(args.phi_contested_neutral_opening_mult) if args.phi_contested_neutral_opening_mult is not None else None),
        "phi_contested_neutral_mid_mult": ("AWBW_PHI_CONTESTED_NEUTRAL_MID_MULT", str(args.phi_contested_neutral_mid_mult) if args.phi_contested_neutral_mid_mult is not None else None),
        "phi_contested_neutral_late_mult": ("AWBW_PHI_CONTESTED_NEUTRAL_LATE_MULT", str(args.phi_contested_neutral_late_mult) if args.phi_contested_neutral_late_mult is not None else None),
        "phi_capture_opening_end_day": ("AWBW_PHI_CAPTURE_OPENING_END_DAY", str(args.phi_capture_opening_end_day) if args.phi_capture_opening_end_day is not None else None),
        "phi_capture_early_mid_end_day": ("AWBW_PHI_CAPTURE_EARLY_MID_END_DAY", str(args.phi_capture_early_mid_end_day) if args.phi_capture_early_mid_end_day is not None else None),
        "phi_capture_mid_end_day": ("AWBW_PHI_CAPTURE_MID_END_DAY", str(args.phi_capture_mid_end_day) if args.phi_capture_mid_end_day is not None else None),
        "phi_capture_late_end_day": ("AWBW_PHI_CAPTURE_LATE_END_DAY", str(args.phi_capture_late_end_day) if args.phi_capture_late_end_day is not None else None),
    }

    for attr, (env_name, value) in env_var_map.items():
        val = getattr(args, attr, None)
        if val is not None and val is not False:
            os.environ[env_name] = value if val is True else str(val)
        else:
            os.environ.pop(env_name, None)

    if args.dual_gradient_self_play:
        os.environ["AWBW_DUAL_GRADIENT_SELF_PLAY"] = "1"
    if args.pairwise_zero_sum_reward:
        os.environ["AWBW_PAIRWISE_ZERO_SUM_REWARD"] = "1"


def _transition_to_payload(t: RheaTransition) -> dict[str, Any]:
    """Convert a RheaTransition to a JSON-serializable dict."""
    return {
        "spatial_before": t.spatial_before.tolist() if isinstance(t.spatial_before, np.ndarray) else t.spatial_before,
        "scalars_before": t.scalars_before.tolist() if isinstance(t.scalars_before, np.ndarray) else t.scalars_before,
        "reward_turn": float(t.reward_turn),
        "spatial_after": t.spatial_after.tolist() if isinstance(t.spatial_after, np.ndarray) else t.spatial_after,
        "scalars_after": t.scalars_after.tolist() if isinstance(t.scalars_after, np.ndarray) else t.scalars_after,
        "done": bool(t.done),
        "winner": t.winner,
        "acting_seat": int(t.acting_seat),
        "day": int(t.day),
        "phi_delta": float(t.phi_delta),
        "value_after_at_search_time": float(t.value_after_at_search_time),
        "search_score": float(t.search_score),
    }


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


def _encode_into(
    state,
    observer_seat: int,
    spatial_buf: np.ndarray,
    scalars_buf: np.ndarray,
) -> None:
    """Encode state into pre-allocated buffers."""
    encode_state(
        state,
        observer=int(observer_seat),
        belief=None,
        out_spatial=spatial_buf,
        out_scalars=scalars_buf,
    )


# ---------------------------------------------------------------------------
# Background sync thread: syncs local staging dir to Samba
# ---------------------------------------------------------------------------
class StagingSyncThread(threading.Thread):
    """Background thread that syncs local transition files to Samba.

    Uses robocopy on Windows (mirror mode, retries on failure) or
    rsync on Linux.  Compressed .jsonl.gz files are synced.
    """

    def __init__(
        self,
        local_dir: Path,
        remote_dir: Path,
        machine_id: str,
        poll_interval: float = 30.0,
    ) -> None:
        super().__init__(name="StagingSync", daemon=True)
        self.local_dir = local_dir
        self.remote_dir = remote_dir
        self.machine_id = machine_id
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self.files_synced = 0
        self.errors = 0

    def run(self) -> None:
        """Sync loop: mirror local -> remote using robocopy/rsync."""
        while not self._stop_event.is_set():
            try:
                self._sync_once()
            except Exception as exc:
                self.errors += 1
                print(json.dumps({
                    "event": "staging_sync_error",
                    "machine_id": self.machine_id,
                    "error": str(exc),
                }), flush=True)

            # Wait for next poll or stop
            self._stop_event.wait(timeout=self.poll_interval)

    def _sync_once(self) -> None:
        """One sync pass: mirror local staging to remote Samba dir."""
        if not self.local_dir.exists():
            return

        # Ensure remote dir exists
        self.remote_dir.mkdir(parents=True, exist_ok=True)

        if sys.platform.startswith("win"):
            self._sync_robocopy()
        else:
            self._sync_rsync()

    def _sync_robocopy(self) -> None:
        """Use robocopy to mirror local -> remote (Windows)."""
        # robocopy arguments:
        #   /MIR - mirror (copy new, delete removed from dest)
        #   /R:3 - retry 3 times
        #   /W:5 - wait 5 sec between retries
        #   /NJH - no job header
        #   /NJS - no job summary
        #   /NP  - no progress display
        local_str = str(self.local_dir)
        remote_str = str(self.remote_dir)

        try:
            result = subprocess.run(
                [
                    "robocopy",
                    local_str,
                    remote_str,
                    "*.jsonl.gz",
                    "/MIR",
                    "/R:3",
                    "/W:5",
                    "/NJH",
                    "/NJS",
                    "/NP",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            # robocopy returns 0-7 as success codes (0=no files copied, 1=files copied, etc.)
            if result.returncode > 7:
                print(json.dumps({
                    "event": "robocopy_failed",
                    "machine_id": self.machine_id,
                    "returncode": result.returncode,
                    "stderr": result.stderr[-500:] if result.stderr else "",
                }), flush=True)
            else:
                # Count .gz files in local dir as a rough metric
                gz_files = list(self.local_dir.glob("*.jsonl.gz"))
                if gz_files:
                    self.files_synced = len(gz_files)
                    if False:  # Set to True for verbose sync logging
                        print(json.dumps({
                            "event": "staging_sync_complete",
                            "machine_id": self.machine_id,
                            "files_synced": self.files_synced,
                        }), flush=True)
        except Exception as exc:
            self.errors += 1
            print(json.dumps({
                "event": "robocopy_exception",
                "machine_id": self.machine_id,
                "error": str(exc),
            }), flush=True)

    def _sync_rsync(self) -> None:
        """Use rsync to sync local -> remote (Linux)."""
        local_str = str(self.local_dir) + "/"
        remote_str = str(self.remote_dir) + "/"

        try:
            result = subprocess.run(
                [
                    "rsync",
                    "-av",
                    "--include=*.jsonl.gz",
                    "--exclude=*",
                    "--remove-source-files",
                    local_str,
                    remote_str,
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                print(json.dumps({
                    "event": "rsync_failed",
                    "machine_id": self.machine_id,
                    "returncode": result.returncode,
                    "stderr": result.stderr[-500:] if result.stderr else "",
                }), flush=True)
        except Exception as exc:
            self.errors += 1
            print(json.dumps({
                "event": "rsync_exception",
                "machine_id": self.machine_id,
                "error": str(exc),
            }), flush=True)

    def stop(self) -> None:
        """Signal the thread to stop and wait for final sync."""
        self._stop_event.set()
        # Do one final sync
        try:
            self._sync_once()
        except Exception:
            pass


def write_compressed_batch(
    transitions: list[dict[str, Any]],
    staging_dir: Path,
    machine_id: str,
    batch_num: int,
) -> Path:
    """Write a batch of transitions to a GZIP-compressed JSONL file in staging dir.

    Returns the path to the written file (local staging dir).
    The background StagingSyncThread will sync it to Samba.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    stem = f"{machine_id}_batch_{batch_num:06d}_{timestamp}"
    gz_path = staging_dir / f"{stem}.jsonl.gz"

    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        for t in transitions:
            f.write(json.dumps(t, separators=(",", ":")) + "\n")

    return gz_path


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Remote RHEA actor for multi-machine training")

    # Shared filesystem
    ap.add_argument("--shared-root", type=str, required=True,
                      help="Path to shared Samba mount (Z:\\ on Windows, /mnt/awbw on Linux)")
    ap.add_argument("--machine-id", type=str, required=True,
                      help="Unique machine ID (e.g. workhorse2, gpu-box1)")
    ap.add_argument("--checkpoint", type=str, default=None,
                      help="Path to initial checkpoint (default: <shared-root>/checkpoints/value_rhea_latest.pt)")

    # Env
    ap.add_argument("--map-id", type=int, default=171596)
    ap.add_argument("--co-p0", type=str, default="14,8,28,7")
    ap.add_argument("--co-p1", type=str, default="14,8,28,7")
    ap.add_argument("--max-days", type=int, default=30)

    # Actor device
    ap.add_argument("--device", type=str, default="cuda", help="Value net device")
    ap.add_argument("--actor-torch-threads", type=int, default=1)

    # RHEA search
    ap.add_argument("--rhea-autotune", action="store_true")
    ap.add_argument("--rhea-population", type=int, default=32)
    ap.add_argument("--rhea-generations", type=int, default=5)
    ap.add_argument("--rhea-elite", type=int, default=4)
    ap.add_argument("--rhea-mutation-rate", type=float, default=0.20)
    ap.add_argument("--rhea-top-k-per-state", type=int, default=24)
    ap.add_argument("--rhea-max-actions-per-turn", type=int, default=128)
    ap.add_argument("--reward-weight", type=float, default=0.90)
    ap.add_argument("--value-weight", type=float, default=0.10)

    # Transition output (optimized for Samba)
    ap.add_argument("--transition-batch-size", type=int, default=500,
                      help="Number of transitions per compressed batch (default: 500, larger = fewer Samba ops)")
    ap.add_argument("--transition-dir", type=str, default=None,
                      help="FINAL transition directory on shared Samba (default: <shared-root>/fleet/<machine-id>/transitions/)")
    ap.add_argument("--local-staging-dir", type=str, default=None,
                      help="LOCAL disk path for staging transitions before Samba sync (default: system temp dir)")
    ap.add_argument("--no-compress", action="store_true",
                      help="Disable gzip compression of transition batches")
    ap.add_argument("--sync-interval", type=float, default=30.0,
                      help="Seconds between background syncs to Samba (default: 30)")

    # Weight refresh
    ap.add_argument("--actor-refresh-seconds", type=float, default=120.0)

    # Dual-gradient self-play
    ap.add_argument("--dual-gradient-self-play", action="store_true")
    ap.add_argument("--dual-gradient-hist-prob", type=float, default=0.0)
    ap.add_argument("--pairwise-zero-sum-reward", action="store_true")

    # COP disable (forces SCOP learning)
    ap.add_argument("--cop-disable-per-seat-p", type=float, default=0.10,
                      help="Probability (0-1) to disable COP for each seat at game start (default 0.10 = 10%%)")

    # RHEA tactical beam
    ap.add_argument("--rhea-use-tactical-beam", action="store_true")
    ap.add_argument("--rhea-tactical-beam-max-width", type=int, default=48)
    ap.add_argument("--rhea-tactical-beam-max-depth", type=int, default=14)
    ap.add_argument("--rhea-tactical-beam-max-expand", type=int, default=24)

    # Phi capture phase weighting
    ap.add_argument("--phi-capture-phase-weighting", action="store_true")
    ap.add_argument("--phi-safe-neutral-opening-mult", type=float, default=None)
    ap.add_argument("--phi-safe-neutral-early-mid-mult", type=float, default=None)
    ap.add_argument("--phi-safe-neutral-mid-mult", type=float, default=None)
    ap.add_argument("--phi-safe-neutral-late-mult", type=float, default=None)
    ap.add_argument("--phi-safe-neutral-endgame-mult", type=float, default=None)
    ap.add_argument("--phi-contested-neutral-opening-mult", type=float, default=None)
    ap.add_argument("--phi-contested-neutral-mid-mult", type=float, default=None)
    ap.add_argument("--phi-contested-neutral-late-mult", type=float, default=None)
    ap.add_argument("--phi-capture-opening-end-day", type=int, default=None)
    ap.add_argument("--phi-capture-early-mid-end-day", type=int, default=None)
    ap.add_argument("--phi-capture-mid-end-day", type=int, default=None)
    ap.add_argument("--phi-capture-late-end-day", type=int, default=None)

    # Run control
    ap.add_argument("--n-envs", type=int, default=1, help="Number of actor processes (default 1 for remote)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-transitions", type=int, default=1_000_000)
    ap.add_argument("--verbose", action="store_true")

    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    _setup_env_vars(args)

    # Resolve paths
    shared_root = Path(args.shared_root).resolve()
    machine_id = args.machine_id

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = shared_root / "checkpoints" / "value_rhea_latest.pt"

    # Transition output directory on Samba (final destination)
    if args.transition_dir:
        transition_dir = Path(args.transition_dir)
    else:
        transition_dir = shared_root / "fleet" / machine_id / "transitions"

    # Local staging directory (fast local disk)
    if args.local_staging_dir:
        staging_dir = Path(args.local_staging_dir)
    else:
        # Use system temp dir to avoid Samba for writes
        staging_dir = Path(tempfile.gettempdir()) / "rhea_actor" / machine_id

    staging_dir.mkdir(parents=True, exist_ok=True)

    # Start background sync thread (syncs local staging -> Samba)
    sync_thread = StagingSyncThread(
        local_dir=staging_dir,
        remote_dir=transition_dir,
        machine_id=machine_id,
        poll_interval=args.sync_interval,
    )
    sync_thread.start()

    # Seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Log startup
    print(json.dumps({
        "event": "remote_actor_start",
        "machine_id": machine_id,
        "shared_root": str(shared_root),
        "checkpoint": str(checkpoint_path),
        "transition_dir": str(transition_dir),
        "staging_dir": str(staging_dir),
        "compress": not args.no_compress,
        "device": args.device,
    }), flush=True)

    # Create environment
    env = _make_env(args)

    # Load initial value model
    actor_device = args.device
    value_model = None
    try:
        value_model = load_value_checkpoint(str(checkpoint_path), device=actor_device)
        print(json.dumps({
            "event": "checkpoint_loaded",
            "checkpoint": str(checkpoint_path),
        }), flush=True)
    except Exception as e:
        print(json.dumps({
            "event": "checkpoint_load_failed",
            "checkpoint": str(checkpoint_path),
            "error": str(e),
            "message": "Creating fresh model as fallback",
        }), flush=True)
        value_model = AWBWValueNet().to(actor_device)

    # Setup fitness
    fitness = RheaFitness(
        env_template=env,
        value_model=value_model,
        device=actor_device,
        reward_weight=args.reward_weight,
        value_weight=args.value_weight,
    )

    # Pre-allocate encode buffers
    spatial_buf = np.empty((GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS), dtype=np.float32)
    scalars_buf = np.empty((N_SCALARS,), dtype=np.float32)

    # Create RHEA planner
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
            tactical_beam_max_width=args.rhea_tactical_beam_max_width,
            tactical_beam_max_depth=args.rhea_tactical_beam_max_depth,
            tactical_beam_max_expand=args.rhea_tactical_beam_max_expand,
        ),
        dynamic_budget=args.rhea_autotune,
        complexity_metrics=None,
    )

    # Main loop
    transitions_batch: list[dict[str, Any]] = []
    batch_num = 0
    transitions_written = 0
    games_done = 0
    last_refresh = 0.0
    latest_path = shared_root / "checkpoints" / "value_rhea_latest.pt"

    print(json.dumps({
        "event": "remote_actor_ready",
        "machine_id": machine_id,
        "staging_dir": str(staging_dir),
        "transition_dir": str(transition_dir),
    }), flush=True)

    try:
        while transitions_written < args.max_transitions:
            # Periodically refresh value model from shared checkpoint
            now = time.time()
            if args.actor_refresh_seconds > 0 and now - last_refresh >= args.actor_refresh_seconds:
                try:
                    # Use a copy to avoid corrupting the live model on partial read
                    tmp_path = shared_root / "checkpoints" / ".value_rhea_latest.pt.tmp"
                    if not tmp_path.exists():
                        tmp_path = latest_path
                    ckpt = torch.load(str(tmp_path), map_location=actor_device)
                    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
                    value_model.load_state_dict(sd, strict=False)
                    value_model.to(actor_device)
                    value_model.eval()
                    last_refresh = now
                    fitness.set_value_model(value_model)
                    if args.verbose:
                        print(json.dumps({
                            "event": "model_refreshed",
                            "machine_id": machine_id,
                        }), flush=True)
                except Exception as e:
                    if args.verbose:
                        print(json.dumps({
                            "event": "model_refresh_failed",
                            "machine_id": machine_id,
                            "error": str(e),
                        }), flush=True)

            # Run one game
            env.reset()
            game_transitions = 0

            # 10% chance to disable COP for each seat at game start (forces SCOP learning)
            cop_disable_p = getattr(args, "cop_disable_per_seat_p", 0.10)
            for seat in (0, 1):
                _maybe_disable_cop_for_seat(env.state.co_states[seat], cop_disable_p)

            while env.state is not None and env.state.winner is None:
                state = env.state
                acting = int(state.active_player)
                day = int(getattr(state, "turn", getattr(state, "day", 0)))

                # Encode before state
                _encode_into(state, acting, spatial_buf, scalars_buf)
                before_spatial = spatial_buf.copy()
                before_scalars = scalars_buf.copy()
                phi_before = fitness.phi(state, acting)

                # Compute complexity metrics for dynamic budgeting
                complexity_metrics = None
                if args.rhea_autotune:
                    try:
                        complexity_metrics = RheaPlanner.compute_complexity_metrics(state, acting)
                    except Exception:
                        pass

                planner.dynamic_budget = args.rhea_autotune
                planner.complexity_metrics = complexity_metrics

                # Plan and execute full turn
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

                # Encode after state
                _encode_into(after, acting, spatial_buf, scalars_buf)
                after_spatial = spatial_buf.copy()
                after_scalars = scalars_buf.copy()
                phi_after = fitness.phi(after, acting)
                reward_turn = float(phi_after - phi_before)
                done = bool(after.winner is not None)

                # Create transition
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

                transitions_batch.append(_transition_to_payload(t))
                game_transitions += 1

                # Write batch if ready
                if len(transitions_batch) >= args.transition_batch_size:
                    try:
                        if args.no_compress:
                            # Write uncompressed (slower over Samba)
                            write_compressed_batch(transitions_batch, staging_dir, machine_id, batch_num)
                        else:
                            # Write compressed (much faster over Samba)
                            write_compressed_batch(transitions_batch, staging_dir, machine_id, batch_num)
                        transitions_written += len(transitions_batch)
                        batch_num += 1
                        transitions_batch.clear()

                        if args.verbose:
                            print(json.dumps({
                                "event": "batch_written",
                                "machine_id": machine_id,
                                "batch_num": batch_num,
                                "total_written": transitions_written,
                            }), flush=True)
                    except Exception as e:
                        print(json.dumps({
                            "event": "batch_write_failed",
                            "machine_id": machine_id,
                            "error": str(e),
                        }), flush=True)

                if day > args.max_days + 1:
                    break

            games_done += 1
            if args.verbose:
                print(json.dumps({
                    "event": "game_done",
                    "machine_id": machine_id,
                    "game_num": games_done,
                    "transitions": game_transitions,
                }), flush=True)

    except KeyboardInterrupt:
        print(json.dumps({
            "event": "remote_actor_stopped",
            "machine_id": machine_id,
            "transitions_written": transitions_written,
            "games_done": games_done,
        }), flush=True)
    finally:
        # Stop the sync thread (it will do a final sync)
        sync_thread.stop()
        sync_thread.join(timeout=60.0)

        # Write any remaining transitions
        if transitions_batch:
            try:
                write_compressed_batch(transitions_batch, staging_dir, machine_id, batch_num)
                transitions_written += len(transitions_batch)
            except Exception as e:
                print(json.dumps({
                    "event": "final_batch_write_failed",
                    "error": str(e),
                }), flush=True)

        print(json.dumps({
            "event": "remote_actor_shutdown",
            "machine_id": machine_id,
            "total_transitions_written": transitions_written,
            "total_games_done": games_done,
            "sync_files_synced": sync_thread.files_synced,
            "sync_errors": sync_thread.errors,
        }), flush=True)


if __name__ == "__main__":
    main()
