"""
Remote RHEA actor for multi-machine value training.

Runs on any machine with access to the shared Samba mount (Z:\\ or /mnt/awbw).
Reads the latest value net from checkpoints/value_rhea_latest.pt, runs RHEA
self-play, and writes transition batches to Z:/fleet/<machine_id>/transitions/
for the learner (running on workhorse1) to ingest.

Usage (on any auxiliary machine):
    python -m scripts.rhea_remote_actor \
        --shared-root Z:\\ \
        --machine-id workhorse2 \
        --checkpoint Z:/checkpoints/value_rhea_latest.pt \
        --map-id 171596 \
        --co-p0 14,8,28,7 --co-p1 14,8,28,7 \
        --max-days 30 \
        --rhea-autotune \
        --reward-weight 0.8 --value-weight 0.2 \
        --transition-batch-size 100 \
        --n-envs 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
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
_maybe_recompile_cython()

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

    # Transition output
    ap.add_argument("--transition-batch-size", type=int, default=100,
                      help="Number of transitions per JSONL file written to shared disk")
    ap.add_argument("--transition-dir", type=str, default=None,
                      help="Override transition output directory (default: <shared-root>/fleet/<machine-id>/transitions/)")

    # Weight refresh
    ap.add_argument("--actor-refresh-seconds", type=float, default=120.0)

    # Dual-gradient self-play
    ap.add_argument("--dual-gradient-self-play", action="store_true")
    ap.add_argument("--dual-gradient-hist-prob", type=float, default=0.0)
    ap.add_argument("--pairwise-zero-sum-reward", action="store_true")

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


def write_transitions_batch(
    transitions: list[dict[str, Any]],
    transition_dir: Path,
    machine_id: str,
    batch_num: int,
) -> None:
    """Write a batch of transitions to a JSONL file atomically."""
    transition_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    stem = f"{machine_id}_batch_{batch_num:06d}_{timestamp}"
    tmp_path = transition_dir / f".{stem}.jsonl.tmp"
    final_path = transition_dir / f"{stem}.jsonl"

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for t in transitions:
                f.write(json.dumps(t, separators=(",", ":")) + "\n")
        # Atomic rename (works on both Windows SMB and Linux)
        os.replace(str(tmp_path), str(final_path))
        return final_path
    except Exception as exc:
        # Clean up tmp file on failure
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise exc


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

    # Transition output directory
    if args.transition_dir:
        transition_dir = Path(args.transition_dir)
    else:
        transition_dir = shared_root / "fleet" / machine_id / "transitions"

    transition_dir.mkdir(parents=True, exist_ok=True)

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
            tactial_beam_max_width=args.rhea_tactical_beam_max_width,
            tactial_beam_max_depth=args.rhea_tactical_beam_max_depth,
            tactial_beam_max_expand=args.rhea_tactical_beam_max_expand,
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
                        write_transitions_batch(transitions_batch, transition_dir, machine_id, batch_num)
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
        # Write any remaining transitions
        if transitions_batch:
            try:
                write_transitions_batch(transitions_batch, transition_dir, machine_id, batch_num)
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
        }), flush=True)


if __name__ == "__main__":
    main()
