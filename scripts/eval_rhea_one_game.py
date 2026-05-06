from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

from rl.env import AWBWEnv, POOL_PATH
from rl.rhea import RheaConfig, RheaPlanner
from rl.rhea_fitness import RheaFitness
from rl.value_net import load_value_checkpoint
from tools.export_awbw_replay import write_awbw_replay


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
    
    # Enable phi shaping
    os.environ["AWBW_REWARD_SHAPING"] = "phi"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--map-id", type=int, default=171596)
    parser.add_argument("--co-p0", type=int, default=14)
    parser.add_argument("--co-p1", type=int, default=14)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--value-weight", type=float, default=0.10)
    parser.add_argument("--reward-weight", type=float, default=0.90)
    parser.add_argument("--max-days", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="replays")
    parser.add_argument("--game-id", type=int, default=None)
    parser.add_argument("--open-viewer", action="store_true", default=True)
    parser.add_argument("--rhea-autotune", action="store_true", help="Use dynamic budgeting for RHEA")
    
    # Tactical beam.
    parser.add_argument("--rhea-use-tactical-beam", action="store_true")
    parser.add_argument("--rhea-tactical-beam-max-width", type=int, default=48)
    parser.add_argument("--rhea-tactical-beam-max-depth", type=int, default=14)
    parser.add_argument("--rhea-tactical-beam-max-expand", type=int, default=24)
    
    # Phi capture phase weighting
    parser.add_argument("--phi-capture-phase-weighting", action="store_true")
    parser.add_argument("--phi-safe-neutral-opening-mult", type=float, default=None)
    parser.add_argument("--phi-safe-neutral-early-mid-mult", type=float, default=None)
    parser.add_argument("--phi-safe-neutral-mid-mult", type=float, default=None)
    parser.add_argument("--phi-safe-neutral-late-mult", type=float, default=None)
    parser.add_argument("--phi-safe-neutral-endgame-mult", type=float, default=None)
    parser.add_argument("--phi-contested-neutral-opening-mult", type=float, default=None)
    parser.add_argument("--phi-contested-neutral-mid-mult", type=float, default=None)
    parser.add_argument("--phi-contested-neutral-late-mult", type=float, default=None)
    parser.add_argument("--phi-capture-opening-end-day", type=int, default=None)
    parser.add_argument("--phi-capture-early-mid-end-day", type=int, default=None)
    parser.add_argument("--phi-capture-mid-end-day", type=int, default=None)
    parser.add_argument("--phi-capture-late-end-day", type=int, default=None)
    
    # Dual-gradient self-play
    parser.add_argument("--dual-gradient-self-play", action="store_true")
    parser.add_argument("--dual-gradient-hist-prob", type=float, default=0.0)
    
    args = parser.parse_args()
    
    # Setup environment variables for phi and dual-gradient features
    _setup_env_vars(args)
    
    # Enable build punishment for base skipping
    os.environ["AWBW_BUILD_PUNISHMENT"] = "1"

    # Load map pool and filter to specific map ID
    with open(POOL_PATH) as f:
        full_pool = json.load(f)
    
    # Filter to only include the specified map ID
    filtered_pool = [m for m in full_pool if m.get("map_id") == args.map_id]
    if not filtered_pool:
        raise ValueError(f"Map ID {args.map_id} not found in map pool")
    
    # Constructor signatures may differ in local branches. If this fails, pass
    # the same env construction kwargs used by scripts/start_solo_training.py.
    env = AWBWEnv(
        map_pool=filtered_pool,
        co_p0=args.co_p0,
        co_p1=args.co_p1,
        max_turns=args.max_days,
    )
    env.reset()

    # Collect snapshots for replay export
    snapshots = [copy.deepcopy(env.state)]

    # Load value model using unified checkpoint loader
    value_model = load_value_checkpoint(args.checkpoint, device=args.device)

    fitness = RheaFitness(
        env_template=env,
        value_model=value_model,
        device=args.device,
        reward_weight=args.reward_weight,
        value_weight=args.value_weight,
    )
    # Create config with smaller parameters since early game has few actions
    config = RheaConfig(
        population=args.population,
        generations=args.generations,
        reward_weight=args.reward_weight,
        value_weight=args.value_weight,
        seed=random.randrange(1 << 30),
        use_tactical_beam=args.rhea_use_tactical_beam,
        tactial_beam_max_width=args.rhea_tactical_beam_max_width,
        tactial_beam_max_depth=args.rhea_tactical_beam_max_depth,
        tactial_beam_max_expand=args.rhea_tactical_beam_max_expand,
    )
    # Override parameters for early game
    config.top_k_per_state = 10  # With wrap-around fix, can be higher
    config.max_actions_per_turn = 10
    
    # Calculate complexity metrics for dynamic budgeting if enabled
    complexity_metrics = None
    if args.rhea_autotune and env.state is not None:
        # For early game, use conservative defaults
        complexity_metrics = (0, 0, 0, 0)
    
    planner = RheaPlanner(
        fitness, 
        config, 
        dynamic_budget=args.rhea_autotune,
        complexity_metrics=complexity_metrics
    )

    while env.state is not None and env.state.winner is None:
        state = env.state
        active = int(state.active_player)
        result = planner.choose_full_turn(state)

        print(
            f"day={getattr(state, 'turn', '?')} active={active} "
            f"score={result.score:.4f} "
            f"phi={result.breakdown.phi_delta:.4f} "
            f"v={result.breakdown.value:.4f} "
            f"illegal={result.illegal_genes} "
            f"actions={len(result.actions)}"
        )

        for action in result.actions:
            if env.state is None or env.state.winner is not None:
                break
            if int(env.state.active_player) != active:
                break
            env.state.step(action)
            # Add snapshot on player turn change
            if env.state is not None and env.state.active_player != active:
                snapshots.append(copy.deepcopy(env.state))

    if env.state is not None:
        # Add final snapshot
        snapshots.append(copy.deepcopy(env.state))
        print("winner:", env.state.winner)
    else:
        print("winner:", None)


# Export replay
    if snapshots:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        game_id = args.game_id or int(time.time()) % 999000 + 1000
        output_path = output_dir / f"{game_id}.zip"
        
        print(f"Exporting replay with {len(snapshots)} snapshots to {output_path}")
        
        try:
            write_awbw_replay(
                snapshots=snapshots,
                output_path=output_path,
                game_id=game_id,
                game_name=f"Rhea eval - map {args.map_id} - CO {args.co_p0}/{args.co_p1}",
                start_date=time.strftime("%Y-%m-%d %H:%M:%S"),
                full_trace=env.state.full_trace if env.state else None,
                luck_seed=None,
            )
            print(f"Replay exported successfully: {output_path}")
            
            # Launch viewer if requested
            if args.open_viewer:
                from rl.paths import resolve_awbw_replay_player_exe
                import subprocess
                import sys
                
                exe = resolve_awbw_replay_player_exe(Path(__file__).parent.parent)
                if exe is not None and exe.is_file():
                    print(f"Launching replay viewer: {exe}")
                    try:
                        if sys.platform == "win32":
                            subprocess.Popen(
                                ["cmd.exe", "/c", "start", "", str(exe), str(output_path.resolve())],
                                cwd=str(exe.parent),
                                close_fds=False,
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        else:
                            subprocess.Popen(
                                [str(exe), str(output_path.resolve())],
                                cwd=str(exe.parent),
                                close_fds=sys.platform != "win32",
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        print("Viewer launched successfully")
                    except Exception as e:
                        print(f"Failed to launch viewer: {e}")
                else:
                    print("Replay viewer not found, opening folder instead")
                    try:
                        if sys.platform == "win32":
                            subprocess.Popen(["explorer.exe", str(output_dir.resolve())])
                        elif sys.platform == "darwin":
                            subprocess.run(["open", str(output_dir)], check=False)
                        else:
                            subprocess.run(["xdg-open", str(output_dir)], check=False)
                    except Exception as e:
                        print(f"Failed to open folder: {e}")
        except Exception as e:
            print(f"Failed to export replay: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()