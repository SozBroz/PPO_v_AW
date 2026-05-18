from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

from rl.env import AWBWEnv, POOL_PATH, _flat_to_action, _BUILD_OFFSET, _ENC_W
from engine.unit import UnitType as _eng_unit_type_fix
from engine.action import ActionStage
from rl.rhea import RheaConfig, RheaPlanner, replay_rhea_actions, _salvage_ender
from engine.action import Action, ActionType, get_legal_actions
from rl.rhea_fitness import RheaFitness
from rl.value_net import load_value_checkpoint
from tools.export_awbw_replay import write_awbw_replay

_N_UNIT_TYPES_BOOK = len(_eng_unit_type_fix)


def _advance_to_select_for_book(state: object) -> None:
    """Step through MOVE/ACTION until SELECT so opening-book flats can decode."""

    guard = 0
    while (
        state is not None
        and getattr(state, "winner", None) is None
        and state.action_stage != ActionStage.SELECT
        and guard < 500
    ):
        legal = get_legal_actions(state)
        if not legal:
            break
        try:
            state.step(legal[0], oracle_mode=True)
        except Exception:
            break
        guard += 1


def _drain_opening_book(env: AWBWEnv, mgr: object) -> int:
    """Replay both seats' book lines in calendar order until exhausted or stale."""

    n_applied = 0
    stale_loops = 0
    while env.state is not None and env.state.winner is None and stale_loops < 16:
        c0 = mgr.controllers.get(0)
        c1 = mgr.controllers.get(1)
        b0_ex = (
            c0 is None
            or not getattr(c0, "episode_enabled", False)
            or c0._book is None
            or c0._cursor >= len(c0._book.action_indices)
        )
        b1_ex = (
            c1 is None
            or not getattr(c1, "episode_enabled", False)
            or c1._book is None
            or c1._cursor >= len(c1._book.action_indices)
        )
        if b0_ex and b1_ex:
            break

        active = int(env.state.active_player)
        ctl = mgr.controllers.get(active)

        if (
            ctl is None
            or not getattr(ctl, "episode_enabled", False)
            or ctl._book is None
            or ctl._cursor >= len(ctl._book.action_indices)
        ):
            stale_loops += 1
            continue

        stale_loops = 0
        cursor = ctl._cursor
        flat_idx = int(ctl._book.action_indices[cursor])
        st = env.state
        _advance_to_select_for_book(st)
        if st.winner is not None:
            break

        legal = get_legal_actions(st)
        if flat_idx == 0:
            action = Action(ActionType.END_TURN)
        else:
            action = _flat_to_action(flat_idx, st, legal=legal)

        # BUILD flat when decoder misses (wrong stage): SELECT factory tile, then BUILD.
        if action is None and flat_idx >= _BUILD_OFFSET:
            flat_off = flat_idx - _BUILD_OFFSET
            br = flat_off // (_ENC_W * _N_UNIT_TYPES_BOOK)
            rem = flat_off % (_ENC_W * _N_UNIT_TYPES_BOOK)
            bc = rem // _N_UNIT_TYPES_BOOK
            sel_flat = 3 + br * _ENC_W + bc
            sel_a = _flat_to_action(sel_flat, st)
            if sel_a is not None:
                try:
                    st.step(sel_a, oracle_mode=True)
                except Exception:
                    pass
            _advance_to_select_for_book(st)
            legal_b = get_legal_actions(st)
            action = _flat_to_action(flat_idx, st, legal=legal_b)
            if action is None:
                ut_idx = int(flat_idx - _BUILD_OFFSET) % _N_UNIT_TYPES_BOOK
                try:
                    ut = _eng_unit_type_fix(ut_idx)
                except ValueError:
                    ut = _eng_unit_type_fix.INFANTRY
                action = Action(ActionType.BUILD, move_pos=(br, bc), unit_type=ut)

        if action is None:
            print(
                f"  [BOOK] seat={active} flat={flat_idx} cursor={cursor} "
                f"not decodable; stage={st.action_stage!r}",
                flush=True,
            )
            break
        try:
            st.step(action, oracle_mode=True)
            ctl._cursor = cursor + 1
            n_applied += 1
            # Post-build micromanagement stays in MOVE/ACTION; next iter advances.
            _advance_to_select_for_book(st)
        except Exception as e:
            print(
                f"  [BOOK] seat={active} flat={flat_idx} FAILED: {e}",
                flush=True,
            )
            break

    return n_applied


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
    parser.add_argument("--co-p0", type=str, default="14")
    parser.add_argument("--co-p1", type=str, default="14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--value-weight", type=float, default=0.10)
    parser.add_argument("--reward-weight", type=float, default=0.90)
    parser.add_argument("--max-days", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="replays")
    parser.add_argument("--game-id", type=int, default=None)
    parser.add_argument("--open-viewer", action="store_true", default=True)
    parser.add_argument("--rhea-monolithic-buy", action="store_true",
                        help="Use legacy single-phase RHEA (moves+builds in one genome)")
    parser.add_argument(
        "--rhea-autotune",
        action="store_true",
        help="Scale move-phase RHEA pop/gen from per-turn game-state complexity",
    )

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
    parser.add_argument("--opening-book-path", type=str, default=None)
    parser.add_argument("--opening-book-prob", type=float, default=1.0)
    
    args = parser.parse_args()
    
    # Setup environment variables for phi and dual-gradient features
    _setup_env_vars(args)
    
    # Enable build punishment for base skipping but segment move/buy phases
    os.environ["AWBW_BUILD_PUNISHMENT"] = "1"
    os.environ["AWBW_SEGMENT_PHASES"] = "1"  # Separate move and buy phases
    
    # Set punishment weights for base skipping - much stronger penalty for debugging
    os.environ["AWBW_BASE_SKIP_PENALTY"] = "-1.0"  # Strong penalty
    os.environ["AWBW_BASE_CAPTURE_EXEMPT"] = "1"  # Exempt punishment during capture
    
    # Extreme incentives for infantry in opener turns
    os.environ["AWBW_INFANTRY_BUILD_BONUS"] = "0.5"  # Very high bonus for infantry
    os.environ["AWBW_INFANTRY_OPENER_MULT"] = "3.0"  # Higher bonus multiplier
    os.environ["AWBW_MECH_BUILD_PENALTY"] = "-0.5"  # Strong penalty for building mechs early
    os.environ["AWBW_MECH_PENALTY_EARLY_MULT"] = "4.0"  # Very strong multiplier for turns 1-3
    
    # Progressive turn-based reduction of infantry boost
    os.environ["AWBW_INFANTRY_BOOST_TURNS"] = "7"  # How many turns the infantry bonus applies

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
        opening_book_path=args.opening_book_path,
        opening_book_prob=args.opening_book_prob,
    )
    env.reset()

    # Drain joint opening book before RHEA (both seats alternate by calendar clock).
    mgr = getattr(env, '_opening_book_manager', None)
    if mgr is not None:
        n_book = _drain_opening_book(env, mgr)
        if n_book:
            print(f"  [BOOK] drained {n_book} opening indices", flush=True)

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
        two_phase_buy_rhea=not bool(getattr(args, "rhea_monolithic_buy", False)),
    )
    # Override parameters for early game
    # (default top_k_per_state = 48 from RheaConfig)
    
    # Calculate complexity metrics for dynamic budgeting if enabled
    complexity_metrics = None

    planner = RheaPlanner(
        fitness, 
        config, 
        dynamic_budget=args.rhea_autotune,
        complexity_metrics=complexity_metrics
    )

    while env.state is not None and env.state.winner is None:
        state = env.state
        active = int(state.active_player)

        # Refresh complexity metrics for dynamic budgeting
        if args.rhea_autotune:
            try:
                planner.complexity_metrics = RheaPlanner.compute_complexity_metrics(
                    state, active
                )
            except Exception:
                planner.complexity_metrics = None

        result = planner.choose_full_turn(state)

        dyn = ""
        if args.rhea_autotune:
            dyn = f" autotune pop={result.population_used} gen={result.generations_used} combos={result.population_used * result.generations_used}"
        init_info = ""
        if result.initial_best_score is not None:
            init_info = f" init={result.initial_best_score:+.4f}"
        gain_info = ""
        if result.evolved_gain is not None:
            gain_info = f" gain={result.evolved_gain:+.4f}"
        print(
            f"day={getattr(state, 'turn', '?')} active={active} "
            f"score={result.score:.4f} "
            f"phi={result.breakdown.phi_delta:.4f} "
            f"v={result.breakdown.value:.4f} "
            f"illegal={result.illegal_genes} "
            f"actions={len(result.actions)}"
            f"{dyn}{init_info}{gain_info}"
        )

        # Replay actions on real environment.
        applied, skipped = replay_rhea_actions(env.state, result.actions, active)

        # Safety: if active player didn't change after replay, force-advance.
        if (env.state is not None and env.state.winner is None
                and int(env.state.active_player) == active):
            _salvage_ender(env.state, active)
            if int(env.state.active_player) == active:
                print(f"[FATAL] cannot advance player {active} after salvage, game stuck!",
                      flush=True)
                break
        elif env.state is not None and int(env.state.active_player) != active:
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