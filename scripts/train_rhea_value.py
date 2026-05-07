from __future__ import annotations

from engine.game import IllegalActionError

"""Train a value-only network from RHEA-generated turn transitions.

This is the RHEA-machine entrypoint. It deliberately does not use PPO rollout
buffers or policy-gradient updates.

Example:

python scripts/train_rhea_value.py ^
  --map-id 171596 ^
  --co-p0 14,8,28,7 ^
  --co-p1 14,8,28,7 ^
  --checkpoint checkpoints/latest.zip ^
  --max-days 30 ^
  --rhea-population 16 ^
  --rhea-generations 3 ^
  --rhea-elite 4 ^
  --rhea-mutation-rate 0.20 ^
  --rhea-top-k-per-state 16 ^
  --rhea-max-actions-per-turn 96 ^
  --reward-weight 0.90 ^
  --value-weight 0.10 ^
  --value-lr 1e-4 ^
  --value-batch-size 128 ^
  --replay-size 50000 ^
  --min-replay-before-train 1000 ^
  --updates-per-turn 1 ^
  --gamma-turn 0.99 ^
  --target-update-interval 1000 ^
  --grad-clip 1.0 ^
  --freeze-encoder
"""

import sys
import os

# Add the parent directory to the Python path so we can import rl and engine modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from rl.encoder import encode_state, GRID_SIZE, N_SPATIAL_CHANNELS, N_SCALARS
from rl.env import AWBWEnv
from rl.rhea import RheaConfig, RheaPlanner
from rl.rhea_fitness import RheaFitness
from rl.rhea_replay import RheaReplayBuffer, RheaTransition
from rl.rhea_value_learner import RheaValueLearner, RheaValueLearnerConfig
from rl.value_net import AWBWValueNet, load_value_from_maskable_ppo_zip


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


def _make_env(args: argparse.Namespace) -> AWBWEnv:
    """Construct an AWBWEnv.

    This is intentionally simple and may need a tiny local adjustment if your
    branch's AWBWEnv constructor is wrapped by start_solo_training.py factories.
    The RHEA/value learner itself is independent from PPO.
    """

    co_p0 = _parse_co_list(args.co_p0)
    co_p1 = _parse_co_list(args.co_p1)
    
    # Load map pool and filter to only include the specified map ID
    import json
    from pathlib import Path
    
    pool_path = Path(__file__).parent.parent / "data" / "gl_map_pool.json"
    with open(pool_path) as f:
        all_maps = json.load(f)
    
    # Filter to only include the specified map ID
    map_pool = [m for m in all_maps if m.get("map_id") == args.map_id]
    
    if not map_pool:
        raise ValueError(f"Map ID {args.map_id} not found in {pool_path}")
    
    # max_days parameter should be max_turns in AWBWEnv
    max_turns = args.max_days

    # Most local branches accept int CO ids or list/sequence pools. Keep the
    # exact user-provided pool when possible; fallback to first element if the
    # constructor is stricter.
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


def _save_checkpoint(path: Path, model: AWBWValueNet, learner_cfg: RheaValueLearnerConfig, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "learner_cfg": learner_cfg.__dict__,
            "step": step,
        },
        path,
    )

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


def _timestamp_str() -> str:
    """Return a compact timestamp string for checkpoint naming."""
    import time
    return time.strftime("%Y%m%d_%H%M%S")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    # Env / checkpoint.
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--map-id", type=int, default=171596)
    ap.add_argument("--co-p0", type=str, default="14,8,28,7")
    ap.add_argument("--co-p1", type=str, default="14,8,28,7")
    ap.add_argument("--max-days", type=int, default=30)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)

    # RHEA search.
    ap.add_argument("--rhea-population", type=int, default=16)
    ap.add_argument("--rhea-generations", type=int, default=3)
    ap.add_argument("--rhea-elite", type=int, default=4)
    ap.add_argument("--rhea-mutation-rate", type=float, default=0.20)
    ap.add_argument("--rhea-top-k-per-state", type=int, default=16)
    ap.add_argument("--rhea-max-actions-per-turn", type=int, default=96)
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

    # Freeze schedule.
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--unfreeze-last-resblocks", type=int, default=0)

    # Run control.
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--save-every-games", type=int, default=25)
    ap.add_argument("--log-every-turns", type=int, default=25)
    ap.add_argument("--output-dir", type=str, default="runs/rhea_value")
    ap.add_argument("--cop-disable-per-seat-p", type=float, default=0.10,
        help="Probability (0-1) to disable COP for each seat at game start (default 0.10 = 10%%)")

    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env(args)

    online = load_value_from_maskable_ppo_zip(args.checkpoint, device=args.device)
    target = copy.deepcopy(online)

    replay = RheaReplayBuffer(args.replay_size, seed=args.seed)

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

    fitness = RheaFitness(
        env_template=env,
        value_model=online,
        device=args.device,
        reward_weight=args.reward_weight,
        value_weight=args.value_weight,
    )
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
            seed=args.seed,
            use_tactical_beam=args.rhea_use_tactical_beam,
            tactial_beam_max_width=args.rhea_tactical_beam_max_width,
            tactial_beam_max_depth=args.rhea_tactical_beam_max_depth,
            tactial_beam_max_expand=args.rhea_tactical_beam_max_expand,
        ),
    )

    hparam_path = output_dir / "hparams.json"
    hparam_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    global_turn = 0
    for game_idx in range(1, args.games + 1):
        obs, info = env.reset()
        game_turns = 0

        # 10% chance to disable COP for each seat at game start (forces SCOP learning)
        cop_disable_p = getattr(args, "cop_disable_per_seat_p", 0.10)
        for seat in (0, 1):
            _maybe_disable_cop_for_seat(env.state.co_states[seat], cop_disable_p)

        while env.state is not None and env.state.winner is None:
            state = env.state
            acting = int(state.active_player)
            day = int(getattr(state, "turn", getattr(state, "day", 0)))

            before_spatial, before_scalars = _encode(state, acting)
            phi_before = fitness.phi(state, acting)

            result = planner.choose_full_turn(state)

            # Execute selected full-turn actions on the real game state.
            # Track abnormal termination in this game
            _game_abnormal_error = None

            for action in result.actions:
                if env.state is None or env.state.winner is not None:
                    break
                if int(env.state.active_player) != acting:
                    break
                try:
                    env.state.step(action)
                except IllegalActionError as illegal_e:
                    import traceback
                    print(json.dumps({
                        "event": "illegal_action",
                        "game": game_idx,
                        "error": repr(illegal_e),
                        "game_turns": game_turns,
                        "day": day,
                        "action": str(action),
                        "traceback": traceback.format_exc(),
                    }), flush=True)
                    _game_abnormal_error = repr(illegal_e)
                    # Try to find a legal action for the CURRENT state
                    try:
                        from engine.action import get_legal_actions, ActionType
                        legal = get_legal_actions(env.state)
                        if legal:
                            env.state.step(legal[0])
                        else:
                            env.state.step(ActionType.END_TURN)
                    except Exception:
                        try:
                            env.state.step(ActionType.END_TURN)
                        except:
                            pass
                    break  # stop executing remaining actions

            after = env.state
            if after is None:
                break

            after_spatial, after_scalars = _encode(after, acting)
            phi_after = fitness.phi(after, acting)
            reward_turn = float(phi_after - phi_before)
            done = bool(after.winner is not None)

            replay.add(
                RheaTransition(
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
            )

            train_logs = learner.maybe_train_after_turn()
            global_turn += 1
            game_turns += 1

            if args.log_every_turns > 0 and global_turn % args.log_every_turns == 0:
                last_train = train_logs[-1] if train_logs else {}
                print(
                    json.dumps(
                        {
                            "game": game_idx,
                            "global_turn": global_turn,
                            "game_turn": game_turns,
                            "day": day,
                            "acting": acting,
                            "winner": after.winner,
                            "replay": len(replay),
                            "search_score": result.score,
                            "phi_delta_search": result.breakdown.phi_delta,
                            "reward_turn_real": reward_turn,
                            "value_after": result.breakdown.value,
                            "initial_best_score": result.initial_best_score,
                            "evolved_gain": result.evolved_gain,
                            "illegal_genes": result.illegal_genes,
                            "actions": len(result.actions),
                            **last_train,
                        },
                        sort_keys=True,
                    )
                )

            # Hard day cap safety if env does not terminal out cleanly.
            if day > args.max_days + 1:
                break

        print(json.dumps({
            "event": "game_done",
            "game": game_idx,
            "winner": None if env.state is None else env.state.winner,
            "turns": game_turns,
            "abnormal_termination": _game_abnormal_error is not None,
            "termination_error": _game_abnormal_error,
        }))

        if args.save_every_games > 0 and game_idx % args.save_every_games == 0:
            _save_checkpoint(output_dir / f"value_rhea_{_timestamp_str()}.pt", online, learner_cfg, global_turn)

    _save_checkpoint(output_dir / "value_rhea_latest.pt", online, learner_cfg, global_turn)


if __name__ == "__main__":
    main()