from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.env import AWBWEnv, POOL_PATH
from rl.opening_book import drain_joint_opening_book
from rl.rhea import RheaConfig, RheaPlanner, replay_rhea_actions, _salvage_ender
from rl.rhea_fitness import RheaFitness
from engine.unit import UNIT_STATS
from rl.value_net import load_value_checkpoint
from scripts.train_rhea_value_parallel import (
    add_rhea_adaptive_args,
    add_rhea_autotune_args,
    rhea_autotune_config_from_args,
    rhea_autotune_config_from_mapping,
)


def _setup_phi_env(args: argparse.Namespace) -> None:
    os.environ["AWBW_REWARD_SHAPING"] = "phi"
    if bool(args.phi_capture_phase_weighting):
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


def _load_json_arg(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    path = Path(raw)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(raw)


def _profile_from_mapping(data: dict[str, Any], args: argparse.Namespace, seed: int) -> dict[str, Any]:
    adaptive_extend_default = bool(getattr(args, "rhea_adaptive_extend", False))
    adaptive_max_extra_default = int(getattr(args, "rhea_adaptive_max_extra_generations", 0))
    adaptive_patience_default = int(getattr(args, "rhea_adaptive_patience_generations", 1))
    adaptive_min_improvement_default = float(getattr(args, "rhea_adaptive_min_improvement", 0.0025))
    adaptive_soft_wall_default = getattr(args, "rhea_adaptive_max_wall_s", None)
    adaptive_hard_wall_default = float(getattr(args, "rhea_adaptive_hard_turn_wall_s", 900.0))
    cfg = {
        "population": int(data.get("population", data.get("rhea_population", args.rhea_population))),
        "generations": int(data.get("generations", data.get("rhea_generations", args.rhea_generations))),
        "elite": int(data.get("elite", data.get("rhea_elite", args.rhea_elite))),
        "mutation_rate": float(data.get("mutation_rate", data.get("rhea_mutation_rate", args.rhea_mutation_rate))),
        "max_actions_per_turn": int(
            data.get(
                "max_actions_per_turn",
                data.get("rhea_max_actions_per_turn", getattr(args, "rhea_max_actions_per_turn", 256)),
            )
        ),
        "top_k_per_state": int(data.get("top_k_per_state", data.get("rhea_top_k_per_state", args.rhea_top_k_per_state))),
        "reward_weight": float(data.get("reward_weight", args.reward_weight)),
        "value_weight": float(data.get("value_weight", args.value_weight)),
        "build_value_weight": float(data.get("build_value_weight", args.build_value_weight)),
        "buy_mode": str(data.get("buy_mode", args.buy_mode)),
        "buy_exhaustive_max_candidates": int(data.get("buy_exhaustive_max_candidates", args.buy_exhaustive_max_candidates)),
        "adaptive_extend": bool(data.get("adaptive_extend", data.get("rhea_adaptive_extend", adaptive_extend_default))),
        "adaptive_max_extra_generations": int(data.get("adaptive_max_extra_generations", data.get("rhea_adaptive_max_extra_generations", adaptive_max_extra_default))),
        "adaptive_patience_generations": int(data.get("adaptive_patience_generations", data.get("rhea_adaptive_patience_generations", adaptive_patience_default))),
        "adaptive_min_improvement": float(data.get("adaptive_min_improvement", data.get("rhea_adaptive_min_improvement", adaptive_min_improvement_default))),
        "adaptive_max_wall_s": data.get("adaptive_max_wall_s", data.get("rhea_adaptive_max_wall_s", adaptive_soft_wall_default)),
        "adaptive_hard_turn_wall_s": float(data.get("adaptive_hard_turn_wall_s", data.get("rhea_adaptive_hard_turn_wall_s", adaptive_hard_wall_default))),
        "use_tactical_beam": bool(data.get("use_tactical_beam", data.get("rhea_use_tactical_beam", args.rhea_use_tactical_beam))),
        "tactial_beam_max_width": int(data.get("tactial_beam_max_width", data.get("rhea_tactical_beam_max_width", args.rhea_tactical_beam_max_width))),
        "tactial_beam_max_depth": int(data.get("tactial_beam_max_depth", data.get("rhea_tactical_beam_max_depth", args.rhea_tactical_beam_max_depth))),
        "tactial_beam_max_expand": int(data.get("tactial_beam_max_expand", data.get("rhea_tactical_beam_max_expand", args.rhea_tactical_beam_max_expand))),
        "capture_completion_bonus": float(data.get("capture_completion_bonus", 0.0)),
        "blunder_exposure_weight": float(data.get("blunder_exposure_weight", 0.0)),
        "hq_defense_weight": float(data.get("hq_defense_weight", 0.0)),
        "capture_interrupt_bonus": float(data.get("capture_interrupt_bonus", 0.0)),
        "neutral_income_gap_weight": float(data.get("neutral_income_gap_weight", 0.0)),
        "capture_progress_bonus": float(data.get("capture_progress_bonus", 0.0)),
        "buy_air_context_penalty": float(data.get("buy_air_context_penalty", 0.0)),
        "disable_expensive_build_bias": bool(data.get("disable_expensive_build_bias", False)),
    }
    return {
        "label": str(data.get("label", "profile")),
        "dynamic_budget": bool(data.get("dynamic_budget", data.get("rhea_autotune", args.rhea_autotune))),
        "fitness_reward_weight": cfg["reward_weight"],
        "fitness_value_weight": cfg["value_weight"],
        "capture_completion_bonus": cfg["capture_completion_bonus"],
        "blunder_exposure_weight": cfg["blunder_exposure_weight"],
        "hq_defense_weight": cfg["hq_defense_weight"],
        "capture_interrupt_bonus": cfg["capture_interrupt_bonus"],
        "neutral_income_gap_weight": cfg["neutral_income_gap_weight"],
        "capture_progress_bonus": cfg["capture_progress_bonus"],
        "config": RheaConfig(
            **cfg,
            autotune=rhea_autotune_config_from_mapping(data.get("autotune", data)),
            seed=seed,
        ),
    }


def _profile_from_args(args: argparse.Namespace, label: str, seed: int) -> dict[str, Any]:
    return {
        "label": label,
        "dynamic_budget": bool(args.rhea_autotune),
        "fitness_reward_weight": float(args.reward_weight),
        "fitness_value_weight": float(args.value_weight),
        "config": RheaConfig(
            population=args.rhea_population,
            generations=args.rhea_generations,
            elite=args.rhea_elite,
            mutation_rate=args.rhea_mutation_rate,
            autotune=rhea_autotune_config_from_args(args),
            max_actions_per_turn=args.rhea_max_actions_per_turn,
            top_k_per_state=args.rhea_top_k_per_state,
            reward_weight=args.reward_weight,
            value_weight=args.value_weight,
            build_value_weight=args.build_value_weight,
            buy_mode=args.buy_mode,
            buy_exhaustive_max_candidates=args.buy_exhaustive_max_candidates,
            adaptive_extend=args.rhea_adaptive_extend,
            adaptive_max_extra_generations=args.rhea_adaptive_max_extra_generations,
            adaptive_patience_generations=args.rhea_adaptive_patience_generations,
            adaptive_min_improvement=args.rhea_adaptive_min_improvement,
            adaptive_max_wall_s=args.rhea_adaptive_max_wall_s,
            adaptive_hard_turn_wall_s=args.rhea_adaptive_hard_turn_wall_s,
            seed=seed,
            use_tactical_beam=args.rhea_use_tactical_beam,
            tactial_beam_max_width=args.rhea_tactical_beam_max_width,
            tactial_beam_max_depth=args.rhea_tactical_beam_max_depth,
            tactial_beam_max_expand=args.rhea_tactical_beam_max_expand,
        ),
    }


def _map_pool(map_id: int) -> list[dict[str, Any]]:
    with open(POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    narrowed = [m for m in pool if int(m.get("map_id", -1)) == int(map_id)]
    if not narrowed:
        raise SystemExit(f"map_id {map_id} not found in {POOL_PATH}")
    return narrowed


def _game_metrics(env: AWBWEnv) -> dict[str, Any]:
    state = env.state
    if state is None:
        return {"winner": None}
    return {
        "winner": state.winner,
        "days": int(getattr(state, "turn", 0)),
        "property_count": [state.count_properties(0), state.count_properties(1)],
        "income_property_count": [state.count_income_properties(0), state.count_income_properties(1)],
        "captures_completed": [
            sum(1 for e in state.game_log if e.get("type") == "capture" and e.get("player") == 0 and e.get("cp_remaining") in (0, 20)),
            sum(1 for e in state.game_log if e.get("type") == "capture" and e.get("player") == 1 and e.get("cp_remaining") in (0, 20)),
        ],
        "funds_end": list(state.funds),
        "gold_spent": list(state.gold_spent),
        "alive_unit_count": [sum(1 for u in state.units[0] if u.is_alive), sum(1 for u in state.units[1] if u.is_alive)],
        "army_value": [
            sum(UNIT_STATS[u.unit_type].cost * max(0, int(u.hp)) / 100.0 for u in state.units[0] if u.is_alive),
            sum(UNIT_STATS[u.unit_type].cost * max(0, int(u.hp)) / 100.0 for u in state.units[1] if u.is_alive),
        ],
    }


def _planner_for(
    env: AWBWEnv,
    value_model: Any,
    profile: dict[str, Any],
    device: str,
    seed: int,
) -> RheaPlanner:
    fitness = RheaFitness(
        env_template=env,
        value_model=value_model,
        device=device,
        reward_weight=float(profile["fitness_reward_weight"]),
        value_weight=float(profile["fitness_value_weight"]),
        capture_completion_bonus=float(profile.get("capture_completion_bonus", 0.0)),
        blunder_exposure_weight=float(profile.get("blunder_exposure_weight", 0.0)),
        hq_defense_weight=float(profile.get("hq_defense_weight", 0.0)),
        capture_interrupt_bonus=float(profile.get("capture_interrupt_bonus", 0.0)),
        neutral_income_gap_weight=float(profile.get("neutral_income_gap_weight", 0.0)),
        capture_progress_bonus=float(profile.get("capture_progress_bonus", 0.0)),
    )
    return RheaPlanner(
        fitness,
        dataclasses.replace(profile["config"], seed=int(seed)),
        dynamic_budget=bool(profile["dynamic_budget"]),
        complexity_metrics=None,
    )


def _run_one_game(
    *,
    args: argparse.Namespace,
    map_pool: list[dict[str, Any]],
    value_model: Any,
    profile_by_seat: dict[int, dict[str, Any]],
    seed: int,
    replay_path: Path | None = None,
) -> dict[str, Any]:
    random.seed(seed)
    game_wall_start = time.perf_counter()
    truncation_reason = None
    env = AWBWEnv(
        map_pool=map_pool,
        co_p0=args.co_p0,
        co_p1=args.co_p1,
        max_turns=args.max_days,
        curriculum_tag="rhea_config_eval",
        opening_book_path=args.opening_book_path,
        opening_book_seats="both",
        opening_book_prob=args.opening_book_prob,
        opening_book_strike_release=bool(args.opening_book_strike_release),
        cop_disable_per_seat_p=args.cop_disable_per_seat_p,
    )
    env._build_punishment = 0.0
    env.reset(seed=seed)
    mgr = getattr(env, "_opening_book_manager", None)
    if mgr is not None:
        drain_joint_opening_book(env, mgr, verbose=False)

    planners = {
        seat: _planner_for(
            env,
            value_model,
            profile_by_seat[seat],
            args.device,
            seed ^ (0x9E3779B9 + seat),
        )
        for seat in (0, 1)
    }
    telemetry = {
        0: {"search_wall_s": 0.0, "turns": 0, "early_wall_s": 0.0, "complex_wall_s": 0.0, "pop": [], "gen": []},
        1: {"search_wall_s": 0.0, "turns": 0, "early_wall_s": 0.0, "complex_wall_s": 0.0, "pop": [], "gen": []},
    }
    max_player_turns = int(args.max_days) * 2 + 4
    max_game_wall_s = float(args.max_game_wall_s or 0.0)
    player_turns = 0
    snapshots: list[Any] = []
    if replay_path is not None and env.state is not None:
        snapshots.append(copy.deepcopy(env.state))
    while env.state is not None and env.state.winner is None and player_turns < max_player_turns:
        if max_game_wall_s > 0.0 and (time.perf_counter() - game_wall_start) >= max_game_wall_s:
            truncation_reason = "max_game_wall_eval"
            break
        state = env.state
        active = int(state.active_player)
        planner = planners[active]
        try:
            metrics = RheaPlanner.compute_complexity_metrics(state, active)
        except Exception:
            metrics = None
        planner.dynamic_budget = bool(profile_by_seat[active]["dynamic_budget"])
        planner.complexity_metrics = metrics
        t0 = time.perf_counter()
        result = planner.choose_full_turn(state)
        dt = time.perf_counter() - t0
        planner.note_turn_wall_time(dt)
        if getattr(result, "adaptive_disabled_reason", None) is None:
            result.adaptive_disabled_reason = planner.adaptive_disabled_reason
        tel = telemetry[active]
        tel["search_wall_s"] += dt
        tel["turns"] += 1
        tel["pop"].append(int(result.population_used))
        tel["gen"].append(int(result.generations_used))
        tel.setdefault("move_gen_floor", []).append(int(getattr(result, "move_generations_floor", 0)))
        tel.setdefault("move_gen_used", []).append(int(getattr(result, "move_generations_used", 0)))
        tel.setdefault("adaptive_extra_gen", []).append(int(getattr(result, "adaptive_extra_generations_used", 0)))
        stop_reason = getattr(result, "adaptive_stop_reason", None)
        if stop_reason is not None:
            tel.setdefault("adaptive_stop_reason", []).append(str(stop_reason))
        disabled_reason = getattr(result, "adaptive_disabled_reason", None)
        if disabled_reason is not None:
            tel.setdefault("adaptive_disabled_reason", []).append(str(disabled_reason))
        best_improvement = getattr(result, "adaptive_best_improvement", None)
        if best_improvement is not None:
            tel.setdefault("adaptive_best_improvement", []).append(float(best_improvement))
        if int(getattr(state, "turn", 0)) <= 5:
            tel["early_wall_s"] += dt
        if metrics is not None and (metrics[2] > 0 or metrics[3] > 0 or metrics[0] >= 8):
            tel["complex_wall_s"] += dt
        applied, skipped = replay_rhea_actions(env.state, result.actions, active)
        tel["actions_skipped"] = int(tel.get("actions_skipped", 0)) + int(skipped)
        if env.state is not None and env.state.winner is None and int(env.state.active_player) == active:
            _salvage_ender(env.state, active)
        elif replay_path is not None and env.state is not None and int(env.state.active_player) != active:
            snapshots.append(copy.deepcopy(env.state))
        player_turns += 1

    truncated = bool(env.state is not None and env.state.winner is None)
    if truncated and truncation_reason is None:
        truncation_reason = "max_days_eval"
    try:
        env.finalize_rhea_episode(
            truncated=truncated,
            truncation_reason=truncation_reason,
        )
    except Exception as exc:
        print(f"[rhea-config-eval] game_log finalize failed: {exc!r}", file=sys.stderr)

    if replay_path is not None and env.state is not None:
        snapshots.append(copy.deepcopy(env.state))
        try:
            from tools.export_awbw_replay import write_awbw_replay

            replay_path.parent.mkdir(parents=True, exist_ok=True)
            labels = " vs ".join(
                profile_by_seat[s]["label"].split(" \u2014")[0] for s in (0, 1)
            )
            write_awbw_replay(
                snapshots=snapshots,
                output_path=replay_path,
                game_id=int(seed) % 999000 + 1000,
                game_name=f"Config eval - {labels} - map {args.map_id}",
                start_date=time.strftime("%Y-%m-%d %H:%M:%S"),
                full_trace=env.state.full_trace,
                luck_seed=None,
            )
            print(f"[rhea-config-eval] replay exported: {replay_path}")
        except Exception as exc:
            print(f"[rhea-config-eval] replay export failed: {exc!r}", file=sys.stderr)

    return {
        **_game_metrics(env),
        "truncated": truncated,
        "truncation_reason": truncation_reason,
        "game_wall_s": time.perf_counter() - game_wall_start,
        "profile_by_seat": {str(k): v["label"] for k, v in profile_by_seat.items()},
        "telemetry": telemetry,
    }


def _summarize(results: list[dict[str, Any]], label_a: str, label_b: str) -> dict[str, Any]:
    wins = {label_a: 0, label_b: 0, "draw": 0}
    captures = {label_a: [], label_b: []}
    props = {label_a: [], label_b: []}
    income_props = {label_a: [], label_b: []}
    army = {label_a: [], label_b: []}
    wall = {label_a: [], label_b: []}
    paired: dict[int, dict[str, Any]] = {}
    truncated_games = 0
    game_wall_s: list[float] = []
    for row in results:
        if bool(row.get("truncated")):
            truncated_games += 1
        game_wall_s.append(float(row.get("game_wall_s", 0.0)))
        seat_labels = {int(k): v for k, v in row["profile_by_seat"].items()}
        winner = row.get("winner")
        if winner in (0, 1):
            wins[seat_labels[int(winner)]] += 1
        else:
            wins["draw"] += 1
        for seat in (0, 1):
            label = seat_labels[seat]
            captures[label].append(row.get("captures_completed", [0, 0])[seat])
            props[label].append(row.get("property_count", [0, 0])[seat])
            income_props[label].append(row.get("income_property_count", [0, 0])[seat])
            army[label].append(row.get("army_value", [0, 0])[seat])
            wall[label].append(row.get("telemetry", {}).get(seat, row.get("telemetry", {}).get(str(seat), {})).get("search_wall_s", 0.0))
        pair_id = row.get("pair_id")
        if pair_id is not None:
            slot = paired.setdefault(int(pair_id), {})
            slot[str(row.get("block"))] = row

    def avg(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    paired_deltas: list[dict[str, float]] = []
    for pair_id, rows in sorted(paired.items()):
        if "a_p0" not in rows or "a_p1" not in rows:
            continue
        totals = {
            label_a: {"captures": 0.0, "properties": 0.0, "income_properties": 0.0, "army_value": 0.0, "search_wall_s": 0.0},
            label_b: {"captures": 0.0, "properties": 0.0, "income_properties": 0.0, "army_value": 0.0, "search_wall_s": 0.0},
        }
        for row in (rows["a_p0"], rows["a_p1"]):
            seat_labels = {int(k): v for k, v in row["profile_by_seat"].items()}
            for seat in (0, 1):
                label = seat_labels[seat]
                tel = row.get("telemetry", {}).get(seat, row.get("telemetry", {}).get(str(seat), {}))
                totals[label]["captures"] += float(row.get("captures_completed", [0, 0])[seat])
                totals[label]["properties"] += float(row.get("property_count", [0, 0])[seat])
                totals[label]["income_properties"] += float(row.get("income_property_count", [0, 0])[seat])
                totals[label]["army_value"] += float(row.get("army_value", [0, 0])[seat])
                totals[label]["search_wall_s"] += float(tel.get("search_wall_s", 0.0))
        paired_deltas.append({
            "pair_id": float(pair_id),
            "captures_delta_a_minus_b": totals[label_a]["captures"] - totals[label_b]["captures"],
            "properties_delta_a_minus_b": totals[label_a]["properties"] - totals[label_b]["properties"],
            "income_properties_delta_a_minus_b": totals[label_a]["income_properties"] - totals[label_b]["income_properties"],
            "army_value_delta_a_minus_b": totals[label_a]["army_value"] - totals[label_b]["army_value"],
            "search_wall_s_delta_a_minus_b": totals[label_a]["search_wall_s"] - totals[label_b]["search_wall_s"],
        })

    return {
        "wins": wins,
        "avg_captures": {k: avg([float(x) for x in v]) for k, v in captures.items()},
        "avg_properties": {k: avg([float(x) for x in v]) for k, v in props.items()},
        "avg_income_properties": {k: avg([float(x) for x in v]) for k, v in income_props.items()},
        "avg_army_value": {k: avg([float(x) for x in v]) for k, v in army.items()},
        "avg_search_wall_s": {k: avg([float(x) for x in v]) for k, v in wall.items()},
        "avg_game_wall_s": avg(game_wall_s),
        "truncated_games": truncated_games,
        "paired_deltas_mean": {
            "captures_a_minus_b": avg([x["captures_delta_a_minus_b"] for x in paired_deltas]),
            "properties_a_minus_b": avg([x["properties_delta_a_minus_b"] for x in paired_deltas]),
            "income_properties_a_minus_b": avg([x["income_properties_delta_a_minus_b"] for x in paired_deltas]),
            "army_value_a_minus_b": avg([x["army_value_delta_a_minus_b"] for x in paired_deltas]),
            "search_wall_s_a_minus_b": avg([x["search_wall_s_delta_a_minus_b"] for x in paired_deltas]),
        },
        "paired_deltas": paired_deltas,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--config-a-json", type=str, default=None)
    ap.add_argument("--config-b-json", type=str, default=None)
    ap.add_argument("--label-a", type=str, default="config_a")
    ap.add_argument("--label-b", type=str, default="config_b")
    ap.add_argument("--json-out", type=Path, default=Path("logs/rhea_config_eval.json"))
    ap.add_argument("--export-replays-dir", type=str, default=None,
                    help="If set, export an AWBW replay zip per game into this directory")
    ap.add_argument("--games-per-seat", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--map-id", type=int, default=171596)
    ap.add_argument("--co-p0", type=str, default="14")
    ap.add_argument("--co-p1", type=str, default="14")
    ap.add_argument("--max-days", type=int, default=30)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-game-wall-s", type=float, default=0.0)
    ap.add_argument("--rhea-autotune", action="store_true")
    ap.add_argument("--rhea-population", type=int, default=64)
    ap.add_argument("--rhea-generations", type=int, default=10)
    ap.add_argument("--rhea-elite", type=int, default=8)
    ap.add_argument("--rhea-mutation-rate", type=float, default=0.20)
    ap.add_argument("--rhea-max-actions-per-turn", type=int, default=256)
    ap.add_argument("--rhea-top-k-per-state", type=int, default=96)
    add_rhea_autotune_args(ap)
    ap.add_argument("--buy-mode", choices=("rhea", "exhaustive"), default="exhaustive")
    ap.add_argument("--buy-exhaustive-max-candidates", type=int, default=8192)
    add_rhea_adaptive_args(ap)
    ap.add_argument("--reward-weight", type=float, default=0.5)
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--build-value-weight", type=float, default=20.0)
    ap.add_argument("--rhea-use-tactical-beam", action="store_true")
    ap.add_argument("--rhea-tactical-beam-max-width", type=int, default=96)
    ap.add_argument("--rhea-tactical-beam-max-depth", type=int, default=28)
    ap.add_argument("--rhea-tactical-beam-max-expand", type=int, default=48)
    ap.add_argument("--opening-book-path", type=str, default=None)
    ap.add_argument("--opening-book-prob", type=float, default=1.0)
    ap.add_argument("--opening-book-strike-release", action="store_true")
    ap.add_argument("--cop-disable-per-seat-p", type=float, default=0.10)
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
    return ap


def main() -> int:
    args = build_parser().parse_args()
    _setup_phi_env(args)
    map_pool = _map_pool(args.map_id)
    value_model = load_value_checkpoint(args.checkpoint, device=args.device)
    rng = random.Random(args.seed)
    cfg_a_data = _load_json_arg(args.config_a_json)
    cfg_b_data = _load_json_arg(args.config_b_json)
    cfg_a_data.setdefault("label", args.label_a)
    cfg_b_data.setdefault("label", args.label_b)
    profile_a = _profile_from_mapping(cfg_a_data, args, rng.randrange(1 << 30)) if cfg_a_data else _profile_from_args(args, args.label_a, rng.randrange(1 << 30))
    profile_b = _profile_from_mapping(cfg_b_data, args, rng.randrange(1 << 30)) if cfg_b_data else _profile_from_args(args, args.label_b, rng.randrange(1 << 30))

    results: list[dict[str, Any]] = []
    for i in range(int(args.games_per_seat)):
        seed = rng.randrange(1 << 30)
        for block, seats in (("a_p0", {0: profile_a, 1: profile_b}), ("a_p1", {0: profile_b, 1: profile_a})):
            replay_path = None
            if args.export_replays_dir:
                replay_path = Path(args.export_replays_dir) / f"{args.json_out.stem}_pair{i + 1}_{block}.zip"
            row = _run_one_game(args=args, map_pool=map_pool, value_model=value_model, profile_by_seat=seats, seed=seed, replay_path=replay_path)
            row.update({"block": block, "game_in_block": i + 1, "pair_id": i + 1, "seed": seed})
            results.append(row)
            print(
                f"[rhea-config-eval] block={block} game={i + 1} seed={seed} "
                f"winner={row.get('winner')} captures={row.get('captures_completed')} "
                f"props={row.get('property_count')} truncated={row.get('truncated')}",
                flush=True,
            )

    payload = {
        "checkpoint": str(Path(args.checkpoint)),
        "map_id": args.map_id,
        "co_p0": args.co_p0,
        "co_p1": args.co_p1,
        "max_days": args.max_days,
        "max_game_wall_s": args.max_game_wall_s,
        "config_a": {"label": profile_a["label"], "dynamic_budget": profile_a["dynamic_budget"], "config": dataclasses.asdict(profile_a["config"])},
        "config_b": {"label": profile_b["label"], "dynamic_budget": profile_b["dynamic_budget"], "config": dataclasses.asdict(profile_b["config"])},
        "summary": _summarize(results, profile_a["label"], profile_b["label"]),
        "games": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"[rhea-config-eval] wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
