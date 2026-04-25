"""Instrumented self-play sweep to validate Φ-based reward shaping.

Plan: ``.cursor/plans/rl_capture-combat_recalibration_4ebf9d22.plan.md``.

Runs N episodes per shaping mode (``phi`` and ``level``) under random
masked self-play. Records per-step shaping magnitudes, per-trajectory
totals, and per-event (CAPTURE / ATTACK-with-kill) shaping. Reports a
side-by-side comparison so we can answer:

  1. Are signals proportional? (cap chip vs unit kill in same order)
  2. Are signals overwhelming the terminal ±1.0? (trajectory total ≪ 1)
  3. Does Φ telescope as designed? (sum_step_shaping ≈ Φ_T − Φ_0)
  4. How does Φ compare to the legacy "level" form on the same trajectories?

Usage:
    python tools/phi_smoke.py --episodes 60 --max-steps 400
    python tools/phi_smoke.py --episodes 30 --map-id 123858 --tier T3
    python tools/phi_smoke.py --learner-seat both --episodes 20
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Per-episode telemetry ──────────────────────────────────────────────────

@dataclass
class StepRow:
    step: int
    reward: float
    shaping: float          # reward minus terminal contribution (learner frame)
    is_terminal: bool
    is_capture: bool
    is_attack: bool
    p0_units_pre: int
    p0_units_post: int
    p1_units_pre: int
    p1_units_post: int
    enemy_alive_pre: int
    enemy_alive_post: int
    phi_before: Optional[float] = None
    phi_after: Optional[float] = None


@dataclass
class EpisodeRow:
    mode: str
    map_id: int
    tier: str
    p0_co: int
    p1_co: int
    learner_seat: int
    steps: int
    winner: Optional[int]
    win_reason: Optional[str]
    terminal_reward_learner: float
    sum_reward: float
    sum_shaping: float
    captures: int
    attacks: int
    kills_enemy: int       # learner attacks that reduced enemy alive count
    rows: list[StepRow] = field(default_factory=list)


# ── Helpers ────────────────────────────────────────────────────────────────

def _terminal_reward_learner(winner: Optional[int], learner_seat: int) -> float:
    if winner is None:
        return 0.0
    return 1.0 if int(winner) == int(learner_seat) else -1.0


def _legal_count(state) -> tuple[int, int]:
    return len(state.units[0]), len(state.units[1])


def _alive_count(state, player: int) -> int:
    return sum(1 for u in state.units[player] if u.is_alive)


def _bind_greedy_opponent(env) -> None:
    """Install a capture-greedy opponent on ``env`` that closes over its state."""
    from rl.self_play import pick_capture_greedy_flat

    def _opp(_obs, mask):
        return int(pick_capture_greedy_flat(env.state, mask))

    env.opponent_policy = _opp


def _make_env(
    mode: str,
    map_id: int,
    tier: str,
    co_p0: int,
    co_p1: int,
    max_steps: int,
    opponent: str,
    *,
    learner_seat: int = 0,
):
    """Construct AWBWEnv with the chosen shaping mode and fixed learner seat."""
    os.environ["AWBW_LEARNER_SEAT"] = str(int(learner_seat))
    os.environ.pop("AWBW_SEAT_BALANCE", None)
    os.environ["AWBW_REWARD_SHAPING"] = mode
    os.environ.setdefault("AWBW_PHI_ALPHA", "2e-5")
    os.environ.setdefault("AWBW_PHI_BETA", "0.05")
    os.environ.setdefault("AWBW_PHI_KAPPA", "0.05")

    # Make sure the engine module-level gate matches the mode (it was read
    # at original import; we need to flip it for both directions).
    import engine.game as engine_game
    engine_game._PHI_SHAPING_ACTIVE = (mode == "phi")

    # Suppress per-episode game_log writes (smoke is read-only).
    import rl.env as rl_env
    rl_env._append_game_log_line = lambda _record: None

    pool_path = ROOT / "data" / "gl_map_pool.json"
    with open(pool_path, encoding="utf-8") as f:
        pool = json.load(f)
    pool_one = [next(m for m in pool if m.get("map_id") == map_id)]

    env = rl_env.AWBWEnv(
        map_pool=pool_one,
        opponent_policy=None,
        co_p0=co_p0,
        co_p1=co_p1,
        tier_name=tier,
        max_env_steps=max_steps,
        max_p1_microsteps=2000,   # bounded; default cap is 30 * max_steps which is huge
    )
    if opponent == "greedy":
        _bind_greedy_opponent(env)
    return env


def _decode(env, action_idx):
    from rl.env import _flat_to_action
    return _flat_to_action(action_idx, env.state)


def _action_kind(env, action_idx) -> str:
    """Return 'capture' / 'attack' / 'other' for the decoded action."""
    from engine.action import ActionType
    a = _decode(env, action_idx)
    if a is None:
        return "other"
    if a.action_type == ActionType.CAPTURE:
        return "capture"
    if a.action_type == ActionType.ATTACK:
        return "attack"
    return "other"


def run_episode(env, mode: str, max_steps: int, p0_policy: str) -> EpisodeRow:
    obs, info = env.reset()
    learner_seat = int(info.get("learner_seat", getattr(env, "_learner_seat", 0)))
    enemy_seat = 1 - learner_seat
    rng = random.Random()
    rng.seed((env._p0_env_steps + int(time.time() * 1e6)) & 0xFFFFFFFF)

    rows: list[StepRow] = []
    captures = attacks = kills_enemy = 0

    for step_i in range(1, max_steps + 1):
        mask = env.action_masks()
        legal_idx = np.flatnonzero(mask)
        if len(legal_idx) == 0:
            break
        if p0_policy == "greedy":
            action_idx = int(pick_capture_greedy_flat(env.state, mask))
        else:
            action_idx = int(rng.choice(legal_idx.tolist()))

        kind = _action_kind(env, action_idx)
        p0_pre, p1_pre = _legal_count(env.state)
        enemy_pre = _alive_count(env.state, enemy_seat)

        phi_before = env._compute_phi(env.state) if mode == "phi" else None

        obs, reward, terminated, truncated, step_info = env.step(action_idx)

        p0_post, p1_post = _legal_count(env.state)
        enemy_post = _alive_count(env.state, enemy_seat)
        phi_after = (
            0.0 if terminated else env._compute_phi(env.state)
        ) if mode == "phi" else None

        is_terminal = bool(terminated)
        if is_terminal:
            term_engine = _terminal_reward_learner(env.state.winner, learner_seat)
            shaping = reward - term_engine
        else:
            shaping = reward

        if kind == "capture":
            captures += 1
        if kind == "attack":
            attacks += 1
            if enemy_post < enemy_pre:
                kills_enemy += 1

        rows.append(StepRow(
            step=step_i,
            reward=float(reward),
            shaping=float(shaping),
            is_terminal=is_terminal,
            is_capture=(kind == "capture"),
            is_attack=(kind == "attack"),
            p0_units_pre=p0_pre,
            p0_units_post=p0_post,
            p1_units_pre=p1_pre,
            p1_units_post=p1_post,
            enemy_alive_pre=enemy_pre,
            enemy_alive_post=enemy_post,
            phi_before=phi_before,
            phi_after=phi_after,
        ))

        if is_terminal or truncated:
            break

    state = env.state
    return EpisodeRow(
        mode=mode,
        map_id=info.get("map_id"),
        tier=info.get("tier"),
        p0_co=info.get("p0_co"),
        p1_co=info.get("p1_co"),
        learner_seat=learner_seat,
        steps=len(rows),
        winner=state.winner,
        win_reason=state.win_reason,
        terminal_reward_learner=_terminal_reward_learner(state.winner, learner_seat),
        sum_reward=sum(r.reward for r in rows),
        sum_shaping=sum(r.shaping for r in rows),
        captures=captures,
        attacks=attacks,
        kills_enemy=kills_enemy,
        rows=rows,
    )


def _stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0}
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    def pct(p: float) -> float:
        if n == 1:
            return xs_sorted[0]
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        return xs_sorted[f] + (xs_sorted[c] - xs_sorted[f]) * (k - f)
    return {
        "n": n,
        "mean": statistics.fmean(xs_sorted),
        "stdev": statistics.pstdev(xs_sorted) if n > 1 else 0.0,
        "min": xs_sorted[0],
        "max": xs_sorted[-1],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "abs_mean": statistics.fmean([abs(x) for x in xs_sorted]),
    }


def summarize(mode: str, episodes: list[EpisodeRow]) -> dict:
    per_step_shaping: list[float] = []
    cap_step_shaping: list[float] = []
    attack_step_shaping: list[float] = []
    kill_step_shaping: list[float] = []
    traj_shaping: list[float] = []
    traj_lengths: list[int] = []
    terminal_phi_drops: list[float] = []   # phi mode only

    n_p0_wins = n_p1_wins = n_draws = n_truncated = 0

    for ep in episodes:
        traj_shaping.append(ep.sum_shaping)
        traj_lengths.append(ep.steps)
        if ep.winner == 0:
            n_p0_wins += 1
        elif ep.winner == 1:
            n_p1_wins += 1
        else:
            if ep.win_reason is None:
                n_truncated += 1
            else:
                n_draws += 1

        for r in ep.rows:
            per_step_shaping.append(r.shaping)
            if r.is_capture:
                cap_step_shaping.append(r.shaping)
            if r.is_attack:
                attack_step_shaping.append(r.shaping)
                if r.enemy_alive_post < r.enemy_alive_pre:
                    kill_step_shaping.append(r.shaping)
            if r.is_terminal and mode == "phi" and r.phi_before is not None:
                terminal_phi_drops.append(-r.phi_before)

    ls = episodes[0].learner_seat if episodes else None
    return {
        "mode": mode,
        "learner_seat": ls,
        "n_episodes": len(episodes),
        "n_p0_wins": n_p0_wins,
        "n_p1_wins": n_p1_wins,
        "n_draws": n_draws,
        "n_truncated": n_truncated,
        "episode_length": _stats([float(x) for x in traj_lengths]),
        "trajectory_shaping": _stats(traj_shaping),
        "per_step_shaping": _stats(per_step_shaping),
        "capture_step_shaping": _stats(cap_step_shaping),
        "attack_step_shaping": _stats(attack_step_shaping),
        "kill_step_shaping": _stats(kill_step_shaping),
        "terminal_phi_refund": _stats(terminal_phi_drops),
    }


def _fmt_stats(label: str, s: dict) -> str:
    if not s or s.get("n", 0) == 0:
        return f"  {label:<30}  n=0"
    return (
        f"  {label:<30}  n={s['n']:>5}  "
        f"mean={s['mean']:+.5f}  |abs|={s['abs_mean']:.5f}  "
        f"sd={s['stdev']:.5f}  "
        f"min={s['min']:+.4f}  p50={s['p50']:+.4f}  "
        f"p90={s['p90']:+.4f}  p99={s['p99']:+.4f}  max={s['max']:+.4f}"
    )


def _print_summary(summary: dict) -> None:
    mode = summary["mode"]
    n = summary["n_episodes"]
    ls = summary.get("learner_seat")
    seat_s = f" learner_seat={ls}" if ls is not None else ""
    print(f"\n=== {mode.upper()} mode{seat_s} -- {n} episodes ===")
    print(
        f"  outcomes: P0_wins={summary['n_p0_wins']}  P1_wins={summary['n_p1_wins']}"
        f"  draws={summary['n_draws']}  truncated={summary['n_truncated']}"
    )
    print(_fmt_stats("episode length (P0 steps)", summary["episode_length"]))
    print(_fmt_stats("trajectory shaping (sum F)", summary["trajectory_shaping"]))
    print(_fmt_stats("per-step shaping", summary["per_step_shaping"]))
    print(_fmt_stats("CAPTURE-step shaping", summary["capture_step_shaping"]))
    print(_fmt_stats("ATTACK-step shaping", summary["attack_step_shaping"]))
    print(_fmt_stats("KILL-step shaping (subset)", summary["kill_step_shaping"]))
    if mode == "phi":
        print(_fmt_stats("terminal Phi-refund (-Phi_pre)", summary["terminal_phi_refund"]))


def _print_verdict(phi_s: dict, level_s: dict) -> None:
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    # 1. Trajectory shaping vs terminal ±1.0
    for label, s in [("phi", phi_s), ("level", level_s)]:
        ts = s["trajectory_shaping"]
        if ts.get("n", 0) == 0:
            continue
        ratio = ts["abs_mean"] / 1.0
        verdict = (
            "OK (shaping does not drown terminal)"
            if ratio < 0.5 else
            "STRONG (shaping in same order as terminal)"
            if ratio < 2.0 else
            "OVERWHELMING (shaping drowns terminal)"
        )
        print(
            f"  [{label:>5}]  mean |trajectory shaping| = {ts['abs_mean']:.4f}  "
            f"({ratio*100:.1f}% of terminal +-1)  -> {verdict}"
        )

    # 2. Cap chip vs kill proportionality
    for label, s in [("phi", phi_s), ("level", level_s)]:
        cap = s["capture_step_shaping"]
        kill = s["kill_step_shaping"]
        if cap.get("n", 0) == 0 or kill.get("n", 0) == 0:
            print(f"  [{label:>5}]  cap or kill events too sparse to compare")
            continue
        c = cap["abs_mean"]
        k = kill["abs_mean"]
        if c == 0:
            ratio_str = "kill/cap = inf"
        else:
            ratio_str = f"kill/cap = {k/c:.2f}x"
        balance = (
            "BALANCED (within 5x)"
            if 0.2 <= (k / max(c, 1e-9)) <= 5.0 else
            "ASYMMETRIC (>5x)"
        )
        print(
            f"  [{label:>5}]  cap-step |shaping|={c:.5f}  "
            f"kill-step |shaping|={k:.5f}  ({ratio_str}) -> {balance}"
        )

    # 3. Per-step max — any single step shouldn't approach ±1
    for label, s in [("phi", phi_s), ("level", level_s)]:
        pss = s["per_step_shaping"]
        if pss.get("n", 0) == 0:
            continue
        peak = max(abs(pss["min"]), abs(pss["max"]))
        verdict = (
            "OK"
            if peak < 0.20 else
            "LOUD (single step > 20% of terminal)"
            if peak < 0.50 else
            "OVERPOWERED (single step approaches terminal)"
        )
        print(f"  [{label:>5}]  per-step peak |shaping| = {peak:.4f}  -> {verdict}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=40, help="Episodes per mode")
    p.add_argument("--max-steps", type=int, default=400, help="P0 step cap per episode")
    p.add_argument("--map-id", type=int, default=123858)
    p.add_argument("--tier", type=str, default="T3")
    p.add_argument("--co-p0", type=int, default=1)
    p.add_argument("--co-p1", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--policy", choices=("random", "greedy"), default="greedy",
        help="Policy used for both P0 and the opponent (default greedy — produces "
             "real games with cap/kill events; random rarely terminates).",
    )
    p.add_argument(
        "--learner-seat",
        choices=("0", "1", "both"),
        default="0",
        help="Fixed AWBW_LEARNER_SEAT per episode, or run 0 then 1 (seat balance off).",
    )
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    seats = [0, 1] if args.learner_seat == "both" else [int(args.learner_seat)]

    print(
        f"[smoke] map_id={args.map_id} tier={args.tier!r} "
        f"co_p0={args.co_p0} co_p1={args.co_p1}  "
        f"episodes/mode={args.episodes}  max_steps={args.max_steps}  "
        f"learner_seat={args.learner_seat!r}"
    )

    for seat in seats:
        summaries: dict[str, dict] = {}
        for mode in ("phi", "level"):
            env = _make_env(
                mode,
                args.map_id,
                args.tier,
                args.co_p0,
                args.co_p1,
                args.max_steps,
                opponent=args.policy,
                learner_seat=seat,
            )
            episodes: list[EpisodeRow] = []
            t0 = time.time()
            for _i in range(args.episodes):
                ep = run_episode(env, mode, args.max_steps, p0_policy=args.policy)
                episodes.append(ep)
            elapsed = time.time() - t0
            print(
                f"[smoke] seat={seat} {mode}: ran {args.episodes} episodes in {elapsed:.1f}s "
                f"(policy={args.policy})"
            )
            s = summarize(mode, episodes)
            summaries[mode] = s
            _print_summary(s)

        _print_verdict(summaries["phi"], summaries["level"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
