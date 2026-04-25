"""
Phase 10I — engine step() latency benchmark (STEP-GATE vs oracle_mode).

Measures per-call wall time for GameState.step(), time inside get_legal_actions
invoked from STEP-GATE (engine.game namespace), and residual (step minus gate).

Run: python tools/_phase10i_step_benchmark.py
"""
from __future__ import annotations

import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Patches: only engine.game.get_legal_actions is wrapped for timing.
# step() resolves get_legal_actions from game.py imports — that symbol only.
# Sampling uses engine.action.get_legal_actions (unpatched) so we do not mix
# mask enumeration into STEP-GATE timings.
# ---------------------------------------------------------------------------

_inside_step: bool = False

_orig_step = None
_orig_game_get_legal = None

_step_times_ns: list[int] = []
_legal_gate_times_ns: list[int] = []


def _pct(samples: list[int], p: float) -> float:
    if not samples:
        return float("nan")
    s = sorted(samples)
    n = len(s)
    idx = min(int(p * (n - 1)), n - 1)
    return float(s[idx])


def _report(name: str, samples_ns: list[int]) -> dict[str, float]:
    if not samples_ns:
        return {
            "n": 0,
            "mean_us": float("nan"),
            "median_us": float("nan"),
            "p95_us": float("nan"),
            "p99_us": float("nan"),
        }
    us = [x / 1000.0 for x in samples_ns]
    return {
        "n": len(samples_ns),
        "mean_us": statistics.mean(us),
        "median_us": statistics.median(us),
        "p95_us": _pct(samples_ns, 0.95) / 1000.0,
        "p99_us": _pct(samples_ns, 0.99) / 1000.0,
    }


def _install_patches(game_mod) -> None:
    global _orig_step, _orig_game_get_legal

    _orig_step = game_mod.GameState.step

    def timed_step(self, action, **kwargs):
        global _inside_step
        t0 = time.perf_counter_ns()
        _inside_step = True
        try:
            return _orig_step(self, action, **kwargs)
        finally:
            _inside_step = False
            _step_times_ns.append(time.perf_counter_ns() - t0)

    _orig_game_get_legal = game_mod.get_legal_actions

    def timed_game_get_legal(state, *a, **kw):
        if _inside_step:
            t0 = time.perf_counter_ns()
            r = _orig_game_get_legal(state, *a, **kw)
            _legal_gate_times_ns.append(time.perf_counter_ns() - t0)
            return r
        return _orig_game_get_legal(state, *a, **kw)

    game_mod.GameState.step = timed_step
    game_mod.get_legal_actions = timed_game_get_legal


def _restore_patches(game_mod) -> None:
    game_mod.GameState.step = _orig_step
    game_mod.get_legal_actions = _orig_game_get_legal


def _reset_samples() -> None:
    _step_times_ns.clear()
    _legal_gate_times_ns.clear()


def run_benchmark_steps(
    *,
    n_steps: int,
    oracle_mode: bool,
    make_state,
    rng: random.Random,
    get_legal_sample,
) -> dict[str, object]:
    """
    get_legal_sample: unpatched get_legal_actions from engine.action (mask for RNG).
    Reinitializes when the episode ends so we reach n_steps total.
    """
    state = make_state()
    steps = 0
    games = 1
    max_games = max(5000, n_steps // 2)
    while steps < n_steps and games <= max_games:
        if state.done:
            state = make_state()
            games += 1
            continue
        legal = get_legal_sample(state)
        if not legal:
            state = make_state()
            games += 1
            continue
        action = rng.choice(legal)
        state.step(action, oracle_mode=oracle_mode)
        steps += 1
    return {
        "steps": steps,
        "games_touched": games,
        "final_done": state.done,
        "hit_game_cap": games > max_games and steps < n_steps,
    }


def main() -> None:
    import engine.game as eg
    from engine.action import get_legal_actions as get_legal_sample
    from engine.game import make_initial_state
    from engine.map_loader import load_map

    # Canonical map: first GL pool id (same pool as Phase 4 fuzzer tests).
    pool_path = _ROOT / "data" / "gl_map_pool.json"
    maps_dir = _ROOT / "data" / "maps"
    map_id = 133665

    map_data = load_map(map_id, pool_path, maps_dir)
    co_id = 1

    n_steps = 10_000
    seed = 42

    def make_state():
        return make_initial_state(map_data, co_id, co_id, tier_name="T2")

    _install_patches(eg)

    results: dict[str, object] = {
        "map_id": map_id,
        "pool_path": str(pool_path),
        "maps_dir": str(maps_dir),
        "n_steps_target": n_steps,
        "seed": seed,
        "python": sys.version,
        "platform": {
            "node": platform.node(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "machine_methodology": (
            "time.perf_counter_ns(); monkeypatch engine.game.GameState.step and "
            "engine.game.get_legal_actions; mask sampling via engine.action.get_legal_actions "
            "(unpatched)."
        ),
    }

    # --- STEP-GATE on (default) ---
    _reset_samples()
    rng = random.Random(seed)
    bench_gate = run_benchmark_steps(
        n_steps=n_steps,
        oracle_mode=False,
        make_state=make_state,
        rng=rng,
        get_legal_sample=get_legal_sample,
    )
    step_gate = _report("step oracle_mode=False", _step_times_ns)
    legal_gate = _report("get_legal_actions STEP-GATE", _legal_gate_times_ns)
    if len(_step_times_ns) == len(_legal_gate_times_ns) and _step_times_ns:
        residual_ns = [a - b for a, b in zip(_step_times_ns, _legal_gate_times_ns)]
        residual = _report("step minus STEP-GATE (trace+apply+win)", residual_ns)
        ratio = (
            statistics.mean(_legal_gate_times_ns) / statistics.mean(_step_times_ns)
            if _step_times_ns
            else float("nan")
        )
    else:
        residual = {}
        ratio = float("nan")

    # --- oracle_mode=True (no STEP-GATE) ---
    _reset_samples()
    rng_o = random.Random(seed)
    bench_oracle = run_benchmark_steps(
        n_steps=n_steps,
        oracle_mode=True,
        make_state=make_state,
        rng=rng_o,
        get_legal_sample=get_legal_sample,
    )
    step_oracle = _report("step oracle_mode=True", _step_times_ns)
    assert len(_legal_gate_times_ns) == 0  # no gate calls

    mean_gate = step_gate["mean_us"]
    mean_ora = step_oracle["mean_us"]
    if mean_ora and mean_ora > 0:
        overhead_pct = (mean_gate - mean_ora) / mean_ora * 100.0
    else:
        overhead_pct = float("nan")

    if overhead_pct < 5:
        verdict = "GREEN"
    elif overhead_pct <= 15:
        verdict = "YELLOW"
    else:
        verdict = "RED"

    results.update(
        {
            "benchmark_with_step_gate": bench_gate,
            "benchmark_oracle_mode": bench_oracle,
            "step_oracle_mode_false": step_gate,
            "step_oracle_mode_true": step_oracle,
            "get_legal_actions_step_gate_only": legal_gate,
            "step_minus_gate_residual": residual,
            "mean_legal_to_mean_step_ratio": ratio,
            "step_gate_overhead_percent_vs_oracle": overhead_pct,
            "verdict": verdict,
        }
    )

    _restore_patches(eg)

    out_json = _ROOT / "logs" / "phase10i_step_latency.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # stdout summary
    print("Phase 10I step() benchmark")
    print(f"  map_id={map_id}  n_target={n_steps}  seed={seed}")
    print(f"  steps (gate): {bench_gate}")
    print(f"  steps (oracle): {bench_oracle}")
    print(f"  step() us — oracle_mode=False: {step_gate}")
    print(f"  step() us — oracle_mode=True:  {step_oracle}")
    print(f"  get_legal_actions (STEP-GATE only): {legal_gate}")
    print(f"  residual (step - gate legal): {residual}")
    print(f"  mean(legal)/mean(step) (gate path): {ratio:.4f}")
    print(f"  STEP-GATE overhead vs oracle: {overhead_pct:.2f}%")
    print(f"  verdict: {verdict}")
    print(f"  wrote {out_json}")


if __name__ == "__main__":
    main()
