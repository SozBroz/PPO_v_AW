"""One-off: mine logs/game_log.jsonl for Phase-0 FPS baselines. Delete after use."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _pct_sorted(sorted_vals: list[float], p: float) -> float | None:
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_vals[0])
    k = (n - 1) * p / 100.0
    f = int(math.floor(k))
    c = min(f + 1, n - 1)
    return float(sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f]))


def _summarize_numeric(
    values: list[float],
) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "median": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    s = sorted(values)
    return {
        "count": len(values),
        "median": float(statistics.median(s)),
        "p25": _pct_sorted(s, 25.0),
        "p75": _pct_sorted(s, 75.0),
        "p95": _pct_sorted(s, 95.0),
        "max": float(s[-1]),
    }


def _get_float(row: dict, key: str) -> float | None:
    v = row.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _get_int(row: dict, key: str) -> int | None:
    v = row.get(key)
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def mine(rows: list[dict[str, Any]]) -> dict[str, Any]:
    schema_counts = Counter(str(r.get("log_schema_version", "<missing>")) for r in rows)

    keys_num = [
        "wall_p0_s",
        "wall_p1_s",
        "worker_rss_mb",
        "episode_wall_s",
        "p0_env_steps",
        "max_p1_microsteps",
        "approx_engine_actions_per_p0_step",
        "n_actions",
    ]
    collected: dict[str, list[float]] = {k: [] for k in keys_num}
    for row in rows:
        for k in keys_num:
            v = _get_float(row, k)
            if v is not None and math.isfinite(v):
                collected[k].append(v)

    ratio_vals: list[float] = []
    for row in rows:
        w0 = _get_float(row, "wall_p0_s")
        w1 = _get_float(row, "wall_p1_s")
        if w0 is None or w1 is None or not math.isfinite(w0) or not math.isfinite(w1):
            continue
        if w0 <= 0:
            continue
        ratio_vals.append(w1 / w0)

    ratio_median = None
    ratio_p95 = None
    if ratio_vals:
        s = sorted(ratio_vals)
        ratio_median = _pct_sorted(s, 50.0) or statistics.median(s)
        ratio_p95 = _pct_sorted(s, 95.0)

    by_opp: dict[str, Any] = defaultdict(lambda: {"episode_wall_s": [], "p0_env_steps": []})
    opp_counts: Counter[str] = Counter()
    for row in rows:
        ot = str(row.get("opponent_type", "<missing>"))
        opp_counts[ot] += 1
        ew = _get_float(row, "episode_wall_s")
        pe = _get_float(row, "p0_env_steps")
        if ew is not None:
            by_opp[ot]["episode_wall_s"].append(ew)
        if pe is not None:
            by_opp[ot]["p0_env_steps"].append(pe)
    by_opp_med: dict[str, Any] = {}
    for ot, d in sorted(by_opp.items()):
        by_opp_med[ot] = {
            "episode_wall_s_median": statistics.median(d["episode_wall_s"])
            if d["episode_wall_s"]
            else None,
            "p0_env_steps_median": statistics.median(d["p0_env_steps"])
            if d["p0_env_steps"]
            else None,
            "n": opp_counts[ot],
        }

    # first_p0_capture_p0_step
    cap_step_vals: list[float] = []
    cap_step_nulls = 0
    for row in rows:
        v = row.get("first_p0_capture_p0_step")
        if v is None:
            cap_step_nulls += 1
        else:
            try:
                cap_step_vals.append(float(v))
            except (TypeError, ValueError):
                cap_step_nulls += 1

    captures_p0: list[float] = []
    for row in rows:
        v = _get_float(row, "captures_completed_p0")
        if v is not None:
            captures_p0.append(v)

    greedy_mix: Counter[str] = Counter()
    for row in rows:
        gm = row.get("learner_greedy_mix")
        greedy_mix[str(gm) if gm is not None else "<null>"] += 1

    teacher: list[float] = []
    for row in rows:
        v = _get_float(row, "learner_teacher_overrides")
        if v is not None:
            teacher.append(v)

    # Verdict
    if ratio_median is None:
        verdict = "insufficient data"
    elif ratio_median > 1.1:
        verdict = "P1 dominant"
    elif ratio_median < 0.9:
        verdict = "P0 dominant"
    else:
        verdict = "balanced"

    return {
        "row_count": len(rows),
        "log_schema_version_distribution": dict(schema_counts),
        "numeric": {k: _summarize_numeric(collected[k]) for k in keys_num},
        "wall_p1_over_wall_p0_ratio": {
            "count": len(ratio_vals),
            "median": ratio_median,
            "p95": ratio_p95,
        },
        "by_opponent_type_median": by_opp_med,
        "first_p0_capture_p0_step": {
            "median": _summarize_numeric(cap_step_vals)["median"] if cap_step_vals else None,
            "null_count": cap_step_nulls,
            "valid_count": len(cap_step_vals),
        },
        "captures_completed_p0": _summarize_numeric(captures_p0),
        "learner_greedy_mix_distribution": dict(greedy_mix),
        "learner_teacher_overrides": _summarize_numeric(teacher),
        "p1_p0_dominance_verdict": verdict,
    }


def print_report(payload: dict[str, Any]) -> None:
    print(f"row_count: {payload['row_count']}")
    print("log_schema_version:", payload["log_schema_version_distribution"])
    for key in [
        "wall_p0_s",
        "wall_p1_s",
        "worker_rss_mb",
        "episode_wall_s",
        "p0_env_steps",
        "max_p1_microsteps",
        "approx_engine_actions_per_p0_step",
        "n_actions",
    ]:
        s = payload["numeric"][key]
        print(
            f"{key:40} n={s['count']:4} med={s['median']} p25={s['p25']} "
            f"p75={s['p75']} p95={s['p95']} max={s['max']}"
        )
    r = payload["wall_p1_over_wall_p0_ratio"]
    print(f"wall_p1_s/wall_p0_s ratio: n={r['count']} median={r['median']} p95={r['p95']}")
    print("by_opponent_type (median episode_wall_s, median p0_env_steps, n):")
    for ot, b in payload["by_opponent_type_median"].items():
        print(
            f"  {ot!r}: episode_wall_s_median={b['episode_wall_s_median']}, "
            f"p0_env_steps_median={b['p0_env_steps_median']}, n={b['n']}"
        )
    c = payload["first_p0_capture_p0_step"]
    print(
        f"first_p0_capture_p0_step: median={c['median']} nulls={c['null_count']} valid={c['valid_count']}"
    )
    cap = payload["captures_completed_p0"]
    print(
        f"captures_completed_p0: n={cap['count']} med={cap['median']} p25={cap['p25']} p95={cap['p95']}"
    )
    print("learner_greedy_mix distribution:", payload["learner_greedy_mix_distribution"])
    te = payload["learner_teacher_overrides"]
    print(
        f"learner_teacher_overrides: n={te['count']} med={te['median']} p25={te['p25']} p95={te['p95']}"
    )
    v = payload["p1_p0_dominance_verdict"]
    print()
    print("--- Verdict (from median wall_p1_s / wall_p0_s) ---")
    print(f"  {v}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log-path",
        type=Path,
        default=Path("logs/game_log.jsonl"),
        help="Path to game_log.jsonl (default: logs/game_log.jsonl)",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write structured JSON summary to this path",
    )
    args = ap.parse_args()
    log_path: Path = args.log_path
    if not log_path.is_file():
        raise SystemExit(f"not found: {log_path.resolve()}")
    rows = _read_rows(log_path)
    payload = mine(rows)
    payload["source_log"] = str(log_path.resolve())
    print_report(payload)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.out_json.resolve()}")


if __name__ == "__main__":
    main()
