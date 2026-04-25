"""
Parse and summarize ``logs/fps_diag.jsonl`` metrics for throughput tooling.

Primary signal: ``env_steps_per_s_total``. Optional fallback: ``env_steps_per_s_collect``.
Async training rows include ``training_backend: \"async\"``; ``ppo_update_s`` is the prior
learner (GPU) step time, and ``env_collect_s`` includes queue wait + batch prep.

Bottleneck summary (``summarize_fps_diag_bottleneck``) compares rollout **collect**
vs **PPO update** wall from the same JSONL rows and classifies straggler spread.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterator


def parse_fps_diag_lines(text: str) -> list[float]:
    """Parse ``env_steps_per_s_total`` from fps_diag-style JSONL *text* (no I/O)."""
    out: list[float] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        raw = row.get("env_steps_per_s_total")
        if raw is None:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0.0:
            out.append(v)
    return out


def parse_fps_diag_collect_lines(text: str) -> list[float]:
    """Parse ``env_steps_per_s_collect`` from fps_diag JSONL *text* (no I/O)."""
    out: list[float] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        raw = row.get("env_steps_per_s_collect")
        if raw is None:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0.0:
            out.append(v)
    return out


def parse_fps_diag_throughput_values(text: str) -> list[float]:
    """
    Values for throughput scoring: totals if any line has them, else collect rates.
    """
    totals = parse_fps_diag_lines(text)
    if totals:
        return totals
    return parse_fps_diag_collect_lines(text)


def percentile(sorted_vals: list[float], p: float) -> float:
    """Linear interpolation percentile, *p* in [0, 100]. *sorted_vals* non-empty."""
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    n = len(sorted_vals)
    k = (n - 1) * (p / 100.0)
    f = math.floor(k)
    c = min(f + 1, n - 1)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def summarize_fps(values: list[float]) -> dict[str, Any]:
    s = sorted(values)
    if not s:
        return {
            "n_samples": 0,
            "p25": None,
            "p50": None,
            "median": None,
            "p75": None,
        }
    return {
        "n_samples": len(s),
        "p25": round(percentile(s, 25), 4),
        "p50": round(percentile(s, 50), 4),
        "median": round(percentile(s, 50), 4),
        "p75": round(percentile(s, 75), 4),
    }


def iter_fps_diag_records(text: str) -> Iterator[dict[str, Any]]:
    """Yield decoded JSON objects from *text* (one JSON object per non-empty line)."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            yield row


def summarize_fps_diag_bottleneck(text: str) -> dict[str, Any]:
    """
    Aggregate ``env_collect_s`` vs ``ppo_update_s`` and worker p99 spread from fps_diag JSONL.

    Returns keys including ``verdict`` in
    ``{"env_collect", "ppo_update", "mixed", "insufficient_data"}``.
    """
    collect_fracs: list[float] = []
    straggler_ratios: list[float] = []
    n_rows = 0
    for row in iter_fps_diag_records(text):
        n_rows += 1
        try:
            ec = row.get("env_collect_s")
            pu = row.get("ppo_update_s")
            if ec is not None and pu is not None:
                ec_f = float(ec)
                pu_f = float(pu)
                tot = ec_f + pu_f
                if tot > 1e-9 and math.isfinite(tot):
                    collect_fracs.append(ec_f / tot)
        except (TypeError, ValueError):
            pass
        try:
            wmax = row.get("worker_step_time_p99_max_s")
            wmin = row.get("worker_step_time_p99_min_s")
            if wmax is not None and wmin is not None:
                a = float(wmax)
                b = float(wmin)
                if a > 0 and b >= 0 and math.isfinite(a) and math.isfinite(b):
                    straggler_ratios.append(a / max(b, 1e-9))
        except (TypeError, ValueError):
            pass

    if not collect_fracs:
        verdict = "insufficient_data"
        median_collect_frac: float | None = None
    else:
        s = sorted(collect_fracs)
        median_collect_frac = percentile(s, 50)
        if median_collect_frac > 0.55:
            verdict = "env_collect"
        elif median_collect_frac < 0.45:
            verdict = "ppo_update"
        else:
            verdict = "mixed"

    straggler_median: float | None = None
    if straggler_ratios:
        straggler_median = percentile(sorted(straggler_ratios), 50)

    return {
        "n_json_rows": n_rows,
        "n_collect_update_pairs": len(collect_fracs),
        "median_collect_fraction": None
        if median_collect_frac is None
        else round(float(median_collect_frac), 4),
        "verdict": verdict,
        "n_straggler_samples": len(straggler_ratios),
        "median_worker_p99_ratio_max_over_min": None
        if straggler_median is None
        else round(float(straggler_median), 4),
    }


def format_bottleneck_report(summary: dict[str, Any]) -> str:
    """Human-readable lines for CLI / logs."""
    lines = [
        f"fps_diag rows (parsed): {summary.get('n_json_rows', 0)}",
        f"collect+update samples: {summary.get('n_collect_update_pairs', 0)}",
        f"median env_collect / (collect+update): {summary.get('median_collect_fraction')}",
        f"bottleneck (heuristic): {summary.get('verdict')}",
        f"worker p99 max/min median ratio: {summary.get('median_worker_p99_ratio_max_over_min')}",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize logs/fps_diag.jsonl throughput and bottleneck.")
    p.add_argument(
        "path",
        nargs="?",
        default="logs/fps_diag.jsonl",
        help="Path to fps_diag.jsonl (default: logs/fps_diag.jsonl)",
    )
    args = p.parse_args()
    path = Path(args.path)
    if not path.is_file():
        raise SystemExit(f"not a file: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    summary = summarize_fps_diag_bottleneck(text)
    print(format_bottleneck_report(summary))
    fps_vals = parse_fps_diag_throughput_values(text)
    if fps_vals:
        print("throughput (env_steps/s):", summarize_fps(fps_vals))


if __name__ == "__main__":
    main()
