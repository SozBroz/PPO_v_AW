"""Unit tests for tools/bench_train_throughput.py (no subprocess)."""

from __future__ import annotations

from tools.fps_diag_metrics import parse_fps_diag_lines, summarize_fps


def test_parse_fps_diag_parses_synthetic_jsonl() -> None:
    text = """
{"iteration": 1, "env_steps_per_s_total": 100.0, "noise": true}
not json
{"iteration": 2, "env_steps_per_s_total": 200}
{"iteration": 3, "env_steps_per_s_total": 0.0}
{"iteration": 4, "env_steps_per_s_total": null}
{"iteration": 5, "env_steps_per_s_total": 400.0}
""".strip()
    vals = parse_fps_diag_lines(text)
    assert vals == [100.0, 200.0, 400.0]

    stats = summarize_fps(vals)
    assert stats["n_samples"] == 3
    assert stats["p50"] == 200.0
    assert stats["median"] == stats["p50"]
    assert stats["p25"] == 150.0
    assert stats["p75"] == 300.0
