"""Bottleneck summary from logs/fps_diag.jsonl (tools.fps_diag_metrics)."""

from __future__ import annotations

from tools.fps_diag_metrics import (
    format_bottleneck_report,
    summarize_fps_diag_bottleneck,
)


def test_bottleneck_env_collect_dominant() -> None:
    text = "\n".join(
        [
            json_line(
                {
                    "env_collect_s": 0.8,
                    "ppo_update_s": 0.2,
                    "worker_step_time_p99_max_s": 0.1,
                    "worker_step_time_p99_min_s": 0.05,
                }
            ),
            json_line(
                {
                    "env_collect_s": 0.9,
                    "ppo_update_s": 0.1,
                }
            ),
        ]
    )
    s = summarize_fps_diag_bottleneck(text)
    assert s["verdict"] == "env_collect"
    assert s["median_collect_fraction"] is not None
    assert s["median_collect_fraction"] > 0.55
    assert "env_collect" in format_bottleneck_report(s)


def test_bottleneck_ppo_update_dominant() -> None:
    text = json_line(
        {"env_collect_s": 0.2, "ppo_update_s": 0.9},
    )
    s = summarize_fps_diag_bottleneck(text)
    assert s["verdict"] == "ppo_update"


def test_bottleneck_mixed() -> None:
    text = json_line({"env_collect_s": 0.5, "ppo_update_s": 0.5})
    s = summarize_fps_diag_bottleneck(text)
    assert s["verdict"] == "mixed"


def test_bottleneck_insufficient() -> None:
    text = json_line({"env_steps_per_s_total": 100.0})
    s = summarize_fps_diag_bottleneck(text)
    assert s["verdict"] == "insufficient_data"


def json_line(obj: dict) -> str:
    import json

    return json.dumps(obj)
