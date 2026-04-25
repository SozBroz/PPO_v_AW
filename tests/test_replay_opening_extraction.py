"""Human demo row iterator: seats and max_turn."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.human_demo_rows import iter_demo_rows_from_trace_record


def test_trace_max_turn_filters_by_awbw_turn() -> None:
    p = Path(__file__).resolve().parents[1] / "replays" / "272176.trace.json"
    if not p.is_file():
        pytest.skip("replays/272176.trace.json not in tree")
    with open(p, encoding="utf-8") as f:
        rec = json.load(f)
    rows = list(
        iter_demo_rows_from_trace_record(
            rec,
            seats=(0, 1),
            max_turn=1,
            opening_only=True,
        )
    )
    assert rows
    assert all(int(r.get("awbw_turn", 99)) <= 1 for r in rows)
