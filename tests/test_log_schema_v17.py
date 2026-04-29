"""Schema 1.11 contract: 1.10 fields plus env-step-cap synthetic ``winner`` / ``win_reason``.

Plan: ``.cursor/plans/train.py_fps_campaign_c26ce6d4.plan.md`` — Phase 10/11
logging prerequisites. Without ``machine_id`` on every row the orchestrator's
per-machine rolling-window logic in 10g/10h/11d is impossible; without
``terrain_usage_p0`` the MCTS health gate (11d) has no terrain signal.

Schema 1.8 added explicit episode-end flags. Schema 1.9 adds ``learner_seat``,
``reward_mode``, ``arch_version``, ``opponent_sampler``; ``agent_plays`` mirrors
``learner_seat``. Schema 1.10 adds optional ``tie_breaker_property_count`` (int)
for step-cap property-lead partial wins. Schema 1.11 logs P0 vs P1 property
tiebreak for env truncation when the engine never set a winner
(``env_step_cap_tie`` / ``env_step_cap_tiebreak``).

The writer (``_append_game_log_line``) stamps ``machine_id`` from the env
var ``AWBW_MACHINE_ID`` at write time so a solo dev box without the var
emits ``None`` (degrades cleanly) and the two write paths in that function
cannot drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "data" / "gl_map_pool.json"
MAP_ID = 123858


def _single_map_pool() -> list[dict]:
    with open(POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    return [next(m for m in pool if m.get("map_id") == MAP_ID)]


def _drive_one_log_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    """Build a real AWBWEnv, reset it, mark its state finished, and let the
    real ``_log_finished_game`` writer land one row into a temp game_log file.

    We keep the real writer (not a monkeypatch) so the writer-boundary
    fields (``game_id``, ``machine_id``) are exercised end-to-end.
    """
    log_path = tmp_path / "game_log.jsonl"
    monkeypatch.setattr("rl.env.GAME_LOG_PATH", log_path)
    # Force the in-process counter path so the test does not depend on a
    # shared SQLite session DB possibly set in the host environment.
    monkeypatch.delenv("AWBW_SESSION_GAME_COUNTER_DB", raising=False)

    from rl.env import AWBWEnv

    env = AWBWEnv(
        map_pool=_single_map_pool(),
        opponent_policy=None,
        co_p0=1,
        co_p1=1,
        tier_name="T3",
    )
    env.reset(seed=0)
    # _log_finished_game only reads attributes off self.state; the engine
    # does not have to have actually terminated for this test.
    env.state.done = True
    env.state.winner = 0
    env.state.win_reason = "test"
    env._log_finished_game()

    raw = log_path.read_text(encoding="utf-8").strip()
    assert raw, "writer produced no rows"
    rows = [json.loads(chunk) for chunk in raw.split("\n\n") if chunk.strip()]
    assert len(rows) == 1, f"expected exactly one row, got {len(rows)}"
    return rows[0]


def test_log_schema_v17_required_fields_machine_id_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("AWBW_MACHINE_ID", raising=False)
    row = _drive_one_log_record(monkeypatch, tmp_path)

    assert row["log_schema_version"] == "1.14"
    assert len(row.get("alive_unit_count", [])) == 2
    assert len(row.get("army_value", [])) == 2
    assert row.get("tie_breaker_property_count") is None

    assert row.get("learner_seat") in (0, 1)
    assert row.get("agent_plays") == row.get("learner_seat")
    assert row.get("reward_mode") in ("phi", "level")
    assert isinstance(row.get("arch_version"), str) and row["arch_version"]
    assert row.get("opponent_sampler") in ("pfsp", "uniform")

    assert "machine_id" in row, "writer must always stamp machine_id"
    assert row["machine_id"] is None, (
        "with AWBW_MACHINE_ID unset, machine_id must be None (not the empty "
        "string, not 'pc-b' — the orchestrator distinguishes 'unknown writer' "
        "from a real id)"
    )

    assert "terrain_usage_p0" in row
    val = row["terrain_usage_p0"]
    assert isinstance(val, float), f"terrain_usage_p0 must be float, got {type(val)}"
    assert 0.0 <= val <= 1.0, f"terrain_usage_p0 out of range: {val!r}"

    assert row.get("terminated") is True
    assert row.get("truncated") is False
    assert row.get("truncation_reason") is None
    assert row.get("phi_enemy_property_captures") == 0


def test_log_schema_v17_machine_id_from_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AWBW_MACHINE_ID", "test-pc")
    row = _drive_one_log_record(monkeypatch, tmp_path)

    assert row["machine_id"] == "test-pc"
    assert row["log_schema_version"] == "1.14"
    assert isinstance(row["terrain_usage_p0"], float)
