# -*- coding: utf-8 -*-
"""Phase 11d schema_version 2 contract for ``tools.desync_audit.AuditRow``.

Pins the new ``machine_id`` / ``recorded_at`` attribution fields the
MCTS escalator (``tools/mcts_eval_summary._count_recent_desyncs``)
filters on. If any of these break, a single fleet-wide desync will
again smear DROP_TO_OFF across every machine.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.desync_audit import (  # noqa: E402
    DESYNC_REGISTER_SCHEMA_VERSION,
    AuditRow,
)


_ISO_Z_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _row(**overrides) -> AuditRow:
    """Minimal valid AuditRow; overrides win."""
    base = dict(
        games_id=1,
        map_id=1,
        tier="T1",
        co_p0_id=1,
        co_p1_id=1,
        matchup="a vs b",
        zip_path="x.zip",
        status="ok",
        cls="ok",
        exception_type="",
        message="",
        approx_day=None,
        approx_action_kind=None,
        approx_envelope_index=None,
        envelopes_total=0,
        envelopes_applied=0,
        actions_applied=0,
    )
    base.update(overrides)
    return AuditRow(**base)


def test_schema_version_constant_is_two() -> None:
    assert DESYNC_REGISTER_SCHEMA_VERSION == 2


def test_to_json_emits_schema_version_two() -> None:
    j = _row().to_json()
    assert j["schema_version"] == 2


def test_to_json_emits_machine_id_and_recorded_at_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AWBW_MACHINE_ID", raising=False)
    j = _row().to_json()
    assert "machine_id" in j
    assert "recorded_at" in j
    assert j["machine_id"] is None
    assert isinstance(j["recorded_at"], str)
    assert _ISO_Z_RE.match(j["recorded_at"]) is not None


def test_recorded_at_parses_as_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWBW_MACHINE_ID", raising=False)
    j = _row().to_json()
    s = j["recorded_at"]
    assert s.endswith("Z")
    parsed = datetime.fromisoformat(s.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def test_machine_id_falls_back_to_env_when_field_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWBW_MACHINE_ID", "pc-b")
    j = _row().to_json()
    assert j["machine_id"] == "pc-b"


def test_machine_id_explicit_field_wins_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWBW_MACHINE_ID", "pc-b")
    j = _row(machine_id="explicit-host").to_json()
    assert j["machine_id"] == "explicit-host"


def test_machine_id_emits_null_when_field_and_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AWBW_MACHINE_ID", raising=False)
    j = _row().to_json()
    assert j["machine_id"] is None


def test_explicit_recorded_at_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWBW_MACHINE_ID", raising=False)
    fixed = "2026-04-23T12:00:00Z"
    j = _row(recorded_at=fixed).to_json()
    assert j["recorded_at"] == fixed


def test_legacy_call_sites_compile_without_new_kwargs() -> None:
    # Defensive: AuditRow constructor must still accept the historical
    # positional/keyword set without machine_id / recorded_at.
    row = AuditRow(
        games_id=42,
        map_id=1,
        tier="T1",
        co_p0_id=1,
        co_p1_id=1,
        matchup="a vs b",
        zip_path="x.zip",
        status="first_divergence",
        cls="engine_bug",
        exception_type="X",
        message="boom",
        approx_day=3,
        approx_action_kind="Move",
        approx_envelope_index=5,
        envelopes_total=10,
        envelopes_applied=4,
        actions_applied=20,
    )
    assert row.machine_id is None
    assert row.recorded_at is None
    j = row.to_json()
    assert j["games_id"] == 42
    assert j["schema_version"] == 2
