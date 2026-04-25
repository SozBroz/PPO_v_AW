"""Sidecar JSONLs + stderr summary for --enable-state-mismatch (gold drift surfacing)."""
from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import desync_audit  # noqa: E402
from tools.desync_audit import (  # noqa: E402
    AuditRow,
    CLS_STATE_MISMATCH_FUNDS,
    CLS_STATE_MISMATCH_MULTI,
    CLS_STATE_MISMATCH_UNITS,
    _print_silent_drift_summary,
    _state_mismatch_sidecar_paths,
    _write_state_mismatch_sidecars,
)


def _row(
    *,
    games_id: int,
    cls: str,
    message: str,
) -> AuditRow:
    return AuditRow(
        games_id=games_id,
        map_id=1,
        tier="T1",
        co_p0_id=1,
        co_p1_id=1,
        matchup="a vs b",
        zip_path=str(ROOT / "replays" / "x.zip"),
        status="first_divergence",
        cls=cls,
        exception_type="StateMismatchError",
        message=message,
        approx_day=3,
        approx_action_kind="End",
        approx_envelope_index=5,
        envelopes_total=10,
        envelopes_applied=6,
        actions_applied=100,
        state_mismatch={"env_i": 5},
    )


def test_sidecar_paths_named_from_register_stem(tmp_path: Path) -> None:
    reg = tmp_path / "my_run.jsonl"
    paths = _state_mismatch_sidecar_paths(reg)
    assert paths[CLS_STATE_MISMATCH_FUNDS] == tmp_path / "my_run_state_mismatch_funds.jsonl"
    assert paths[CLS_STATE_MISMATCH_UNITS] == tmp_path / "my_run_state_mismatch_units.jsonl"


def test_write_state_mismatch_sidecars_splits_classes(tmp_path: Path) -> None:
    reg = tmp_path / "batch.jsonl"
    rows = [
        _row(games_id=111, cls=CLS_STATE_MISMATCH_FUNDS, message="P0 funds engine=1 php=2"),
        _row(games_id=222, cls=CLS_STATE_MISMATCH_FUNDS, message="P1 drift"),
        _row(games_id=333, cls=CLS_STATE_MISMATCH_UNITS, message="tile mismatch"),
        _row(games_id=444, cls=CLS_STATE_MISMATCH_MULTI, message="funds+hp"),
        _row(games_id=555, cls=desync_audit.CLS_OK, message=""),
    ]
    _write_state_mismatch_sidecars(reg, rows)
    funds_path = tmp_path / "batch_state_mismatch_funds.jsonl"
    units_path = tmp_path / "batch_state_mismatch_units.jsonl"
    multi_path = tmp_path / "batch_state_mismatch_multi.jsonl"
    assert funds_path.is_file()
    fl = [json.loads(l) for l in funds_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert {r["games_id"] for r in fl} == {111, 222}
    ul = [json.loads(l) for l in units_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [r["games_id"] for r in ul] == [333]
    ml = [json.loads(l) for l in multi_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [r["games_id"] for r in ml] == [444]


def test_print_silent_drift_summary_lists_funds_gids(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    reg = tmp_path / "out.jsonl"
    reg.touch()
    rows = [
        _row(games_id=9, cls=CLS_STATE_MISMATCH_FUNDS, message="small"),
        _row(games_id=1, cls=CLS_STATE_MISMATCH_FUNDS, message="first"),
    ]
    counts = {
        CLS_STATE_MISMATCH_FUNDS: 2,
        CLS_STATE_MISMATCH_UNITS: 3,
        CLS_STATE_MISMATCH_MULTI: 1,
        desync_audit.CLS_STATE_MISMATCH_INVESTIGATE: 0,
    }
    _print_silent_drift_summary(reg, rows, counts)
    err = capsys.readouterr().err
    assert "SILENT DRIFT" in err
    assert "gold_drift (state_mismatch_funds): 2" in err
    assert "gid=1" in err and "gid=9" in err  # sorted by games_id


def test_main_fail_on_without_enable_returns_1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["desync_audit", "--fail-on-state-mismatch-funds"])
    assert desync_audit.main() == 1
