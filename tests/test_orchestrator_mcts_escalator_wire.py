# -*- coding: utf-8 -*-
"""Phase 11 Slice D — orchestrator wiring for ``tools/mcts_escalator``.

These tests pin the contract for ``run_mcts_escalator``:

* skip silently when ``--mcts-mode`` is missing or ``"off"``;
* refuse to escalate without a baseline (audit-only ``mcts_baseline_missing``);
* in DOUBLE the new ``--mcts-sims`` is only written when ``auto_apply`` AND
  the host gate would allow it; otherwise the row is audit-only;
* in DROP_TO_OFF the proposed args are switched to ``--mcts-mode off`` AND
  the health-gate hysteresis streak is reset to 0;
* ``mcts_skip_host`` fires when the orchestrator is asked to escalate the
  host machine without ``--enable-mcts-here``;
* ``logs/mcts_escalator.jsonl`` always gains a row when the escalator runs.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parents[1]


def _fo():
    p = REPO / "scripts" / "fleet_orchestrator.py"
    name = "fleet_orchestrator_escalator_wire"
    spec = importlib.util.spec_from_file_location(name, p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


fo = _fo()
FleetOrchestrator = fo.FleetOrchestrator


def _orch(tmp_path: Path, *, pools: list[str], **kw: object) -> object:
    defaults: dict = {
        "shared_root": tmp_path,
        "pools": pools,
        "dry_run": False,
        "repo_root": tmp_path,
        "keep_newest": 8,
        "keep_top_winrate": 12,
        "keep_diversity": 4,
        "curator_min_age_minutes": 0.0,
        "map_id": 123858,
        "tier": "T3",
        "co_p0": 1,
        "co_p1": 1,
        "games_first_seat": 4,
        "games_second_seat": 3,
        "reload_margin": 0.25,
        "reload_consecutive": 2,
        "stuck_threshold_seconds": 1200.0,
        "audit_log": tmp_path / "audit.jsonl",
        "state_file": tmp_path / "state.json",
        "eval_timeout_seconds": 1800.0,
        "eval_seed": 0,
        "auto_apply": False,
        "apply_cooldown_s": 600.0,
        "train_pid_file_template": "fleet/{machine_id}/train.pid",
        "train_launch_cmd_file_template": "fleet/{machine_id}/train_launch_cmd.json",
        "curriculum_enabled": False,
        "mcts_health_refresh_every_ticks": 0,
        "host_machine_id": "no-such-host",
        "enable_mcts_here": False,
        "mcts_gate_required_consecutive": 1,
    }
    defaults.update(kw)
    return FleetOrchestrator(**defaults)  # type: ignore[misc]


def _write_proposed(
    tmp_path: Path,
    mid: str,
    *,
    mode: str | None = "eval_only",
    sims: int = 16,
    auto_apply: bool = True,
) -> Path:
    d = tmp_path / "fleet" / mid
    d.mkdir(parents=True, exist_ok=True)
    args: dict = {"--n-envs": 4, "--n-steps": 512, "--batch-size": 256}
    if mode is not None:
        args["--mcts-mode"] = mode
        args["--mcts-sims"] = int(sims)
    doc = {"machine_id": mid, "args": args, "auto_apply": auto_apply}
    p = d / "proposed_args.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _write_baseline(tmp_path: Path, mid: str, *, wr: float = 0.40) -> None:
    from tools.mcts_baseline import MctsOffBaseline, write_baseline

    obj = MctsOffBaseline(
        schema_version=1,
        machine_id=mid,
        captured_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        checkpoint_zip=f"checkpoints/pool/{mid}/latest.zip",
        checkpoint_zip_sha256="c" * 64,
        games_decided=200,
        winrate_vs_pool=wr,
        mcts_mode="off",
        source="tools/capture_mcts_baseline.py",
    )
    write_baseline(obj, tmp_path / "fleet" / mid)


def _cycle(*, sims: int, wr: float = 0.55, baseline: float = 0.40,
           ev: float = 0.7, games: int = 250, desyncs: int = 0):
    from tools.mcts_escalator import EscalatorCycleResult

    return EscalatorCycleResult(
        cycle_ts=1.0,
        sims=int(sims),
        winrate_vs_pool=float(wr),
        mcts_off_baseline=float(baseline),
        games_decided=int(games),
        explained_variance=float(ev),
        engine_desyncs_in_cycle=int(desyncs),
        wall_s_per_decision_p50=0.0,
    )


# ---------------------------------------------------------------------------
# Skipping rules
# ---------------------------------------------------------------------------


def test_escalator_skipped_when_mcts_mode_off(tmp_path: Path) -> None:
    mid = "m-aux"
    _write_proposed(tmp_path, mid, mode="off")
    _write_baseline(tmp_path, mid)
    o = _orch(tmp_path, pools=[mid])
    decs = o.run_mcts_escalator(o.read_fleet_state())
    assert decs == []


def test_escalator_skipped_when_no_mcts_mode_key(tmp_path: Path) -> None:
    mid = "m-aux"
    _write_proposed(tmp_path, mid, mode=None)
    _write_baseline(tmp_path, mid)
    o = _orch(tmp_path, pools=[mid])
    decs = o.run_mcts_escalator(o.read_fleet_state())
    assert decs == []


def test_escalator_baseline_missing_emits_audit_only(tmp_path: Path) -> None:
    mid = "m-aux"
    _write_proposed(tmp_path, mid, mode="eval_only", sims=16)
    o = _orch(tmp_path, pools=[mid], auto_apply=True)
    decs = o.run_mcts_escalator(o.read_fleet_state())
    assert len(decs) == 1
    d = decs[0]
    assert d.kind == "mcts_baseline_missing"
    assert d.applied is False
    assert "capture_mcts_baseline" in d.reason
    # Proposed args untouched
    doc = json.loads(
        (tmp_path / "fleet" / mid / "proposed_args.json").read_text(encoding="utf-8")
    )
    assert doc["args"]["--mcts-sims"] == 16


def test_escalator_baseline_stale_treated_as_missing(tmp_path: Path) -> None:
    mid = "m-aux"
    _write_proposed(tmp_path, mid, mode="eval_only", sims=16)
    from tools.mcts_baseline import MctsOffBaseline, write_baseline

    old = (
        datetime.now(timezone.utc) - timedelta(hours=24 * 30)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_baseline(
        MctsOffBaseline(
            schema_version=1,
            machine_id=mid,
            captured_at=old,
            checkpoint_zip=f"checkpoints/pool/{mid}/latest.zip",
            checkpoint_zip_sha256="d" * 64,
            games_decided=200,
            winrate_vs_pool=0.40,
            mcts_mode="off",
            source="tools/capture_mcts_baseline.py",
        ),
        tmp_path / "fleet" / mid,
    )
    o = _orch(tmp_path, pools=[mid], auto_apply=True)
    decs = o.run_mcts_escalator(o.read_fleet_state())
    assert any(d.kind == "mcts_baseline_missing" for d in decs)


def test_escalator_no_data_when_build_cycle_returns_none(tmp_path: Path) -> None:
    mid = "m-aux"
    _write_proposed(tmp_path, mid, mode="eval_only", sims=16)
    _write_baseline(tmp_path, mid)
    o = _orch(tmp_path, pools=[mid], auto_apply=True)
    with patch.object(fo._mcts_eval_summary, "build_cycle_result", return_value=None):
        decs = o.run_mcts_escalator(o.read_fleet_state())
    assert len(decs) == 1
    assert decs[0].kind == "mcts_escalator_no_data"
    assert decs[0].applied is False


def test_escalator_host_skipped_without_enable_flag(tmp_path: Path) -> None:
    mid = "pc-b"
    _write_proposed(tmp_path, mid, mode="eval_only", sims=16)
    _write_baseline(tmp_path, mid)
    o = _orch(
        tmp_path,
        pools=[mid],
        auto_apply=True,
        host_machine_id=mid,
        enable_mcts_here=False,
    )
    with patch.object(
        fo._mcts_eval_summary,
        "build_cycle_result",
        return_value=_cycle(sims=16, ev=0.8, wr=0.7, baseline=0.4),
    ):
        decs = o.run_mcts_escalator(o.read_fleet_state())
    assert len(decs) == 1
    assert decs[0].kind == "mcts_skip_host"
    assert decs[0].applied is False
    # Proposed args not mutated
    doc = json.loads(
        (tmp_path / "fleet" / mid / "proposed_args.json").read_text(encoding="utf-8")
    )
    assert doc["args"]["--mcts-sims"] == 16


# ---------------------------------------------------------------------------
# DOUBLE
# ---------------------------------------------------------------------------


def test_escalator_double_writes_proposed_sims_when_auto_apply(tmp_path: Path) -> None:
    mid = "m-aux"
    _write_proposed(tmp_path, mid, mode="eval_only", sims=16)
    _write_baseline(tmp_path, mid)
    o = _orch(tmp_path, pools=[mid], auto_apply=True)
    # State already has 1 cycle warm-up so the next pass triggers DOUBLE.
    from tools.mcts_escalator import EscalatorState, write_state, default_state_path

    write_state(
        default_state_path(mid, tmp_path),
        EscalatorState(
            current_sims=16,
            mcts_off_baseline=0.40,
            last_double_at_ts=0.0,
            sims_plateau_at=None,
            cycles_at_current_sims=1,
        ),
    )
    with patch.object(
        fo._mcts_eval_summary,
        "build_cycle_result",
        return_value=_cycle(sims=16, wr=0.7, baseline=0.4, ev=0.8, games=300),
    ):
        decs = o.run_mcts_escalator(o.read_fleet_state())
    doubles = [d for d in decs if d.kind == "mcts_escalator_double"]
    assert len(doubles) == 1
    assert doubles[0].applied is True
    assert doubles[0].details["proposed_sims"] == 32
    doc = json.loads(
        (tmp_path / "fleet" / mid / "proposed_args.json").read_text(encoding="utf-8")
    )
    assert int(doc["args"]["--mcts-sims"]) == 32
    assert doc["args"]["--mcts-mode"] == "eval_only"


def test_escalator_double_audit_only_when_no_auto_apply(tmp_path: Path) -> None:
    mid = "m-aux"
    _write_proposed(tmp_path, mid, mode="eval_only", sims=16)
    _write_baseline(tmp_path, mid)
    o = _orch(tmp_path, pools=[mid], auto_apply=False)
    from tools.mcts_escalator import EscalatorState, write_state, default_state_path

    write_state(
        default_state_path(mid, tmp_path),
        EscalatorState(
            current_sims=16,
            mcts_off_baseline=0.40,
            last_double_at_ts=0.0,
            sims_plateau_at=None,
            cycles_at_current_sims=1,
        ),
    )
    with patch.object(
        fo._mcts_eval_summary,
        "build_cycle_result",
        return_value=_cycle(sims=16, wr=0.7, baseline=0.4, ev=0.8, games=300),
    ):
        decs = o.run_mcts_escalator(o.read_fleet_state())
    doubles = [d for d in decs if d.kind == "mcts_escalator_double"]
    assert len(doubles) == 1
    assert doubles[0].applied is False
    doc = json.loads(
        (tmp_path / "fleet" / mid / "proposed_args.json").read_text(encoding="utf-8")
    )
    assert int(doc["args"]["--mcts-sims"]) == 16


# ---------------------------------------------------------------------------
# DROP_TO_OFF
# ---------------------------------------------------------------------------


def test_escalator_drop_to_off_resets_hysteresis_and_clears_sims(tmp_path: Path) -> None:
    mid = "m-aux"
    _write_proposed(tmp_path, mid, mode="eval_only", sims=32)
    _write_baseline(tmp_path, mid)
    o = _orch(tmp_path, pools=[mid], auto_apply=True)
    o._mcts_pass_streak_by_machine[mid] = 5

    with patch.object(
        fo._mcts_eval_summary,
        "build_cycle_result",
        return_value=_cycle(sims=32, wr=0.4, baseline=0.4, ev=0.5, desyncs=1),
    ):
        decs = o.run_mcts_escalator(o.read_fleet_state())
    drops = [d for d in decs if d.kind == "mcts_escalator_drop_to_off"]
    assert len(drops) == 1
    assert drops[0].applied is True
    assert "ALERT" in drops[0].reason
    doc = json.loads(
        (tmp_path / "fleet" / mid / "proposed_args.json").read_text(encoding="utf-8")
    )
    assert doc["args"]["--mcts-mode"] == "off"
    assert "--mcts-sims" not in doc["args"]
    assert o._mcts_pass_streak_by_machine[mid] == 0


def test_escalator_drop_to_off_audit_only_when_no_auto_apply(tmp_path: Path) -> None:
    mid = "m-aux"
    _write_proposed(tmp_path, mid, mode="eval_only", sims=32)
    _write_baseline(tmp_path, mid)
    o = _orch(tmp_path, pools=[mid], auto_apply=False)
    o._mcts_pass_streak_by_machine[mid] = 5
    with patch.object(
        fo._mcts_eval_summary,
        "build_cycle_result",
        return_value=_cycle(sims=32, wr=0.4, baseline=0.4, ev=0.5, desyncs=1),
    ):
        decs = o.run_mcts_escalator(o.read_fleet_state())
    drops = [d for d in decs if d.kind == "mcts_escalator_drop_to_off"]
    assert len(drops) == 1
    assert drops[0].applied is False
    # Hysteresis streak NOT reset when we did not actually drop the proposed args.
    assert o._mcts_pass_streak_by_machine[mid] == 5


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


def test_escalator_appends_cycle_log_jsonl(tmp_path: Path) -> None:
    mid = "m-aux"
    _write_proposed(tmp_path, mid, mode="eval_only", sims=16)
    _write_baseline(tmp_path, mid)
    o = _orch(tmp_path, pools=[mid], auto_apply=True)
    with patch.object(
        fo._mcts_eval_summary,
        "build_cycle_result",
        return_value=_cycle(sims=16, wr=0.55, baseline=0.40, ev=0.7),
    ):
        o.run_mcts_escalator(o.read_fleet_state())
    log = tmp_path / "logs" / "mcts_escalator.jsonl"
    assert log.is_file()
    lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["sims"] == 16
    assert row["winrate_vs_pool"] == pytest.approx(0.55)
    assert row["mcts_off_baseline"] == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# Phase 11d EV scrape — mcts_ev_unavailable DecKind
# ---------------------------------------------------------------------------


def test_escalator_emits_ev_unavailable_when_scrape_returns_none(
    tmp_path: Path,
) -> None:
    mid = "m-aux"
    _write_proposed(tmp_path, mid, mode="eval_only", sims=16)
    _write_baseline(tmp_path, mid)
    o = _orch(tmp_path, pools=[mid], auto_apply=True)
    with patch.object(
        fo._mcts_eval_summary,
        "build_cycle_result",
        return_value=_cycle(sims=16, wr=0.55, baseline=0.40, ev=0.0),
    ), patch.object(
        fo._mcts_eval_summary, "latest_explained_variance", return_value=None
    ):
        decs = o.run_mcts_escalator(o.read_fleet_state())
    kinds = [d.kind for d in decs]
    assert "mcts_ev_unavailable" in kinds
    ev_row = next(d for d in decs if d.kind == "mcts_ev_unavailable")
    assert ev_row.applied is False
    assert "train/explained_variance" in ev_row.details["scalar_tag"]
    assert "no recent train/explained_variance samples" in ev_row.reason
    # The cycle still ran (escalator emitted a hold/double row alongside it).
    assert any(d.kind.startswith("mcts_escalator_") for d in decs)


def test_escalator_appends_cycle_log_when_ev_unavailable(tmp_path: Path) -> None:
    mid = "m-aux"
    _write_proposed(tmp_path, mid, mode="eval_only", sims=16)
    _write_baseline(tmp_path, mid)
    o = _orch(tmp_path, pools=[mid], auto_apply=True)
    with patch.object(
        fo._mcts_eval_summary,
        "build_cycle_result",
        return_value=_cycle(sims=16, wr=0.55, baseline=0.40, ev=0.0),
    ), patch.object(
        fo._mcts_eval_summary, "latest_explained_variance", return_value=None
    ):
        o.run_mcts_escalator(o.read_fleet_state())
    log = tmp_path / "logs" / "mcts_escalator.jsonl"
    assert log.is_file()
    lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1


def test_escalator_no_ev_unavailable_when_scrape_returns_value(tmp_path: Path) -> None:
    mid = "m-aux"
    _write_proposed(tmp_path, mid, mode="eval_only", sims=16)
    _write_baseline(tmp_path, mid)
    o = _orch(tmp_path, pools=[mid], auto_apply=True)
    with patch.object(
        fo._mcts_eval_summary,
        "build_cycle_result",
        return_value=_cycle(sims=16, wr=0.55, baseline=0.40, ev=0.7),
    ), patch.object(
        fo._mcts_eval_summary, "latest_explained_variance", return_value=0.7
    ):
        decs = o.run_mcts_escalator(o.read_fleet_state())
    assert all(d.kind != "mcts_ev_unavailable" for d in decs)


def test_escalator_per_machine_exception_isolated(tmp_path: Path) -> None:
    """One bad machine must not prevent other machines from being processed."""
    bad = "m-bad"
    good = "m-good"
    _write_proposed(tmp_path, bad, mode="eval_only", sims=16)
    _write_proposed(tmp_path, good, mode="eval_only", sims=16)
    _write_baseline(tmp_path, bad)
    _write_baseline(tmp_path, good)
    o = _orch(tmp_path, pools=[bad, good], auto_apply=True)

    def _stub(mid, *_a, **_kw):
        if mid == bad:
            raise RuntimeError("boom")
        return _cycle(sims=16, wr=0.55, baseline=0.40, ev=0.7)

    with patch.object(fo._mcts_eval_summary, "build_cycle_result", side_effect=_stub):
        decs = o.run_mcts_escalator(o.read_fleet_state())
    by_mid = {d.machine_id: d for d in decs}
    assert by_mid[bad].kind == "mcts_escalator_no_data"
    assert "boom" in by_mid[bad].details["error"]
    assert by_mid[good].kind == "mcts_escalator_hold"
