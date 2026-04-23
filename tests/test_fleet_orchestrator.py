"""Phase 10e: scripts/fleet_orchestrator (filesystem driver; eval subprocess mocked)."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parents[1]


def _fo():
    p = REPO / "scripts" / "fleet_orchestrator.py"
    name = "fleet_orchestrator"
    spec = importlib.util.spec_from_file_location(name, p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


fo = _fo()
FleetOrchestrator = fo.FleetOrchestrator
main = fo.main
read_proposed_args = fo.read_proposed_args


def _orch(tmp_path: Path, *, pools: list[str], **kw: object) -> object:
    defaults: dict = {
        "shared_root": tmp_path,
        "pools": pools,
        "dry_run": True,
        "repo_root": REPO,
        "keep_newest": 8,
        "keep_top_winrate": 12,
        "keep_diversity": 4,
        "curator_min_age_minutes": 5.0,
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
    }
    defaults.update(kw)
    return FleetOrchestrator(**defaults)  # type: ignore[misc]


def _write_fake_verdict(
    json_out_path: Path, *, candidate_wins: int, baseline_wins: int, **kw: object
) -> None:
    json_out_path.parent.mkdir(parents=True, exist_ok=True)
    total = candidate_wins + baseline_wins
    wr = (candidate_wins / total) if total else 0.0
    payload: dict = {
        "schema_version": 1,
        "candidate_wins": candidate_wins,
        "baseline_wins": baseline_wins,
        "games_decided": total,
        "winrate": wr,
        "map_id": kw.get("map_id", 123858),
        "tier": "T3",
        "co_p0": 1,
        "co_p1": 1,
        "ckpt": kw.get("ckpt", "checkpoint_test.zip"),
        "timestamp": time.time(),
    }
    json_out_path.write_text(json.dumps(payload), encoding="utf-8")


def test_read_fleet_state_collects_machines(tmp_path: Path) -> None:
    (tmp_path / "fleet" / "pc-b").mkdir(parents=True)
    (tmp_path / "fleet" / "keras-aux").mkdir(parents=True)
    now = time.time()
    (tmp_path / "fleet" / "pc-b" / "status.json").write_text(
        json.dumps({"last_poll": now, "role": "auxiliary", "machine_id": "pc-b"}),
        encoding="utf-8",
    )
    (tmp_path / "fleet" / "keras-aux" / "status.json").write_text(
        json.dumps(
            {
                "last_poll": now,
                "role": "auxiliary",
                "machine_id": "keras-aux",
            }
        ),
        encoding="utf-8",
    )
    for sub in ("pc-b", "keras-aux"):
        pdir = tmp_path / "checkpoints" / "pool" / sub
        pdir.mkdir(parents=True)
        (pdir / "checkpoint_0000.zip").write_bytes(b"z")
    o = _orch(tmp_path, pools=["pc-b", "keras-aux"])
    st = o.read_fleet_state()
    assert "pc-b" in st.machines
    assert "keras-aux" in st.machines
    for mid in ("pc-b", "keras-aux"):
        m = st.machines[mid]
        assert m.status_path == tmp_path / "fleet" / mid / "status.json"
        assert m.last_seen_seconds_ago is not None
        assert m.last_seen_seconds_ago < 2.0
        assert m.pool_dir == tmp_path / "checkpoints" / "pool" / mid


def test_check_heartbeats_flags_stuck(tmp_path: Path) -> None:
    (tmp_path / "fleet" / "box1").mkdir(parents=True)
    old = time.time() - 3600
    (tmp_path / "fleet" / "box1" / "status.json").write_text(
        json.dumps({"last_poll": old, "timestamp": old}),
        encoding="utf-8",
    )
    (tmp_path / "checkpoints").mkdir(parents=True)
    o = _orch(tmp_path, pools=["box1"], stuck_threshold_seconds=1200.0)
    st = o.read_fleet_state()
    d = o.check_heartbeats(st)
    assert len(d) == 1
    assert d[0].kind == "heartbeat_alert"
    assert d[0].machine_id == "box1"


def test_curate_pools_dry_run_does_not_delete(tmp_path: Path) -> None:
    ck = tmp_path / "checkpoints"
    pdir = tmp_path / "checkpoints" / "pool" / "m1"
    pdir.mkdir(parents=True)
    for i in range(30):
        (ck / f"checkpoint_{i:04d}.zip").write_bytes(b"1")
    for i in range(30):
        (pdir / f"checkpoint_{i:04d}.zip").write_bytes(b"1")
    o = _orch(tmp_path, pools=["m1"], curator_min_age_minutes=0.0)
    o.tick()
    n_root = list(ck.glob("checkpoint_*.zip"))
    n_pool = list(pdir.glob("checkpoint_*.zip"))
    assert len(n_root) == 30
    assert len(n_pool) == 30
    assert o.audit_log.is_file()
    lines = o.audit_log.read_text(encoding="utf-8").strip().splitlines()
    assert any("curate" in json.loads(L)["kind"] for L in lines)


def test_run_symmetric_evals_dry_run_does_not_subprocess(tmp_path: Path) -> None:
    mdir = tmp_path / "checkpoints" / "pool" / "p1"
    mdir.mkdir(parents=True)
    (tmp_path / "checkpoints" / "latest.zip").write_bytes(b"a")
    (mdir / "checkpoint_x.zip").write_bytes(b"b")
    o = _orch(tmp_path, pools=["p1"], curator_min_age_minutes=0.0)
    with patch("subprocess.run") as m:
        o.tick()
        m.assert_not_called()


def test_run_symmetric_evals_apply_calls_subprocess(tmp_path: Path) -> None:
    mdir = tmp_path / "checkpoints" / "pool" / "p1"
    mdir.mkdir(parents=True)
    (tmp_path / "checkpoints" / "latest.zip").write_bytes(b"a")
    mdir / "checkpoint_x.zip"
    (mdir / "checkpoint_x.zip").write_bytes(b"b")
    o = _orch(
        tmp_path, pools=["p1"], dry_run=False, curator_min_age_minutes=0.0
    )

    def _fake_run(cmd: list, **kwargs: object) -> MagicMock:
        i = cmd.index("--json-out")
        p = Path(cmd[i + 1])
        _write_fake_verdict(p, candidate_wins=4, baseline_wins=3)
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run) as m:
        o.tick()
    assert m.call_count >= 1


def test_decide_promotion_winner_above_margin(tmp_path: Path) -> None:
    latest = tmp_path / "checkpoints" / "latest.zip"
    latest.parent.mkdir(parents=True)
    latest.write_bytes(b"u")
    cz = tmp_path / "cand.zip"
    cz.write_bytes(b"u")
    o = _orch(
        tmp_path, pools=["x"], games_first_seat=4, reload_margin=0.25
    )
    st = o.read_fleet_state()
    v = {
        "schema_version": 1,
        "candidate_wins": 8,
        "baseline_wins": 2,
        "games_decided": 10,
        "winrate": 0.8,
        "_candidate_path": str(cz),
    }
    d = o.decide_promotion(st, {"x": v})
    assert len(d) == 1
    assert d[0].kind == "promote"


def test_decide_promotion_skipped_when_too_few_games(tmp_path: Path) -> None:
    latest = tmp_path / "checkpoints" / "latest.zip"
    latest.parent.mkdir(parents=True)
    latest.write_bytes(b"u")
    cz = tmp_path / "cand.zip"
    cz.write_bytes(b"u")
    o = _orch(
        tmp_path, pools=["x"], games_first_seat=4, reload_margin=0.25
    )
    st = o.read_fleet_state()
    v = {
        "candidate_wins": 4,
        "baseline_wins": 1,
        "games_decided": 5,
        "winrate": 0.8,
        "_candidate_path": str(cz),
    }
    d = o.decide_promotion(st, {"x": v})
    assert d == []


def test_decide_reload_increments_counter_then_fires(
    tmp_path: Path,
) -> None:
    ev = tmp_path / "fleet" / "k0" / "eval"
    ev.mkdir(parents=True)
    _write_fake_verdict(ev / "v1.json", candidate_wins=1, baseline_wins=4, map_id=1)
    (tmp_path / "checkpoints").mkdir(parents=True)
    o = _orch(
        tmp_path, pools=["k0"], reload_margin=0.25, reload_consecutive=2
    )
    o.tick()
    assert o._laggard_cycles.get("k0", 0) == 1
    d2 = o.tick()
    assert any(x.kind == "reload_request" for x in d2)
    assert o._laggard_cycles.get("k0", 0) == 0


def test_decide_reload_resets_on_recovery(
    tmp_path: Path,
) -> None:
    ev = tmp_path / "fleet" / "k0" / "eval"
    ev.mkdir(parents=True)
    _write_fake_verdict(
        ev / "v1.json", candidate_wins=1, baseline_wins=4, map_id=1
    )
    (tmp_path / "checkpoints").mkdir(parents=True)
    o = _orch(tmp_path, pools=["k0"], reload_margin=0.25)
    o.tick()
    time.sleep(0.05)
    (ev / "v2.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_wins": 11,
                "baseline_wins": 9,
                "games_decided": 20,
                "map_id": 1,
            }
        ),
        encoding="utf-8",
    )
    o.tick()
    assert o._laggard_cycles.get("k0", 0) == 0


def test_apply_promote_writes_atomic_publishing_then_replace(
    tmp_path: Path,
) -> None:
    latest = tmp_path / "checkpoints" / "latest.zip"
    latest.parent.mkdir(parents=True)
    latest.write_text("old", encoding="utf-8")
    win = tmp_path / "cand.zip"
    win.write_text("new", encoding="utf-8")
    calls: list[tuple[str, str]] = []
    real_replace = os.replace

    def _track(s: str, d: str) -> None:
        calls.append((s, d))
        real_replace(s, d)

    o = _orch(
        tmp_path, pools=["p0"], dry_run=False, games_first_seat=4, reload_margin=0.25
    )
    st = o.read_fleet_state()
    v = {
        "schema_version": 1,
        "candidate_wins": 8,
        "baseline_wins": 1,
        "games_decided": 9,
        "winrate": 0.88,
        "_candidate_path": str(win),
    }
    with patch("os.replace", side_effect=_track):
        o.decide_promotion(st, {"p0": v})
    assert any(
        Path(s).name == "latest.zip.publishing" and Path(d).name == "latest.zip"
        for s, d in calls
    )
    cands = list((tmp_path / "checkpoints" / "promoted").glob("candidate_*.zip"))
    assert len(cands) == 1


def test_apply_reload_writes_to_correct_fleet_path(
    tmp_path: Path,
) -> None:
    ev = tmp_path / "fleet" / "m3" / "eval"
    ev.mkdir(parents=True)
    _write_fake_verdict(ev / "v.json", candidate_wins=0, baseline_wins=4)
    o = _orch(
        tmp_path,
        pools=["m3"],
        dry_run=False,
        reload_margin=0.25,
        reload_consecutive=2,
    )
    o.tick()
    o.tick()
    p = tmp_path / "fleet" / "m3" / "reload_request.json"
    assert p.is_file()
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert "target_zip" in raw
    assert "reason" in raw
    assert "issued_at" in raw
    assert raw.get("min_steps_done") == 0


def test_audit_log_one_row_per_decision(
    tmp_path: Path,
) -> None:
    (tmp_path / "checkpoints").mkdir(parents=True)
    (tmp_path / "checkpoints" / "pool" / "a").mkdir(parents=True)
    (tmp_path / "checkpoints" / "pool" / "b").mkdir(parents=True)
    for mid in ("a", "b"):
        (tmp_path / "fleet" / mid / "q").parent.mkdir(
            parents=True, exist_ok=True
        )  # dirs without status.json: missing heartbeat
    o = _orch(
        tmp_path, pools=["a", "b"], curator_min_age_minutes=0.0
    )
    o.tick()
    lines = o.audit_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 7
    kinds = {json.loads(L)["kind"] for L in lines}
    assert "heartbeat_alert" in kinds
    assert "fleet_diagnosis" in kinds
    for L in lines:
        row = json.loads(L)
        for k in ("kind", "machine_id", "applied", "reason", "details", "tick_id"):
            assert k in row


def test_audit_log_noop_tick_emits_one_row(
    tmp_path: Path,
) -> None:
    o = _orch(
        tmp_path, pools=[], audit_log=tmp_path / "a.jsonl", state_file=tmp_path / "s.json"
    )
    o.tick()
    lines = o.audit_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["kind"] == "noop"


def test_state_file_persists_laggard_counter_across_instances(
    tmp_path: Path,
) -> None:
    ev = tmp_path / "fleet" / "k9" / "eval"
    ev.mkdir(parents=True)
    _write_fake_verdict(
        ev / "v1.json", candidate_wins=1, baseline_wins=4, map_id=1
    )
    (tmp_path / "checkpoints").mkdir(parents=True)
    stf = tmp_path / "lag.json"
    o1 = _orch(
        tmp_path,
        pools=["k9"],
        reload_margin=0.25,
        reload_consecutive=2,
        state_file=stf,
    )
    o1.tick()
    o2 = _orch(
        tmp_path,
        pools=["k9"],
        reload_margin=0.25,
        reload_consecutive=2,
        state_file=stf,
    )
    d = o2.tick()
    assert any(x.kind == "reload_request" for x in d)


def test_subprocess_eval_failure_logged_not_raised(
    tmp_path: Path,
) -> None:
    mdir = tmp_path / "checkpoints" / "pool" / "p1"
    mdir.mkdir(parents=True)
    (tmp_path / "checkpoints" / "latest.zip").write_bytes(b"a")
    (mdir / "checkpoint_x.zip").write_bytes(b"b")
    o = _orch(
        tmp_path, pools=["p1"], dry_run=False, curator_min_age_minutes=0.0
    )
    with patch("subprocess.run", return_value=MagicMock(
        returncode=1, stdout="", stderr="boom"
    )):
        d = o.tick()
    ev = [x for x in d if x.kind == "eval"]
    assert len(ev) == 1
    assert ev[0].details.get("failure") is True
    assert "boom" in (ev[0].details.get("stderr_tail") or "")


def test_main_one_tick_runs_without_real_fleet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alog = tmp_path / "a.jsonl"
    stf = tmp_path / "s.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fleet_orchestrator",
            "--once",
            "--shared-root",
            str(tmp_path),
            "--pools",
            "",
            "--audit-log",
            str(alog),
            "--state-file",
            str(stf),
        ],
    )
    assert main() == 0
    assert alog.is_file()
    line = alog.read_text(encoding="utf-8").strip().splitlines()
    assert line and json.loads(line[0])["kind"] == "noop"


def test_read_proposed_args_missing_returns_none(tmp_path: Path) -> None:
    assert read_proposed_args("nope", tmp_path) is None


def test_read_proposed_args_loads_json(tmp_path: Path) -> None:
    p = tmp_path / "fleet" / "m1" / "proposed_args.json"
    p.parent.mkdir(parents=True)
    payload = {"machine_id": "m1", "args": {"--n-envs": 2}}
    p.write_text(json.dumps(payload), encoding="utf-8")
    got = read_proposed_args("m1", tmp_path)
    assert got == payload


def test_proposed_args_row_in_audit_when_file_present(
    tmp_path: Path,
) -> None:
    prop = {
        "machine_id": "a",
        "proposed_at": "2026-04-22T00:00:00Z",
        "based_on_probe_at": "2026-04-22T00:00:00Z",
        "args": {"--n-envs": 4, "--n-steps": 512, "--batch-size": 256},
        "reasoning": "test fixture",
        "auto_apply": False,
    }
    (tmp_path / "fleet" / "a" / "proposed_args.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (tmp_path / "fleet" / "a" / "proposed_args.json").write_text(
        json.dumps(prop), encoding="utf-8"
    )
    (tmp_path / "checkpoints").mkdir(parents=True)
    o = _orch(
        tmp_path, pools=["a"], curator_min_age_minutes=0.0
    )
    o.tick()
    lines = o.audit_log.read_text(encoding="utf-8").strip().splitlines()
    rows = [json.loads(L) for L in lines]
    kinds = [r["kind"] for r in rows]
    assert "proposed_args" in kinds
    pr = [r for r in rows if r["kind"] == "proposed_args"]
    assert len(pr) == 1
    row = pr[0]
    assert row["machine_id"] == "a"
    assert row["applied"] is False
    assert row["details"]["proposed"] == prop
    assert "path" in row["details"]