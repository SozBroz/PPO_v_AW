"""fleet_orchestrator: MCTS verdict merged into proposed_args when pass_overall."""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]


def _fo():
    p = REPO / "scripts" / "fleet_orchestrator.py"
    name = "fleet_orchestrator_mcts_merge"
    spec = importlib.util.spec_from_file_location(name, p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


fo = _fo()
FleetOrchestrator = fo.FleetOrchestrator


def _mh():
    # Use the real logical module name so @dataclass / slots resolution finds
    # ``sys.modules[cls.__module__]`` during class creation (Python 3.12+).
    name = "tools.mcts_health"
    spec = importlib.util.spec_from_file_location(
        name, REPO / "tools" / "mcts_health.py"
    )
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


mh = _mh()


def _orch(tmp_path: Path, *, pools: list[str], **kw: object) -> object:
    defaults: dict = {
        "shared_root": tmp_path,
        "pools": pools,
        "dry_run": False,
        "repo_root": tmp_path,
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
        "auto_apply": False,
        "apply_cooldown_s": 600.0,
        "train_pid_file_template": "fleet/{machine_id}/train.pid",
        "train_launch_cmd_file_template": "fleet/{machine_id}/train_launch_cmd.json",
        "curriculum_enabled": False,
        "curriculum_window_games": 200,
        "curriculum_state_file_template": "fleet/{machine_id}/curriculum_state.json",
        "mcts_health_refresh_every_ticks": 0,
        "mcts_gate_required_consecutive": 1,
    }
    defaults.update(kw)
    return FleetOrchestrator(**defaults)  # type: ignore[misc]


def _probe(mid: str) -> dict:
    return {
        "machine_id": mid,
        "probed_at": "2026-04-22T12:00:00Z",
        "cpu": {"physical_cores": 8},
        "ram": {"total_gb": 32.0},
    }


def _verdict(*, pass_overall: bool, mode: str, sims: int) -> object:
    metrics = mh.MctsHealthMetrics(
        games_in_window=80,
        capture_sense_score=0.5,
        avg_capture_completions_per_game=2.0,
        avg_terrain_usage_score=0.55,
        army_value_lead_pos_rate=0.5,
        win_rate=0.5,
        avg_episode_length_turns=30.0,
        early_resign_rate=0.1,
    )
    return mh.MctsHealthVerdict(
        machine_id="t-mcts",
        measured_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        games_in_window=80,
        metrics=metrics,
        pass_capture=pass_overall,
        pass_terrain=pass_overall,
        pass_army_value=pass_overall,
        pass_episode_quality=pass_overall,
        pass_overall=pass_overall,
        proposed_mcts_mode=mode,
        proposed_mcts_sims=sims,
        reasoning="test",
    )


def test_pass_false_no_mcts_args(tmp_path: Path) -> None:
    mid = "t-mcts"
    (tmp_path / "fleet" / mid).mkdir(parents=True)
    (tmp_path / "fleet" / mid / "probe.json").write_text(
        json.dumps(_probe(mid)), encoding="utf-8"
    )
    (tmp_path / "checkpoints").mkdir(parents=True)
    o = _orch(tmp_path, pools=[mid], curriculum_enabled=False)
    with patch.object(fo, "read_mcts_health", return_value=_verdict(pass_overall=False, mode="off", sims=0)):
        o.tick()
    doc = json.loads((tmp_path / "fleet" / mid / "proposed_args.json").read_text(encoding="utf-8"))
    assert "--mcts-mode" not in doc["args"]
    assert "--mcts-sims" not in doc["args"]


def test_pass_true_alphazero_merges_sims(tmp_path: Path) -> None:
    mid = "t-mcts"
    (tmp_path / "fleet" / mid).mkdir(parents=True)
    (tmp_path / "fleet" / mid / "probe.json").write_text(
        json.dumps(_probe(mid)), encoding="utf-8"
    )
    (tmp_path / "checkpoints").mkdir(parents=True)
    o = _orch(tmp_path, pools=[mid], curriculum_enabled=False)
    with patch.object(
        fo,
        "read_mcts_health",
        return_value=_verdict(pass_overall=True, mode="alphazero", sims=16),
    ):
        o.tick()
    doc = json.loads((tmp_path / "fleet" / mid / "proposed_args.json").read_text(encoding="utf-8"))
    assert doc["args"]["--mcts-mode"] == "alphazero"
    assert doc["args"]["--mcts-sims"] == 16


def test_mcts_merge_idempotent_second_tick(tmp_path: Path) -> None:
    mid = "t-mcts"
    (tmp_path / "fleet" / mid).mkdir(parents=True)
    (tmp_path / "fleet" / mid / "probe.json").write_text(
        json.dumps(_probe(mid)), encoding="utf-8"
    )
    (tmp_path / "checkpoints").mkdir(parents=True)
    o = _orch(tmp_path, pools=[mid], curriculum_enabled=False)
    v = _verdict(pass_overall=True, mode="alphazero", sims=8)
    with patch.object(fo, "read_mcts_health", return_value=v):
        o.tick()
        p = tmp_path / "fleet" / mid / "proposed_args.json"
        b1 = p.read_bytes()
        o.tick()
        b2 = p.read_bytes()
    assert b1 == b2


def _bare_probe(tmp_path: Path, mid: str) -> None:
    (tmp_path / "fleet" / mid).mkdir(parents=True, exist_ok=True)
    (tmp_path / "fleet" / mid / "probe.json").write_text(
        json.dumps(_probe(mid)), encoding="utf-8"
    )
    if not (tmp_path / "checkpoints").is_dir():
        (tmp_path / "checkpoints").mkdir(parents=True)


def test_hysteresis_two_passes_required_for_merge(tmp_path: Path) -> None:
    """Slice B: pass_overall on tick 1 → pending; tick 2 → merge."""
    mid = "t-aux"
    _bare_probe(tmp_path, mid)
    o = _orch(
        tmp_path,
        pools=[mid],
        curriculum_enabled=False,
        host_machine_id="other-host",
        mcts_gate_required_consecutive=2,
    )
    v = _verdict(pass_overall=True, mode="alphazero", sims=16)
    p = tmp_path / "fleet" / mid / "proposed_args.json"
    with patch.object(fo, "read_mcts_health", return_value=v):
        d1 = o.tick()
        doc1 = json.loads(p.read_text(encoding="utf-8"))
        assert "--mcts-mode" not in doc1["args"]
        assert any(
            x.kind == "mcts_gate_pending"
            and x.machine_id == mid
            and x.details["streak"] == 1
            and x.details["required"] == 2
            for x in d1
        )
        d2 = o.tick()
    doc2 = json.loads(p.read_text(encoding="utf-8"))
    assert doc2["args"]["--mcts-mode"] == "alphazero"
    assert doc2["args"]["--mcts-sims"] == 16
    assert not any(x.kind == "mcts_gate_pending" for x in d2)


def test_hysteresis_resets_on_failing_verdict(tmp_path: Path) -> None:
    """Slice B: a failing verdict resets the streak; subsequent passes restart counting."""
    mid = "t-aux"
    _bare_probe(tmp_path, mid)
    o = _orch(
        tmp_path,
        pools=[mid],
        curriculum_enabled=False,
        host_machine_id="other-host",
        mcts_gate_required_consecutive=2,
    )
    p = tmp_path / "fleet" / mid / "proposed_args.json"
    v_pass = _verdict(pass_overall=True, mode="alphazero", sims=16)
    v_fail = _verdict(pass_overall=False, mode="off", sims=0)
    with patch.object(fo, "read_mcts_health", return_value=v_pass):
        o.tick()
    assert o._mcts_pass_streak_by_machine[mid] == 1
    with patch.object(fo, "read_mcts_health", return_value=v_fail):
        o.tick()
    assert o._mcts_pass_streak_by_machine[mid] == 0
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert "--mcts-mode" not in doc["args"]
    with patch.object(fo, "read_mcts_health", return_value=v_pass):
        o.tick()
        d = o.tick()
    doc2 = json.loads(p.read_text(encoding="utf-8"))
    assert doc2["args"]["--mcts-mode"] == "alphazero"
    assert any(x.kind == "mcts_gate_pending" for x in d) is False


def test_host_gate_skips_merge_without_enable_flag(tmp_path: Path) -> None:
    """Slice C: host machine + enable_mcts_here=False → skip merge, emit mcts_skip_host."""
    mid = "pc-b"
    _bare_probe(tmp_path, mid)
    o = _orch(
        tmp_path,
        pools=[mid],
        curriculum_enabled=False,
        host_machine_id=mid,
        enable_mcts_here=False,
        mcts_gate_required_consecutive=1,
    )
    v = _verdict(pass_overall=True, mode="alphazero", sims=32)
    with patch.object(fo, "read_mcts_health", return_value=v):
        d = o.tick()
    doc = json.loads(
        (tmp_path / "fleet" / mid / "proposed_args.json").read_text(encoding="utf-8")
    )
    assert "--mcts-mode" not in doc["args"]
    assert any(
        x.kind == "mcts_skip_host"
        and x.machine_id == mid
        and "operator-only" in x.details["reason"]
        for x in d
    )


def test_host_gate_allows_merge_with_enable_flag(tmp_path: Path) -> None:
    """Slice C: host machine + enable_mcts_here=True → merge after hysteresis."""
    mid = "pc-b"
    _bare_probe(tmp_path, mid)
    o = _orch(
        tmp_path,
        pools=[mid],
        curriculum_enabled=False,
        host_machine_id=mid,
        enable_mcts_here=True,
        mcts_gate_required_consecutive=2,
    )
    v = _verdict(pass_overall=True, mode="alphazero", sims=32)
    p = tmp_path / "fleet" / mid / "proposed_args.json"
    with patch.object(fo, "read_mcts_health", return_value=v):
        o.tick()
        doc1 = json.loads(p.read_text(encoding="utf-8"))
        assert "--mcts-mode" not in doc1["args"]
        o.tick()
    doc2 = json.loads(p.read_text(encoding="utf-8"))
    assert doc2["args"]["--mcts-mode"] == "alphazero"
    assert doc2["args"]["--mcts-sims"] == 32


def test_non_host_machine_merges_regardless_of_host_flag(tmp_path: Path) -> None:
    """Slice C: non-host machine merges (after hysteresis) even with enable_mcts_here=False."""
    mid = "aux-1"
    _bare_probe(tmp_path, mid)
    o = _orch(
        tmp_path,
        pools=[mid],
        curriculum_enabled=False,
        host_machine_id="pc-b",
        enable_mcts_here=False,
        mcts_gate_required_consecutive=1,
    )
    v = _verdict(pass_overall=True, mode="alphazero", sims=8)
    with patch.object(fo, "read_mcts_health", return_value=v):
        d = o.tick()
    doc = json.loads(
        (tmp_path / "fleet" / mid / "proposed_args.json").read_text(encoding="utf-8")
    )
    assert doc["args"]["--mcts-mode"] == "alphazero"
    assert doc["args"]["--mcts-sims"] == 8
    assert not any(x.kind == "mcts_skip_host" for x in d)


def test_train_advisor_mode_refused_even_on_host(tmp_path: Path) -> None:
    """Slice C: train_advisor mode is refused even when --enable-mcts-here is set."""
    mid = "pc-b"
    _bare_probe(tmp_path, mid)
    o = _orch(
        tmp_path,
        pools=[mid],
        curriculum_enabled=False,
        host_machine_id=mid,
        enable_mcts_here=True,
        mcts_gate_required_consecutive=1,
    )
    v = _verdict(pass_overall=True, mode="train_advisor", sims=16)
    with patch.object(fo, "read_mcts_health", return_value=v):
        d = o.tick()
    doc = json.loads(
        (tmp_path / "fleet" / mid / "proposed_args.json").read_text(encoding="utf-8")
    )
    assert "--mcts-mode" not in doc["args"]
    assert any(
        x.kind == "mcts_refuse_train_advisor"
        and x.machine_id == mid
        and x.details["proposed_mcts_mode"] == "train_advisor"
        for x in d
    )
