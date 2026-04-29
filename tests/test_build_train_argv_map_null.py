"""build_train_argv_from_proposed_args: JSON null omits --map-id / co / tier (GL diversity)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _fo():
    p = REPO / "scripts" / "fleet_orchestrator.py"
    spec = importlib.util.spec_from_file_location("fo_bta_null", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["fo_bta_null"] = m
    spec.loader.exec_module(m)
    return m


def test_live_games_id_list_emits_repeated_flags() -> None:
    fo = _fo()
    snap = str(REPO / "replays" / "amarinner_my_games")
    doc = {
        "args": {
            "--n-envs": 4,
            "--live-games-id": [111, 222],
            "--live-snapshot-dir": snap,
        }
    }
    argv = fo.build_train_argv_from_proposed_args(doc, repo_root=REPO)
    assert argv.count("--live-games-id") == 2
    i = argv.index("--live-games-id")
    assert argv[i + 1] == "111"
    j = argv.index("--live-games-id", i + 1)
    assert argv[j + 1] == "222"
    assert snap in argv


def test_dual_gradient_self_play_flag_emits_from_proposed_args() -> None:
    fo = _fo()
    doc = {
        "args": {
            "--n-envs": 4,
            "--training-backend": "async",
            "--dual-gradient-self-play": True,
        }
    }
    argv = fo.build_train_argv_from_proposed_args(doc, repo_root=REPO)
    assert "--dual-gradient-self-play" in argv


def test_omit_map_tier_co_when_null() -> None:
    fo = _fo()
    doc = {
        "args": {
            "--n-envs": 4,
            "--map-id": None,
            "--tier": None,
            "--co-p0": None,
            "--co-p1": None,
            "--learner-greedy-mix": 0.0,
            "--curriculum-broad-prob": 0.2,
        }
    }
    argv = fo.build_train_argv_from_proposed_args(doc, repo_root=REPO)
    flat = " ".join(argv)
    assert "--map-id" not in flat
    assert "--tier" not in flat
    assert "--co-p0" not in flat
    assert "--co-p1" not in flat
    assert "0.2" in flat
    assert "--curriculum-broad-prob" in flat


def test_read_state_migrates_old_terminal_name(tmp_path: Path) -> None:
    from tools.curriculum_advisor import read_state, write_state, CurriculumState

    p = tmp_path / "st.json"
    p.write_text(
        '{"current_stage_name": "stage_d_self_play_pure", '
        '"games_observed_in_stage": 3, "entered_stage_at_ts": 0.0, "last_proposal_ts": 0.0, "last_seen_finished_games": 0}',
        encoding="utf-8",
    )
    st = read_state(p)
    assert st.current_stage_name == "stage_f0_full_random_stub"
    # round-trip with migrated name
    st2 = CurriculumState(
        current_stage_name=st.current_stage_name,
        games_observed_in_stage=st.games_observed_in_stage,
        entered_stage_at_ts=st.entered_stage_at_ts,
        last_proposal_ts=st.last_proposal_ts,
        last_seen_finished_games=st.last_seen_finished_games,
    )
    write_state(p, st2)
    st3 = read_state(p)
    assert st3.current_stage_name == "stage_f0_full_random_stub"
