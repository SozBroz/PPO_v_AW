"""Unit tests for rl/heuristic_termination.py (spirit + diag)."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from engine.game import GameState, make_initial_state
from engine.map_loader import load_map
from rl.encoder import GRID_SIZE, N_SCALARS, N_SPATIAL_CHANNELS
from rl.heuristic_termination import (
    SpiritConfig,
    SpiritStreaks,
    army_value_for_player,
    income_props_and_counts,
    material_margins,
    maybe_log_disagreements,
    raw_value_to_p_win,
    resign_crush_holds,
    run_calendar_day,
    snowball_holds,
    SPIRIT_BROKEN_REASON,
)


def _tiny_state() -> GameState:
    root = Path(__file__).resolve().parent.parent / "data"
    pool = root / "gl_map_pool.json"
    maps = root / "maps"
    md = load_map(133665, pool, maps)
    return make_initial_state(md, 1, 7, tier_name="T2", max_turns=30)


def test_army_value_empty() -> None:
    s = _tiny_state()
    v0 = army_value_for_player(s, 0)
    v1 = army_value_for_player(s, 1)
    assert v0 > 0 and v1 > 0
    m = income_props_and_counts(s)
    assert m["income_p0"] >= 0
    d1, d2, p0l, p1l = material_margins(m, 0.10)
    assert isinstance(d1, int)


def test_raw_value_to_p_win() -> None:
    assert 0.49 < raw_value_to_p_win(0.0, 1.0) < 0.51
    assert raw_value_to_p_win(4.0, 1.0) > 0.9


def test_snowball_and_resign_streaks_mock() -> None:
    s = _tiny_state()
    cfg = SpiritConfig(
        p_snowball=0.3,
        p_trailer_resign_max=0.4,
        value_margin=0.10,
    )
    streaks = SpiritStreaks()
    p0, p1 = 0.7, 0.3
    m = {
        "income_p0": 10,
        "income_p1": 6,
        "count_p0": 20,
        "count_p1": 8,
        "value_p0": 80_000.0,
        "value_p1": 20_000.0,
    }
    assert snowball_holds(m, 0, p0, cfg) is True
    assert snowball_holds(m, 1, p1, cfg) is False
    m_lose = {
        "income_p0": 4,
        "income_p1": 10,
        "count_p0": 2,
        "count_p1": 20,
        "value_p0": 5_000.0,
        "value_p1": 80_000.0,
    }
    _, _, _, p1v = material_margins(m_lose, cfg.value_margin)
    assert p1v is True
    assert resign_crush_holds(m_lose, 0, 0.2, cfg) is True

    def _enc(_st, o):
        import numpy as np

        return {
            "spatial": np.zeros((GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS), dtype=np.float32),
            "scalars": np.zeros((N_SCALARS,), dtype=np.float32),
        }

    class Pol:
        def obs_to_tensor(self, obs):
            import torch

            o = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}
            return o, None

        def predict_values(self, obs_t):
            import torch

            return torch.zeros((1, 1))

    mdl = MagicMock()
    mdl.device = "cpu"
    mdl.policy = Pol()

    os.environ["AWBW_SPIRIT_BROKEN"] = "0"
    os.environ["AWBW_HEURISTIC_VALUE_DIAG"] = "0"
    kind, _ = run_calendar_day(
        s,
        mdl,
        cfg,
        streaks,
        _enc,
        is_std_map=True,
        map_tier_ok=True,
        episode_id=1,
        map_id=1,
        learner_seat=0,
    )
    assert kind is None

    mdl2 = MagicMock()
    mdl2.device = "cpu"
    mdl2.policy = Pol()
    os.environ["AWBW_SPIRIT_BROKEN"] = "1"
    s2 = _tiny_state()
    s2.winner = None
    s2.done = False
    # will not meet 3-day streak; expect None
    st = SpiritStreaks()
    k2, _ = run_calendar_day(
        s2,
        mdl2,
        cfg,
        st,
        _enc,
        is_std_map=True,
        map_tier_ok=True,
        episode_id=1,
        map_id=1,
        learner_seat=0,
    )
    assert k2 is None
    del os.environ["AWBW_SPIRIT_BROKEN"]


def test_spirit_broken_constant() -> None:
    assert SPIRIT_BROKEN_REASON == "spirit_broken"


def test_diag_uses_snowball_and_resign_bars_not_half(tmp_path) -> None:
    """Value diag aligns with p_snowball / p_trailer_resign, not 0.5 vs 0.5."""
    s = _tiny_state()
    cfg = SpiritConfig()
    m_win = {
        "income_p0": 10,
        "income_p1": 6,
        "count_p0": 20,
        "count_p1": 8,
        "value_p0": 80_000.0,
        "value_p1": 20_000.0,
    }
    p = tmp_path / "d.jsonl"
    os.environ["AWBW_HEURISTIC_VALUE_DIAG"] = "1"
    try:
        n = maybe_log_disagreements(
            s,
            m_win,
            0.50,
            0.30,
            0.0,
            0.0,
            cfg,
            episode_id=1,
            map_id=1,
            tier_name="T2",
            learner_seat=0,
            log_path=p,
            lines_used=0,
        )
        assert n == 1
        line = p.read_text(encoding="utf-8").strip()
        import json

        rec = json.loads(line)
        assert rec["case"] == "spirit_snowball_material_p_below_bar"
        assert rec["seat"] == 0

        # p0 == snowball bar: not "below bar", so no new line
        p2 = tmp_path / "d2.jsonl"
        n2 = maybe_log_disagreements(
            s,
            m_win,
            0.65,
            0.30,
            0.0,
            0.0,
            cfg,
            episode_id=1,
            map_id=1,
            tier_name="T2",
            learner_seat=0,
            log_path=p2,
            lines_used=0,
        )
        assert n2 == 0
        assert not p2.exists() or p2.read_text().strip() == ""

        m_lose = {
            "income_p0": 4,
            "income_p1": 10,
            "count_p0": 2,
            "count_p1": 20,
            "value_p0": 5_000.0,
            "value_p1": 80_000.0,
        }
        p3 = tmp_path / "d3.jsonl"
        # p1 high: seat 1 is materially ahead so would match snowball material; p1 is not
        # "below bar", so we only get the P0 resign-side disagreement line.
        n3 = maybe_log_disagreements(
            s,
            m_lose,
            0.40,
            0.70,
            0.0,
            0.0,
            cfg,
            episode_id=1,
            map_id=1,
            tier_name="T2",
            learner_seat=0,
            log_path=p3,
            lines_used=0,
        )
        assert n3 == 1
        r3 = json.loads(p3.read_text(encoding="utf-8").strip().splitlines()[-1])
        assert r3["case"] == "spirit_resign_material_p_above_bar"
    finally:
        del os.environ["AWBW_HEURISTIC_VALUE_DIAG"]
