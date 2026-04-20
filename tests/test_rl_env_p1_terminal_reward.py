"""Regression: terminal engine reward from P1 microsteps must reach the P0-facing return."""

from __future__ import annotations

import json
import random as random_module
from pathlib import Path

import pytest

from engine.action import Action, ActionType, ActionStage, get_legal_actions
from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, Unit, UnitType
from rl.env import AWBWEnv

ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"
MAP_ID = 123858


def _single_map_pool() -> list[dict]:
    with open(POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    row = next(m for m in pool if m.get("map_id") == MAP_ID)
    return [row]


def _craft_p0_active_p1_hq_capture_win() -> tuple[object, tuple[int, int]]:
    """P0 at SELECT; P1 infantry on P0 HQ with 1 capture point left; P1 wins on CAPTURE."""
    m = load_map(MAP_ID, POOL_PATH, MAPS_DIR)
    s = make_initial_state(m, 1, 5, tier_name="T3")
    hq = next(p for p in s.properties if p.is_hq and p.owner == 0)
    hr, hc = hq.row, hq.col
    for plist in s.units.values():
        plist[:] = [u for u in plist if u.pos != (hr, hc)]
    st = UNIT_STATS[UnitType.INFANTRY]
    inf = Unit(
        UnitType.INFANTRY,
        1,
        100,
        st.max_ammo,
        st.max_fuel,
        (hr, hc),
        False,
        [],
        False,
        20,
        1,
    )
    s.units[1].append(inf)
    hq.capture_points = 1
    hq.owner = 0
    # P0 must be able to pass immediately (``env.step`` uses flat index 0 = END_TURN).
    for u in s.units[0]:
        if u.is_alive:
            u.moved = True
    s.active_player = 0
    s.action_stage = ActionStage.SELECT
    s.selected_unit = None
    s.done = False
    s.winner = None
    s.win_reason = None
    return s, (hr, hc)


def _same_action_shape(a: Action, template: Action) -> bool:
    if a.action_type != template.action_type:
        return False
    if a.unit_pos != template.unit_pos:
        return False
    if template.move_pos is not None and a.move_pos != template.move_pos:
        return False
    return True


def test_p1_hq_capture_terminal_reward_reaches_p0(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the episode ends on P1's clock, P0 must still receive the decisive -1/+1."""
    monkeypatch.setattr("rl.env._append_game_log_line", lambda _record: None)

    crafted, (hr, hc) = _craft_p0_active_p1_hq_capture_win()
    scripted = [
        Action(ActionType.SELECT_UNIT, unit_pos=(hr, hc)),
        Action(
            ActionType.SELECT_UNIT,
            unit_pos=(hr, hc),
            move_pos=(hr, hc),
        ),
        Action(
            ActionType.CAPTURE,
            unit_pos=(hr, hc),
            move_pos=(hr, hc),
        ),
    ]

    _real_choice = random_module.choice

    def controlled_choice(legal):
        # ``reset`` / sampling use ``random.choice`` on non-action lists.
        if not legal or not isinstance(legal[0], Action):
            return _real_choice(legal)
        if not scripted:
            return _real_choice(legal)
        want = scripted[0]
        for a in legal:
            if _same_action_shape(a, want):
                scripted.pop(0)
                return a
        # Opening random-opponent walk during ``reset()`` — do not burn scripted P1 capture.
        return _real_choice(legal)

    monkeypatch.setattr("rl.env.random.choice", controlled_choice)

    env = AWBWEnv(
        map_pool=_single_map_pool(),
        opponent_policy=None,
        co_p0=1,
        co_p1=5,
        tier_name="T3",
    )
    env.reset(seed=0)
    env.state = crafted

    _obs, reward, terminated, _truncated, info = env.step(0)

    assert terminated is True
    assert info.get("winner") == 1
    # Without the fix this stayed near dense-shaping scale (~0); P1's +1 must become -1 for P0.
    assert reward < -0.5
    assert scripted == []

    # Sanity: scripted actions are legal in order on a fresh state opened on P1.
    s, (hr, hc) = _craft_p0_active_p1_hq_capture_win()
    s.active_player = 1
    s.action_stage = ActionStage.SELECT
    s.selected_unit = None
    for want in [
        Action(ActionType.SELECT_UNIT, unit_pos=(hr, hc)),
        Action(ActionType.SELECT_UNIT, unit_pos=(hr, hc), move_pos=(hr, hc)),
        Action(ActionType.CAPTURE, unit_pos=(hr, hc), move_pos=(hr, hc)),
    ]:
        legal = get_legal_actions(s)
        picked = next(a for a in legal if _same_action_shape(a, want))
        s, _r, d = s.step(picked)
    assert d is True and s.winner == 1
