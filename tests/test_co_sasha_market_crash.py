"""Phase 11J-SASHA-MARKETCRASH-FIX — Sasha COP "Market Crash" formula.

AWBW canon (Tier 1, AWBW CO Chart Sasha row):
  *"Market Crash -- Reduces enemy power bar(s) by (10 * Funds / 5000)%
  of their maximum power bar."*
  https://awbw.amarriner.com/co.php

Engine model (see engine/game.py::_apply_power_effects Sasha co_id==19
COP branch):

* drain = opp_max_bar * sasha_funds // 50000
* opp_max_bar = opp.scop_stars * (9000 + opp.power_uses * 1800)
  (= opp's SCOP charge ceiling, the visual top of the bar; matches
  ``COState._scop_threshold`` in engine/co.py)
* opp.power_bar = max(0, opp.power_bar - drain)

Replaces the pre-fix ``count_properties(player) * 9000`` formula
(CO-SURVEY a29a6462; ≥2 oracle_gap closure: gids 1626284 and 1628953).

Coordinated with VONBOLT-SCOP-SHIP (different co_id branch in the same
function — no overlap) and L1-BUILD-FUNDS (does not touch Sasha branch).
"""
from __future__ import annotations

import pytest

from engine.action import ActionStage
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData

PLAIN = 1

SASHA_CO_ID = 19
ANDY_CO_ID = 1


def _make_state(*, p0_co_id: int = SASHA_CO_ID,
                p1_co_id: int = ANDY_CO_ID) -> GameState:
    md = MapData(
        map_id=0, name="market_crash", map_type="std",
        terrain=[[PLAIN] * 5 for _ in range(5)],
        height=5, width=5,
        cap_limit=99, unit_limit=50, unit_bans=[], tiers=[],
        objective_type=None, properties=[],
        hq_positions={0: [], 1: []}, lab_positions={0: [], 1: []},
        country_to_player={},
    )
    return GameState(
        map_data=md,
        units={0: [], 1: []},
        funds=[0, 0],
        co_states=[make_co_state_safe(p0_co_id), make_co_state_safe(p1_co_id)],
        properties=[],
        turn=1,
        active_player=0,
        action_stage=ActionStage.SELECT,
        selected_unit=None,
        selected_move_pos=None,
        done=False,
        winner=None,
        win_reason=None,
        game_log=[],
        tier_name="T2",
        full_trace=[],
    )


def _opp_max_bar(state: GameState, opp_player: int) -> int:
    """Mirror engine formula so tests stay in lock-step with engine/co.py."""
    opp_co = state.co_states[opp_player]
    return opp_co.scop_stars * (9000 + opp_co.power_uses * 1800)


# ---------------------------------------------------------------------------
# 1. Full drain: Sasha funds = 50000 → exactly 100% of opp max → bar to 0
# ---------------------------------------------------------------------------


def test_market_crash_full_drain_at_50000_funds() -> None:
    """AWBW Tier 1: (10 * 50000 / 5000)% = 100% of opp max → full drain."""
    state = _make_state()
    state.active_player = 0
    state.funds[0] = 50_000

    max_bar = _opp_max_bar(state, opp_player=1)
    state.co_states[1].power_bar = max_bar

    state._apply_power_effects(player=0, cop=True)

    assert state.co_states[1].power_bar == 0


# ---------------------------------------------------------------------------
# 2. Partial drain: Sasha funds = 25000 → 50% of opp max → half drain
# ---------------------------------------------------------------------------


def test_market_crash_partial_drain_at_25000_funds() -> None:
    """AWBW Tier 1: (10 * 25000 / 5000)% = 50% of opp max."""
    state = _make_state()
    state.active_player = 0
    state.funds[0] = 25_000

    max_bar = _opp_max_bar(state, opp_player=1)
    state.co_states[1].power_bar = max_bar

    state._apply_power_effects(player=0, cop=True)

    expected_remaining = max_bar - (max_bar * 25_000) // 50_000
    assert state.co_states[1].power_bar == expected_remaining
    assert state.co_states[1].power_bar == max_bar // 2


# ---------------------------------------------------------------------------
# 3. Tiny drain: Sasha funds = 1000 → 2% of opp max
# ---------------------------------------------------------------------------


def test_market_crash_tiny_drain_at_1000_funds() -> None:
    """AWBW Tier 1: (10 * 1000 / 5000)% = 2% of opp max."""
    state = _make_state()
    state.active_player = 0
    state.funds[0] = 1_000

    max_bar = _opp_max_bar(state, opp_player=1)
    state.co_states[1].power_bar = max_bar

    state._apply_power_effects(player=0, cop=True)

    expected_drain = (max_bar * 1_000) // 50_000
    assert state.co_states[1].power_bar == max_bar - expected_drain
    # Sanity: drain == 2% of max (within integer floor).
    assert expected_drain == max_bar * 2 // 100


# ---------------------------------------------------------------------------
# 4. Floor at zero: massive funds vs low opp bar — clamps at 0, never negative
# ---------------------------------------------------------------------------


def test_market_crash_drain_floors_at_zero() -> None:
    """Drain magnitudes that would over-shoot must clamp at 0, not go negative."""
    state = _make_state()
    state.active_player = 0
    state.funds[0] = 1_000_000  # enormous treasury, drain dwarfs the bar

    state.co_states[1].power_bar = 5_000

    state._apply_power_effects(player=0, cop=True)

    assert state.co_states[1].power_bar == 0


# ---------------------------------------------------------------------------
# 5. SCOP path is War Bonds, NOT Market Crash — opp power bar must NOT drain
# ---------------------------------------------------------------------------


def test_market_crash_does_not_fire_on_scop() -> None:
    """Sasha SCOP = War Bonds (treasury credit on damage). Opp power bar
    must remain untouched when SCOP fires; only COP triggers Market Crash."""
    state = _make_state()
    state.active_player = 0
    state.funds[0] = 50_000

    max_bar = _opp_max_bar(state, opp_player=1)
    state.co_states[1].power_bar = max_bar

    state._apply_power_effects(player=0, cop=False)

    assert state.co_states[1].power_bar == max_bar
    # Sanity: SCOP path correctly arms War Bonds bookkeeping.
    assert state.co_states[0].war_bonds_active is True
    assert state.co_states[0].pending_war_bonds_funds == 0
