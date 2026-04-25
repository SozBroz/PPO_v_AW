"""Phase 11J-CLUSTER-B-SHIP — Von Bolt SCOP "Ex Machina" AOE override.

Covers the new ``_oracle_power_aoe_positions`` channel introduced in
``docs/oracle_exception_audit/phase11j_cluster_b_ship.md``:

* Engine ``_apply_power_effects`` co_id 30 SCOP branch consumes the
  oracle-supplied 2-range Manhattan (13-tile) AOE when present, applying
  -30 internal HP only to enemy units inside the AOE; falls back to the
  historical global -30 when no override is set (preserving RL /
  non-oracle path semantics).
* Override is one-shot: cleared after consumption so the next
  activation rolls fresh.
* Oracle Power handler in ``tools/oracle_zip_replay.py`` parses
  ``Power.missileCoords`` and expands the chosen center into the 13-tile
  Manhattan diamond; raises ``UnsupportedOracleAction`` when
  ``missileCoords`` is missing/malformed for a Von Bolt SCOP.

Diagnostic source for the fix is recorded in the file header above and
on the engine-side comment block at ``engine/game.py`` co_id 30 branch.
"""
from __future__ import annotations

import pytest

from engine.action import Action, ActionStage, ActionType
from engine.co import COState, make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData, PropertyState
from engine.unit import Unit, UnitType, UNIT_STATS

from tools.oracle_zip_replay import (
    UnsupportedOracleAction,
    apply_oracle_action_json,
)

PLAIN = 1


def _make_state(*, width: int = 5, height: int = 5, p0_co_id: int = 30) -> GameState:
    md = MapData(
        map_id=0,
        name="phase11j_cluster_b",
        map_type="std",
        terrain=[[PLAIN] * width for _ in range(height)],
        height=height,
        width=width,
        cap_limit=99,
        unit_limit=50,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=[],
        hq_positions={0: [], 1: []},
        lab_positions={0: [], 1: []},
        country_to_player={},
    )
    co0 = make_co_state_safe(p0_co_id)
    co1 = make_co_state_safe(7)  # Max — irrelevant; just a non-Sasha non-VB pick
    state = GameState(
        map_data=md,
        units={0: [], 1: []},
        funds=[0, 0],
        co_states=[co0, co1],
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
    # ACTIVATE_SCOP path expects a charged power bar — populate to the
    # SCOP threshold so ``step`` doesn't bail; we sidestep ``step`` and
    # call ``_apply_power_effects`` directly anyway.
    return state


def _spawn(state: GameState, ut: UnitType, player: int, pos: tuple[int, int],
           *, unit_id: int, hp: int = 100) -> Unit:
    stats = UNIT_STATS[ut]
    u = Unit(
        unit_type=ut,
        player=player,
        hp=hp,
        ammo=stats.max_ammo if stats.max_ammo > 0 else 0,
        fuel=stats.max_fuel,
        pos=pos,
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
        unit_id=unit_id,
    )
    state.units[player].append(u)
    return u


# ---------------------------------------------------------------------------
# Engine consumption — _apply_power_effects co_id 30 SCOP branch
# ---------------------------------------------------------------------------


def test_cluster_b_engine_aoe_override_pins_only_inside_diamond():
    """Oracle AOE: only enemy units inside the 13-tile Manhattan diamond lose
    30 HP. (4, 7) is in the diamond around center (4, 5) but outside the old
    3x3 box — PHP gid 1622328 env 28 ``unitReplace`` includes that tile.
    """
    state = _make_state(width=20, height=20, p0_co_id=30)
    inside = _spawn(state, UnitType.INFANTRY, 1, (4, 5), unit_id=1, hp=100)
    ring = _spawn(state, UnitType.INFANTRY, 1, (4, 7), unit_id=2, hp=100)
    far_a = _spawn(state, UnitType.TANK, 1, (16, 20 - 1), unit_id=3, hp=100)
    far_b = _spawn(state, UnitType.B_COPTER, 1, (16, 7), unit_id=4, hp=100)

    cy, cx = 4, 5
    state._oracle_power_aoe_positions = {
        (cy + dr, cx + dc)
        for dr in range(-2, 3)
        for dc in range(-2, 3)
        if abs(dr) + abs(dc) <= 2
    }
    state._apply_power_effects(player=0, cop=False)

    assert inside.hp == 70
    assert ring.hp == 70
    assert far_a.hp == 100
    assert far_b.hp == 100
    # One-shot consumption — second activation falls back to the engine
    # default (which would damage everyone) when no fresh override is set.
    assert state._oracle_power_aoe_positions is None


def test_cluster_b_engine_no_override_keeps_global_fallback():
    """Override is opt-in: when ``None`` (RL / non-oracle path), the
    engine keeps its historical "all enemy units lose 30 HP" behaviour
    so existing checkpoints / training runs don't regress.
    """
    state = _make_state(width=10, height=10, p0_co_id=30)
    a = _spawn(state, UnitType.INFANTRY, 1, (0, 0), unit_id=1, hp=100)
    b = _spawn(state, UnitType.TANK, 1, (9, 9), unit_id=2, hp=100)
    state._oracle_power_aoe_positions = None

    state._apply_power_effects(player=0, cop=False)

    assert a.hp == 70
    assert b.hp == 70


def test_cluster_b_engine_aoe_override_floors_hp_at_one():
    """Damage floor at 1 internal HP matches the rest of
    ``_apply_power_effects`` (Hawke / Drake / Olaf). Units inside the AOE
    that were already below 30 HP are pinned to 1, not driven to 0
    (which would leak a kill via SCOP that AWBW does not).
    """
    state = _make_state(width=5, height=5, p0_co_id=30)
    weak = _spawn(state, UnitType.INFANTRY, 1, (2, 2), unit_id=1, hp=10)
    state._oracle_power_aoe_positions = {(2, 2)}

    state._apply_power_effects(player=0, cop=False)

    assert weak.hp == 1
    assert weak.is_alive


# ---------------------------------------------------------------------------
# Oracle handler — Power.missileCoords parsing + dispatch into the engine
# ---------------------------------------------------------------------------


def test_cluster_b_oracle_handler_pins_aoe_from_missilecoords(monkeypatch):
    """``missileCoords`` expands to the 13-tile Manhattan diamond before
    ``ACTIVATE_SCOP``. Ring tile (4, 7) is inside for center (y=4, x=5)."""
    state = _make_state(width=20, height=20, p0_co_id=30)
    state.action_stage = ActionStage.SELECT
    state.co_states[0].power_bar = 10**9  # plenty for SCOP
    state.co_states[0].cop_stars = 6
    state.co_states[0].scop_stars = 6
    inside = _spawn(state, UnitType.INFANTRY, 1, (4, 5), unit_id=1, hp=100)
    ring = _spawn(state, UnitType.INFANTRY, 1, (4, 7), unit_id=3, hp=100)
    far = _spawn(state, UnitType.INFANTRY, 1, (16, 16), unit_id=2, hp=100)
    awbw_to_engine = {1001: 0, 1002: 1}

    apply_oracle_action_json(
        state,
        {
            "action": "Power",
            "playerID": 1001,
            "coName": "Von Bolt",
            "coPower": "S",
            "missileCoords": [{"x": "5", "y": "4"}],
        },
        awbw_to_engine,
        envelope_awbw_player_id=1001,
    )

    assert inside.hp == 70
    assert ring.hp == 70
    assert far.hp == 100
    assert state._oracle_power_aoe_positions is None


def test_cluster_b_oracle_handler_raises_when_missilecoords_missing():
    """Defensive: a Von Bolt SCOP without parseable ``missileCoords`` is
    flagged ``UnsupportedOracleAction`` rather than silently falling back
    to the global -30 path (which is the bug we just fixed). Forces the
    audit register to surface the gap so we don't regress quietly.
    """
    state = _make_state(width=10, height=10, p0_co_id=30)
    state.action_stage = ActionStage.SELECT
    state.co_states[0].power_bar = 10**9
    state.co_states[0].cop_stars = 6
    state.co_states[0].scop_stars = 6
    _spawn(state, UnitType.INFANTRY, 1, (0, 0), unit_id=1, hp=100)
    awbw_to_engine = {1001: 0, 1002: 1}

    with pytest.raises(UnsupportedOracleAction, match="missileCoords"):
        apply_oracle_action_json(
            state,
            {
                "action": "Power",
                "playerID": 1001,
                "coName": "Von Bolt",
                "coPower": "S",
                # missileCoords intentionally absent
            },
            awbw_to_engine,
            envelope_awbw_player_id=1001,
        )
