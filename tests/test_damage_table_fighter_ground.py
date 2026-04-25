"""Phase 11J-FIRE-DAMAGE-FIGHTER-TANK: chart parity for Fighter vs ground.

Verdict B: AWBW canonical chart (https://awbw.amarriner.com/damage.php) shows
no base damage for Fighter vs Tank ('-'). ``get_base_damage`` stays ``None``;
oracle Fire raises :class:`UnsupportedOracleAction` instead of mis-applying HP.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from engine.action import ActionStage
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData
from engine.unit import UNIT_STATS, Unit, UnitType

from engine.combat import get_base_damage
from tools.oracle_zip_replay import UnsupportedOracleAction, apply_oracle_action_json

PLAIN = 1


def _minimal_state() -> GameState:
    md = MapData(
        map_id=0,
        name="fighter_tank_oracle",
        map_type="std",
        terrain=[[PLAIN] * 4 for _ in range(4)],
        height=4,
        width=4,
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
    return GameState(
        map_data=md,
        units={0: [], 1: []},
        funds=[0, 0],
        co_states=[make_co_state_safe(0), make_co_state_safe(0)],
        properties=[],
        turn=1,
        active_player=0,
        action_stage=ActionStage.ACTION,
        selected_unit=None,
        selected_move_pos=None,
        done=False,
        winner=None,
        win_reason=None,
        game_log=[],
        tier_name="T2",
        full_trace=[],
    )


def _spawn(
    state: GameState,
    ut: UnitType,
    player: int,
    pos: tuple[int, int],
    *,
    unit_id: int,
    hp: int = 100,
) -> Unit:
    st = UNIT_STATS[ut]
    u = Unit(
        unit_type=ut,
        player=player,
        hp=hp,
        ammo=st.max_ammo if st.max_ammo > 0 else 0,
        fuel=st.max_fuel,
        pos=pos,
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
        unit_id=unit_id,
    )
    state.units[player].append(u)
    return u


def test_fighter_vs_tank_base_damage_is_none():
    assert get_base_damage(UnitType.FIGHTER, UnitType.TANK) is None


def test_apply_oracle_fire_fighter_vs_tank_raises_unsupported():
    """No-path Fire reaches the damage-table guard when the striker is pinned (GL drift).

    Vanilla ``get_attack_targets`` never lists a Tank for a Fighter, so the oracle
    would not otherwise resolve the attacker. Patch the resolver to return the
    Fighter as in 1631494-style board/combatInfo mismatch; then the honest
    ``UnsupportedOracleAction`` is the chart miss, not ``_apply_attack``.
    """
    state = _minimal_state()
    fighter = _spawn(state, UnitType.FIGHTER, 0, (1, 1), unit_id=101, hp=80)
    _spawn(state, UnitType.TANK, 1, (1, 2), unit_id=202, hp=100)
    state.active_player = 0
    awbw_pid = 90001
    obj = {
        "action": "Fire",
        "Move": [],
        "Fire": {
            "action": "Fire",
            "combatInfoVision": {
                "global": {
                    "combatInfo": {
                        "attacker": {
                            "units_id": 101,
                            "units_y": 1,
                            "units_x": 1,
                            "units_hit_points": 8,
                            "units_players_id": awbw_pid,
                        },
                        "defender": {
                            "units_id": 202,
                            "units_y": 1,
                            "units_x": 2,
                            "units_hit_points": 10,
                        },
                    }
                }
            },
        },
    }
    with (
        patch(
            "tools.oracle_zip_replay._resolve_fire_or_seam_attacker",
            return_value=fighter,
        ),
        pytest.raises(UnsupportedOracleAction, match="no damage entry"),
    ):
        apply_oracle_action_json(
            state, obj, {awbw_pid: 0}, envelope_awbw_player_id=awbw_pid
        )
