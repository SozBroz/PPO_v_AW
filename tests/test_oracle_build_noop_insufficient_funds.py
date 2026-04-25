"""Engine-only insufficient-funds BUILD no-ops: oracle returns (PHP may still list ``Build``)."""

from __future__ import annotations

from test_build_guard import _minimal_state
from tools.oracle_zip_replay import apply_oracle_action_json


def test_oracle_build_noop_insufficient_funds_returns_silently() -> None:
    state = _minimal_state(active_player=0, factory_owner=0)
    state.funds[0] = 0
    obj = {
        "action": "Build",
        "newUnit": {
            "global": {
                "units_id": 999001,
                "units_players_id": 9001,
                "units_name": "Infantry",
                "units_y": 0,
                "units_x": 1,
                "units_movement_points": 3,
                "units_vision": 2,
                "units_fuel": 99,
                "units_fuel_per_turn": 0,
                "units_sub_dive": "N",
                "units_ammo": 0,
                "units_short_range": 0,
                "units_long_range": 0,
                "units_second_weapon": "N",
                "units_symbol": "G",
                "units_cost": 1000,
                "units_movement_type": "F",
                "units_moved": 0,
                "units_capture": 0,
                "units_fired": 0,
                "units_hit_points": 10,
                "units_cargo1_units_id": 0,
                "units_cargo2_units_id": 0,
                "units_carried": "N",
                "countries_code": "os",
            }
        },
    }
    apply_oracle_action_json(state, obj, {9001: 0}, envelope_awbw_player_id=9001)
    assert len(state.units[0]) == 0
    assert state.funds[0] == 0
