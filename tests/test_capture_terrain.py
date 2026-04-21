"""Full capture swaps property terrain IDs; owned-property resupply includes AWBW day heal.

These tests verify the ``_apply_capture`` handler in isolation: they
parachute a CAPTURE Action onto a SELECT-stage state without walking
SELECT_UNIT → MOVE first, so under STEP-GATE (Phase 3
``desync_purge_engine_harden``) they pass ``oracle_mode=True`` to bypass
the legality gate and exercise the handler directly.
"""

from __future__ import annotations

import pytest

from engine.action import Action, ActionType
from engine.game import make_initial_state
from engine.map_loader import PropertyState, load_map
from engine.terrain import (
    country_id_for_player_seat,
    property_terrain_id_after_owner_change,
    property_terrain_id_for_country_and_kind,
)
from engine.unit import UNIT_STATS, Unit, UnitType

from server.play_human import MAPS_DIR, POOL_PATH


def test_property_terrain_id_neutral_city_os_p0():
    """Neutral city (34) + P0 seated as OS (country 1) → OS city (38)."""
    tid = property_terrain_id_after_owner_change(34, 0, {1: 0, 5: 1})
    assert tid == 38


def test_full_capture_updates_terrain_on_misery_neutral_city():
    m = load_map(123858, POOL_PATH, MAPS_DIR)
    s = make_initial_state(m, 1, 5, tier_name="T3")

    r, c = 0, 4
    assert s.map_data.terrain[r][c] == 34
    prop = next(p for p in s.properties if p.row == r and p.col == c)
    assert prop.owner is None
    prop.capture_points = 10

    assert s.get_unit_at(r, c) is None
    st = UNIT_STATS[UnitType.INFANTRY]
    inf = Unit(
        UnitType.INFANTRY,
        0,
        100,
        st.max_ammo,
        st.max_fuel,
        (r, c),
        False,
        [],
        False,
        20,
        1,
    )
    s.units[0].append(inf)
    s.active_player = 0

    s.step(Action(ActionType.CAPTURE, unit_pos=(r, c), move_pos=(r, c)), oracle_mode=True)

    assert prop.owner == 0
    cid = country_id_for_player_seat(s.map_data.country_to_player, 0)
    assert cid is not None
    expected = property_terrain_id_for_country_and_kind(
        cid,
        is_hq=False,
        is_lab=False,
        is_comm_tower=False,
        is_base=False,
        is_airport=False,
        is_port=False,
    )
    assert expected is not None
    assert s.map_data.terrain[r][c] == expected
    assert prop.terrain_id == expected


def test_full_capture_neutral_comm_tower_swaps_tid():
    """Neutral comm tower (133) flips to the capturer's country (GL seating parity)."""
    from test_lander_and_fuel import _fresh_state

    s = _fresh_state()
    s.map_data.country_to_player = {5: 0, 1: 1}
    r, c = 2, 2
    s.map_data.terrain[r][c] = 133
    s.properties.append(
        PropertyState(
            terrain_id=133,
            row=r,
            col=c,
            owner=None,
            capture_points=10,
            is_hq=False,
            is_lab=False,
            is_comm_tower=True,
            is_base=False,
            is_airport=False,
            is_port=False,
        )
    )
    prop = s.properties[-1]
    st = UNIT_STATS[UnitType.INFANTRY]
    s.units[0].append(
        Unit(
            UnitType.INFANTRY,
            0,
            100,
            st.max_ammo,
            st.max_fuel,
            (r, c),
            False,
            [],
            False,
            20,
            1,
        )
    )
    s.active_player = 0
    s.step(Action(ActionType.CAPTURE, unit_pos=(r, c), move_pos=(r, c)), oracle_mode=True)
    assert prop.owner == 0
    exp = property_terrain_id_after_owner_change(133, 0, s.map_data.country_to_player)
    assert exp == 128
    assert s.map_data.terrain[r][c] == 128
    assert prop.terrain_id == 128


def test_full_capture_neutral_lab_swaps_tid():
    """Neutral lab (145) flips to the capturer's country."""
    from test_lander_and_fuel import _fresh_state

    s = _fresh_state()
    s.map_data.country_to_player = {5: 0, 1: 1}
    r, c = 2, 2
    s.map_data.terrain[r][c] = 145
    s.properties.append(
        PropertyState(
            terrain_id=145,
            row=r,
            col=c,
            owner=None,
            capture_points=10,
            is_hq=False,
            is_lab=True,
            is_comm_tower=False,
            is_base=False,
            is_airport=False,
            is_port=False,
        )
    )
    prop = s.properties[-1]
    st = UNIT_STATS[UnitType.INFANTRY]
    s.units[0].append(
        Unit(
            UnitType.INFANTRY,
            0,
            100,
            st.max_ammo,
            st.max_fuel,
            (r, c),
            False,
            [],
            False,
            20,
            1,
        )
    )
    s.active_player = 0
    s.step(Action(ActionType.CAPTURE, unit_pos=(r, c), move_pos=(r, c)), oracle_mode=True)
    assert prop.owner == 0
    exp = property_terrain_id_after_owner_change(145, 0, s.map_data.country_to_player)
    assert exp == 139
    assert s.map_data.terrain[r][c] == 139
    assert prop.terrain_id == 139


def _ground_repair_tile(prop) -> bool:
    if prop.is_lab or prop.is_comm_tower:
        return False
    is_city = not (
        prop.is_hq
        or prop.is_lab
        or prop.is_comm_tower
        or prop.is_base
        or prop.is_airport
        or prop.is_port
    )
    return prop.is_hq or prop.is_base or is_city


def test_resupply_heals_infantry_on_owned_hq_base_or_city():
    m = load_map(123858, POOL_PATH, MAPS_DIR)
    s = make_initial_state(m, 1, 5, tier_name="T3")
    city = next(
        p for p in s.properties if p.owner == 0 and _ground_repair_tile(p)
    )
    if s.get_unit_at(city.row, city.col) is not None:
        pytest.skip("Expected an unoccupied P0 HQ/base/city on this map fixture")

    st = UNIT_STATS[UnitType.INFANTRY]
    u = Unit(
        UnitType.INFANTRY,
        0,
        50,
        st.max_ammo,
        st.max_fuel,
        (city.row, city.col),
        False,
        [],
        False,
        20,
        1,
    )
    s.funds[0] = 500
    s.units[0].append(u)
    s._resupply_on_properties(0)
    # Infantry 1000g: 20% = 200 for full +20 internal (+2 bars)
    assert u.hp == 70
    assert s.funds[0] == 300


def test_property_day_repair_partial_hp_charges_proportionally():
    """Only +10 internal to max → 10% of deployment cost (infantry 1000 → 100g)."""
    m = load_map(123858, POOL_PATH, MAPS_DIR)
    s = make_initial_state(m, 1, 5, tier_name="T3")
    city = next(
        p for p in s.properties if p.owner == 0 and _ground_repair_tile(p)
    )
    if s.get_unit_at(city.row, city.col) is not None:
        pytest.skip("Expected an unoccupied P0 HQ/base/city on this map fixture")

    st = UNIT_STATS[UnitType.INFANTRY]
    u = Unit(
        UnitType.INFANTRY,
        0,
        90,
        st.max_ammo,
        st.max_fuel,
        (city.row, city.col),
        False,
        [],
        False,
        20,
        1,
    )
    s.funds[0] = 500
    s.units[0].append(u)
    s._resupply_on_properties(0)
    assert u.hp == 100
    assert s.funds[0] == 400  # 1000 * 10 // 100 = 100


def test_property_day_repair_respects_insufficient_funds():
    """Heal only as much HP as the player can pay for (integer cost per HP chunk)."""
    m = load_map(123858, POOL_PATH, MAPS_DIR)
    s = make_initial_state(m, 1, 5, tier_name="T3")
    city = next(
        p for p in s.properties if p.owner == 0 and _ground_repair_tile(p)
    )
    if s.get_unit_at(city.row, city.col) is not None:
        pytest.skip("Expected an unoccupied P0 HQ/base/city on this map fixture")

    st = UNIT_STATS[UnitType.INFANTRY]
    u = Unit(
        UnitType.INFANTRY,
        0,
        50,
        st.max_ammo,
        st.max_fuel,
        (city.row, city.col),
        False,
        [],
        False,
        20,
        1,
    )
    s.funds[0] = 150
    s.units[0].append(u)
    s._resupply_on_properties(0)
    # Full +20 costs 200; largest h with cost <= 150 is h=15 (150g)
    assert u.hp == 65
    assert s.funds[0] == 0
