"""Engine ⊂ AWBW regression tests.

Validates the four parity fixes from the engine-vs-AWBW audit:

* A1 — BUILD is **never** offered as a Stage-2 ACTION terminator (free-action
       exploit removed; Stage-0 factory BUILD is the only AWBW-correct path).
* A3 — ``prop.capture_points`` resets to 20 when the capturer dies on the tile
       (counter-kill on a mid-capture property, defender killed on its own
       mid-capture property). Tile-vacated cases were already handled by
       ``_move_unit``.
* B5 — Carrier accepts air units (Fighter / Bomber / Stealth / B-Copter /
       T-Copter / Black Bomb) as cargo.
* relax — ``_apply_wait`` no longer raises when CAPTURE would also be legal
       (AWBW lets the player decline the capture and just WAIT).
"""

from __future__ import annotations

from engine.action import (
    Action,
    ActionStage,
    ActionType,
    get_legal_actions,
    get_loadable_into,
)
from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, Unit, UnitType

from server.play_human import MAPS_DIR, POOL_PATH


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _blank_state(map_id: int = 123858, p0_co: int = 1, p1_co: int = 1):
    m = load_map(map_id, POOL_PATH, MAPS_DIR)
    s = make_initial_state(m, p0_co, p1_co, tier_name="T3")
    s.units[0] = []
    s.units[1] = []
    s.active_player = 0
    s.action_stage = ActionStage.SELECT
    s.selected_unit = None
    s.selected_move_pos = None
    return s


def _spawn(s, ut: UnitType, player: int, pos, hp: int = 100, unit_id: int = 0):
    st = UNIT_STATS[ut]
    u = Unit(
        ut, player, hp, st.max_ammo, st.max_fuel, pos,
        False, [], False, 20, unit_id or len(s.units[0]) + len(s.units[1]) + 1,
    )
    s.units[player].append(u)
    return u


def _find_owned_factory(s, player: int):
    """Return (row, col) of an empty owned factory/airport/port, or None."""
    from engine.terrain import get_terrain
    for prop in s.properties:
        if prop.owner != player:
            continue
        terr = get_terrain(s.map_data.terrain[prop.row][prop.col])
        if not (terr.is_base or terr.is_airport or terr.is_port):
            continue
        if s.get_unit_at(prop.row, prop.col) is not None:
            continue
        return (prop.row, prop.col)
    return None


# ---------------------------------------------------------------------------
# A1: BUILD as Stage-2 ACTION terminator must be gone
# ---------------------------------------------------------------------------

def test_a1_build_never_appears_as_stage2_action():
    """No Stage-2 BUILD even when the selected unit ended its move on its own
    empty factory tile. (AWBW: factories produce from the SELECT stage, the
    moved unit cannot also issue a build.)"""
    s = _blank_state()
    factory = _find_owned_factory(s, 0)
    assert factory is not None, "test fixture needs at least one owned base"

    # Spawn an infantry directly on the factory tile so it can "end its move"
    # there with zero MP cost. This is the worst-case Stage-2 BUILD shape.
    inf = _spawn(s, UnitType.INFANTRY, 0, factory)
    s.funds[0] = 999_999

    # Drive into Stage-2 ACTION via SELECT_UNIT → SELECT_UNIT(move_pos=same).
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=inf.pos))
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=inf.pos, move_pos=inf.pos))
    assert s.action_stage == ActionStage.ACTION

    legal = get_legal_actions(s)
    builds = [a for a in legal if a.action_type == ActionType.BUILD]
    assert builds == [], (
        f"Stage-2 BUILD leaked into legal_actions: {builds!r} — "
        "this would let the unit build a unit and remain unmoved."
    )


def test_a1_stage0_build_still_works():
    """The AWBW-correct Stage-0 BUILD path stays available with no selected unit."""
    s = _blank_state()
    factory = _find_owned_factory(s, 0)
    assert factory is not None
    s.funds[0] = 999_999

    legal = get_legal_actions(s)
    stage0_builds = [
        a for a in legal
        if a.action_type == ActionType.BUILD
        and a.unit_pos is None
        and a.move_pos == factory
    ]
    assert stage0_builds, "Stage-0 BUILD must remain legal on owned factory tiles"


# ---------------------------------------------------------------------------
# A3: capture_progress reset on capturer death
# ---------------------------------------------------------------------------

def _setup_capturer_kill_state():
    """Build a hand-crafted scenario: P1 infantry mid-capture on a city; P0
    Mega Tank adjacent on ground. P0 attack kills the infantry — the
    property's capture_points must reset to 20 from the partial value."""
    from engine.terrain import get_move_cost, MOVE_TREAD, INF_PASSABLE
    s = _blank_state()

    # Owner doesn't matter — we set capture_points directly. Just need an
    # infantry-passable property with an adjacent ground tile (most cities).
    for target_prop in s.properties:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ar, ac = target_prop.row + dr, target_prop.col + dc
            if not (0 <= ar < s.map_data.height and 0 <= ac < s.map_data.width):
                continue
            tid = s.map_data.terrain[ar][ac]
            if get_move_cost(tid, MOVE_TREAD) >= INF_PASSABLE:
                continue
            if s.get_unit_at(ar, ac) is not None:
                continue
            target_prop.capture_points = 12  # mid-capture
            inf = _spawn(s, UnitType.INFANTRY, 1, (target_prop.row, target_prop.col), hp=10)
            attacker = _spawn(s, UnitType.MEGA_TANK, 0, (ar, ac))
            return s, attacker, inf, target_prop
    raise RuntimeError("no neutral property with an adjacent ground tile in fixture map")


def test_a3_capture_points_reset_when_defender_dies():
    s, attacker, defender, prop = _setup_capturer_kill_state()
    assert prop.capture_points == 12  # pre-condition

    # Drive: SELECT MegaTank → end move on its own tile → ATTACK defender.
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=attacker.pos))
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=attacker.pos, move_pos=attacker.pos))
    s.step(Action(
        ActionType.ATTACK,
        unit_pos=attacker.pos,
        move_pos=attacker.pos,
        target_pos=defender.pos,
    ))

    # Defender must have died. Otherwise this fixture is mis-tuned.
    survivors = [u for u in s.units[1] if u.unit_id == defender.unit_id]
    assert not survivors, "fixture failure: defender survived the Mega Tank hit"
    assert prop.capture_points == 20, (
        f"capture_points must reset to 20 on defender death; got {prop.capture_points}"
    )


# ---------------------------------------------------------------------------
# B5: Carrier loadable list
# ---------------------------------------------------------------------------

def test_b5_carrier_loads_air_units():
    loadable = set(get_loadable_into(UnitType.CARRIER))
    expected = {
        UnitType.FIGHTER, UnitType.BOMBER, UnitType.STEALTH,
        UnitType.B_COPTER, UnitType.T_COPTER, UnitType.BLACK_BOMB,
    }
    assert loadable == expected, (
        f"Carrier cargo set drifted from AWBW: got {loadable!r}, want {expected!r}"
    )


def test_b5_carrier_rejects_ground_naval():
    loadable = set(get_loadable_into(UnitType.CARRIER))
    forbidden = {
        UnitType.INFANTRY, UnitType.MECH, UnitType.TANK, UnitType.MED_TANK,
        UnitType.RECON, UnitType.APC, UnitType.ARTILLERY,
        UnitType.CRUISER, UnitType.BLACK_BOAT, UnitType.LANDER, UnitType.SUBMARINE,
    }
    assert not (loadable & forbidden), (
        f"Carrier accepts non-air cargo: {loadable & forbidden!r}"
    )


# ---------------------------------------------------------------------------
# Relax: WAIT on capturable property no longer raises
# ---------------------------------------------------------------------------

def test_relax_wait_on_capturable_property_does_not_raise():
    """AWBW allows WAIT on an enemy property even when CAPTURE is also legal —
    the player can simply decline to capture. ``_apply_wait`` must not raise."""
    s = _blank_state()
    target_prop = next((p for p in s.properties if p.owner is None), None)
    assert target_prop is not None

    inf = _spawn(s, UnitType.INFANTRY, 0, (target_prop.row, target_prop.col))
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=inf.pos))
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=inf.pos, move_pos=inf.pos))

    # ``step`` must accept this WAIT without raising — even though the
    # ``get_legal_actions`` mask hides it for RL shaping.
    s.step(Action(ActionType.WAIT, unit_pos=inf.pos, move_pos=inf.pos))

    assert inf.moved is True
    assert target_prop.capture_points == 20  # WAIT does not start a capture


# ---------------------------------------------------------------------------
# Oracle: SELECT_UNIT disambiguation when drawable stack shares one tile
# ---------------------------------------------------------------------------


def test_select_unit_id_pins_engine_unit_when_tile_stacked():
    """Site replay can list two AWBW drawables on one tile; ``get_unit_at`` only
    sees the first engine unit — ``select_unit_id`` must pick the mover."""
    s = _blank_state()
    pos: tuple[int, int] | None = None
    for r, row in enumerate(s.map_data.terrain):
        for c, tid in enumerate(row):
            if tid == 1:
                pos = (r, c)
                break
        if pos is not None:
            break
    assert pos is not None
    u_first = _spawn(s, UnitType.INFANTRY, 0, pos, unit_id=101)
    u_second = _spawn(s, UnitType.INFANTRY, 0, pos, unit_id=202)
    assert s.get_unit_at(*pos) is u_first
    s.step(Action(ActionType.SELECT_UNIT, unit_pos=pos, select_unit_id=202))
    assert s.selected_unit is u_second
