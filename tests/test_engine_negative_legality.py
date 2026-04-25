"""Phase 4 NEG-TESTS — engine negative legality regressions.

Each test below asserts that ``GameState.step`` either rejects an illegal
action (negative tests) or accepts a legal one (positive guards). The vast
majority of negative tests are EXPECTED TO FAIL today: the engine ships its
legality rules in ``get_legal_actions`` / ``get_attack_targets`` but does not
yet enforce them inside ``step``. Phase 3 (STEP-GATE, SEAM, POWER+TURN+CAPTURE,
ATTACK-INV) closes those gaps. A red bar here is a Phase 3 acceptance criterion.

See ``.cursor/plans/desync_purge_engine_harden_d85bd82c.plan.md`` §"Phase 4
Thread NEG-TESTS" for the campaign context. Tests are deliberately
self-contained: each builds the smallest GameState the assertion needs.
"""
from __future__ import annotations

from typing import Optional

import pytest

from engine.action import (
    Action,
    ActionStage,
    ActionType,
    get_attack_targets,
)
from engine.co import make_co_state_safe
from engine.combat import get_seam_base_damage
from engine.game import GameState
from engine.map_loader import MapData, PropertyState
from engine.unit import Unit, UnitType, UNIT_STATS


# ---------------------------------------------------------------------------
# Terrain ID constants (see engine/terrain.py _build_table)
# ---------------------------------------------------------------------------
PLAIN = 1
MOUNTAIN = 2
SEA = 28
NEUTRAL_BASE = 35       # neutral factory
OS_BASE = 39            # Orange Star base (country_id=1)
BM_BASE = 44            # Blue Moon base (country_id=2)
HPIPE_SEAM = 113        # Horizontal pipe seam (intact)
HPIPE = 102             # Plain horizontal pipe (impassable except piperunner)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _make_state(
    *,
    width: int = 8,
    height: int = 1,
    terrain: Optional[list[list[int]]] = None,
    properties: Optional[list[PropertyState]] = None,
    units: Optional[dict[int, list[Unit]]] = None,
    funds: tuple[int, int] = (0, 0),
    p0_co: int = 1,
    p1_co: int = 1,
    active_player: int = 0,
    action_stage: ActionStage = ActionStage.SELECT,
    seam_hp: Optional[dict[tuple[int, int], int]] = None,
) -> GameState:
    """Build a minimal in-memory GameState. Terrain defaults to all PLAIN."""
    if terrain is None:
        terrain = [[PLAIN] * width for _ in range(height)]
    else:
        height = len(terrain)
        width = len(terrain[0])
    if properties is None:
        properties = []
    if units is None:
        units = {0: [], 1: []}
    if seam_hp is None:
        seam_hp = {
            (r, c): 99
            for r, row in enumerate(terrain)
            for c, tid in enumerate(row)
            if tid in (113, 114)
        }

    md = MapData(
        map_id=999_001,
        name="neg_legality_probe",
        map_type="std",
        terrain=[row[:] for row in terrain],
        height=height,
        width=width,
        cap_limit=999,
        unit_limit=999,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=properties,
        hq_positions={0: [], 1: []},
        lab_positions={0: [], 1: []},
        country_to_player={},
        predeployed_specs=[],
    )
    state = GameState(
        map_data=md,
        units=units,
        funds=[funds[0], funds[1]],
        co_states=[make_co_state_safe(p0_co), make_co_state_safe(p1_co)],
        properties=properties,
        turn=1,
        active_player=active_player,
        action_stage=action_stage,
        selected_unit=None,
        selected_move_pos=None,
        done=False,
        winner=None,
        win_reason=None,
        game_log=[],
        tier_name="T2",
        full_trace=[],
        seam_hp=seam_hp,
    )
    return state


_NEXT_UID = [1000]


def _spawn(
    state: GameState,
    ut: UnitType,
    player: int,
    pos: tuple[int, int],
    *,
    hp: int = 100,
    fuel: Optional[int] = None,
    ammo: Optional[int] = None,
    moved: bool = False,
    is_submerged: bool = False,
    loaded: Optional[list[Unit]] = None,
) -> Unit:
    stats = UNIT_STATS[ut]
    _NEXT_UID[0] += 1
    u = Unit(
        unit_type=ut,
        player=player,
        hp=hp,
        ammo=stats.max_ammo if ammo is None else ammo,
        fuel=stats.max_fuel if fuel is None else fuel,
        pos=pos,
        moved=moved,
        loaded_units=loaded or [],
        is_submerged=is_submerged,
        capture_progress=20,
        unit_id=_NEXT_UID[0],
    )
    state.units[player].append(u)
    return u


def _prop(
    row: int, col: int, terrain_id: int, owner: Optional[int],
    *, is_base: bool = False, is_airport: bool = False, is_port: bool = False,
    is_hq: bool = False, is_lab: bool = False, is_comm_tower: bool = False,
) -> PropertyState:
    return PropertyState(
        terrain_id=terrain_id, row=row, col=col, owner=owner, capture_points=20,
        is_hq=is_hq, is_lab=is_lab, is_comm_tower=is_comm_tower,
        is_base=is_base, is_airport=is_airport, is_port=is_port,
    )


# ---------------------------------------------------------------------------
# SEAM — indirect attackers (Phase 3 SEAM thread targets these)
# ---------------------------------------------------------------------------

def test_artillery_cannot_fire_on_pipe_seam():
    """Artillery is indirect; AWBW reserves seam attacks for direct fire +
    Piperunner. Engine currently routes it through ``_apply_seam_attack``."""
    terrain = [[PLAIN, PLAIN, PLAIN, HPIPE_SEAM, PLAIN]]
    state = _make_state(terrain=terrain)
    _spawn(state, UnitType.ARTILLERY, player=0, pos=(0, 0))
    action = Action(
        ActionType.ATTACK,
        unit_pos=(0, 0), move_pos=(0, 0), target_pos=(0, 3),
    )
    with pytest.raises(Exception):
        state.step(action)


def test_rocket_cannot_fire_on_pipe_seam():
    terrain = [[PLAIN] * 8]
    terrain[0][5] = HPIPE_SEAM
    state = _make_state(terrain=terrain)
    _spawn(state, UnitType.ROCKET, player=0, pos=(0, 0))
    action = Action(
        ActionType.ATTACK,
        unit_pos=(0, 0), move_pos=(0, 0), target_pos=(0, 5),
    )
    with pytest.raises(Exception):
        state.step(action)


@pytest.mark.xfail(strict=False, reason="pending Phase 3 SEAM canon decision: BB-on-seam may be canonical AWBW")
def test_battleship_cannot_fire_on_pipe_seam():
    """Battleship indirect on seam — canon-dependent. Phase 2.5 recon owns
    the verdict; xfail until SEAM thread decides."""
    terrain = [[SEA, SEA, SEA, SEA, HPIPE_SEAM]]
    state = _make_state(terrain=terrain)
    _spawn(state, UnitType.BATTLESHIP, player=0, pos=(0, 0))
    action = Action(
        ActionType.ATTACK,
        unit_pos=(0, 0), move_pos=(0, 0), target_pos=(0, 4),
    )
    with pytest.raises(Exception):
        state.step(action)


def test_mech_cannot_attack_adjacent_pipe_seam():
    """Commander's canonical example — Mech adjacent to a seam attempting
    to attack it. Engine ⊂ AWBW: must raise."""
    terrain = [[PLAIN, HPIPE_SEAM]]
    state = _make_state(terrain=terrain)
    _spawn(state, UnitType.MECH, player=0, pos=(0, 0))
    action = Action(
        ActionType.ATTACK,
        unit_pos=(0, 0), move_pos=(0, 0), target_pos=(0, 1),
    )
    with pytest.raises(Exception):
        state.step(action)


@pytest.mark.xfail(strict=False, reason="pending Phase 3 SEAM canon decision: AWBW direct-fire vs seam policy unclear for Infantry")
def test_infantry_cannot_attack_pipe_seam():
    terrain = [[PLAIN, HPIPE_SEAM]]
    state = _make_state(terrain=terrain)
    _spawn(state, UnitType.INFANTRY, player=0, pos=(0, 0))
    action = Action(
        ActionType.ATTACK,
        unit_pos=(0, 0), move_pos=(0, 0), target_pos=(0, 1),
    )
    with pytest.raises(Exception):
        state.step(action)


@pytest.mark.xfail(strict=False, reason="pending Phase 3 SEAM canon decision: direct-fire tanks may legitimately attack seams")
def test_tank_cannot_attack_pipe_seam():
    terrain = [[PLAIN, HPIPE_SEAM]]
    state = _make_state(terrain=terrain)
    _spawn(state, UnitType.TANK, player=0, pos=(0, 0))
    action = Action(
        ActionType.ATTACK,
        unit_pos=(0, 0), move_pos=(0, 0), target_pos=(0, 1),
    )
    with pytest.raises(Exception):
        state.step(action)


# ---------------------------------------------------------------------------
# CO power activation (Phase 3 POWER+TURN+CAPTURE)
# ---------------------------------------------------------------------------

def test_cop_activation_with_zero_power_bar_raises():
    state = _make_state()
    state.co_states[0].power_bar = 0
    assert not state.co_states[0].can_activate_cop()
    with pytest.raises(Exception):
        state.step(Action(ActionType.ACTIVATE_COP))


def test_scop_activation_with_insufficient_power_bar_raises():
    state = _make_state()
    state.co_states[0].power_bar = 100
    assert not state.co_states[0].can_activate_scop()
    with pytest.raises(Exception):
        state.step(Action(ActionType.ACTIVATE_SCOP))


# ---------------------------------------------------------------------------
# END_TURN with unmoved units (Phase 3 POWER+TURN+CAPTURE / STEP-GATE)
# ---------------------------------------------------------------------------

def test_end_turn_with_unmoved_infantry_raises():
    """END_TURN is illegal while any acting-seat unit still has ``moved=False``
    (ignoring loaded-transport carve-out). Engine currently lets it through."""
    state = _make_state()
    _spawn(state, UnitType.INFANTRY, player=0, pos=(0, 0), moved=False)
    with pytest.raises(Exception):
        state.step(Action(ActionType.END_TURN))


# ---------------------------------------------------------------------------
# BUILD legality (Phase 3 POWER+TURN+CAPTURE)
# ---------------------------------------------------------------------------

def test_build_on_enemy_owned_factory_raises():
    """A factory owned by P1 must never accept BUILD orders from P0."""
    terrain = [[BM_BASE, PLAIN]]
    properties = [_prop(0, 0, BM_BASE, owner=1, is_base=True)]
    state = _make_state(terrain=terrain, properties=properties, funds=(50_000, 0))
    action = Action(
        ActionType.BUILD,
        move_pos=(0, 0), unit_type=UnitType.TANK,
    )
    with pytest.raises(Exception):
        state.step(action)


def test_build_with_insufficient_funds_raises():
    terrain = [[OS_BASE, PLAIN]]
    properties = [_prop(0, 0, OS_BASE, owner=0, is_base=True)]
    state = _make_state(terrain=terrain, properties=properties, funds=(100, 0))
    action = Action(
        ActionType.BUILD,
        move_pos=(0, 0), unit_type=UnitType.TANK,  # Tank costs 7000
    )
    with pytest.raises(Exception):
        state.step(action)


# ---------------------------------------------------------------------------
# CAPTURE legality (Phase 3 POWER+TURN+CAPTURE)
# ---------------------------------------------------------------------------

def test_capture_by_tank_raises():
    """Only ``can_capture`` units (Infantry / Mech) may issue CAPTURE."""
    terrain = [[PLAIN, NEUTRAL_BASE]]
    properties = [_prop(0, 1, NEUTRAL_BASE, owner=None, is_base=True)]
    state = _make_state(terrain=terrain, properties=properties)
    _spawn(state, UnitType.TANK, player=0, pos=(0, 0))
    action = Action(
        ActionType.CAPTURE,
        unit_pos=(0, 0), move_pos=(0, 1),
    )
    with pytest.raises(Exception):
        state.step(action)


def test_capture_by_infantry_on_plain_ground_raises():
    """CAPTURE is only valid on a capturable property tile (no PropertyState)."""
    terrain = [[PLAIN, PLAIN]]
    state = _make_state(terrain=terrain)
    _spawn(state, UnitType.INFANTRY, player=0, pos=(0, 0))
    action = Action(
        ActionType.CAPTURE,
        unit_pos=(0, 0), move_pos=(0, 1),
    )
    with pytest.raises(Exception):
        state.step(action)


def test_capture_by_infantry_on_mountain_raises():
    """Mountains are not capturable property; CAPTURE must be rejected."""
    terrain = [[PLAIN, MOUNTAIN]]
    state = _make_state(terrain=terrain)
    _spawn(state, UnitType.INFANTRY, player=0, pos=(0, 0))
    action = Action(
        ActionType.CAPTURE,
        unit_pos=(0, 0), move_pos=(0, 1),
    )
    with pytest.raises(Exception):
        state.step(action)


# ---------------------------------------------------------------------------
# ATTACK range / friendly-fire (Phase 3 ATTACK-INV)
# ---------------------------------------------------------------------------

def test_direct_attack_outside_manhattan_1_raises():
    """Tank (range 1-1) cannot attack a target two tiles away.

    AWBW canon: direct units attack at Manhattan distance 1 only —
    the four orthogonal neighbours. See Phase 6 fix in
    ``logs/desync_regression_log.md``."""
    terrain = [[PLAIN, PLAIN, PLAIN]]
    state = _make_state(terrain=terrain)
    _spawn(state, UnitType.TANK, player=0, pos=(0, 0))
    _spawn(state, UnitType.INFANTRY, player=1, pos=(0, 2))
    action = Action(
        ActionType.ATTACK,
        unit_pos=(0, 0), move_pos=(0, 0), target_pos=(0, 2),
    )
    with pytest.raises(Exception):
        state.step(action)


def test_indirect_attack_outside_max_range_raises():
    """Artillery max_range=3; range-4 strike must be rejected."""
    terrain = [[PLAIN] * 6]
    state = _make_state(terrain=terrain)
    _spawn(state, UnitType.ARTILLERY, player=0, pos=(0, 0))
    _spawn(state, UnitType.INFANTRY, player=1, pos=(0, 4))
    action = Action(
        ActionType.ATTACK,
        unit_pos=(0, 0), move_pos=(0, 0), target_pos=(0, 4),
    )
    with pytest.raises(Exception):
        state.step(action)


def test_friendly_fire_attack_raises():
    """Same-player ATTACK is illegal; ``get_attack_targets`` already filters
    friendlies. Engine ``_apply_attack`` must mirror that."""
    terrain = [[PLAIN, PLAIN]]
    state = _make_state(terrain=terrain)
    _spawn(state, UnitType.TANK, player=0, pos=(0, 0))
    _spawn(state, UnitType.TANK, player=0, pos=(0, 1))
    action = Action(
        ActionType.ATTACK,
        unit_pos=(0, 0), move_pos=(0, 0), target_pos=(0, 1),
    )
    with pytest.raises(Exception):
        state.step(action)


def test_indirect_unit_move_then_attack_same_turn_raises():
    """Indirects (Artillery) cannot move and fire in the same turn — AWBW
    rule, also encoded in ``get_attack_targets`` (returns [] when move_pos
    differs from unit.pos)."""
    terrain = [[PLAIN] * 6]
    state = _make_state(terrain=terrain)
    _spawn(state, UnitType.ARTILLERY, player=0, pos=(0, 0))
    _spawn(state, UnitType.INFANTRY, player=1, pos=(0, 3))
    action = Action(
        ActionType.ATTACK,
        unit_pos=(0, 0), move_pos=(0, 1), target_pos=(0, 3),
    )
    with pytest.raises(Exception):
        state.step(action)


# ---------------------------------------------------------------------------
# JOIN legality (already raises today — positive-pass guards)
# ---------------------------------------------------------------------------

def test_join_with_mismatched_unit_types_raises():
    """Tank cannot JOIN onto Infantry — ``units_can_join`` rejects, and
    ``_apply_join`` already raises ValueError today."""
    terrain = [[PLAIN, PLAIN]]
    state = _make_state(terrain=terrain)
    _spawn(state, UnitType.TANK, player=0, pos=(0, 0))
    _spawn(state, UnitType.INFANTRY, player=0, pos=(0, 1), hp=50)
    action = Action(
        ActionType.JOIN,
        unit_pos=(0, 0), move_pos=(0, 1),
    )
    with pytest.raises(Exception):
        state.step(action)


def test_join_onto_full_hp_friendly_raises():
    """Two full-HP allies cannot merge — both at 100 internal HP."""
    terrain = [[PLAIN, PLAIN]]
    state = _make_state(terrain=terrain)
    _spawn(state, UnitType.INFANTRY, player=0, pos=(0, 0), hp=100)
    _spawn(state, UnitType.INFANTRY, player=0, pos=(0, 1), hp=100)
    action = Action(
        ActionType.JOIN,
        unit_pos=(0, 0), move_pos=(0, 1),
    )
    with pytest.raises(Exception):
        state.step(action)


# ---------------------------------------------------------------------------
# LOAD legality (already raises today — positive-pass guards)
# ---------------------------------------------------------------------------

def test_load_incompatible_cargo_tank_into_apc_raises():
    """APC carries Infantry/Mech only; Tank LOAD must raise."""
    terrain = [[PLAIN, PLAIN]]
    state = _make_state(terrain=terrain)
    _spawn(state, UnitType.TANK, player=0, pos=(0, 0))
    _spawn(state, UnitType.APC, player=0, pos=(0, 1))
    action = Action(
        ActionType.LOAD,
        unit_pos=(0, 0), move_pos=(0, 1),
    )
    with pytest.raises(Exception):
        state.step(action)


def test_load_infantry_into_cruiser_raises():
    """Cruiser carries B-Copter / T-Copter only; Infantry LOAD must raise."""
    terrain = [[SEA, SEA]]
    state = _make_state(terrain=terrain)
    _spawn(state, UnitType.INFANTRY, player=0, pos=(0, 0))
    _spawn(state, UnitType.CRUISER, player=0, pos=(0, 1))
    action = Action(
        ActionType.LOAD,
        unit_pos=(0, 0), move_pos=(0, 1),
    )
    with pytest.raises(Exception):
        state.step(action)


# ---------------------------------------------------------------------------
# UNLOAD legality (Phase 3 STEP-GATE / POWER+TURN+CAPTURE)
# ---------------------------------------------------------------------------

def _apc_with_cargo(state: GameState, pos: tuple[int, int]) -> Unit:
    cargo_stats = UNIT_STATS[UnitType.INFANTRY]
    _NEXT_UID[0] += 1
    cargo = Unit(
        unit_type=UnitType.INFANTRY,
        player=0,
        hp=100,
        ammo=cargo_stats.max_ammo,
        fuel=cargo_stats.max_fuel,
        pos=pos,
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
        unit_id=_NEXT_UID[0],
    )
    apc = _spawn(state, UnitType.APC, player=0, pos=pos, loaded=[cargo])
    return apc


def test_unload_to_non_adjacent_tile_raises():
    """UNLOAD drop tile must be Manhattan-1 from the transport's destination."""
    terrain = [[PLAIN] * 6]
    state = _make_state(terrain=terrain)
    _apc_with_cargo(state, (0, 1))
    action = Action(
        ActionType.UNLOAD,
        unit_pos=(0, 1), move_pos=(0, 1), target_pos=(0, 5),
        unit_type=UnitType.INFANTRY,
    )
    with pytest.raises(Exception):
        state.step(action)


def test_unload_to_occupied_tile_raises():
    """UNLOAD into a tile already occupied by another unit must raise."""
    terrain = [[PLAIN, PLAIN, PLAIN]]
    state = _make_state(terrain=terrain)
    _apc_with_cargo(state, (0, 1))
    _spawn(state, UnitType.INFANTRY, player=0, pos=(0, 2))
    action = Action(
        ActionType.UNLOAD,
        unit_pos=(0, 1), move_pos=(0, 1), target_pos=(0, 2),
        unit_type=UnitType.INFANTRY,
    )
    with pytest.raises(Exception):
        state.step(action)


# ---------------------------------------------------------------------------
# Black Boat REPAIR adjacency (proxy for "supply from APC to non-adjacent
# ally" — the engine has no targeted SUPPLY action; APC supply is implicit
# on WAIT and only adjacent. Black Boat REPAIR is the closest targeted
# heal/resupply primitive and exercises the same rule.)
# ---------------------------------------------------------------------------

def test_supply_from_apc_to_non_adjacent_ally_raises():
    terrain = [[SEA, SEA, SEA, SEA]]
    state = _make_state(terrain=terrain)
    _spawn(state, UnitType.BLACK_BOAT, player=0, pos=(0, 0))
    _spawn(state, UnitType.BATTLESHIP, player=0, pos=(0, 3), fuel=10, ammo=2)
    action = Action(
        ActionType.REPAIR,
        unit_pos=(0, 0), move_pos=(0, 0), target_pos=(0, 3),
    )
    with pytest.raises(Exception):
        state.step(action)


# ---------------------------------------------------------------------------
# Sub / Stealth dive fuel guard (Phase 3 STEP-GATE)
# ---------------------------------------------------------------------------

def test_sub_dive_with_low_fuel_raises():
    """Subs cannot dive below 5 fuel — submerged drain (+4/turn) would
    immediately sink them. AWBW blocks the action."""
    terrain = [[SEA, SEA]]
    state = _make_state(terrain=terrain)
    _spawn(state, UnitType.SUBMARINE, player=0, pos=(0, 0), fuel=3)
    action = Action(
        ActionType.DIVE_HIDE,
        unit_pos=(0, 0), move_pos=(0, 0),
    )
    with pytest.raises(Exception):
        state.step(action)


def test_stealth_hide_with_low_fuel_raises():
    """Stealth cannot hide below 5 fuel for the same reason as Subs."""
    terrain = [[PLAIN, PLAIN]]
    state = _make_state(terrain=terrain)
    _spawn(state, UnitType.STEALTH, player=0, pos=(0, 0), fuel=3)
    action = Action(
        ActionType.DIVE_HIDE,
        unit_pos=(0, 0), move_pos=(0, 0),
    )
    with pytest.raises(Exception):
        state.step(action)


# ---------------------------------------------------------------------------
# Positive guards — must remain LEGAL through Phase 3
# ---------------------------------------------------------------------------

def _walk_select_to_action(state: GameState, unit_pos, move_pos) -> None:
    """Walk SELECT → MOVE → ACTION via legal mask actions so the subsequent
    ATTACK / WAIT / etc. is issued from the correct stage. STEP-GATE
    (Phase 3) makes the mask authoritative for non-oracle callers; positive
    guards therefore must drive the pipeline rather than parachute an
    ACTION-stage Action onto a SELECT-stage state."""
    state.step(Action(ActionType.SELECT_UNIT, unit_pos=unit_pos))
    state.step(Action(ActionType.SELECT_UNIT, unit_pos=unit_pos, move_pos=move_pos))


@pytest.mark.parametrize(
    "unit_type",
    [
        UnitType.INFANTRY,
        UnitType.MECH,
        UnitType.RECON,
        UnitType.TANK,
        UnitType.MED_TANK,
        UnitType.NEO_TANK,
        UnitType.MEGA_TANK,
        UnitType.ANTI_AIR,
        UnitType.B_COPTER,
    ],
    ids=lambda u: u.name,
)
def test_direct_r1_unit_cannot_attack_diagonally(unit_type):
    """AWBW canon (Phase 6): direct range-1 units attack the four orthogonal
    neighbours only — never diagonals.

    Evidence: AWBW Wiki ("directly adjacent"), Carnaghi 2022 ("on axis not
    diagonally"), 936 GL std-tier replays = 62,614 direct-r1 Fire envelopes,
    zero diagonals. The inverse test ``test_mech_can_attack_diagonal_chebyshev_1``
    that previously codified the Chebyshev bug was deleted.
    """
    terrain = [[PLAIN, PLAIN], [PLAIN, PLAIN]]
    state = _make_state(terrain=terrain)
    _spawn(state, unit_type, player=0, pos=(0, 0))
    _spawn(state, UnitType.INFANTRY, player=1, pos=(1, 1))
    attacker = state.units[0][0]
    targets = get_attack_targets(state, attacker, (0, 0))
    assert (1, 1) not in targets, (
        f"{unit_type.name} diagonal (1,1) MUST NOT be in legal targets "
        f"from (0,0) — got targets={targets}"
    )
    _walk_select_to_action(state, unit_pos=(0, 0), move_pos=(0, 0))
    action = Action(
        ActionType.ATTACK,
        unit_pos=(0, 0), move_pos=(0, 0), target_pos=(1, 1),
    )
    with pytest.raises(Exception):
        state.step(action)


@pytest.mark.parametrize(
    "unit_type,target_pos",
    [
        (UnitType.INFANTRY, (0, 1)),
        (UnitType.INFANTRY, (1, 0)),
        (UnitType.MECH,     (0, 1)),
        (UnitType.RECON,    (1, 0)),
        (UnitType.TANK,     (0, 1)),
        (UnitType.MED_TANK, (1, 0)),
        (UnitType.B_COPTER, (0, 1)),
        (UnitType.ANTI_AIR, (1, 0)),
    ],
    ids=lambda v: str(v),
)
def test_direct_r1_unit_can_attack_orthogonally(unit_type, target_pos):
    """AWBW canon: direct range-1 units can attack any of the four orthogonal
    neighbours from their current tile (positive guard, complement to the
    diagonal-rejection test above)."""
    terrain = [[PLAIN, PLAIN], [PLAIN, PLAIN]]
    state = _make_state(terrain=terrain)
    _spawn(state, unit_type, player=0, pos=(0, 0))
    # Pick a defender unit type that the attacker has a damage entry for.
    defender_type = (
        UnitType.B_COPTER if unit_type in (UnitType.ANTI_AIR, UnitType.B_COPTER)
        else UnitType.INFANTRY
    )
    _spawn(state, defender_type, player=1, pos=target_pos)
    attacker = state.units[0][0]
    targets = get_attack_targets(state, attacker, (0, 0))
    assert target_pos in targets, (
        f"{unit_type.name} orth target {target_pos} must be in legal targets "
        f"from (0,0) — got targets={targets}"
    )
    _walk_select_to_action(state, unit_pos=(0, 0), move_pos=(0, 0))
    state.step(Action(
        ActionType.ATTACK,
        unit_pos=(0, 0), move_pos=(0, 0), target_pos=target_pos,
    ))


def test_piperunner_can_fire_on_pipe_seam_within_range():
    """Piperunner is the canonical seam attacker (AWBW Pipes & Pipeseams
    wiki); range 2-5 indirect. If Phase 3 SEAM bans this it will flip to
    a negative test."""
    terrain = [[HPIPE, HPIPE, HPIPE, HPIPE_SEAM]]
    state = _make_state(terrain=terrain)
    _spawn(state, UnitType.PIPERUNNER, player=0, pos=(0, 0))
    assert get_seam_base_damage(UnitType.PIPERUNNER) is not None
    _walk_select_to_action(state, unit_pos=(0, 0), move_pos=(0, 0))
    action = Action(
        ActionType.ATTACK,
        unit_pos=(0, 0), move_pos=(0, 0), target_pos=(0, 3),
    )
    state.step(action)


def test_direct_adjacent_attack_on_unit_standing_on_seam_tile():
    """Attacking the unit on a seam (not the seam itself) is legal direct
    fire — the seam tile is irrelevant; defender exists."""
    terrain = [[PLAIN, HPIPE_SEAM]]
    state = _make_state(terrain=terrain)
    _spawn(state, UnitType.TANK, player=0, pos=(0, 0))
    # Piperunner can stand on a seam (its native terrain).
    _spawn(state, UnitType.PIPERUNNER, player=1, pos=(0, 1))
    _walk_select_to_action(state, unit_pos=(0, 0), move_pos=(0, 0))
    action = Action(
        ActionType.ATTACK,
        unit_pos=(0, 0), move_pos=(0, 0), target_pos=(0, 1),
    )
    state.step(action)


def test_co_discount_build_applies():
    """Colin (CO 15) builds at 80% cost — Light Tank 7000 → 5600. Verify
    funds are debited at the discounted rate."""
    terrain = [[OS_BASE, PLAIN]]
    properties = [_prop(0, 0, OS_BASE, owner=0, is_base=True)]
    state = _make_state(
        terrain=terrain, properties=properties,
        funds=(7_000, 0), p0_co=15,
    )
    starting = state.funds[0]
    action = Action(
        ActionType.BUILD,
        move_pos=(0, 0), unit_type=UnitType.TANK,
    )
    state.step(action)
    spent = starting - state.funds[0]
    assert spent == 5_600, (
        f"Colin Tank discount: expected 5600 spent, got {spent} "
        f"(funds went {starting} → {state.funds[0]})"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
