"""
AWBW hierarchical action system.

Three-stage turn structure per unit:
  Stage 0 – SELECT : choose a unit (or END_TURN / activate power)
  Stage 1 – MOVE   : choose a destination tile
  Stage 2 – ACTION : ATTACK / CAPTURE / WAIT / LOAD / JOIN / UNLOAD / BUILD

Legal action generation is called at each stage. The GameState tracks
which stage it is in and which unit / move destination was selected.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from engine.game import GameState

from engine.unit import UnitType, UNIT_STATS, Unit
from engine.terrain import get_terrain, get_move_cost, INF_PASSABLE
from engine.weather import effective_move_cost


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionStage(IntEnum):
    SELECT = 0
    MOVE   = 1
    ACTION = 2


class ActionType(IntEnum):
    # Stage 0
    SELECT_UNIT  = 0
    END_TURN     = 1
    ACTIVATE_COP  = 2
    ACTIVATE_SCOP = 3
    # Stage 1 — move destination encoded in Action.move_pos
    # (reuses SELECT_UNIT with move_pos set, see _get_move_actions)
    # Stage 2
    ATTACK  = 10
    CAPTURE = 11
    WAIT    = 12
    LOAD    = 13
    UNLOAD  = 14
    BUILD   = 15
    # Black Boat "Repair" command (AWBW name): explicit one-target heal/resupply
    # of an orthogonally adjacent ally. ``target_pos`` = ally tile. Replaces
    # the old mass auto-repair that used to piggy-back on WAIT.
    REPAIR  = 16
    # Merge two damaged same-type allies: mover ends on partner tile (AWBW join).
    JOIN    = 17


# ---------------------------------------------------------------------------
# Action dataclass
# ---------------------------------------------------------------------------

@dataclass
class Action:
    action_type: ActionType
    unit_pos:   Optional[tuple[int, int]] = None   # position of acting unit
    move_pos:   Optional[tuple[int, int]] = None   # destination after move
    target_pos: Optional[tuple[int, int]] = None   # attack / unload target
    unit_type:  Optional[UnitType]        = None   # for BUILD
    unload_pos: Optional[tuple[int, int]] = None   # unload destination

    def __repr__(self) -> str:
        parts = [self.action_type.name]
        if self.unit_pos:  parts.append(f"from={self.unit_pos}")
        if self.move_pos:  parts.append(f"to={self.move_pos}")
        if self.target_pos: parts.append(f"tgt={self.target_pos}")
        if self.unit_type:  parts.append(f"unit={self.unit_type.name}")
        return f"Action({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Transport load-compatibility table
# ---------------------------------------------------------------------------

_LOADABLE_INTO: dict[UnitType, list[UnitType]] = {
    UnitType.APC: [UnitType.INFANTRY, UnitType.MECH],
    UnitType.T_COPTER: [UnitType.INFANTRY, UnitType.MECH],
    UnitType.LANDER: [
        UnitType.INFANTRY, UnitType.MECH, UnitType.RECON,
        UnitType.TANK, UnitType.MED_TANK, UnitType.NEO_TANK, UnitType.MEGA_TANK,
        UnitType.APC, UnitType.ARTILLERY, UnitType.ROCKET,
        UnitType.ANTI_AIR, UnitType.MISSILES,
    ],
    UnitType.CRUISER: [UnitType.B_COPTER, UnitType.T_COPTER],
    UnitType.BLACK_BOAT: [UnitType.INFANTRY, UnitType.MECH],
    UnitType.GUNBOAT: [UnitType.INFANTRY, UnitType.MECH],
}


def get_loadable_into(transport_type: UnitType) -> list[UnitType]:
    return _LOADABLE_INTO.get(transport_type, [])


def units_can_join(mover: Unit, occupant: Unit) -> bool:
    """True if ``mover`` may legally end on ``occupant`` to AWBW-join (merge).

    Requires same owner and ``UnitType``, an **injured** partner (``hp < 100``),
    and neither unit carrying cargo. Transport ``LOAD`` takes precedence when
    the mover can board the occupant.
    """
    if mover is occupant:
        return False
    if mover.player != occupant.player:
        return False
    if mover.unit_type != occupant.unit_type:
        return False
    if occupant.hp >= 100:
        return False
    if mover.loaded_units or occupant.loaded_units:
        return False
    return True


# ---------------------------------------------------------------------------
# Movement reachability (Dijkstra/BFS with fuel as cost)
# ---------------------------------------------------------------------------

def compute_reachable_costs(state: GameState, unit: Unit) -> dict[tuple[int, int], int]:
    """
    Return a mapping of legal end-tile positions to the minimum movement-point
    cost the unit pays to reach them. Movement cost equals the sum of terrain
    move-points along the cheapest path; this same value is consumed from
    ``unit.fuel`` when the engine commits the move (see ``_move_unit``).

    Rules:
    - Cap on movement: ``min(stats.move_range + CO bonuses, unit.fuel)``. A unit
      with zero fuel can only stay put.
    - Cannot pass through enemy-occupied tiles.
    - Cannot end on a friendly unit unless loading into a transport with room,
      or joining (same type) onto an injured ally.
    """
    stats      = UNIT_STATS[unit.unit_type]
    move_range = stats.move_range
    co         = state.co_states[unit.player]

    # Adder: +1 move (COP +1, SCOP +2 on top of DTD +1)
    if co.co_id == 11:
        move_range += 1
        if co.cop_active:
            move_range += 1
        if co.scop_active:
            move_range += 2

    # Eagle SCOP: copters/air get +2 move
    if co.co_id == 10 and co.scop_active:
        if stats.unit_class in ("air", "copter"):
            move_range += 2

    # Sami COP: infantry +1, SCOP +2
    if co.co_id == 8:
        if stats.unit_class == "infantry":
            if co.scop_active:
                move_range += 2
            elif co.cop_active:
                move_range += 1

    # Grimm SCOP: all ground +3
    if co.co_id == 20 and co.scop_active:
        if stats.unit_class in ("infantry", "mech", "vehicle", "pipe"):
            move_range += 3

    # Fuel hard-caps movement: a unit cannot spend more MP than it has fuel.
    move_range = min(move_range, unit.fuel)

    start   = unit.pos
    visited: dict[tuple[int, int], int] = {start: 0}
    queue: collections.deque[tuple[tuple[int, int], int]] = collections.deque([(start, 0)])

    while queue:
        (r, c), fuel_used = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < state.map_data.height and 0 <= nc < state.map_data.width):
                continue
            tid  = state.map_data.terrain[nr][nc]
            cost = effective_move_cost(state, unit, tid)
            if cost >= INF_PASSABLE:
                continue
            new_fuel = fuel_used + cost
            if new_fuel > move_range:
                continue

            # Cannot pass through enemy units
            occupant = state.get_unit_at(nr, nc)
            if occupant is not None and occupant.player != unit.player:
                continue

            if (nr, nc) not in visited or visited[(nr, nc)] > new_fuel:
                visited[(nr, nc)] = new_fuel
                queue.append(((nr, nc), new_fuel))

    # Filter to tiles where the unit can legally stop, preserving the cost.
    result: dict[tuple[int, int], int] = {}
    for pos, cost in visited.items():
        occupant = state.get_unit_at(*pos)
        if occupant is None or pos == unit.pos:
            result[pos] = cost
        elif occupant.player == unit.player:
            # Can stop here only to load into this transport
            cap = UNIT_STATS[occupant.unit_type].carry_capacity
            if cap > 0 and unit.unit_type in get_loadable_into(occupant.unit_type):
                if len(occupant.loaded_units) < cap:
                    result[pos] = cost
            elif units_can_join(unit, occupant):
                result[pos] = cost

    return result


def get_reachable_tiles(state: GameState, unit: Unit) -> set[tuple[int, int]]:
    """Backwards-compatible wrapper around :func:`compute_reachable_costs`."""
    return set(compute_reachable_costs(state, unit).keys())


# ---------------------------------------------------------------------------
# Attack target enumeration
# ---------------------------------------------------------------------------

def get_attack_targets(
    state: GameState,
    unit: Unit,
    move_pos: tuple[int, int],
) -> list[tuple[int, int]]:
    """
    Return all enemy tile positions this unit can attack from move_pos.
    Respects ammo, attack range, and submarine visibility rules.

    Pipe seams (terrain 113/114) are also returned when the tile is empty
    and the unit has a non-zero base damage vs seams (AWBW: seams are
    legitimate attack targets without any defender unit present).
    """
    from engine.combat import get_base_damage, get_seam_base_damage  # local import

    stats = UNIT_STATS[unit.unit_type]
    if stats.max_ammo == 0:
        return []
    if unit.ammo == 0 and stats.max_ammo > 0:
        return []

    # Indirect units cannot move and attack in the same turn
    if stats.is_indirect and move_pos != unit.pos:
        return []

    targets: list[tuple[int, int]] = []
    min_r, max_r = stats.min_range, stats.max_range
    mr, mc = move_pos

    for dr in range(-max_r, max_r + 1):
        for dc in range(-max_r, max_r + 1):
            dist = abs(dr) + abs(dc)
            if dist < min_r or dist > max_r:
                continue
            tr, tc = mr + dr, mc + dc
            if not (0 <= tr < state.map_data.height and 0 <= tc < state.map_data.width):
                continue
            enemy = state.get_unit_at(tr, tc)
            if enemy is not None:
                if enemy.player == unit.player:
                    continue
                if enemy.is_submerged and not _can_detect_submerged(unit):
                    continue
                if get_base_damage(unit.unit_type, enemy.unit_type) is not None:
                    targets.append((tr, tc))
                continue

            # No defender: still targetable if this tile is an intact pipe seam
            # and the attacker has a seam damage entry.
            tid = state.map_data.terrain[tr][tc]
            if tid in (113, 114):
                seam_base = get_seam_base_damage(unit.unit_type)
                if seam_base is not None and seam_base > 0:
                    targets.append((tr, tc))

    return targets


def _can_detect_submerged(unit: Unit) -> bool:
    """Naval units (non-sub) can spot submerged submarines."""
    cls = UNIT_STATS[unit.unit_type].unit_class
    return cls == "naval" and not UNIT_STATS[unit.unit_type].is_submarine


# ---------------------------------------------------------------------------
# Producible units per building type
# ---------------------------------------------------------------------------

_GROUND_UNITS: list[UnitType] = [
    UnitType.INFANTRY, UnitType.MECH, UnitType.RECON,
    UnitType.TANK, UnitType.MED_TANK, UnitType.NEO_TANK, UnitType.MEGA_TANK,
    UnitType.APC, UnitType.ARTILLERY, UnitType.ROCKET,
    UnitType.ANTI_AIR, UnitType.MISSILES,
]
_AIR_UNITS: list[UnitType] = [
    UnitType.FIGHTER, UnitType.BOMBER, UnitType.STEALTH,
    UnitType.B_COPTER, UnitType.T_COPTER, UnitType.BLACK_BOMB,
]
_NAVAL_UNITS: list[UnitType] = [
    UnitType.BATTLESHIP, UnitType.CARRIER, UnitType.SUBMARINE,
    UnitType.CRUISER, UnitType.LANDER, UnitType.GUNBOAT, UnitType.BLACK_BOAT,
]

_BAN_MAP: dict[str, UnitType] = {
    "Black Bomb":  UnitType.BLACK_BOMB,
    "Stealth":     UnitType.STEALTH,
    "Piperunner":  UnitType.PIPERUNNER,
    "Oozium":      UnitType.OOZIUM,
}


def get_producible_units(terrain_info, unit_bans: list[str]) -> list[UnitType]:
    banned = {_BAN_MAP[b] for b in unit_bans if b in _BAN_MAP}
    if terrain_info.is_base:
        return [u for u in _GROUND_UNITS if u not in banned]
    if terrain_info.is_airport:
        return [u for u in _AIR_UNITS if u not in banned]
    if terrain_info.is_port:
        return [u for u in _NAVAL_UNITS if u not in banned]
    return []


# ---------------------------------------------------------------------------
# Top-level legal action generator
# ---------------------------------------------------------------------------

def get_legal_actions(state: GameState) -> list[Action]:
    player = state.active_player
    if state.action_stage == ActionStage.SELECT:
        return _get_select_actions(state, player)
    if state.action_stage == ActionStage.MOVE:
        return _get_move_actions(state, player)
    if state.action_stage == ActionStage.ACTION:
        return _get_action_actions(state, player)
    return []


def _get_select_actions(state: GameState, player: int) -> list[Action]:
    actions: list[Action] = []

    co = state.co_states[player]
    if co.can_activate_cop():
        actions.append(Action(ActionType.ACTIVATE_COP))
    if co.can_activate_scop():
        actions.append(Action(ActionType.ACTIVATE_SCOP))

    has_unmoved = False
    for unit in state.units[player]:
        if not unit.moved:
            has_unmoved = True
            actions.append(Action(ActionType.SELECT_UNIT, unit_pos=unit.pos))

    # END_TURN is only legal once every friendly unit has acted (or if there
    # are no units at all). This forces the agent to move every unit each turn;
    # choosing a no-op is still possible via SELECT_UNIT → WAIT on the same tile.
    if not has_unmoved:
        actions.append(Action(ActionType.END_TURN))

    # Direct factory BUILD actions (AWBW-correct: factories build without unit activation)
    # Generate BUILD actions for each owned empty factory/airport/port
    if len(state.units[player]) < state.map_data.unit_limit:
        for prop in state.properties:
            if prop.owner == player:
                terrain = get_terrain(state.map_data.terrain[prop.row][prop.col])
                if terrain.is_base or terrain.is_airport or terrain.is_port:
                    # Check if factory tile is empty
                    if state.get_unit_at(prop.row, prop.col) is None:
                        # Generate BUILD actions for all affordable units
                        for ut in get_producible_units(terrain, state.map_data.unit_bans):
                            cost = _build_cost(ut, state, player, (prop.row, prop.col))
                            if state.funds[player] >= cost:
                                actions.append(Action(
                                    ActionType.BUILD,
                                    unit_pos=None,  # No unit required for factory builds
                                    move_pos=(prop.row, prop.col),
                                    unit_type=ut,
                                ))

    return actions


def _get_move_actions(state: GameState, player: int) -> list[Action]:
    unit = state.selected_unit
    if unit is None:
        return []
    reachable = get_reachable_tiles(state, unit)
    return [
        Action(ActionType.SELECT_UNIT, unit_pos=unit.pos, move_pos=pos)
        for pos in reachable
    ]


def _get_action_actions(state: GameState, player: int) -> list[Action]:
    unit     = state.selected_unit
    move_pos = state.selected_move_pos
    if unit is None or move_pos is None:
        return []

    # If the destination is a friendly transport (not the unit itself), the
    # *only* legal terminator is LOAD. Allowing WAIT here would co-occupy the
    # tile, leaving two drawable units on one square in snapshots.
    occupant = state.get_unit_at(*move_pos)
    boarding = (
        occupant is not None
        and occupant.player == player
        and occupant.pos != unit.pos
    )
    if boarding:
        cap = UNIT_STATS[occupant.unit_type].carry_capacity
        if (
            cap > 0
            and unit.unit_type in get_loadable_into(occupant.unit_type)
            and len(occupant.loaded_units) < cap
        ):
            return [Action(ActionType.LOAD, unit_pos=unit.pos, move_pos=move_pos)]
        if units_can_join(unit, occupant):
            return [Action(ActionType.JOIN, unit_pos=unit.pos, move_pos=move_pos)]
        # Friendly non-loadable occupant should never be a reachable end-tile,
        # but fail closed if it ever is.
        return []

    actions: list[Action] = [Action(ActionType.WAIT, unit_pos=unit.pos, move_pos=move_pos)]

    # --- Attack ---
    for tpos in get_attack_targets(state, unit, move_pos):
        actions.append(Action(
            ActionType.ATTACK,
            unit_pos=unit.pos,
            move_pos=move_pos,
            target_pos=tpos,
        ))

    # --- Capture ---
    dest_tid  = state.map_data.terrain[move_pos[0]][move_pos[1]]
    dest_info = get_terrain(dest_tid)
    stats     = UNIT_STATS[unit.unit_type]

    if dest_info.is_property and stats.can_capture:
        prop = state.get_property_at(*move_pos)
        if prop is not None and prop.owner != player:
            actions.append(Action(ActionType.CAPTURE, unit_pos=unit.pos, move_pos=move_pos))

    # --- Unload (transport with cargo) ---
    if stats.carry_capacity > 0 and unit.loaded_units:
        for cargo in unit.loaded_units:
            cargo_stats = UNIT_STATS[cargo.unit_type]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                tr, tc = move_pos[0] + dr, move_pos[1] + dc
                if not (0 <= tr < state.map_data.height and 0 <= tc < state.map_data.width):
                    continue
                tid = state.map_data.terrain[tr][tc]
                if effective_move_cost(state, cargo, tid) >= INF_PASSABLE:
                    continue
                # Drop tile must be empty (or be the transport's own pre-move tile,
                # which will become empty once the transport vacates).
                drop_occupant = state.get_unit_at(tr, tc)
                if drop_occupant is not None and drop_occupant.pos != unit.pos:
                    continue
                actions.append(Action(
                    ActionType.UNLOAD,
                    unit_pos=unit.pos,
                    move_pos=move_pos,
                    target_pos=(tr, tc),
                    unit_type=cargo.unit_type,
                ))

    # --- Build (from an owned base/airport/port — unit must be one being built; the
    #     BUILD action is issued by the producing unit which is the *factory itself* in
    #     AWBW, but for simplicity here the active player issues BUILD from the factory
    #     tile; this works when the game loop checks "is there a factory here I own?" ---
    if dest_info.is_base or dest_info.is_airport or dest_info.is_port:
        prop_at = state.get_property_at(*move_pos)
        if prop_at is not None and prop_at.owner == player:
            existing = state.get_unit_at(*move_pos)
            # Must match ``GameState._apply_build``: spawn only on an *empty*
            # factory tile. If ``existing`` is the acting unit, BUILD is legal in
            # the mask but ``step`` no-ops — the policy can spin in ACTION forever.
            if existing is None:
                if len(state.units[player]) < state.map_data.unit_limit:
                    for ut in get_producible_units(dest_info, state.map_data.unit_bans):
                        cost = _build_cost(ut, state, player, move_pos)
                        if state.funds[player] >= cost:
                            actions.append(Action(
                                ActionType.BUILD,
                                unit_pos=unit.pos,
                                move_pos=move_pos,
                                unit_type=ut,
                            ))

    # --- Black Boat REPAIR (one per orthogonally adjacent ally) ---
    # AWBW "Repair" command: 1 HP / 10% target cost / resupply fuel+ammo.
    # One action per eligible neighbour; the agent picks which ally to heal.
    # Eligibility follows ``_black_boat_repair_eligible``: ally not the boat,
    # not submerged/airborne cargo, and either below max HP *or* needing
    # fuel / ammo (resupply-only is still a meaningful action).
    if unit.unit_type == UnitType.BLACK_BOAT:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            tr, tc = move_pos[0] + dr, move_pos[1] + dc
            if not (0 <= tr < state.map_data.height and 0 <= tc < state.map_data.width):
                continue
            ally = state.get_unit_at(tr, tc)
            if ally is None or ally.player != unit.player or ally is unit:
                continue
            if _black_boat_repair_eligible(state, ally):
                actions.append(Action(
                    ActionType.REPAIR,
                    unit_pos=unit.pos,
                    move_pos=move_pos,
                    target_pos=(tr, tc),
                ))

    # --- Prune WAIT for capture-capable units on neutral/enemy property ---
    # Infantry/mech standing on a building they could act on must either attack
    # or capture — no idling on contested ground. Missile silos (is_property
    # False) and owned buildings are untouched; if capture is structurally
    # impossible (e.g. no PropertyState for this tile) we leave WAIT in place
    # so the unit is never deadlocked.
    if stats.can_capture and dest_info.is_property:
        prop_here = state.get_property_at(*move_pos)
        if prop_here is not None and prop_here.owner != player:
            has_attack = any(a.action_type == ActionType.ATTACK for a in actions)
            has_capture = any(a.action_type == ActionType.CAPTURE for a in actions)
            if has_attack or has_capture:
                pruned = [a for a in actions if a.action_type != ActionType.WAIT]
                if pruned:
                    actions = pruned

    # --- Prune no-op WAIT for empty APCs when a better resupply tile exists ---
    # AWBW APCs resupply every adjacent allied unit at the end of WAIT (all
    # classes — ground, air, naval — see ``GameState._apc_resupply``). An APC
    # WAITing on a tile with no adjacent allied unit needing fuel or ammo is a
    # strictly dominated no-op *if* another reachable tile would trigger a
    # real resupply. Cargo drops are preserved: UNLOAD stays, and an APC with
    # cargo aboard still keeps WAIT (the cargo is the productive payload).
    # Never prune WAIT when it is the only legal ACTION terminator (MOVE
    # already committed; dominated-move theory cannot apply with no alternative).
    if (
        unit.unit_type == UnitType.APC
        and not unit.loaded_units
        and not any(a.action_type == ActionType.UNLOAD for a in actions)
        and not _apc_tile_benefits_allies(state, unit, move_pos)
    ):
        reachable = compute_reachable_costs(state, unit)
        if any(
            pos != move_pos and _apc_tile_benefits_allies(state, unit, pos)
            for pos in reachable
        ):
            pruned = [a for a in actions if a.action_type != ActionType.WAIT]
            if pruned:
                actions = pruned

    return actions


def _apc_tile_benefits_allies(
    state: GameState,
    apc: Unit,
    tile: tuple[int, int],
) -> bool:
    """Return True if a WAIT by ``apc`` at ``tile`` would refuel or rearm any
    adjacent allied unit. Mirrors the filter in ``GameState._apc_resupply``:
    all allied unit classes are eligible (AWBW rule), so we only check
    whether anyone is below ``max_fuel`` / ``max_ammo``.
    """
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        adj = state.get_unit_at(tile[0] + dr, tile[1] + dc)
        if adj is None or adj.player != apc.player or adj is apc:
            continue
        astats = UNIT_STATS[adj.unit_type]
        if adj.fuel < astats.max_fuel:
            return True
        if astats.max_ammo > 0 and adj.ammo < astats.max_ammo:
            return True
    return False


def _black_boat_repair_eligible(state: GameState, ally: Unit) -> bool:
    """Return True if ``ally`` would benefit from a Black Boat REPAIR.

    Eligibility mirrors the AWBW wiki: the target must be an ally below
    max HP (so the heal applies) *or* needing fuel / ammo (so resupply
    applies). Either branch is enough because AWBW always resupplies even
    when the HP heal is skipped (unaffordable or full HP).
    """
    stats = UNIT_STATS[ally.unit_type]
    if ally.hp < 100:
        return True
    if ally.fuel < stats.max_fuel:
        return True
    if stats.max_ammo > 0 and ally.ammo < stats.max_ammo:
        return True
    return False


def _build_cost(ut: UnitType, state: GameState, player: int, pos: tuple[int, int]) -> int:
    """Adjusted build cost after CO modifiers."""
    cost = UNIT_STATS[ut].cost
    co   = state.co_states[player]
    if co.co_id == 3:            # Kanbei: 120% cost
        cost = int(cost * 1.2)
    elif co.co_id == 15:        # Colin: 80% cost
        cost = int(cost * 0.8)
    elif co.co_id == 17:        # Hachi: 50% from cities (bases in AWBW context)
        terrain = get_terrain(state.map_data.terrain[pos[0]][pos[1]])
        if terrain.is_base:
            cost = cost // 2
    return cost
