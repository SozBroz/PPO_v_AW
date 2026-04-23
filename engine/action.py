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

import os as _os

# Opt-in RL training: when set (see _get_move_actions), infantry/mech MOVE
# masks are restricted to capturable enemy/neutral property tiles when any
# exist in range. Default OFF — replays call step() directly and never hit
# get_legal_actions, so they are unaffected.
_CAPTURE_MOVE_GATE_ENV = "AWBW_CAPTURE_MOVE_GATE"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionStage(IntEnum):
    SELECT = 0
    MOVE   = 1
    ACTION = 2


# RL legality contract:
# - This enum is the COMPLETE list of actions the RL bot may emit.
# - "Delete Unit" (AWBW player-issued unit scrap) is INTENTIONALLY ABSENT.
#   AWBW replays use it to free production tiles, but giving the RL bot the
#   ability to scrap its own units enables degenerate self-trading strategies
#   (scrap blocker -> spawn stronger replacement -> repeat).
# - The Delete action is reproduced ONLY by the replay oracle
#   (tools/oracle_zip_replay.py::_oracle_kill_friendly_unit), which is never
#   imported by the engine and never reachable from get_legal_actions().
# - DO NOT add a DELETE member here without explicit Imperator approval.
# - Phase 11J-DELETE-GUARD-PIN regression tests in
#   tests/test_no_delete_action_legality.py enforce this contract at runtime.
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
    # Sub "Dive" / Stealth "Hide": toggle ``Unit.is_submerged`` after movement
    # (https://awbw.fandom.com/wiki/Sub , https://awbw.fandom.com/wiki/Stealth).
    DIVE_HIDE = 18
    # Immediate forfeit (oracle / AWBW replay ``Resign``). Not exposed in RL legal actions.
    RESIGN = 19


# ---------------------------------------------------------------------------
# RL action-space allowlist (Phase 11J-RL-DELETE-GUARD-SHIP)
# ---------------------------------------------------------------------------
# CANONICAL set of ActionTypes the RL agent is permitted to emit. Any
# ``get_legal_actions()`` output MUST be a subset of this set; the dispatcher
# below enforces it at runtime. Pinned here to lock out oracle-only constructs
# (e.g. AWBW "Delete Unit", reproduced by tools/oracle_zip_replay.py via
# ``_oracle_kill_friendly_unit``, which is NOT a legal RL action).
_RL_LEGAL_ACTION_TYPES: frozenset = frozenset({
    ActionType.SELECT_UNIT,
    ActionType.END_TURN,
    ActionType.ACTIVATE_COP,
    ActionType.ACTIVATE_SCOP,
    ActionType.ATTACK,
    ActionType.CAPTURE,
    ActionType.WAIT,
    ActionType.LOAD,
    ActionType.UNLOAD,
    ActionType.BUILD,
    ActionType.REPAIR,
    ActionType.JOIN,
    ActionType.DIVE_HIDE,
    # NOTE: ActionType.RESIGN is intentionally excluded — replays can encode
    # an explicit forfeit but the RL agent must never voluntarily resign.
    # NOTE: there is no ActionType.DELETE — AWBW "Delete Unit" is oracle-path
    # only (see tools/oracle_zip_replay.py::_oracle_kill_friendly_unit).
})


# ---------------------------------------------------------------------------
# Action dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Action:
    action_type: ActionType
    unit_pos:   Optional[tuple[int, int]] = None   # position of acting unit
    move_pos:   Optional[tuple[int, int]] = None   # destination after move
    target_pos: Optional[tuple[int, int]] = None   # attack / unload target
    unit_type:  Optional[UnitType]        = None   # for BUILD
    unload_pos: Optional[tuple[int, int]] = None   # unload destination
    # Oracle / replay: disambiguate when multiple engine units share ``unit_pos``
    # (AWBW drawable stack — oracle_zip_replay ``_apply_move_paths_then_terminator``).
    select_unit_id: Optional[int] = None

    def __repr__(self) -> str:
        parts = [self.action_type.name]
        if self.unit_pos:  parts.append(f"from={self.unit_pos}")
        if self.move_pos:  parts.append(f"to={self.move_pos}")
        if self.target_pos: parts.append(f"tgt={self.target_pos}")
        if self.unit_type is not None:
            parts.append(f"unit={self.unit_type.name}")
        return f"Action({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Transport load-compatibility table
# ---------------------------------------------------------------------------

_LOADABLE_INTO: dict[UnitType, list[UnitType]] = {
    # AWBW: APC carries foot soldiers only (Infantry + Mech), never vehicles.
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
    # AWBW Carrier holds 2 air units of any class. https://awbw.fandom.com/wiki/Carrier
    UnitType.CARRIER: [
        UnitType.FIGHTER, UnitType.BOMBER, UnitType.STEALTH,
        UnitType.B_COPTER, UnitType.T_COPTER, UnitType.BLACK_BOMB,
    ],
}


def get_loadable_into(transport_type: UnitType) -> list[UnitType]:
    return _LOADABLE_INTO.get(transport_type, [])


def units_can_join(mover: Unit, occupant: Unit) -> bool:
    """True if ``mover`` may legally end on ``occupant`` to AWBW-join (merge).

    Requires same owner and ``UnitType``, **at least one** of the two below
    full HP (AWBW allows joining a full-HP partner with a damaged mover and
    vice versa — only the both-full case is forbidden), and neither unit
    carrying cargo. Transport ``LOAD`` takes precedence when the mover can
    board the occupant.
    """
    if mover is occupant:
        return False
    if mover.player != occupant.player:
        return False
    if mover.unit_type != occupant.unit_type:
        return False
    if mover.hp >= 100 and occupant.hp >= 100:
        return False
    if mover.loaded_units or occupant.loaded_units:
        return False
    return True


# ---------------------------------------------------------------------------
# Per-call unit occupancy (Phase 2c / 2d): (row, col) -> Unit for O(1) tile lookup
# ---------------------------------------------------------------------------

def _build_occupancy(state: GameState) -> dict[tuple[int, int], Unit]:
    """Build per-call (row, col) -> Unit lookup over alive units in both players.

    Phase 2c (extended Phase 2d): replaces O(N) state.get_unit_at scans with
    O(1) dict lookups. Caller must guarantee state.units is not mutated for
    the lifetime of the dict (true throughout get_legal_actions and helpers).
    """
    occ: dict[tuple[int, int], Unit] = {}
    for player_units in state.units.values():
        for u in player_units:
            if u.is_alive:
                occ[u.pos] = u
    return occ


# ---------------------------------------------------------------------------
# Movement reachability (Dijkstra/BFS with fuel as cost)
# ---------------------------------------------------------------------------

def compute_reachable_costs(
    state: GameState,
    unit: Unit,
    *,
    occupancy: dict[tuple[int, int], Unit] | None = None,
) -> dict[tuple[int, int], int]:
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
    if occupancy is None:
        occupancy = _build_occupancy(state)

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

    # Jess COP/SCOP: +2 movement for all vehicles (Turbo Charge / Overdrive).
    if co.co_id == 14 and (co.cop_active or co.scop_active):
        if stats.unit_class == "vehicle":
            move_range += 2

    # Andy SCOP (Hyper Upgrade): +1 movement for all units (AWBW Power envelope
    # ``global.units_movement_points``; COP is heal-only — no movement bonus).
    if co.co_id == 1 and co.scop_active:
        move_range += 1

    # Koal COP (Forced March): +1 movement to all own units globally. The wiki
    # text "+1 on road tiles" is misleading; live AWBW Power envelopes for Koal
    # COP carry ``global.units_movement_points: 1`` and unit snapshots show
    # ``movement_points = base + 1`` regardless of starting tile (Phase 11D-F2
    # recon, gids 1605367 and 1630794, both with no roads on the failing path).
    # The road -1 cost discount is applied separately in
    # ``engine/weather.py::effective_move_cost`` and stacks with this bonus.
    # SCOP "Trail of Woe" is intentionally NOT bumped here: weather.py already
    # applies -2 cost per road tile, which is sufficient for the SCOP's road
    # behavior; a global +2 has not been confirmed by replay evidence.
    if co.co_id == 21 and co.cop_active:
        move_range += 1

    # Fuel hard-caps movement: a unit cannot spend more MP than it has fuel.
    move_range = min(move_range, unit.fuel)

    start   = unit.pos
    visited: dict[tuple[int, int], int] = {start: 0}
    queue: collections.deque[tuple[tuple[int, int], int]] = collections.deque([(start, 0)])

    # Phase 2b: per-call effective_move_cost memoization. unit and state are fixed
    # for this BFS; the same tid is hit many times during neighbor expansion.
    # Cache lives only inside this function call's stack frame -> no cross-call
    # invalidation risk (weather/CO changes invalidate by re-entering this fn).
    # Phase 2e: lookup inlined into the BFS loop body to remove ~200 closure
    # invocations per BFS call (1M+ over a single-proc 5000-step microbench);
    # behavior is byte-identical to the prior _cached_cost helper.
    _cost_cache: dict[int, int] = {}

    while queue:
        (r, c), fuel_used = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < state.map_data.height and 0 <= nc < state.map_data.width):
                continue
            tid  = state.map_data.terrain[nr][nc]
            cost = _cost_cache.get(tid)
            if cost is None:
                cost = effective_move_cost(state, unit, tid)
                _cost_cache[tid] = cost
            if cost >= INF_PASSABLE:
                continue
            new_fuel = fuel_used + cost
            if new_fuel > move_range:
                continue

            # Cannot pass through enemy units
            occupant = occupancy.get((nr, nc))
            if occupant is not None and occupant.player != unit.player:
                continue

            if (nr, nc) not in visited or visited[(nr, nc)] > new_fuel:
                visited[(nr, nc)] = new_fuel
                queue.append(((nr, nc), new_fuel))

    # Filter to tiles where the unit can legally stop, preserving the cost.
    result: dict[tuple[int, int], int] = {}
    for pos, cost in visited.items():
        occupant = occupancy.get(pos)
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
    *,
    occupancy: dict[tuple[int, int], Unit] | None = None,
) -> list[tuple[int, int]]:
    """
    Return all enemy tile positions this unit can attack from move_pos.
    Respects ammo, attack range, and submarine visibility rules.

    Pipe seams (terrain 113/114) are also returned when the tile is empty
    and the unit has a non-zero base damage vs seams (AWBW: seams are
    legitimate attack targets without any defender unit present).
    """
    if occupancy is None:
        occupancy = _build_occupancy(state)
    from engine.combat import get_base_damage, get_seam_base_damage  # local import

    stats = UNIT_STATS[unit.unit_type]
    # max_ammo == 0: MG-only / transport (AWBW chart) — still attack if the
    # damage table allows; no magazine to exhaust.
    if stats.max_ammo > 0 and unit.ammo == 0:
        return []

    # Indirect units cannot move and attack in the same turn
    if stats.is_indirect and move_pos != unit.pos:
        return []

    targets: list[tuple[int, int]] = []
    min_r, max_r = stats.min_range, stats.max_range
    # CO powers extend **max** indirect range (AWBW: Grit COP +1 / SCOP +2; Jake
    # COP/SCOP +1 land indirects only — not Battleship).
    if stats.is_indirect:
        co = state.co_states[int(unit.player)]
        if co.co_id == 2:  # Grit — all indirects including naval
            if co.scop_active:
                max_r += 2
            elif co.cop_active:
                max_r += 1
        elif co.co_id == 22 and (co.cop_active or co.scop_active):
            if stats.unit_class != "naval":
                max_r += 1
    mr, mc = move_pos

    for dr in range(-max_r, max_r + 1):
        for dc in range(-max_r, max_r + 1):
            # AWBW canon: attack range is **Manhattan** distance for ALL units.
            # Direct (min=max=1) hits the four orthogonal neighbours only —
            # NEVER diagonals. Indirects use the standard min..max Manhattan
            # ring. Verified against AWBW Wiki (Units / Overview), Carnaghi
            # 2022 ("on axis not diagonally"), and 936 GL std-tier replays
            # (62,614 direct-r1 Fire envelopes, zero diagonals). The prior
            # Chebyshev-1 special case for direct r1 was the Phase 6 bug.
            dist = abs(dr) + abs(dc)
            if dist < min_r or dist > max_r:
                continue
            tr, tc = mr + dr, mc + dc
            if not (0 <= tr < state.map_data.height and 0 <= tc < state.map_data.width):
                continue
            enemy = occupancy.get((tr, tc))
            if enemy is not None:
                if enemy.player == unit.player:
                    continue
                if enemy.is_submerged and not _can_attack_submerged_or_hidden(unit, enemy):
                    continue
                if get_base_damage(unit.unit_type, enemy.unit_type) is not None:
                    targets.append((tr, tc))
                continue

            # No defender: still targetable if this tile is an intact pipe seam
            # and the attacker has a seam damage entry.
            tid = state.map_data.terrain[tr][tc]
            # Intact seams (113/114) and broken rubble (115/116) use the same seam
            # damage chart on AWBW; replays may show AttackSeam after rubble forms.
            if tid in (113, 114, 115, 116):
                seam_base = get_seam_base_damage(unit.unit_type)
                if seam_base is not None and seam_base > 0:
                    targets.append((tr, tc))

    return targets


def _can_attack_submerged_or_hidden(attacker: Unit, enemy: Unit) -> bool:
    """Whether ``attacker`` may target ``enemy`` while ``enemy.is_submerged``.

    - Hidden Stealth: only Fighter or Stealth (https://awbw.fandom.com/wiki/Stealth).
    - Submerged Sub: Cruiser or Submarine (standard AWBW / Units chart; subs fight subs).
    """
    if enemy.unit_type == UnitType.STEALTH:
        return attacker.unit_type in (UnitType.FIGHTER, UnitType.STEALTH)
    if UNIT_STATS[enemy.unit_type].is_submarine:
        return attacker.unit_type in (UnitType.CRUISER, UnitType.SUBMARINE)
    return False


# ---------------------------------------------------------------------------
# Producible units per building type
# ---------------------------------------------------------------------------

_GROUND_UNITS: list[UnitType] = [
    UnitType.INFANTRY, UnitType.MECH, UnitType.RECON,
    UnitType.TANK, UnitType.MED_TANK, UnitType.NEO_TANK, UnitType.MEGA_TANK,
    UnitType.APC, UnitType.ARTILLERY, UnitType.ROCKET,
    UnitType.ANTI_AIR, UnitType.MISSILES,
    # Piperunner: AWBW allows production from any owned Base unless the map
    # bans it via ``unit_bans`` ("Piperunner"). Movement is restricted to pipe
    # tiles by ``MOVE_PIPELINE``; build legality is independent.
    UnitType.PIPERUNNER,
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
    occupancy = _build_occupancy(state)
    player = state.active_player
    if state.action_stage == ActionStage.SELECT:
        actions = _get_select_actions(state, player, occupancy=occupancy)
    elif state.action_stage == ActionStage.MOVE:
        actions = _get_move_actions(state, player, occupancy=occupancy)
    elif state.action_stage == ActionStage.ACTION:
        actions = _get_action_actions(state, player, occupancy=occupancy)
    else:
        actions = []
    # Phase 11J-RL-DELETE-GUARD-SHIP: defense-in-depth — fail loud if any
    # action type slipped past the per-builder filters into the RL action space.
    for _a in actions:
        if _a.action_type not in _RL_LEGAL_ACTION_TYPES:
            raise AssertionError(
                f"get_legal_actions emitted non-RL-legal action {_a.action_type.name}; "
                f"_RL_LEGAL_ACTION_TYPES is the canonical allowlist (see engine/action.py)."
            )
    return actions


def _get_select_actions(
    state: GameState,
    player: int,
    *,
    occupancy: dict[tuple[int, int], Unit] | None = None,
) -> list[Action]:
    if occupancy is None:
        occupancy = _build_occupancy(state)
    actions: list[Action] = []

    # COP/SCOP: emit only when meter/threshold and exclusivity rules allow
    # (mirrors ``COState.can_activate_cop`` / ``can_activate_scop``).
    co = state.co_states[player]
    if co.can_activate_cop():
        actions.append(Action(ActionType.ACTIVATE_COP))
    if co.can_activate_scop():
        actions.append(Action(ActionType.ACTIVATE_SCOP))

    has_unmoved = False
    for unit in state.units[player]:
        # Phase 11J-VONBOLT-SCOP-SHIP — Von Bolt "Ex Machina" stun gate.
        # AWBW canon (https://awbw.fandom.com/wiki/Von_Bolt + the AWBW CO
        # Chart https://awbw.amarriner.com/co.php Von Bolt row): Ex Machina
        # *"prevents all affected enemy units from acting next turn."*
        # ``Unit.is_stunned`` is set in ``GameState._apply_power_effects``
        # (co_id 30 SCOP branch) and cleared in ``GameState._end_turn`` on
        # the units of the player whose turn just ended — the stunned army
        # serves the stun across exactly one of its own turns. While the
        # flag is set we suppress SELECT_UNIT for that unit so the legal
        # mask never offers it to RL / agents / tests, AND the unit does
        # NOT count as "unmoved" — stunned units must not block END_TURN
        # (the stun would otherwise wedge the player on a turn where the
        # only legal option is to do nothing). The STEP-GATE in
        # ``GameState.step`` then rejects any direct attempt to act on a
        # stunned unit because that action will not appear in the mask.
        if unit.is_stunned:
            continue
        if not unit.moved:
            actions.append(Action(ActionType.SELECT_UNIT, unit_pos=unit.pos))
            stats = UNIT_STATS[unit.unit_type]
            carve_out = stats.carry_capacity > 0 and len(unit.loaded_units) > 0
            if not carve_out:
                has_unmoved = True

    # END_TURN: only when no unmoved unit blocks (loaded-transport carve-out
    # above). Crafted ``step(END_TURN)`` while unmoved units exist is rejected
    # by Phase 3 STEP-GATE in ``GameState.step``, not here.
    #
    # Loaded transports (carry_capacity > 0) with cargo aboard do not block
    # END_TURN: loaded transports are tactically optional — see
    # _get_select_actions carve-out. SELECT_UNIT is still emitted so the agent
    # may act on the transport if desired.
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
                    if occupancy.get((prop.row, prop.col)) is None:
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


def _get_move_actions(
    state: GameState,
    player: int,
    *,
    occupancy: dict[tuple[int, int], Unit] | None = None,
) -> list[Action]:
    if occupancy is None:
        occupancy = _build_occupancy(state)
    unit = state.selected_unit
    if unit is None:
        return []
    reachable = set(compute_reachable_costs(state, unit, occupancy=occupancy).keys())
    # --- AWBW_CAPTURE_MOVE_GATE (RL / get_legal_actions only; default OFF) ---
    # When the env var is "1"/"true"/"yes"/"on", capturers with at least one
    # reachable capturable tile (enemy/neutral property; not comm tower / lab;
    # same notion as rl/env._has_capturable_property per-tile) may only MOVE
    # onto those property tiles — closing the "sidestep to grass then WAIT"
    # loophole. If the filtered set would be empty, we keep the full reachable
    # set (defensive). Unset env → identical behaviour to pre-gate engine.
    gate_raw = _os.environ.get(_CAPTURE_MOVE_GATE_ENV, "").strip().lower()
    if (
        gate_raw in ("1", "true", "yes", "on")
        and UNIT_STATS[unit.unit_type].can_capture
    ):
        capturable: set[tuple[int, int]] = set()
        for pos in reachable:
            tid = state.map_data.terrain[pos[0]][pos[1]]
            if not get_terrain(tid).is_property:
                continue
            prop = state.get_property_at(*pos)
            if prop is None or prop.is_comm_tower or prop.is_lab:
                continue
            if prop.owner is not None and prop.owner == player:
                continue
            capturable.add(pos)
        if capturable:
            reachable = capturable
    return [
        Action(ActionType.SELECT_UNIT, unit_pos=unit.pos, move_pos=pos)
        for pos in reachable
    ]


def _get_action_actions(
    state: GameState,
    player: int,
    *,
    occupancy: dict[tuple[int, int], Unit] | None = None,
) -> list[Action]:
    if occupancy is None:
        occupancy = _build_occupancy(state)
    unit     = state.selected_unit
    move_pos = state.selected_move_pos
    if unit is None or move_pos is None:
        return []

    # If the destination is a friendly transport (not the unit itself), the
    # *only* legal terminator is LOAD. Allowing WAIT here would co-occupy the
    # tile, leaving two drawable units on one square in snapshots.
    occupant = occupancy.get(move_pos)
    # Phase 11J-FUNDS-EXTERMINATION-JOIN-SNAP-FIX: when the oracle replay
    # path-tail snap (tools/oracle_zip_replay.py::
    # _oracle_path_tail_occupant_allows_forced_snap) co-places the mover
    # onto a JOIN partner before terminator selection, ``get_unit_at`` may
    # return either unit. Resolve the partner explicitly by scanning for a
    # *different* friendly unit on the same tile so JOIN remains in the
    # legal-action mask. Without this, ``_get_action_actions`` falls through
    # to ATTACK / CAPTURE / WAIT and ``GameState._apply_join`` is never
    # called — the AWBW excess-HP fund refund (canon: AWBW Wiki *War
    # Funds*, ``(unit_cost / 10) * (HPA + HPB - 10)``) is silently lost
    # (closes GL 1628849; +400 g recovered → Build no-op cleared).
    if occupant is unit:
        for lst in state.units.values():
            for u in lst:
                if (
                    u is not unit
                    and u.is_alive
                    and u.pos == move_pos
                    and u.player == player
                ):
                    occupant = u
                    break
            if occupant is not unit:
                break
    boarding = (
        occupant is not None
        and occupant is not unit
        and occupant.player == player
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

    stats = UNIT_STATS[unit.unit_type]
    actions: list[Action] = [Action(ActionType.WAIT, unit_pos=unit.pos, move_pos=move_pos)]
    if stats.can_dive:
        actions.append(
            Action(ActionType.DIVE_HIDE, unit_pos=unit.pos, move_pos=move_pos)
        )

    # --- Attack ---
    for tpos in get_attack_targets(state, unit, move_pos, occupancy=occupancy):
        actions.append(Action(
            ActionType.ATTACK,
            unit_pos=unit.pos,
            move_pos=move_pos,
            target_pos=tpos,
        ))

    # --- Capture ---
    dest_tid  = state.map_data.terrain[move_pos[0]][move_pos[1]]
    dest_info = get_terrain(dest_tid)

    # CAPTURE: only foot units (``UNIT_STATS.can_capture``) on an opponent or
    # neutral income property (excludes owned tiles).
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
                drop_occupant = occupancy.get((tr, tc))
                if drop_occupant is not None and drop_occupant.pos != unit.pos:
                    continue
                actions.append(Action(
                    ActionType.UNLOAD,
                    unit_pos=unit.pos,
                    move_pos=move_pos,
                    target_pos=(tr, tc),
                    unit_type=cargo.unit_type,
                ))

    # --- BUILD removed from Stage-2 ACTION ---
    # AWBW factories produce units directly from the SELECT stage (see
    # ``_get_select_actions``); a Stage-2 BUILD piggy-backed on a moved unit
    # would let the unit issue a build *and* remain unmoved (``_apply_build``
    # never touches the acting unit), giving a free extra action per turn that
    # AWBW does not allow. Engine ⊂ AWBW: removed.

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
            ally = occupancy.get((tr, tc))
            if ally is None or ally.player != unit.player:
                continue
            if int(ally.unit_id) == int(unit.unit_id):
                continue
            if _black_boat_repair_eligible(state, ally):
                actions.append(Action(
                    ActionType.REPAIR,
                    unit_pos=unit.pos,
                    move_pos=move_pos,
                    target_pos=(tr, tc),
                ))

    # --- Prune WAIT when CAPTURE is available ---
    # If CAPTURE is a legal ACTION terminator, WAIT is not. (ATTACK-only does not
    # remove WAIT here.) Same for fresh vs mid-capture. Missile silos
    # (is_property False) and owned buildings are untouched; if CAPTURE is
    # not offered we leave WAIT so the unit is never deadlocked.
    if stats.can_capture and dest_info.is_property:
        prop_here = state.get_property_at(*move_pos)
        if prop_here is not None and prop_here.owner != player:
            has_capture = any(a.action_type == ActionType.CAPTURE for a in actions)
            if has_capture:
                pruned = [
                    a for a in actions
                    if a.action_type not in (ActionType.WAIT, ActionType.DIVE_HIDE)
                ]
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
        reachable = compute_reachable_costs(state, unit, occupancy=occupancy)
        if any(
            pos != move_pos and _apc_tile_benefits_allies(state, unit, pos)
            for pos in reachable
        ):
            pruned = [
                a for a in actions
                if a.action_type not in (ActionType.WAIT, ActionType.DIVE_HIDE)
            ]
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
    """Return True if ``ally`` is a valid Black Boat REPAIR target.

    AWBW emits the ``Repair`` envelope for any orthogonally adjacent ally
    even when the heal+resupply is a no-op (target at full HP / fuel /
    ammo — common for fresh INFANTRY ferried by a Black Boat). The legal
    action mask must therefore mirror that permissive rule; ``_apply_repair``
    already handles the no-op cleanly (skip heal when affordable/below-max
    fails, skip resupply when already full).
    """
    return ally is not None


def _build_cost(ut: UnitType, state: GameState, player: int, pos: tuple[int, int]) -> int:
    """Adjusted build cost after CO modifiers.

    Source: AWBW CO Chart https://awbw.amarriner.com/co.php
      * Kanbei  (3)  — units cost +20% more  → ×1.20
      * Colin   (15) — units cost  20% less  → ×0.80
      * Hachi   (17) — units cost  10% less  → ×0.90 (D2D, **all build sites**,
        not just bases). Phase 10T section 3 flagged the previous "50% on
        ``terrain.is_base`` only" heuristic as a HIGH-priority canon gap; the
        chart line is "Units cost 10% less". See
        ``docs/oracle_exception_audit/phase11a_kindle_hachi_canon.md``.
    """
    cost = UNIT_STATS[ut].cost
    co   = state.co_states[player]
    if co.co_id == 3:            # Kanbei: 120% cost
        cost = int(cost * 1.2)
    elif co.co_id == 15:         # Colin: 80% cost
        cost = int(cost * 0.8)
    elif co.co_id == 17:         # Hachi: 90% cost on every build (CO Chart "Units cost 10% less")
        cost = int(cost * 0.9)
    return cost


# ---------------------------------------------------------------------------
# Phase 11J-DELETE-GUARD-PIN — refuse to load if anyone adds a Delete-shaped
# action to the engine's RL action space. Cheap import-time assertion.
#
# AWBW players may scrap their own units (the "Delete unit" UI control) to
# free a production tile. The replay oracle (tools/oracle_zip_replay.py
# ::_oracle_kill_friendly_unit) reproduces that envelope so AWBW zip replays
# can be reconstructed faithfully, but the RL bot must NEVER be able to emit
# this action: a degenerate scrap-and-rebuild loop (scrap a low-value blocker
# -> spawn a stronger replacement -> repeat) would let the policy print
# arbitrary value out of the production system. This guard pins the contract
# at module-load time so a future refactor cannot quietly add a DELETE
# member without the test suite (tests/test_no_delete_action_legality.py)
# AND every Python process that imports engine.action breaking immediately.
# ---------------------------------------------------------------------------
_FORBIDDEN_RL_ACTION_NAMES = {
    "DELETE",
    "DELETE_UNIT",
    "SCRAP",
    "SCRAP_UNIT",
    "DESTROY_OWN_UNIT",
    "KILL_OWN_UNIT",
}
_existing = {m.name for m in ActionType}
_collision = _FORBIDDEN_RL_ACTION_NAMES & _existing
assert not _collision, (
    f"ActionType contains forbidden RL action(s): {_collision}. "
    f"See Phase 11J-DELETE-GUARD-PIN. Delete must remain oracle-only."
)
del _existing, _collision
