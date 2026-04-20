"""
AWBW unit definitions: all 27 unit types with full stats, and mutable Unit instance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from engine.terrain import (
    MOVE_INF, MOVE_MECH, MOVE_TREAD, MOVE_TIRE_A, MOVE_TIRE_B,
    MOVE_AIR, MOVE_SEA, MOVE_LANDER, MOVE_PIPELINE,
)


# ---------------------------------------------------------------------------
# Unit type enum
# ---------------------------------------------------------------------------
class UnitType(IntEnum):
    INFANTRY   =  0
    MECH       =  1
    RECON      =  2
    TANK       =  3
    MED_TANK   =  4
    NEO_TANK   =  5
    MEGA_TANK  =  6
    APC        =  7
    ARTILLERY  =  8
    ROCKET     =  9
    ANTI_AIR   = 10
    MISSILES   = 11
    FIGHTER    = 12
    BOMBER     = 13
    STEALTH    = 14
    B_COPTER   = 15
    T_COPTER   = 16
    BATTLESHIP = 17
    CARRIER    = 18
    SUBMARINE  = 19
    CRUISER    = 20
    LANDER     = 21
    GUNBOAT    = 22
    BLACK_BOAT = 23
    BLACK_BOMB = 24
    PIPERUNNER = 25
    OOZIUM     = 26


# ---------------------------------------------------------------------------
# UnitStats: immutable per-type data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UnitStats:
    unit_type: UnitType
    name: str
    move_type: str          # MOVE_* constant
    move_range: int
    max_fuel: int
    fuel_per_turn: int      # consumed at start of each of this unit's turns (see AWBW chart)
    # Expendable ammo capacity (AWBW ``units.php`` / amarriner Unit Chart):
    #   0 = no tracked magazine (MG-only infantry/recon, transports, bombs…);
    #       attacks still use ``min_range``/``max_range`` / damage table; no −1 ammo.
    #   >0 = finite shots; each direct/seam fire consumes 1 from ``Unit.ammo``.
    max_ammo: int
    vision: int
    cost: int               # in funds
    unit_class: str         # "infantry" | "mech" | "vehicle" | "copter" | "air" | "naval" | "pipe"
    can_capture: bool
    carry_capacity: int     # 0 = cannot carry
    min_range: int          # 1 for direct; 2+ for indirect
    max_range: int          # 1 for direct; 3–8 for indirect
    is_indirect: bool
    is_submarine: bool
    can_dive: bool          # stealth/sub: can submerge/hide


# ---------------------------------------------------------------------------
# Full stats table for all 27 units
# ---------------------------------------------------------------------------
# Fuel / ammo: chart numbers from https://awbw.fandom.com/wiki/Units .
# Movement fuel is terrain-based in ``terrain.py`` / ``weather.py``.
# Start-of-turn idle drain for air/sea is ``idle_start_of_day_fuel_drain``
# (Sub dive + Stealth hide, Eagle air discount) per the same page § Fuel.
UNIT_STATS: dict[UnitType, UnitStats] = {
    UnitType.INFANTRY: UnitStats(
        unit_type=UnitType.INFANTRY, name="Infantry",
        move_type=MOVE_INF, move_range=3,
        max_fuel=99, fuel_per_turn=0,
        max_ammo=0, vision=2, cost=1000,
        unit_class="infantry", can_capture=True,
        carry_capacity=0, min_range=1, max_range=1,
        is_indirect=False, is_submarine=False, can_dive=False,
    ),
    UnitType.MECH: UnitStats(
        unit_type=UnitType.MECH, name="Mech",
        move_type=MOVE_MECH, move_range=2,
        max_fuel=70, fuel_per_turn=0,
        max_ammo=3, vision=2, cost=3000,
        unit_class="mech", can_capture=True,
        carry_capacity=0, min_range=1, max_range=1,
        is_indirect=False, is_submarine=False, can_dive=False,
    ),
    UnitType.RECON: UnitStats(
        unit_type=UnitType.RECON, name="Recon",
        move_type=MOVE_TIRE_A, move_range=8,
        max_fuel=80, fuel_per_turn=0,
        max_ammo=0, vision=5, cost=4000,
        unit_class="vehicle", can_capture=False,
        carry_capacity=0, min_range=1, max_range=1,
        is_indirect=False, is_submarine=False, can_dive=False,
    ),
    UnitType.TANK: UnitStats(
        unit_type=UnitType.TANK, name="Tank",
        move_type=MOVE_TREAD, move_range=6,
        max_fuel=70, fuel_per_turn=0,
        max_ammo=9, vision=3, cost=7000,
        unit_class="vehicle", can_capture=False,
        carry_capacity=0, min_range=1, max_range=1,
        is_indirect=False, is_submarine=False, can_dive=False,
    ),
    UnitType.MED_TANK: UnitStats(
        unit_type=UnitType.MED_TANK, name="Medium Tank",
        move_type=MOVE_TREAD, move_range=5,
        max_fuel=50, fuel_per_turn=0,
        max_ammo=8, vision=1, cost=16000,
        unit_class="vehicle", can_capture=False,
        carry_capacity=0, min_range=1, max_range=1,
        is_indirect=False, is_submarine=False, can_dive=False,
    ),
    UnitType.NEO_TANK: UnitStats(
        unit_type=UnitType.NEO_TANK, name="Neotank",
        move_type=MOVE_TREAD, move_range=6,
        max_fuel=99, fuel_per_turn=0,
        max_ammo=9, vision=1, cost=22000,
        unit_class="vehicle", can_capture=False,
        carry_capacity=0, min_range=1, max_range=1,
        is_indirect=False, is_submarine=False, can_dive=False,
    ),
    UnitType.MEGA_TANK: UnitStats(
        unit_type=UnitType.MEGA_TANK, name="Megatank",
        move_type=MOVE_TREAD, move_range=4,
        max_fuel=50, fuel_per_turn=0,
        max_ammo=3, vision=1, cost=28000,
        unit_class="vehicle", can_capture=False,
        carry_capacity=0, min_range=1, max_range=1,
        is_indirect=False, is_submarine=False, can_dive=False,
    ),
    UnitType.APC: UnitStats(
        unit_type=UnitType.APC, name="APC",
        move_type=MOVE_TREAD, move_range=6,
        max_fuel=70, fuel_per_turn=0,
        max_ammo=0, vision=1, cost=5000,
        unit_class="vehicle", can_capture=False,
        carry_capacity=1, min_range=1, max_range=1,
        is_indirect=False, is_submarine=False, can_dive=False,
    ),
    UnitType.ARTILLERY: UnitStats(
        unit_type=UnitType.ARTILLERY, name="Artillery",
        move_type=MOVE_TREAD, move_range=5,
        max_fuel=50, fuel_per_turn=0,
        max_ammo=9, vision=1, cost=6000,
        unit_class="vehicle", can_capture=False,
        carry_capacity=0, min_range=2, max_range=3,
        is_indirect=True, is_submarine=False, can_dive=False,
    ),
    UnitType.ROCKET: UnitStats(
        unit_type=UnitType.ROCKET, name="Rocket",
        move_type=MOVE_TREAD, move_range=5,
        max_fuel=50, fuel_per_turn=0,
        max_ammo=6, vision=1, cost=15000,
        unit_class="vehicle", can_capture=False,
        carry_capacity=0, min_range=3, max_range=5,
        is_indirect=True, is_submarine=False, can_dive=False,
    ),
    UnitType.ANTI_AIR: UnitStats(
        unit_type=UnitType.ANTI_AIR, name="Anti-Air",
        move_type=MOVE_TREAD, move_range=6,
        max_fuel=60, fuel_per_turn=0,
        max_ammo=9, vision=2, cost=8000,
        unit_class="vehicle", can_capture=False,
        carry_capacity=0, min_range=1, max_range=1,
        is_indirect=False, is_submarine=False, can_dive=False,
    ),
    UnitType.MISSILES: UnitStats(
        unit_type=UnitType.MISSILES, name="Missiles",
        move_type=MOVE_TREAD, move_range=4,
        max_fuel=50, fuel_per_turn=0,
        max_ammo=6, vision=2, cost=12000,
        unit_class="vehicle", can_capture=False,
        carry_capacity=0, min_range=3, max_range=5,
        is_indirect=True, is_submarine=False, can_dive=False,
    ),
    UnitType.FIGHTER: UnitStats(
        unit_type=UnitType.FIGHTER, name="Fighter",
        move_type=MOVE_AIR, move_range=9,
        max_fuel=99, fuel_per_turn=5,
        max_ammo=9, vision=2, cost=20000,
        unit_class="air", can_capture=False,
        carry_capacity=0, min_range=1, max_range=1,
        is_indirect=False, is_submarine=False, can_dive=False,
    ),
    UnitType.BOMBER: UnitStats(
        unit_type=UnitType.BOMBER, name="Bomber",
        move_type=MOVE_AIR, move_range=7,
        max_fuel=99, fuel_per_turn=5,
        max_ammo=9, vision=2, cost=22000,
        unit_class="air", can_capture=False,
        carry_capacity=0, min_range=1, max_range=1,
        is_indirect=False, is_submarine=False, can_dive=False,
    ),
    UnitType.STEALTH: UnitStats(
        unit_type=UnitType.STEALTH, name="Stealth",
        move_type=MOVE_AIR, move_range=6,
        max_fuel=60, fuel_per_turn=5,
        max_ammo=6, vision=4, cost=24000,
        unit_class="air", can_capture=False,
        carry_capacity=0, min_range=1, max_range=1,
        is_indirect=False, is_submarine=False, can_dive=True,
    ),
    UnitType.B_COPTER: UnitStats(
        unit_type=UnitType.B_COPTER, name="B-Copter",
        move_type=MOVE_AIR, move_range=6,
        max_fuel=99, fuel_per_turn=2,
        max_ammo=6, vision=3, cost=9000,
        unit_class="copter", can_capture=False,
        carry_capacity=0, min_range=1, max_range=1,
        is_indirect=False, is_submarine=False, can_dive=False,
    ),
    UnitType.T_COPTER: UnitStats(
        unit_type=UnitType.T_COPTER, name="T-Copter",
        move_type=MOVE_AIR, move_range=6,
        max_fuel=99, fuel_per_turn=2,
        max_ammo=0, vision=2, cost=5000,
        unit_class="copter", can_capture=False,
        carry_capacity=1, min_range=1, max_range=1,
        is_indirect=False, is_submarine=False, can_dive=False,
    ),
    UnitType.BATTLESHIP: UnitStats(
        unit_type=UnitType.BATTLESHIP, name="Battleship",
        move_type=MOVE_SEA, move_range=5,
        max_fuel=99, fuel_per_turn=1,
        max_ammo=9, vision=2, cost=28000,
        unit_class="naval", can_capture=False,
        carry_capacity=0, min_range=2, max_range=6,
        is_indirect=True, is_submarine=False, can_dive=False,
    ),
    UnitType.CARRIER: UnitStats(
        unit_type=UnitType.CARRIER, name="Carrier",
        move_type=MOVE_SEA, move_range=5,
        max_fuel=99, fuel_per_turn=1,
        max_ammo=9, vision=4, cost=30000,
        unit_class="naval", can_capture=False,
        carry_capacity=2, min_range=3, max_range=8,
        is_indirect=True, is_submarine=False, can_dive=False,
    ),
    UnitType.SUBMARINE: UnitStats(
        unit_type=UnitType.SUBMARINE, name="Submarine",
        move_type=MOVE_SEA, move_range=5,
        max_fuel=60, fuel_per_turn=1,
        max_ammo=6, vision=5, cost=20000,
        unit_class="naval", can_capture=False,
        carry_capacity=0, min_range=1, max_range=1,
        is_indirect=False, is_submarine=True, can_dive=True,
    ),
    UnitType.CRUISER: UnitStats(
        unit_type=UnitType.CRUISER, name="Cruiser",
        move_type=MOVE_SEA, move_range=6,
        max_fuel=99, fuel_per_turn=1,
        max_ammo=9, vision=3, cost=18000,
        unit_class="naval", can_capture=False,
        carry_capacity=2, min_range=1, max_range=1,
        is_indirect=False, is_submarine=False, can_dive=False,
    ),
    UnitType.LANDER: UnitStats(
        unit_type=UnitType.LANDER, name="Lander",
        move_type=MOVE_LANDER, move_range=6,
        max_fuel=99, fuel_per_turn=1,
        max_ammo=0, vision=1, cost=12000,
        unit_class="naval", can_capture=False,
        carry_capacity=2, min_range=1, max_range=1,
        is_indirect=False, is_submarine=False, can_dive=False,
    ),
    UnitType.GUNBOAT: UnitStats(
        unit_type=UnitType.GUNBOAT, name="Gunboat",
        move_type=MOVE_LANDER, move_range=7,
        max_fuel=99, fuel_per_turn=1,
        max_ammo=1, vision=2, cost=6000,
        unit_class="naval", can_capture=False,
        carry_capacity=1, min_range=1, max_range=1,
        is_indirect=False, is_submarine=False, can_dive=False,
    ),
    UnitType.BLACK_BOAT: UnitStats(
        unit_type=UnitType.BLACK_BOAT, name="Black Boat",
        move_type=MOVE_LANDER, move_range=7,
        max_fuel=60, fuel_per_turn=1,
        max_ammo=0, vision=1, cost=7500,
        unit_class="naval", can_capture=False,
        carry_capacity=2, min_range=1, max_range=1,
        is_indirect=False, is_submarine=False, can_dive=False,
    ),
    UnitType.BLACK_BOMB: UnitStats(
        unit_type=UnitType.BLACK_BOMB, name="Black Bomb",
        move_type=MOVE_AIR, move_range=9,
        max_fuel=45, fuel_per_turn=5,
        max_ammo=0, vision=1, cost=25000,
        unit_class="air", can_capture=False,
        carry_capacity=0, min_range=1, max_range=1,
        is_indirect=False, is_submarine=False, can_dive=False,
    ),
    UnitType.PIPERUNNER: UnitStats(
        unit_type=UnitType.PIPERUNNER, name="Piperunner",
        move_type=MOVE_PIPELINE, move_range=9,
        max_fuel=99, fuel_per_turn=0,
        max_ammo=9, vision=4, cost=20000,
        unit_class="pipe", can_capture=False,
        carry_capacity=0, min_range=2, max_range=5,
        is_indirect=True, is_submarine=False, can_dive=False,
    ),
    UnitType.OOZIUM: UnitStats(
        unit_type=UnitType.OOZIUM, name="Oozium",
        move_type=MOVE_INF, move_range=1,
        max_fuel=99, fuel_per_turn=0,
        max_ammo=0, vision=1, cost=0,
        unit_class="vehicle", can_capture=False,
        carry_capacity=0, min_range=1, max_range=1,
        is_indirect=False, is_submarine=False, can_dive=False,
    ),
}


def idle_start_of_day_fuel_drain(unit: "Unit", co_id: int) -> int:
    """Fuel consumed at the start of this unit owner's turn (AWBW idle / \"day\" drain).

    Authoritative rules: https://awbw.fandom.com/wiki/Units#Fuel — sea 1/day
    with +4 when a Sub is submerged (5 total); copters 2/day; planes 5/day
    with +3 when a Stealth is hidden (8 total); Eagle (CO id 10) applies −2/day
    to his *air* units (wiki: copters 0, planes 3, hidden Stealth 6). Naval
    idle drain is not reduced by Eagle.
    """
    st = UNIT_STATS[unit.unit_type]
    cls = st.unit_class
    eagle_air = 2 if co_id == 10 else 0

    if cls in ("infantry", "mech", "vehicle", "pipe"):
        return 0

    if cls == "naval":
        n = 1
        if st.is_submarine and unit.is_submerged:
            n += 4
        return n

    if cls == "copter":
        return max(0, 2 - eagle_air)

    if cls == "air":
        plane_base = max(0, 5 - eagle_air)
        if unit.unit_type == UnitType.STEALTH and unit.is_submerged:
            return plane_base + 3
        return plane_base

    return 0


# ---------------------------------------------------------------------------
# Mutable unit instance
# ---------------------------------------------------------------------------
@dataclass
class Unit:
    unit_type: UnitType
    player: int              # 0 or 1
    hp: int                  # 0–100 internal scale
    ammo: int
    fuel: int
    pos: tuple[int, int]     # (row, col)
    moved: bool              # has acted this turn
    loaded_units: list       # list[Unit] for transports
    is_submerged: bool       # submarines / stealth hidden
    capture_progress: int    # capture points remaining on current property (20 = not capturing)
    # Stable identity that persists across turns. Assigned exactly once at
    # creation (predeploy, build, or CO power spawn) and never reused after
    # death. AWBW replay viewers index `DrawableUnit` by this id, so reusing
    # it causes the wrong sprite/color to be updated (see export_awbw_replay).
    unit_id: int = 0

    @property
    def display_hp(self) -> int:
        """HP shown in-game: 1–10 (ceiling of internal 1–100)."""
        return (self.hp + 9) // 10

    @property
    def is_alive(self) -> bool:
        return self.hp > 0

    def __repr__(self) -> str:
        return (
            f"Unit({self.unit_type.name}, P{self.player}, "
            f"hp={self.hp}, pos={self.pos}, moved={self.moved})"
        )
