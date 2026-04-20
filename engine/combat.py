"""
AWBW combat: damage formula, luck, and counterattack.

Damage formula (AWBW canonical, cross-checked against community damage
calculators):

    raw = (B × AV/100 + L - LB) × (HPA_bars / 10) × (200 - (DV + DTR × HPD_bars)) / 100

  B         = base damage percent (from damage_table.json)
  AV        = attacker value  (100 + CO ATK modifiers; Lash terrain bonus)
  L         = luck roll (positive component, per CO)
  LB        = bad luck (negative component; 0 for most COs)
  HPA_bars  = attacker display HP (1–10 bars; ``Unit.display_hp``)
  DV        = defender value  (100 + CO DEF modifiers)
  DTR       = defender terrain defense stars (0–4)
  HPD_bars  = defender display HP (1–10 bars; ``Unit.display_hp``)

Both HPA and HPD are expressed in **display bars** (1–10) to keep the
``DTR × HPD`` term on the same scale as ``DV`` (percent). Using raw internal
HP (1–100) for HPD causes ``200 - (DV + DTR × HPD)`` to go negative for
defenders on any terrain-star tile, which incorrectly clamps damage to 0
(historically manifested as "Sami Infantry does 0% damage to a Mech on a
city"). See ``calculate_damage`` for the in-line justification and the
anchor combat regression test in ``test_combat_anchor.py``.

Rounding: ceil(raw / 0.05) * 0.05, then floor to int — matches AWBW's
"nearest 0.05 up, then floor" convention for integer HP damage.
"""
from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Optional

from engine.unit import Unit, UnitType, UNIT_STATS
from engine.terrain import TerrainInfo, get_terrain
from engine.co import COState

DAMAGE_TABLE_PATH = Path(__file__).parent.parent / "data" / "damage_table.json"

_damage_table: Optional[list[list]] = None


# ---------------------------------------------------------------------------
# Damage table loading
# ---------------------------------------------------------------------------

def load_damage_table() -> list[list]:
    """
    Load the 27×27 base-damage matrix.

    Expected JSON shape:
      { "table": [[row0...], [row1...], ...] }

    Each entry is an integer percentage (e.g. 55 = 55%) or null if the
    attacker cannot hit the defender.

    Raises FileNotFoundError if data/damage_table.json is absent.
    """
    global _damage_table
    if _damage_table is not None:
        return _damage_table
    if not DAMAGE_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Damage table not found at {DAMAGE_TABLE_PATH}. "
            "Run the data-generation script to create data/damage_table.json."
        )
    with open(DAMAGE_TABLE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    _damage_table = data["table"]
    return _damage_table


def get_base_damage(attacker_type: UnitType, defender_type: UnitType) -> Optional[int]:
    """
    Return the base damage percentage for attacker → defender.
    Returns None if the attacker cannot target that unit type.
    """
    table = load_damage_table()
    val = table[int(attacker_type)][int(defender_type)]
    return val  # None or int


# ---------------------------------------------------------------------------
# Seam damage
# ---------------------------------------------------------------------------
# AWBW pipe-seam base damage per attacker, no luck, defender behaves like a
# 100-HP Neotank at 0★ terrain (see https://awbw.fandom.com/wiki/Pipes_and_Pipeseams).
# Values not listed (0 / None) mean the attacker cannot damage a seam.
_SEAM_BASE_DAMAGE: dict[UnitType, int] = {
    UnitType.INFANTRY:    5,
    UnitType.MECH:       55,   # machine gun (wiki lists 55 / 65 MG/Bazooka; MG is the seam weapon)
    UnitType.RECON:       8,
    UnitType.TANK:       55,
    UnitType.MED_TANK:  105,
    UnitType.NEO_TANK:  125,
    UnitType.MEGA_TANK: 135,
    UnitType.ARTILLERY:  70,
    UnitType.ROCKET:     80,
    UnitType.ANTI_AIR:   10,
    UnitType.BOMBER:    110,
    UnitType.STEALTH:    75,
    UnitType.B_COPTER:   55,
    UnitType.BATTLESHIP: 80,
    UnitType.PIPERUNNER: 80,
}


def get_seam_base_damage(attacker_type: UnitType) -> Optional[int]:
    """Base damage percent for ``attacker_type`` vs an intact pipe seam.

    Returns None when the unit cannot damage seams (unarmed transports,
    air-only missile platforms, other naval non-BB units, etc.). Used by
    ``get_attack_targets`` to decide whether a seam tile is selectable.
    """
    return _SEAM_BASE_DAMAGE.get(attacker_type)


def calculate_seam_damage(
    attacker: Unit,
    attacker_terrain: TerrainInfo,
    attacker_co: COState,
) -> Optional[int]:
    """Compute damage the attacker deals to an intact pipe seam.

    Follows the AWBW formula with luck disabled (per wiki) and defender
    stats frozen at Neotank-on-0★ (DV=100, DTR=0, HPD=10). CO ATK/D2D/
    comm tower bonuses still apply via ``attacker_co.total_atk``. Lash's
    terrain-ATK bonus also applies when her power is active — matching
    the live AWBW formula for vs-unit combat.
    """
    base = get_seam_base_damage(attacker.unit_type)
    if base is None or base <= 0:
        return None

    unit_class = UNIT_STATS[attacker.unit_type].unit_class
    av = attacker_co.total_atk(unit_class)
    if attacker_co.co_id == 16:
        if attacker_co.scop_active or attacker_co.cop_active:
            av += attacker_terrain.defense * 10

    hpa_bars = attacker.display_hp
    # Defender side: Neotank at 0 defense stars, full HP, no luck applied.
    # (200 - (100 + 0 * 10)) / 100 == 1.0, so the multiplier reduces to 1.
    raw = (base * av / 100) * (hpa_bars / 10) * 1.0

    rounded = math.ceil(raw / 0.05) * 0.05
    return max(0, math.floor(rounded))


# ---------------------------------------------------------------------------
# Luck calculation per CO
# ---------------------------------------------------------------------------

def _get_luck(co_id: int, roll: int, cop: bool, scop: bool) -> int:
    """
    Return the effective luck value for this CO.
    Positive = good luck (adds to damage), negative = bad luck (subtracts).

    Roll is always in [0, 9] from the caller; COs scale/shift it.

    CO IDs used here must match co_data.json:
      24 = Nell, 28 = Rachel, 25 = Flak, 26 = Jugger
    """
    # Nell: COP ×3 luck (max 29), SCOP ×6 (max 59)
    if co_id == 24:
        if scop:
            return min(roll * 6, 59)
        if cop:
            return min(roll * 3, 29)
        return min(roll, 9)

    # Rachel: COP ×2 luck (max 19)
    if co_id == 28:
        if cop or scop:
            return min(roll * 2, 19)
        return min(roll, 9)

    # Flak: negative to positive luck, scaled by power state
    if co_id == 25:
        if scop:
            return roll * 4 - 10
        if cop:
            return roll * 2 - 5
        return roll - 2

    # Jugger: similar extreme variance to Flak but slightly wider
    if co_id == 26:
        if scop:
            return roll * 5 - 10
        if cop:
            return roll * 3 - 5
        return roll * 2 - 3

    return min(roll, 9)


# ---------------------------------------------------------------------------
# Main damage calculation
# ---------------------------------------------------------------------------

def calculate_damage(
    attacker: Unit,
    defender: Unit,
    attacker_terrain: TerrainInfo,
    defender_terrain: TerrainInfo,
    attacker_co: COState,
    defender_co: COState,
    luck_roll: Optional[int] = None,
) -> Optional[int]:
    """
    Calculate damage dealt by attacker to defender.

    Returns damage in HP points (0–100 internal scale), or None if the
    attacker cannot attack the defender (missing table entry).

    luck_roll: explicit 0–9 value for deterministic tests; random if None.
    """
    base = get_base_damage(attacker.unit_type, defender.unit_type)
    if base is None:
        return None

    # Hidden Stealth: only Fighter or Stealth may attack (Fandom Stealth page).
    # Submerged Sub: Cruiser or Submarine (Fandom Units § Fuel + standard AWBW naval rules).
    if defender.is_submerged:
        if defender.unit_type == UnitType.STEALTH:
            if attacker.unit_type not in (UnitType.FIGHTER, UnitType.STEALTH):
                return None
        elif UNIT_STATS[defender.unit_type].is_submarine:
            if attacker.unit_type not in (UnitType.CRUISER, UnitType.SUBMARINE):
                return None

    unit_class     = UNIT_STATS[attacker.unit_type].unit_class
    def_unit_class = UNIT_STATS[defender.unit_type].unit_class

    # --- Attack Value ---
    av = attacker_co.total_atk(unit_class)

    # Lash (co_id=16): terrain stars add to ATK when her power is active
    if attacker_co.co_id == 16:
        if attacker_co.scop_active or attacker_co.cop_active:
            av += attacker_terrain.defense * 10

    # --- Luck ---
    if luck_roll is None:
        luck_roll = random.randint(0, 9)
    luck_val = _get_luck(attacker_co.co_id, luck_roll, attacker_co.cop_active, attacker_co.scop_active)
    l_val  = max(0, luck_val)
    lb_val = max(0, -luck_val)

    # AWBW damage uses display HP bars (1–10, ceilinged) on both sides. Using
    # raw internal HP (1–100) for HPD makes ``dtr × hpd`` 10× too large and
    # flips ``200 - (dv + dtr × hpd)`` negative on any terrain-star tile,
    # clamping damage to 0.
    hpa_bars = attacker.display_hp
    hpd_bars = defender.display_hp

    # --- Defense Value ---
    dv = defender_co.total_def(def_unit_class)

    # Terrain defense stars
    dtr = defender_terrain.defense
    # Submerged subs/stealth get no terrain bonus (they're at sea)
    if defender.is_submerged:
        dtr = 0

    # --- Formula ---
    raw = (base * av / 100 + l_val - lb_val) * (hpa_bars / 10) * (200 - (dv + dtr * hpd_bars)) / 100

    # Round up to nearest 0.05, then floor
    rounded = math.ceil(raw / 0.05) * 0.05
    result  = math.floor(rounded)

    return max(0, result)


# ---------------------------------------------------------------------------
# Damage range (observer belief layer)
# ---------------------------------------------------------------------------

def damage_range(
    attacker: Unit,
    defender: Unit,
    attacker_terrain: TerrainInfo,
    defender_terrain: TerrainInfo,
    attacker_co: COState,
    defender_co: COState,
) -> Optional[tuple[int, int]]:
    """Min/max damage (inclusive) the attacker *could* deal to the defender
    given the observer's view of CO state, terrain, and display-HP buckets.

    Used by ``engine.belief.BeliefState`` to shrink per-unit HP intervals
    after a visible combat event. Inputs are the same objects ``calculate_damage``
    consumes, so CO powers (Lash terrain ATK, Sonja SCOP side-effects, etc.)
    are honoured identically.

    Returns ``None`` if the attacker cannot hit the defender (missing damage
    table entry). Otherwise returns ``(min_dmg, max_dmg)`` on the internal
    0–100 scale with ``0 <= min_dmg <= max_dmg``.

    Implementation note: instead of reimplementing the formula, we sweep
    every legal ``luck_roll`` in ``[0, 9]`` through ``calculate_damage``
    (luck is the only stochastic input; everything else is deterministic
    from these args) and return the observed extremes. This automatically
    covers Nell's ×6 scaling, Flak/Jugger's negative-luck tails, and any
    future CO-specific luck curve without duplication.
    """
    if get_base_damage(attacker.unit_type, defender.unit_type) is None:
        return None

    vals: list[int] = []
    for roll in range(10):
        d = calculate_damage(
            attacker, defender,
            attacker_terrain, defender_terrain,
            attacker_co, defender_co,
            luck_roll=roll,
        )
        if d is None:
            return None
        vals.append(d)
    return min(vals), max(vals)


# ---------------------------------------------------------------------------
# Counterattack
# ---------------------------------------------------------------------------

def calculate_counterattack(
    attacker: Unit,
    defender: Unit,
    attacker_terrain: TerrainInfo,
    defender_terrain: TerrainInfo,
    attacker_co: COState,
    defender_co: COState,
    attack_damage: int,
    luck_roll: Optional[int] = None,
) -> Optional[int]:
    """
    Calculate the defender's counterattack against the attacker.

    Indirect units and unarmed units cannot counter.
    Sonja SCOP (co_id=18): defender counters with the **pre-attack** HP
    instead of the reduced post-attack HP.

    Contract with caller: by the time this function is invoked, the caller
    (``GameState._apply_attack``) has already applied ``attack_damage`` to
    ``defender.hp`` — the defender's HP here is **post-attack**. Previously
    this function re-subtracted ``attack_damage`` internally, which double-
    counted the forward strike and silently halved the counter damage for
    every non-Sonja engagement. ``counter_unit`` now uses ``defender`` in-
    place for the standard path; Sonja SCOP reverses the reduction.
    """
    def_stats = UNIT_STATS[defender.unit_type]

    if def_stats.is_indirect:
        return None
    if not defender.is_alive:
        return None

    # No counter if this defender cannot damage the attacker (MG-only units use
    # max_ammo == 0 per AWBW chart but still have machine-gun entries).
    if get_base_damage(defender.unit_type, attacker.unit_type) is None:
        return None

    # Cannot counter if out of expendable ammo (rockets / tank shells, etc.)
    if def_stats.max_ammo > 0 and defender.ammo == 0:
        return None

    if defender_co.co_id == 18 and defender_co.scop_active:
        # Sonja SCOP "counter break": restore the forward damage so the
        # counter rolls against pre-attack HP (capped at 100).
        counter_unit = copy.copy(defender)
        counter_unit.hp = min(100, defender.hp + attack_damage)
    else:
        counter_unit = defender

    return calculate_damage(
        counter_unit, attacker,
        defender_terrain, attacker_terrain,
        defender_co, attacker_co,
        luck_roll=luck_roll,
    )
