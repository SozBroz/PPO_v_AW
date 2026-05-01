"""
AWBW combat: damage formula, luck, and counterattack.

Damage formula (AWBW canonical, cross-checked against community damage
calculators):

    raw = (B × AV/100 + L - LB) × (HPA_bars / 10) × (200 - (DV + DTR × HPD_bars)) / 100

  B         = base damage percent (from damage_table.json)
  AV        = attacker value  (100 + CO ATK modifiers; Lash terrain bonus)
  L         = good luck (positive component, per CO)
  LB        = bad luck (subtracted in the attack term; 0 for most COs)

When a CO's configured luck range spans **both** negative and positive
(`luck_modifiers` low ``< 0`` and high ``> 0``, or Sonja-style symmetric
``low < 0`` with high ``== 0`` meaning ±|low|), AWBW rolls **two**
independent uniform digits 0–9: one maps into the bad-luck arm (LB) and one
into the good-luck arm (L). Extremes remain reachable (e.g. full LB with no L),
but the distribution is smoother than a single roll mapped across the net span.
COs with only-positive or only-negative (single-interval) luck still use one roll.
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
# Kindle (co_id=23) attack rider — Phase 11J-L1-BUILD-FUNDS-SHIP
# ---------------------------------------------------------------------------
# AWBW canon (Tier 1 — AWBW CO Chart https://awbw.amarriner.com/co.php Kindle row):
#   *"Units (even air units) gain +40% attack while on urban terrain. HQs,
#    bases, airports, ports, cities, labs, and comtowers count as urban
#    terrain. Urban Blight -- All enemy units lose -3 HP on urban terrain.
#    Urban bonus is increased to +80%. High Society -- Urban bonus is
#    increased to +130%, and attack for all units is increased by +3% for
#    each of your owned urban terrain."*
#
# Power tiers REPLACE each other on urban terrain (not stack):
#   D2D   → +40 AV
#   COP   → +80 AV   (Urban Blight — the +10 SCOPB is still added separately
#                     by COState.cop_atk_modifier as with every CO.)
#   SCOP  → +130 AV  (High Society)
# Plus, under SCOP only, +3 AV per owned urban property (HQs, bases,
# airports, ports, cities, labs, comm towers — all entries in
# ``GameState.properties``), which applies off-urban too. ``urban_props``
# is refreshed each turn by ``GameState._refresh_comm_towers``.
def _kindle_atk_rider(attacker_co: COState, attacker_terrain: TerrainInfo) -> int:
    if attacker_co.co_id != 23:
        return 0
    av = 0
    if attacker_terrain.is_property:
        if attacker_co.scop_active:
            av += 130
        elif attacker_co.cop_active:
            av += 80
        else:
            av += 40
    if attacker_co.scop_active:
        av += 3 * attacker_co.urban_props
    return av


# ---------------------------------------------------------------------------
# Colin (co_id=15) attack rider — Phase 11J-COLIN-IMPL-SHIP
# ---------------------------------------------------------------------------
# AWBW canon (Tier 1, both AWBW canonicals agree — see
# docs/oracle_exception_audit/phase11y_colin_scrape.md §0.1, §0.3, §0.4):
#   * D2D — *"Units cost −20 % less to build and lose −10 % attack."*
#   * COP "Gold Rush" — *"Funds are multiplied by 1.5x."* (NO attack rider;
#     funds payout handled in ``GameState._apply_power_effects``.)
#   * SCOP "Power of Money" — *"Unit attack percentage increases by
#     (3 * Funds / 1000)%."*
#   * Sources: https://awbw.amarriner.com/co.php (Colin row) and
#     https://awbw.fandom.com/wiki/Colin
#
# Stacking model (per scrape §0.4): D2D −10 %% PERSISTS during COP and SCOP and
# stacks with the universal +10 %% SCOPB rider that ``COState.cop_atk_modifier``
# already adds. Net AV deltas vs base 100 (this rider's contribution only,
# SCOPB applied separately by ``COState.cop_atk_modifier``):
#   D2D    →  −10 AV
#   COP    →  −10 AV   (Gold Rush has no attack effect; SCOPB still adds +10
#                       universally for net 100 AV during COP — matches
#                       scrape §0.4 "≈99 %%" wording, which is multiplicative
#                       prose for the additive engine.)
#   SCOP   →  −10 + int(3 * funds_snapshot / 1000) AV
#
# Funds source for SCOP: snapshotted into ``COState.colin_pom_funds_snapshot``
# at SCOP activation by ``GameState._apply_power_effects`` (so post-SCOP
# spending does not erode the bonus mid-turn). Float division → ``int(...)``
# floor matches AWBW PHP integer arithmetic for non-negative funds.
def _colin_atk_rider(attacker_co: COState) -> int:
    if attacker_co.co_id != 15:
        return 0
    av = -10  # D2D −10 %% attack, persists through COP and SCOP per scrape §0.4.
    if attacker_co.scop_active:
        av += int(3 * attacker_co.colin_pom_funds_snapshot / 1000)
    return av


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
    av = attacker_co.total_atk_for_unit(attacker.unit_type)
    if attacker_co.co_id == 16:
        if attacker_co.scop_active or attacker_co.cop_active:
            av += attacker_terrain.defense * 10
    av += _kindle_atk_rider(attacker_co, attacker_terrain)
    av += _colin_atk_rider(attacker_co)

    hpa_bars = attacker.display_hp
    # Defender side: Neotank at 0 defense stars, full HP, no luck applied.
    # (200 - (100 + 0 * 10)) / 100 == 1.0, so the multiplier reduces to 1.
    raw = (base * av / 100) * (hpa_bars / 10) * 1.0

    rounded = math.ceil(raw / 0.05) * 0.05
    return max(0, math.floor(rounded))


# ---------------------------------------------------------------------------
# Luck calculation per CO
# ---------------------------------------------------------------------------

def _bounded_luck_digit(roll: int) -> int:
    return max(0, min(int(roll), 9))


def _single_roll_net_luck(low: int, high_exclusive: int, roll: int) -> int:
    """Interpolate one 0–9 digit across inclusive endpoints ``low .. high_exclusive-1``.

    Used when luck is a **single** stochastic degree of freedom (no mixed good+bad arm).
    Bounds match ``co_data.json``: ``high_exclusive`` is exclusive (``\"0,10\"`` → 0..9).
    """
    br = _bounded_luck_digit(roll)
    span = high_exclusive - low
    if span <= 1:
        return low
    return low + ((span - 1) * br) // 9


def _is_dual_luck_bounds(bounds: tuple[int, int]) -> bool:
    """True when AWBW uses separate bad-luck and good-luck dice (mixed or symmetric Sonja)."""
    low, high_ex = bounds
    return low < 0 and (high_ex > 0 or high_ex == 0)


def luck_net_bounds_for_co(co: COState) -> tuple[int, int]:
    """Min/max net luck contribution ``L - LB`` (inclusive) for ``co``'s active power tier."""
    bounds = co.luck_bounds()
    if bounds is None:
        return 0, 9
    low, high_ex = bounds
    if _is_dual_luck_bounds(bounds):
        if high_ex == 0:
            cap = -low
            return -cap, cap
        return -(-low), high_ex - 1
    vals = [_single_roll_net_luck(low, high_ex, r) for r in range(10)]
    return min(vals), max(vals)


def _draw_luck_digit(luck_rng: Optional[random.Random]) -> int:
    if luck_rng is not None:
        return luck_rng.randint(0, 9)
    return random.randint(0, 9)


def _scale_dual_luck_arm(low: int, high_exclusive: int, roll: int) -> int:
    """Map one luck digit into one arm: inclusive ``low .. high_exclusive-1``."""
    return _single_roll_net_luck(low, high_exclusive, roll)


def _resolve_attack_luck_terms(
    attacker_co: COState,
    luck_roll: Optional[int],
    luck_roll_bad: Optional[int],
    luck_rng: Optional[random.Random],
) -> tuple[int, int]:
    """Return ``(L, LB)`` good/bad luck percentages for the attack term."""
    bounds = attacker_co.luck_bounds()
    if bounds is None:
        r = luck_roll if luck_roll is not None else _draw_luck_digit(luck_rng)
        return _bounded_luck_digit(r), 0
    low, high_ex = bounds
    if _is_dual_luck_bounds(bounds):
        rg = luck_roll if luck_roll is not None else _draw_luck_digit(luck_rng)
        rb = luck_roll_bad if luck_roll_bad is not None else _draw_luck_digit(luck_rng)
        if high_ex == 0:
            cap = -low
            l_val = _scale_dual_luck_arm(0, cap + 1, rg)
            lb_val = _scale_dual_luck_arm(0, cap + 1, rb)
        else:
            l_val = _scale_dual_luck_arm(0, high_ex, rg)
            lb_val = _scale_dual_luck_arm(0, (-low) + 1, rb)
        return l_val, lb_val
    r = luck_roll if luck_roll is not None else _draw_luck_digit(luck_rng)
    luck_val = _single_roll_net_luck(low, high_ex, r)
    return max(0, luck_val), max(0, -luck_val)


def _get_luck(co: COState, roll: int) -> int:
    """Legacy net luck from a **single** 0–9 digit (non–dual-luck COs only).

    Dual-luck COs (Sonja / Flak / Jugger day-to-day and powers) raise:
    use :func:`luck_net_bounds_for_co` or pass ``luck_roll`` and
    ``luck_roll_bad`` into :func:`calculate_damage`.
    """
    bounds = co.luck_bounds()
    if bounds is None:
        return min(roll, 9)
    low, high_ex = bounds
    if _is_dual_luck_bounds(bounds):
        raise ValueError(
            "dual-luck CO: use luck_net_bounds_for_co or calculate_damage(..., luck_roll=..., luck_roll_bad=...)"
        )
    return _single_roll_net_luck(low, high_ex, roll)


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
    luck_roll_bad: Optional[int] = None,
    *,
    counter_amp: float = 1.0,
    luck_rng: Optional[random.Random] = None,
) -> Optional[int]:
    """
    Calculate damage dealt by attacker to defender.

    Returns damage in HP points (0–100 internal scale), or None if the
    attacker cannot attack the defender (missing table entry).

    luck_roll: explicit 0–9 good-luck digit for deterministic tests; random if None.
    luck_roll_bad: second digit for dual-luck COs only (bad-luck arm); random if None.
    luck_rng: when a digit is None and this is set, draws use ``randint(0, 9)`` here;
        otherwise fall back to the process-global ``random`` module (legacy).
    counter_amp: multiplier applied to the raw damage before AWBW rounding.
        Used by ``calculate_counterattack`` to inject Sonja's D2D counter ×1.5
        (https://awbw.amarriner.com/co.php Sonja row, "counterattacks do
        1.5x more damage"). Equivalent to scaling AV inside the formula —
        applied to ``raw`` so AWBW's ceil-to-0.05/floor rounding stays the
        sole source of HP-tick truncation.
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
    av = attacker_co.total_atk_for_unit(attacker.unit_type)

    # Lash (co_id=16): terrain stars add to ATK when her power is active
    if attacker_co.co_id == 16:
        if attacker_co.scop_active or attacker_co.cop_active:
            av += attacker_terrain.defense * 10

    # Kindle (co_id=23): urban-terrain attack rider (D2D / COP / SCOP).
    av += _kindle_atk_rider(attacker_co, attacker_terrain)

    # Colin (co_id=15): D2D −10 %% + SCOP "Power of Money" attack rider.
    av += _colin_atk_rider(attacker_co)

    # --- Luck ---
    l_val, lb_val = _resolve_attack_luck_terms(
        attacker_co, luck_roll, luck_roll_bad, luck_rng,
    )

    # AWBW damage uses display HP bars (1–10, ceilinged) on both sides. Using
    # raw internal HP (1–100) for HPD makes ``dtr × hpd`` 10× too large and
    # flips ``200 - (dv + dtr × hpd)`` negative on any terrain-star tile,
    # clamping damage to 0.
    hpa_bars = attacker.display_hp
    hpd_bars = defender.display_hp

    # Sonja (co_id=18) "Hidden HP" is intentionally NOT applied to the damage
    # formula. The Fandom Damage_Formula page is explicit: "damage taken by
    # the defending unit will be calculated in the form of its true health".
    # AWBW PHP server has full information and computes damage with true HP;
    # Hidden HP is a UI-only deception that affects what the *opponent* sees
    # in the on-screen indicator (and so what they can infer in Fog of War).
    # An earlier Phase 11J-SONJA-D2D-IMPL attempt added a `hpd_bars - 1`
    # rider here; it shifted the engine's Sonja-defender damage in the right
    # direction for ~3 mid-range gids but introduced offsetting overshoot on
    # other Sonja-bearing games (per-unit cumulative drift up to 23 HP) and
    # was reverted. See docs/oracle_exception_audit/phase11j_sonja_d2d_impl.md
    # § "Reverted: Hidden HP damage rider".

    # --- Defense Value ---
    dv = defender_co.total_def_for_unit_against(defender.unit_type, attacker.unit_type)

    # Terrain defense stars
    dtr = defender_terrain.defense
    # Submerged subs/stealth get no terrain bonus (they're at sea)
    if defender.is_submerged:
        dtr = 0

    # --- Formula ---
    raw = (base * av / 100 + l_val - lb_val) * (hpa_bars / 10) * (200 - (dv + dtr * hpd_bars)) / 100
    # Sonja D2D counter ×1.5 amplification (caller-supplied; see docstring).
    raw *= counter_amp

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

    Implementation note: instead of reimplementing the formula, we sweep every
    legal luck digit through ``calculate_damage`` (one digit for single-luck
    COs; the Cartesian product ``[0,9]²`` for dual-luck COs such as Sonja /
    Flak / Jugger). Luck is the only stochastic input; everything else is
    deterministic from these args.
    """
    if get_base_damage(attacker.unit_type, defender.unit_type) is None:
        return None

    bounds = attacker_co.luck_bounds()
    dual = bounds is not None and _is_dual_luck_bounds(bounds)

    vals: list[int] = []
    if dual:
        for rg in range(10):
            for rb in range(10):
                d = calculate_damage(
                    attacker, defender,
                    attacker_terrain, defender_terrain,
                    attacker_co, defender_co,
                    luck_roll=rg,
                    luck_roll_bad=rb,
                )
                if d is None:
                    return None
                vals.append(d)
    else:
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
    luck_roll_bad: Optional[int] = None,
    *,
    luck_rng: Optional[random.Random] = None,
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

    Luck mirrors :func:`calculate_damage`: dual-luck COs consume ``luck_roll``
    (good arm) and ``luck_roll_bad`` (bad arm); omitting either draws from
    ``luck_rng`` / ``random`` independently (two draws per counter).
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
        # Sonja SCOP "Counter Break": defender attacks first, so the counter
        # rolls against pre-attack HP (capped at 100). AWBW canon:
        # https://awbw.amarriner.com/co.php — "A unit being attacked will
        # attack first (even if it would be destroyed by the attack)."
        counter_unit = copy.copy(defender)
        counter_unit.hp = min(100, defender.hp + attack_damage)
    else:
        counter_unit = defender

    # Sonja D2D "counterattacks do 1.5x more damage" (amarriner Sonja row).
    # No canon language disables this under COP/SCOP — the ×1.5 stacks on
    # top of SCOP's "attack first" pre-attack-HP path above.
    counter_amp = 1.5 if defender_co.co_id == 18 else 1.0

    return calculate_damage(
        counter_unit, attacker,
        defender_terrain, attacker_terrain,
        defender_co, attacker_co,
        luck_roll=luck_roll,
        luck_roll_bad=luck_roll_bad,
        counter_amp=counter_amp,
        luck_rng=luck_rng,
    )
