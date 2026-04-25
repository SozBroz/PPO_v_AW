"""Phase 11J-SONJA-D2D-IMPL — Sonja CO 18 D2D counter ×1.5.

AWBW canon:

* Tier 1 — https://awbw.amarriner.com/co.php Sonja row:
  *"Units gain +1 vision in Fog of War, have hidden HP, and counterattacks
   do 1.5x more damage. Luck is reduced to -9% to +9%.
   Enhanced Vision -- All units gain +1 vision, and can see into forests
   and reefs.
   Counter Break -- All units gain +1 vision, and can see into forests and
   reefs. A unit being attacked will attack first (even if it would be
   destroyed by the attack)."*

* Tier 2 (Damage Formula scope) — https://awbw.fandom.com/wiki/Damage_Formula:
  *"Due to the way the formula is set, damage taken by the defending unit
   will be calculated in the form of its true health."*

Engine implementation in ``engine/combat.py``:

* **Counter ×1.5** — when Sonja is the counter-attacker, ``calculate_damage``
  receives ``counter_amp=1.5`` from ``calculate_counterattack`` and scales
  raw damage by 1.5 before AWBW's ceil-to-0.05 / floor rounding. Always
  active. Stacks with SCOP "Counter Break" (which restores pre-attack HP
  via the existing branch).

* **Hidden HP** — explicitly NOT a formula change. The Fandom Damage
  Formula page is unambiguous that AWBW's PHP server uses true HP for
  damage; "hidden HP" is UI-only deception affecting what the *opponent*
  sees in the on-screen indicator (relevant in Fog of War). An earlier
  rider that injected ``display_hp - 1`` into the formula partially closed
  3 mid-range Sonja gids but introduced offsetting overshoot on other
  Sonja-bearing games (per-unit cumulative drift up to 23 HP); reverted.
  See ``docs/oracle_exception_audit/phase11j_sonja_d2d_impl.md``.
"""
from __future__ import annotations

import math

import pytest

from engine.combat import calculate_counterattack, calculate_damage
from engine.co import make_co_state_safe
from engine.terrain import get_terrain
from engine.unit import Unit, UnitType, UNIT_STATS

SONJA_CO_ID = 18
ANDY_CO_ID = 1

PLAIN = get_terrain(1)   # 0 defense stars
WOOD = get_terrain(3)    # 2 defense stars
MOUNTAIN = get_terrain(2)  # 4 defense stars


def _unit(ut: UnitType, player: int, *, hp: int = 100, pos=(0, 0)) -> Unit:
    stats = UNIT_STATS[ut]
    return Unit(
        unit_type=ut,
        player=player,
        hp=hp,
        ammo=stats.max_ammo if stats.max_ammo > 0 else 0,
        fuel=stats.max_fuel,
        pos=pos,
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
        unit_id=player * 100,
    )


# ---------------------------------------------------------------------------
# Hidden HP is UI-only — no damage formula impact (Fandom Damage_Formula)
# ---------------------------------------------------------------------------
def test_hidden_hp_does_not_alter_damage_formula() -> None:
    """Andy TANK attacks Sonja TANK on WOOD (2★). The Fandom Damage_Formula
    page states damage is calculated against true health; Sonja's "hidden
    HP" is a UI-only deception (and so does NOT shift the engine's number).
    Damage vs Sonja must equal damage vs Andy under identical setup.
    """
    attacker = _unit(UnitType.TANK, 0, hp=100)
    sonja_def = _unit(UnitType.TANK, 1, hp=70)
    andy_def = _unit(UnitType.TANK, 1, hp=70)

    andy_atk_co = make_co_state_safe(ANDY_CO_ID)
    sonja_co = make_co_state_safe(SONJA_CO_ID)
    andy_def_co = make_co_state_safe(ANDY_CO_ID)

    sonja_dmg = calculate_damage(
        attacker, sonja_def, PLAIN, WOOD,
        andy_atk_co, sonja_co, luck_roll=0,
    )
    andy_dmg = calculate_damage(
        attacker, andy_def, PLAIN, WOOD,
        andy_atk_co, andy_def_co, luck_roll=0,
    )
    # raw = 55 * 1 * (200 - 100 - 2*7)/100 = 47.3 → 47 for both.
    assert sonja_dmg == 47, sonja_dmg
    assert andy_dmg == 47, andy_dmg


# ---------------------------------------------------------------------------
# Hidden HP — neutral under COP "Enhanced Vision" too
# ---------------------------------------------------------------------------
def test_sonja_cop_does_not_alter_inbound_damage_via_hidden_hp() -> None:
    """COP "Enhanced Vision" only changes vision/fog mechanics. Inbound
    damage shifts only by the universal SCOPB DEF +10, never by a
    Hidden-HP rider.
    """
    attacker = _unit(UnitType.TANK, 0, hp=100)
    sonja_def = _unit(UnitType.TANK, 1, hp=70)

    andy_atk_co = make_co_state_safe(ANDY_CO_ID)
    sonja_cop = make_co_state_safe(SONJA_CO_ID)
    sonja_cop.cop_active = True
    sonja_d2d = make_co_state_safe(SONJA_CO_ID)

    cop_dmg = calculate_damage(
        attacker, sonja_def, PLAIN, WOOD,
        andy_atk_co, sonja_cop, luck_roll=0,
    )
    d2d_dmg = calculate_damage(
        attacker, sonja_def, PLAIN, WOOD,
        andy_atk_co, sonja_d2d, luck_roll=0,
    )
    # raw_d2d = 55 * (200 - 100 - 14)/100 = 47.3 → 47.
    # raw_cop = 55 * (200 - 110 - 14)/100 = 41.8 → 41.
    assert d2d_dmg == 47
    assert cop_dmg == 41
    assert cop_dmg < d2d_dmg


# ---------------------------------------------------------------------------
# Hidden HP — neutral under SCOP "Counter Break" too
# ---------------------------------------------------------------------------
def test_sonja_scop_does_not_alter_inbound_damage_via_hidden_hp() -> None:
    """SCOP "Counter Break" only flips counter timing. Inbound damage
    shifts only by SCOPB DEF +10.
    """
    attacker = _unit(UnitType.TANK, 0, hp=100)
    sonja_def = _unit(UnitType.TANK, 1, hp=70)

    andy_atk_co = make_co_state_safe(ANDY_CO_ID)
    sonja_scop = make_co_state_safe(SONJA_CO_ID)
    sonja_scop.scop_active = True

    scop_dmg = calculate_damage(
        attacker, sonja_def, PLAIN, WOOD,
        andy_atk_co, sonja_scop, luck_roll=0,
    )
    # raw = 55 * (200 - 110 - 14)/100 = 41.8 → 41.
    assert scop_dmg == 41


# ---------------------------------------------------------------------------
# Counter ×1.5 — D2D base case
# ---------------------------------------------------------------------------
def test_sonja_counter_amplifier_d2d() -> None:
    """Sonja TANK at hp=50 counters Andy TANK on plains (dtr=1).
    Baseline raw = 55 * (5/10) * (200 - 100 - 1*10)/100 = 24.75 → 24.
    With ×1.5 → 37.125; ceil-to-0.05 → 37.15; floor → 37.
    AWBW canon: Sonja D2D "counterattacks do 1.5x more damage".
    """
    attacker = _unit(UnitType.TANK, 0, hp=100)
    sonja_def = _unit(UnitType.TANK, 1, hp=50)
    andy_def = _unit(UnitType.TANK, 1, hp=50)

    andy_atk_co = make_co_state_safe(ANDY_CO_ID)
    sonja_co = make_co_state_safe(SONJA_CO_ID)
    andy_def_co = make_co_state_safe(ANDY_CO_ID)

    sonja_counter = calculate_counterattack(
        attacker, sonja_def, PLAIN, PLAIN,
        andy_atk_co, sonja_co,
        attack_damage=0, luck_roll=0,
    )
    andy_counter = calculate_counterattack(
        attacker, andy_def, PLAIN, PLAIN,
        andy_atk_co, andy_def_co,
        attack_damage=0, luck_roll=0,
    )
    assert andy_counter == 24, andy_counter
    assert sonja_counter == 37, sonja_counter
    # Approximately ×1.5 with AWBW rounding noise.
    assert sonja_counter / andy_counter == pytest.approx(1.5, abs=0.06)


# ---------------------------------------------------------------------------
# Counter ×1.5 — non-Sonja CO unaffected
# ---------------------------------------------------------------------------
def test_no_counter_amplifier_for_andy() -> None:
    """Sanity: removing Sonja from the defender slot removes the ×1.5.
    Confirms the gate is strictly ``defender_co.co_id == 18``.
    """
    attacker = _unit(UnitType.TANK, 0, hp=100)
    andy_def = _unit(UnitType.TANK, 1, hp=50)

    andy_atk_co = make_co_state_safe(ANDY_CO_ID)
    andy_def_co = make_co_state_safe(ANDY_CO_ID)

    counter = calculate_counterattack(
        attacker, andy_def, PLAIN, PLAIN,
        andy_atk_co, andy_def_co,
        attack_damage=0, luck_roll=0,
    )
    # Baseline (plains dtr=1): 55 * 0.5 * (200 - 100 - 10)/100 = 24.75 → 24.
    assert counter == 24


# ---------------------------------------------------------------------------
# Counter ×1.5 — still active under COP / SCOP
# ---------------------------------------------------------------------------
def test_sonja_counter_amplifier_active_under_cop() -> None:
    attacker = _unit(UnitType.TANK, 0, hp=100)
    sonja_def = _unit(UnitType.TANK, 1, hp=50)
    sonja_cop = make_co_state_safe(SONJA_CO_ID)
    sonja_cop.cop_active = True
    andy_atk_co = make_co_state_safe(ANDY_CO_ID)

    counter = calculate_counterattack(
        attacker, sonja_def, PLAIN, PLAIN,
        andy_atk_co, sonja_cop,
        attack_damage=0, luck_roll=0,
    )
    # COP grants Sonja the universal SCOPB +10 ATK → AV 110.
    # raw = 55 * 110/100 * 0.5 * (200 - 100 - 1*10)/100 = 27.225
    # *1.5 = 40.8375; ceil-to-0.05 → 40.85; floor → 40.
    assert counter == 40


def test_sonja_counter_amplifier_stacks_with_scop_counter_break() -> None:
    """SCOP "Counter Break" restores pre-attack HP for the counter, AND
    Sonja's D2D ×1.5 still applies on top. Defender at hp=50 takes a
    20-HP hit → would normally counter at hp=30 (display 3); under SCOP
    the counter rolls at hp=50 (display 5), then ×1.5.
    """
    attacker = _unit(UnitType.TANK, 0, hp=100)
    sonja_def = _unit(UnitType.TANK, 1, hp=30)  # post-attack HP
    sonja_scop = make_co_state_safe(SONJA_CO_ID)
    sonja_scop.scop_active = True
    andy_atk_co = make_co_state_safe(ANDY_CO_ID)

    counter = calculate_counterattack(
        attacker, sonja_def, PLAIN, PLAIN,
        andy_atk_co, sonja_scop,
        attack_damage=20, luck_roll=0,  # restored to 50 → display 5
    )
    # SCOPB +10 ATK on Sonja under SCOP → AV 110.
    # raw = 55 * 110/100 * (5/10) * (200 - 100 - 1*10)/100 = 27.225
    # *1.5 = 40.8375 → 40. (Same as COP-active counter — Counter Break
    # restores HP via the existing branch above; ×1.5 stacks on top.)
    assert counter == 40


# ---------------------------------------------------------------------------
# Combined: forward damage AND counter ×1.5 fires (terrain-stars + counter)
# ---------------------------------------------------------------------------
def test_forward_then_counter_amp_chain() -> None:
    """Andy TANK attacks Sonja TANK on WOOD (2★) at hp=70 (no Hidden-HP
    rider — true HP used). Counter from surviving Sonja unit at hp=70-47=23
    (display 3) on plains rolls ×1.5.
    """
    attacker = _unit(UnitType.TANK, 0, hp=100)
    sonja_def = _unit(UnitType.TANK, 1, hp=70)

    andy_atk_co = make_co_state_safe(ANDY_CO_ID)
    sonja_co = make_co_state_safe(SONJA_CO_ID)

    fwd = calculate_damage(
        attacker, sonja_def, PLAIN, WOOD,
        andy_atk_co, sonja_co, luck_roll=0,
    )
    # raw = 55 * (200 - 100 - 14)/100 = 47.3 → 47.
    assert fwd == 47
    sonja_def.hp = max(0, sonja_def.hp - fwd)  # 23 → display 3

    counter = calculate_counterattack(
        attacker, sonja_def, PLAIN, PLAIN,
        andy_atk_co, sonja_co,
        attack_damage=fwd, luck_roll=0,
    )
    # raw = 55 * 1.0 * (3/10) * (200 - 100 - 1*10)/100 = 14.85
    # *1.5 = 22.275; ceil-to-0.05 → 22.30; floor → 22.
    assert counter == 22


# ---------------------------------------------------------------------------
# Hidden HP audit pin — Sonja never alters inbound damage at any HP/terrain
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hp", [1, 15, 25, 50, 70, 95, 100])
@pytest.mark.parametrize("terrain", [PLAIN, WOOD, MOUNTAIN])
def test_sonja_inbound_damage_matches_andy_at_every_hp_and_terrain(
    hp: int, terrain
) -> None:
    """Pin: across every HP bucket and terrain-star tier, Sonja's inbound
    damage equals Andy's inbound damage. Guards against any future regression
    that would re-introduce a Hidden-HP damage rider.
    """
    attacker = _unit(UnitType.TANK, 0, hp=100)
    sonja_def = _unit(UnitType.TANK, 1, hp=hp)
    andy_def = _unit(UnitType.TANK, 1, hp=hp)
    andy_atk_co = make_co_state_safe(ANDY_CO_ID)
    sonja_co = make_co_state_safe(SONJA_CO_ID)
    andy_def_co = make_co_state_safe(ANDY_CO_ID)

    sonja_dmg = calculate_damage(
        attacker, sonja_def, PLAIN, terrain,
        andy_atk_co, sonja_co, luck_roll=0,
    )
    andy_dmg = calculate_damage(
        attacker, andy_def, PLAIN, terrain,
        andy_atk_co, andy_def_co, luck_roll=0,
    )
    assert sonja_dmg == andy_dmg, (hp, terrain.name, sonja_dmg, andy_dmg)
