"""
Regression test: Sami vs Eagle infantry–mech beach/city anchor combat.

Anchors the AWBW damage formula fix (see ``engine/combat.py`` module
docstring). Expected bands — sourced from the AWBW community calculator —
for Sami (CO id 8, +30% ATK infantry/mech) Infantry full HP on a shoal/beach
firing at an Eagle (CO id 10, no ground bonuses) Mech full HP on a city
(3 defence stars):

  Forward strike:  **~41–47%** HP damage (luck 0–9 → 40–46).
  Counterattack:   **~39–44%** HP damage against the attacker (defender HP
                   reduced by the forward strike first).

Both bands match under the display-bar formulation of the damage formula.
Prior to the fix, the forward strike evaluated to **0%** because ``hpd``
was used as raw internal HP (1–100) inside ``dtr × hpd``, sending the
defence term negative.
"""
from __future__ import annotations

import unittest

from engine.combat import calculate_damage, calculate_counterattack
from engine.co import make_co_state
from engine.terrain import get_terrain
from engine.unit import Unit, UnitType, UNIT_STATS


def _make_unit(unit_type: UnitType, player: int, hp: int = 100) -> Unit:
    stats = UNIT_STATS[unit_type]
    return Unit(
        unit_type=unit_type,
        player=player,
        hp=hp,
        ammo=stats.max_ammo if stats.max_ammo > 0 else 0,
        fuel=stats.max_fuel,
        pos=(0, 0),
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
    )


# Terrain IDs used by the anchor:
#   VShoalE (beach)  = 32, 0 stars
#   City             = 34, 3 stars
BEACH_TID = 32
CITY_TID  = 34

# CO ids (per data/co_data.json)
SAMI_ID  = 8
EAGLE_ID = 10


class TestSamiVsEagleAnchorCombat(unittest.TestCase):
    """Damage must land inside AWBW's published bands for the anchor case."""

    def setUp(self) -> None:
        self.sami  = make_co_state(SAMI_ID)
        self.eagle = make_co_state(EAGLE_ID)

        self.attacker = _make_unit(UnitType.INFANTRY, player=1, hp=100)
        self.defender = _make_unit(UnitType.MECH,     player=0, hp=100)

        self.att_terrain = get_terrain(BEACH_TID)  # 0 stars
        self.def_terrain = get_terrain(CITY_TID)   # 3 stars

    def test_forward_strike_is_in_41_to_47_band(self) -> None:
        """With Sami +30% ATK, all luck rolls 0..9 must land in 40–46 HP."""
        results = []
        for roll in range(10):
            dmg = calculate_damage(
                self.attacker, self.defender,
                self.att_terrain, self.def_terrain,
                self.sami, self.eagle,
                luck_roll=roll,
            )
            self.assertIsNotNone(dmg)
            results.append(dmg)

        # Zero-luck baseline is the published band's mathematical floor (40).
        self.assertEqual(results[0], 40, f"unexpected baseline forward damage: {results}")
        # Full luck sweep must fall inside the published AWBW band (≈40–47).
        self.assertTrue(
            all(40 <= r <= 47 for r in results),
            f"luck sweep outside expected forward band 40..47: {results}",
        )

    def test_counter_strike_is_in_39_to_44_band(self) -> None:
        """After a ~40% forward strike, Mech counter at 6 bars must land ~39–44%.

        Matches the contract used by ``GameState._apply_attack``: the caller
        applies the forward damage to ``defender.hp`` before invoking
        ``calculate_counterattack``.
        """
        forward = calculate_damage(
            self.attacker, self.defender,
            self.att_terrain, self.def_terrain,
            self.sami, self.eagle,
            luck_roll=0,
        )
        self.assertEqual(forward, 40)
        # Post-attack defender HP, as the engine would have it.
        self.defender.hp = max(0, self.defender.hp - forward)

        results = []
        for roll in range(10):
            counter = calculate_counterattack(
                self.attacker, self.defender,
                self.att_terrain, self.def_terrain,
                self.sami, self.eagle,
                attack_damage=forward,
                luck_roll=roll,
            )
            self.assertIsNotNone(counter)
            results.append(counter)

        self.assertEqual(results[0], 39, f"unexpected baseline counter damage: {results}")
        self.assertTrue(
            all(39 <= r <= 44 for r in results),
            f"luck sweep outside expected counter band 39..44: {results}",
        )


if __name__ == "__main__":
    unittest.main()
