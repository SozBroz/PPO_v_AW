"""Damage-formula baseline validation.

At zero luck, full HP, 0★ terrain, neutral COs, the engine's
``calculate_damage`` reduces algebraically to the raw ``data/damage_table.json``
entry::

    raw = (B * 100/100 + 0 - 0) * (10/10) * (200 - (100 + 0*10)) / 100 = B

Any deviation from ``B`` for any (attacker, defender) pair indicates a bug in
the formula chain (CO modifiers, terrain stars, HP scaling, or rounding).

This is the no-modifier anchor; ``test_combat_anchor.py`` covers the
Sami-on-beach vs Eagle-on-city case (CO ATK + terrain stars + HP scaling
all firing simultaneously). Together they validate the formula end-to-end.
"""
from __future__ import annotations

import unittest

from engine.combat import calculate_damage, get_base_damage
from engine.co import make_co_state
from engine.terrain import get_terrain
from engine.unit import Unit, UnitType, UNIT_STATS

ROAD_TID = 15  # 0★ defense terrain
ANDY_ID = 1


def _mk(unit_type: UnitType, player: int, hp: int = 100) -> Unit:
    s = UNIT_STATS[unit_type]
    return Unit(
        unit_type=unit_type,
        player=player,
        hp=hp,
        ammo=s.max_ammo if s.max_ammo > 0 else 0,
        fuel=s.max_fuel,
        pos=(0, 0),
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
    )


class TestCombatFormulaBaseline(unittest.TestCase):
    def test_road_neutral_andy_zero_luck_reproduces_table(self) -> None:
        co0 = make_co_state(ANDY_ID)
        co1 = make_co_state(ANDY_ID)
        road = get_terrain(ROAD_TID)
        self.assertEqual(road.defense, 0)

        checked = 0
        bad: list[tuple[str, str, int, int]] = []
        for atk in UnitType:
            for dfn in UnitType:
                base = get_base_damage(atk, dfn)
                if base is None:
                    continue
                attacker = _mk(atk, player=1)
                defender = _mk(dfn, player=0)
                dmg = calculate_damage(
                    attacker, defender, road, road, co0, co1, luck_roll=0
                )
                self.assertIsNotNone(dmg)
                checked += 1
                if dmg != base:
                    bad.append((atk.name, dfn.name, base, int(dmg)))

        self.assertGreater(checked, 250, "damage table sweep too small")
        self.assertEqual(
            bad, [],
            f"{len(bad)} (atk,dfn) pairs deviate from raw table at no-luck "
            f"baseline (expected B == result): first {bad[:5]}",
        )


if __name__ == "__main__":
    unittest.main()
