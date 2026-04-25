"""Phase 10A regression: B_COPTER air-unit ``Fire`` parity.

The Phase 10A briefing framed the 47 GL std-tier engine_bug rows as a
B_COPTER **movement parity** bug (analogous to Lane M's Andy SCOP +1 fix).
Drilling the smallest-drift games (gid 1631621 drift=0, gid 1621170 drift=1)
showed the actual root cause is **not** movement: the engine and AWBW agree
on the firing tile reachability. The first divergence is ``get_attack_targets``
returning [] for two unrelated reasons, both AWBW canon violations:

  1. **Damage-table holes**: B_COPTER vs LANDER / BLACK_BOAT and
     RECON vs B_COPTER / T_COPTER were ``None``, so adjacent direct fire
     was hidden even though AWBW's primary-source damage chart
     (https://awbw.amarriner.com/damage.php) lists them as legal hits
     (B-Copter row entries 25 / 25; Recon row entries 10 / 35).

  2. **Secondary MG consumed primary ammo**: the engine drained one round
     of primary magazine on **every** strike, including ones AWBW resolves
     with the secondary Machine Gun (https://awbw.fandom.com/wiki/Machine_Gun).
     B-Copter / Mech / Tank / Md.Tank / Neotank / Mega Tank attacks against
     **Infantry** or **Mech** defenders use the unlimited MG; primary ammo
     must not tick down. Pre-fix, magazines bottomed out 2-5 turns before
     AWBW expected, raising ``_apply_attack`` range / ammo errors when the
     engine refused a later legitimate primary-weapon strike.

This file is named ``test_b_copter_movement_parity`` to match the Phase 10A
plan's expected artifact name; the docstring above is the corrective record
of what actually shipped.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.action import Action, ActionType, get_attack_targets  # noqa: E402
from engine.combat import get_base_damage  # noqa: E402
from engine.unit import UNIT_STATS, UnitType  # noqa: E402

from test_lander_and_fuel import _fresh_state, _make_unit  # noqa: E402


# ---------------------------------------------------------------------------
# Damage-table parity — proves Bucket 1 is closed.
# ---------------------------------------------------------------------------
class TestDamageTableAirNavalGaps(unittest.TestCase):
    """Entries that were ``None`` pre-Phase-10A and now match AWBW canon."""

    def test_b_copter_vs_lander(self) -> None:
        self.assertEqual(get_base_damage(UnitType.B_COPTER, UnitType.LANDER), 25)

    def test_b_copter_vs_black_boat(self) -> None:
        self.assertEqual(
            get_base_damage(UnitType.B_COPTER, UnitType.BLACK_BOAT), 25
        )

    def test_recon_vs_b_copter(self) -> None:
        self.assertEqual(get_base_damage(UnitType.RECON, UnitType.B_COPTER), 10)

    def test_recon_vs_t_copter(self) -> None:
        self.assertEqual(get_base_damage(UnitType.RECON, UnitType.T_COPTER), 35)


# ---------------------------------------------------------------------------
# get_attack_targets parity — proves Bucket 1 unblocks `_apply_attack`.
# ---------------------------------------------------------------------------
class TestBCopterFireOnNavalTransports(unittest.TestCase):
    """Mirrors the gid 1621170 / 1633990 / 1634571 failure shape: B_COPTER
    adjacent to enemy BLACK_BOAT / LANDER must list the boat as a legal
    attack target.
    """

    def _state_with_copter_and(self, defender_type: UnitType):
        st = _fresh_state()
        st.active_player = 0
        # Sea row 0; copter sits over sea on (0, 2), defender on (0, 3).
        copter = _make_unit(st, UnitType.B_COPTER, 0, (0, 2))
        _make_unit(st, defender_type, 1, (0, 3))
        return st, copter

    def test_b_copter_can_strike_adjacent_black_boat(self) -> None:
        st, copter = self._state_with_copter_and(UnitType.BLACK_BOAT)
        self.assertIn((0, 3), get_attack_targets(st, copter, copter.pos))

    def test_b_copter_can_strike_adjacent_lander(self) -> None:
        st, copter = self._state_with_copter_and(UnitType.LANDER)
        self.assertIn((0, 3), get_attack_targets(st, copter, copter.pos))


# ---------------------------------------------------------------------------
# MG ammo bookkeeping — proves Bucket 2 is closed.
# ---------------------------------------------------------------------------
def _select_attack(state, attacker, target_pos):
    state.step(Action(ActionType.SELECT_UNIT, unit_pos=attacker.pos))
    state.step(Action(
        ActionType.SELECT_UNIT, unit_pos=attacker.pos, move_pos=attacker.pos
    ))
    state.step(Action(
        ActionType.ATTACK,
        unit_pos=attacker.pos,
        move_pos=attacker.pos,
        target_pos=target_pos,
    ))


class TestMGSecondaryDoesNotConsumePrimaryAmmo(unittest.TestCase):
    """AWBW canon: when MECH / TANK / MED_TANK / NEO_TANK / MEGA_TANK /
    B_COPTER hit an INFANTRY or MECH defender, the secondary Machine Gun
    fires (no ammo). Strikes vs other unit classes still consume primary
    ammo (one round per shot).
    """

    def _atk_state(self, attacker_type, defender_type, atk_pos, def_pos):
        st = _fresh_state()
        st.active_player = 0
        atk = _make_unit(st, attacker_type, 0, atk_pos)
        defn = _make_unit(st, defender_type, 1, def_pos)
        return st, atk, defn

    def test_b_copter_vs_infantry_mg_does_not_consume(self) -> None:
        # B-Copter on plains (2, 2), enemy infantry on plains (2, 3).
        st, copter, _ = self._atk_state(
            UnitType.B_COPTER, UnitType.INFANTRY, (2, 2), (2, 3)
        )
        starting_ammo = copter.ammo
        self.assertEqual(starting_ammo, UNIT_STATS[UnitType.B_COPTER].max_ammo)
        _select_attack(st, copter, (2, 3))
        self.assertEqual(copter.ammo, starting_ammo,
                         "MG attack must not draw from primary magazine")

    def test_b_copter_vs_tank_primary_consumes(self) -> None:
        st, copter, _ = self._atk_state(
            UnitType.B_COPTER, UnitType.TANK, (2, 2), (2, 3)
        )
        starting_ammo = copter.ammo
        _select_attack(st, copter, (2, 3))
        self.assertEqual(copter.ammo, starting_ammo - 1,
                         "Primary missile vs TANK consumes one round")

    def test_mech_vs_infantry_mg_does_not_consume(self) -> None:
        # Mech (2,2) vs Infantry (2,3), both on plains so foot can fight.
        st, mech, _ = self._atk_state(
            UnitType.MECH, UnitType.INFANTRY, (2, 2), (2, 3)
        )
        starting_ammo = mech.ammo
        _select_attack(st, mech, (2, 3))
        self.assertEqual(mech.ammo, starting_ammo,
                         "Mech MG attack must not consume bazooka ammo")

    def test_mech_vs_tank_primary_consumes(self) -> None:
        st, mech, _ = self._atk_state(
            UnitType.MECH, UnitType.TANK, (2, 2), (2, 3)
        )
        starting_ammo = mech.ammo
        _select_attack(st, mech, (2, 3))
        self.assertEqual(mech.ammo, starting_ammo - 1,
                         "Bazooka vs TANK consumes one round")

    def test_mega_tank_vs_infantry_mg_does_not_consume(self) -> None:
        st, mega, _ = self._atk_state(
            UnitType.MEGA_TANK, UnitType.INFANTRY, (2, 2), (2, 3)
        )
        starting_ammo = mega.ammo
        _select_attack(st, mega, (2, 3))
        self.assertEqual(mega.ammo, starting_ammo)

    def test_tank_vs_recon_primary_consumes(self) -> None:
        # Recon is *vehicle* (not in MG target set), so cannon fires.
        st, tank, _ = self._atk_state(
            UnitType.TANK, UnitType.RECON, (2, 2), (2, 3)
        )
        starting_ammo = tank.ammo
        _select_attack(st, tank, (2, 3))
        self.assertEqual(tank.ammo, starting_ammo - 1,
                         "TANK vs RECON uses cannon, must consume ammo")


if __name__ == "__main__":
    unittest.main()
