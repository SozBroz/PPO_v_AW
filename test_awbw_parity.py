"""AWBW parity: predeployed units from sidecar JSON; army wipe only after combat."""
import unittest
from pathlib import Path

from engine.action import Action, ActionStage, ActionType
from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import Unit, UnitType, UNIT_STATS

ROOT = Path(__file__).parent
POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


class TestPredeployedMap133665(unittest.TestCase):
    def test_load_map_has_two_predeployed_specs(self) -> None:
        md = load_map(133665, POOL, MAPS_DIR)
        self.assertEqual(len(md.predeployed_specs), 2)
        players = {s.player for s in md.predeployed_specs}
        self.assertEqual(players, {0, 1})

    def test_make_initial_state_spawns_units(self) -> None:
        md = load_map(133665, POOL, MAPS_DIR)
        st = make_initial_state(md, 1, 7, starting_funds=0, tier_name="T2")
        self.assertEqual(len(st.units[0]) + len(st.units[1]), 2)
        self.assertGreater(len(st.units[0]), 0)
        self.assertGreater(len(st.units[1]), 0)


class TestUnitWipeCombatOnly(unittest.TestCase):
    def test_global_check_does_not_army_wipe(self) -> None:
        """Empty vs non-empty alone must not end the game (build / economy)."""
        md = load_map(133665, POOL, MAPS_DIR)
        st = make_initial_state(md, 1, 7, starting_funds=0, tier_name="T2")
        st.units = {
            0: st.units[0],
            1: [],
        }
        st.done = False
        st.winner = None
        r = st._check_win_conditions(0)
        self.assertFalse(st.done)
        self.assertEqual(r, 0.0)

    def test_wipe_after_attack_eliminates_last_unit(self) -> None:
        md = load_map(133665, POOL, MAPS_DIR)
        st = make_initial_state(md, 1, 7, starting_funds=0, tier_name="T2")
        # Two infantries adjacent orthogonally: P0 at (0,2), move to (1,2) and hit P1 at (1,3)?
        # Simpler: place both on same row adjacent
        inf = UNIT_STATS[UnitType.INFANTRY]
        u0 = Unit(
            unit_type=UnitType.INFANTRY,
            player=0,
            hp=100,
            ammo=inf.max_ammo if inf.max_ammo > 0 else 0,
            fuel=inf.max_fuel,
            pos=(5, 5),
            moved=False,
            loaded_units=[],
            is_submerged=False,
            capture_progress=20,
        )
        u1 = Unit(
            unit_type=UnitType.INFANTRY,
            player=1,
            hp=1,
            ammo=inf.max_ammo if inf.max_ammo > 0 else 0,
            fuel=inf.max_fuel,
            pos=(5, 6),
            moved=False,
            loaded_units=[],
            is_submerged=False,
            capture_progress=20,
        )
        st.units = {0: [u0], 1: [u1]}
        st.active_player = 0
        st.action_stage = ActionStage.ACTION
        st.selected_unit = u0
        st.selected_move_pos = (5, 5)

        action = Action(
            ActionType.ATTACK,
            unit_pos=u0.pos,
            move_pos=(5, 5),
            target_pos=(5, 6),
        )
        st.step(action)
        self.assertTrue(st.done)
        self.assertEqual(st.winner, 0)
        self.assertEqual(st.win_reason, "army_wipe")


if __name__ == "__main__":
    unittest.main()
