"""Tests for BUILD flat encoding (per-factory tile + unit type)."""
import unittest
from unittest.mock import patch

from engine.action import Action, ActionType
from engine.unit import UnitType
from rl.env import _action_to_flat, _flat_to_action, _get_action_mask
from rl.network import ACTION_SPACE_SIZE


class TestBuildActionEncoding(unittest.TestCase):
    def test_distinct_indices_same_unit_type_different_tile(self) -> None:
        a = Action(ActionType.BUILD, move_pos=(5, 5), unit_type=UnitType.INFANTRY)
        b = Action(ActionType.BUILD, move_pos=(10, 3), unit_type=UnitType.INFANTRY)
        self.assertNotEqual(_action_to_flat(a), _action_to_flat(b))

    def test_max_index_fits_action_space(self) -> None:
        a = Action(ActionType.BUILD, move_pos=(29, 29), unit_type=UnitType.OOZIUM)
        idx = _action_to_flat(a)
        self.assertLess(idx, ACTION_SPACE_SIZE)

    def test_flat_to_action_roundtrip(self) -> None:
        a1 = Action(ActionType.BUILD, move_pos=(1, 1), unit_type=UnitType.TANK)
        a2 = Action(ActionType.BUILD, move_pos=(2, 2), unit_type=UnitType.TANK)
        with patch("rl.env.get_legal_actions", return_value=[a1, a2]):
            self.assertIs(_flat_to_action(_action_to_flat(a1), None), a1)
            self.assertIs(_flat_to_action(_action_to_flat(a2), None), a2)

    def test_action_mask_sets_distinct_bits(self) -> None:
        a1 = Action(ActionType.BUILD, move_pos=(1, 1), unit_type=UnitType.INFANTRY)
        a2 = Action(ActionType.BUILD, move_pos=(2, 2), unit_type=UnitType.INFANTRY)
        with patch("rl.env.get_legal_actions", return_value=[a1, a2]):
            mask = _get_action_mask(None)
            self.assertTrue(mask[_action_to_flat(a1)])
            self.assertTrue(mask[_action_to_flat(a2)])
            self.assertEqual(int(mask.sum()), 2)


if __name__ == "__main__":
    unittest.main()
