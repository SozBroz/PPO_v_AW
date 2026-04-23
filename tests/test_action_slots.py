"""Regression pins for Action dataclass __slots__ (Phase 8a)."""

from dataclasses import fields

import pytest

from engine.action import Action, ActionType
from engine.unit import UnitType


def test_action_has_slots() -> None:
    assert hasattr(Action, "__slots__")
    inst = Action(ActionType.END_TURN)
    assert "__dict__" not in dir(inst)


def test_action_attributes_settable() -> None:
    a = Action(ActionType.END_TURN)
    a.action_type = ActionType.WAIT
    assert a.action_type == ActionType.WAIT
    a.unit_pos = (1, 2)
    assert a.unit_pos == (1, 2)
    a.move_pos = (3, 4)
    assert a.move_pos == (3, 4)
    a.target_pos = (5, 6)
    assert a.target_pos == (5, 6)
    a.unit_type = UnitType.TANK
    assert a.unit_type == UnitType.TANK
    a.unload_pos = (7, 8)
    assert a.unload_pos == (7, 8)
    a.select_unit_id = 99
    assert a.select_unit_id == 99


def test_action_unknown_attribute_raises() -> None:
    a = Action(ActionType.END_TURN)
    with pytest.raises(AttributeError):
        a.bogus_field = 1  # type: ignore[attr-defined]


def test_action_repr_unchanged() -> None:
    assert repr(Action(action_type=ActionType.END_TURN)) == "Action(END_TURN)"


def test_action_dataclass_fields_intact() -> None:
    assert {f.name for f in fields(Action)} == {
        "action_type",
        "unit_pos",
        "move_pos",
        "target_pos",
        "unit_type",
        "unload_pos",
        "select_unit_id",
    }
