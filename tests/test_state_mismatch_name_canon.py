"""Regression for Phase 11J unit-type name canonicalization (state-mismatch only)."""

from __future__ import annotations

from tools.desync_audit import _canonicalize_unit_type_name


def test_missile_missiles_plural_fold() -> None:
    assert _canonicalize_unit_type_name("Missile") == _canonicalize_unit_type_name(
        "Missiles"
    )


def test_mega_tank_spacing_fold() -> None:
    assert _canonicalize_unit_type_name("Megatank") == _canonicalize_unit_type_name(
        "Mega Tank"
    )


def test_medium_tank_spellings_fold() -> None:
    canon = _canonicalize_unit_type_name("Medium Tank")
    assert canon == _canonicalize_unit_type_name("Md.Tank")
    assert canon == _canonicalize_unit_type_name("Md. Tank")
    assert canon == _canonicalize_unit_type_name("MdTank")
    assert canon == _canonicalize_unit_type_name("MED_TANK")


def test_bcopter_spellings_fold() -> None:
    canon = _canonicalize_unit_type_name("B-Copter")
    assert canon == _canonicalize_unit_type_name("B_COPTER")
    assert canon == _canonicalize_unit_type_name("B Copter")


def test_tank_not_mega_tank() -> None:
    assert _canonicalize_unit_type_name("Tank") != _canonicalize_unit_type_name(
        "Mega Tank"
    )


def test_bomber_not_bcopter() -> None:
    assert _canonicalize_unit_type_name("Bomber") != _canonicalize_unit_type_name(
        "B_COPTER"
    )


def test_submarine_sub_abbrev_fold() -> None:
    assert _canonicalize_unit_type_name("Submarine") == _canonicalize_unit_type_name(
        "Sub"
    )
