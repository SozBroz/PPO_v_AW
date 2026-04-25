"""Phase 11Z — canonical unit-name resolver invariants.

These tests are the contract for ``engine/unit_naming.py``. Every consumer
(oracle replay, state-mismatch comparator, PHP exporter, predeployed
fetcher) routes through that module; if the contract here changes, the
audit doc ``docs/oracle_exception_audit/phase11z_unit_naming_canon_audit.md``
should be updated in lockstep.
"""
from __future__ import annotations

from types import MappingProxyType

import pytest

from engine.unit import UnitType
from engine.unit_naming import (
    UnitNameSurface,
    UnknownUnitName,
    all_known_aliases,
    from_unit_type,
    is_known_alias,
    normalize_alias_key,
    to_unit_type,
)


def test_every_unittype_has_every_surface_canonical_render() -> None:
    """``from_unit_type`` must succeed for all (UT, surface) pairs."""
    for ut in UnitType:
        for surface in UnitNameSurface:
            rendered = from_unit_type(ut, surface)
            assert isinstance(rendered, str) and rendered, (ut, surface)


def test_roundtrip_every_unittype_every_surface() -> None:
    """``to_unit_type(from_unit_type(ut, s), surface=s) == ut`` for all pairs."""
    for ut in UnitType:
        for surface in UnitNameSurface:
            rendered = from_unit_type(ut, surface)
            assert to_unit_type(rendered, surface=surface) == ut, (
                ut, surface, rendered,
            )


def test_missile_singular_and_plural_both_resolve() -> None:
    """The eagle GL bleeder: ``Missile`` (site singular) vs ``Missiles`` (engine)."""
    assert to_unit_type("Missile") == UnitType.MISSILES
    assert to_unit_type("Missiles") == UnitType.MISSILES


def test_submarine_abbrev_and_full() -> None:
    assert to_unit_type("Sub") == UnitType.SUBMARINE
    assert to_unit_type("Submarine") == UnitType.SUBMARINE


def test_med_tank_all_known_spellings() -> None:
    for s in ("Md.Tank", "Md. Tank", "MdTank", "Medium Tank", "Med Tank", "MED_TANK"):
        assert to_unit_type(s) == UnitType.MED_TANK, s


def test_neo_tank_all_known_spellings() -> None:
    for s in ("Neotank", "Neo Tank", "NEO_TANK", "NeoTank"):
        assert to_unit_type(s) == UnitType.NEO_TANK, s


def test_mega_tank_all_known_spellings() -> None:
    for s in ("Megatank", "Mega Tank", "MEGA_TANK", "MegaTank"):
        assert to_unit_type(s) == UnitType.MEGA_TANK, s


def test_anti_air_all_known_spellings() -> None:
    for s in ("Anti-Air", "Anti Air", "ANTI_AIR", "AntiAir"):
        assert to_unit_type(s) == UnitType.ANTI_AIR, s


def test_copter_all_known_spellings() -> None:
    for s in ("B-Copter", "B Copter", "B_COPTER", "BCopter"):
        assert to_unit_type(s) == UnitType.B_COPTER, s
    for s in ("T-Copter", "T Copter", "T_COPTER", "TCopter"):
        assert to_unit_type(s) == UnitType.T_COPTER, s


def test_rocket_singular_and_plural() -> None:
    assert to_unit_type("Rocket") == UnitType.ROCKET
    assert to_unit_type("Rockets") == UnitType.ROCKET


def test_case_and_punctuation_insensitive() -> None:
    """``normalize_alias_key`` strips space/hyphen/period/underscore + lowercases."""
    assert to_unit_type("md tank") == UnitType.MED_TANK
    assert to_unit_type("MEDIUMTANK") == UnitType.MED_TANK
    assert to_unit_type("anti  air") == UnitType.ANTI_AIR
    assert to_unit_type("anti-air") == UnitType.ANTI_AIR
    assert to_unit_type("ANTI-AIR") == UnitType.ANTI_AIR
    assert to_unit_type("  Sub  ") == UnitType.SUBMARINE


def test_unknown_raises_with_surface_list() -> None:
    with pytest.raises(UnknownUnitName) as exc_info:
        to_unit_type("Definitely Not A Unit")
    err = exc_info.value
    assert err.name == "Definitely Not A Unit"
    assert UnitNameSurface.AWBW_PHP in err.tried_surfaces
    assert UnitNameSurface.AWBW_VIEWER in err.tried_surfaces
    msg = str(err)
    assert "Definitely Not A Unit" in msg
    for s in UnitNameSurface:
        assert s.value in msg


def test_unit_type_passthrough() -> None:
    """Passing a UnitType in returns it unchanged."""
    assert to_unit_type(UnitType.MISSILES) == UnitType.MISSILES


def test_is_known_alias_predicate() -> None:
    assert is_known_alias("Missile")
    assert is_known_alias("Missiles")
    assert is_known_alias("MED_TANK")
    assert not is_known_alias("Definitely Not A Unit")
    assert not is_known_alias("")


def test_all_known_aliases_is_frozen_mapping() -> None:
    aliases = all_known_aliases()
    assert isinstance(aliases, MappingProxyType)
    with pytest.raises(TypeError):
        aliases["FooBar"] = UnitType.INFANTRY  # type: ignore[index]


def test_all_known_aliases_coverage_floor() -> None:
    """Coverage gate: any deletion of a known alias trips this test.

    Empirical floor at audit time: 27 enum members + 27 engine canon
    entries + AWBW PHP / VIEWER / DAMAGE_PHP / DISPLAY entries +
    explicit aliases = 72 distinct original-spelling aliases. The
    ``>= 70`` floor leaves a tiny cushion for renames inside a single
    surface but blocks accidental drops of multi-surface aliases.
    """
    assert len(all_known_aliases()) >= 70


def test_engine_canonical_matches_unit_stats_name() -> None:
    """``ENGINE`` surface render == ``UNIT_STATS[ut].name`` for every UT."""
    from engine.unit import UNIT_STATS

    for ut in UnitType:
        assert from_unit_type(ut, UnitNameSurface.ENGINE) == UNIT_STATS[ut].name, ut


def test_awbw_php_surface_matches_export_table() -> None:
    """The ``_AWBW_UNIT_NAMES`` exporter table must equal canon AWBW_PHP renders."""
    from tools.export_awbw_replay import _AWBW_UNIT_NAMES

    for ut in UnitType:
        assert _AWBW_UNIT_NAMES[ut] == from_unit_type(ut, UnitNameSurface.AWBW_PHP), ut


def test_normalize_alias_key_idempotent() -> None:
    for s in ("Md.Tank", "Md. Tank", "Mega Tank", "B-Copter", "Anti Air"):
        n = normalize_alias_key(s)
        assert normalize_alias_key(n) == n
