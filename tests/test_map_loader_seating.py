"""Tests for optional p0_country_id seating (red=P0, blue=P1)."""

from __future__ import annotations

from engine.map_loader import PropertyState, apply_p0_country_id_seating
from engine.predeployed import PredeployedUnitSpec
from engine.unit import UnitType


def _prop(
    tid: int,
    r: int,
    c: int,
    owner: int | None,
    *,
    is_hq: bool = False,
    is_base: bool = False,
) -> PropertyState:
    return PropertyState(
        terrain_id=tid,
        row=r,
        col=c,
        owner=owner,
        capture_points=20,
        is_hq=is_hq,
        is_lab=False,
        is_comm_tower=False,
        is_base=is_base,
        is_airport=False,
        is_port=False,
    )


def test_apply_p0_country_id_identity():
    props = [
        _prop(95, 0, 0, 0, is_hq=True),
        _prop(39, 1, 0, 1, is_base=True),
    ]
    scan = {5: 0, 1: 1}
    spec = PredeployedUnitSpec(0, 1, 0, UnitType.INFANTRY)
    new_ctp, specs, hq, lab = apply_p0_country_id_seating(
        props, scan, 5, [spec], map_id=0, map_name="test"
    )
    assert new_ctp == {5: 0, 1: 1}
    assert props[0].owner == 0 and props[1].owner == 1
    assert specs[0].player == 0
    assert hq[0] == [(0, 0)] and hq[1] == []


def test_apply_p0_country_id_swaps_os_to_red_seat():
    """BH (5) was scan-P0; force OS (1) onto player 0 and remap predeployed."""
    props = [
        _prop(95, 0, 0, 0, is_hq=True),
        _prop(39, 1, 0, 1, is_base=True),
    ]
    scan = {5: 0, 1: 1}
    spec = PredeployedUnitSpec(0, 1, 0, UnitType.INFANTRY)
    new_ctp, specs, hq, lab = apply_p0_country_id_seating(
        props, scan, 1, [spec], map_id=0, map_name="test"
    )
    assert new_ctp == {1: 0, 5: 1}
    assert props[0].owner == 1
    assert props[1].owner == 0
    assert specs[0].player == 1
    assert hq[1] == [(0, 0)]
    assert hq[0] == []
    assert lab[0] == [] and lab[1] == []


def test_apply_p0_country_id_rejects_bad_country():
    props = [_prop(95, 0, 0, 0, is_hq=True)]
    try:
        apply_p0_country_id_seating(
            props, {5: 0, 1: 1}, 99, [], map_id=0, map_name="test"
        )
    except ValueError as e:
        assert "p0_country_id=99" in str(e)
    else:
        raise AssertionError("expected ValueError")
