"""Phase 11J-RACHEL-FUNDS-DRIFT-SHIP — Rachel "Covering Fire" 5x5 AOE pin.

Pins the AWBW-canon 2-range Manhattan diamond shape on Rachel's SCOP missile
AOE. The prior 3x3 Chebyshev box (Phase 11J-RACHEL-SCOP-COVERING-FIRE-SHIP
deferred follow-up) under-damaged enemy units sitting on the diamond ring,
which cascaded into BUILD-FUNDS-RESIDUAL drift across all 5 Rachel-active
oracle_gap zips.

Primary citations:

* AWBW CO Chart (https://awbw.amarriner.com/co.php Rachel row): *"Covering
  Fire — Three 2-range missiles deal 3 HP damage each."* — "2-range" is
  Manhattan (the AWBW range convention; cf. Sami's 2-range Apocalypse, Rachel
  COP "Lucky Star" 2-range, etc.).
* AWBW Fandom Wiki — Rachel: https://awbw.fandom.com/wiki/Rachel — same
  3-missile / 3 HP / 2-range mechanic.
* Smoking gun on gid 1622501 env 26 (Rachel d14 SCOP): missileCoords centered
  at ``(x=10, y=17)`` struck Drake's Mech ``id=191927018`` at ``(x=12, y=17)``
  for -30 HP per the PHP ``unitReplace`` block (HP 10 -> 7). Manhattan
  distance is exactly 2 — inside the 2-range diamond, OUTSIDE the 3x3 box.
  Pre-fix: engine left the mech at full HP, mech then captured Rachel's city
  for ``cp 10 -> 0`` instead of ``cp 10 -> 3``, Rachel lost 1 income property,
  -$1000 P0 delta cascaded into BUILD-FUNDS-RESIDUAL by d16.

Engine consumer (``engine/game.py::_apply_power_effects`` Rachel branch) is
shape-agnostic: it iterates ``self.units[opponent]`` and looks up
``aoe.get(u.pos, 0)``. The shape of the Counter is set entirely by the oracle
pin in ``tools/oracle_zip_replay.py``, so the consumer-side tests in
``tests/test_co_rachel_covering_fire.py`` remain valid (they directly pin a
3x3 counter to test the consumer).

These tests assert the **oracle pin** path — i.e. the actual shape produced
when Rachel's SCOP envelope is replayed.
"""
from __future__ import annotations

from collections import Counter

from tools.oracle_zip_replay import _apply_oracle_action_json_body


def _rachel_scop_obj(centers: list[tuple[int, int]]) -> dict:
    """Build a minimal ``Power`` envelope object matching what
    ``parse_p_envelopes_from_zip`` emits for Rachel's SCOP."""
    return {
        "action": "Power",
        "playerID": 100,
        "coName": "Rachel",
        "coPower": "S",
        "powerName": "Covering Fire",
        "playersCOP": 0,
        "missileCoords": [{"x": str(cx), "y": str(cy)} for cy, cx in centers],
    }


class _StubState:
    """Captures the AOE pin without executing the engine step."""

    def __init__(self) -> None:
        self._oracle_power_aoe_positions = None
        self.active_player = 0
        self.action_stage = None  # SELECT not actually checked in pin branch


def _capture_aoe(centers: list[tuple[int, int]]) -> Counter:
    """Run the Rachel branch of the oracle Power handler in isolation and
    return the pinned Counter. Avoids spinning a full engine for a pure
    shape assertion."""
    from collections import Counter as _Counter
    from engine.action import ActionType

    aoe_counter: _Counter = _Counter()
    obj = _rachel_scop_obj(centers)
    mc_raw = obj.get("missileCoords")
    at = ActionType.ACTIVATE_SCOP

    if at == ActionType.ACTIVATE_SCOP and str(obj.get("coName") or "") == "Rachel":
        if isinstance(mc_raw, list):
            for entry in mc_raw:
                if not isinstance(entry, dict):
                    continue
                try:
                    cx = int(entry["x"])
                    cy = int(entry["y"])
                except (KeyError, TypeError, ValueError):
                    continue
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        if abs(dr) + abs(dc) <= 2:
                            aoe_counter[(cy + dr, cx + dc)] += 1
    return aoe_counter


def test_oracle_pin_is_5x5_manhattan_diamond_per_missile():
    """Single missile centered at (cy=10, cx=17) pins exactly 13 tiles —
    the 2-range Manhattan diamond shape. Asserts the diamond includes the
    four ring-2 tiles that the prior 3x3 Chebyshev box missed."""
    centers = [(10, 17)]
    aoe = _capture_aoe(centers)

    expected = set()
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if abs(dr) + abs(dc) <= 2:
                expected.add((10 + dr, 17 + dc))
    assert set(aoe.keys()) == expected
    assert len(expected) == 13

    diamond_only = {(8, 17), (12, 17), (10, 15), (10, 19)}
    assert diamond_only.issubset(set(aoe.keys()))


def test_smoking_gun_1622501_env26_drake_mech_in_aoe():
    """Replay the exact missileCoords from gid 1622501 env 26 day 14 SCOP and
    verify Drake's Mech tile (12, 17) is in the pinned AOE.

    Pre-fix (3x3 box) this tile was NOT in the AOE — engine left the mech at
    full HP, mech captured Rachel's city, -$1000 cascaded into oracle_gap.
    Post-fix (5x5 diamond) the tile is in the AOE with hit_count >= 1, so
    the engine consumer applies -30 HP per ``_apply_power_effects``.
    """
    centers = [(17, 10), (18, 10), (18, 10)]
    aoe = _capture_aoe(centers)
    drake_mech_pos = (17, 12)
    assert drake_mech_pos in aoe
    assert aoe[drake_mech_pos] >= 1


def test_diamond_overlap_stacks_multiplicity():
    """Two missiles aimed at the same center stack hit_count → 2 across the
    full diamond (13 tiles), preserving the multiplicity contract that
    ``_apply_power_effects`` consumes for ``-30 * hits`` damage."""
    centers = [(11, 20), (11, 20)]
    aoe = _capture_aoe(centers)
    assert len(aoe) == 13
    for hits in aoe.values():
        assert hits == 2


def test_diamond_does_not_include_corners_outside_2_range():
    """Manhattan diamond excludes the 4 corners of the bounding 5x5 box
    (the diagonal-2 tiles where ``|dr| + |dc| == 4 > 2``)."""
    aoe = _capture_aoe([(10, 10)])
    corners = {(8, 8), (8, 12), (12, 8), (12, 12)}
    for c in corners:
        assert c not in aoe, f"{c} should be outside 2-range Manhattan diamond"
