"""State-mismatch tolerance regression pin (Phase 11J-STATE-MISMATCH-RETUNE-SHIP).

Empirical justification for the tolerance=10 default:

- AWBW ``combatInfo`` records DISPLAY HP only (integer 1-10).
- The engine pins post-combat HP to ``display × 10`` via the existing oracle
  override (``_oracle_combat_damage_override``).
- PHP per-day snapshots use sub-display ``hit_points`` decimals (e.g. ``9.4``
  ≡ internal HP 94).
- Most drift is |Δ|≤9 (sub-display remainder). A smaller set (~12 / 936 GL games
  at tolerance 9) lands at **exactly** |Δ|=10: one full display bucket between
  lossy combatInfo ×10 and snapshot ``round(hit_points×10)`` — not limited to
  Sonja; absorbing |Δ|≤10 clears that quantization class without masking
  larger combat drift (|Δ|≥11 still flags).

If the engine tracks sub-display HP natively, this tolerance can be reduced.

See ``docs/oracle_exception_audit/phase11j_state_mismatch_full_triage.md`` and
``phase11j_state_mismatch_retune_ship.md``.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.unit import Unit, UnitType  # noqa: E402

from tools import desync_audit  # noqa: E402
from tools.desync_audit import _diff_engine_vs_snapshot  # noqa: E402


ROUNDING_NOISE_CEILING = 10  # |Δ| ≤ 10 absorbs sub-display + single bucket step.


def _make_unit(unit_type: UnitType, player: int, pos: tuple[int, int], hp: int) -> Unit:
    return Unit(
        unit_type=unit_type,
        player=player,
        hp=hp,
        ammo=0,
        fuel=99,
        pos=pos,
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
    )


def _make_state(*, units_p0, units_p1):
    return types.SimpleNamespace(
        funds=[9000, 9000],
        units={0: list(units_p0), 1: list(units_p1)},
    )


def _php_frame_one_unit(*, players_id: int, x: int, y: int, hp_decimal: float) -> dict:
    return {
        "day": 3,
        "players": {
            "k100": {"id": 100, "funds": 9000, "order": 0},
            "k200": {"id": 200, "funds": 9000, "order": 1},
        },
        "units": {
            "u0": {
                "x": x,
                "y": y,
                "players_id": players_id,
                "name": "Infantry",
                "hit_points": hp_decimal,
            }
        },
    }


# ---------------------------------------------------------------------------
# Pin 1 — CLI default is 9, NOT 0. Prevents accidental revert.
# ---------------------------------------------------------------------------
def test_cli_default_tolerance_is_10():
    parser = desync_audit._build_arg_parser()
    args = parser.parse_args([])
    assert args.state_mismatch_hp_tolerance == ROUNDING_NOISE_CEILING, (
        "CLI default for --state-mismatch-hp-tolerance changed; review "
        "docs/oracle_exception_audit/phase11j_state_mismatch_retune_ship.md "
        "and tests/test_state_mismatch_tolerance.py before reducing (would "
        "reintroduce single-bucket |Δ|=10 rows to the state_mismatch register)."
    )


def test_cli_zero_override_still_accepted():
    """Operators can still opt back into EXACT comparison for forensic runs."""
    parser = desync_audit._build_arg_parser()
    args = parser.parse_args(["--state-mismatch-hp-tolerance", "0"])
    assert args.state_mismatch_hp_tolerance == 0


# ---------------------------------------------------------------------------
# Pin 2 — Diff function honors the tolerance: |Δ| ≤ 10 absorbed silently;
# |Δ| ≥ 11 surfaces as state_mismatch_units.
# ---------------------------------------------------------------------------
def test_diff_absorbs_sub_display_rounding_noise():
    awbw_to_engine = {100: 0, 200: 1}
    state = _make_state(
        units_p0=[_make_unit(UnitType.INFANTRY, 0, (3, 4), hp=100)],
        units_p1=[],
    )
    frame = _php_frame_one_unit(players_id=100, x=4, y=3, hp_decimal=9.1)
    diff = _diff_engine_vs_snapshot(state, frame, awbw_to_engine, hp_internal_tolerance=10)
    assert diff == {}, (
        f"|Δ|=9 should be absorbed at tolerance=10, but diff surfaced: {diff!r}"
    )


def test_diff_absorbs_single_bucket_quantization_at_10():
    awbw_to_engine = {100: 0, 200: 1}
    state = _make_state(
        units_p0=[_make_unit(UnitType.INFANTRY, 0, (3, 4), hp=100)],
        units_p1=[],
    )
    frame = _php_frame_one_unit(players_id=100, x=4, y=3, hp_decimal=9.0)
    diff = _diff_engine_vs_snapshot(state, frame, awbw_to_engine, hp_internal_tolerance=10)
    assert diff == {}, (
        f"|Δ|=10 should be absorbed at tolerance=10, but diff surfaced: {diff!r}"
    )


def test_diff_surfaces_signal_above_tolerance():
    awbw_to_engine = {100: 0, 200: 1}
    state = _make_state(
        units_p0=[_make_unit(UnitType.INFANTRY, 0, (3, 4), hp=100)],
        units_p1=[],
    )
    frame = _php_frame_one_unit(players_id=100, x=4, y=3, hp_decimal=8.9)
    diff = _diff_engine_vs_snapshot(state, frame, awbw_to_engine, hp_internal_tolerance=10)
    assert "units_hp" in diff.get("axes", []), (
        f"|Δ|=11 must surface at tolerance=10; got diff={diff!r}"
    )
    assert diff["unit_hp_mismatch_count"] == 1


def test_diff_legacy_exact_mode_still_flags_any_delta():
    """tolerance=0 keeps the original strict behavior for forensic reruns."""
    awbw_to_engine = {100: 0, 200: 1}
    state = _make_state(
        units_p0=[_make_unit(UnitType.INFANTRY, 0, (3, 4), hp=95)],
        units_p1=[],
    )
    frame = _php_frame_one_unit(players_id=100, x=4, y=3, hp_decimal=9.4)
    diff = _diff_engine_vs_snapshot(state, frame, awbw_to_engine, hp_internal_tolerance=0)
    assert "units_hp" in diff.get("axes", []), (
        "tolerance=0 must still flag |Δ|=1 (legacy EXACT semantics for forensic mode)"
    )
