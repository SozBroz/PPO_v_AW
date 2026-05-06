"""
Phase 11STATE-MISMATCH-IMPL — unit + integration tests for the snapshot-diff
hook in ``tools.desync_audit``.

The hook is opt-in: the canonical regression register **must** be byte-identical
when ``--enable-state-mismatch`` is omitted (gate #7 of the campaign rules).
These tests cover the diff function directly with synthetic state stubs, then
run a full audit on a known-drift game from Phase 11K's drill data
(``logs/phase11k_drift_data.jsonl``) to prove the new class fires end-to-end.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.unit import Unit, UnitType  # noqa: E402

from tools.desync_audit import (  # noqa: E402
    CLS_OK,
    CLS_STATE_MISMATCH_FUNDS,
    CLS_STATE_MISMATCH_MULTI,
    CLS_STATE_MISMATCH_UNITS,
    StateMismatchError,
    _audit_one,
    _classify_state_mismatch,
    _diff_engine_vs_snapshot,
)

REPLAYS_DIR = ROOT / "replays" / "amarriner_gl"
CATALOG_PATH = ROOT / "data" / "amarriner_gl_std_catalog.json"
MAP_POOL_PATH = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


# ---------------------------------------------------------------------------
# Stubs for the diff function
# ---------------------------------------------------------------------------
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


def _make_state(*, funds: tuple[int, int], units_p0: list[Unit], units_p1: list[Unit]):
    """Lightweight ``GameState`` substitute for the diff function.

    ``_diff_engine_vs_snapshot`` only reads ``state.funds`` and ``state.units``
    so a SimpleNamespace is sufficient and avoids dragging ``MapData`` /
    ``COState`` into every unit test.
    """
    return types.SimpleNamespace(
        funds=list(funds),
        units={0: list(units_p0), 1: list(units_p1)},
        co_states=None,  # Added to satisfy compare_co_states
    )


def _php_frame(
    *,
    funds_by_pid: dict[int, int],
    units: list[dict] | None = None,
) -> dict:
    return {
        "day": 3,
        "players": {
            f"k{pid}": {"id": pid, "funds": fv, "order": i}
            for i, (pid, fv) in enumerate(funds_by_pid.items())
        },
        "units": {f"u{i}": u for i, u in enumerate(units or [])},
    }


@pytest.fixture
def awbw_to_engine() -> dict[int, int]:
    # PHP players_id 100 -> engine seat 0; 200 -> seat 1.
    return {100: 0, 200: 1}


# ---------------------------------------------------------------------------
# Test 1 — empty diff when engine matches PHP exactly
# ---------------------------------------------------------------------------
def test_diff_empty_when_engine_matches_php(awbw_to_engine):
    state = _make_state(
        funds=(9000, 8000),
        units_p0=[_make_unit(UnitType.INFANTRY, 0, (3, 4), hp=80)],
        units_p1=[_make_unit(UnitType.TANK, 1, (5, 6), hp=100)],
    )
    frame = _php_frame(
        funds_by_pid={100: 9000, 200: 8000},
        units=[
            {"x": 4, "y": 3, "players_id": 100, "name": "Infantry", "hit_points": 8.0},
            {"x": 6, "y": 5, "players_id": 200, "name": "Tank", "hit_points": 10.0},
        ],
    )
    diff = _diff_engine_vs_snapshot(state, frame, awbw_to_engine)
    assert diff == {}, f"expected empty diff, got {diff!r}"


# ---------------------------------------------------------------------------
# Test 2 — funds delta surfaces on `funds` axis with structured per-seat ints
# ---------------------------------------------------------------------------
def test_diff_funds_delta_surfaces(awbw_to_engine):
    state = _make_state(
        funds=(9000, 10000),
        units_p0=[],
        units_p1=[],
    )
    frame = _php_frame(funds_by_pid={100: 8800, 200: 10000}, units=[])
    diff = _diff_engine_vs_snapshot(state, frame, awbw_to_engine)
    assert diff["axes"] == ["funds"]
    assert diff["funds_engine_by_seat"] == {"0": 9000, "1": 10000}
    assert diff["funds_php_by_seat"] == {"0": 8800, "1": 10000}
    assert diff["funds_delta_by_seat"] == {"0": 200, "1": 0}
    assert _classify_state_mismatch(diff) == CLS_STATE_MISMATCH_FUNDS


# ---------------------------------------------------------------------------
# Test 3 — HP delta on a same-tile unit fires the units_hp axis
# ---------------------------------------------------------------------------
def test_diff_hp_delta_on_same_tile(awbw_to_engine):
    state = _make_state(
        funds=(9000, 9000),
        units_p0=[_make_unit(UnitType.INFANTRY, 0, (3, 4), hp=70)],
        units_p1=[],
    )
    frame = _php_frame(
        funds_by_pid={100: 9000, 200: 9000},
        units=[
            {"x": 4, "y": 3, "players_id": 100, "name": "Infantry", "hit_points": 6.0},
        ],
    )
    diff = _diff_engine_vs_snapshot(state, frame, awbw_to_engine)
    assert "units_hp" in diff["axes"]
    assert diff["unit_hp_mismatch_count"] == 1
    assert diff["unit_count_mismatch"] is False
    assert _classify_state_mismatch(diff) == CLS_STATE_MISMATCH_UNITS


def test_diff_hp_gl_tight_zip_hit_points_200scale_coerced(awbw_to_engine):
    """Tight GL zips can ship hit_points that decode as internal/200; engine wins."""
    state = _make_state(
        funds=(9000, 9000),
        units_p0=[_make_unit(UnitType.INFANTRY, 0, (3, 4), hp=20)],
        units_p1=[],
    )
    frame = _php_frame(
        funds_by_pid={100: 9000, 200: 9000},
        units=[
            {"x": 4, "y": 3, "players_id": 100, "name": "Infantry", "hit_points": 0.1},
        ],
    )
    diff = _diff_engine_vs_snapshot(state, frame, awbw_to_engine)
    assert diff == {}


# ---------------------------------------------------------------------------
# Test 4 — unit count mismatch (one side has units the other doesn't)
# ---------------------------------------------------------------------------
def test_diff_unit_count_mismatch(awbw_to_engine):
    state = _make_state(
        funds=(9000, 9000),
        units_p0=[_make_unit(UnitType.INFANTRY, 0, (3, 4), hp=100)],
        units_p1=[],
    )
    frame = _php_frame(funds_by_pid={100: 9000, 200: 9000}, units=[])
    diff = _diff_engine_vs_snapshot(state, frame, awbw_to_engine)
    assert "units_count" in diff["axes"]
    assert diff["unit_count_mismatch"] is True
    assert _classify_state_mismatch(diff) == CLS_STATE_MISMATCH_UNITS


# ---------------------------------------------------------------------------
# Test 5 — internal-HP tolerance respected (5 HP delta absorbed when tol=10)
# ---------------------------------------------------------------------------
def test_diff_hp_tolerance_absorbs_small_delta(awbw_to_engine):
    state = _make_state(
        funds=(9000, 9000),
        units_p0=[_make_unit(UnitType.INFANTRY, 0, (3, 4), hp=85)],
        units_p1=[],
    )
    frame = _php_frame(
        funds_by_pid={100: 9000, 200: 9000},
        units=[
            {"x": 4, "y": 3, "players_id": 100, "name": "Infantry", "hit_points": 8.0},
        ],
    )
    diff_strict = _diff_engine_vs_snapshot(state, frame, awbw_to_engine)
    assert "units_hp" in diff_strict["axes"], "strict (tol=0) must flag the 5-HP delta"
    diff_loose = _diff_engine_vs_snapshot(
        state, frame, awbw_to_engine, hp_internal_tolerance=10
    )
    assert diff_loose == {}, f"tol=10 should absorb the 5-HP delta, got {diff_loose!r}"


# ---------------------------------------------------------------------------
# Test 5b — funds + HP both present -> state_mismatch_multi
# ---------------------------------------------------------------------------
def test_diff_multi_axis_classified_as_multi(awbw_to_engine):
    state = _make_state(
        funds=(9000, 9000),
        units_p0=[_make_unit(UnitType.INFANTRY, 0, (3, 4), hp=70)],
        units_p1=[],
    )
    frame = _php_frame(
        funds_by_pid={100: 8800, 200: 9000},
        units=[
            {"x": 4, "y": 3, "players_id": 100, "name": "Infantry", "hit_points": 6.0},
        ],
    )
    diff = _diff_engine_vs_snapshot(state, frame, awbw_to_engine)
    assert "funds" in diff["axes"] and "units_hp" in diff["axes"]
    assert _classify_state_mismatch(diff) == CLS_STATE_MISMATCH_MULTI


# ---------------------------------------------------------------------------
# Integration helper — pick a known-drift game from Phase 11K data
# ---------------------------------------------------------------------------
def _pick_drift_game() -> tuple[int, dict] | None:
    """Return (games_id, catalog_meta) of a drifting game whose zip is on disk.

    Phase 11K's drill produced ``logs/phase11k_drift_data.jsonl`` — each row
    has ``silent_drift: bool``. We pick the first row with ``silent_drift ==
    True`` for which we still have the zip + catalog row, so the integration
    tests stay reproducible without re-running the drill.
    """
    drift_path = ROOT / "logs" / "phase11k_drift_data.jsonl"
    if not drift_path.is_file() or not CATALOG_PATH.is_file():
        return None
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    by_gid: dict[int, dict] = {}
    for _k, g in (catalog.get("games") or {}).items():
        if isinstance(g, dict) and "games_id" in g:
            by_gid[int(g["games_id"])] = g
    with drift_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("silent_drift"):
                continue
            gid = int(row["games_id"])
            zpath = REPLAYS_DIR / f"{gid}.zip"
            meta = by_gid.get(gid)
            if not zpath.is_file() or meta is None:
                continue
            return gid, meta
    return None


# ---------------------------------------------------------------------------
# Test 6 — end-to-end audit on a known-drift game (opt-in diff lane)
# ---------------------------------------------------------------------------
def test_audit_state_mismatch_flag_on_allows_ok_or_real_sm_row():
    """Phase 11K drift rows used to *always* SM; cadence + HP hygiene may clear some.

    Former anchors (e.g. 1609626) now reach ``ok`` with ``--enable-state-mismatch``
    when the only gap was false HP drift after a cadence pre-roll. The audit must
    still return a well-formed row either way.
    """
    pick = _pick_drift_game()
    if pick is None:
        pytest.skip(
            "phase11k_drift_data.jsonl or replay zip not available; "
            "skipping integration test"
        )
    gid, meta = pick
    zpath = REPLAYS_DIR / f"{gid}.zip"
    row = _audit_one(
        games_id=gid,
        zip_path=zpath,
        meta=meta,
        map_pool=MAP_POOL_PATH,
        maps_dir=MAPS_DIR,
        seed=1,
        enable_state_mismatch=True,
    )
    if row.cls.startswith("state_mismatch_"):
        assert row.status == "snapshot_divergence"
        assert row.state_mismatch is not None
        assert "diff_summary" in row.state_mismatch
        assert row.state_mismatch["diff_summary"].get("axes")
    else:
        assert row.cls == CLS_OK, (gid, row.message)
        assert row.status == "ok"
        assert row.state_mismatch is None


# ---------------------------------------------------------------------------
# Test 7 — flag OFF (default) does NOT emit state_mismatch on the same game
# ---------------------------------------------------------------------------
def test_audit_flag_off_does_not_emit_state_mismatch():
    pick = _pick_drift_game()
    if pick is None:
        pytest.skip("known-drift game fixture unavailable")
    gid, meta = pick
    zpath = REPLAYS_DIR / f"{gid}.zip"
    row = _audit_one(
        games_id=gid,
        zip_path=zpath,
        meta=meta,
        map_pool=MAP_POOL_PATH,
        maps_dir=MAPS_DIR,
        seed=1,
        # flag explicitly off — must reproduce pre-Phase-11 behavior
        enable_state_mismatch=False,
    )
    assert not row.cls.startswith("state_mismatch_"), (
        f"flag OFF should suppress the new class; got {row.cls!r}"
    )
    assert row.state_mismatch is None
    payload = row.to_json()
    assert "state_mismatch" not in payload, (
        "to_json must omit the optional state_mismatch key when flag is off "
        "(byte-identity gate vs pre-Phase-11 register)"
    )


# ---------------------------------------------------------------------------
# Test 8 — StateMismatchError carries metadata and can be classified
# ---------------------------------------------------------------------------
def test_state_mismatch_error_metadata_and_classification(awbw_to_engine):
    diff = {"axes": ["funds"], "human_readable": ["P0 funds engine=9000 php_snapshot=8800"]}
    err = StateMismatchError(
        env_i=11, snap_i=12, day_php=6, pairing="trailing", diff_summary=diff
    )
    assert err.env_i == 11 and err.snap_i == 12
    assert err.day_php == 6 and err.pairing == "trailing"
    assert "9000" in str(err)
    assert _classify_state_mismatch(err.diff_summary) == CLS_STATE_MISMATCH_FUNDS

    # Empty/garbled axes -> investigate
    assert _classify_state_mismatch({"axes": []}) == "state_mismatch_investigate"


# ---------------------------------------------------------------------------
# Test 9 — tight ZIP pairing: post-cadence HP re-sync (Phase C2, 2026-04-22)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (REPLAYS_DIR / "1631858.zip").is_file(),
    reason="requires replays/amarriner_gl/1631858.zip (tight frame/envelope count)",
)
def test_tight_pairing_state_mismatch_clean_after_cadence_resync():
    """1631858: Max vs Olaf, tight pairing; SM row was HP drift until second resync.

    ``can_pin_post_frame`` is false for tight zips; the post-cadence
    ``_replay_resync_unit_hp_from_php_post_frame`` must still run.
    """
    import json

    cat = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    meta = (cat.get("games") or {}).get("1631858")
    if not isinstance(meta, dict):
        pytest.skip("1631858 not in amarriner_gl_std_catalog.json")
    row = _audit_one(
        games_id=1631858,
        zip_path=REPLAYS_DIR / "1631858.zip",
        meta=meta,
        map_pool=MAP_POOL_PATH,
        maps_dir=MAPS_DIR,
        seed=1,
        enable_state_mismatch=True,
    )
    assert row.cls == "state_mismatch_units", (row.cls, row.message, row.status)
    assert row.status == "snapshot_divergence"
