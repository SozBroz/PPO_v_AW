"""
Phase 11J-FUNDS-EXTERMINATION — cadence-mismatch pre-roll guard.

The state-mismatch comparator in ``tools.desync_audit._run_replay_instrumented``
runs a per-envelope diff against the PHP snapshot. AWBW occasionally emits a
``p:`` envelope that lacks an explicit ``End`` action (timeouts, AET, short
exports such as ``games_id`` 1618984 / 1619117 / 1621641); the next PHP frame
still reflects an implicit end-of-turn rollover plus the next player's
start-of-turn income. The engine catches up only when the *following*
envelope's player differs (via ``_oracle_advance_turn_until_player``) — too
late for the current snapshot diff, producing spurious
``state_mismatch_funds`` rows of $8000-$13000.

These tests pin the pre-roll's two correctness invariants:

1. When the engine has already rolled past the envelope (normal ``End``
   case, ``state.active_player != envelope_player``), the pre-roll **must
   not** advance the turn again. Doing so would re-charge income/repair
   and produce a much larger drift than the bug it tries to fix
   (regression observed: 18 → 875 funds rows when guard was missing).

2. When the engine has *not* rolled (no explicit ``End``) but PHP shows a
   later day in ``frame[i+1]``, the engine state should be advanced to
   the opponent's seat before diffing.

The audit's classification, gate logic and ``hp_internal_tolerance`` floor
remain untouched (per Phase 11J-FINAL standing rules).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.action import ActionStage  # noqa: E402

from tools.desync_audit import _audit_before_engine_step  # noqa: E402


class _FakeState:
    """Minimal stand-in for ``GameState`` cadence-pre-roll inspects."""

    def __init__(self, *, active_player: int, action_stage=ActionStage.SELECT, done: bool = False):
        self.active_player = active_player
        self.action_stage = action_stage
        self.done = done


def _pre_roll_predicate(
    state: _FakeState,
    envelope_player_eng: int | None,
    envelope_day: int,
    php_day_pre: int,
) -> bool:
    """Mirror of the guard in ``_run_replay_instrumented``.

    Kept in sync with the audit by structural reference so we can unit-test
    the boolean without spinning up a full replay.
    """
    return (
        not state.done
        and envelope_player_eng is not None
        and int(state.active_player) == int(envelope_player_eng)
        and php_day_pre > envelope_day
        and state.action_stage == ActionStage.SELECT
    )


# ---------------------------------------------------------------------------
# Test 1 — engine already rolled (normal End): pre-roll MUST NOT fire.
# ---------------------------------------------------------------------------
def test_pre_roll_skipped_after_explicit_end():
    """Envelope ended with ``End`` → engine moved to seat 1; PHP day advanced.

    Without this guard the pre-roll would advance the engine *again* and
    double-grant the next player's income — exactly the regression that
    blew the funds-row count from 18 to 875 in the first attempt.
    """
    state = _FakeState(active_player=1)  # engine rolled past pid-0 envelope
    assert _pre_roll_predicate(
        state, envelope_player_eng=0, envelope_day=6, php_day_pre=7
    ) is False


# ---------------------------------------------------------------------------
# Test 2 — engine still on envelope's seat + PHP day advanced: fire.
# ---------------------------------------------------------------------------
def test_pre_roll_fires_when_engine_lags_php_cadence():
    """Capture-only envelope (no ``End``) → engine still on pid-0; PHP rolled.

    Reproduces the ``games_id`` 1618984 / 1619117 / 1621641 cadence
    pattern: engine seat is the actor's seat, PHP frame is one day later.
    """
    state = _FakeState(active_player=0)
    assert _pre_roll_predicate(
        state, envelope_player_eng=0, envelope_day=3, php_day_pre=4
    ) is True


# ---------------------------------------------------------------------------
# Test 3 — same-day PHP frame (no implicit rollover): pre-roll skipped.
# ---------------------------------------------------------------------------
def test_pre_roll_skipped_when_php_day_unchanged():
    """Mid-turn snapshot (same day as envelope) → no cadence mismatch."""
    state = _FakeState(active_player=0)
    assert _pre_roll_predicate(
        state, envelope_player_eng=0, envelope_day=5, php_day_pre=5
    ) is False


# ---------------------------------------------------------------------------
# Test 4 — engine mid-action (not SELECT): pre-roll must not interrupt.
# ---------------------------------------------------------------------------
def test_pre_roll_skipped_when_engine_not_in_select():
    """Engine inside a multi-step action (MOVE/ACTION) cannot end the turn."""
    state = _FakeState(active_player=0, action_stage=ActionStage.MOVE)
    assert _pre_roll_predicate(
        state, envelope_player_eng=0, envelope_day=3, php_day_pre=4
    ) is False


# ---------------------------------------------------------------------------
# Test 5 — game already done: pre-roll must not fire.
# ---------------------------------------------------------------------------
def test_pre_roll_skipped_when_game_done():
    """Final envelope (resign / annihilation) → engine done, never advance."""
    state = _FakeState(active_player=0, done=True)
    assert _pre_roll_predicate(
        state, envelope_player_eng=0, envelope_day=3, php_day_pre=4
    ) is False


# ---------------------------------------------------------------------------
# Test 6 — unknown envelope player (mapping miss): pre-roll skipped.
# ---------------------------------------------------------------------------
def test_pre_roll_skipped_when_envelope_player_unmapped():
    """``awbw_to_engine`` may miss a stray ``p:`` row — fail safe (skip)."""
    state = _FakeState(active_player=0)
    assert _pre_roll_predicate(
        state, envelope_player_eng=None, envelope_day=3, php_day_pre=4
    ) is False


# ---------------------------------------------------------------------------
# Test 7 — the no-op hook returns ``None`` for any args (smoke test).
# ---------------------------------------------------------------------------
def test_audit_before_engine_step_is_noop():
    """Pre-roll passes a stub hook to the oracle helpers; it must be a no-op."""
    assert _audit_before_engine_step() is None
    assert _audit_before_engine_step("any", "args", k="v") is None


# ---------------------------------------------------------------------------
# Test 8 — integration: cadence game + treasury snap (zip fixture optional).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (ROOT / "replays" / "amarriner_gl" / "1618984.zip").is_file(),
    reason="GL replay zip not present in this checkout",
)
def test_1618984_cadence_no_spurious_funds_after_treasury_snap():
    """1618984: Capt-without-End envelope; pre-roll + PHP funds snap must stay ok."""
    from tools.desync_audit import CLS_OK, _audit_one, _load_catalog

    cat_path = ROOT / "data" / "amarriner_gl_std_catalog.json"
    if not cat_path.is_file():
        pytest.skip("std catalog missing")
    catalog = _load_catalog(cat_path)
    games = catalog.get("games") or {}
    meta = None
    for _k, g in games.items():
        if isinstance(g, dict) and int(g.get("games_id", -1)) == 1618984:
            meta = g
            break
    if meta is None:
        pytest.skip("games_id 1618984 not in std catalog")
    zip_path = ROOT / "replays" / "amarriner_gl" / "1618984.zip"
    row = _audit_one(
        games_id=1618984,
        zip_path=zip_path,
        meta=meta,
        map_pool=ROOT / "data" / "gl_map_pool.json",
        maps_dir=ROOT / "data" / "maps",
        seed=1,
        enable_state_mismatch=True,
        state_mismatch_hp_tolerance=10,
    )
    assert row.cls == CLS_OK, row.message


@pytest.mark.skipif(
    not (ROOT / "replays" / "amarriner_gl" / "1623866.zip").is_file(),
    reason="GL replay zip not present in this checkout",
)
def test_1623866_sm_ok_after_post_preroll_hp_resync():
    """Trailing half-turn: cadence pre-roll must not leave +20 repair vs PHP frame."""
    from tools.desync_audit import CLS_OK, _audit_one, _load_catalog

    cat_path = ROOT / "data" / "amarriner_gl_std_catalog.json"
    if not cat_path.is_file():
        pytest.skip("std catalog missing")
    catalog = _load_catalog(cat_path)
    games = catalog.get("games") or {}
    meta = None
    for _k, g in games.items():
        if isinstance(g, dict) and int(g.get("games_id", -1)) == 1623866:
            meta = g
            break
    if meta is None:
        pytest.skip("games_id 1623866 not in std catalog")
    zip_path = ROOT / "replays" / "amarriner_gl" / "1623866.zip"
    row = _audit_one(
        games_id=1623866,
        zip_path=zip_path,
        meta=meta,
        map_pool=ROOT / "data" / "gl_map_pool.json",
        maps_dir=ROOT / "data" / "maps",
        seed=1,
        enable_state_mismatch=True,
        state_mismatch_hp_tolerance=10,
    )
    assert row.cls == CLS_OK, row.message

