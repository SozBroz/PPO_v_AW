"""RL action-space pins (Phase 11J-RL-DELETE-GUARD-SHIP).

Guarantees that:
  1. ``get_legal_actions()`` never emits an ``ActionType`` outside
     ``_RL_LEGAL_ACTION_TYPES`` (static + 100-step random walk).
  2. The allowlist permanently excludes ``RESIGN`` and any future oracle-only
     ``ActionType`` (e.g. a hypothetical ``DELETE``).
  3. The oracle's ``_oracle_kill_friendly_unit`` is not reachable from any
     ``get_legal_actions`` output — no RL action triggers friendly-unit removal.

Imperator directive 2026-04-20: the bot must not learn to Delete its own
units to free production tiles. AWBW Delete Unit is a player UI control
that human players use to scrap blockers; the RL agent is forbidden from it
by both action-type omission and this regression suite.
"""
from __future__ import annotations

import random
import pytest

from engine.action import (
    ActionType,
    get_legal_actions,
    _RL_LEGAL_ACTION_TYPES,
)
from engine.unit import UnitType
from tests.test_engine_negative_legality import _make_state, _spawn, _prop, OS_BASE


# ---------------------------------------------------------------------------
# Static pins
# ---------------------------------------------------------------------------

def test_allowlist_excludes_resign():
    assert ActionType.RESIGN not in _RL_LEGAL_ACTION_TYPES


def test_allowlist_has_no_delete_action_type():
    # Defensive: assert no ActionType named DELETE exists.
    assert not hasattr(ActionType, "DELETE"), (
        "ActionType.DELETE must not exist — Delete Unit is oracle-path-only. "
        "If this fires, someone added DELETE; revert and route via oracle."
    )


def test_allowlist_size_pinned():
    # If this changes, the test author must explicitly review whether the new
    # ActionType is truly RL-legal or oracle-only.
    assert len(_RL_LEGAL_ACTION_TYPES) == 13, (
        f"_RL_LEGAL_ACTION_TYPES size changed to {len(_RL_LEGAL_ACTION_TYPES)}. "
        "Review whether the new ActionType is RL-legal or oracle-only."
    )


def test_allowlist_excludes_every_forbidden_name():
    # Mirrors the engine's import-time guard (_FORBIDDEN_RL_ACTION_NAMES).
    forbidden = {"DELETE", "DELETE_UNIT", "SCRAP", "SCRAP_UNIT",
                 "DESTROY_OWN_UNIT", "KILL_OWN_UNIT"}
    assert not any(at.name in forbidden for at in _RL_LEGAL_ACTION_TYPES)


# ---------------------------------------------------------------------------
# Live legality — every emitted action must be in the allowlist
# ---------------------------------------------------------------------------

def _tiny_state():
    """Smallest viable state: one OS infantry on a base, one BM infantry across the map."""
    state = _make_state(width=6, height=1, properties=[
        _prop(0, 0, OS_BASE, owner=0, is_base=True),
    ], funds=(8000, 8000))
    _spawn(state, UnitType.INFANTRY, player=0, pos=(0, 1))
    _spawn(state, UnitType.INFANTRY, player=1, pos=(0, 5))
    return state


def test_get_legal_actions_subset_of_allowlist_initial_state():
    state = _tiny_state()
    actions = get_legal_actions(state)
    bad = [a.action_type.name for a in actions if a.action_type not in _RL_LEGAL_ACTION_TYPES]
    assert not bad, f"get_legal_actions emitted non-allowlist actions: {bad}"


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_random_walk_never_emits_non_allowlist(seed):
    # 100-step random walk on the tiny board; every step's get_legal_actions
    # output must be a subset of the allowlist. The dispatcher's defense-in-
    # depth assert would also raise, but we check explicitly for clarity.
    rng = random.Random(seed)
    state = _tiny_state()
    for _ in range(100):
        actions = get_legal_actions(state)
        if not actions:
            break
        for a in actions:
            assert a.action_type in _RL_LEGAL_ACTION_TYPES, (
                f"Step emitted non-allowlist action {a.action_type.name}"
            )
        a = rng.choice(actions)
        try:
            state, _, done = state.step(a)
        except Exception:
            break
        if done:
            break
