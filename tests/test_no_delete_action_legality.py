"""Phase 11J-DELETE-GUARD-PIN — RL action-space safeguard regression tests.

Strategic concern
-----------------
AWBW players can issue a "Delete unit" action to scrap their own unit (no
funds refund). The replay oracle (``tools/oracle_zip_replay.py
::_oracle_kill_friendly_unit``) reproduces that envelope so AWBW zip
replays can be reconstructed faithfully (Phase 11J-L2-BUILD-OCCUPIED-SHIP).

The RL bot must NEVER be able to emit this action. Allowing it would unlock
a degenerate scrap-and-rebuild policy: scrap a low-value blocker on a
production tile -> spawn a stronger replacement -> repeat. That loop lets
the policy print arbitrary value out of the production system without the
opportunity cost AWBW intends.

What this test file pins
------------------------
1. ``engine.action.ActionType`` contains no DELETE-shaped member.
2. ``get_legal_actions(state)`` never returns one across many random states.
3. ``GameState.step`` outside ``oracle_mode`` rejects a synthetic Delete
   action via the STEP-GATE.
4. No file in ``engine/`` imports the oracle delete helper or its module.
5. The helper itself remains in ``tools/oracle_zip_replay.py`` (oracle path
   only) and is not exported from ``engine/__init__.py``.

Companion guard: an import-time assertion in ``engine/action.py``
(``Phase 11J-DELETE-GUARD-PIN``) refuses to load if any forbidden RL
action name is added to the enum. These pytest tests are the runtime
backstop for that contract.
"""
from __future__ import annotations

import ast
import importlib
import random
from enum import IntEnum
from pathlib import Path
from typing import Optional

import pytest

from engine.action import (
    Action,
    ActionStage,
    ActionType,
    _FORBIDDEN_RL_ACTION_NAMES,
    get_legal_actions,
)
from engine.co import make_co_state_safe
from engine.game import GameState, IllegalActionError
from engine.map_loader import MapData, PropertyState
from engine.unit import Unit, UnitType, UNIT_STATS


_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Local state builders (inlined to keep this test self-contained — same
# pattern as tests/test_engine_negative_legality.py::_make_state / _spawn).
# ---------------------------------------------------------------------------
PLAIN = 1
OS_BASE = 39
BM_BASE = 44
NEUTRAL_BASE = 35


def _make_state(
    *,
    width: int = 6,
    height: int = 4,
    terrain: Optional[list[list[int]]] = None,
    properties: Optional[list[PropertyState]] = None,
    units: Optional[dict[int, list[Unit]]] = None,
    funds: tuple[int, int] = (0, 0),
    p0_co: int = 1,
    p1_co: int = 1,
    active_player: int = 0,
    action_stage: ActionStage = ActionStage.SELECT,
) -> GameState:
    if terrain is None:
        terrain = [[PLAIN] * width for _ in range(height)]
    else:
        height = len(terrain)
        width = len(terrain[0])
    if properties is None:
        properties = []
    if units is None:
        units = {0: [], 1: []}
    md = MapData(
        map_id=999_777,
        name="delete_guard_pin_probe",
        map_type="std",
        terrain=[row[:] for row in terrain],
        height=height,
        width=width,
        cap_limit=999,
        unit_limit=999,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=properties,
        hq_positions={0: [], 1: []},
        lab_positions={0: [], 1: []},
        country_to_player={},
        predeployed_specs=[],
    )
    state = GameState(
        map_data=md,
        units=units,
        funds=[funds[0], funds[1]],
        co_states=[make_co_state_safe(p0_co), make_co_state_safe(p1_co)],
        properties=properties,
        turn=1,
        active_player=active_player,
        action_stage=action_stage,
        selected_unit=None,
        selected_move_pos=None,
        done=False,
        winner=None,
        win_reason=None,
        game_log=[],
        tier_name="T2",
        full_trace=[],
        seam_hp={},
    )
    return state


_NEXT_UID = [9000]


def _spawn(
    state: GameState,
    ut: UnitType,
    player: int,
    pos: tuple[int, int],
    *,
    moved: bool = False,
) -> Unit:
    stats = UNIT_STATS[ut]
    _NEXT_UID[0] += 1
    u = Unit(
        unit_type=ut,
        player=player,
        hp=100,
        ammo=stats.max_ammo,
        fuel=stats.max_fuel,
        pos=pos,
        moved=moved,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
        unit_id=_NEXT_UID[0],
    )
    state.units[player].append(u)
    return u


def _prop(row: int, col: int, terrain_id: int, owner: Optional[int],
          *, is_base: bool = False) -> PropertyState:
    return PropertyState(
        terrain_id=terrain_id, row=row, col=col, owner=owner, capture_points=20,
        is_hq=False, is_lab=False, is_comm_tower=False,
        is_base=is_base, is_airport=False, is_port=False,
    )


# Pool of unit types varied enough to exercise different mask branches
# (infantry capture, indirect, transport, naval, air).
_UT_POOL = [
    UnitType.INFANTRY,
    UnitType.MECH,
    UnitType.RECON,
    UnitType.TANK,
    UnitType.ARTILLERY,
    UnitType.ROCKET,
    UnitType.APC,
    UnitType.B_COPTER,
    UnitType.FIGHTER,
]


def _random_state(rng: random.Random) -> GameState:
    """Build a small, varied GameState for property-style mask probing."""
    h = rng.randint(3, 6)
    w = rng.randint(3, 6)
    # Sprinkle a few owned bases so SELECT can produce BUILD branches.
    terrain = [[PLAIN] * w for _ in range(h)]
    props: list[PropertyState] = []
    if rng.random() < 0.6:
        # Plant 1-2 owned bases for player 0 / player 1.
        for _ in range(rng.randint(1, 2)):
            r = rng.randint(0, h - 1)
            c = rng.randint(0, w - 1)
            terrain[r][c] = OS_BASE
            props.append(_prop(r, c, OS_BASE, owner=0, is_base=True))
        for _ in range(rng.randint(0, 1)):
            r = rng.randint(0, h - 1)
            c = rng.randint(0, w - 1)
            terrain[r][c] = BM_BASE
            props.append(_prop(r, c, BM_BASE, owner=1, is_base=True))
    funds = (rng.randint(0, 30000), rng.randint(0, 30000))
    state = _make_state(
        terrain=terrain,
        properties=props,
        funds=funds,
        active_player=rng.randint(0, 1),
    )
    # Spawn 0-4 units per player at unique tiles.
    used: set[tuple[int, int]] = set()
    for player in (0, 1):
        for _ in range(rng.randint(0, 4)):
            for _attempt in range(20):
                r = rng.randint(0, h - 1)
                c = rng.randint(0, w - 1)
                if (r, c) in used:
                    continue
                used.add((r, c))
                _spawn(
                    state,
                    rng.choice(_UT_POOL),
                    player,
                    (r, c),
                    moved=rng.random() < 0.3,
                )
                break
    return state


# ---------------------------------------------------------------------------
# Test 1 — enum membership
# ---------------------------------------------------------------------------

def test_action_type_enum_has_no_delete_member():
    """Phase 11J-DELETE-GUARD-PIN: the RL action enum must contain no
    DELETE-shaped member. Delete is oracle-only (replay reproduction); giving
    the bot the ability to scrap its own units enables a degenerate
    scrap-and-rebuild loop. Mirrors the import-time assertion in
    ``engine/action.py`` so the contract is enforced at both load time and
    in CI.
    """
    members = {m.name for m in ActionType}
    collision = _FORBIDDEN_RL_ACTION_NAMES & members
    assert not collision, (
        f"ActionType contains forbidden RL action(s): {collision}. "
        f"See Phase 11J-DELETE-GUARD-PIN."
    )


# ---------------------------------------------------------------------------
# Test 2 — property-style sweep over random states
# ---------------------------------------------------------------------------

def test_get_legal_actions_never_returns_delete_across_random_states():
    """Phase 11J-DELETE-GUARD-PIN: ``get_legal_actions`` must never surface a
    Delete-shaped action regardless of state shape. We sweep 200 random
    GameStates (varied map size, unit composition, owned bases, funds, active
    player) and assert the returned mask carries no member whose
    ``action_type.name`` falls in the forbidden RL set. This is the runtime
    backstop for the strategic concern (RL degenerate scrap-and-rebuild).
    """
    rng = random.Random(0xDE1E7E)
    sweeps = 200
    for i in range(sweeps):
        state = _random_state(rng)
        try:
            actions = get_legal_actions(state)
        except Exception as e:  # defensive: random shape may surprise the engine
            pytest.fail(
                f"get_legal_actions raised on random state #{i}: {type(e).__name__}: {e}"
            )
        for a in actions:
            assert a.action_type.name not in _FORBIDDEN_RL_ACTION_NAMES, (
                f"sweep #{i} produced forbidden action {a.action_type.name!r}; "
                f"Phase 11J-DELETE-GUARD-PIN violated."
            )


# ---------------------------------------------------------------------------
# Test 3 — STEP-GATE rejects a synthetic Delete-shaped action
# ---------------------------------------------------------------------------

def test_step_rejects_synthetic_delete_action_via_step_gate():
    """Phase 11J-DELETE-GUARD-PIN: even if a caller hand-crafts an Action
    whose ``action_type`` falls outside the enum (sentinel value 999), the
    STEP-GATE in ``GameState.step`` (oracle_mode=False) must refuse it
    because the synthetic action cannot appear in
    ``get_legal_actions(state)``. This is the second line of defence
    behind the enum contract — even an attacker who adds a member would
    still need to pipe it through the legal-action generator. Confirms
    the gate raises ``IllegalActionError`` (subclass of ``ValueError``).
    """
    # A separate IntEnum carrying value 999 so it is interchangeable with int
    # at compare sites (IntEnum comparison uses the int value) but still
    # exposes ``.name`` for the STEP-GATE's error-formatting path. This is
    # exactly the shape a future "DELETE = 999" addition would have.
    class _RogueActionType(IntEnum):
        DELETE = 999

    state = _make_state()
    _spawn(state, UnitType.INFANTRY, player=0, pos=(0, 0))
    bogus = Action(action_type=_RogueActionType.DELETE, unit_pos=(0, 0))  # type: ignore[arg-type]
    with pytest.raises((IllegalActionError, ValueError)):
        state.step(bogus)


# ---------------------------------------------------------------------------
# Test 4 — engine/ never imports the oracle delete helper
# ---------------------------------------------------------------------------

_ENGINE_DIR = _REPO_ROOT / "engine"

# Module names whose import from engine/ would (re)expose the oracle
# delete helper to the RL stack.
_FORBIDDEN_IMPORT_MODULES = ("tools.oracle_zip_replay", "oracle_zip_replay")
# Identifiers that would only show up in engine code if someone wired the
# delete helper into the engine. Comments/docstrings/string literals are
# intentionally excluded — only AST identifier references count, so this
# test does not trip on the Phase 11J-DELETE-GUARD-PIN guard banner that
# necessarily *names* the helper in prose.
_FORBIDDEN_IDENTIFIERS = ("_oracle_kill_friendly_unit", "kill_friendly_unit")


def test_oracle_kill_friendly_helper_is_not_imported_by_engine():
    """Phase 11J-DELETE-GUARD-PIN: ``engine/`` must not import or call the
    oracle delete helper. The helper lives in ``tools/oracle_zip_replay.py``
    and is reachable ONLY through the oracle replay path
    (``GameState.step(..., oracle_mode=True)``). If a future refactor pulled
    the helper into engine code — even just for typing or re-export — it
    could become reachable from the RL action stack and silently nullify
    this guard.

    Scan strategy: parse every ``engine/**/*.py`` with ``ast`` and assert
    no ``Import`` / ``ImportFrom`` references the oracle module, and no
    identifier reference (``Name``/``Attribute``/``FunctionDef``) names a
    forbidden helper. Comments, docstrings, and string literals are
    excluded — the guard banner in ``engine/action.py`` necessarily
    mentions these names in prose.
    """
    offenders: list[tuple[str, str]] = []
    assert _ENGINE_DIR.is_dir(), f"engine/ not found at {_ENGINE_DIR}"
    for py in sorted(_ENGINE_DIR.rglob("*.py")):
        rel = str(py.relative_to(_REPO_ROOT))
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as e:
            pytest.fail(f"failed to parse {rel}: {e}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(m in alias.name for m in _FORBIDDEN_IMPORT_MODULES):
                        offenders.append((rel, f"import {alias.name}"))
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if any(m in mod for m in _FORBIDDEN_IMPORT_MODULES):
                    offenders.append((rel, f"from {mod} import ..."))
                for alias in node.names:
                    if any(ident in alias.name for ident in _FORBIDDEN_IDENTIFIERS):
                        offenders.append((rel, f"from {mod} import {alias.name}"))
            elif isinstance(node, ast.Name):
                if node.id in _FORBIDDEN_IDENTIFIERS:
                    offenders.append((rel, f"Name:{node.id}"))
            elif isinstance(node, ast.Attribute):
                if node.attr in _FORBIDDEN_IDENTIFIERS:
                    offenders.append((rel, f"Attribute:.{node.attr}"))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in _FORBIDDEN_IDENTIFIERS:
                    offenders.append((rel, f"def {node.name}"))
    assert not offenders, (
        "engine/ AST references oracle delete helper or module — "
        "Phase 11J-DELETE-GUARD-PIN violated. Offenders: "
        + ", ".join(f"{p}:{n}" for p, n in offenders)
    )


# ---------------------------------------------------------------------------
# Test 5 — helper exists only on the oracle path
# ---------------------------------------------------------------------------

def test_oracle_delete_helper_only_callable_from_oracle_path():
    """Phase 11J-DELETE-GUARD-PIN: confirm the delete helper exists in
    ``tools/oracle_zip_replay.py`` (so the L2-BUILD-OCCUPIED replay handler
    still works) AND is not exported from ``engine/__init__.py`` or any
    engine top-level surface. Importing through ``tools.oracle_zip_replay``
    must succeed; importing through ``engine`` must fail. Locks in the
    architectural separation that keeps Delete oracle-only.
    """
    oracle_mod = importlib.import_module("tools.oracle_zip_replay")
    assert hasattr(oracle_mod, "_oracle_kill_friendly_unit"), (
        "tools/oracle_zip_replay.py must still expose _oracle_kill_friendly_unit "
        "for the L2-BUILD-OCCUPIED replay path; missing it breaks Phase 11J replays."
    )
    helper = getattr(oracle_mod, "_oracle_kill_friendly_unit")
    assert callable(helper), "_oracle_kill_friendly_unit must be callable."

    engine_mod = importlib.import_module("engine")
    for name in ("_oracle_kill_friendly_unit", "kill_friendly_unit", "delete_unit"):
        assert not hasattr(engine_mod, name), (
            f"engine module unexpectedly exposes {name!r}; "
            f"Phase 11J-DELETE-GUARD-PIN requires the helper stay oracle-only."
        )

    # Belt-and-braces: parse engine/__init__.py for any star-export or
    # explicit re-export of the helper. Catches `from tools.oracle_zip_replay
    # import _oracle_kill_friendly_unit` style leaks even if the runtime
    # hasattr check is somehow defeated.
    init_src = (_ENGINE_DIR / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(init_src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", None) or ""
            assert "oracle_zip_replay" not in mod, (
                "engine/__init__.py must not import from tools.oracle_zip_replay; "
                "Phase 11J-DELETE-GUARD-PIN."
            )
            for alias in node.names:
                assert "oracle_kill_friendly" not in alias.name.lower(), (
                    "engine/__init__.py re-exports the oracle delete helper; "
                    "Phase 11J-DELETE-GUARD-PIN."
                )
