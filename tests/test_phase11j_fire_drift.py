"""Phase 11J FIRE-DRIFT regressions — engine + oracle edits.

Covers the three distinct fix families introduced in
``docs/oracle_exception_audit/phase11j_fire_drift_fix.md``:

* **Edit A (P-COLO-ATTACKER)** — ``_apply_attack`` prefers
  ``state.selected_unit`` over ``get_unit_at`` when STEP-GATE has pinned the
  actor on a co-occupied tile (GL 1634664 friendly-fire false positive).
* **Edit B (P-AMMO override-bypass)** — ``_apply_attack`` skips the
  defense-in-depth range check when ``_oracle_combat_damage_override`` is
  set, so MG-only / ammo=0 secondary strikes the oracle pinned still apply
  (GL 1622104, 1625784, 1630983, 1635025, 1635846).
* **Edit C (P-DRIFT-DEFENDER)** —
  ``_oracle_assert_fire_damage_table_compatible`` raises
  ``UnsupportedOracleAction`` when the resolved engine defender has no
  damage entry for the attacker (GL 1631494 FIGHTER vs TANK).
"""
from __future__ import annotations

import pytest

from engine.action import Action, ActionStage, ActionType
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData, PropertyState
from engine.unit import Unit, UnitType, UNIT_STATS

from tools.oracle_zip_replay import (
    UnsupportedOracleAction,
    _oracle_assert_fire_damage_table_compatible,
)

PLAIN = 1


def _make_state(width: int, height: int) -> GameState:
    md = MapData(
        map_id=0,
        name="phase11j",
        map_type="std",
        terrain=[[PLAIN] * width for _ in range(height)],
        height=height,
        width=width,
        cap_limit=99,
        unit_limit=50,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=[],
        hq_positions={0: [], 1: []},
        lab_positions={0: [], 1: []},
        country_to_player={},
    )
    return GameState(
        map_data=md,
        units={0: [], 1: []},
        funds=[0, 0],
        co_states=[make_co_state_safe(0), make_co_state_safe(0)],
        properties=[],
        turn=1,
        active_player=0,
        action_stage=ActionStage.ACTION,
        selected_unit=None,
        selected_move_pos=None,
        done=False,
        winner=None,
        win_reason=None,
        game_log=[],
        tier_name="T2",
        full_trace=[],
    )


def _spawn(
    state: GameState,
    ut: UnitType,
    player: int,
    pos: tuple[int, int],
    *,
    unit_id: int,
    ammo: int | None = None,
    hp: int = 100,
) -> Unit:
    stats = UNIT_STATS[ut]
    if ammo is None:
        ammo = stats.max_ammo if stats.max_ammo > 0 else 0
    u = Unit(
        unit_type=ut,
        player=player,
        hp=hp,
        ammo=ammo,
        fuel=stats.max_fuel,
        pos=pos,
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
        unit_id=unit_id,
    )
    state.units[player].append(u)
    return u


# ---------------------------------------------------------------------------
# Edit A — selected_unit beats get_unit_at on co-occupied tile (GL 1634664)
# ---------------------------------------------------------------------------


def test_phase11j_edit_a_selected_unit_wins_over_first_unit_on_tile():
    """Two INFs co-located at (0,0): P0 stationary (older entry), P1 just
    moved. ``get_unit_at`` returns the *first* unit at that tile, which on a
    naive lookup is the P0 unit — the friendly-fire guard would then trip
    when the action targets a P0 INF at (0,1). With Edit A, ``selected_unit``
    (the P1 attacker) is preferred and the strike resolves cleanly.
    """
    state = _make_state(width=3, height=1)
    p0_stationary = _spawn(state, UnitType.INFANTRY, 0, (0, 0), unit_id=10)
    p1_actor = _spawn(state, UnitType.INFANTRY, 1, (0, 0), unit_id=11)
    p0_target = _spawn(state, UnitType.INFANTRY, 0, (0, 1), unit_id=12, hp=40)
    state.active_player = 1
    state.selected_unit = p1_actor
    state.selected_move_pos = (0, 0)

    assert state.units[0][0] is p0_stationary
    assert state.get_unit_at(0, 0) is p0_stationary

    state._apply_attack(
        Action(
            ActionType.ATTACK,
            unit_pos=(0, 0),
            move_pos=(0, 0),
            target_pos=(0, 1),
        )
    )

    assert p0_target.hp < 40
    assert p1_actor.is_alive
    assert state.selected_unit is None


def test_phase11j_edit_a_no_selected_unit_falls_back_to_get_unit_at():
    """Edit A is inert when ``selected_unit`` is None — legacy paths
    (tests, seam attacks) keep the old behaviour."""
    state = _make_state(width=3, height=1)
    p1_actor = _spawn(state, UnitType.INFANTRY, 1, (0, 0), unit_id=20)
    p0_target = _spawn(state, UnitType.INFANTRY, 0, (0, 1), unit_id=21, hp=40)
    state.active_player = 1
    state.selected_unit = None
    state.selected_move_pos = (0, 0)

    state._apply_attack(
        Action(
            ActionType.ATTACK,
            unit_pos=(0, 0),
            move_pos=(0, 0),
            target_pos=(0, 1),
        )
    )

    assert p0_target.hp < 40
    assert p1_actor.is_alive


# ---------------------------------------------------------------------------
# Edit B — override-bypass admits ammo=0 / MG-only strikes the oracle pinned
# ---------------------------------------------------------------------------


def test_phase11j_edit_b_override_bypasses_range_check_for_ammo_zero_mech():
    """Mech with primary ammo=0 versus an adjacent Tank — primary table has
    no entry (Mech anti-tank is the bazooka, requires ammo) but secondary MG
    is unmetered. ``get_attack_targets`` shorts to ``[]`` because Mech has
    ``max_ammo > 0`` and ``unit.ammo == 0``. Without the override-bypass,
    ``_apply_attack`` raises 'target not in attack range'. With it, the
    oracle-supplied damage applies cleanly.
    """
    state = _make_state(width=3, height=1)
    mech = _spawn(state, UnitType.MECH, 0, (0, 0), unit_id=30, ammo=0)
    tank = _spawn(state, UnitType.TANK, 1, (0, 1), unit_id=31, hp=100)
    state.selected_unit = mech
    state.selected_move_pos = (0, 0)
    state._oracle_combat_damage_override = (10, 0)

    state._apply_attack(
        Action(
            ActionType.ATTACK,
            unit_pos=(0, 0),
            move_pos=(0, 0),
            target_pos=(0, 1),
        )
    )

    assert tank.hp == 90
    assert state._oracle_combat_damage_override is None


def test_phase11j_edit_b_no_override_still_enforces_range_check():
    """Without an override, ammo=0 Mech vs Tank is still rejected — the
    bypass is gated strictly on ``_oracle_combat_damage_override``."""
    state = _make_state(width=3, height=1)
    mech = _spawn(state, UnitType.MECH, 0, (0, 0), unit_id=40, ammo=0)
    _spawn(state, UnitType.TANK, 1, (0, 1), unit_id=41, hp=100)
    state.selected_unit = mech
    state.selected_move_pos = (0, 0)
    state._oracle_combat_damage_override = None

    with pytest.raises(ValueError, match="not in attack range"):
        state._apply_attack(
            Action(
                ActionType.ATTACK,
                unit_pos=(0, 0),
                move_pos=(0, 0),
                target_pos=(0, 1),
            )
        )


def test_phase11j_edit_b_override_does_not_bypass_friendly_fire_guard():
    """The override-bypass is scoped to the range check only — friendly
    fire remains an unconditional invariant."""
    state = _make_state(width=3, height=1)
    mech = _spawn(state, UnitType.MECH, 0, (0, 0), unit_id=50, ammo=0)
    _spawn(state, UnitType.INFANTRY, 0, (0, 1), unit_id=51, hp=100)
    state.selected_unit = mech
    state.selected_move_pos = (0, 0)
    state._oracle_combat_damage_override = (10, 0)

    with pytest.raises(ValueError, match="friendly fire"):
        state._apply_attack(
            Action(
                ActionType.ATTACK,
                unit_pos=(0, 0),
                move_pos=(0, 0),
                target_pos=(0, 1),
            )
        )


# ---------------------------------------------------------------------------
# Edit C — oracle damage-table guard reclassifies engine_bug -> oracle_gap
# ---------------------------------------------------------------------------


def test_phase11j_edit_c_fighter_vs_tank_raises_unsupported_oracle_action():
    """Mirrors GL 1631494: Fighter resolved by oracle to strike a Tank tile
    (no AWBW Fighter-vs-Tank damage entry). Helper raises
    ``UnsupportedOracleAction`` so the audit reclassifies the row away
    from ``engine_bug``.
    """
    state = _make_state(width=3, height=3)
    fighter = _spawn(state, UnitType.FIGHTER, 0, (1, 0), unit_id=60)
    _spawn(state, UnitType.TANK, 1, (1, 1), unit_id=61, hp=100)

    with pytest.raises(UnsupportedOracleAction, match="no damage entry"):
        _oracle_assert_fire_damage_table_compatible(state, fighter, (1, 1))


def test_phase11j_edit_c_compatible_pair_passes_silently():
    """Mech vs Infantry has a table entry — guard is a no-op."""
    state = _make_state(width=3, height=3)
    mech = _spawn(state, UnitType.MECH, 0, (1, 0), unit_id=70, ammo=0)
    _spawn(state, UnitType.INFANTRY, 1, (1, 1), unit_id=71, hp=100)

    _oracle_assert_fire_damage_table_compatible(state, mech, (1, 1))


def test_phase11j_edit_c_empty_tile_passes_silently():
    """No engine defender at the resolved tile — Fire branch will hit the
    seam logic or the post-move ``defender is None`` path. Guard does not
    interfere.
    """
    state = _make_state(width=3, height=3)
    fighter = _spawn(state, UnitType.FIGHTER, 0, (1, 0), unit_id=80)

    _oracle_assert_fire_damage_table_compatible(state, fighter, (1, 1))
