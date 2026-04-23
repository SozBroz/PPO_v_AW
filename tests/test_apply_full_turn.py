"""Phase 11a: ``GameState.apply_full_turn`` — turn-level rollout for MCTS foundation."""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.action import Action, ActionStage, ActionType, get_legal_actions  # noqa: E402
from engine.game import GameState, make_initial_state  # noqa: E402
from engine.map_loader import MapData  # noqa: E402
from engine.unit import Unit, UnitType  # noqa: E402

_PLAIN = 1


def _tiny_plains(w: int, h: int = 1) -> MapData:
    return MapData(
        map_id=991_201,
        name="apply_full_turn_tiny",
        map_type="std",
        terrain=[[_PLAIN] * w for _ in range(h)],
        height=h,
        width=w,
        cap_limit=999,
        unit_limit=50,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=[],
        hq_positions={0: [], 1: []},
        lab_positions={0: [], 1: []},
        country_to_player={},
        predeployed_specs=[],
    )


def _empty_p0_state() -> GameState:
    """P0, SELECT, no units — only unmoved block cleared so END_TURN is legal by itself."""
    m = _tiny_plains(2, 1)
    s = make_initial_state(m, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0)
    s.units = {0: [], 1: []}
    return s


def _p0_one_inf_state() -> GameState:
    """P0 with one unmoved infantry — must SELECT→MOVE→WAIT (or similar) before END_TURN."""
    m = _tiny_plains(4, 1)
    s = make_initial_state(m, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0)
    s.units = {0: [], 1: []}
    u0 = Unit(
        UnitType.INFANTRY,
        0,
        100,
        0,
        99,
        (0, 0),
        False,
        [],
        False,
        20,
    )
    u0.unit_id = s._allocate_unit_id()
    s.units[0].append(u0)
    return s


def _end_turn_from(s: GameState) -> Action:
    for a in get_legal_actions(s):
        if a.action_type == ActionType.END_TURN:
            return a
    raise AssertionError("no END_TURN in get_legal_actions; fix fixture")


def _p0_p1_adjacent_infantry() -> GameState:
    """1×2 map: P0 and P1 infantry adjacent — direct combat uses luck."""
    m = _tiny_plains(2, 1)
    s = make_initial_state(m, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0)
    s.units = {0: [], 1: []}
    u0 = Unit(
        UnitType.INFANTRY,
        0,
        100,
        0,
        99,
        (0, 0),
        False,
        [],
        False,
        20,
    )
    u1 = Unit(
        UnitType.INFANTRY,
        1,
        100,
        0,
        99,
        (0, 1),
        False,
        [],
        False,
        20,
    )
    u0.unit_id = s._allocate_unit_id()
    u1.unit_id = s._allocate_unit_id()
    s.units[0].append(u0)
    s.units[1].append(u1)
    return s


def _first_of_type(leg: list[Action], t: ActionType) -> Action:
    for a in leg:
        if a.action_type == t:
            return a
    raise AssertionError(f"no {t.name} in legality list ({len(leg)} actions)")


def _combat_then_end_plan(s: GameState) -> list[Action]:
    """SELECT P0 inf → move stay → ATTACK P1 at (0,1) → END_TURN."""
    st = s
    assert st.action_stage == ActionStage.SELECT
    u0 = st.units[0][0]
    a_sel = _first_of_type(
        get_legal_actions(st),
        ActionType.SELECT_UNIT,
    )
    if a_sel.unit_pos != u0.pos:
        a_sel = next(
            a
            for a in get_legal_actions(st)
            if a.action_type == ActionType.SELECT_UNIT and a.unit_pos == u0.pos
        )
    st = copy.deepcopy(s)
    st.step(a_sel)
    a_move = _first_of_type(get_legal_actions(st), ActionType.SELECT_UNIT)
    assert a_move.move_pos is not None
    st2 = copy.deepcopy(s)
    st2.step(a_sel)
    st2.step(a_move)
    atk = next(
        a
        for a in get_legal_actions(st2)
        if a.action_type == ActionType.ATTACK and a.target_pos == (0, 1)
    )
    a_end = Action(ActionType.END_TURN)
    return [a_sel, a_move, atk, a_end]


def _structural_same(a: GameState, b: GameState) -> None:
    assert a.turn == b.turn
    assert a.active_player == b.active_player
    assert a.action_stage == b.action_stage
    assert a.funds == b.funds
    assert a.done == b.done
    for p in (0, 1):
        assert len(a.units[p]) == len(b.units[p])
    for p in (0, 1):
        sa = sorted(a.units[p], key=lambda u: u.unit_id)
        sb = sorted(b.units[p], key=lambda u: u.unit_id)
        for ua, ub in zip(sa, sb, strict=True):
            assert ua.pos == ub.pos
            assert ua.hp == ub.hp
            assert ua.player == ub.player
            assert ua.unit_type == ub.unit_type
            assert ua.ammo == ub.ammo
            assert ua.fuel == ub.fuel


def _state_fingerprint(st: GameState) -> tuple:
    ubits = []
    for p in (0, 1):
        for u in sorted(st.units[p], key=lambda x: x.unit_id):
            ubits.append((p, u.unit_id, u.pos, u.hp, u.ammo, u.fuel, u.moved))
    return (
        st.turn,
        st.active_player,
        int(st.action_stage),
        st.done,
        tuple(st.funds),
        tuple(ubits),
    )


def test_random_policy_completes_turn() -> None:
    s = _p0_one_inf_state()
    starting = s.active_player
    random.seed(7)

    def policy(st: GameState) -> Action:
        leg = get_legal_actions(st)
        assert leg
        return random.choice(leg)

    out, taken, _rtot, d = s.apply_full_turn(policy, copy=True)
    assert out is not s
    assert len(taken) >= 1
    assert d or out.active_player == 1 - starting


def test_plan_replay_matches_step_by_step() -> None:
    s0 = _p0_one_inf_state()
    starting = s0.active_player
    acts: list[Action] = []
    st = copy.deepcopy(s0)
    random.seed(11)
    while st.active_player == starting and not st.done:
        a = random.choice(get_legal_actions(st))
        st.step(a)
        acts.append(a)

    fresh = copy.deepcopy(s0)
    out, replayed, _, d2 = fresh.apply_full_turn(acts, copy=False)
    assert not d2  # this scenario should not end the game
    _structural_same(st, out)
    assert replayed == acts


def test_copy_true_does_not_mutate_input() -> None:
    s0 = _empty_p0_state()
    snap = copy.deepcopy(s0)
    s0.apply_full_turn([_end_turn_from(s0)], copy=True)
    _structural_same(s0, snap)


def test_copy_false_mutates_input() -> None:
    s0 = _empty_p0_state()
    et = _end_turn_from(s0)
    out, _, _, _ = s0.apply_full_turn(
        [et],
        copy=False,
    )
    assert out is s0
    assert s0.active_player == 1


def test_non_select_stage_raises_value_error() -> None:
    s0 = _p0_one_inf_state()
    u0 = s0.units[0][0]
    a0 = next(
        a
        for a in get_legal_actions(s0)
        if a.action_type == ActionType.SELECT_UNIT and a.unit_pos == u0.pos
    )
    s0.step(a0)
    assert s0.action_stage == ActionStage.MOVE
    with pytest.raises(ValueError, match="action_stage==SELECT"):
        s0.apply_full_turn([Action(ActionType.END_TURN)])


def test_plan_exhaustion_raises_runtime_error() -> None:
    s0 = _empty_p0_state()
    with pytest.raises(RuntimeError, match="[Ee]xhausted"):
        s0.apply_full_turn([])


def test_max_actions_cap() -> None:
    s0 = _p0_one_inf_state()
    u0 = s0.units[0][0]
    a_sel = next(
        a
        for a in get_legal_actions(s0)
        if a.action_type == ActionType.SELECT_UNIT and a.unit_pos == u0.pos
    )
    st_mid = copy.deepcopy(s0)
    st_mid.step(a_sel)
    a_move_leg = _first_of_type(get_legal_actions(st_mid), ActionType.SELECT_UNIT)
    plan = [a_sel, a_move_leg, Action(ActionType.END_TURN)]
    with pytest.raises(RuntimeError, match="max_actions"):
        s0.apply_full_turn(plan, max_actions=1)


def test_rng_seed_determinism() -> None:
    s0 = _p0_p1_adjacent_infantry()
    plan = _combat_then_end_plan(s0)
    t1, d1, f1 = _run_seeded(s0, plan, 2026)
    t2, d2, f2 = _run_seeded(s0, plan, 2026)
    assert (t1, d1, f1) == (t2, d2, f2)
    t3, d3, f3 = _run_seeded(s0, plan, 1337)
    assert (t1, f1) != (t3, f3) or d1 != d3


def _run_seeded(
    s0: GameState, plan: list[Action], seed: int
) -> tuple[float, bool, tuple]:
    _out, _acts, tot, d = s0.apply_full_turn(plan, copy=True, rng_seed=seed)
    return tot, d, _state_fingerprint(_out)


def test_rng_state_restored_after_call() -> None:
    s0 = _empty_p0_state()
    random.seed(0)
    a = random.random()
    random.seed(0)
    s0.apply_full_turn([_end_turn_from(s0)], rng_seed=999)
    b = random.random()
    assert a == b


def test_on_step_callback_invoked_per_action() -> None:
    s0 = _empty_p0_state()
    log: list[tuple] = []
    et = _end_turn_from(s0)

    def on_step(
        st: GameState, act: Action, r: float, d: bool
    ) -> None:
        log.append((act.action_type, r, d))

    _out, taken, _tr, _d = s0.apply_full_turn(
        [et],
        on_step=on_step,
    )
    assert len(log) == len(taken) == 1

