"""Sign/scale tests for RHEA competency fitness knobs."""
from __future__ import annotations

import copy
import os
import random
from types import SimpleNamespace

import pytest

os.environ.setdefault("AWBW_REWARD_SHAPING", "phi")
os.environ.setdefault("AWBW_LEARNER_SEAT", "0")

from engine.action import ActionStage, ActionType
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData, PropertyState
from engine.unit import UNIT_STATS, Unit, UnitType

from rl.buy_exhaustive import enemy_air_threat_possible, score_buy_candidate
from rl.env import AWBWEnv
from rl.rhea import (
    ActionPool,
    BuildIntent,
    BuySpendGenome,
    RheaConfig,
    RheaGenome,
    RheaPlanner,
    UnitIntent,
    _sample_build_intent_biased_expensive,
)
from rl.rhea_fitness import RheaFitness


class _DummyFitness:
    pass


def _blank_map(*, props: list[PropertyState] | None = None) -> MapData:
    terrain = [[1] * 5 for _ in range(5)]
    return MapData(
        map_id=0,
        name="competency-toy",
        map_type="std",
        terrain=terrain,
        height=5,
        width=5,
        cap_limit=99,
        unit_limit=50,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=props or [],
        hq_positions={0: [], 1: []},
        lab_positions={0: [], 1: []},
        country_to_player={},
    )


def _state_from_map(
    map_data: MapData,
    *,
    units: dict[int, list[Unit]] | None = None,
    funds: list[int] | None = None,
    active: int = 0,
) -> GameState:
    return GameState(
        map_data=map_data,
        units=units or {0: [], 1: []},
        funds=funds or [10_000, 10_000],
        co_states=[make_co_state_safe(0), make_co_state_safe(0)],
        properties=list(map_data.properties),
        turn=3,
        active_player=active,
        action_stage=ActionStage.SELECT,
        selected_unit=None,
        selected_move_pos=None,
        done=False,
        winner=None,
        win_reason=None,
        game_log=[],
        tier_name="T2",
        full_trace=[],
    )


def _unit(tp: UnitType, player: int, pos: tuple[int, int], *, hp: int = 100, uid: int = 1) -> Unit:
    stats = UNIT_STATS[tp]
    return Unit(
        unit_id=uid,
        unit_type=tp,
        player=player,
        pos=pos,
        hp=hp,
        ammo=stats.max_ammo,
        fuel=stats.max_fuel,
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=0,
        is_stunned=False,
    )


def _fitness_env() -> AWBWEnv:
    return AWBWEnv(
        map_pool=[],
        opponent_policy="random",
        max_env_steps=100,
        max_p1_microsteps=100,
    )


def _capture_completion_states() -> tuple[GameState, GameState]:
    prop = PropertyState(
        terrain_id=34,
        row=2,
        col=2,
        owner=1,
        capture_points=20,
        is_hq=False,
        is_lab=False,
        is_comm_tower=False,
        is_base=False,
        is_airport=False,
        is_port=False,
    )
    md = _blank_map(props=[prop])
    before = _state_from_map(md)
    after = copy.deepcopy(before)
    after.properties[0].owner = 0
    after.properties[0].capture_points = 20
    return before, after


def test_capture_completion_bonus_raises_score_on_flip() -> None:
    before, after = _capture_completion_states()
    env = _fitness_env()
    off = RheaFitness(env_template=env, value_model=None, device="cpu")
    on = RheaFitness(
        env_template=env,
        value_model=None,
        device="cpu",
        capture_completion_bonus=0.05,
    )
    s_off = off.score(before, after, observer_seat=0).total
    s_on = on.score(before, after, observer_seat=0).total
    assert s_on > s_off
    assert s_on - s_off == pytest.approx(0.05)


def test_capture_completion_bonus_zero_is_noop() -> None:
    before, after = _capture_completion_states()
    env = _fitness_env()
    a = RheaFitness(env_template=env, value_model=None, device="cpu", capture_completion_bonus=0.0)
    b = RheaFitness(env_template=env, value_model=None, device="cpu")
    assert a.score(before, after, observer_seat=0).total == pytest.approx(
        b.score(before, after, observer_seat=0).total,
    )


def test_blunder_exposure_penalizes_threatened_unit() -> None:
    md = _blank_map()
    before = _state_from_map(
        md,
        units={
            0: [_unit(UnitType.INFANTRY, 0, (2, 2), uid=10)],
            1: [_unit(UnitType.TANK, 1, (2, 3), uid=20)],
        },
    )
    exposed = copy.deepcopy(before)
    safe = copy.deepcopy(before)
    safe.units[0][0].pos = (0, 0)
    safe.units[1][0].pos = (4, 4)
    env = _fitness_env()
    off = RheaFitness(env_template=env, value_model=None, device="cpu")
    on = RheaFitness(
        env_template=env,
        value_model=None,
        device="cpu",
        blunder_exposure_weight=0.5,
    )
    s_exposed_off = off.score(before, exposed, observer_seat=0).total
    s_exposed_on = on.score(before, exposed, observer_seat=0).total
    assert s_exposed_on < s_exposed_off
    s_safe_off = off.score(before, safe, observer_seat=0).total
    s_safe_on = on.score(before, safe, observer_seat=0).total
    assert (s_exposed_off - s_exposed_on) > (s_safe_off - s_safe_on)
    assert s_safe_on == pytest.approx(s_safe_off)


def test_blunder_exposure_exempts_unit_on_capturable_property() -> None:
    """Exposure while standing on a capturable property is purposeful — free."""
    prop = PropertyState(
        terrain_id=42,
        row=2,
        col=2,
        owner=1,
        capture_points=20,
        is_hq=False,
        is_lab=False,
        is_comm_tower=False,
        is_base=False,
        is_airport=False,
        is_port=False,
    )
    md = _blank_map(props=[prop])
    md.terrain[2][2] = 42  # city terrain so is_capturable_property_at sees property
    state = _state_from_map(
        md,
        units={
            0: [_unit(UnitType.INFANTRY, 0, (2, 2), uid=10)],
            1: [_unit(UnitType.TANK, 1, (2, 3), uid=20)],
        },
    )
    env = _fitness_env()
    on = RheaFitness(
        env_template=env,
        value_model=None,
        device="cpu",
        blunder_exposure_weight=0.5,
    )
    assert on._blunder_exposure_penalty(state, 0) == pytest.approx(0.0)


def test_blunder_exposure_scales_with_threat_magnitude() -> None:
    """Chip threat charges less than kill threat (risk-proportional term)."""
    md = _blank_map()
    chip = _state_from_map(
        md,
        units={
            # Infantry threatening a tank: chip damage only.
            0: [_unit(UnitType.TANK, 0, (2, 2), uid=10)],
            1: [_unit(UnitType.INFANTRY, 1, (2, 3), uid=20)],
        },
    )
    kill = _state_from_map(
        md,
        units={
            # Md tank threatening a tank: near-lethal.
            0: [_unit(UnitType.TANK, 0, (2, 2), uid=10)],
            1: [_unit(UnitType.MED_TANK, 1, (2, 3), uid=20)],
        },
    )
    env = _fitness_env()
    on = RheaFitness(
        env_template=env,
        value_model=None,
        device="cpu",
        blunder_exposure_weight=0.5,
    )
    p_chip = on._blunder_exposure_penalty(chip, 0)
    p_kill = on._blunder_exposure_penalty(kill, 0)
    assert 0.0 < p_chip < p_kill


def _hq_map() -> MapData:
    md = _blank_map()
    md.hq_positions = {0: [(0, 0)], 1: []}
    return md


def test_hq_defense_penalizes_undefended_snipe_threat() -> None:
    """Enemy infantry near my HQ with my whole army far away → penalty."""
    md = _hq_map()
    state = _state_from_map(
        md,
        units={
            # Infantry defender 8 tiles out (3 turns) cannot contest a
            # 1-turn-out sniper (window is threat_turns + 1 = 2 turns).
            0: [_unit(UnitType.INFANTRY, 0, (4, 4), uid=10)],
            1: [_unit(UnitType.INFANTRY, 1, (0, 2), uid=20)],  # 1 turn from HQ
        },
    )
    env = _fitness_env()
    on = RheaFitness(env_template=env, value_model=None, device="cpu", hq_defense_weight=0.1)
    off = RheaFitness(env_template=env, value_model=None, device="cpu")
    assert on._hq_defense_penalty(state, 0) > 0.0
    assert off._hq_defense_penalty(state, 0) == pytest.approx(0.0)


def test_hq_defense_no_penalty_with_defender_in_range() -> None:
    """A defender that can contest within threat_turns + 1 silences the term."""
    md = _hq_map()
    state = _state_from_map(
        md,
        units={
            0: [_unit(UnitType.INFANTRY, 0, (1, 0), uid=10)],  # adjacent to HQ
            1: [_unit(UnitType.INFANTRY, 1, (0, 2), uid=20)],
        },
    )
    env = _fitness_env()
    on = RheaFitness(env_template=env, value_model=None, device="cpu", hq_defense_weight=0.1)
    assert on._hq_defense_penalty(state, 0) == pytest.approx(0.0)


def test_hq_defense_ignores_distant_and_noncapture_threats() -> None:
    md = _hq_map()
    # Tank can't capture; lone enemy tank next to HQ is the blunder term's
    # business, not the HQ-defense term's.
    tank_only = _state_from_map(
        md,
        units={0: [], 1: [_unit(UnitType.TANK, 1, (0, 1), uid=20)]},
    )
    env = _fitness_env()
    on = RheaFitness(env_template=env, value_model=None, device="cpu", hq_defense_weight=0.1)
    assert on._hq_defense_penalty(tank_only, 0) == pytest.approx(0.0)


def test_hq_defense_rewards_moving_defender_closer() -> None:
    """A defender partway home must shrink the penalty (selection gradient)."""
    md = _hq_map()
    # Mech defenders (move 2): 8 tiles = 4 turns vs 6 tiles = 3 turns, both
    # outside the contest window (threat 1 turn + 1), so the gap term alone
    # differentiates them.
    far_def = _state_from_map(
        md,
        units={
            0: [_unit(UnitType.MECH, 0, (4, 4), uid=10)],
            1: [_unit(UnitType.INFANTRY, 1, (0, 2), uid=20)],
        },
    )
    closer_def = _state_from_map(
        md,
        units={
            0: [_unit(UnitType.MECH, 0, (3, 3), uid=10)],
            1: [_unit(UnitType.INFANTRY, 1, (0, 2), uid=20)],
        },
    )
    env = _fitness_env()
    on = RheaFitness(env_template=env, value_model=None, device="cpu", hq_defense_weight=0.1)
    p_far = on._hq_defense_penalty(far_def, 0)
    p_closer = on._hq_defense_penalty(closer_def, 0)
    assert p_far > p_closer > 0.0


def test_hq_defense_scales_with_threat_proximity() -> None:
    md = _hq_map()
    near = _state_from_map(
        md,
        units={0: [], 1: [_unit(UnitType.INFANTRY, 1, (0, 1), uid=20)]},
    )
    far = _state_from_map(
        md,
        units={0: [], 1: [_unit(UnitType.INFANTRY, 1, (4, 4), uid=20)]},
    )
    env = _fitness_env()
    on = RheaFitness(env_template=env, value_model=None, device="cpu", hq_defense_weight=0.1)
    p_near = on._hq_defense_penalty(near, 0)
    p_far = on._hq_defense_penalty(far, 0)
    assert p_near > p_far > 0.0


def _interrupt_states(
    *, is_hq: bool = False, is_base: bool = False, is_airport: bool = False,
    is_comm_tower: bool = False, is_port: bool = False, is_lab: bool = False,
) -> tuple[GameState, GameState]:
    """My property mid-capture by the enemy in before; progress wiped in after."""
    prop = PropertyState(
        terrain_id=34,
        row=2,
        col=2,
        owner=0,
        capture_points=10,
        is_hq=is_hq,
        is_lab=is_lab,
        is_comm_tower=is_comm_tower,
        is_base=is_base,
        is_airport=is_airport,
        is_port=is_port,
    )
    md = _blank_map(props=[prop])
    before = _state_from_map(md)
    after = copy.deepcopy(before)
    after.properties[0].capture_points = 20
    return before, after


def test_capture_interrupt_bonus_tiers() -> None:
    """City/tower/port/lab < base/airport < HQ."""
    env = _fitness_env()
    fit = RheaFitness(
        env_template=env, value_model=None, device="cpu", capture_interrupt_bonus=0.05,
    )
    def shaped(**kw: bool) -> float:
        before, after = _interrupt_states(**kw)
        return fit.competency_shaping(before, after, observer_seat=0)

    city = shaped()
    tower = shaped(is_comm_tower=True)
    port = shaped(is_port=True)
    lab = shaped(is_lab=True)
    base = shaped(is_base=True)
    airport = shaped(is_airport=True)
    hq = shaped(is_hq=True)
    assert city == tower == port == lab == pytest.approx(0.05)
    assert base == airport == pytest.approx(0.10)
    assert hq == pytest.approx(0.20)


def test_capture_interrupt_requires_actual_reset() -> None:
    env = _fitness_env()
    fit = RheaFitness(
        env_template=env, value_model=None, device="cpu", capture_interrupt_bonus=0.05,
    )
    # Capture still in progress in after — no reward.
    before, after = _interrupt_states()
    after.properties[0].capture_points = 10
    assert fit.competency_shaping(before, after, observer_seat=0) == pytest.approx(0.0)
    # Property flipped to the enemy — no reward.
    before, after = _interrupt_states()
    after.properties[0].owner = 1
    assert fit.competency_shaping(before, after, observer_seat=0) == pytest.approx(0.0)
    # Enemy property mid-capture (by me) resetting is not an interrupt for me.
    before, after = _interrupt_states()
    before.properties[0].owner = 1
    after.properties[0].owner = 1
    assert fit.competency_shaping(before, after, observer_seat=0) == pytest.approx(0.0)


def test_neutral_income_gap_penalty_scales_with_remaining_neutrals() -> None:
    prop_neu = PropertyState(
        terrain_id=34, row=2, col=2, owner=None, capture_points=20,
        is_hq=False, is_lab=False, is_comm_tower=False, is_base=False,
        is_airport=False, is_port=False,
    )
    prop_owned = PropertyState(
        terrain_id=38, row=2, col=3, owner=0, capture_points=20,
        is_hq=False, is_lab=False, is_comm_tower=False, is_base=False,
        is_airport=False, is_port=False,
    )
    md = _blank_map(props=[prop_neu, prop_owned])
    many_neu = _state_from_map(md)
    flipped = copy.deepcopy(many_neu)
    flipped.properties[0].owner = 0
    flipped.properties[0].capture_points = 20
    env = _fitness_env()
    on = RheaFitness(
        env_template=env, value_model=None, device="cpu", neutral_income_gap_weight=0.04,
    )
    p_many = on._neutral_income_gap_fraction(many_neu)
    p_few = on._neutral_income_gap_fraction(flipped)
    assert p_many > p_few
    assert on.competency_shaping(many_neu, flipped, observer_seat=0) > on.competency_shaping(
        many_neu, many_neu, observer_seat=0,
    )


def test_capture_progress_bonus_rewards_partial_neutral_capture() -> None:
    prop = PropertyState(
        terrain_id=34, row=2, col=2, owner=None, capture_points=20,
        is_hq=False, is_lab=False, is_comm_tower=False, is_base=False,
        is_airport=False, is_port=False,
    )
    md = _blank_map(props=[prop])
    before = _state_from_map(md)
    after = copy.deepcopy(before)
    after.properties[0].capture_points = 10
    env = _fitness_env()
    on = RheaFitness(
        env_template=env, value_model=None, device="cpu", capture_progress_bonus=0.02,
    )
    off = RheaFitness(env_template=env, value_model=None, device="cpu")
    assert on.competency_shaping(before, after, observer_seat=0) == pytest.approx(0.01)
    assert off.competency_shaping(before, after, observer_seat=0) == pytest.approx(0.0)


def test_capture_interrupt_default_off() -> None:
    env = _fitness_env()
    fit = RheaFitness(env_template=env, value_model=None, device="cpu")
    before, after = _interrupt_states(is_hq=True)
    assert fit.competency_shaping(before, after, observer_seat=0) == pytest.approx(0.0)


def _airport_buy_state(*, enemy_fighter: bool) -> tuple[GameState, tuple, dict, BuySpendGenome]:
    props = [
        PropertyState(
            terrain_id=36,
            row=0,
            col=1,
            owner=0,
            capture_points=20,
            is_hq=False,
            is_lab=False,
            is_comm_tower=False,
            is_base=False,
            is_airport=True,
            is_port=False,
        ),
    ]
    md = _blank_map(props=props)
    units: dict[int, list[Unit]] = {0: [], 1: []}
    if enemy_fighter:
        units[1] = [_unit(UnitType.FIGHTER, 1, (4, 4), uid=99)]
    post = _state_from_map(md, units=units, funds=[20_000, 10_000])
    fo = ((0, 1),)
    catalog = {fo[0]: (UnitType.INFANTRY, UnitType.MISSILES)}
    genome = BuySpendGenome(fo, [UnitType.MISSILES])
    return post, fo, catalog, genome


def test_buy_air_context_penalty_without_enemy_air() -> None:
    post, _fo, _catalog, genome = _airport_buy_state(enemy_fighter=False)
    assert enemy_air_threat_possible(post, 0) is False
    env = _fitness_env()
    fitness = RheaFitness(env_template=env, value_model=None, device="cpu")
    planner = RheaPlanner(fitness, RheaConfig(seed=1, buy_air_context_penalty=0.05))
    rw, vw = 0.90, 0.10
    common = dict(
        execute_buy=planner._execute_buy_spend_allocation,
        fitness=fitness,
        post_moves=post,
        genome=genome,
        acting_seat=0,
        reward_weight=rw,
        value_weight=vw,
        buy_value_scale=0.70,
        buy_shaping_weight=2.5,
        illegal_gene_penalty=0.02,
        v_ref_buy=float(fitness.value(post, 0)),
        phi_ref_buy=float(fitness.phi(post, 0)),
        gold_before=int(post.funds[0]),
        buy_air_context_penalty=0.0,
        air_threat_possible=False,
    )
    off = score_buy_candidate(**common)
    on = score_buy_candidate(**{**common, "buy_air_context_penalty": 0.05})
    assert on.total < off.total
    assert off.total - on.total == pytest.approx(0.05)


def test_buy_air_context_penalty_skipped_when_enemy_has_air() -> None:
    post, _fo, _catalog, genome = _airport_buy_state(enemy_fighter=True)
    assert enemy_air_threat_possible(post, 0) is True
    env = _fitness_env()
    fitness = RheaFitness(env_template=env, value_model=None, device="cpu")
    planner = RheaPlanner(fitness, RheaConfig(seed=1))
    common = dict(
        execute_buy=planner._execute_buy_spend_allocation,
        fitness=fitness,
        post_moves=post,
        genome=genome,
        acting_seat=0,
        reward_weight=0.90,
        value_weight=0.10,
        buy_value_scale=0.70,
        buy_shaping_weight=2.5,
        illegal_gene_penalty=0.02,
        v_ref_buy=float(fitness.value(post, 0)),
        phi_ref_buy=float(fitness.phi(post, 0)),
        gold_before=int(post.funds[0]),
        air_threat_possible=True,
    )
    off = score_buy_candidate(**common, buy_air_context_penalty=0.0)
    on = score_buy_candidate(**common, buy_air_context_penalty=0.05)
    assert on.total == pytest.approx(off.total)


def test_disable_expensive_build_bias_is_noop_on_genome() -> None:
    planner = RheaPlanner(_DummyFitness(), RheaConfig(seed=1, disable_expensive_build_bias=True))
    fac = (0, 1)
    cheap = BuildIntent(factory_pos=fac, unit_type=UnitType.INFANTRY)
    costly = BuildIntent(factory_pos=fac, unit_type=UnitType.MEGA_TANK)
    genome = RheaGenome(unit_segment=[], build_segment=[cheap])
    pool = ActionPool(
        unmoved_positions=[],
        unit_options={},
        build_options=[cheap, costly],
        cop_legal=False,
        scop_legal=False,
        power_active=False,
    )
    planner.rng = SimpleNamespace(random=lambda: 0.0, choice=lambda opts: opts[-1])
    planner._bias_genome_toward_expensive_builds(genome, pool)
    assert genome.build_segment[0].unit_type == UnitType.INFANTRY


def test_sample_build_intent_uniform_when_bias_disabled() -> None:
    fac = (0, 0)
    cheap = BuildIntent(factory_pos=fac, unit_type=UnitType.INFANTRY)
    costly = BuildIntent(factory_pos=fac, unit_type=UnitType.MEGA_TANK)
    opts = [cheap, costly]
    rng = random.Random(7)
    picks = {
        _sample_build_intent_biased_expensive(
            opts, rng, p_pick_max_cost=0.99, uniform=True,
        ).unit_type
        for _ in range(300)
    }
    assert UnitType.INFANTRY in picks
    assert UnitType.MEGA_TANK in picks


def test_default_off_regression_fixed_scenario() -> None:
    before, after = _capture_completion_states()
    md = _blank_map()
    threatened = _state_from_map(
        md,
        units={
            0: [_unit(UnitType.TANK, 0, (2, 2), uid=1)],
            1: [_unit(UnitType.TANK, 1, (2, 3), uid=2)],
        },
    )
    env = _fitness_env()
    explicit = RheaFitness(
        env_template=env,
        value_model=None,
        device="cpu",
        capture_completion_bonus=0.0,
        blunder_exposure_weight=0.0,
    )
    default = RheaFitness(env_template=env, value_model=None, device="cpu")
    for b, a in ((before, after), (before, threatened)):
        assert explicit.score(b, a, observer_seat=0).total == pytest.approx(
            default.score(b, a, observer_seat=0).total,
        )
