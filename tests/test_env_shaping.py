"""Invariant tests for potential-based reward shaping.

Plan: ``.cursor/plans/rl_capture-combat_recalibration_4ebf9d22.plan.md``.

The four invariants verified here:

1. **Value potential**     — losing a full-HP Inf for P0 changes Φ by exactly −α·1000.
2. **Capture refund on reset**
                            — chip credit is automatically returned when the
                              engine resets ``capture_points`` (capturer dies),
                              so a "cap chip → Inf killed" sequence nets to the
                              Inf's value cost (−α·1000), not zero.
3. **Flip transfer**        — flipping a chipped property moves κ·chip credit
                              out of Φ_cap and adds β to Φ_props, with no
                              hidden constants.
4. **Φ telescoping**        — across a non-terminal trajectory, the sum of
                              per-step Φ-shaping equals Φ(s_T) − Φ(s_0)
                              (potential-based shaping is policy-invariant
                              and bounded).
"""

from __future__ import annotations

import json
import random as random_module
from pathlib import Path

import pytest

from engine.action import Action, ActionType, ActionStage, get_legal_actions
from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, Unit, UnitType


def _inject_inf(state, player: int, pos: tuple[int, int], hp: int = 100) -> Unit:
    """Inject a fresh full-HP Infantry into ``state`` for tests that need a
    deterministic owner. Map 123858 starts P0 with no units, so the value /
    capture-refund tests need one planted by hand."""
    st = UNIT_STATS[UnitType.INFANTRY]
    inf = Unit(
        UnitType.INFANTRY,
        player,
        hp,
        st.max_ammo,
        st.max_fuel,
        pos,
        False,    # moved
        [],       # loaded_units
        False,    # hidden
        20,       # capture_points carried (unused on most actions)
        max((u.unit_id for plist in state.units.values() for u in plist), default=0) + 1,
    )
    state.units[player].append(inf)
    return inf


ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"
MAP_ID = 123858


def _single_map_pool() -> list[dict]:
    with open(POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    return [next(m for m in pool if m.get("map_id") == MAP_ID)]


@pytest.fixture
def phi_env(monkeypatch: pytest.MonkeyPatch):
    """Construct an AWBWEnv with potential-based shaping active.

    Sets the env vars *before* env construction (read once in ``__init__``)
    and patches the engine-side ``_PHI_SHAPING_ACTIVE`` gate so capture
    shaping is suppressed in ``_apply_capture`` (otherwise we'd double-count
    the chip via the engine's legacy 0.04/0.20/0.01 constants).
    """
    monkeypatch.setenv("AWBW_REWARD_SHAPING", "phi")
    monkeypatch.setenv("AWBW_PHI_ALPHA", "2e-5")
    monkeypatch.setenv("AWBW_PHI_BETA", "0.05")
    monkeypatch.setenv("AWBW_PHI_KAPPA", "0.05")
    monkeypatch.setattr("engine.game._PHI_SHAPING_ACTIVE", True)
    monkeypatch.setattr("rl.env._append_game_log_line", lambda _record: None)

    from rl.env import AWBWEnv

    env = AWBWEnv(
        map_pool=_single_map_pool(),
        opponent_policy=None,
        co_p0=1,
        co_p1=1,
        tier_name="T3",
    )
    env.reset(seed=0)
    return env


# ── Test 1: value potential ────────────────────────────────────────────────

def test_phi_value_potential_inf_loss(phi_env) -> None:
    """Killing one full-HP P0 Infantry shifts Φ by exactly −α·1000."""
    state = phi_env.state
    inf = _inject_inf(state, player=0, pos=(0, 0))
    phi_before = phi_env._compute_phi(state)

    inf.hp = 0  # ``is_alive`` is derived from hp; engine prunes elsewhere.

    phi_after = phi_env._compute_phi(state)

    assert phi_after - phi_before == pytest.approx(
        -phi_env._phi_alpha * 1000.0, rel=1e-9, abs=1e-12
    )


# ── Test 2: capture refund on reset ────────────────────────────────────────

def test_phi_chip_then_capturer_dies_refunds(phi_env) -> None:
    """Chip a building, then kill the capturer → ΔΦ = −α·(unit value).

    The engine resets ``cp = 20`` on capturer death; Φ_cap drops back to its
    pre-chip value automatically, so the only net Φ change is the value
    coin loss. No first-attempt pat masks the wasted Inf.
    """
    state = phi_env.state
    p0_inf = _inject_inf(state, player=0, pos=(0, 0))
    phi_initial = phi_env._compute_phi(state)

    contested = next(
        p for p in state.properties
        if p.owner != 0 and p.capture_points == 20
    )

    contested.capture_points = 10  # simulate chip 10/20
    phi_chipped = phi_env._compute_phi(state)
    expected_chip_delta = phi_env._phi_kappa * 0.5
    if contested.owner is None:
        # Neutral: contested for both seats; flips one entry on each side.
        # cap_p0 gains +0.5 (target for P0); cap_p1 gains +0.5 (target for P1
        # too — currently neutral). κ·(Δcap_p0 − Δcap_p1) = 0.
        expected_chip_delta = 0.0
    assert phi_chipped - phi_initial == pytest.approx(
        expected_chip_delta, rel=1e-9, abs=1e-12
    )

    # Capturer dies → engine canon resets cp back to 20.
    contested.capture_points = 20
    p0_inf.hp = 0

    phi_after = phi_env._compute_phi(state)
    delta_total = phi_after - phi_initial

    assert delta_total == pytest.approx(
        -phi_env._phi_alpha * 1000.0, rel=1e-9, abs=1e-12
    )


# ── Test 3: flip transfer ──────────────────────────────────────────────────

def test_phi_flip_transfers_cap_credit_to_property(phi_env) -> None:
    """Flipping a chipped P1-owned property to P0 produces a clean transfer:
    chip credit returns out of Φ_cap and the property-diff jumps by +2.

    Pre-flip  (owner=1, cp=0):  cap_p0 += 1.0, cap_p1 += 0.0
    Post-flip (owner=0, cp=20): cap_p0 += 0.0, cap_p1 += 0.0
    Δ(p0_props − p1_props) = +1 − (−1) = +2  →  β-term contributes 2β.
    Δ(cap_p0 − cap_p1)     = −1.0 − 0      = −1.0  →  κ-term contributes −κ.

    Step net = 2β − κ.  Trajectory total over (chip×2, flip) = 2β
    (one β for "gained property", one β for "opponent lost it").

    Also verify the trajectory total separately by snapshotting Φ at the
    pre-chip baseline; this is the load-bearing invariant the agent sees.
    """
    state = phi_env.state

    target = next(
        p for p in state.properties
        if p.owner == 1 and p.capture_points == 20
    )

    phi_pre_chip = phi_env._compute_phi(state)

    target.capture_points = 0
    phi_pre_flip = phi_env._compute_phi(state)

    target.owner = 0
    target.capture_points = 20
    phi_post_flip = phi_env._compute_phi(state)

    flip_step_delta = phi_post_flip - phi_pre_flip
    assert flip_step_delta == pytest.approx(
        2 * phi_env._phi_beta - phi_env._phi_kappa, rel=1e-9, abs=1e-12
    )

    trajectory_total = phi_post_flip - phi_pre_chip
    assert trajectory_total == pytest.approx(
        2 * phi_env._phi_beta, rel=1e-9, abs=1e-12
    )


# ── Test 4: Φ telescoping ──────────────────────────────────────────────────

def test_phi_telescoping_over_short_trajectory(
    monkeypatch: pytest.MonkeyPatch, phi_env
) -> None:
    """Sum of per-step Φ-shaping over a non-terminal episode equals
    Φ(s_final) − Φ(s_initial). Verifies the potential-based property and
    catches any drift in snapshot ordering (e.g. forgetting to refund chip
    on terminal, or computing Φ on wrong state).

    Verifies the **pure potential** slice: ``Σ phi_potential_delta`` equals
    ``Φ(s_final) − Φ(s_initial)``. Full ``reward`` also includes non-potential
    terms (power-use bonus, kill bonus, …) and is not expected to telescope.
    """
    # Make sure the random opponent doesn't end the game in two turns.
    rng = random_module.Random(0)
    monkeypatch.setattr("rl.env.random.choice", lambda seq: rng.choice(seq))

    env = phi_env
    state = env.state
    phi_initial = env._compute_phi(state)

    # END_TURN flat index is 0. Run a handful of P0 steps; if any terminate,
    # bail out (terminal Φ:=0 makes the invariant trivially hold but we're
    # testing the non-terminal telescoping path explicitly).
    total_reward = 0.0
    total_phi_pd = 0.0
    steps_taken = 0
    for _ in range(3):
        _obs, reward, terminated, truncated, info = env.step(0)
        total_reward += reward
        total_phi_pd += float(
            (info.get("reward_components") or {}).get("phi_potential_delta", 0.0)
        )
        steps_taken += 1
        if terminated or truncated:
            break

    if not env.state.done:
        phi_final = env._compute_phi(env.state)
        assert total_phi_pd == pytest.approx(
            phi_final - phi_initial, rel=1e-6, abs=1e-9
        ), (
            f"Φ potential telescoping broken: Σphi_potential_delta={total_phi_pd}, "
            f"Φ_T−Φ_0={phi_final - phi_initial}; "
            f"Σreward={total_reward} (includes non-potential terms e.g. power/kill bonuses)"
        )
    else:
        # Terminal hit: trajectory shaping = engine_terminal_reward + (0 − Φ_pre_terminal).
        # We can't cleanly separate the two from outside, so just assert finite.
        assert abs(total_reward) <= 2.0


# ── Pipe seam: no positive RL shaping for striking terrain ───────────────────


def test_seam_attack_yields_zero_engine_reward_and_unchanged_phi(
    phi_env,
) -> None:
    """Striking an intact pipe seam must not accrue engine dense reward or Φ.

    Seam HP is not a unit, not capture progress, and not a property — the RL
    stack (engine ``step`` + ``_compute_phi``) must stay blind to seam damage
    so the policy is not accidentally rewarded for chipping or breaking seams.
    """
    from engine.action import Action, ActionType, ActionStage
    from engine.co import make_co_state_safe
    from engine.game import GameState
    from engine.map_loader import MapData
    from engine.unit import Unit, UnitType, UNIT_STATS

    HPIPE, HPIPE_SEAM = 102, 113
    terrain = [[HPIPE, HPIPE, HPIPE, HPIPE_SEAM]]
    md = MapData(
        map_id=999_888,
        name="seam_rl_probe",
        map_type="std",
        terrain=[row[:] for row in terrain],
        height=1,
        width=4,
        cap_limit=999,
        unit_limit=999,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=[],
        hq_positions={0: [], 1: []},
        lab_positions={0: [], 1: []},
        country_to_player={},
        predeployed_specs=[],
    )
    st = UNIT_STATS[UnitType.PIPERUNNER]
    pr = Unit(
        unit_type=UnitType.PIPERUNNER,
        player=0,
        hp=100,
        ammo=st.max_ammo,
        fuel=st.max_fuel,
        pos=(0, 0),
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
        unit_id=1,
    )
    state = GameState(
        map_data=md,
        units={0: [pr], 1: []},
        funds=[0, 0],
        co_states=[make_co_state_safe(1), make_co_state_safe(1)],
        properties=[],
        turn=1,
        active_player=0,
        action_stage=ActionStage.SELECT,
        selected_unit=None,
        selected_move_pos=None,
        done=False,
        winner=None,
        win_reason=None,
        game_log=[],
        tier_name="T2",
        full_trace=[],
        seam_hp={(0, 3): 99},
    )
    state.step(Action(ActionType.SELECT_UNIT, unit_pos=(0, 0)))
    state.step(
        Action(ActionType.SELECT_UNIT, unit_pos=(0, 0), move_pos=(0, 0))
    )
    phi_before = phi_env._compute_phi(state)

    _s, r_eng, done = state.step(
        Action(
            ActionType.ATTACK,
            unit_pos=(0, 0),
            move_pos=(0, 0),
            target_pos=(0, 3),
        )
    )
    assert r_eng == 0.0
    assert not done
    assert phi_env._compute_phi(state) == pytest.approx(phi_before, rel=1e-9, abs=1e-9)


def test_phi_cop_scop_activation_bonus_learner_frame(phi_env) -> None:
    """COP has no Φ bonus; SCOP bonus is 5% of nominal (meter-scaled)."""
    from rl.env import PHI_SCOP_ACTIVATION_BONUS, PHI_VON_BOLT_SCOP_REF_THRESHOLD

    env = phi_env
    st = env.state
    assert int(env._learner_seat) == 0
    vb_ref = PHI_VON_BOLT_SCOP_REF_THRESHOLD
    assert st.co_states[0]._cop_threshold / vb_ref == pytest.approx(0.3)
    assert st.co_states[0]._scop_threshold / vb_ref == pytest.approx(0.6)
    cop = Action(ActionType.ACTIVATE_COP)
    scop = Action(ActionType.ACTIVATE_SCOP)
    b0, _b_act0, rc0, _pa0, _as0 = env._phi_power_activation_and_attack_bonus(cop, 0, st, learner_frame=True)
    assert b0 == 0.0 and _b_act0 == 0.0 and rc0 == {}
    b1, _b_act1, rc1, _pa1, _as1 = env._phi_power_activation_and_attack_bonus(cop, 1, st, learner_frame=True)
    assert b1 == 0.0 and _b_act1 == 0.0 and rc1 == {}
    s0, _s_act, rcs, _pas, _sas = env._phi_power_activation_and_attack_bonus(scop, 0, st, learner_frame=True)
    assert rcs["phi_power_bonus_meter_scale_vs_vb_scop"] == pytest.approx(0.6)
    assert s0 == pytest.approx(PHI_SCOP_ACTIVATION_BONUS * 0.6 * 0.05)
    assert rcs["phi_scop_activation_bonus"] == pytest.approx(s0)


def test_phi_scop_bonus_adder_cheaper_than_hawke() -> None:
    from engine.co import make_co_state

    from rl.env import PHI_SCOP_ACTIVATION_BONUS, PHI_VON_BOLT_SCOP_REF_THRESHOLD

    adder = make_co_state(11)
    hawke = make_co_state(12)
    von_bolt = make_co_state(30)
    ref = PHI_VON_BOLT_SCOP_REF_THRESHOLD
    b_adder = PHI_SCOP_ACTIVATION_BONUS * (float(adder._scop_threshold) / ref)
    b_hawke = PHI_SCOP_ACTIVATION_BONUS * (float(hawke._scop_threshold) / ref)
    b_von_bolt = PHI_SCOP_ACTIVATION_BONUS * (float(von_bolt._scop_threshold) / ref)
    assert b_adder < b_hawke
    assert b_hawke < b_von_bolt
    assert b_von_bolt == pytest.approx(PHI_SCOP_ACTIVATION_BONUS)


def test_phi_attack_stats_advantage_bonus(phi_env) -> None:
    """ATTACK shaping uses strike stats advantage (not COP/SCOP gating)."""
    env = phi_env
    st = env.state
    atk = Action(ActionType.ATTACK, unit_pos=(0, 0), move_pos=(0, 0), target_pos=(0, 3))
    b0, _ba0, rc0, _, _ = env._phi_power_activation_and_attack_bonus(atk, 0, st, learner_frame=True)
    assert "phi_attack_stats_advantage_mult" in rc0
    assert "phi_power_turn_attack_bonus" not in rc0
    mult0 = float(rc0["phi_attack_stats_advantage_mult"])
    if mult0 > 0:
        assert b0 == pytest.approx(0.001 * mult0)
        assert rc0["phi_attack_stats_advantage_bonus"] == pytest.approx(0.001 * mult0)
    else:
        assert b0 == pytest.approx(0.0)
    st.co_states[0].cop_active = True
    b1, _ba1, rc1, _, _ = env._phi_power_activation_and_attack_bonus(atk, 0, st, learner_frame=True)
    mult1 = float(rc1["phi_attack_stats_advantage_mult"])
    if mult1 > 0:
        assert b1 == pytest.approx(0.001 * mult1)
    st.co_states[0].cop_active = False
    st.co_states[0].scop_active = True
    b2, _ba2, rc2, _, _ = env._phi_power_activation_and_attack_bonus(atk, 0, st, learner_frame=True)
    mult2 = float(rc2["phi_attack_stats_advantage_mult"])
    if mult2 > 0:
        assert b2 == pytest.approx(0.001 * mult2)


def test_designed_desires_checkpoint_penalty_sum() -> None:
    from rl.env import AWBWEnv, PHI_DESIGNED_DESIRES_MAP_ID

    env = object.__new__(AWBWEnv)
    env._reward_shaping_mode = "phi"
    env._episode_info = {"map_id": PHI_DESIGNED_DESIRES_MAP_ID}
    env._designed_desires_checkpoint_misses = 0

    class _St:
        def count_properties(self, p: int) -> int:
            return 3

    env.state = _St()
    assert AWBWEnv._designed_desires_checkpoint_penalty_sum_for_turn_bump(
        env, 5, 6, 0
    ) == pytest.approx(-0.1)
    assert AWBWEnv._designed_desires_checkpoint_penalty_sum_for_turn_bump(
        env, 6, 7, 0
    ) == pytest.approx(-0.01)
    assert env._designed_desires_checkpoint_misses == 2


def test_designed_desires_shaping_active_only_phi_and_map() -> None:
    from rl.env import AWBWEnv, PHI_DESIGNED_DESIRES_MAP_ID

    env = object.__new__(AWBWEnv)
    env._reward_shaping_mode = "level"
    env._episode_info = {"map_id": PHI_DESIGNED_DESIRES_MAP_ID}
    assert not AWBWEnv._designed_desires_shaping_active(env)
    env._reward_shaping_mode = "phi"
    env._episode_info = {"map_id": 1}
    assert not AWBWEnv._designed_desires_shaping_active(env)


def test_reward_contract_seat_local_includes_designed_desires_stack() -> None:
    from types import SimpleNamespace

    from rl.env import AWBWEnv

    env = SimpleNamespace(_learner_seat=0, _pairwise_zero_sum_reward=True)
    contract = AWBWEnv._reward_contract_info(
        env,
        competitive_learner=0.22,
        common_learner=-0.01,
        seat_local_learner=-0.05 + (-0.1),
        final_reward=0.06,
    )
    assert contract["seat_local_by_seat"] == pytest.approx([-0.15, 0.0])
    assert contract["training_reward"] == pytest.approx(0.06)


def test_phi_episode_power_activation_counters(phi_env) -> None:
    env = phi_env
    env._episode_cop_by_seat = [0, 0]
    env._episode_scop_by_seat = [0, 0]
    env._phi_after_step_record_power_activations(Action(ActionType.ACTIVATE_COP), 0)
    env._phi_after_step_record_power_activations(Action(ActionType.ACTIVATE_SCOP), 1)
    assert env._episode_cop_by_seat == [1, 0]
    assert env._episode_scop_by_seat == [0, 1]
