"""Phase 11J-COLIN-IMPL-SHIP — Colin (CO 15) D2D + Gold Rush + Power of Money.

Three CO mechanics, ≤60 LOC engine, all wiki-anchored.

AWBW canon (Tier 1 — both AWBW canonicals agree, see
``docs/oracle_exception_audit/phase11y_colin_scrape.md``):

  * **D2D** — *"Units cost −20 % less to build and lose −10 % attack."*
    Sources: https://awbw.amarriner.com/co.php (Colin row) and
    https://awbw.fandom.com/wiki/Colin

  * **COP "Gold Rush"** — *"Funds are multiplied by 1.5x."*
    Sources: same as above.
    Rounding: **round-half-up** (PHP ``round()`` default), confirmed on
    15 / 15 sub=0 COP envelopes via PHP-payload drill (scrape §7.3).
    The three boundary cases on the .5 mark all matched ``round_half_up``,
    NOT ``int()`` floor — using ``int()`` would silently desync ~20 %
    of COP fires. Engine implements ``(3 * pre + 1) // 2`` (pure integer
    round-half-up for ``pre >= 0``), clamped to 999 999.

  * **SCOP "Power of Money"** — *"Unit attack percentage increases by
    (3 * Funds / 1000)%."*
    Stacks with D2D −10 %% AND with the universal +10 %% SCOPB rider that
    ``COState.cop_atk_modifier`` already applies (scrape §0.4).
    Funds source: snapshotted into ``COState.colin_pom_funds_snapshot`` at
    SCOP activation so post-SCOP spending does not erode the bonus
    mid-turn (one-turn AW power semantics).

Engine sites:
  * ``engine.action._build_cost`` Colin elif (×0.8) — pre-existing,
    pinned for parity by ``test_colin_d2d_cost_tank_5600``.
  * ``engine.combat._colin_atk_rider`` (additive AV delta, mirrors
    ``_kindle_atk_rider`` pattern).
  * ``engine.game._apply_power_effects`` Colin co_id==15 branch:
    COP → funds round_half_up(×1.5); SCOP → funds snapshot.

Corpus closure note: zero Colin gids in
``logs/desync_register_post_phase11j_v2_936.jsonl`` per CO-WAVE-2 and
COLIN-SCRAPE §1 (Colin sits in disabled tier T0 of the GL std rotation).
This suite is the canonical pin until non-std Colin replays are added
to the audit harness.
"""
from __future__ import annotations

import unittest

from engine.action import ActionStage, _build_cost
from engine.co import make_co_state_safe
from engine.combat import calculate_damage, _colin_atk_rider
from engine.game import GameState
from engine.map_loader import MapData, PropertyState
from engine.terrain import get_terrain
from engine.unit import Unit, UnitType, UNIT_STATS

# ---- terrain ids (engine/terrain.py) -------------------------------------
PLAIN           = 1
NEUTRAL_BASE    = 35

COLIN_CO_ID = 15
ANDY_CO_ID  = 1


def _state(*, p0_co_id: int = COLIN_CO_ID,
           p1_co_id: int = ANDY_CO_ID,
           weather: str = "clear",
           width: int = 5, height: int = 5) -> GameState:
    """Plains map with a single P0-owned base in the corner."""
    terrain = [[PLAIN] * width for _ in range(height)]
    terrain[0][0] = NEUTRAL_BASE
    properties = [PropertyState(
        terrain_id=NEUTRAL_BASE, row=0, col=0, owner=0, capture_points=20,
        is_hq=False, is_lab=False, is_comm_tower=False,
        is_base=True, is_airport=False, is_port=False,
    )]
    md = MapData(
        map_id=0, name="colin", map_type="std",
        terrain=terrain, height=height, width=width,
        cap_limit=99, unit_limit=50, unit_bans=[], tiers=[],
        objective_type=None, properties=properties,
        hq_positions={0: [], 1: []}, lab_positions={0: [], 1: []},
        country_to_player={},
    )
    state = GameState(
        map_data=md,
        units={0: [], 1: []},
        funds=[0, 0],
        co_states=[make_co_state_safe(p0_co_id), make_co_state_safe(p1_co_id)],
        properties=md.properties,
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
    )
    state.weather = weather
    return state


def _spawn(state: GameState, ut: UnitType, player: int,
           pos: tuple[int, int], *, unit_id: int, hp: int = 100) -> Unit:
    stats = UNIT_STATS[ut]
    u = Unit(
        unit_type=ut,
        player=player,
        hp=hp,
        ammo=stats.max_ammo,
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


# ===========================================================================
# 1. D2D −20 %% unit cost (parity with pre-existing Colin elif in _build_cost)
# ===========================================================================

class TestColinD2DCost(unittest.TestCase):
    """AWBW canon: Colin's units cost 80 %% of base (×0.8) on every build.
    Source: co.php Colin row + awbw.fandom.com/wiki/Colin (scrape §0.1)."""

    def test_colin_tank_costs_5600(self) -> None:
        # Tank list cost 7000 → Colin pays int(7000 * 0.8) = 5600.
        st = _state()
        cost = _build_cost(UnitType.TANK, st, player=0, pos=(0, 0))
        self.assertEqual(cost, 5600)


# ===========================================================================
# 2. D2D −10 %% attack rider (pure rider check, no full damage formula)
# ===========================================================================

class TestColinD2DAttackRider(unittest.TestCase):
    """AWBW canon: Colin's units lose −10 %% attack D2D (scrape §0.1)."""

    def test_d2d_rider_returns_minus_10(self) -> None:
        co = make_co_state_safe(COLIN_CO_ID)
        # No power active → pure D2D.
        self.assertEqual(_colin_atk_rider(co), -10)

    def test_non_colin_rider_returns_zero(self) -> None:
        # Andy must be untouched.
        co = make_co_state_safe(ANDY_CO_ID)
        self.assertEqual(_colin_atk_rider(co), 0)

    def test_d2d_attack_floors_at_90_pct_via_damage_calc(self) -> None:
        """Cross-check: identical Colin-vs-Andy and Andy-vs-Andy tank fights
        at the same luck roll — Colin attacker must deal 90 %% of Andy's
        attacker damage (within rounding). 0 luck pins determinism."""
        st_colin = _state(p0_co_id=COLIN_CO_ID, p1_co_id=ANDY_CO_ID)
        atk_colin = _spawn(st_colin, UnitType.TANK, 0, (0, 1), unit_id=1)
        def_colin = _spawn(st_colin, UnitType.TANK, 1, (0, 2), unit_id=2)

        st_andy = _state(p0_co_id=ANDY_CO_ID, p1_co_id=ANDY_CO_ID)
        atk_andy = _spawn(st_andy, UnitType.TANK, 0, (0, 1), unit_id=1)
        def_andy = _spawn(st_andy, UnitType.TANK, 1, (0, 2), unit_id=2)

        plains = get_terrain(PLAIN)
        d_colin = calculate_damage(
            atk_colin, def_colin, plains, plains,
            st_colin.co_states[0], st_colin.co_states[1], luck_roll=0,
        )
        d_andy = calculate_damage(
            atk_andy, def_andy, plains, plains,
            st_andy.co_states[0], st_andy.co_states[1], luck_roll=0,
        )
        # AV: Colin = 100 - 10 = 90; Andy = 100. Damage scales linearly with
        # AV in the AWBW formula (luck = 0 collapses the +L − LB term), so
        # ratio ≈ 0.9 modulo the ceil-to-0.05 / floor rounding.
        assert d_andy is not None and d_colin is not None and d_andy > 0
        ratio = d_colin / d_andy
        self.assertAlmostEqual(
            ratio, 0.9, delta=0.02,
            msg=f"Colin/Andy attacker tank-vs-tank damage ratio "
                f"{ratio:.3f} should be ≈ 0.90 (D2D −10 %%); "
                f"got Colin={d_colin}, Andy={d_andy}.",
        )


# ===========================================================================
# 3. COP "Gold Rush" — funds × 1.5 (round_half_up, PHP-canonical)
# ===========================================================================

class TestColinCopGoldRush(unittest.TestCase):
    """AWBW canon: funds × 1.5 on COP activation, round_half_up.
    Sources: co.php + awbw.fandom.com/wiki/Colin (scrape §0.2);
    rounding from PHP-payload drill (scrape §7.3, 15/15 envelopes)."""

    def test_gold_rush_funds_at_10000_yields_15000(self) -> None:
        # 10 000 × 1.5 = 15 000 (integer — no rounding ambiguity).
        st = _state()
        st.funds[0] = 10_000
        st._apply_power_effects(player=0, cop=True)
        self.assertEqual(st.funds[0], 15_000)

    def test_gold_rush_round_half_up_at_50835(self) -> None:
        # Anchor from scrape §7.3: zip 1637153 env 38, pre=50835, payload=76253.
        # int(50835 * 1.5) = 76 252 (WRONG); round_half_up = 76 253 (CANON).
        st = _state()
        st.funds[0] = 50_835
        st._apply_power_effects(player=0, cop=True)
        self.assertEqual(
            st.funds[0], 76_253,
            "Gold Rush must use round_half_up (PHP canon, scrape §7.3 "
            "1637153 env 38: pre=50835 → payload=76253). int() floor "
            "would give 76252 and silently desync ~20 %% of COP fires.",
        )

    def test_gold_rush_round_half_up_at_7777(self) -> None:
        # SHIP order originally specified int() = 11 665; canon override
        # documented in phase11j_colin_impl_ship.md Section "Canon overrides".
        # 7 777 × 1.5 = 11 665.5 → round_half_up = 11 666.
        st = _state()
        st.funds[0] = 7_777
        st._apply_power_effects(player=0, cop=True)
        self.assertEqual(st.funds[0], 11_666)


# ===========================================================================
# 4. COP has NO attack modifier (Gold Rush is funds-only)
# ===========================================================================

class TestColinCopHasNoAttackBonus(unittest.TestCase):
    """AWBW canon: COP "Gold Rush" payout is funds-only — D2D −10 %%
    persists, no extra ATK rider. Net AV during COP = -10 (this rider) +
    +10 (universal SCOPB via COState.cop_atk_modifier) = 0 vs base 100
    (i.e. ≈ 100 %% effective attack, matching scrape §0.4 wording)."""

    def test_cop_active_rider_still_minus_10(self) -> None:
        co = make_co_state_safe(COLIN_CO_ID)
        co.cop_active = True
        self.assertEqual(
            _colin_atk_rider(co), -10,
            "Colin COP (Gold Rush) must NOT add an attack bonus — only the "
            "funds × 1.5 payout. D2D −10 %% rider persists during COP.",
        )


# ===========================================================================
# 5. SCOP "Power of Money" formula at 50 000 funds → 2.5× attack vs base
# ===========================================================================

class TestColinScopPowerOfMoney(unittest.TestCase):
    """AWBW canon: + (3 * Funds / 1000) %% attack on all units during SCOP.
    Funds snapshotted at activation; rider stacks with D2D −10 %% AND the
    universal +10 %% SCOPB rider.

    At 50 000 funds, the three components conveniently cancel/stack to
    exactly 2.5× base: −10 (D2D) + 10 (SCOPB) + 150 (PoM) = +150 AV →
    av = 250 → 2.5× damage scaling vs base AV = 100."""

    def test_scop_snapshot_recorded_at_50000(self) -> None:
        st = _state()
        st.funds[0] = 50_000
        st._apply_power_effects(player=0, cop=False)
        self.assertEqual(st.co_states[0].colin_pom_funds_snapshot, 50_000)

    def test_scop_rider_at_50000_funds_returns_140(self) -> None:
        # Rider returns -10 (D2D) + int(3 * 50000 / 1000) = -10 + 150 = 140.
        # SCOPB +10 is added separately by COState.cop_atk_modifier so the
        # net AV delta vs Andy at base 100 is exactly +150 → av=250 → 2.5×.
        co = make_co_state_safe(COLIN_CO_ID)
        co.scop_active = True
        co.colin_pom_funds_snapshot = 50_000
        self.assertEqual(_colin_atk_rider(co), 140)

    def test_scop_2_5x_damage_ratio_vs_andy(self) -> None:
        """End-to-end: Colin SCOP @ 50k vs Andy baseline → ≈ 2.5× damage.

        Note: ``_apply_power_effects`` only snapshots the funds; the
        ``scop_active`` flag is set by ``_activate_power`` (its caller).
        We replicate the activation contract here by setting both."""
        st_colin = _state(p0_co_id=COLIN_CO_ID, p1_co_id=ANDY_CO_ID)
        st_colin.funds[0] = 50_000
        st_colin._apply_power_effects(player=0, cop=False)
        st_colin.co_states[0].scop_active = True  # set by _activate_power
        atk_colin = _spawn(st_colin, UnitType.TANK, 0, (0, 1), unit_id=1)
        def_colin = _spawn(st_colin, UnitType.TANK, 1, (0, 2), unit_id=2)

        st_andy = _state(p0_co_id=ANDY_CO_ID, p1_co_id=ANDY_CO_ID)
        atk_andy = _spawn(st_andy, UnitType.TANK, 0, (0, 1), unit_id=1)
        def_andy = _spawn(st_andy, UnitType.TANK, 1, (0, 2), unit_id=2)

        plains = get_terrain(PLAIN)
        d_colin = calculate_damage(
            atk_colin, def_colin, plains, plains,
            st_colin.co_states[0], st_colin.co_states[1], luck_roll=0,
        )
        d_andy = calculate_damage(
            atk_andy, def_andy, plains, plains,
            st_andy.co_states[0], st_andy.co_states[1], luck_roll=0,
        )
        # Colin SCOP @ 50k: AV = 100 - 10 (D2D) + 10 (SCOPB) + 150 (PoM) = 250.
        # Andy baseline:    AV = 100.
        # Damage scales linearly with AV (luck=0 collapses + L − LB).
        assert d_andy is not None and d_colin is not None and d_andy > 0
        ratio = d_colin / d_andy
        self.assertAlmostEqual(
            ratio, 2.5, delta=0.05,
            msg=f"Colin SCOP @ 50k attacker damage ratio vs Andy = "
                f"{ratio:.3f}, expected ≈ 2.50; got Colin={d_colin}, "
                f"Andy={d_andy}.",
        )


# ===========================================================================
# 6. SCOP at low funds (1 000) → ≈ 1.03× (rider returns −10 + 3 = −7)
# ===========================================================================

class TestColinScopLowFunds(unittest.TestCase):
    """AWBW canon: at low funds, PoM bonus is small. 1 000 funds →
    int(3 * 1000 / 1000) = 3 AV; rider returns -10 + 3 = -7 AV; with
    SCOPB +10 the net delta vs base is +3 → av=103 → ≈ 1.03×."""

    def test_scop_rider_at_1000_funds_returns_minus_7(self) -> None:
        co = make_co_state_safe(COLIN_CO_ID)
        co.scop_active = True
        co.colin_pom_funds_snapshot = 1_000
        self.assertEqual(_colin_atk_rider(co), -7)


# ===========================================================================
# 7. Cost discount unaffected by weather (snow / rain / sandstorm)
# ===========================================================================

class TestColinCostDiscountWeatherIndependent(unittest.TestCase):
    """AWBW canon: build cost is purely a CO-side modifier (Kanbei +20 %%,
    Colin −20 %%, Hachi −10 %%). Weather affects movement / terrain
    defense, not list price. Pin the invariant explicitly per the SHIP
    order's coordination notice with L1-WAVE-2 (which may add other CO
    cost branches but must not introduce weather-coupled cost math)."""

    def test_colin_tank_cost_5600_in_snow(self) -> None:
        st = _state(weather="snow")
        cost = _build_cost(UnitType.TANK, st, player=0, pos=(0, 0))
        self.assertEqual(cost, 5600)

    def test_colin_tank_cost_5600_in_rain(self) -> None:
        st = _state(weather="rain")
        cost = _build_cost(UnitType.TANK, st, player=0, pos=(0, 0))
        self.assertEqual(cost, 5600)


# ===========================================================================
# 8. COP integer floor at .5 boundary (canon override of SHIP order)
# ===========================================================================

class TestColinCopRoundingBoundary(unittest.TestCase):
    """SHIP order originally requested ``int(7777 * 1.5) == 11665``. The
    PHP-payload drill (scrape §7.3, 15 / 15 sub=0 COP envelopes) proved
    AWBW uses **round_half_up**, not floor. Anchors:

      pre=50 835 → payload=76 253  (zip 1637153 env 38)
      pre=48 533 → payload=72 800  (zip 1637153 env 44)
      pre=23 331 → payload=34 997  (zip 1619141 env 35)

    All three .5-boundary cases match round_half_up. Implementation uses
    pure integer ``(3 * pre + 1) // 2``. This test pins all three PHP
    anchors so any future regression to ``int()`` floor surfaces here.
    """

    def test_payload_anchor_50835(self) -> None:
        st = _state()
        st.funds[0] = 50_835
        st._apply_power_effects(player=0, cop=True)
        self.assertEqual(st.funds[0], 76_253)

    def test_payload_anchor_48533(self) -> None:
        st = _state()
        st.funds[0] = 48_533
        st._apply_power_effects(player=0, cop=True)
        self.assertEqual(st.funds[0], 72_800)

    def test_payload_anchor_23331(self) -> None:
        st = _state()
        st.funds[0] = 23_331
        st._apply_power_effects(player=0, cop=True)
        self.assertEqual(st.funds[0], 34_997)

    def test_999999_funds_cap(self) -> None:
        # Engine-universal cap: funds may not exceed 999 999 (matches
        # GameState._grant_income clamp). 800 000 × 1.5 = 1 200 000 →
        # clamped to 999 999.
        st = _state()
        st.funds[0] = 800_000
        st._apply_power_effects(player=0, cop=True)
        self.assertEqual(st.funds[0], 999_999)


if __name__ == "__main__":
    unittest.main()
