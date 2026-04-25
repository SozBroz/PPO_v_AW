"""Phase 11J-FUNDS-SHIP — start-of-turn ordering and per-unit repair canon.

Three engine rules ship together in this phase, anchored to the AWBW canon
documented in ``docs/oracle_exception_audit/phase11j_funds_deep.md`` and
``docs/oracle_exception_audit/phase11j_f2_koal_fu_oracle_funds.md``:

* **R1 — Income before property-day repair.** ``GameState._end_turn`` now
  calls ``_grant_income(opponent)`` BEFORE ``_resupply_on_properties(opponent)``.
  The user-confirmed AWBW canon (Phase 11J-FUNDS-SHIP, Imperator) plus the
  empirical 69/69 Tier-3 PHP-snapshot match across the 100-game corpus
  (``phase11j_funds_corpus_derivation.md`` §3) and 37/39 NEITHER rows under
  the IBR hypothetical (``phase11j_funds_deep.md`` §3.4) settle the order.

* **R2 — All-or-nothing per-unit repair.** ``_resupply_on_properties`` now
  computes the FULL step (+20 internal, or +30 for Rachel co_id 28) and
  either pays for it in full or skips the unit entirely. No partial heals.
  Three Tier-2 AWBW Wiki citations (Units / Advance Wars Overview /
  Black-Boat — see ``phase11j_funds_deep.md`` §4 R2).

* **R3 — Deterministic iteration order: ``(prop.col, prop.row)`` ascending
  (column-major-from-left).** Required to keep the Rachel game ``1622501``
  green under R1 + R2 — the engine and PHP would otherwise pick different
  units to heal when the treasury exactly straddles one full-step's cost.
  Tier-4 supporting source (RPGHQ AWBW Q&A — *"Repair priority is checked
  by columns (top to bottom) starting from the left."*); see
  ``phase11j_f2_koal_fu_oracle_funds.md`` §2 Tier-4 supporting note.

These tests are the canonical regression fixture for the FUNDS-SHIP bundle.
If any of them fails, R1/R2/R3 is broken (or has been re-litigated and
needs a new ship plan). Two pre-existing partial-heal asserts were
canon-aligned in the same phase: ``tests/test_capture_terrain.py::
test_property_day_repair_respects_insufficient_funds`` and
``tests/test_co_repair_rachel.py::TestRachelDayRepair::
test_rachel_all_or_nothing_under_budget``.
"""

from __future__ import annotations

import unittest

from engine.action import ActionStage
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData, PropertyState
from engine.unit import UNIT_STATS, Unit, UnitType


# Terrain ids (engine/terrain.py).
NEUTRAL_CITY = 34

# CO ids per data/co_data.json.
CO_ANDY   = 1
CO_RACHEL = 28


def _make_state(
    *,
    width: int,
    prop_owner: int,
    co_id_p0: int = CO_ANDY,
    co_id_p1: int = CO_ANDY,
    funds_p0: int = 0,
    funds_p1: int = 0,
    active_player: int = 0,
    units_p0: list[Unit] | None = None,
    units_p1: list[Unit] | None = None,
) -> GameState:
    """Build a 1×width map of neutral-city tiles, all owned by ``prop_owner``.

    Single row (row=0), columns 0..width-1. Every tile is a city (no flags
    set ⇒ ``is_city`` derivation in ``_resupply_on_properties``). Useful for
    income (count_income_properties = width) and per-unit repair tests.
    """
    properties = [
        PropertyState(
            terrain_id=NEUTRAL_CITY, row=0, col=c, owner=prop_owner,
            capture_points=20,
            is_hq=False, is_lab=False, is_comm_tower=False,
            is_base=False, is_airport=False, is_port=False,
        )
        for c in range(width)
    ]
    map_data = MapData(
        map_id=0, name="funds-ship", map_type="std",
        terrain=[[NEUTRAL_CITY] * width], height=1, width=width,
        cap_limit=99, unit_limit=50, unit_bans=[], tiers=[],
        objective_type=None, properties=properties,
        hq_positions={0: [], 1: []}, lab_positions={0: [], 1: []},
        country_to_player={},
    )
    return GameState(
        map_data=map_data,
        units={0: list(units_p0 or []), 1: list(units_p1 or [])},
        funds=[funds_p0, funds_p1],
        co_states=[make_co_state_safe(co_id_p0), make_co_state_safe(co_id_p1)],
        properties=map_data.properties,
        turn=9,
        active_player=active_player,
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


def _mk_unit(
    *,
    unit_type: UnitType,
    player: int,
    hp: int,
    pos: tuple[int, int],
    unit_id: int = 1,
) -> Unit:
    stats = UNIT_STATS[unit_type]
    return Unit(
        unit_type=unit_type, player=player, hp=hp,
        ammo=stats.max_ammo, fuel=stats.max_fuel,
        pos=pos, moved=False, loaded_units=[],
        is_submerged=False, capture_progress=20, unit_id=unit_id,
    )


# ---------------------------------------------------------------------------
# R1 — income before property-day repair
# ---------------------------------------------------------------------------

class TestR1IncomeBeforeRepair(unittest.TestCase):
    """``_end_turn`` must grant income BEFORE running the heal pass.

    Pre-FUNDS-SHIP order (RBI) ran the heal pass on pre-income funds, which
    typically meant 0g — the partial-heal loop then silently healed nothing
    even though the player would have had budget to spare after income.
    """

    def test_end_turn_runs_income_then_heal_for_opponent(self) -> None:
        """P0 ends turn → opponent (P1) collects income, then heals.

        Setup: 14 P1-owned cities (income = +14_000g). One P1 INFANTRY at
        internal HP 40 (display 4) sits on city col=0. P1 funds = 0 before
        ``_end_turn``. Expected post-``_end_turn``:
          * P1 funds = 14_000 (income) − 200 (full +20 INF heal) = 13_800.
          * INFANTRY internal HP 40 → 60 (display 4 → 6).

        Under the pre-FUNDS-SHIP order the heal pass would run with 0g,
        the partial loop would heal nothing, and final P1 funds would be
        14_000 with INFANTRY still at HP 40. The post-FUNDS-SHIP order
        produces the canonical AWBW outcome.
        """
        inf = _mk_unit(unit_type=UnitType.INFANTRY, player=1, hp=40, pos=(0, 0))
        state = _make_state(
            width=14, prop_owner=1,
            funds_p0=0, funds_p1=0,
            active_player=0,  # opponent = 1 → P1 receives income/heal
            units_p1=[inf],
        )
        self.assertEqual(state.count_income_properties(1), 14)

        state._end_turn()

        self.assertEqual(state.active_player, 1, "Active flips to opponent.")
        self.assertEqual(
            state.funds[1], 13_800,
            "P1 funds = 14_000 income − 200 INF heal cost (income FIRST).",
        )
        self.assertEqual(
            inf.hp, 60,
            "INFANTRY heals +20 internal (display 4 → 6) once income is in "
            "the bank — pre-FUNDS-SHIP order would have left it at 40.",
        )


# ---------------------------------------------------------------------------
# R2 — all-or-nothing per-unit repair
# ---------------------------------------------------------------------------

class TestR2AllOrNothing(unittest.TestCase):
    """A unit either gets the full +20 (or +30 Rachel) heal or no heal at all.

    Three Tier-2 AWBW Wiki citations (Units / Advance Wars Overview /
    Black-Boat) anchor the rule. Pre-FUNDS-SHIP engine carried a
    ``while h > 0: h -= 1`` decrement loop that quietly healed a smaller
    increment to fit budget — that loop is gone.
    """

    def test_single_unit_underbudget_skipped_entirely(self) -> None:
        """TANK HP 70 with funds 1000 < 1400 full-step cost → no heal, no charge."""
        tank = _mk_unit(unit_type=UnitType.TANK, player=0, hp=70, pos=(0, 0))
        state = _make_state(
            width=1, prop_owner=0, funds_p0=1000, units_p0=[tank],
        )

        state._resupply_on_properties(0)

        self.assertEqual(
            tank.hp, 70,
            "Cannot afford +20 step (cost 1400 > funds 1000) → no heal.",
        )
        self.assertEqual(
            state.funds[0], 1000,
            "Treasury must be untouched on the all-or-nothing skip path.",
        )

    def test_two_units_straddle_treasury_first_heals_second_skipped(self) -> None:
        """Funds 1500, two TANKs HP 70: one full heal (1400g), one skipped.

        One full +20 step costs 1400g. The first eligible unit by the
        column-major iteration order pays 1400 and heals fully (HP 70 →
        90). The second unit cannot afford the next 1400 (residual funds
        100) and is skipped — NOT partial-healed.
        """
        # Both TANKs on the same row; sort key is (col, row) so col=0 wins.
        tank_a = _mk_unit(
            unit_type=UnitType.TANK, player=0, hp=70, pos=(0, 0), unit_id=1,
        )
        tank_b = _mk_unit(
            unit_type=UnitType.TANK, player=0, hp=70, pos=(0, 1), unit_id=2,
        )
        state = _make_state(
            width=2, prop_owner=0, funds_p0=1500,
            units_p0=[tank_a, tank_b],
        )

        state._resupply_on_properties(0)

        healed = [t for t in (tank_a, tank_b) if t.hp == 90]
        skipped = [t for t in (tank_a, tank_b) if t.hp == 70]
        self.assertEqual(
            len(healed), 1,
            "Exactly ONE tank heals fully (+20 internal) at this treasury.",
        )
        self.assertEqual(
            len(skipped), 1,
            "Exactly ONE tank is skipped entirely — no partial heal.",
        )
        self.assertEqual(
            state.funds[0], 100,
            "Funds = 1500 − 1400 (one full heal) = 100; skipped tank is free.",
        )


# ---------------------------------------------------------------------------
# R3 — deterministic iteration order: (col, row) ascending
# ---------------------------------------------------------------------------

class TestR3SortOrder(unittest.TestCase):
    """Eligible units iterated in column-major-from-left order.

    Sort key: ``(prop.col, prop.row)`` ascending — column first (left → right),
    then row within column (top → bottom). Tier-4 supporting source: RPGHQ
    AWBW Q&A. Required for the Rachel game ``1622501`` not to regress on
    R1 + R2 alone (see ``phase11j_funds_deep.md`` §6).
    """

    def test_lower_col_unit_heals_first_when_treasury_straddles(self) -> None:
        """Two TANKs HP 70, funds 1400. Tank at (col=1, row=8) heals first.

        Plan-test mapping: position notation is ``(prop_x, prop_y) =
        (col, row)``. Tank A at (col=3, row=5), Tank B at (col=1, row=8).
        Column-major-from-left ⇒ B (col=1) precedes A (col=3) regardless
        of row. With funds = 1400 (exactly one full +20 TANK step),
        B heals fully, A is skipped.
        """
        # Need a wider map so cols 1 and 3 both exist; row 8 needs height ≥ 9.
        # Build manually here rather than use the 1×width helper.
        properties = [
            PropertyState(
                terrain_id=NEUTRAL_CITY, row=5, col=3, owner=0,
                capture_points=20,
                is_hq=False, is_lab=False, is_comm_tower=False,
                is_base=False, is_airport=False, is_port=False,
            ),
            PropertyState(
                terrain_id=NEUTRAL_CITY, row=8, col=1, owner=0,
                capture_points=20,
                is_hq=False, is_lab=False, is_comm_tower=False,
                is_base=False, is_airport=False, is_port=False,
            ),
        ]
        # Map needs to span the property coordinates; fill terrain with cities.
        height, width = 9, 4
        terrain = [[NEUTRAL_CITY] * width for _ in range(height)]
        map_data = MapData(
            map_id=0, name="sort-order", map_type="std",
            terrain=terrain, height=height, width=width,
            cap_limit=99, unit_limit=50, unit_bans=[], tiers=[],
            objective_type=None, properties=properties,
            hq_positions={0: [], 1: []}, lab_positions={0: [], 1: []},
            country_to_player={},
        )
        tank_a = _mk_unit(
            unit_type=UnitType.TANK, player=0, hp=70, pos=(5, 3), unit_id=1,
        )
        tank_b = _mk_unit(
            unit_type=UnitType.TANK, player=0, hp=70, pos=(8, 1), unit_id=2,
        )
        state = GameState(
            map_data=map_data,
            units={0: [tank_a, tank_b], 1: []},
            funds=[1400, 0],
            co_states=[make_co_state_safe(CO_ANDY), make_co_state_safe(CO_ANDY)],
            properties=map_data.properties,
            turn=9,
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

        state._resupply_on_properties(0)

        self.assertEqual(
            tank_b.hp, 90,
            "TANK at (col=1, row=8) heals first by column-major sort.",
        )
        self.assertEqual(
            tank_a.hp, 70,
            "TANK at (col=3, row=5) is skipped — treasury exhausted.",
        )
        self.assertEqual(
            state.funds[0], 0,
            "Funds = 1400 − 1400 (single full heal) = 0.",
        )

    def test_lower_col_wins_over_lower_row(self) -> None:
        """Order test asserts col precedes row, not the other way around.

        TANK A at (col=2, row=0) and TANK B at (col=0, row=2). If sort
        were ``(row, col)`` instead of ``(col, row)``, A would win on
        row=0; under the canon-correct ``(col, row)`` sort, B wins on
        col=0. Funds = 1400 → exactly one heal.
        """
        properties = [
            PropertyState(
                terrain_id=NEUTRAL_CITY, row=0, col=2, owner=0,
                capture_points=20,
                is_hq=False, is_lab=False, is_comm_tower=False,
                is_base=False, is_airport=False, is_port=False,
            ),
            PropertyState(
                terrain_id=NEUTRAL_CITY, row=2, col=0, owner=0,
                capture_points=20,
                is_hq=False, is_lab=False, is_comm_tower=False,
                is_base=False, is_airport=False, is_port=False,
            ),
        ]
        height, width = 3, 3
        terrain = [[NEUTRAL_CITY] * width for _ in range(height)]
        map_data = MapData(
            map_id=0, name="sort-order-axis", map_type="std",
            terrain=terrain, height=height, width=width,
            cap_limit=99, unit_limit=50, unit_bans=[], tiers=[],
            objective_type=None, properties=properties,
            hq_positions={0: [], 1: []}, lab_positions={0: [], 1: []},
            country_to_player={},
        )
        tank_a = _mk_unit(
            unit_type=UnitType.TANK, player=0, hp=70, pos=(0, 2), unit_id=1,
        )
        tank_b = _mk_unit(
            unit_type=UnitType.TANK, player=0, hp=70, pos=(2, 0), unit_id=2,
        )
        state = GameState(
            map_data=map_data,
            units={0: [tank_a, tank_b], 1: []},
            funds=[1400, 0],
            co_states=[make_co_state_safe(CO_ANDY), make_co_state_safe(CO_ANDY)],
            properties=map_data.properties,
            turn=9,
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

        state._resupply_on_properties(0)

        self.assertEqual(
            tank_b.hp, 90,
            "TANK at (col=0, row=2) heals first — col precedes row.",
        )
        self.assertEqual(tank_a.hp, 70, "TANK at (col=2, row=0) skipped.")


# ---------------------------------------------------------------------------
# Rachel R1 + R2 combined
# ---------------------------------------------------------------------------

class TestRachelR1R2Combined(unittest.TestCase):
    """Rachel CO 28 — +30 internal step at 30% of unit cost is all-or-nothing.

    Rachel D2D pays 30% of deployment cost for one full +30 step (display
    +3 bars). With funds tight enough to cover exactly one Rachel TANK
    step (2100g), only the first eligible unit by column-major sort heals;
    the rest are skipped.
    """

    def test_rachel_two_tanks_funds_for_one_step(self) -> None:
        """Two Rachel TANKs HP 70, funds 2100. First by sort heals fully."""
        tank_a = _mk_unit(
            unit_type=UnitType.TANK, player=0, hp=70, pos=(0, 0), unit_id=1,
        )
        tank_b = _mk_unit(
            unit_type=UnitType.TANK, player=0, hp=70, pos=(0, 1), unit_id=2,
        )
        state = _make_state(
            width=2, prop_owner=0,
            co_id_p0=CO_RACHEL, funds_p0=2100,
            units_p0=[tank_a, tank_b],
        )

        state._resupply_on_properties(0)

        healed = [t for t in (tank_a, tank_b) if t.hp == 100]
        skipped = [t for t in (tank_a, tank_b) if t.hp == 70]
        self.assertEqual(
            len(healed), 1,
            "Exactly ONE Rachel TANK heals +30 (HP 70 → 100, cost 2100g).",
        )
        self.assertEqual(
            len(skipped), 1,
            "The second Rachel TANK is skipped — all-or-nothing canon.",
        )
        self.assertEqual(
            state.funds[0], 0,
            "Funds = 2100 − 2100 (one Rachel full step) = 0.",
        )

    def test_rachel_end_turn_grants_income_then_full_step(self) -> None:
        """R1 + R2 + Rachel together: P1 (Rachel) starts day with funds 0.

        14 Rachel-owned cities (income +14_000) + one Rachel TANK at HP 70
        on col=0. After ``_end_turn`` (P0 ends → P1 receives income/heal):
          * P1 income = 14_000.
          * Rachel +30 step on a TANK costs 2100g.
          * Final P1 funds = 14_000 − 2100 = 11_900, TANK HP 70 → 100.
        """
        tank = _mk_unit(unit_type=UnitType.TANK, player=1, hp=70, pos=(0, 0))
        state = _make_state(
            width=14, prop_owner=1,
            co_id_p1=CO_RACHEL,
            funds_p0=0, funds_p1=0,
            active_player=0,
            units_p1=[tank],
        )

        state._end_turn()

        self.assertEqual(state.active_player, 1)
        self.assertEqual(
            tank.hp, 100,
            "Rachel TANK heals full +30 step once income is granted.",
        )
        self.assertEqual(
            state.funds[1], 11_900,
            "P1 funds = 14_000 income − 2_100 Rachel TANK heal cost.",
        )


# ---------------------------------------------------------------------------
# R4 — display-cap repair cost canon (non-Rachel)
# ---------------------------------------------------------------------------

class TestR4DisplayCapRepairCanon(unittest.TestCase):
    """Non-Rachel D2D repair charges by DISPLAY HP, not internal HP.

    Per the AWBW Wiki "Units" article (already cited in
    ``_resupply_on_properties``):

        *"If a unit is not specifically at 9HP, repair costs will be
        calculated only in increments of 2HP."*

    Combined with the AWBW Wiki "Changes_in_AWBW" rule

        *"Repairs will only take place in increments of exactly 20
        hitpoints, or 2 full visual hitpoints."*

    The PHP repair canon is:

      * display 10 (internal 91–99)         : NO REPAIR, charge 0g
      * display  9 (internal 81–90)         : +1 display (+10 internal),
                                              cost = 10% of unit cost
      * display 1–8 (internal 1–80)         : +2 display (+20 internal),
                                              cost = 20% of unit cost

    Pre-R4 the engine charged ``min(20, 100-hp)`` × ``unit_cost / 100``,
    which produced phantom partial repairs at internal HP 91–99 and
    over-charged at internal HP 81–90 (e.g. TANK at HP 85 paid 1050g for
    +15 internal, canon is 700g for +10 internal). Phase
    11J-BUILD-NO-OP-CLUSTER-CLOSE — six oracle_gap closures.

    Recon: ``docs/oracle_exception_audit/phase11j_build_no_op_cluster_close.md``.
    """

    def test_display_10_internal_91_to_99_skipped_no_charge(self) -> None:
        """TANK HP 94 (display 10) → no repair, no charge.

        Pre-R4 the engine charged ``(6 * 7000) // 100 = 420g`` and healed
        the unit to internal HP 100 (still display 10, no visible change).
        AWBW PHP refuses the repair: the bar is already maxed.
        """
        tank = _mk_unit(unit_type=UnitType.TANK, player=0, hp=94, pos=(0, 0))
        state = _make_state(
            width=1, prop_owner=0, funds_p0=10_000, units_p0=[tank],
        )

        state._resupply_on_properties(0)

        self.assertEqual(
            tank.hp, 94,
            "Display 10 (internal 91-99) must NOT be healed — bar is full.",
        )
        self.assertEqual(
            state.funds[0], 10_000,
            "Treasury must be untouched on the display-10 skip path.",
        )

    def test_display_9_internal_85_heals_plus_one_display_at_ten_percent(self) -> None:
        """TANK HP 85 (display 9) → +10 internal HP, cost 700g (10%).

        Pre-R4 the engine charged ``(15 * 7000) // 100 = 1050g`` and
        healed +15 internal HP. AWBW PHP heals exactly one display bar
        (+10 internal HP) at 10% of unit cost.
        """
        tank = _mk_unit(unit_type=UnitType.TANK, player=0, hp=85, pos=(0, 0))
        state = _make_state(
            width=1, prop_owner=0, funds_p0=5000, units_p0=[tank],
        )

        state._resupply_on_properties(0)

        self.assertEqual(
            tank.hp, 95,
            "Display 9 → +10 internal HP (display 9 → 10), not +15.",
        )
        self.assertEqual(
            state.funds[0], 5000 - 700,
            "Cost = 10% of 7000 = 700g (NOT 15% = 1050g).",
        )

    def test_display_9_infantry_hp_88_charges_one_hundred(self) -> None:
        """INFANTRY HP 88 (display 9) → +10 internal HP, cost 100g.

        Per gid 1632289 (Andy/Sonja) trace: pre-R4 engine charged
        ``(12 * 1000) // 100 = 120g`` and healed +12 internal. PHP
        charges 100g for +10 internal — drift = +20g over-charge.
        """
        inf = _mk_unit(unit_type=UnitType.INFANTRY, player=0, hp=88, pos=(0, 0))
        state = _make_state(
            width=1, prop_owner=0, funds_p0=500, units_p0=[inf],
        )

        state._resupply_on_properties(0)

        self.assertEqual(
            inf.hp, 98,
            "Display 9 INFANTRY heals +10 internal HP (88 → 98).",
        )
        self.assertEqual(
            state.funds[0], 400,
            "Cost = 10% of 1000 = 100g (NOT 12% = 120g).",
        )

    def test_display_7_tank_two_bar_step_twenty_percent(self) -> None:
        """TANK HP 65 (display 7) → +20 internal HP, cost 1400g (20%).

        Display-8 non-decile band (71–80 internal) uses one +10 tick first
        (10%); lower displays still take the full two-bar +20 step when
        affordable (R2 all-or-nothing on that computed step).
        """
        tank = _mk_unit(unit_type=UnitType.TANK, player=0, hp=65, pos=(0, 0))
        state = _make_state(
            width=1, prop_owner=0, funds_p0=5000, units_p0=[tank],
        )

        state._resupply_on_properties(0)

        self.assertEqual(tank.hp, 85, "Display 7 TANK heals +20 internal HP (65 → 85).")
        self.assertEqual(
            state.funds[0], 5000 - 1400,
            "Cost = 20% of 7000 = 1400g (full two-bar step).",
        )

    def test_display_8_internal_73_one_bar_ten_percent_gid1624307(self) -> None:
        """TANK HP 73 (ceil display 8, non-decile) → +10 internal, 700g (10%).

        PHP property-day repair charges the one-bar rate first for 71–80
        internal HP (display 8); the prior engine path treated all display-8
        units as +20 / 20% only (gid 1624307 env 36).
        """
        tank = _mk_unit(unit_type=UnitType.TANK, player=0, hp=73, pos=(0, 0))
        state = _make_state(
            width=1, prop_owner=0, funds_p0=5000, units_p0=[tank],
        )

        state._resupply_on_properties(0)

        self.assertEqual(tank.hp, 83)
        self.assertEqual(
            state.funds[0], 5000 - 700,
            "Cost = 10% of 7000 = 700g (one +10 internal tick).",
        )

    def test_display_8_internal_80_one_bar_ten_percent(self) -> None:
        """TANK HP 80 (ceil display 8) → +10 internal, 700g (10%).

        Upper end of the display-8 band matches PHP one-tick-first (gid 1624307
        class); must not demand a single +20 / 20% step for the whole band.
        """
        tank = _mk_unit(unit_type=UnitType.TANK, player=0, hp=80, pos=(0, 0))
        state = _make_state(
            width=1, prop_owner=0, funds_p0=5000, units_p0=[tank],
        )

        state._resupply_on_properties(0)

        self.assertEqual(tank.hp, 100, "80 → 90 → 100 (71–80 band + chained display-9 tick).")
        self.assertEqual(state.funds[0], 5000 - 700 - 700)

    def test_rachel_display_10_no_heal_matches_php(self) -> None:
        """Rachel display-10 units do NOT heal — matches the non-Rachel R4 cap.

        Phase 11J-FINAL-LASTMILE moved Rachel onto the same display-bar
        canon as standard COs: a Rachel TANK at internal HP 94 has
        display HP 10 (the bar is already maxed in the AWBW UI), so PHP
        refuses repair and charges 0g. The pre-LASTMILE legacy path
        charged ``(100 - hp)% * unit_cost = 6% * 7000 = 420g`` for a
        +6 internal phantom heal that never appeared on the bar; that
        path is gone. Anchor: gid 1607045 closeout
        (``docs/oracle_exception_audit/phase11j_final_lastmile_v2.md``).
        """
        tank = _mk_unit(unit_type=UnitType.TANK, player=0, hp=94, pos=(0, 0))
        state = _make_state(
            width=1, prop_owner=0,
            co_id_p0=CO_RACHEL, funds_p0=5000,
            units_p0=[tank],
        )

        state._resupply_on_properties(0)

        self.assertEqual(
            tank.hp, 94,
            "Rachel TANK at display 10 is bar-maxed; PHP refuses repair.",
        )
        self.assertEqual(
            state.funds[0], 5000,
            "Display-10 repair cost = 0g (canon parity with non-Rachel branch).",
        )


if __name__ == "__main__":
    unittest.main()
