"""Rachel (CO 28) day-property repair canon parity.

Sources (all aligned; PHP wins on disagreement per Phase 11A Kindle precedent):

* **AWBW CO Chart** https://awbw.amarriner.com/co.php — Rachel row:
  *"Units repair +1 additional HP (note: liable for costs)."*
* **AWBW Fandom Wiki — Rachel** https://awbw.fandom.com/wiki/Rachel —
  Day-to-Day: *"Units repair +1 additional HP on properties (note: liable
  for costs)."* (The ``amarriner.com/wiki/`` path returns HTTP 404; the
  Fandom wiki is the canonical community wiki.)
* **AWBW Fandom Wiki — Changes in AWBW** https://awbw.fandom.com/wiki/Changes_in_AWBW
  — *"Repairs will only take place in increments of exactly 20 hitpoints,
  or 2 full visual hitpoints."* Combined with Rachel's +1 ⇒ exactly
  **+30 internal HP** (+3 visual bars).
* **Advance Wars Wiki — Repairing** https://advancewars.fandom.com/wiki/Repairing
  — *"10% cost per 10% health, or 1HP."* ⇒ +30 internal HP costs 30% of
  deployment cost.
* **PHP snapshot cross-check** (``tools/_phase11y_rachel_php_check.py``):
  43 of 48 positive Rachel heal events on properties show exactly +3 bars
  across 7 Rachel-bearing zips; the remaining 5 are HP-cap or post-heal
  combat. Andy control on 5 zips: 39 of 39 positive heals = +2 bars (no
  Rachel-bonus drift on standard COs).

Phase 11Y-RACHEL-IMPL bumps the per-property-day heal from +20 internal HP
(+2 bars) to +30 (+3 bars) when ``co_id == 28``. ``_property_day_repair_gold``
is linear in internal HP, so a full +30 step naturally costs 30% of the
unit's deployment cost — Rachel pays for the extra bar exactly as the
chart, wiki, and PHP all confirm. ``data/co_data.json`` only mentions luck
for Rachel and is **not** trusted (see
``docs/oracle_exception_audit/phase11y_co_wave_2.md`` §5).
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
NEUTRAL_BASE = 35


# CO ids per data/co_data.json.
CO_ANDY   = 1
CO_RACHEL = 28


def _make_state(
    *,
    co_id_p0: int,
    terrain_id: int,
    unit_type: UnitType,
    unit_hp: int,
    funds_p0: int = 100_000,
    prop_owner: int = 0,
) -> tuple[GameState, Unit]:
    """1x1 fixture: P0 unit on a single tile owned by ``prop_owner``.

    The tile's flag-set is derived from ``terrain_id``: 35 = base, 34 = city
    (no flag set ⇒ ``is_city`` derivation in ``_resupply_on_properties``).
    Returns the state plus a handle to the placed unit.
    """
    is_base = (terrain_id == NEUTRAL_BASE)
    properties = [PropertyState(
        terrain_id=terrain_id, row=0, col=0, owner=prop_owner, capture_points=20,
        is_hq=False, is_lab=False, is_comm_tower=False,
        is_base=is_base, is_airport=False, is_port=False,
    )]
    map_data = MapData(
        map_id=0, name="rachel-repair", map_type="std",
        terrain=[[terrain_id]], height=1, width=1,
        cap_limit=99, unit_limit=50, unit_bans=[], tiers=[],
        objective_type=None, properties=properties,
        hq_positions={0: [], 1: []}, lab_positions={0: [], 1: []},
        country_to_player={},
    )
    stats = UNIT_STATS[unit_type]
    unit = Unit(
        unit_type=unit_type, player=0, hp=unit_hp,
        ammo=stats.max_ammo, fuel=stats.max_fuel,
        pos=(0, 0), moved=False, loaded_units=[],
        is_submerged=False, capture_progress=20, unit_id=1,
    )
    state = GameState(
        map_data=map_data,
        units={0: [unit], 1: []},
        funds=[funds_p0, 0],
        co_states=[make_co_state_safe(co_id_p0), make_co_state_safe(CO_ANDY)],
        properties=map_data.properties,
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
    return state, unit


class TestRachelDayRepair(unittest.TestCase):
    """AWBW chart: Rachel D2D — units heal +1 extra bar on owned props (and pay)."""

    def test_rachel_infantry_on_city_full_band(self) -> None:
        """Infantry HP 40 on Rachel-owned city → HP 70, cost 30% of 1000 = 300."""
        state, unit = _make_state(
            co_id_p0=CO_RACHEL, terrain_id=NEUTRAL_CITY,
            unit_type=UnitType.INFANTRY, unit_hp=40, funds_p0=10_000,
        )
        funds_before = state.funds[0]
        state._resupply_on_properties(0)
        self.assertEqual(unit.hp, 70, "Rachel +30 internal HP heal on city.")
        cost = funds_before - state.funds[0]
        self.assertEqual(
            cost, 300,
            "Rachel pays 30% of Infantry cost (300g) for full +30 heal "
            "(standard COs would pay 200g for +20).",
        )

    def test_rachel_tank_on_base_full_band(self) -> None:
        """Tank HP 70 on Rachel-owned base → HP 100, cost 2100."""
        state, unit = _make_state(
            co_id_p0=CO_RACHEL, terrain_id=NEUTRAL_BASE,
            unit_type=UnitType.TANK, unit_hp=70, funds_p0=50_000,
        )
        funds_before = state.funds[0]
        state._resupply_on_properties(0)
        self.assertEqual(unit.hp, 100, "Rachel heals +30 to cap at HP 100.")
        cost = funds_before - state.funds[0]
        self.assertEqual(
            cost, 2100,
            "Rachel Tank +30 heal: 30% of 7000 = 2100g (vs Andy 1400g for +20).",
        )

    def test_rachel_tank_capped_at_max_hp(self) -> None:
        """Tank HP 90 on Rachel-owned city → HP 100 only (cap), cost for +10 only."""
        state, unit = _make_state(
            co_id_p0=CO_RACHEL, terrain_id=NEUTRAL_CITY,
            unit_type=UnitType.TANK, unit_hp=90, funds_p0=50_000,
        )
        funds_before = state.funds[0]
        state._resupply_on_properties(0)
        self.assertEqual(unit.hp, 100, "Heal must clamp at max HP, not over-heal.")
        cost = funds_before - state.funds[0]
        # +10 internal HP × 7000 listed / 100 = 700g.
        self.assertEqual(
            cost, 700,
            "Cost for the partial heal must match the (h * cost) // 100 formula.",
        )

    def test_andy_baseline_unchanged(self) -> None:
        """Non-Rachel CO (Andy) keeps the standard +20 heal at 20% cost."""
        state, unit = _make_state(
            co_id_p0=CO_ANDY, terrain_id=NEUTRAL_CITY,
            unit_type=UnitType.TANK, unit_hp=70, funds_p0=50_000,
        )
        funds_before = state.funds[0]
        state._resupply_on_properties(0)
        self.assertEqual(
            unit.hp, 90,
            "Andy must still heal +20 internal (HP 70 → 90), no Rachel bonus.",
        )
        cost = funds_before - state.funds[0]
        self.assertEqual(cost, 1400, "Andy Tank +20 heal: 20% of 7000 = 1400g.")

    def test_rachel_unit_on_opponent_property_no_heal(self) -> None:
        """Rachel-owned unit standing on enemy-owned property → no heal applied.

        Property ownership is the gate; CO bonus does not bypass it. (Just-
        captured tile that flips owner only after the heal pass would still
        belong to the other side at heal time — same outcome.)
        """
        state, unit = _make_state(
            co_id_p0=CO_RACHEL, terrain_id=NEUTRAL_CITY,
            unit_type=UnitType.TANK, unit_hp=60, funds_p0=50_000,
            prop_owner=1,  # property belongs to P1, not Rachel
        )
        funds_before = state.funds[0]
        state._resupply_on_properties(0)
        self.assertEqual(
            unit.hp, 60,
            "No heal on enemy-owned property regardless of CO.",
        )
        self.assertEqual(
            state.funds[0], funds_before,
            "No funds spent when no heal occurred.",
        )

    def test_rachel_all_or_nothing_under_budget(self) -> None:
        """Rachel Tank HP 70, funds 1500 → cannot afford full +30 step → no heal.

        Phase 11J-FUNDS-SHIP (R2) replaced the partial-decrement loop with
        AWBW-canon all-or-nothing per-unit repair (three Tier-2 AWBW Wiki
        citations: Units / Advance Wars Overview / Black-Boat — see
        ``docs/oracle_exception_audit/phase11j_funds_deep.md`` §4 R2).
        Rachel's full step is +30 internal HP at 30% of unit cost; for the
        Tank at HP 70 that is ``(30*7000)//100 = 2100g``. Funds 1500 < 2100
        ⇒ no heal at all, treasury untouched.

        The pre-FUNDS-SHIP engine wrongly degraded the step until it fit
        (HP 70 → 91, funds 1500 → 30). That divergence was exactly the
        partial-loop bug R2 was designed to close, and the deep drill
        confirms PHP enforces all-or-nothing at this boundary
        (``tools/_phase11y_rachel_php_check.py`` Rachel funds parity).
        """
        state, unit = _make_state(
            co_id_p0=CO_RACHEL, terrain_id=NEUTRAL_BASE,
            unit_type=UnitType.TANK, unit_hp=70, funds_p0=1500,
        )
        state._resupply_on_properties(0)
        self.assertEqual(
            unit.hp, 70,
            "All-or-nothing: Rachel cannot afford the full +30 step on a "
            "Tank with only 1500g, so the unit must NOT heal at all.",
        )
        self.assertEqual(
            state.funds[0], 1500,
            "Treasury must be untouched when the full step is unaffordable.",
        )


if __name__ == "__main__":
    unittest.main()
