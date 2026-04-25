"""Black Boat repair — display-10 (91–99 internal) charges no gold (R4 parity).

Anchor: GL **1635742** env 38 — Medium Tank at 97 internal / display 10 receives
Black Boat REPAIR; PHP keeps treasury at 2600g (resupply-only). Pre-fix engine
applied ``_black_boat_heal_cost`` (1600g for Md.Tank) and broke the following
twin INF builds.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from engine.action import Action, ActionStage, ActionType
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData
from engine.unit import UNIT_STATS, Unit, UnitType

SEA = 28

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"


def _sea_state(
    *,
    funds: tuple[int, int],
    units: dict[int, list[Unit]],
) -> GameState:
    md = MapData(
        map_id=990_1635742,
        name="bb-repair-display-cap",
        map_type="std",
        terrain=[[SEA, SEA]],
        height=1,
        width=2,
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
        units=units,
        funds=[funds[0], funds[1]],
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
        seam_hp={},
    )


class TestBlackBoatRepairDisplayCap(unittest.TestCase):
    def test_display10_internal_97_no_charge_med_tank_gid1635742(self) -> None:
        st = UNIT_STATS[UnitType.MED_TANK]
        boat = Unit(
            unit_type=UnitType.BLACK_BOAT,
            player=0,
            hp=100,
            ammo=UNIT_STATS[UnitType.BLACK_BOAT].max_ammo,
            fuel=UNIT_STATS[UnitType.BLACK_BOAT].max_fuel,
            pos=(0, 0),
            moved=False,
            loaded_units=[],
            is_submerged=False,
            capture_progress=20,
            unit_id=1,
        )
        tank = Unit(
            unit_type=UnitType.MED_TANK,
            player=0,
            hp=97,
            ammo=st.max_ammo,
            fuel=st.max_fuel - 5,
            pos=(0, 1),
            moved=False,
            loaded_units=[],
            is_submerged=False,
            capture_progress=20,
            unit_id=2,
        )
        state = _sea_state(funds=(2600, 0), units={0: [boat, tank], 1: []})
        state.step(
            Action(
                ActionType.REPAIR,
                unit_pos=(0, 0),
                move_pos=(0, 0),
                target_pos=(0, 1),
            ),
            oracle_mode=True,
        )
        self.assertEqual(state.funds[0], 2600, "No gold — display HP already 10.")
        self.assertEqual(tank.hp, 100)
        self.assertEqual(tank.fuel, st.max_fuel)

    def test_display9_internal_88_still_charges_one_tick(self) -> None:
        st = UNIT_STATS[UnitType.MED_TANK]
        boat = Unit(
            unit_type=UnitType.BLACK_BOAT,
            player=0,
            hp=100,
            ammo=UNIT_STATS[UnitType.BLACK_BOAT].max_ammo,
            fuel=UNIT_STATS[UnitType.BLACK_BOAT].max_fuel,
            pos=(0, 0),
            moved=False,
            loaded_units=[],
            is_submerged=False,
            capture_progress=20,
            unit_id=1,
        )
        tank = Unit(
            unit_type=UnitType.MED_TANK,
            player=0,
            hp=88,
            ammo=st.max_ammo,
            fuel=st.max_fuel,
            pos=(0, 1),
            moved=False,
            loaded_units=[],
            is_submerged=False,
            capture_progress=20,
            unit_id=2,
        )
        state = _sea_state(funds=(5000, 0), units={0: [boat, tank], 1: []})
        state.step(
            Action(
                ActionType.REPAIR,
                unit_pos=(0, 0),
                move_pos=(0, 0),
                target_pos=(0, 1),
            ),
            oracle_mode=True,
        )
        cost = max(1, st.cost // 10)
        self.assertEqual(state.funds[0], 5000 - cost)
        self.assertEqual(tank.hp, 98)

    def test_oracle_snaps_treasury_from_repair_funds_global(self) -> None:
        """PHP embeds post-repair funds on the Repair line — mirror for oracle replay."""
        from tools.oracle_zip_replay import (
            _oracle_snap_treasury_from_repair_funds_block_if_present,
        )

        st = UNIT_STATS[UnitType.MED_TANK]
        boat = Unit(
            unit_type=UnitType.BLACK_BOAT,
            player=0,
            hp=100,
            ammo=UNIT_STATS[UnitType.BLACK_BOAT].max_ammo,
            fuel=UNIT_STATS[UnitType.BLACK_BOAT].max_fuel,
            pos=(0, 0),
            moved=False,
            loaded_units=[],
            is_submerged=False,
            capture_progress=20,
            unit_id=1,
        )
        tank = Unit(
            unit_type=UnitType.MED_TANK,
            player=0,
            hp=97,
            ammo=st.max_ammo,
            fuel=st.max_fuel,
            pos=(0, 1),
            moved=False,
            loaded_units=[],
            is_submerged=False,
            capture_progress=20,
            unit_id=2,
        )
        state = _sea_state(funds=(0, 0), units={0: [boat, tank], 1: []})
        _oracle_snap_treasury_from_repair_funds_block_if_present(
            state, {"funds": {"global": 2600}}
        )
        self.assertEqual(state.funds[0], 2600)


@unittest.skipUnless(
    (ROOT / "replays" / "amarriner_gl" / "1635742.zip").is_file(),
    "requires replays/amarriner_gl/1635742.zip",
)
class TestGid1635742DesyncAuditClean(unittest.TestCase):
    """Integration: full oracle + audit passes after Repair treasury snap (Ft. Fantasy)."""

    def test_desync_audit_ok_merged_catalog(self) -> None:
        from tools.desync_audit import CLS_OK, _audit_one

        cat = _merge_extras()
        row = (cat.get("games") or {}).get("1635742")
        self.assertIsInstance(row, dict, "1635742 must be in merged catalog")
        z = ROOT / "replays" / "amarriner_gl" / "1635742.zip"
        audit = _audit_one(
            games_id=1635742,
            zip_path=z,
            meta=row,
            map_pool=POOL,
            maps_dir=MAPS_DIR,
            seed=1,
        )
        self.assertEqual(audit.cls, CLS_OK, msg=audit.message)


def _merge_extras() -> dict:
    std = json.loads(
        (ROOT / "data" / "amarriner_gl_std_catalog.json").read_text(encoding="utf-8")
    )
    ex = json.loads(
        (ROOT / "data" / "amarriner_gl_extras_catalog.json").read_text(encoding="utf-8")
    )
    g0 = (std.get("games") or {}).copy()
    g1 = ex.get("games") or {}
    for k, v in g1.items():
        if isinstance(v, dict) and "games_id" in v:
            g0[k] = v
    return {"games": g0}


if __name__ == "__main__":
    unittest.main()
