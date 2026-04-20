"""Regression pins for GL tail fixes (AttackSeam / map CSV vs site snapshot)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from engine.action import compute_reachable_costs
from engine.game import make_initial_state
from engine.map_loader import MapData
from engine.unit import UNIT_STATS, Unit, UnitType

from server.play_human import MAPS_DIR, POOL_PATH
from tools.oracle_zip_replay import (
    _furthest_reachable_path_stop_for_seam_attack,
    _resolve_attackseam_no_path_attacker,
)
from tools.verify_map_csv_vs_zip import diff_csv_vs_zip_frame0, verify_zip

ROOT = Path(__file__).resolve().parents[1]


def _tiny_seam_join_map() -> MapData:
    """One row: plain — HPipe seam — plain — plain (cols 0..3)."""
    plain, seam = 1, 113
    return MapData(
        map_id=990_102,
        name="seam_join_probe",
        map_type="std",
        terrain=[[plain, seam, plain, plain]],
        height=1,
        width=4,
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


class TestSeamPathEndSkipsJoinOnlyTile(unittest.TestCase):
    """GL 1629178: do not end AttackSeam move on a friendly join tile when an earlier waypoint still strikes."""

    def test_furthest_stop_skips_boarding_only_destination(self) -> None:
        md = _tiny_seam_join_map()
        st = make_initial_state(md, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0)
        ist = UNIT_STATS[UnitType.INFANTRY]
        mover = Unit(
            UnitType.INFANTRY,
            0,
            55,
            ist.max_ammo,
            ist.max_fuel,
            (0, 0),
            False,
            [],
            False,
            20,
            unit_id=501,
        )
        partner = Unit(
            UnitType.INFANTRY,
            0,
            100,
            ist.max_ammo,
            ist.max_fuel,
            (0, 2),
            False,
            [],
            False,
            20,
            unit_id=502,
        )
        st.units = {0: [mover, partner], 1: []}
        paths = [{"y": 0, "x": 0}, {"y": 0, "x": 2}]
        reach = compute_reachable_costs(st, mover)
        end = _furthest_reachable_path_stop_for_seam_attack(
            st,
            mover,
            paths,
            reach,
            (0, 1),
            json_path_end=(0, 2),
            start_fallback=(0, 0),
        )
        self.assertEqual(end, (0, 0))


class TestAttackSeamNoPathLooseResolver(unittest.TestCase):
    """GL 1609533 family: seam strike when combatInfo anchor tile is empty."""

    def test_resolves_striker_from_off_anchor_when_seam_still_in_range(self) -> None:
        md = _tiny_seam_join_map()
        st = make_initial_state(md, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0)
        ist = UNIT_STATS[UnitType.INFANTRY]
        striker = Unit(
            UnitType.INFANTRY,
            0,
            100,
            ist.max_ammo,
            ist.max_fuel,
            (0, 0),
            False,
            [],
            False,
            20,
            unit_id=901,
        )
        st.units = {0: [striker], 1: []}
        u = _resolve_attackseam_no_path_attacker(
            st,
            eng=0,
            awbw_units_id=999_999,
            anchor_r=0,
            anchor_c=3,
            seam_row=0,
            seam_col=1,
            hp_hint=None,
        )
        self.assertIsNotNone(u)
        assert u is not None
        self.assertEqual(int(u.unit_id), 901)


class TestVerifyMapCsvVsZip(unittest.TestCase):
    def test_diff_flags_csv_mismatch(self) -> None:
        csv_terrain = [[1, 2]]
        frame0 = {
            "buildings": {
                "0": {
                    "__class__": "awbwBuilding",
                    "x": 1,
                    "y": 0,
                    "terrain_id": 28,
                }
            }
        }
        mism = diff_csv_vs_zip_frame0(csv_terrain=csv_terrain, frame0=frame0)
        self.assertEqual(len(mism), 1)
        self.assertEqual(mism[0].row, 0)
        self.assertEqual(mism[0].col, 1)
        self.assertEqual(mism[0].csv_tid, 2)
        self.assertEqual(mism[0].php_tid, 28)

    def test_diff_ignores_cosmetic_recolor_when_move_costs_match(self) -> None:
        csv_terrain = [[1, 43]]
        frame0 = {
            "buildings": {
                "0": {
                    "__class__": "awbwBuilding",
                    "x": 1,
                    "y": 0,
                    "terrain_id": 172,
                }
            }
        }
        mism = diff_csv_vs_zip_frame0(csv_terrain=csv_terrain, frame0=frame0)
        self.assertEqual(mism, [])
        strict = diff_csv_vs_zip_frame0(
            csv_terrain=csv_terrain, frame0=frame0, strict_ids=True
        )
        self.assertEqual(len(strict), 1)


@unittest.skipUnless(
    (ROOT / "replays" / "amarriner_gl" / "1629178.zip").is_file(),
    "requires replays/amarriner_gl/1629178.zip",
)
class TestVerifyMapCsvVsZipIntegration(unittest.TestCase):
    def test_zip_matches_csv_movement_layer(self) -> None:
        z = ROOT / "replays" / "amarriner_gl" / "1629178.zip"
        mid, mism = verify_zip(z, maps_dir=MAPS_DIR)
        self.assertEqual(mid, 180298)
        self.assertEqual(mism, [])


def _catalog_row(games_id: int) -> dict:
    cat = json.loads((ROOT / "data" / "amarriner_gl_std_catalog.json").read_text(encoding="utf-8"))
    games = cat.get("games") or cat
    row = games[str(games_id)]
    if not isinstance(row, dict):
        raise KeyError(games_id)
    return row


@unittest.skipUnless(
    (ROOT / "replays" / "amarriner_gl" / "1609533.zip").is_file()
    and (ROOT / "replays" / "amarriner_gl" / "1628985.zip").is_file()
    and (ROOT / "replays" / "amarriner_gl" / "1629178.zip").is_file(),
    "requires replays/amarriner_gl/{1609533,1628985,1629178}.zip",
)
class TestGlZipOracleReplaysClean(unittest.TestCase):
    """GL tail games: full oracle zip replay + desync_audit ``ok``."""

    def test_replay_oracle_zip_completes(self) -> None:
        from tools.oracle_zip_replay import replay_oracle_zip

        for gid in (1609533, 1628985, 1629178):
            row = _catalog_row(gid)
            z = ROOT / "replays" / "amarriner_gl" / f"{gid}.zip"
            with self.subTest(games_id=gid):
                replay_oracle_zip(
                    z,
                    map_pool=POOL_PATH,
                    maps_dir=MAPS_DIR,
                    map_id=int(row["map_id"]),
                    co0=int(row["co_p0_id"]),
                    co1=int(row["co_p1_id"]),
                    tier_name=str(row.get("tier") or "T2"),
                )

    def test_desync_audit_ok(self) -> None:
        from tools.desync_audit import CLS_OK, _audit_one

        for gid in (1609533, 1628985, 1629178):
            row = _catalog_row(gid)
            z = ROOT / "replays" / "amarriner_gl" / f"{gid}.zip"
            with self.subTest(games_id=gid):
                audit = _audit_one(
                    games_id=gid,
                    zip_path=z,
                    meta=row,
                    map_pool=POOL_PATH,
                    maps_dir=MAPS_DIR,
                )
                self.assertEqual(audit.cls, CLS_OK, msg=audit.message)


if __name__ == "__main__":
    unittest.main()
