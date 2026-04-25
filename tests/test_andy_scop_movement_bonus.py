"""Andy Hyper Upgrade (SCOP) grants +1 movement to every unit (AWBW parity)."""

from __future__ import annotations

import json
import random
import unittest
from pathlib import Path

from engine.action import compute_reachable_costs
from engine.game import make_initial_state
from engine.map_loader import MapData, load_map
from engine.unit import UNIT_STATS, Unit, UnitType

from tools.amarriner_catalog_cos import pair_catalog_cos_ids
from tools.diff_replay_zips import load_replay
from tools.oracle_zip_replay import (
    apply_oracle_action_json,
    map_snapshot_player_ids_to_engine,
    parse_p_envelopes_from_zip,
    resolve_replay_first_mover,
)

ROOT = Path(__file__).resolve().parents[1]


def _plain_strip_map() -> MapData:
    plain = 1
    terrain = [[plain for _ in range(6)]]
    return MapData(
        map_id=990_201,
        name="andy_scop_move_probe",
        map_type="std",
        terrain=terrain,
        height=1,
        width=6,
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


class TestAndyScopMovementBonus(unittest.TestCase):
    def test_infantry_four_plain_tiles_requires_scop_bonus(self) -> None:
        md = _plain_strip_map()
        st = make_initial_state(md, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=0)
        ist = UNIT_STATS[UnitType.INFANTRY]
        inf = Unit(
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
            unit_id=1,
        )
        st.units = {0: [inf], 1: []}

        st.co_states[0].scop_active = False
        r0 = compute_reachable_costs(st, inf)
        self.assertNotIn((0, 4), r0, "base Infantry move is 3; fourth step is out of range")

        st.co_states[0].scop_active = True
        r1 = compute_reachable_costs(st, inf)
        self.assertIn((0, 4), r1)
        self.assertEqual(r1[(0, 4)], 4)


@unittest.skipUnless(
    (ROOT / "replays" / "amarriner_gl" / "1616284.zip").is_file(),
    "requires replays/amarriner_gl/1616284.zip",
)
class TestGl1616284CaptAfterAndyScop(unittest.TestCase):
    """Phase 9 Lane M: GL replay Capt nested Move needs Andy SCOP +1 MP (Hyper Upgrade)."""

    def test_full_replay_no_oracle_gap_on_capt_envelope(self) -> None:
        gid = 1616284
        cat = json.loads((ROOT / "data/amarriner_gl_std_catalog.json").read_text(encoding="utf-8"))
        meta = next(g for g in cat["games"].values() if int(g["games_id"]) == gid)
        co0, co1 = pair_catalog_cos_ids(meta)
        zpath = ROOT / "replays" / "amarriner_gl" / f"{gid}.zip"
        frames = load_replay(zpath)
        awbw_to_engine = map_snapshot_player_ids_to_engine(frames[0], co0, co1)
        map_data = load_map(
            int(meta["map_id"]),
            ROOT / "data/gl_map_pool.json",
            ROOT / "data/maps",
        )
        envs = parse_p_envelopes_from_zip(zpath)
        first_mover = resolve_replay_first_mover(envs, frames[0], awbw_to_engine)
        random.seed(((1 & 0xFFFFFFFF) << 32) | (int(gid) & 0xFFFFFFFF))
        st = make_initial_state(
            map_data,
            co0,
            co1,
            starting_funds=0,
            tier_name=str(meta.get("tier") or "T2"),
            replay_first_mover=first_mover,
        )
        for _env_i, (pid, _day, actions) in enumerate(envs):
            for obj in actions:
                if st.done:
                    return
                apply_oracle_action_json(st, obj, awbw_to_engine, envelope_awbw_player_id=pid)
