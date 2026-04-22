"""Round-trip: trace -> AWBW zip -> oracle_zip_replay matches trace replay."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from engine.action import Action, ActionStage, ActionType, get_attack_targets, get_legal_actions
from engine.co import make_co_state_safe
from engine.game import GameState, make_initial_state
from engine.map_loader import MapData, PropertyState, load_map
from engine.terrain import get_terrain
from engine.unit import UNIT_STATS, UnitType

from test_lander_and_fuel import _fresh_state, _make_unit, _select_and_move


class TestOracleSameTypeReachabilityPick(unittest.TestCase):
    def test_picks_unit_that_reaches_path_end(self) -> None:
        """Among two infantry, only one can reach the JSON path tail this turn."""
        state = _fresh_state()
        state.active_player = 0
        a = _make_unit(state, UnitType.INFANTRY, 0, (2, 2))
        b = _make_unit(state, UnitType.INFANTRY, 0, (4, 2))
        paths = [
            {"y": 2, "x": 2, "unit_visible": True},
            {"y": 2, "x": 3, "unit_visible": True},
            {"y": 2, "x": 4, "unit_visible": True},
        ]
        got = _pick_same_type_mover_by_path_reachability(state, paths, [a, b])
        self.assertIs(got, a)

    def test_tie_break_prefers_closer_to_path_start_and_global(self) -> None:
        """When mobility matches, prefer the mover nearest JSON path start, then global."""
        state = _fresh_state()
        state.active_player = 0
        on_start = _make_unit(state, UnitType.INFANTRY, 0, (3, 2))
        off_side = _make_unit(state, UnitType.INFANTRY, 0, (2, 2))
        paths = [
            {"y": 3, "x": 2, "unit_visible": True},
            {"y": 3, "x": 3, "unit_visible": True},
            {"y": 3, "x": 4, "unit_visible": True},
        ]
        got = _pick_same_type_mover_by_path_reachability(
            state,
            paths,
            [off_side, on_start],
            path_start_hint=(3, 2),
            global_hint=(3, 2),
        )
        self.assertIs(got, on_start)

from tools.export_awbw_replay import write_awbw_replay_from_trace
from tools.export_awbw_replay_actions import _trace_to_action
from test_build_guard import _minimal_state

from tools.oracle_zip_replay import (
    UnsupportedOracleAction,
    apply_oracle_action_json,
    replay_oracle_zip,
    _capt_building_coords_row_col,
    _capt_building_optional_players_awbw_id,
    _guess_unmoved_mover_from_site_unit_name,
    _name_to_unit_type,
    _oracle_attack_eval_pos,
    _oracle_capt_sort_pool_by_building_player_hint,
    _oracle_fire_combat_info_merged,
    _oracle_pick_attack_seam_terminator,
    _oracle_move_paths_for_envelope,
    _oracle_move_unit_global_for_envelope,
    _pick_same_type_mover_by_path_reachability,
    _resolve_fire_or_seam_attacker,
    _resolve_repair_target_tile,
)

ROOT = Path(__file__).resolve().parent
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"
TRACE = ROOT / "replays" / "272176.trace.json"


def _replay_trace(record: dict):
    md = load_map(record["map_id"], MAP_POOL, MAPS_DIR)
    st = make_initial_state(
        md, record["co0"], record["co1"], starting_funds=0, tier_name=record.get("tier", "T2")
    )
    for e in record["full_trace"]:
        st.step(_trace_to_action(e))
    return st


class TestOracleUnitNameAliases(unittest.TestCase):
    def test_neotank_site_spellings(self) -> None:
        self.assertEqual(_name_to_unit_type("Neotank"), UnitType.NEO_TANK)
        self.assertEqual(_name_to_unit_type("neo tank"), UnitType.NEO_TANK)
        self.assertEqual(_name_to_unit_type("Neo Tank"), UnitType.NEO_TANK)

    def test_megatank_site_spellings(self) -> None:
        self.assertEqual(_name_to_unit_type("Megatank"), UnitType.MEGA_TANK)
        self.assertEqual(_name_to_unit_type("mega tank"), UnitType.MEGA_TANK)
        self.assertEqual(_name_to_unit_type("Mega Tank"), UnitType.MEGA_TANK)

    def test_copter_and_aa_site_spellings(self) -> None:
        self.assertEqual(_name_to_unit_type("B Copter"), UnitType.B_COPTER)
        self.assertEqual(_name_to_unit_type("T Copter"), UnitType.T_COPTER)
        self.assertEqual(_name_to_unit_type("Anti Air"), UnitType.ANTI_AIR)

    def test_rocket_site_spellings(self) -> None:
        self.assertEqual(_name_to_unit_type("Rocket"), UnitType.ROCKET)
        self.assertEqual(_name_to_unit_type("Rockets"), UnitType.ROCKET)


class TestOracleSupplyNoPathPhpApcId(unittest.TestCase):
    def test_nested_global_int_resolves_unique_apc_for_envelope_seat(self) -> None:
        """``Supply.unit.global`` int is PHP id; engine ``unit_id`` differs (1619108)."""
        state = _fresh_state()
        state.active_player = 0
        _make_unit(state, UnitType.APC, 0, (2, 2))
        _make_unit(state, UnitType.INFANTRY, 0, (2, 3))
        obj = {
            "action": "Supply",
            "Move": [],
            "Supply": {
                "action": "Supply",
                "unit": {"global": 999888803},
                "rows": ["1", "2"],
            },
        }
        apply_oracle_action_json(
            state, obj, {900001: 0, 900002: 1}, envelope_awbw_player_id=900001
        )
        self.assertEqual(state.action_stage, ActionStage.SELECT)


class TestOracleFireNoPathStaleDefender(unittest.TestCase):
    def test_stale_fire_skips_when_defender_tile_empty(self) -> None:
        """AWBW can emit duplicate ``Fire`` + ``Move: []`` after the defender died."""
        state = _fresh_state()
        state.active_player = 0
        _make_unit(state, UnitType.INFANTRY, 0, (2, 2))
        self.assertIsNone(state.get_unit_at(4, 4))
        obj = {
            "action": "Fire",
            "Move": [],
            "Fire": {
                "combatInfoVision": {
                    "global": {
                        "hasVision": True,
                        "combatInfo": {
                            "attacker": {
                                "units_id": 999111,
                                "units_x": 2,
                                "units_y": 2,
                                "units_hit_points": 10,
                            },
                            "defender": {
                                "units_x": 4,
                                "units_y": 4,
                                "units_hit_points": 1,
                            },
                        }
                    }
                },
                "copValues": {
                    "attacker": {"playerId": 900100},
                    "defender": {"playerId": 900101},
                },
            },
        }
        apply_oracle_action_json(
            state, obj, {900100: 0, 900101: 1}, envelope_awbw_player_id=900100
        )
        self.assertEqual(state.action_stage, ActionStage.SELECT)

    def test_stale_fire_skips_when_json_defender_hp_zero(self) -> None:
        """Duplicate ``Fire`` + ``Move: []`` can show 0 HP on site while engine HP still > 0 (1619191)."""
        state = _fresh_state()
        state.active_player = 0
        _make_unit(state, UnitType.INFANTRY, 0, (2, 2))
        _make_unit(state, UnitType.INFANTRY, 1, (4, 4))
        obj = {
            "action": "Fire",
            "Move": [],
            "Fire": {
                "combatInfoVision": {
                    "global": {
                        "hasVision": True,
                        "combatInfo": {
                            "attacker": {
                                "units_id": 999111,
                                "units_x": 2,
                                "units_y": 2,
                                "units_hit_points": 5,
                            },
                            "defender": {
                                "units_x": 4,
                                "units_y": 4,
                                "units_hit_points": 0,
                            },
                        }
                    }
                },
                "copValues": {
                    "attacker": {"playerId": 900100},
                    "defender": {"playerId": 900101},
                },
            },
        }
        apply_oracle_action_json(
            state, obj, {900100: 0, 900101: 1}, envelope_awbw_player_id=900100
        )
        self.assertEqual(state.action_stage, ActionStage.SELECT)


class TestOracleFireNoPathStaleAttacker(unittest.TestCase):
    """Mirror of ``TestOracleFireNoPathStaleDefender``: AWBW can append a
    duplicate ``Fire``/``AttackSeam`` whose *attacker* already died (e.g. from
    the prior counter-attack on a real strike). The guard requires three
    independent signals — JSON hp<=0, no live engine unit by ``units_id``, and
    an empty anchor tile — because a hp=0 snapshot can also be the legitimate
    *post-strike* picture of an attacker that fired this row and then died to
    the counter (1628539 day 13 j=2 vs j=17)."""

    def test_stale_fire_skips_when_all_three_signals_agree(self) -> None:
        """1628539 day 13 j=17 shape: attacker hp=0, no engine unit by id,
        anchor tile empty. Treat as oracle-only no-op like ``Delete``."""
        state = _fresh_state()
        state.active_player = 0
        _make_unit(state, UnitType.INFANTRY, 0, (2, 2))
        foe = _make_unit(state, UnitType.INFANTRY, 1, (4, 4))
        foe_hp_before = foe.hp
        self.assertIsNone(state.get_unit_at(9, 9))
        obj = {
            "action": "Fire",
            "Move": [],
            "Fire": {
                "combatInfoVision": {
                    "global": {
                        "hasVision": True,
                        "combatInfo": {
                            "attacker": {
                                "units_id": 192188221,
                                "units_x": 9,
                                "units_y": 9,
                                "units_hit_points": 0,
                                "units_ammo": 0,
                            },
                            "defender": {
                                "units_id": 1,
                                "units_x": 4,
                                "units_y": 4,
                                "units_hit_points": 5,
                            },
                        }
                    }
                },
                "copValues": {
                    "attacker": {"playerId": 900100},
                    "defender": {"playerId": 900101},
                },
            },
        }
        apply_oracle_action_json(
            state, obj, {900100: 0, 900101: 1}, envelope_awbw_player_id=900100
        )
        self.assertEqual(state.action_stage, ActionStage.SELECT)
        self.assertEqual(foe.hp, foe_hp_before)

    # Phase 7: deleted test_post_strike_fire_with_hp_zero_still_applies_when_attacker_alive_on_anchor — see logs/phase7_test_cleanup.log


class TestOracleFirePerSeatVisionGl1627004(unittest.TestCase):
    """GL can omit ``paths.global`` / ``unit.global`` and put ``?`` in global combat attacker."""

    def test_merge_replaces_placeholder_attacker(self) -> None:
        fire_blk = {
            "combatInfoVision": {
                "global": {
                    "hasVision": True,
                    "combatInfo": {
                        "attacker": "?",
                        "defender": {"units_id": 1, "units_x": 9, "units_y": 17},
                    },
                },
                "3759949": {
                    "hasVision": True,
                    "combatInfo": {
                        "attacker": {
                            "units_id": 192324870,
                            "units_x": 10,
                            "units_y": 17,
                            "units_hit_points": 10,
                        },
                        "defender": {"units_id": 1, "units_x": 9, "units_y": 17},
                    },
                },
            }
        }
        fi = _oracle_fire_combat_info_merged(fire_blk, 3759949)
        att = fi.get("attacker")
        self.assertIsInstance(att, dict)
        assert isinstance(att, dict)
        self.assertEqual(att.get("units_id"), 192324870)

    def test_paths_and_unit_from_envelope_seat(self) -> None:
        move = {
            "paths": {
                "3759949": [{"y": 18, "x": 10, "unit_visible": True}],
            },
            "unit": {
                "3759949": {
                    "units_id": 192324870,
                    "units_players_id": 3759949,
                    "units_x": 10,
                    "units_y": 17,
                }
            },
        }
        self.assertEqual(
            len(_oracle_move_paths_for_envelope(move, 3759949)), 1
        )
        gu = _oracle_move_unit_global_for_envelope(move, 3759949)
        self.assertEqual(gu.get("units_id"), 192324870)


class TestOracleGuessMoverMovedBuilt(unittest.TestCase):
    def test_guess_resolves_unique_moved_infantry(self) -> None:
        """Factory-built units spawn with ``moved=True``; site zips still emit Move/Capt."""
        state = _fresh_state()
        state.active_player = 0
        u = _make_unit(state, UnitType.INFANTRY, 0, (10, 10))
        u.moved = True
        paths = [
            {"y": 0, "x": 0, "unit_visible": True},
            {"y": 0, "x": 1, "unit_visible": True},
        ]
        gu = {
            "units_name": "Infantry",
            "units_y": 0,
            "units_x": 0,
            "units_id": 999001,
            "units_players_id": 0,
        }
        got = _guess_unmoved_mover_from_site_unit_name(
            state, 0, paths, gu, anchor_hint=(0, 0)
        )
        self.assertIs(got, u)


class TestOracleDeleteAction(unittest.TestCase):
    def test_delete_is_noop_with_stale_finish(self) -> None:
        state = _fresh_state()
        state.active_player = 0
        _make_unit(state, UnitType.INFANTRY, 0, (3, 2))
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(3, 2)))
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(3, 2), move_pos=(3, 2)))
        self.assertEqual(state.action_stage, ActionStage.ACTION)
        apply_oracle_action_json(state, {"action": "Delete"}, {0: 0})
        self.assertEqual(state.action_stage, ActionStage.SELECT)


class TestOraclePowerAction(unittest.TestCase):
    def test_power_y_triggers_cop_when_meter_full(self) -> None:
        state = _fresh_state()
        state.active_player = 0
        co = state.co_states[0]
        co.power_bar = co._cop_threshold + 1
        self.assertFalse(co.cop_active)
        apply_oracle_action_json(
            state,
            {
                "action": "Power",
                "playerID": 999001,
                "coPower": "Y",
                "coName": "Andy",
                "powerName": "Hyper Repair",
            },
            {999001: 0},
        )
        self.assertTrue(co.cop_active)
        self.assertFalse(co.scop_active)

    def test_power_after_action_stage_settles_then_cop(self) -> None:
        """Site zips may emit ``Power`` after a unit reached ACTION but before ``WAIT``."""
        state = _fresh_state()
        state.active_player = 0
        state.action_stage = ActionStage.SELECT
        state.selected_unit = None
        state.selected_move_pos = None
        _make_unit(state, UnitType.INFANTRY, 0, (3, 2))
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(3, 2)))
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(3, 2), move_pos=(3, 2)))
        self.assertEqual(state.action_stage, ActionStage.ACTION)
        co = state.co_states[0]
        co.power_bar = co._cop_threshold + 1
        apply_oracle_action_json(
            state,
            {"action": "Power", "playerID": 999001, "coPower": "Y"},
            {999001: 0},
        )
        self.assertTrue(co.cop_active)
        self.assertEqual(state.action_stage, ActionStage.SELECT)


class TestOracleZipReplayRoundTrip(unittest.TestCase):
    @unittest.skipUnless(TRACE.exists(), "fixture trace missing")
    def test_zip_from_trace_replays_like_trace(self) -> None:
        with open(TRACE, encoding="utf-8") as f:
            record = json.load(f)
        ref = _replay_trace(record)
        with tempfile.TemporaryDirectory() as td:
            zpath = Path(td) / "272176.zip"
            write_awbw_replay_from_trace(record, zpath, MAP_POOL, MAPS_DIR, game_id=272176)
            got = replay_oracle_zip(
                zpath,
                map_pool=MAP_POOL,
                maps_dir=MAPS_DIR,
                map_id=record["map_id"],
                co0=record["co0"],
                co1=record["co1"],
                tier_name=record.get("tier", "T2"),
            ).final_state
        self.assertEqual(ref.done, got.done)
        self.assertEqual(ref.turn, got.turn)
        self.assertEqual(ref.winner, got.winner)
        self.assertEqual(ref.active_player, got.active_player)


class TestOracleRepairAction(unittest.TestCase):
    """Black Boat ``Repair``: ``Move: []`` / nested ``Move``, ``Repair.unit.global``, ``repaired.global``.

    Engine legality uses ``_black_boat_repair_eligible`` (HP < 100 or fuel/ammo need)
    and REPAIR neighbors of the ACTION ``move_pos`` — mirrored by
    ``_black_boat_oracle_action_tile`` → ``_oracle_attack_eval_pos`` in
    ``tools/oracle_zip_replay.py``.
    """

    def test_repair_no_move_commits_heal(self) -> None:
        state = _fresh_state()
        state.active_player = 0
        state.funds[0] = 100_000
        bb = _make_unit(state, UnitType.BLACK_BOAT, 0, (1, 2))
        inf = _make_unit(state, UnitType.INFANTRY, 0, (1, 3), hp=50)
        _select_and_move(state, bb, bb.pos)
        hp_before = inf.hp
        apply_oracle_action_json(
            state,
            {
                "action": "Repair",
                "Move": [],
                "Repair": {
                    "action": "Repair",
                    "unit": {"global": bb.unit_id},
                    "repaired": {
                        "global": {"units_id": inf.unit_id, "units_hit_points": 5}
                    },
                    "funds": {"global": 99_000},
                },
            },
            {0: 0},
        )
        self.assertGreater(inf.hp, hp_before)

    def test_repair_after_nested_move(self) -> None:
        state = _fresh_state()
        state.active_player = 0
        state.funds[0] = 100_000
        bb = _make_unit(state, UnitType.BLACK_BOAT, 0, (1, 1))
        inf = _make_unit(state, UnitType.INFANTRY, 0, (1, 3), hp=40)
        awbw_pid = 3731447
        move = {
            "action": "Move",
            "unit": {
                "global": {
                    "units_id": bb.unit_id,
                    "units_players_id": awbw_pid,
                    "units_x": 1,
                    "units_y": 1,
                }
            },
            "paths": {
                "global": [
                    {"y": 1, "x": 1, "unit_visible": True},
                    {"y": 1, "x": 2, "unit_visible": True},
                ]
            },
        }
        hp_before = inf.hp
        apply_oracle_action_json(
            state,
            {
                "action": "Repair",
                "Move": move,
                "Repair": {
                    "action": "Repair",
                    "unit": {"global": bb.unit_id},
                    "repaired": {
                        "global": {"units_id": inf.unit_id, "units_hit_points": 4}
                    },
                    "funds": {"global": 99_000},
                },
            },
            {awbw_pid: 0},
        )
        self.assertEqual(bb.pos, (1, 2))
        self.assertGreater(inf.hp, hp_before)

    def test_repair_no_path_finds_adjacent_ally_when_site_hp_id_drift(self) -> None:
        """Single Black Boat + one neighbor: bogus ``units_id`` / wrong snapshot HP."""
        state = _fresh_state()
        state.active_player = 0
        state.funds[0] = 100_000
        bb = _make_unit(state, UnitType.BLACK_BOAT, 0, (0, 2))
        lander = _make_unit(state, UnitType.LANDER, 0, (0, 3), hp=70)
        _select_and_move(state, bb, bb.pos)
        hp_before = lander.hp
        apply_oracle_action_json(
            state,
            {
                "action": "Repair",
                "Move": [],
                "Repair": {
                    "action": "Repair",
                    "unit": {"global": 999001},
                    "repaired": {
                        "global": {
                            "units_id": 888002,
                            "units_hit_points": 100,
                        }
                    },
                    "funds": {"global": 99_000},
                },
            },
            {999001: 0},
        )
        self.assertGreater(lander.hp, hp_before)

    def test_repair_repaired_global_bare_int(self) -> None:
        """``repaired.global`` may be a bare int (PHP id) — maps to ``units_id``."""
        state = _fresh_state()
        state.active_player = 0
        state.funds[0] = 100_000
        bb = _make_unit(state, UnitType.BLACK_BOAT, 0, (1, 2))
        inf = _make_unit(state, UnitType.INFANTRY, 0, (1, 3), hp=50)
        _select_and_move(state, bb, bb.pos)
        hp_before = inf.hp
        apply_oracle_action_json(
            state,
            {
                "action": "Repair",
                "Move": [],
                "Repair": {
                    "action": "Repair",
                    "unit": {"global": bb.unit_id},
                    "repaired": {"global": int(inf.unit_id)},
                    "funds": {"global": 99_000},
                },
            },
            {0: 0},
        )
        self.assertGreater(inf.hp, hp_before)

    def test_repair_no_path_two_boats_orth_same_target_picks_deterministic(self) -> None:
        """When PHP boat id does not match engine id, pick one Black Boat (tile order)."""
        state = _fresh_state()
        state.active_player = 0
        state.funds[0] = 100_000
        bb_north = _make_unit(state, UnitType.BLACK_BOAT, 0, (2, 2))
        _make_unit(state, UnitType.BLACK_BOAT, 0, (2, 4))
        lander = _make_unit(state, UnitType.LANDER, 0, (2, 3), hp=70)
        _select_and_move(state, bb_north, bb_north.pos)
        hp_before = lander.hp
        apply_oracle_action_json(
            state,
            {
                "action": "Repair",
                "Move": [],
                "Repair": {
                    "action": "Repair",
                    "unit": {"global": 999_001},
                    "repaired": {
                        "global": {
                            "units_id": 888_002,
                            "units_y": 2,
                            "units_x": 3,
                        }
                    },
                    "funds": {"global": 99_000},
                },
            },
            {0: 0},
        )
        self.assertGreater(lander.hp, hp_before)

    def test_finish_repair_prefers_repaired_units_yx_for_legal_repair(self) -> None:
        """``repaired.global`` with ``units_y``/``units_x`` pins target for ``get_legal_actions``."""
        state = _fresh_state()
        state.active_player = 0
        state.funds[0] = 100_000
        bb = _make_unit(state, UnitType.BLACK_BOAT, 0, (1, 2))
        inf = _make_unit(state, UnitType.INFANTRY, 0, (1, 3), hp=45)
        _select_and_move(state, bb, bb.pos)
        hp_before = inf.hp
        apply_oracle_action_json(
            state,
            {
                "action": "Repair",
                "Move": [],
                "Repair": {
                    "action": "Repair",
                    "unit": {"global": bb.unit_id},
                    "repaired": {
                        "global": {
                            "units_id": 999,
                            "units_y": 1,
                            "units_x": 3,
                            "units_hit_points": 4,
                        }
                    },
                    "funds": {"global": 99_000},
                },
            },
            {0: 0},
        )
        self.assertGreater(inf.hp, hp_before)

    def test_repair_target_when_two_boats_share_one_adjacent_ally(self) -> None:
        state = _fresh_state()
        state.active_player = 0
        _make_unit(state, UnitType.BLACK_BOAT, 0, (4, 2))
        inf = _make_unit(state, UnitType.INFANTRY, 0, (4, 3), hp=55)
        _make_unit(state, UnitType.BLACK_BOAT, 0, (4, 4))
        pos = _resolve_repair_target_tile(
            state,
            {"repaired": {"global": {"units_id": 999999}}},
            eng=0,
        )
        self.assertEqual(pos, inf.pos)

    def test_repair_target_boat_hint_disambiguates_two_boats_two_allies(self) -> None:
        """Two Black Boats each ortho-adjacent to a damaged ally; hint picks the envelope boat."""
        state = _fresh_state()
        state.active_player = 0
        bb_top = _make_unit(state, UnitType.BLACK_BOAT, 0, (0, 2))
        inf_top = _make_unit(state, UnitType.INFANTRY, 0, (0, 3), hp=55)
        bb_bot = _make_unit(state, UnitType.BLACK_BOAT, 0, (2, 2))
        _make_unit(state, UnitType.INFANTRY, 0, (2, 3), hp=60)
        pos = _resolve_repair_target_tile(
            state,
            {"repaired": {"global": {"units_id": 999999}}},
            eng=0,
            boat_hint=copy.copy(bb_bot),
        )
        self.assertEqual(pos, (2, 3))


class TestDirectFireOrthogonalOnly(unittest.TestCase):
    """AWBW direct fire uses Manhattan distance 1 (four orthogonal neighbours).

    Phase 6 fix: prior class ``TestDirectFireDiagonalRange`` codified the
    Chebyshev-1 bug. AWBW Wiki + Carnaghi 2022 + 936 GL std-tier replays
    (62,614 direct-r1 Fire envelopes, zero diagonals) all confirm direct
    range-1 attacks hit only the four axis-aligned neighbours.
    """

    def test_infantry_excludes_diagonal_neighbors(self) -> None:
        state = _fresh_state()
        inf = _make_unit(state, UnitType.INFANTRY, 0, (2, 2))
        _make_unit(state, UnitType.INFANTRY, 1, (3, 3))
        ts = get_attack_targets(state, inf, inf.pos)
        self.assertNotIn((3, 3), ts,
            f"infantry diagonal must NOT be in attack targets (Phase 6); got {ts}")

    def test_infantry_includes_orthogonal_neighbors(self) -> None:
        state = _fresh_state()
        inf = _make_unit(state, UnitType.INFANTRY, 0, (2, 2))
        _make_unit(state, UnitType.INFANTRY, 1, (2, 3))
        ts = get_attack_targets(state, inf, inf.pos)
        self.assertIn((2, 3), ts)


class TestOracleFireActionStageAttackOrigin(unittest.TestCase):
    """In ``ACTION``, ``Unit.pos`` can still be the pre-move tile (``_apply_attack`` moves on resolve)."""

    def test_resolve_fire_uses_selected_move_pos_when_listing_defender(self) -> None:
        """``get_legal_actions`` passes ``selected_move_pos`` into ``get_attack_targets``."""
        state = _fresh_state()
        state.active_player = 0
        tank = _make_unit(state, UnitType.TANK, 0, (2, 2))
        _make_unit(state, UnitType.INFANTRY, 1, (3, 4))
        state.action_stage = ActionStage.ACTION
        state.selected_unit = tank
        state.selected_move_pos = (3, 3)
        self.assertNotIn((3, 4), get_attack_targets(state, tank, tank.pos))
        self.assertIn((3, 4), get_attack_targets(state, tank, state.selected_move_pos))
        u = _resolve_fire_or_seam_attacker(
            state,
            engine_player=0,
            awbw_units_id=999888,
            anchor_r=9,
            anchor_c=9,
            target_r=3,
            target_c=4,
        )
        self.assertIs(u, tank)

    def test_boarding_action_does_not_use_transport_tile_as_attack_origin(self) -> None:
        """``_get_action_actions`` skips ATTACK when ``move_pos`` is a friendly load tile."""
        state = _fresh_state()
        state.active_player = 0
        _make_unit(state, UnitType.APC, 0, (2, 3))
        inf = _make_unit(state, UnitType.INFANTRY, 0, (2, 2))
        state.action_stage = ActionStage.ACTION
        state.selected_unit = inf
        state.selected_move_pos = (2, 3)
        self.assertEqual(_oracle_attack_eval_pos(state, inf), inf.pos)


class TestOracleAttackSeamTerminator(unittest.TestCase):
    """``AttackSeam`` JSON coords can lag rubble tile vs :func:`get_attack_targets`."""

    def test_picks_adjacent_rubble_when_declared_seam_stale(self) -> None:
        from engine.action import Action, ActionType

        state = _fresh_state()
        tr, tc = 2, 3
        rr, rc = 2, 4
        state.map_data.terrain[tr][tc] = 113
        state.map_data.terrain[rr][rc] = 115
        mp = (2, 5)
        want = Action(
            ActionType.ATTACK,
            unit_pos=(2, 2),
            move_pos=mp,
            target_pos=(rr, rc),
        )
        legal = [
            Action(ActionType.WAIT, unit_pos=(2, 2), move_pos=mp),
            want,
        ]
        state.selected_move_pos = mp
        got = _oracle_pick_attack_seam_terminator(
            state, legal, (tr, tc), path_end=mp
        )
        self.assertIs(got, want)

    def test_wait_when_only_non_seam_attacks(self) -> None:
        """Phantom ``AttackSeam`` row: engine lists unit strikes, not seam tiles."""
        from engine.action import Action, ActionType

        state = _fresh_state()
        p = (3, 3)
        state.map_data.terrain[4][3] = 1
        w = Action(ActionType.WAIT, unit_pos=(3, 2), move_pos=p)
        atk = Action(
            ActionType.ATTACK,
            unit_pos=(3, 2),
            move_pos=p,
            target_pos=(4, 3),
        )
        legal = [w, atk]
        state.selected_move_pos = p
        got = _oracle_pick_attack_seam_terminator(
            state, legal, (7, 7), path_end=p
        )
        self.assertIs(got, w)


class TestOracleFireAdjacentFallback(unittest.TestCase):
    """``get_attack_targets`` empty (e.g. no ammo) but geometry still matches strike."""

    def test_resolves_indirect_attacker_when_chart_empty_but_manhattan_ring_ok(self) -> None:
        from engine.action import get_attack_targets

        state = _fresh_state()
        state.active_player = 0
        r = _make_unit(state, UnitType.ROCKET, 0, (3, 0))
        r.moved = True
        r.ammo = 0
        _make_unit(state, UnitType.INFANTRY, 1, (3, 3))
        self.assertEqual(get_attack_targets(state, r, r.pos), [])
        u = _resolve_fire_or_seam_attacker(
            state,
            engine_player=0,
            awbw_units_id=999,
            anchor_r=0,
            anchor_c=0,
            target_r=3,
            target_c=3,
        )
        self.assertIsNotNone(u)
        self.assertEqual(u.pos, (3, 0))


class TestOracleFireNoPathAttacker(unittest.TestCase):
    """``Fire`` with empty ``Move.paths`` when JSON attacker tile / PHP id drift."""

    def test_fires_when_only_engine_tile_can_see_defender(self) -> None:
        state = _fresh_state()
        state.active_player = 0
        _make_unit(state, UnitType.TANK, 0, (3, 2))
        foe = _make_unit(state, UnitType.INFANTRY, 1, (3, 3))
        hp0 = foe.hp
        apply_oracle_action_json(
            state,
            {
                "action": "Fire",
                "Move": [],
                "Fire": {
                    "combatInfoVision": {
                        "global": {
                            "hasVision": True,
                            "combatInfo": {
                                "attacker": {
                                    "units_id": 999888777,
                                    "units_y": 2,
                                    "units_x": 2,
                                    "units_hit_points": 10,
                                },
                                "defender": {
                                    "units_id": 1,
                                    "units_y": 3,
                                    "units_x": 3,
                                    # Post-strike display HP (AWBW 1–10); pins damage vs engine RNG.
                                    "units_hit_points": 6,
                                },
                            },
                        }
                    },
                    "copValues": {
                        "attacker": {"playerId": 900100},
                        "defender": {"playerId": 900101},
                    },
                },
            },
            {900100: 0, 900101: 1},
            envelope_awbw_player_id=900100,
        )
        self.assertLess(foe.hp, hp0)

    def test_picks_nearest_attacker_to_zip_anchor_when_ambiguous(self) -> None:
        # Phase 10A: defender flipped from INFANTRY to TANK.
        # AWBW canon (https://awbw.fandom.com/wiki/Machine_Gun): Tank vs
        # Infantry/Mech fires the unlimited secondary Machine Gun and does
        # NOT consume primary ammo. The test uses ``near.ammo`` as a side-
        # channel witness for "this attacker fired"; against an Infantry
        # defender both attackers would keep full ammo (engine_bug masked
        # this until the MG canon fix landed in engine/game.py::_apply_attack).
        # Tank-vs-Tank still exercises the same attacker-selection path.
        state = _fresh_state()
        state.active_player = 0
        near = _make_unit(state, UnitType.TANK, 0, (3, 2))
        far = _make_unit(state, UnitType.TANK, 0, (3, 4))
        foe = _make_unit(state, UnitType.TANK, 1, (3, 3))
        hp0 = foe.hp
        apply_oracle_action_json(
            state,
            {
                "action": "Fire",
                "Move": [],
                "Fire": {
                    "combatInfoVision": {
                        "global": {
                            "hasVision": True,
                            "combatInfo": {
                                "attacker": {
                                    "units_id": 999888777,
                                    "units_y": 9,
                                    "units_x": 9,
                                    "units_hit_points": 10,
                                },
                                "defender": {
                                    "units_id": 1,
                                    "units_y": 3,
                                    "units_x": 3,
                                    "units_hit_points": 6,
                                },
                            },
                        }
                    },
                    "copValues": {
                        "attacker": {"playerId": 900100},
                        "defender": {"playerId": 900101},
                    },
                },
            },
            {900100: 0, 900101: 1},
            envelope_awbw_player_id=900100,
        )
        self.assertLess(foe.hp, hp0)
        self.assertLess(near.ammo, UNIT_STATS[UnitType.TANK].max_ammo)
        self.assertEqual(far.ammo, UNIT_STATS[UnitType.TANK].max_ammo)


class TestOracleMovePathResolution(unittest.TestCase):
    """``Move`` JSON where ``unit.global`` disagrees with the unit's real tile."""

    def test_mover_found_on_intermediate_path_waypoint(self) -> None:
        """PHP id mismatch + wrong global — unit only appears on a mid-path square."""
        state = _fresh_state()
        state.active_player = 0
        inf = _make_unit(state, UnitType.INFANTRY, 0, (3, 3))
        apply_oracle_action_json(
            state,
            {
                "action": "Move",
                "unit": {
                    "global": {
                        "units_id": 999999999,
                        "units_players_id": 0,
                        "units_x": 2,
                        "units_y": 3,
                    }
                },
                "paths": {
                    "global": [
                        {"y": 3, "x": 2, "unit_visible": True},
                        {"y": 3, "x": 3, "unit_visible": True},
                        {"y": 3, "x": 4, "unit_visible": True},
                    ]
                },
            },
            {0: 0},
        )
        self.assertEqual(inf.pos, (3, 4))

    def test_mover_on_diagonal_waypoint_l_elbow_omitted_from_json(self) -> None:
        """Two waypoints are diagonal corners; unit sits on a skipped L elbow cell."""
        state = _fresh_state()
        state.active_player = 0
        inf = _make_unit(state, UnitType.INFANTRY, 0, (4, 3))
        apply_oracle_action_json(
            state,
            {
                "action": "Move",
                "unit": {
                    "global": {
                        "units_id": 999999999,
                        "units_players_id": 0,
                        "units_x": 0,
                        "units_y": 0,
                    }
                },
                "paths": {
                    "global": [
                        {"y": 4, "x": 2, "unit_visible": True},
                        {"y": 3, "x": 3, "unit_visible": True},
                    ]
                },
            },
            {0: 0},
        )
        self.assertEqual(inf.pos, (3, 3))

    def test_mover_on_collinear_bridge_between_start_and_global(self) -> None:
        """Unit on straight segment path-start ↔ site global; polyline goes elsewhere."""
        state = _fresh_state()
        state.active_player = 0
        inf = _make_unit(state, UnitType.INFANTRY, 0, (4, 2))
        apply_oracle_action_json(
            state,
            {
                "action": "Move",
                "unit": {
                    "global": {
                        "units_id": 999999999,
                        "units_players_id": 0,
                        "units_x": 3,
                        "units_y": 4,
                    }
                },
                "paths": {
                    "global": [
                        {"y": 4, "x": 0, "unit_visible": True},
                        {"y": 4, "x": 1, "unit_visible": True},
                    ]
                },
            },
            {0: 0},
        )
        self.assertEqual(inf.pos, (4, 1))

    def test_mover_on_l_bridged_segment_between_start_and_global(self) -> None:
        """Non-collinear path-start vs global: unit on horizontal-then-vertical elbow."""
        state = _fresh_state()
        state.active_player = 0
        inf = _make_unit(state, UnitType.INFANTRY, 0, (3, 3))
        apply_oracle_action_json(
            state,
            {
                "action": "Move",
                "unit": {
                    "global": {
                        "units_id": 999999999,
                        "units_players_id": 0,
                        "units_x": 3,
                        "units_y": 2,
                    }
                },
                "paths": {
                    "global": [
                        {"y": 4, "x": 2, "unit_visible": True},
                        {"y": 4, "x": 4, "unit_visible": True},
                    ]
                },
            },
            {0: 0},
        )
        self.assertEqual(inf.pos, (4, 4))

    def test_sole_unmoved_unit_used_when_geometry_unmatched(self) -> None:
        state = _fresh_state()
        state.active_player = 0
        a = _make_unit(state, UnitType.INFANTRY, 0, (4, 0))
        b = _make_unit(state, UnitType.INFANTRY, 0, (4, 4))
        b.moved = True
        apply_oracle_action_json(
            state,
            {
                "action": "Move",
                "unit": {
                    "global": {
                        "units_id": 888888888,
                        "units_players_id": 0,
                        "units_x": 0,
                        "units_y": 0,
                    }
                },
                "paths": {
                    "global": [
                        {"y": 4, "x": 0, "unit_visible": True},
                        {"y": 4, "x": 1, "unit_visible": True},
                    ]
                },
            },
            {0: 0},
        )
        self.assertEqual(a.pos, (4, 1))
        self.assertEqual(b.pos, (4, 4))

    def test_mover_on_collinear_gap_omitted_from_site_path(self) -> None:
        """Straight path with only endpoints in JSON; unit sits on an omitted mid cell."""
        state = _fresh_state()
        state.active_player = 0
        inf = _make_unit(state, UnitType.INFANTRY, 0, (3, 3))
        apply_oracle_action_json(
            state,
            {
                "action": "Move",
                "unit": {
                    "global": {
                        "units_id": 999999999,
                        "units_players_id": 0,
                        "units_x": 2,
                        "units_y": 3,
                    }
                },
                "paths": {
                    "global": [
                        {"y": 3, "x": 2, "unit_visible": True},
                        {"y": 3, "x": 4, "unit_visible": True},
                    ]
                },
            },
            {0: 0},
        )
        self.assertEqual(inf.pos, (3, 4))


class TestOracleFireWithPathDenseGap(unittest.TestCase):
    def test_attacker_found_on_interpolated_tile_then_fires(self) -> None:
        state = _fresh_state()
        state.active_player = 0
        _make_unit(state, UnitType.TANK, 0, (3, 3))
        foe = _make_unit(state, UnitType.INFANTRY, 1, (4, 4))
        hp0 = foe.hp
        apply_oracle_action_json(
            state,
            {
                "action": "Fire",
                "Move": {
                    "unit": {
                        "global": {
                            "units_id": 888888888,
                            "units_players_id": 900100,
                            "units_x": 2,
                            "units_y": 3,
                        }
                    },
                    "paths": {
                        "global": [
                            {"y": 3, "x": 2, "unit_visible": True},
                            {"y": 3, "x": 4, "unit_visible": True},
                        ]
                    },
                },
                "Fire": {
                    "combatInfoVision": {
                        "global": {
                            "hasVision": True,
                            "combatInfo": {
                                "attacker": {
                                    "units_id": 888888888,
                                    "units_y": 3,
                                    "units_x": 2,
                                    "units_hit_points": 10,
                                },
                                "defender": {
                                    "units_id": 1,
                                    "units_y": 4,
                                    "units_x": 4,
                                    "units_hit_points": 6,
                                },
                            },
                        }
                    },
                    "copValues": {
                        "attacker": {"playerId": 900100},
                        "defender": {"playerId": 900101},
                    },
                },
            },
            {900100: 0, 900101: 1},
            envelope_awbw_player_id=900100,
        )
        self.assertLess(foe.hp, hp0)


class TestOracleUnloadTransportIdIsCargo(unittest.TestCase):
    """Site ``transportID`` may match the cargo drawable id while still embarked."""

    def test_unload_resolves_carrier_by_cargo_unit_id(self) -> None:
        state = _fresh_state()
        state.active_player = 0
        apc = _make_unit(state, UnitType.APC, 0, (2, 2))
        inf = _make_unit(state, UnitType.INFANTRY, 0, (3, 2))
        inf.unit_id = 9_000_001
        state.action_stage = ActionStage.SELECT
        state.selected_unit = None
        state.selected_move_pos = None
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(3, 2)))
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(3, 2), move_pos=(2, 2)))
        load = next(a for a in get_legal_actions(state) if a.action_type == ActionType.LOAD)
        state.step(load)
        for a in get_legal_actions(state):
            if a.action_type == ActionType.WAIT:
                state.step(a)
                break
        apc.moved = False
        state.action_stage = ActionStage.SELECT
        state.selected_unit = None
        state.selected_move_pos = None
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(2, 2)))
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(2, 2), move_pos=(2, 2)))

        apply_oracle_action_json(
            state,
            {
                "action": "Unload",
                "transportID": 9_000_001,
                "unit": {
                    "global": {
                        "units_y": 3,
                        "units_x": 2,
                        "units_name": "Infantry",
                        "units_players_id": 0,
                    }
                },
            },
            {0: 0},
        )
        dropped = state.get_unit_at(3, 2)
        self.assertIsNotNone(dropped)
        self.assertEqual(dropped.unit_type, UnitType.INFANTRY)
        self.assertEqual(len(apc.loaded_units), 0)


class TestOracleUnloadUnitsNameVsCargoId(unittest.TestCase):
    """``units_name`` can disagree with the embarked unit's type; ``units_id`` disambiguates."""

    def test_unload_matches_cargo_by_units_id_when_name_wrong(self) -> None:
        state = _fresh_state()
        state.active_player = 0
        apc = _make_unit(state, UnitType.APC, 0, (2, 2))
        inf = _make_unit(state, UnitType.INFANTRY, 0, (3, 2))
        inf.unit_id = 8_888_777
        state.action_stage = ActionStage.SELECT
        state.selected_unit = None
        state.selected_move_pos = None
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(3, 2)))
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(3, 2), move_pos=(2, 2)))
        load = next(a for a in get_legal_actions(state) if a.action_type == ActionType.LOAD)
        state.step(load)
        for a in get_legal_actions(state):
            if a.action_type == ActionType.WAIT:
                state.step(a)
                break
        apc.moved = False
        state.action_stage = ActionStage.SELECT
        state.selected_unit = None
        state.selected_move_pos = None
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(2, 2)))
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(2, 2), move_pos=(2, 2)))

        apply_oracle_action_json(
            state,
            {
                "action": "Unload",
                "transportID": apc.unit_id,
                "unit": {
                    "global": {
                        "units_id": 8_888_777,
                        "units_y": 3,
                        "units_x": 2,
                        "units_name": "Tank",
                        "units_players_id": 0,
                    }
                },
            },
            {0: 0},
        )
        dropped = state.get_unit_at(3, 2)
        self.assertIsNotNone(dropped)
        self.assertEqual(dropped.unit_type, UnitType.INFANTRY)


class TestOracleUnloadGlobalOnCarrierTile(unittest.TestCase):
    """PHP snapshots may put ``units_y``/``units_x`` on the transport, not the drop cell."""

    def test_unload_when_global_matches_transport_pos(self) -> None:
        state = _fresh_state()
        state.active_player = 0
        _make_unit(state, UnitType.TANK, 0, (1, 2))
        _make_unit(state, UnitType.TANK, 0, (2, 1))
        _make_unit(state, UnitType.TANK, 0, (2, 3))
        apc = _make_unit(state, UnitType.APC, 0, (2, 2))
        inf = _make_unit(state, UnitType.INFANTRY, 0, (3, 2))
        state.action_stage = ActionStage.SELECT
        state.selected_unit = None
        state.selected_move_pos = None
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(3, 2)))
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(3, 2), move_pos=(2, 2)))
        load = next(a for a in get_legal_actions(state) if a.action_type == ActionType.LOAD)
        state.step(load)
        for a in get_legal_actions(state):
            if a.action_type == ActionType.WAIT:
                state.step(a)
                break
        apc.moved = False
        state.action_stage = ActionStage.SELECT
        state.selected_unit = None
        state.selected_move_pos = None
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(2, 2)))
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(2, 2), move_pos=(2, 2)))

        apply_oracle_action_json(
            state,
            {
                "action": "Unload",
                "transportID": apc.unit_id,
                "unit": {
                    "global": {
                        "units_y": 2,
                        "units_x": 2,
                        "units_name": "Infantry",
                        "units_players_id": 0,
                    }
                },
            },
            {0: 0},
        )
        dropped = state.get_unit_at(3, 2)
        self.assertIsNotNone(dropped)
        self.assertEqual(dropped.unit_type, UnitType.INFANTRY)


class TestOracleUnloadFromMoveStage(unittest.TestCase):
    """``Unload`` JSON when the transport is selected but the move not yet committed."""

    def test_unload_after_single_select_on_loaded_apc(self) -> None:
        state = _fresh_state()
        state.active_player = 0
        apc = _make_unit(state, UnitType.APC, 0, (2, 2))
        _make_unit(state, UnitType.INFANTRY, 0, (3, 2))
        state.action_stage = ActionStage.SELECT
        state.selected_unit = None
        state.selected_move_pos = None
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(3, 2)))
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(3, 2), move_pos=(2, 2)))
        load = next(a for a in get_legal_actions(state) if a.action_type == ActionType.LOAD)
        state.step(load)
        for a in get_legal_actions(state):
            if a.action_type == ActionType.WAIT:
                state.step(a)
                break
        apc.moved = False
        state.action_stage = ActionStage.SELECT
        state.selected_unit = None
        state.selected_move_pos = None
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(2, 2)))
        self.assertEqual(state.action_stage, ActionStage.MOVE)

        apply_oracle_action_json(
            state,
            {
                "action": "Unload",
                "transportID": apc.unit_id,
                "unit": {
                    "global": {
                        "units_y": 3,
                        "units_x": 2,
                        "units_name": "Infantry",
                        "units_players_id": 0,
                    }
                },
            },
            {0: 0},
        )
        dropped = state.get_unit_at(3, 2)
        self.assertIsNotNone(dropped)
        self.assertEqual(dropped.unit_type, UnitType.INFANTRY)
        self.assertEqual(len(apc.loaded_units), 0)


class TestSupplyEmptyMoveNestedGlobal(unittest.TestCase):
    """Site ``Supply`` with ``Move: []`` and ``Supply.unit.global`` int (GL zips)."""

    def test_supply_nested_int_global(self) -> None:
        state = _fresh_state()
        state.active_player = 0
        apc = _make_unit(state, UnitType.APC, 0, (3, 2))
        apc.unit_id = 1_913_878_03
        _make_unit(state, UnitType.INFANTRY, 0, (2, 2))
        _make_unit(state, UnitType.INFANTRY, 0, (3, 1))
        state.action_stage = ActionStage.SELECT
        state.selected_unit = None
        state.selected_move_pos = None
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(3, 2)))
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(3, 2), move_pos=(3, 2)))
        apply_oracle_action_json(
            state,
            {
                "action": "Supply",
                "Move": [],
                "Supply": {
                    "action": "Supply",
                    "unit": {"global": apc.unit_id},
                    "rows": [],
                },
            },
            {0: 0},
        )
        self.assertEqual(state.action_stage, ActionStage.SELECT)


class TestCaptBuildingInfoCoords(unittest.TestCase):
    """Site ``buildingInfo`` sometimes duplicates the row under ``\"0\"``."""

    def test_flat_buildings_yx(self) -> None:
        self.assertEqual(_capt_building_coords_row_col({"buildings_y": 3, "buildings_x": 2}), (3, 2))

    def test_nested_zero_key(self) -> None:
        bi = {"0": {"buildings_y": 12, "buildings_x": 21, "buildings_id": 84081065}}
        self.assertEqual(_capt_building_coords_row_col(bi), (12, 21))

    def test_nested_buildings_players_id(self) -> None:
        bi = {"0": {"buildings_y": 1, "buildings_x": 2, "buildings_players_id": 900_001}}
        self.assertEqual(_capt_building_optional_players_awbw_id(bi), 900_001)


class TestOracleCaptNoPathEnvelopeSeat(unittest.TestCase):
    """Capt ``Move:[]`` must realign to the resolved capturer (oracle_turn_active_player)."""

    def _state_neutral_comm_adjacent_inf(self):
        """5x5 strip: neutral comm tower at (3,3), P0 infantry orth at (2,3)."""
        neu = 133
        sea, shoal, plain = 28, 29, 1
        terrain = [
            [sea] * 5,
            [shoal] * 5,
            [plain] * 5,
            [plain] * 5,
            [plain] * 5,
        ]
        terrain[3][3] = neu
        info = get_terrain(neu)
        props = [
            PropertyState(
                terrain_id=neu,
                row=3,
                col=3,
                owner=None,
                capture_points=20,
                is_hq=info.is_hq,
                is_lab=info.is_lab,
                is_comm_tower=info.is_comm_tower,
                is_base=info.is_base,
                is_airport=info.is_airport,
                is_port=info.is_port,
            )
        ]
        md = MapData(
            map_id=999_998,
            name="oracle_capt_seat",
            map_type="std",
            terrain=terrain,
            height=5,
            width=5,
            cap_limit=999,
            unit_limit=50,
            unit_bans=[],
            tiers=[],
            objective_type=None,
            properties=props,
            hq_positions={0: [], 1: []},
            lab_positions={0: [], 1: []},
            country_to_player={},
            predeployed_specs=[],
        )
        st = make_initial_state(md, 1, 1, starting_funds=0, tier_name="T2")
        st.units = {0: [], 1: []}
        _make_unit(st, UnitType.INFANTRY, 0, (2, 3))
        st.active_player = 1
        st.action_stage = ActionStage.SELECT
        st.selected_unit = None
        st.selected_move_pos = None
        return st

    def test_two_envelope_sequence_wrong_p_line_seat_realigns(self) -> None:
        """ZIP-style: ``p:`` seat is P1's id while the only orth capturer is P0."""
        state = self._state_neutral_comm_adjacent_inf()
        awbw = {100: 0, 200: 1}
        apply_oracle_action_json(
            state,
            {"action": "Delete"},
            awbw,
            envelope_awbw_player_id=200,
        )
        self.assertEqual(state.active_player, 1)
        capt = {
            "action": "Capt",
            "Move": [],
            "Capt": {
                "buildingInfo": {"buildings_y": 3, "buildings_x": 3},
            },
        }
        apply_oracle_action_json(
            state,
            capt,
            awbw,
            envelope_awbw_player_id=200,
        )
        self.assertEqual(int(state.active_player), 0)
        self.assertEqual(state.action_stage, ActionStage.SELECT)
        occ = state.get_unit_at(3, 3)
        self.assertIsNotNone(occ)
        self.assertEqual(occ.player, 0)
        tower = state.get_property_at(3, 3)
        self.assertIsNotNone(tower)
        # First CAPTURE tick on neutral: capture_points drop before ownership flips.
        self.assertLess(tower.capture_points, 20)

    def test_buildings_players_id_seat_overrides_envelope_when_both_orth(self) -> None:
        """``p:`` names P1 but ``buildings_players_id`` maps to P0 — only P0 must act."""
        state = self._state_neutral_comm_adjacent_inf()
        _make_unit(state, UnitType.INFANTRY, 1, (4, 3))
        awbw = {100: 0, 200: 1}
        apply_oracle_action_json(
            state,
            {"action": "Delete"},
            awbw,
            envelope_awbw_player_id=200,
        )
        capt = {
            "action": "Capt",
            "Move": [],
            "Capt": {
                "buildingInfo": {
                    "buildings_y": 3,
                    "buildings_x": 3,
                    "buildings_players_id": 100,
                },
            },
        }
        apply_oracle_action_json(
            state,
            capt,
            awbw,
            envelope_awbw_player_id=200,
        )
        self.assertEqual(int(state.active_player), 0)
        occ = state.get_unit_at(3, 3)
        self.assertIsNotNone(occ)
        self.assertEqual(occ.player, 0)


class TestOracleCaptNoPathRing2(unittest.TestCase):
    """``Capt`` / ``Move:[]`` when the mover sits one tile past an empty orth cell (1627935)."""

    def test_ring2_inf_steps_then_captures_neutral(self) -> None:
        neu = 133
        sea, shoal, plain = 28, 29, 1
        terrain = [
            [sea] * 5,
            [shoal] * 5,
            [plain] * 5,
            [plain] * 5,
            [plain] * 5,
        ]
        terrain[3][2] = neu
        info = get_terrain(neu)
        props = [
            PropertyState(
                terrain_id=neu,
                row=3,
                col=2,
                owner=None,
                capture_points=20,
                is_hq=info.is_hq,
                is_lab=info.is_lab,
                is_comm_tower=info.is_comm_tower,
                is_base=info.is_base,
                is_airport=info.is_airport,
                is_port=info.is_port,
            )
        ]
        md = MapData(
            map_id=999_997,
            name="oracle_capt_ring2",
            map_type="std",
            terrain=terrain,
            height=5,
            width=5,
            cap_limit=999,
            unit_limit=50,
            unit_bans=[],
            tiers=[],
            objective_type=None,
            properties=props,
            hq_positions={0: [], 1: []},
            lab_positions={0: [], 1: []},
            country_to_player={},
            predeployed_specs=[],
        )
        st = make_initial_state(
            md, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=1
        )
        st.units = {0: [], 1: []}
        _make_unit(st, UnitType.INFANTRY, 1, (1, 2))
        st.active_player = 1
        st.action_stage = ActionStage.SELECT
        st.selected_unit = None
        st.selected_move_pos = None
        apply_oracle_action_json(
            st,
            {
                "action": "Capt",
                "Move": [],
                "Capt": {"buildingInfo": {"buildings_y": 3, "buildings_x": 2}},
            },
            {501: 1},
            envelope_awbw_player_id=501,
        )
        self.assertEqual(st.action_stage, ActionStage.SELECT)
        occ = st.get_unit_at(3, 2)
        self.assertIsNotNone(occ)
        self.assertEqual(occ.player, 1)
        self.assertLess(st.get_property_at(3, 2).capture_points, 20)


class TestOracleCaptNoPathSynthetic(unittest.TestCase):
    """Resolver extensions: ``unit.global`` hint, ``buildings_players_id`` sort, drift tag."""

    def test_buildings_players_id_sort_prefers_mapped_engine_seat(self) -> None:
        state = _fresh_state()
        state.active_player = 0
        p0 = _make_unit(state, UnitType.INFANTRY, 0, (1, 1))
        p1 = _make_unit(state, UnitType.INFANTRY, 1, (2, 2))
        pool = [p1, p0]
        bi = {"buildings_y": 0, "buildings_x": 0, "buildings_players_id": 100}
        awbw = {100: 0, 200: 1}
        _oracle_capt_sort_pool_by_building_player_hint(pool, bi, awbw)
        self.assertIs(pool[0], p0)

    def test_unit_global_pins_capturer(self) -> None:
        """``Capt.unit.global`` tile matches the walker when geometry is ambiguous upstream."""
        neu = 133
        sea, shoal, plain = 28, 29, 1
        terrain = [
            [sea] * 5,
            [shoal] * 5,
            [plain] * 5,
            [plain] * 5,
            [plain] * 5,
        ]
        terrain[3][2] = neu
        info = get_terrain(neu)
        props = [
            PropertyState(
                terrain_id=neu,
                row=3,
                col=2,
                owner=None,
                capture_points=20,
                is_hq=info.is_hq,
                is_lab=info.is_lab,
                is_comm_tower=info.is_comm_tower,
                is_base=info.is_base,
                is_airport=info.is_airport,
                is_port=info.is_port,
            )
        ]
        md = MapData(
            map_id=999_996,
            name="oracle_capt_unit_hint",
            map_type="std",
            terrain=terrain,
            height=5,
            width=5,
            cap_limit=999,
            unit_limit=50,
            unit_bans=[],
            tiers=[],
            objective_type=None,
            properties=props,
            hq_positions={0: [], 1: []},
            lab_positions={0: [], 1: []},
            country_to_player={},
            predeployed_specs=[],
        )
        st = make_initial_state(
            md, 1, 1, starting_funds=0, tier_name="T2", replay_first_mover=1
        )
        st.units = {0: [], 1: []}
        capper = _make_unit(st, UnitType.INFANTRY, 1, (3, 1))
        _make_unit(st, UnitType.INFANTRY, 1, (4, 2))
        st.active_player = 1
        st.action_stage = ActionStage.SELECT
        st.selected_unit = None
        st.selected_move_pos = None
        apply_oracle_action_json(
            st,
            {
                "action": "Capt",
                "Move": [],
                "Capt": {
                    "buildingInfo": {"buildings_y": 3, "buildings_x": 2},
                    "unit": {
                        "global": {
                            "units_y": 3,
                            "units_x": 1,
                            "units_players_id": 501,
                            "units_id": 999001,
                        }
                    },
                },
            },
            {501: 1},
            envelope_awbw_player_id=501,
        )
        self.assertLess(st.get_property_at(3, 2).capture_points, 20)
        self.assertEqual(st.get_unit_at(3, 2).player, 1)

    def test_no_capturer_tags_drift_not_resolver(self) -> None:
        """Empty neighborhood on a property tile → [drift] (engine vs zip mismatch)."""
        neu = 133
        sea, shoal, plain = 28, 29, 1
        terrain = [
            [sea] * 5,
            [shoal] * 5,
            [plain] * 5,
            [plain] * 5,
            [plain] * 5,
        ]
        terrain[3][2] = neu
        info = get_terrain(neu)
        props = [
            PropertyState(
                terrain_id=neu,
                row=3,
                col=2,
                owner=None,
                capture_points=20,
                is_hq=info.is_hq,
                is_lab=info.is_lab,
                is_comm_tower=info.is_comm_tower,
                is_base=info.is_base,
                is_airport=info.is_airport,
                is_port=info.is_port,
            )
        ]
        md = MapData(
            map_id=999_995,
            name="oracle_capt_drift",
            map_type="std",
            terrain=terrain,
            height=5,
            width=5,
            cap_limit=999,
            unit_limit=50,
            unit_bans=[],
            tiers=[],
            objective_type=None,
            properties=props,
            hq_positions={0: [], 1: []},
            lab_positions={0: [], 1: []},
            country_to_player={},
            predeployed_specs=[],
        )
        st = make_initial_state(md, 1, 1, starting_funds=0, tier_name="T2")
        st.units = {0: [], 1: []}
        st.active_player = 0
        st.action_stage = ActionStage.SELECT
        with self.assertRaises(UnsupportedOracleAction) as ctx:
            apply_oracle_action_json(
                st,
                {
                    "action": "Capt",
                    "Move": [],
                    "Capt": {"buildingInfo": {"buildings_y": 3, "buildings_x": 2}},
                },
                {100: 0},
                envelope_awbw_player_id=100,
            )
        self.assertIn("drift", str(ctx.exception).lower())


class TestOracleUnloadMisalignedGlobalTile(unittest.TestCase):
    """Site ``unit.global`` can sit far from any legal orthogonal drop (oracle_unload)."""

    def test_unload_picks_closest_legal_drop_to_hint(self) -> None:
        state = _fresh_state()
        state.active_player = 0
        _make_unit(state, UnitType.TANK, 0, (1, 2))
        _make_unit(state, UnitType.TANK, 0, (2, 1))
        apc = _make_unit(state, UnitType.APC, 0, (2, 2))
        inf = _make_unit(state, UnitType.INFANTRY, 0, (3, 2))
        state.action_stage = ActionStage.SELECT
        state.selected_unit = None
        state.selected_move_pos = None
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(3, 2)))
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(3, 2), move_pos=(2, 2)))
        load = next(a for a in get_legal_actions(state) if a.action_type == ActionType.LOAD)
        state.step(load)
        for a in get_legal_actions(state):
            if a.action_type == ActionType.WAIT:
                state.step(a)
                break
        apc.moved = False
        state.action_stage = ActionStage.SELECT
        state.selected_unit = None
        state.selected_move_pos = None
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(2, 2)))
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(2, 2), move_pos=(2, 2)))

        apply_oracle_action_json(
            state,
            {
                "action": "Unload",
                "transportID": apc.unit_id,
                "unit": {
                    "global": {
                        "units_id": inf.unit_id,
                        "units_y": 9,
                        "units_x": 9,
                        "units_name": "Infantry",
                        "units_players_id": 0,
                    }
                },
            },
            {0: 0},
        )
        dropped = state.get_unit_at(2, 3)
        self.assertIsNotNone(dropped)
        self.assertEqual(dropped.unit_type, UnitType.INFANTRY)
        self.assertEqual(len(apc.loaded_units), 0)


class TestOracleUnloadPerSeatUnitOnly(unittest.TestCase):
    """GL-style ``Unload``: no ``unit.global``; cargo lives under ``unit[<p: seat id>]``."""

    def test_unload_resolves_cargo_from_envelope_seat_bucket(self) -> None:
        state = _fresh_state()
        state.active_player = 0
        _make_unit(state, UnitType.TANK, 0, (1, 2))
        _make_unit(state, UnitType.TANK, 0, (2, 1))
        apc = _make_unit(state, UnitType.APC, 0, (2, 2))
        inf = _make_unit(state, UnitType.INFANTRY, 0, (3, 2))
        state.action_stage = ActionStage.SELECT
        state.selected_unit = None
        state.selected_move_pos = None
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(3, 2)))
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(3, 2), move_pos=(2, 2)))
        load = next(a for a in get_legal_actions(state) if a.action_type == ActionType.LOAD)
        state.step(load)
        for a in get_legal_actions(state):
            if a.action_type == ActionType.WAIT:
                state.step(a)
                break
        apc.moved = False
        state.action_stage = ActionStage.SELECT
        state.selected_unit = None
        state.selected_move_pos = None
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(2, 2)))
        state.step(Action(ActionType.SELECT_UNIT, unit_pos=(2, 2), move_pos=(2, 2)))

        apply_oracle_action_json(
            state,
            {
                "action": "Unload",
                "transportID": apc.unit_id,
                "unit": {
                    "0": {
                        "units_id": inf.unit_id,
                        "units_y": 9,
                        "units_x": 9,
                        "units_name": "Infantry",
                        "units_players_id": 0,
                    }
                },
            },
            {0: 0},
            envelope_awbw_player_id=0,
        )
        dropped = state.get_unit_at(2, 3)
        self.assertIsNotNone(dropped)
        self.assertEqual(dropped.unit_type, UnitType.INFANTRY)
        self.assertEqual(len(apc.loaded_units), 0)


class TestOracleBuildNoopGuard(unittest.TestCase):
    """Site ``Build``: most engine refusals surface as ``UnsupportedOracleAction`` (Build no-op).

    When the engine refuses only for ``insufficient funds``, the handler returns
    (PHP may still emit ``Build`` for a no-spend client/server edge).
    """

    def _build_site_json(self) -> dict:
        return {
            "action": "Build",
            "newUnit": {
                "global": {
                    "units_id": 999001,
                    "units_players_id": 9001,
                    "units_name": "Infantry",
                    "units_y": 0,
                    "units_x": 1,
                    "units_movement_points": 3,
                    "units_vision": 2,
                    "units_fuel": 99,
                    "units_fuel_per_turn": 0,
                    "units_sub_dive": "N",
                    "units_ammo": 0,
                    "units_short_range": 0,
                    "units_long_range": 0,
                    "units_second_weapon": "N",
                    "units_symbol": "G",
                    "units_cost": 1000,
                    "units_movement_type": "F",
                    "units_moved": 0,
                    "units_capture": 0,
                    "units_fired": 0,
                    "units_hit_points": 10,
                    "units_cargo1_units_id": 0,
                    "units_cargo2_units_id": 0,
                    "units_carried": "N",
                    "countries_code": "os",
                }
            },
        }

    def test_build_noop_raises_under_strict_default(self) -> None:
        state = _minimal_state(active_player=0, factory_owner=1)
        obj = self._build_site_json()
        with self.assertRaises(UnsupportedOracleAction) as ctx:
            apply_oracle_action_json(
                state, obj, {9001: 0}, envelope_awbw_player_id=9001
            )
        self.assertIn("Build no-op", str(ctx.exception))
        self.assertIn("(0,1)", str(ctx.exception))

    # Phase 7: deleted test_build_noop_silent_when_strict_disabled — see logs/phase7_test_cleanup.log
    # Phase 7: deleted test_build_on_neutral_factory_snaps_owner_then_succeeds — see logs/phase7_test_cleanup.log

    def test_build_success_does_not_trigger_guard(self) -> None:
        state = _minimal_state(active_player=0, factory_owner=0)
        obj = self._build_site_json()
        apply_oracle_action_json(
            state, obj, {9001: 0}, envelope_awbw_player_id=9001
        )
        self.assertEqual(len(state.units[0]), 1)

    # Phase 7: deleted test_site_trusted_build_snaps_wrong_owner_factory — see logs/phase7_test_cleanup.log
    # Phase 7: deleted test_site_trusted_build_funds_hint_unblocks_build — see logs/phase7_test_cleanup.log

    def test_build_nudges_friendly_off_factory_tile(self) -> None:
        """Friendly unit sitting on the base: zip still lists a legal Build there."""
        terrain = [[1, 1, 35]]
        prop = PropertyState(
            terrain_id=35,
            row=0,
            col=2,
            owner=0,
            capture_points=20,
            is_hq=False,
            is_lab=False,
            is_comm_tower=False,
            is_base=True,
            is_airport=False,
            is_port=False,
        )
        md = MapData(
            map_id=777_001,
            name="build-nudge",
            map_type="std",
            terrain=terrain,
            height=1,
            width=3,
            cap_limit=99,
            unit_limit=50,
            unit_bans=[],
            tiers=[],
            objective_type=None,
            properties=[prop],
            hq_positions={0: [], 1: []},
            lab_positions={0: [], 1: []},
            country_to_player={},
            predeployed_specs=[],
        )
        state = GameState(
            map_data=md,
            units={0: [], 1: []},
            funds=[10_000, 10_000],
            co_states=[make_co_state_safe(0), make_co_state_safe(0)],
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
        _make_unit(state, UnitType.INFANTRY, 0, (0, 2))
        obj = self._build_site_json()
        obj["newUnit"]["global"]["units_x"] = 2
        obj["newUnit"]["global"]["units_y"] = 0
        apply_oracle_action_json(
            state, obj, {9001: 0, 9002: 1}, envelope_awbw_player_id=9001
        )
        self.assertEqual(len(state.units[0]), 2)
        self.assertIsNotNone(state.get_unit_at(0, 2))
        self.assertIsNotNone(state.get_unit_at(0, 1))


class TestOracleCaptOuterRing(unittest.TestCase):
    """``_oracle_capt_no_path_outer_ring_capturers`` when an approach cell is occupied."""

    def test_finds_capturer_when_cardinal_of_building_is_blocked(self) -> None:
        from tools.oracle_zip_replay import _oracle_capt_no_path_outer_ring_capturers

        state = _fresh_state()
        er, ec = 2, 2
        state.map_data.terrain[er][ec] = 34  # neutral city
        _make_unit(state, UnitType.INFANTRY, 0, (2, 3))
        cap = _make_unit(state, UnitType.INFANTRY, 1, (2, 4))
        out = _oracle_capt_no_path_outer_ring_capturers(state, er, ec)
        self.assertEqual(len(out), 1)
        self.assertIs(out[0], cap)


class TestOracleAttackSeamCatalogZip(unittest.TestCase):
    """Regression: ``AttackSeam`` with ``Move: []`` and ``combatInfo`` under player-id keys only."""

    @unittest.skipUnless(
        (ROOT / "replays" / "amarriner_gl" / "1628539.zip").is_file(),
        "catalog replay zip not present",
    )
    def test_gl_1628539_attackseam_no_path_replays(self) -> None:
        from tools.amarriner_catalog_cos import pair_catalog_cos_ids

        z = ROOT / "replays" / "amarriner_gl" / "1628539.zip"
        cat = json.loads((ROOT / "data" / "amarriner_gl_std_catalog.json").read_text())[
            "games"
        ]["1628539"]
        co0, co1 = pair_catalog_cos_ids(cat)
        r = replay_oracle_zip(
            z,
            map_pool=MAP_POOL,
            maps_dir=MAPS_DIR,
            map_id=int(cat["map_id"]),
            co0=co0,
            co1=co1,
            tier_name=str(cat.get("tier") or "T2"),
        )
        self.assertGreater(r.actions_applied, 300)


if __name__ == "__main__":
    unittest.main()
