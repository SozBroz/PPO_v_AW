"""Regression for ``tools.oracle_zip_replay._resolve_fire_or_seam_attacker`` (oracle_fire cluster)."""

from __future__ import annotations

import unittest

from engine.action import ActionStage, get_attack_targets
from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, Unit, UnitType

from server.play_human import MAPS_DIR, POOL_PATH
from tools.oracle_zip_replay import (
    _oracle_fire_attack_move_pos_candidates,
    _oracle_fire_no_attacker_message_suffix,
    _oracle_fire_resolve_defender_target_pos,
    _oracle_resolve_fire_move_pos,
    _oracle_try_grit_jake_indirect_fire,
    _resolve_fire_or_seam_attacker,
    apply_oracle_action_json,
)


def _tank(player: int, pos: tuple[int, int], *, hp: int, uid: int) -> Unit:
    st = UNIT_STATS[UnitType.TANK]
    return Unit(
        UnitType.TANK,
        player,
        hp,
        st.max_ammo,
        st.max_fuel,
        pos,
        False,
        [],
        False,
        20,
        uid,
    )


class TestOracleFireResolve(unittest.TestCase):
    def test_defender_combatinfo_tile_off_by_one_gl_1628008_shape(self) -> None:
        """Vision ``defender.units_y``/``x`` can sit one step off the engine occupant (GL 1628008)."""
        m = load_map(77060, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 1, 1, tier_name="T3", starting_funds=0)
        s.units[0] = []
        s.units[1] = []
        s.units[0].append(_tank(0, (12, 3), hp=70, uid=9001))
        s.units[1].append(_tank(1, (11, 3), hp=30, uid=9002))
        defender = {
            "units_y": 12,
            "units_x": 3,
            "units_id": 9002,
            "units_hit_points": 3,
        }
        tr, tc = _oracle_fire_resolve_defender_target_pos(
            s, defender, attacker_eng=0
        )
        self.assertEqual((tr, tc), (11, 3))

    def test_defender_prefers_combatinfo_tile_when_two_adjacent_foes_gl_1630151(self) -> None:
        """Tank orth to two enemy infantry: ``units_hit_points`` can tie-break to the wrong unit (1630151)."""
        ist = UNIT_STATS[UnitType.INFANTRY]
        s = make_initial_state(
            load_map(159501, POOL_PATH, MAPS_DIR),
            1,
            2,
            tier_name="T2",
            starting_funds=0,
            replay_first_mover=0,
        )
        s.units[0] = []
        s.units[1] = []
        s.units[0].append(
            Unit(
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
                17,
            )
        )
        s.units[0].append(
            Unit(
                UnitType.INFANTRY,
                0,
                47,
                ist.max_ammo,
                ist.max_fuel,
                (1, 1),
                False,
                [],
                False,
                20,
                9,
            )
        )
        s.units[1].append(_tank(1, (1, 2), hp=100, uid=19))
        defender = {
            "units_y": 0,
            "units_x": 2,
            "units_id": 192257579,
            "units_hit_points": 4,
        }
        tr, tc = _oracle_fire_resolve_defender_target_pos(
            s, defender, attacker_eng=1
        )
        self.assertEqual((tr, tc), (0, 2))

    def test_defender_ring_prefers_tile_strikable_from_attacker_anchor_gl_1632283(self) -> None:
        """PHP defender coords can neighbor two enemies; Manhattan tie-break picked the wrong foe (1632283)."""
        ist = UNIT_STATS[UnitType.INFANTRY]
        bst = UNIT_STATS[UnitType.B_COPTER]
        m = load_map(69201, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 1, 1, tier_name="T4", starting_funds=0)
        s.units[0] = []
        s.units[1] = []
        s.units[1].append(
            Unit(
                UnitType.INFANTRY,
                1,
                90,
                ist.max_ammo,
                ist.max_fuel,
                (9, 4),
                False,
                [],
                False,
                20,
                101,
            )
        )
        s.units[1].append(
            Unit(
                UnitType.INFANTRY,
                1,
                50,
                ist.max_ammo,
                ist.max_fuel,
                (10, 4),
                False,
                [],
                False,
                20,
                102,
            )
        )
        s.units[0].append(_tank(0, (9, 5), hp=17, uid=201))
        s.units[0].append(
            Unit(
                UnitType.B_COPTER,
                0,
                100,
                bst.max_ammo,
                bst.max_fuel,
                (11, 4),
                False,
                [],
                False,
                20,
                202,
            )
        )
        defender = {
            "units_y": 10,
            "units_x": 4,
            "units_id": 192517820,
            "units_hit_points": "?",
        }
        tr, tc = _oracle_fire_resolve_defender_target_pos(
            s, defender, attacker_eng=1, attacker_anchor=(9, 4)
        )
        self.assertEqual((tr, tc), (9, 5))

    def test_cross_seat_anchor_matches_gl_1609589(self) -> None:
        """AWBW ``combatInfo.attacker`` tile can hold the striking unit while ``p:`` envelope is the other seat (1609589)."""
        m = load_map(77060, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 1, 1, tier_name="T3", starting_funds=0)
        s.units[0] = []
        s.units[1] = []
        s.units[0].append(_tank(0, (11, 11), hp=56, uid=1002))
        s.units[1].append(_tank(1, (10, 11), hp=89, uid=1001))
        s.active_player = 0
        s.action_stage = ActionStage.SELECT
        s.selected_unit = None
        s.selected_move_pos = None
        u = _resolve_fire_or_seam_attacker(
            s,
            engine_player=0,
            awbw_units_id=191218076,
            anchor_r=10,
            anchor_c=11,
            target_r=11,
            target_c=11,
            hp_hint=9,
        )
        self.assertIsNotNone(u)
        assert u is not None
        self.assertEqual(u.player, 1)
        self.assertEqual(u.pos, (10, 11))

    def test_all_player_fallback_matches_gl_1613840(self) -> None:
        """Stale ``attacker.units_{y,x}`` on wrong tile; a legal striker exists on the other seat (1613840)."""
        m = load_map(173170, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 11, 11, tier_name="T4", starting_funds=0)
        s.units[0] = []
        s.units[1] = []
        s.units[0].append(_tank(0, (10, 6), hp=100, uid=2001))
        s.units[1].append(_tank(1, (10, 5), hp=68, uid=2002))
        s.units[1].append(_tank(1, (10, 4), hp=100, uid=2003))
        s.active_player = 1
        s.action_stage = ActionStage.SELECT
        s.selected_unit = None
        s.selected_move_pos = None
        u = _resolve_fire_or_seam_attacker(
            s,
            engine_player=1,
            awbw_units_id=191097245,
            anchor_r=10,
            anchor_c=4,
            target_r=10,
            target_c=5,
            hp_hint=10,
        )
        self.assertIsNotNone(u)
        assert u is not None
        self.assertEqual(u.player, 0)
        self.assertEqual(u.pos, (10, 6))

    def test_grit_cop_probe_matches_gl_1627004(self) -> None:
        """Grit artillery at distance 4 needs COP +1; site may omit ``Power`` before ``Fire``."""
        from engine.co import make_co_state

        m = load_map(77060, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 2, 9, tier_name="T2", starting_funds=0)
        s.units[0] = []
        s.units[1] = []
        s.co_states[1] = make_co_state(2)
        s.co_states[1].cop_active = False
        s.co_states[1].scop_active = False
        ast = UNIT_STATS[UnitType.ARTILLERY]
        ar = Unit(
            UnitType.ARTILLERY,
            1,
            100,
            ast.max_ammo,
            ast.max_fuel,
            (10, 12),
            False,
            [],
            False,
            20,
            501,
        )
        s.units[1].append(ar)
        ist = UNIT_STATS[UnitType.INFANTRY]
        s.units[0].append(
            Unit(
                UnitType.INFANTRY,
                0,
                100,
                ist.max_ammo,
                ist.max_fuel,
                (6, 12),
                False,
                [],
                False,
                20,
                502,
            )
        )
        s.active_player = 1
        s.action_stage = ActionStage.SELECT
        got = _oracle_try_grit_jake_indirect_fire(
            s, ar, ar.pos, 1, (6, 12)
        )
        self.assertIs(got, ar)
        self.assertTrue(s.co_states[1].cop_active ^ s.co_states[1].scop_active)

    def test_diag_triage_includes_grit_co_readonly_gl_1627004(self) -> None:
        """``strike_possible_in_engine`` must see Grit COP range like the grit probe (GL 1627004)."""
        from engine.co import make_co_state

        m = load_map(77060, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 2, 9, tier_name="T2", starting_funds=0)
        s.units[0] = []
        s.units[1] = []
        s.co_states[1] = make_co_state(2)
        s.co_states[1].cop_active = False
        s.co_states[1].scop_active = False
        ast = UNIT_STATS[UnitType.ARTILLERY]
        ar = Unit(
            UnitType.ARTILLERY,
            1,
            100,
            ast.max_ammo,
            ast.max_fuel,
            (10, 12),
            False,
            [],
            False,
            20,
            501,
        )
        s.units[1].append(ar)
        ist = UNIT_STATS[UnitType.INFANTRY]
        s.units[0].append(
            Unit(
                UnitType.INFANTRY,
                0,
                100,
                ist.max_ammo,
                ist.max_fuel,
                (6, 12),
                False,
                [],
                False,
                20,
                502,
            )
        )
        s.active_player = 1
        s.action_stage = ActionStage.SELECT
        self.assertNotIn((6, 12), get_attack_targets(s, ar, ar.pos))
        suf = _oracle_fire_no_attacker_message_suffix(s, 6, 12)
        self.assertIn("strike_possible_in_engine=1", suf)
        self.assertFalse(s.co_states[1].cop_active)
        self.assertFalse(s.co_states[1].scop_active)

    def test_no_attacker_suffix_strike_possible_flags_resolver_triage(self) -> None:
        """Probe tags resolver-side gaps when the engine still allows some strike to the tile."""
        m = load_map(77060, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 1, 1, tier_name="T3", starting_funds=0)
        s.units[0] = []
        s.units[1] = []
        s.units[0].append(_tank(0, (11, 11), hp=56, uid=1002))
        s.units[1].append(_tank(1, (10, 11), hp=89, uid=1001))
        s.active_player = 0
        s.action_stage = ActionStage.SELECT
        suf = _oracle_fire_no_attacker_message_suffix(s, 11, 11)
        self.assertIn("strike_possible_in_engine=1", suf)
        self.assertIn("resolver_gap_or_anchor", suf)

    def test_no_attacker_suffix_no_strike_flags_drift_triage(self) -> None:
        """Probe tags drift/unmapped-range when no unit can legally target that cell."""
        m = load_map(77060, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 1, 1, tier_name="T3", starting_funds=0)
        s.units[0] = []
        s.units[1] = []
        s.units[0].append(_tank(0, (11, 11), hp=56, uid=1002))
        s.active_player = 0
        s.action_stage = ActionStage.SELECT
        suf = _oracle_fire_no_attacker_message_suffix(s, 5, 5)
        self.assertIn("strike_possible_in_engine=0", suf)
        self.assertIn("drift_range_los_or_unmapped_co", suf)

    def test_direct_fire_move_pos_candidates_gl_bucket(self) -> None:
        """Direct strike legal from ``unit.pos`` while ``selected_move_pos`` is one step off."""
        m = load_map(77060, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 1, 1, tier_name="T3", starting_funds=0)
        s.units[0] = []
        s.units[1] = []
        atk = _tank(0, (5, 5), hp=55, uid=7001)
        s.units[0].append(atk)
        s.units[1].append(_tank(1, (4, 4), hp=70, uid=7002))
        s.active_player = 0
        s.action_stage = ActionStage.ACTION
        s.selected_unit = atk
        s.selected_move_pos = (5, 6)
        cands = _oracle_fire_attack_move_pos_candidates(s, atk)
        self.assertEqual(cands[0], (5, 5))
        self.assertIn((5, 6), cands)
        u = _resolve_fire_or_seam_attacker(
            s,
            engine_player=0,
            awbw_units_id=7001,
            anchor_r=5,
            anchor_c=6,
            target_r=4,
            target_c=4,
            hp_hint=None,
        )
        self.assertIsNotNone(u)
        assert u is not None
        self.assertIs(u, atk)

    def test_fire_move_pos_skips_transport_hex_gl_1624281(self) -> None:
        """Reachability marks a friendly transport as a walk end for boarding; ``ATTACK`` must not use that hex as ``move_pos``.

        :meth:`GameState._move_unit` does not auto-board — using the APC cell as
        ``move_pos`` stacks infantry and APC on one tile (GL 1624281).
        """
        m = load_map(77060, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 1, 8, tier_name="T3", starting_funds=0)
        s.units[0] = []
        s.units[1] = []
        ist = UNIT_STATS[UnitType.INFANTRY]
        ast = UNIT_STATS[UnitType.APC]
        inf = Unit(
            UnitType.INFANTRY,
            0,
            100,
            ist.max_ammo,
            ist.max_fuel,
            (11, 11),
            False,
            [],
            False,
            20,
            17001,
        )
        apc = Unit(
            UnitType.APC,
            0,
            100,
            ast.max_ammo,
            ast.max_fuel,
            (11, 12),
            False,
            [],
            False,
            20,
            17002,
        )
        s.units[0].extend([inf, apc])
        s.units[1].append(_tank(1, (12, 12), hp=70, uid=9901))
        s.active_player = 0
        s.action_stage = ActionStage.SELECT
        paths = [{"y": 11, "x": 12}]
        fire_pos = _oracle_resolve_fire_move_pos(s, inf, paths, (11, 12), (12, 12))
        self.assertEqual(fire_pos, (11, 11))

    def test_fire_no_path_applies_when_defender_snapshot_hp0_gl1629178(self) -> None:
        """ZIP lists post-strike defender HP=0 while the engine still has the unit — apply the kill (GL 1629178)."""
        md = load_map(159501, POOL_PATH, MAPS_DIR)
        s = make_initial_state(
            md, 1, 2, tier_name="T2", starting_funds=0, replay_first_mover=0
        )
        ist = UNIT_STATS[UnitType.INFANTRY]
        ast = UNIT_STATS[UnitType.ARTILLERY]
        s.units[0] = []
        s.units[1] = []
        inf = Unit(
            UnitType.INFANTRY,
            0,
            90,
            ist.max_ammo,
            ist.max_fuel,
            (1, 2),
            False,
            [],
            False,
            20,
            99001,
        )
        arty = Unit(
            UnitType.ARTILLERY,
            1,
            20,
            ast.max_ammo,
            ast.max_fuel,
            (1, 3),
            False,
            [],
            False,
            20,
            99002,
        )
        s.units[0].append(inf)
        s.units[1].append(arty)
        s.active_player = 0
        s.action_stage = ActionStage.SELECT
        obj = {
            "action": "Fire",
            "Move": [],
            "Fire": {
                "action": "Fire",
                "combatInfoVision": {
                    "global": {
                        "combatInfo": {
                            "attacker": {
                                "units_id": 99001,
                                "units_y": 2,
                                "units_x": 1,
                                "units_hit_points": 9,
                                "units_players_id": 90001,
                            },
                            "defender": {
                                "units_id": 99002,
                                "units_y": 3,
                                "units_x": 1,
                                "units_hit_points": 0,
                            },
                        }
                    }
                },
            },
        }
        apply_oracle_action_json(s, obj, {90001: 0}, envelope_awbw_player_id=90001)
        self.assertIsNone(s.get_unit_at(1, 3))


if __name__ == "__main__":
    unittest.main()
