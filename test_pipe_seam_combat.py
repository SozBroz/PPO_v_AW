"""Pipe seam combat — AWBW parity.

Pins seam-combat rules from https://awbw.fandom.com/wiki/Pipes_and_Pipeseams
and the ``awbw-engine-parity`` plan:

1. Intact seams (terrain 113 / 114) are **legal ATTACK targets** even when
   no defender unit occupies the tile — the legal-action pipeline must
   expose seam coordinates to eligible attackers.
2. Seams have 99 HP. Artillery (70 base damage vs seam) at full HP breaks
   a fresh seam in two hits (70 + 70 >= 99), not one.
3. On break the terrain ID flips: 113 → 115 (HPipe Rubble), 114 → 116
   (VPipe Rubble). Seam HP state is cleared from ``GameState.seam_hp``.
4. Broken seams behave like Plains for movement — Piperunners **cannot**
   traverse them, and other ground types gain access.
5. Luck does not apply to seam damage (deterministic).
"""
from __future__ import annotations

import unittest

from engine.action import (
    Action, ActionType, ActionStage,
    get_legal_actions, get_attack_targets,
)
from engine.combat import (
    calculate_seam_damage, get_seam_base_damage,
)
from engine.game import make_initial_state, SEAM_MAX_HP
from engine.map_loader import MapData
from engine.unit import Unit, UnitType, UNIT_STATS
from engine.terrain import get_move_cost, INF_PASSABLE, MOVE_PIPELINE, MOVE_TREAD


PLAIN = 1
SEAM_H = 113
SEAM_V = 114
RUBBLE_H = 115
RUBBLE_V = 116


def _seam_map(seam_row: int = 2, seam_col: int = 2, seam_id: int = SEAM_H) -> MapData:
    """5x5 plains with a single seam tile at (row, col)."""
    terrain = [[PLAIN] * 5 for _ in range(5)]
    terrain[seam_row][seam_col] = seam_id
    return MapData(
        map_id=999_990 + seam_id,
        name="seam_test",
        map_type="std",
        terrain=terrain,
        height=5,
        width=5,
        cap_limit=999,
        unit_limit=50,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=[],
        hq_positions={},
        lab_positions={},
        country_to_player={},
        predeployed_specs=[],
    )


def _fresh(seam_id: int = SEAM_H) -> tuple:
    md = _seam_map(seam_id=seam_id)
    st = make_initial_state(md, 1, 1, starting_funds=0, tier_name="T2")
    st.units = {0: [], 1: []}
    st.active_player = 0
    return st


def _make_unit(
    state,
    unit_type: UnitType,
    player: int,
    pos: tuple[int, int],
    *,
    hp: int = 100,
) -> Unit:
    stats = UNIT_STATS[unit_type]
    u = Unit(
        unit_type=unit_type,
        player=player,
        hp=hp,
        ammo=stats.max_ammo if stats.max_ammo > 0 else 0,
        fuel=stats.max_fuel,
        pos=pos,
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
        unit_id=state._allocate_unit_id(),
    )
    state.units[player].append(u)
    return u


def _select_and_move(state, unit: Unit, dest: tuple[int, int]) -> None:
    state.action_stage = ActionStage.SELECT
    state.selected_unit = None
    state.selected_move_pos = None
    state.step(Action(ActionType.SELECT_UNIT, unit_pos=unit.pos))
    state.step(Action(ActionType.SELECT_UNIT, unit_pos=unit.pos, move_pos=dest))


# ---------------------------------------------------------------------------
# Targetability
# ---------------------------------------------------------------------------

class TestSeamTargetability(unittest.TestCase):
    """Seams must be enumerated as attack targets and legal ACTION targets."""

    def test_get_attack_targets_includes_empty_seam_tile(self) -> None:
        state = _fresh()
        # Artillery at (2, 0); seam at (2, 2), Manhattan distance 2 (in range).
        arty = _make_unit(state, UnitType.ARTILLERY, 0, (2, 0))
        targets = get_attack_targets(state, arty, arty.pos)
        self.assertIn((2, 2), targets)

    def test_legal_actions_expose_seam_attack(self) -> None:
        state = _fresh()
        arty = _make_unit(state, UnitType.ARTILLERY, 0, (2, 0))
        _select_and_move(state, arty, arty.pos)
        acts = get_legal_actions(state)
        seam_attacks = [
            a for a in acts
            if a.action_type == ActionType.ATTACK and a.target_pos == (2, 2)
        ]
        self.assertEqual(len(seam_attacks), 1)

    def test_unarmed_transport_does_not_target_seam(self) -> None:
        state = _fresh()
        apc = _make_unit(state, UnitType.APC, 0, (2, 0))
        targets = get_attack_targets(state, apc, apc.pos)
        self.assertNotIn((2, 2), targets)

    def test_seam_base_damage_table_covers_key_units(self) -> None:
        self.assertEqual(get_seam_base_damage(UnitType.ARTILLERY), 70)
        self.assertEqual(get_seam_base_damage(UnitType.ROCKET), 80)
        self.assertEqual(get_seam_base_damage(UnitType.BATTLESHIP), 80)
        self.assertEqual(get_seam_base_damage(UnitType.PIPERUNNER), 80)
        self.assertIsNone(get_seam_base_damage(UnitType.APC))


# ---------------------------------------------------------------------------
# Damage resolution / break / terrain flip
# ---------------------------------------------------------------------------

class TestSeamAttackResolution(unittest.TestCase):

    def test_artillery_damages_seam_without_breaking_it_first_hit(self) -> None:
        state = _fresh()
        arty = _make_unit(state, UnitType.ARTILLERY, 0, (2, 0))
        _select_and_move(state, arty, arty.pos)
        state.step(Action(
            ActionType.ATTACK,
            unit_pos=arty.pos, move_pos=arty.pos, target_pos=(2, 2),
        ))

        # Seam HP drops from 99 → 29 (70 base, full HP, 100 AV).
        self.assertEqual(state.seam_hp.get((2, 2)), 29)
        # Terrain still intact (still a seam).
        self.assertEqual(state.map_data.terrain[2][2], SEAM_H)
        # Ammo consumed on the attacker.
        self.assertEqual(
            arty.ammo,
            UNIT_STATS[UnitType.ARTILLERY].max_ammo - 1,
        )

    def test_two_artillery_hits_break_the_seam(self) -> None:
        state = _fresh()
        arty = _make_unit(state, UnitType.ARTILLERY, 0, (2, 0))

        for _ in range(2):
            _select_and_move(state, arty, arty.pos)
            state.step(Action(
                ActionType.ATTACK,
                unit_pos=arty.pos, move_pos=arty.pos, target_pos=(2, 2),
            ))
            arty.moved = False  # allow the same unit to fire again this test

        # Seam broke → rubble terrain, no HP tracked.
        self.assertEqual(state.map_data.terrain[2][2], RUBBLE_H)
        self.assertNotIn((2, 2), state.seam_hp)

    def test_vertical_seam_flips_to_vertical_rubble(self) -> None:
        state = _fresh(seam_id=SEAM_V)
        # Megatank one-shots a full seam (135 base >= 99).
        mega = _make_unit(state, UnitType.MEGA_TANK, 0, (2, 1))
        _select_and_move(state, mega, mega.pos)
        state.step(Action(
            ActionType.ATTACK,
            unit_pos=mega.pos, move_pos=mega.pos, target_pos=(2, 2),
        ))
        self.assertEqual(state.map_data.terrain[2][2], RUBBLE_V)

    def test_rubble_tile_stays_rubble_across_extra_seam_strikes(self) -> None:
        """AWBW allows repeated AttackSeam vs 115/116; terrain must not flip to plain(1)."""
        state = _fresh(seam_id=SEAM_V)
        arty = _make_unit(state, UnitType.ARTILLERY, 0, (2, 0))
        state.map_data.terrain[2][2] = RUBBLE_V
        state.seam_hp.pop((2, 2), None)
        _select_and_move(state, arty, arty.pos)
        state.step(Action(
            ActionType.ATTACK,
            unit_pos=arty.pos, move_pos=arty.pos, target_pos=(2, 2),
        ))
        self.assertEqual(state.map_data.terrain[2][2], RUBBLE_V)

    def test_calculate_seam_damage_luck_free(self) -> None:
        """Seam damage must not vary across calls (no luck roll)."""
        from engine.co import make_co_state_safe
        from engine.terrain import get_terrain
        att = Unit(
            unit_type=UnitType.ARTILLERY, player=0, hp=100,
            ammo=9, fuel=50, pos=(2, 0), moved=False,
            loaded_units=[], is_submerged=False, capture_progress=20,
            unit_id=1,
        )
        co = make_co_state_safe(1)
        plain = get_terrain(PLAIN)
        results = {calculate_seam_damage(att, plain, co) for _ in range(50)}
        self.assertEqual(len(results), 1, f"Luck leaked into seam damage: {results}")

    def test_initial_state_populates_seam_hp(self) -> None:
        state = _fresh()
        self.assertEqual(state.seam_hp.get((2, 2)), SEAM_MAX_HP)


# ---------------------------------------------------------------------------
# Post-break movement rules
# ---------------------------------------------------------------------------

class TestBrokenSeamMovement(unittest.TestCase):
    """Broken seam (Rubble) tiles are Plains-like for ground, closed to Piperunner."""

    def test_piperunner_cannot_cross_broken_seam(self) -> None:
        # Broken H-seam uses plain costs; Piperunner's MOVE_PIPELINE is absent.
        self.assertGreaterEqual(
            get_move_cost(RUBBLE_H, MOVE_PIPELINE), INF_PASSABLE,
        )
        self.assertGreaterEqual(
            get_move_cost(RUBBLE_V, MOVE_PIPELINE), INF_PASSABLE,
        )

    def test_tank_can_cross_broken_seam(self) -> None:
        # Rubble = plains → tread units pay 1 MP.
        self.assertEqual(get_move_cost(RUBBLE_H, MOVE_TREAD), 1)
        self.assertEqual(get_move_cost(RUBBLE_V, MOVE_TREAD), 1)


# ---------------------------------------------------------------------------
# Terrain grid isolation across games
# ---------------------------------------------------------------------------

class TestMapDataTerrainIsolation(unittest.TestCase):
    """Breaking a seam in one game must not leak into another with same MapData."""

    def test_independent_games_keep_independent_terrain(self) -> None:
        md = _seam_map()
        s1 = make_initial_state(md, 1, 1, starting_funds=0, tier_name="T2")
        s2 = make_initial_state(md, 1, 1, starting_funds=0, tier_name="T2")

        # Break the seam in s1 manually.
        s1.map_data.terrain[2][2] = RUBBLE_H
        s1.seam_hp.pop((2, 2), None)

        self.assertEqual(s2.map_data.terrain[2][2], SEAM_H,
                         "Second game's terrain must remain intact.")
        self.assertEqual(s2.seam_hp.get((2, 2)), SEAM_MAX_HP)


if __name__ == "__main__":
    unittest.main()
