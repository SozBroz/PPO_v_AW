"""Lander logic, transport co-occupancy, cargo death, and per-move fuel.

Pins the rules fixed alongside the replay 163425 investigation:

  1. A unit cannot WAIT onto a friendly transport that could carry it — the
     only legal end-state is LOAD.
  2. A Lander holds at most 2 cargo (capacity guard in `_apply_load`).
  3. When a transport is destroyed in combat, every unit aboard dies with it.
  4. UNLOAD drops one cargo onto an adjacent legal tile; multi-drop turns are
     supported (stage stays in ACTION until the transport is empty).
  5. Movement deducts fuel equal to the path's movement-point cost.
  6. Out-of-fuel and low-fuel units have shrunken reachability.
"""
from __future__ import annotations

import unittest

from engine.action import (
    Action, ActionType, ActionStage,
    get_legal_actions, get_reachable_tiles, compute_reachable_costs,
)
from engine.game import GameState, make_initial_state
from engine.map_loader import MapData
from engine.unit import Unit, UnitType, UNIT_STATS


# Terrain ids: 1=Plain, 28=Sea, 29=Shoal (accepts ground + lander)
SEA, SHOAL, PLAIN = 28, 29, 1


def _build_map() -> MapData:
    """5x5 map: row 0 sea, row 1 shoal, rows 2-4 plain."""
    terrain = [
        [SEA] * 5,
        [SHOAL] * 5,
        [PLAIN] * 5,
        [PLAIN] * 5,
        [PLAIN] * 5,
    ]
    return MapData(
        map_id=999_999,
        name="lander_fuel_test",
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


def _make_unit(
    state: GameState,
    unit_type: UnitType,
    player: int,
    pos: tuple[int, int],
    *,
    fuel: int | None = None,
    hp: int = 100,
) -> Unit:
    stats = UNIT_STATS[unit_type]
    u = Unit(
        unit_type=unit_type,
        player=player,
        hp=hp,
        ammo=stats.max_ammo if stats.max_ammo > 0 else 0,
        fuel=stats.max_fuel if fuel is None else fuel,
        pos=pos,
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
        unit_id=state._allocate_unit_id(),
    )
    state.units[player].append(u)
    return u


def _fresh_state() -> GameState:
    md = _build_map()
    st = make_initial_state(md, 1, 1, starting_funds=0, tier_name="T2")
    st.units = {0: [], 1: []}
    return st


def _select_and_move(state: GameState, unit: Unit, dest: tuple[int, int]) -> None:
    """Walk a unit through SELECT -> MOVE stages, leaving it ready for ACTION."""
    state.action_stage      = ActionStage.SELECT
    state.selected_unit     = None
    state.selected_move_pos = None
    state.step(Action(ActionType.SELECT_UNIT, unit_pos=unit.pos))
    state.step(Action(ActionType.SELECT_UNIT, unit_pos=unit.pos, move_pos=dest))


# ---------------------------------------------------------------------------
# Transport co-occupancy and capacity
# ---------------------------------------------------------------------------

class TestLanderLoadingRules(unittest.TestCase):
    """LOAD must replace WAIT when the destination tile is a friendly transport."""

    def setUp(self) -> None:
        self.state = _fresh_state()
        self.state.active_player = 0
        # Lander parked on shoal tile (1, 2); infantry below on plains.
        self.lander = _make_unit(self.state, UnitType.LANDER, 0, (1, 2))
        self.inf    = _make_unit(self.state, UnitType.INFANTRY, 0, (2, 2))

    def test_load_is_only_terminator_on_friendly_transport_tile(self) -> None:
        _select_and_move(self.state, self.inf, (1, 2))
        legal = get_legal_actions(self.state)
        types = {a.action_type for a in legal}
        self.assertEqual(
            types, {ActionType.LOAD},
            "Boarding a friendly transport must be the *only* legal terminator "
            "(WAIT here would co-occupy the tile).",
        )

    def test_wait_onto_loadable_transport_is_rejected(self) -> None:
        _select_and_move(self.state, self.inf, (1, 2))
        with self.assertRaises(ValueError):
            self.state.step(Action(
                ActionType.WAIT, unit_pos=self.inf.pos, move_pos=(1, 2),
            ))

    def test_third_unit_cannot_board_full_lander(self) -> None:
        # Pre-load 2 infantry directly into lander to simulate full state.
        for r in (3, 4):
            extra = _make_unit(self.state, UnitType.INFANTRY, 0, (r, 2))
            self.state.units[0].remove(extra)
            self.lander.loaded_units.append(extra)
        self.assertEqual(len(self.lander.loaded_units), 2)

        # Reachability must not include the lander tile any more.
        reachable = get_reachable_tiles(self.state, self.inf)
        self.assertNotIn((1, 2), reachable)

        # And a hand-crafted LOAD must be rejected with a capacity error.
        self.inf.moved = False
        with self.assertRaises(ValueError):
            self.state._apply_load(Action(
                ActionType.LOAD, unit_pos=self.inf.pos, move_pos=(1, 2),
            ))

    def test_apply_load_rejects_incompatible_cargo_type(self) -> None:
        # Cruiser cannot carry infantry — a manual LOAD must raise.
        cruiser = _make_unit(self.state, UnitType.CRUISER, 0, (0, 4))
        with self.assertRaises(ValueError):
            self.state._apply_load(Action(
                ActionType.LOAD,
                unit_pos=self.inf.pos,
                move_pos=cruiser.pos,
            ))


# ---------------------------------------------------------------------------
# Transport death takes its cargo
# ---------------------------------------------------------------------------

class TestTransportDeathKillsCargo(unittest.TestCase):
    """A destroyed Lander must not leave its cargo orphaned."""

    def test_cargo_dies_with_lander(self) -> None:
        st = _fresh_state()
        st.active_player = 1   # P1 attacks the P0 lander

        lander  = _make_unit(st, UnitType.LANDER, 0, (1, 2), hp=10)  # 1 display HP
        cargo_a = _make_unit(st, UnitType.INFANTRY, 0, (3, 2))
        cargo_b = _make_unit(st, UnitType.INFANTRY, 0, (3, 3))
        # Move cargo into the lander manually (skip the SELECT/MOVE machinery).
        for c in (cargo_a, cargo_b):
            st.units[0].remove(c)
            lander.loaded_units.append(c)
        self.assertEqual(len(lander.loaded_units), 2)

        # Adjacent enemy battleship — strong enough to one-shot the lander.
        bship = _make_unit(st, UnitType.BATTLESHIP, 1, (0, 2))

        before_p0_units = st.losses_units[0]

        # Indirect attack (battleship is range 2-6) — keep at original tile.
        st.step(Action(ActionType.SELECT_UNIT, unit_pos=bship.pos))
        st.step(Action(
            ActionType.SELECT_UNIT, unit_pos=bship.pos, move_pos=bship.pos,
        ))
        st.step(Action(
            ActionType.ATTACK,
            unit_pos=bship.pos, move_pos=bship.pos,
            target_pos=lander.pos,
        ))

        # Lander gone from the board, cargo gone too, losses tallied.
        self.assertNotIn(lander, st.units[0],
                         "Destroyed lander should have been pruned from units[].")
        self.assertEqual(cargo_a.hp, 0)
        self.assertEqual(cargo_b.hp, 0)
        self.assertEqual(lander.loaded_units, [])
        self.assertEqual(
            st.losses_units[0] - before_p0_units, 3,
            "Three P0 units (lander + 2 cargo) should be tallied as lost.",
        )


# ---------------------------------------------------------------------------
# Unload mechanics
# ---------------------------------------------------------------------------

class TestUnloadMechanics(unittest.TestCase):
    """UNLOAD drops a single cargo onto an adjacent walkable tile."""

    def setUp(self) -> None:
        self.state = _fresh_state()
        self.state.active_player = 0

        self.lander = _make_unit(self.state, UnitType.LANDER, 0, (1, 2))
        cargo = _make_unit(self.state, UnitType.INFANTRY, 0, (2, 2))
        self.state.units[0].remove(cargo)
        self.lander.loaded_units.append(cargo)
        self.cargo = cargo

    def test_unload_emits_legal_drop_tiles(self) -> None:
        _select_and_move(self.state, self.lander, self.lander.pos)
        legal = get_legal_actions(self.state)
        unload_actions = [a for a in legal if a.action_type == ActionType.UNLOAD]
        # Drop tiles must be 4-adjacent shoals/plains, never the sea tile (0, 2).
        drop_tiles = {a.target_pos for a in unload_actions}
        self.assertIn((2, 2), drop_tiles)   # plain south
        self.assertIn((1, 1), drop_tiles)   # shoal west
        self.assertIn((1, 3), drop_tiles)   # shoal east
        self.assertNotIn((0, 2), drop_tiles, "Sea is impassable for infantry.")

    def test_unload_places_cargo_and_finalizes(self) -> None:
        _select_and_move(self.state, self.lander, self.lander.pos)
        self.state.step(Action(
            ActionType.UNLOAD,
            unit_pos=self.lander.pos,
            move_pos=self.lander.pos,
            target_pos=(2, 2),
            unit_type=UnitType.INFANTRY,
        ))
        self.assertEqual(self.lander.loaded_units, [])
        # Cargo back on the board at the drop tile, marked as moved.
        dropped = self.state.get_unit_at(2, 2)
        self.assertIsNotNone(dropped)
        self.assertEqual(dropped.unit_type, UnitType.INFANTRY)
        self.assertTrue(dropped.moved)
        # Empty transport: stage finalised.
        self.assertEqual(self.state.action_stage, ActionStage.SELECT)
        self.assertTrue(self.lander.moved)


# ---------------------------------------------------------------------------
# Per-move fuel deduction
# ---------------------------------------------------------------------------

class TestFuelOnMove(unittest.TestCase):
    """Movement deducts fuel equal to the path's movement-point cost."""

    def test_infantry_move_burns_fuel(self) -> None:
        st = _fresh_state()
        st.active_player = 0
        inf = _make_unit(st, UnitType.INFANTRY, 0, (4, 0))
        start_fuel = inf.fuel
        st._move_unit(inf, (4, 3))   # 3 plains tiles, cost 1 each
        self.assertEqual(inf.pos, (4, 3))
        self.assertEqual(inf.fuel, start_fuel - 3)

    def test_lander_move_burns_fuel(self) -> None:
        st = _fresh_state()
        st.active_player = 0
        lander = _make_unit(st, UnitType.LANDER, 0, (0, 0))
        start_fuel = lander.fuel
        st._move_unit(lander, (0, 4))   # 4 sea tiles, cost 1 each
        self.assertEqual(lander.fuel, start_fuel - 4)

    def test_low_fuel_shrinks_reachability(self) -> None:
        st = _fresh_state()
        st.active_player = 0
        inf = _make_unit(st, UnitType.INFANTRY, 0, (4, 0), fuel=1)
        costs = compute_reachable_costs(st, inf)
        # With 1 fuel and infantry move_range=3, only 1-step neighbours reachable.
        max_cost = max(costs.values())
        self.assertLessEqual(max_cost, 1)
        self.assertIn((3, 0), costs)
        self.assertIn((4, 1), costs)
        self.assertNotIn((4, 2), costs)

    def test_zero_fuel_cannot_move(self) -> None:
        st = _fresh_state()
        st.active_player = 0
        inf = _make_unit(st, UnitType.INFANTRY, 0, (4, 0), fuel=0)
        reachable = get_reachable_tiles(st, inf)
        self.assertEqual(reachable, {(4, 0)},
                         "Out-of-fuel unit can only stay on its starting tile.")
        with self.assertRaises(ValueError):
            st._move_unit(inf, (4, 1))


if __name__ == "__main__":
    unittest.main()
