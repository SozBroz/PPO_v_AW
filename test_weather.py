"""
Tests for AWBW weather simulation:
  1. effective_move_cost returns correct costs for Rain and Snow.
  2. CO immunity: Olaf units ignore snow; Drake units ignore rain.
  3. Clear weather is unchanged from base get_move_cost.
  4. Two-segment lifecycle: weather expires after exactly 2 _end_turn calls.
  5. Power hooks: Olaf COP/SCOP sets snow; Drake SCOP sets rain; Drake COP does not.
  6. Power conflict: second power overwrites the first.
  7. encode_state includes weather scalars at indices 13-15.
"""
from __future__ import annotations

import unittest

from engine.action import Action, ActionType, ActionStage
from engine.co import make_co_state_safe
from engine.game import GameState, make_initial_state
from engine.map_loader import MapData, PropertyState
from engine.terrain import (
    MOVE_INF, MOVE_MECH, MOVE_TREAD, MOVE_TIRE_A, MOVE_TIRE_B,
    MOVE_AIR, MOVE_SEA, MOVE_LANDER, INF_PASSABLE,
    get_move_cost,
)
from engine.unit import Unit, UnitType, UNIT_STATS
from engine.weather import effective_move_cost

# terrain IDs used in tests
TID_PLAIN    = 1
TID_MOUNTAIN = 2
TID_WOOD     = 3
TID_ROAD     = 15  # horizontal road
TID_SEA      = 28
TID_SHOAL    = 29


# ---------------------------------------------------------------------------
# Minimal map helpers
# ---------------------------------------------------------------------------

def _small_map(terrain: list[list[int]]) -> MapData:
    h = len(terrain)
    w = len(terrain[0])
    return MapData(
        map_id=0,
        name="weather-test",
        map_type="std",
        terrain=terrain,
        height=h,
        width=w,
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


def _state_with_weather(weather: str, p0_co_id: int = 1, p1_co_id: int = 1) -> GameState:
    """Minimal 3×3 all-plain state with an explicit weather."""
    md = _small_map([[TID_PLAIN, TID_PLAIN, TID_PLAIN],
                     [TID_PLAIN, TID_PLAIN, TID_PLAIN],
                     [TID_PLAIN, TID_PLAIN, TID_PLAIN]])
    return GameState(
        map_data=md,
        units={0: [], 1: []},
        funds=[10_000, 10_000],
        co_states=[make_co_state_safe(p0_co_id), make_co_state_safe(p1_co_id)],
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
        weather=weather,
        default_weather="clear",
        co_weather_segments_remaining=0,
    )


def _unit(player: int, unit_type: UnitType, pos=(0, 0)) -> Unit:
    stats = UNIT_STATS[unit_type]
    return Unit(
        unit_type=unit_type,
        player=player,
        hp=100,
        ammo=stats.max_ammo,
        fuel=stats.max_fuel,
        pos=pos,
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
    )


# ---------------------------------------------------------------------------
# 1. effective_move_cost — clear is identical to base
# ---------------------------------------------------------------------------

class TestClearWeatherIdentity(unittest.TestCase):
    def _check_all_move_types(self, tid: int) -> None:
        """Each unit's effective_move_cost in clear == get_move_cost for its own move_type."""
        from engine.unit import UNIT_STATS
        state = _state_with_weather("clear")
        for ut in [
            UnitType.INFANTRY,
            UnitType.MECH,
            UnitType.TANK,
            UnitType.RECON,
            UnitType.ARTILLERY,   # MOVE_TREAD (AWBW indirects use treads)
            UnitType.FIGHTER,
            UnitType.BATTLESHIP,
            UnitType.LANDER,
        ]:
            u = _unit(0, ut)
            move_type = UNIT_STATS[ut].move_type
            expected = get_move_cost(tid, move_type)
            actual   = effective_move_cost(state, u, tid)
            self.assertEqual(
                actual, expected,
                f"clear: tid={tid} unit={ut.name} move_type={move_type}: expected {expected}, got {actual}",
            )

    def test_clear_plain(self):
        self._check_all_move_types(TID_PLAIN)

    def test_clear_mountain(self):
        self._check_all_move_types(TID_MOUNTAIN)

    def test_clear_sea(self):
        self._check_all_move_types(TID_SEA)


# ---------------------------------------------------------------------------
# 2. Rain movement costs
# ---------------------------------------------------------------------------

class TestRainMovement(unittest.TestCase):
    def _cost(self, ut: UnitType, tid: int) -> int:
        state = _state_with_weather("rain")
        return effective_move_cost(state, _unit(0, ut), tid)

    def test_tread_plain_is_2(self):
        self.assertEqual(self._cost(UnitType.TANK, TID_PLAIN), 2)

    def test_tire_a_plain_is_3(self):
        self.assertEqual(self._cost(UnitType.RECON, TID_PLAIN), 3)

    def test_tread_wood_is_3(self):
        self.assertEqual(self._cost(UnitType.TANK, TID_WOOD), 3)

    def test_artillery_wood_matches_tread_rain(self):
        # Artillery uses MOVE_TREAD (same rain cost as tanks).
        self.assertEqual(self._cost(UnitType.ARTILLERY, TID_WOOD), 3)

    def test_infantry_plain_unchanged(self):
        self.assertEqual(self._cost(UnitType.INFANTRY, TID_PLAIN), 1)

    def test_air_plain_unchanged(self):
        self.assertEqual(self._cost(UnitType.FIGHTER, TID_PLAIN), 1)

    def test_road_unchanged(self):
        self.assertEqual(self._cost(UnitType.TANK, TID_ROAD), 1)


# ---------------------------------------------------------------------------
# 3. Snow movement costs
# ---------------------------------------------------------------------------

class TestSnowMovement(unittest.TestCase):
    def _cost(self, ut: UnitType, tid: int) -> int:
        state = _state_with_weather("snow")
        return effective_move_cost(state, _unit(0, ut), tid)

    def test_infantry_plain_is_2(self):
        self.assertEqual(self._cost(UnitType.INFANTRY, TID_PLAIN), 2)

    def test_infantry_mountain_is_4(self):
        # doubled from clear cost of 2
        self.assertEqual(self._cost(UnitType.INFANTRY, TID_MOUNTAIN), 4)

    def test_mech_mountain_is_2(self):
        # doubled from clear cost of 1
        self.assertEqual(self._cost(UnitType.MECH, TID_MOUNTAIN), 2)

    def test_air_plain_is_2(self):
        self.assertEqual(self._cost(UnitType.FIGHTER, TID_PLAIN), 2)

    def test_air_road_is_2(self):
        self.assertEqual(self._cost(UnitType.FIGHTER, TID_ROAD), 2)

    def test_sea_sea_is_2(self):
        self.assertEqual(self._cost(UnitType.BATTLESHIP, TID_SEA), 2)

    def test_lander_sea_is_2(self):
        self.assertEqual(self._cost(UnitType.LANDER, TID_SEA), 2)

    def test_road_ground_unchanged(self):
        self.assertEqual(self._cost(UnitType.TANK, TID_ROAD), 1)


# ---------------------------------------------------------------------------
# 4. CO immunity
# ---------------------------------------------------------------------------

class TestCoImmunity(unittest.TestCase):
    def test_olaf_immune_to_snow_plain(self):
        # Olaf co_id == 9
        state = _state_with_weather("snow", p0_co_id=9)
        u = _unit(0, UnitType.INFANTRY)
        # Should get clear-weather cost (1), not snow cost (2)
        self.assertEqual(effective_move_cost(state, u, TID_PLAIN), 1)

    def test_olaf_immune_to_snow_mountain_infantry(self):
        state = _state_with_weather("snow", p0_co_id=9)
        u = _unit(0, UnitType.INFANTRY)
        # Clear-weather mountain cost for infantry = 2
        self.assertEqual(effective_move_cost(state, u, TID_MOUNTAIN), 2)

    def test_olaf_opponent_not_immune_to_snow(self):
        # P0 is Olaf, P1 is Andy. P1 unit should still pay snow costs.
        state = _state_with_weather("snow", p0_co_id=9, p1_co_id=1)
        u = _unit(1, UnitType.INFANTRY)
        self.assertEqual(effective_move_cost(state, u, TID_PLAIN), 2)

    def test_drake_immune_to_rain_plain(self):
        # Drake co_id == 5
        state = _state_with_weather("rain", p0_co_id=5)
        u = _unit(0, UnitType.TANK)
        # Clear-weather plain cost for tread = 1, not rain cost (2)
        self.assertEqual(effective_move_cost(state, u, TID_PLAIN), 1)

    def test_drake_opponent_not_immune_to_rain(self):
        state = _state_with_weather("rain", p0_co_id=5, p1_co_id=1)
        u = _unit(1, UnitType.TANK)
        self.assertEqual(effective_move_cost(state, u, TID_PLAIN), 2)

    def test_olaf_not_immune_to_rain(self):
        # Olaf is penalised under rain (same as everyone else)
        state = _state_with_weather("rain", p0_co_id=9)
        u = _unit(0, UnitType.TANK)
        self.assertEqual(effective_move_cost(state, u, TID_PLAIN), 2)


# ---------------------------------------------------------------------------
# 5. Two-segment weather expiry via _end_turn
# ---------------------------------------------------------------------------

class TestWeatherExpiry(unittest.TestCase):
    def _state_with_segments(self, n: int) -> GameState:
        state = _state_with_weather("snow")
        state.co_weather_segments_remaining = n
        return state

    def test_expires_after_two_end_turns(self):
        state = self._state_with_segments(2)
        # First end-turn: 2 → 1, still snow
        state._end_turn()
        self.assertEqual(state.weather, "snow")
        self.assertEqual(state.co_weather_segments_remaining, 1)
        # Second end-turn: 1 → 0, reverts to default (clear)
        state._end_turn()
        self.assertEqual(state.weather, "clear")
        self.assertEqual(state.co_weather_segments_remaining, 0)

    def test_zero_segments_does_not_decrement(self):
        state = self._state_with_segments(0)
        state.weather = "clear"
        state._end_turn()
        self.assertEqual(state.weather, "clear")
        self.assertEqual(state.co_weather_segments_remaining, 0)


# ---------------------------------------------------------------------------
# 6. Power hooks: weather set on activation
# ---------------------------------------------------------------------------

class TestPowerHooks(unittest.TestCase):
    """Use make_initial_state with a real map so _activate_power works end-to-end."""

    def setUp(self):
        from pathlib import Path
        from engine.map_loader import load_map
        pool = Path(__file__).parent / "data" / "gl_map_pool.json"
        maps = Path(__file__).parent / "data" / "maps"
        from engine.action import get_legal_actions
        self._pool = pool
        self._maps = maps

    def _load_map_123858(self):
        from engine.map_loader import load_map
        return load_map(123858, self._pool, self._maps)

    def _state_for_co(self, co_id: int) -> GameState:
        md = self._load_map_123858()
        return make_initial_state(md, co_id, 1)

    def _charge_to_scop(self, state: GameState) -> None:
        """Charge P0's power bar past SCOP threshold by direct assignment."""
        state.active_player = 0
        co = state.co_states[0]
        co.power_bar = co._scop_threshold + 1

    def _charge_to_cop(self, state: GameState) -> None:
        state.active_player = 0
        co = state.co_states[0]
        if co.cop_stars:
            co.power_bar = co._cop_threshold + 1

    def test_olaf_cop_sets_snow(self):
        state = self._state_for_co(9)  # Olaf
        self._charge_to_cop(state)
        state._activate_power(cop=True)
        self.assertEqual(state.weather, "snow")
        self.assertEqual(state.co_weather_segments_remaining, 2)

    def test_olaf_scop_sets_snow(self):
        state = self._state_for_co(9)
        self._charge_to_scop(state)
        state._activate_power(cop=False)
        self.assertEqual(state.weather, "snow")
        self.assertEqual(state.co_weather_segments_remaining, 2)

    def test_drake_scop_sets_rain(self):
        state = self._state_for_co(5)  # Drake
        self._charge_to_scop(state)
        state._activate_power(cop=False)
        self.assertEqual(state.weather, "rain")
        self.assertEqual(state.co_weather_segments_remaining, 2)

    def test_drake_cop_does_not_set_weather(self):
        state = self._state_for_co(5)
        self._charge_to_cop(state)
        state._activate_power(cop=True)
        # COP Tsunami: no weather change
        self.assertEqual(state.weather, "clear")
        self.assertEqual(state.co_weather_segments_remaining, 0)

    def test_second_power_overrides_first(self):
        # Start with snow active, then Drake SCOP fires rain
        md = self._load_map_123858()
        state = make_initial_state(md, 9, 5)  # P0=Olaf, P1=Drake
        state.weather = "snow"
        state.co_weather_segments_remaining = 2
        # Switch to P1 to activate Drake SCOP
        state.active_player = 1
        co1 = state.co_states[1]
        co1.power_bar = co1._scop_threshold + 1
        state._activate_power(cop=False)
        self.assertEqual(state.weather, "rain")
        self.assertEqual(state.co_weather_segments_remaining, 2)


# ---------------------------------------------------------------------------
# 7. encode_state includes weather scalars
# ---------------------------------------------------------------------------

class TestEncodeStateWeatherScalars(unittest.TestCase):
    def test_clear_weather_scalars_zero(self):
        from rl.encoder import encode_state, N_SCALARS
        state = _state_with_weather("clear")
        _, scalars = encode_state(state)
        self.assertEqual(len(scalars), N_SCALARS)
        self.assertAlmostEqual(float(scalars[13]), 0.0)  # weather_rain
        self.assertAlmostEqual(float(scalars[14]), 0.0)  # weather_snow
        self.assertAlmostEqual(float(scalars[15]), 0.0)  # weather_turns

    def test_rain_scalar(self):
        from rl.encoder import encode_state
        state = _state_with_weather("rain")
        state.co_weather_segments_remaining = 2
        _, scalars = encode_state(state)
        self.assertAlmostEqual(float(scalars[13]), 1.0)  # rain
        self.assertAlmostEqual(float(scalars[14]), 0.0)  # snow
        self.assertAlmostEqual(float(scalars[15]), 1.0)  # 2/2

    def test_snow_scalar(self):
        from rl.encoder import encode_state
        state = _state_with_weather("snow")
        state.co_weather_segments_remaining = 1
        _, scalars = encode_state(state)
        self.assertAlmostEqual(float(scalars[13]), 0.0)  # rain
        self.assertAlmostEqual(float(scalars[14]), 1.0)  # snow
        self.assertAlmostEqual(float(scalars[15]), 0.5)  # 1/2

    def test_n_scalars_matches_encoder(self):
        from rl.encoder import N_SCALARS
        self.assertEqual(N_SCALARS, 17)

    def test_income_share_scalar_bounded(self):
        from rl.encoder import encode_state, N_SCALARS
        state = _state_with_weather("clear")
        _, scalars = encode_state(state)
        self.assertEqual(len(scalars), N_SCALARS)
        self.assertGreaterEqual(float(scalars[16]), 0.0)
        self.assertLessEqual(float(scalars[16]), 1.0)


if __name__ == "__main__":
    unittest.main()
