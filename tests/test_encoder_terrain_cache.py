"""Phase 3b: lazy terrain one-hot cache on MapData instances."""
from __future__ import annotations

from engine.action import ActionStage
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData
from rl.encoder import GRID_SIZE, N_TERRAIN_CHANNELS, encode_state


def _st(mid: int) -> GameState:
    t = [[1] * 8 for _ in range(8)]
    md = MapData(
        mid, "enc", "std", t, 8, 8, 99, 99, [], [], None, [], {0: [], 1: []},
        {0: [], 1: []}, {}, [],
    )
    return GameState(
        map_data=md,
        units={0: [], 1: []},
        funds=[0, 0],
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
        tier_name="T1",
    )


def test_terrain_cache_attached_after_first_encode() -> None:
    st = _st(999_911)
    assert getattr(st.map_data, "_encoded_terrain_channels", None) is None
    encode_state(st)
    ch = getattr(st.map_data, "_encoded_terrain_channels", None)
    assert ch is not None and ch.shape == (GRID_SIZE, GRID_SIZE, N_TERRAIN_CHANNELS)


def test_terrain_cache_reused_second_call() -> None:
    st = _st(999_912)
    encode_state(st)
    a = st.map_data._encoded_terrain_channels
    encode_state(st)
    assert st.map_data._encoded_terrain_channels is a


def test_terrain_cache_distinct_per_mapdata() -> None:
    s0, s1 = _st(999_913), _st(999_914)
    encode_state(s0)
    encode_state(s1)
    assert s0.map_data._encoded_terrain_channels is not s1.map_data._encoded_terrain_channels
