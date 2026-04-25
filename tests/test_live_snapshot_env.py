import os
from pathlib import Path

import pytest

from engine.game import make_initial_state
from engine.map_loader import load_map
from rl.env import LEARNER_SEAT_ENV
from rl.live_snapshot import write_live_snapshot


def test_reset_loads_pickle_and_marks_episode_info(tmp_path: Path) -> None:
    pytest.importorskip("gymnasium")
    from rl.env import AWBWEnv

    root = Path(__file__).resolve().parents[1]
    pool = root / "data" / "gl_map_pool.json"
    maps = root / "data" / "maps"
    map_data = load_map(133665, pool, maps)
    st = make_initial_state(
        map_data, 1, 7, starting_funds=0, tier_name="T1", replay_first_mover=0
    )
    ls = int(st.active_player)
    p = tmp_path / "9999999.pkl"
    write_live_snapshot(p, st, games_id=9999999, learner_seat=ls)
    old = os.environ.get(LEARNER_SEAT_ENV)
    try:
        os.environ[LEARNER_SEAT_ENV] = str(ls)
        env = AWBWEnv(
            live_snapshot_path=p,
            live_games_id=9999999,
            live_fallback_curriculum=False,
            opponent_policy=None,
        )
        _obs, info = env.reset()
    finally:
        if old is None:
            os.environ.pop(LEARNER_SEAT_ENV, None)
        else:
            os.environ[LEARNER_SEAT_ENV] = old
    assert info.get("live") is True
    assert info.get("games_id") == 9999999
    assert int(env.state.active_player) == ls  # type: ignore[union-attr]
