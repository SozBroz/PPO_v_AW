"""RHEA game corpus bridge: write, read, replay determinism."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.action import Action, ActionStage, ActionType
from engine.game import GameState, make_initial_state
from engine.map_loader import MapData
from engine.predeployed import PredeployedUnitSpec
from engine.unit import UnitType
from rl.game_corpus import (
    build_corpus_row,
    corpus_enabled,
    replay_corpus_row,
    state_fingerprint,
)

WOOD = 3
TINY_MAP_ID = 9_000_042


@pytest.fixture
def tiny_map_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    def _loader(map_id: int, pool: Path, maps_dir: Path) -> MapData:
        if int(map_id) == TINY_MAP_ID:
            return _tiny_combat_map()
        raise ValueError(f"unexpected map_id {map_id}")

    monkeypatch.setattr("rl.game_corpus.load_map", _loader)
    monkeypatch.setattr("engine.map_loader.load_map", _loader)


def _tiny_combat_map() -> MapData:
    return MapData(
        map_id=TINY_MAP_ID,
        name="corpus_combat_tiny",
        map_type="std",
        terrain=[[WOOD, WOOD], [WOOD, WOOD]],
        height=2,
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
        predeployed_specs=[
            PredeployedUnitSpec(row=0, col=0, player=0, unit_type=UnitType.TANK),
            PredeployedUnitSpec(row=0, col=1, player=1, unit_type=UnitType.INFANTRY),
        ],
    )


def _play_scripted_combat_game(*, luck_seed: int = 42_001) -> GameState:
    """Short deterministic game with one attack (luck-sensitive)."""
    md = _tiny_combat_map()
    st = make_initial_state(
        md,
        14,
        14,
        starting_funds=0,
        tier_name="T2",
        replay_first_mover=0,
        luck_seed=luck_seed,
    )
    tank = st.get_unit_at(0, 0)
    inf = st.get_unit_at(0, 1)
    assert tank is not None and inf is not None
    scripted = [
        Action(ActionType.SELECT_UNIT, unit_pos=tank.pos),
        Action(ActionType.SELECT_UNIT, unit_pos=tank.pos, move_pos=tank.pos),
        Action(
            ActionType.ATTACK,
            unit_pos=tank.pos,
            move_pos=tank.pos,
            target_pos=inf.pos,
        ),
    ]
    for act in scripted:
        st.step(act)
    assert st.full_trace
    return st


def test_corpus_row_replay_matches_terminal_state(
    tmp_path: Path, tiny_map_loader: None
) -> None:
    original = _play_scripted_combat_game()
    row = build_corpus_row(
        original,
        map_id=TINY_MAP_ID,
        map_name="corpus_combat_tiny",
        tier="T2",
        p0_co_id=14,
        p1_co_id=14,
        luck_seed=42_001,
        replay_first_mover=0,
    )
    assert row["corpus_schema_version"] == "1.0"
    assert row["full_trace"]
    assert row["action_indices"]
    assert len(row["action_indices"]) == len(row["full_trace"])
    assert row["luck_seed"] == 42_001

    replayed = replay_corpus_row(row)
    assert state_fingerprint(replayed) == state_fingerprint(original)


def test_env_finalize_writes_corpus_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tiny_map_loader: None,
) -> None:
    from rl.env import AWBWEnv

    corpus_file = tmp_path / "rhea_games.jsonl"
    monkeypatch.setenv("AWBW_GAME_CORPUS_PATH", str(corpus_file))
    monkeypatch.setenv("AWBW_GAME_CORPUS", "1")
    monkeypatch.setenv("AWBW_MACHINE_ID", "test-corpus")

    original = _play_scripted_combat_game()
    env = AWBWEnv(map_pool=[{
        "map_id": TINY_MAP_ID,
        "name": "corpus_combat_tiny",
        "tier": "T2",
        "type": "std",
        "tiers": [{"tier_name": "T2", "enabled": True, "co_ids": [14]}],
    }])
    monkeypatch.setattr(env, "_load_map", lambda _mid: _tiny_combat_map())
    env.reset(seed=42_001, options={"map_id": TINY_MAP_ID})
    assert env.state is not None
    # Replace terminal state with scripted game (same map/CO setup).
    env.state = original
    env._episode_info.update({
        "map_id": TINY_MAP_ID,
        "map_name": "corpus_combat_tiny",
        "tier": "T2",
        "p0_co": 14,
        "p1_co": 14,
        "luck_seed": 42_001,
    })
    env.finalize_rhea_episode()

    assert corpus_file.is_file()
    row = json.loads(corpus_file.read_text(encoding="utf-8").strip())
    assert row["machine_id"] == "test-corpus"
    replayed = replay_corpus_row(row)
    assert state_fingerprint(replayed) == state_fingerprint(original)


def test_corpus_write_failure_does_not_raise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tiny_map_loader: None,
) -> None:
    from rl.env import AWBWEnv

    original = _play_scripted_combat_game()
    env = AWBWEnv(map_pool=[{
        "map_id": TINY_MAP_ID,
        "name": "corpus_combat_tiny",
        "tier": "T2",
        "type": "std",
        "tiers": [{"tier_name": "T2", "enabled": True, "co_ids": [14]}],
    }])
    env.state = original
    env._episode_info = {
        "map_id": TINY_MAP_ID,
        "map_name": "corpus_combat_tiny",
        "tier": "T2",
        "p0_co": 14,
        "p1_co": 14,
        "luck_seed": 42_001,
    }
    env._opening_book_log = {}
    env._opening_player = 0
    env._log_episode_truncated = False
    env._log_episode_truncation_reason = None

    monkeypatch.setattr(
        "rl.game_corpus.append_corpus_row",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )
    env._try_write_rhea_corpus()

    env._try_write_rhea_corpus()


def test_corpus_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWBW_GAME_CORPUS", "0")
    assert not corpus_enabled()


def test_luck_rng_state_replay_without_seed(tiny_map_loader: None) -> None:
    md = _tiny_combat_map()
    st = make_initial_state(md, 14, 14, starting_funds=0, tier_name="T2", replay_first_mover=0)
    from rl.game_corpus import _serialize_rng_state

    initial_rng_state = _serialize_rng_state(st.luck_rng)
    tank = st.get_unit_at(0, 0)
    inf = st.get_unit_at(0, 1)
    assert tank is not None and inf is not None
    st.step(Action(ActionType.SELECT_UNIT, unit_pos=tank.pos))
    st.step(Action(ActionType.SELECT_UNIT, unit_pos=tank.pos, move_pos=tank.pos))
    st.step(Action(
        ActionType.ATTACK,
        unit_pos=tank.pos,
        move_pos=tank.pos,
        target_pos=inf.pos,
    ))

    row = build_corpus_row(
        st,
        map_id=TINY_MAP_ID,
        map_name="corpus_combat_tiny",
        tier="T2",
        p0_co_id=14,
        p1_co_id=14,
        luck_seed=None,
        luck_rng_state=initial_rng_state,
        replay_first_mover=0,
    )
    assert "luck_rng_state" in row
    assert "luck_seed" not in row
    replayed = replay_corpus_row(row)
    assert state_fingerprint(replayed) == state_fingerprint(st)
