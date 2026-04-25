"""Unit tests for ``rl.live_games_resync`` argv parsing."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rl.live_games_resync import parse_train_cmd_live_ppo


def test_parse_train_cmd_live_ppo_empty() -> None:
    assert parse_train_cmd_live_ppo([]) == ([], None)
    assert parse_train_cmd_live_ppo([sys.executable, "train.py"]) == ([], None)


def test_parse_train_cmd_live_ppo_ids_and_dir() -> None:
    cmd = [
        sys.executable,
        str(Path("train.py")),
        "--n-envs",
        "4",
        "--live-games-id",
        "1638496",
        "--live-snapshot-dir",
        "replays/amarinner_my_games",
        "--live-games-id",
        "1638514",
    ]
    ids, snap = parse_train_cmd_live_ppo(cmd)
    assert ids == [1638496, 1638514]
    assert snap == Path("replays/amarinner_my_games")


@pytest.mark.parametrize("bad", ("", "x"))
def test_parse_skips_bad_game_id(bad: str) -> None:
    cmd = [sys.executable, "train.py", "--live-games-id", bad]
    assert parse_train_cmd_live_ppo(cmd)[0] == []
