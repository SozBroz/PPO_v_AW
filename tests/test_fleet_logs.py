"""Tests for rl.fleet_logs path helper (formalize-fleet-logs-layout)."""
from __future__ import annotations

import pytest

from rl.fleet_logs import (
    fleet_log_path,
    infer_machine_id_from_path,
    iter_fleet_log_paths,
)
from rl.paths import LOGS_DIR, REPO_ROOT


def test_fleet_log_path_local_machine():
    expected = REPO_ROOT / "logs" / "game_log.jsonl"
    assert fleet_log_path(None, "game_log.jsonl") == expected
    assert fleet_log_path("", "game_log.jsonl") == expected


def test_fleet_log_path_main_mirror():
    expected = REPO_ROOT / "logs" / "logs" / "game_log.jsonl"
    assert fleet_log_path("main", "game_log.jsonl") == expected
    assert fleet_log_path("MAIN", "game_log.jsonl") == expected
    assert fleet_log_path("Main", "game_log.jsonl") == expected


def test_fleet_log_path_aux_machine():
    expected = REPO_ROOT / "logs" / "keras-aux" / "game_log.jsonl"
    assert fleet_log_path("keras-aux", "game_log.jsonl") == expected

    expected2 = REPO_ROOT / "logs" / "fake_aux_1" / "slow_games.jsonl"
    assert fleet_log_path("fake_aux_1", "slow_games.jsonl") == expected2


@pytest.mark.parametrize("bad_id", ["../etc", "a/b", "a;b", ".", "..", "a b", "a\\b"])
def test_fleet_log_path_rejects_traversal(bad_id):
    with pytest.raises(ValueError):
        fleet_log_path(bad_id, "game_log.jsonl")


def test_iter_fleet_log_paths_local_only():
    paths = iter_fleet_log_paths("game_log.jsonl")

    assert "_local" in paths
    assert paths["_local"] == LOGS_DIR / "game_log.jsonl"

    main_mirror = LOGS_DIR / "logs" / "game_log.jsonl"
    if main_mirror.is_file():
        assert paths.get("main") == main_mirror

    for key, p in paths.items():
        if key in {"_local", "main"}:
            continue
        assert p.is_file(), f"discovered aux entry {key} should point at an existing file"
        assert p.parent.parent == LOGS_DIR


def test_infer_machine_id_from_path_round_trip():
    cases = [
        (None, None),
        ("", None),
        ("main", "main"),
        ("MAIN", "main"),
        ("keras-aux", "keras-aux"),
        ("fake_aux_1", "fake_aux_1"),
    ]
    for in_id, expected_out in cases:
        p = fleet_log_path(in_id, "game_log.jsonl")
        assert infer_machine_id_from_path(p) == expected_out, (
            f"round-trip failed for {in_id!r} -> {p} -> {infer_machine_id_from_path(p)} "
            f"(expected {expected_out!r})"
        )


def test_infer_machine_id_from_path_unrelated_returns_none():
    assert infer_machine_id_from_path(REPO_ROOT / "README.md") is None
