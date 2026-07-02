"""RHEA parallel learner: multi-map pool filtering and opening-book knobs."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_train_parallel():
    path = ROOT / "scripts" / "train_rhea_value_parallel.py"
    spec = importlib.util.spec_from_file_location("train_rhea_value_parallel", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["train_rhea_value_parallel"] = mod
    spec.loader.exec_module(mod)
    return mod


def _synthetic_pool() -> list[dict]:
    def _row(map_id: int, *, map_type: str = "std", name: str = "m") -> dict:
        return {
            "map_id": map_id,
            "name": name,
            "type": map_type,
            "tiers": [{"tier_name": "T4", "enabled": True, "co_ids": [14, 8]}],
        }

    return [
        _row(171596, name="designed_desires"),
        _row(123858, name="amarriner"),
        _row(133665, name="legacy", map_type="legacy"),
    ]


def test_parse_map_id_cli_single_and_list() -> None:
    mod = _load_train_parallel()
    assert mod._parse_map_id_cli("171596") == [171596]
    assert mod._parse_map_id_cli("171596,123858") == [171596, 123858]
    assert mod._parse_map_id_cli("171596, 123858 ,171596") == [171596, 123858]


def test_parse_map_id_cli_rejects_invalid() -> None:
    mod = _load_train_parallel()
    with pytest.raises(Exception):
        mod._parse_map_id_cli("not-a-map")


def test_build_arg_parser_map_id_defaults() -> None:
    mod = _load_train_parallel()
    args = mod.build_arg_parser().parse_args(["--checkpoint", "x.pt"])
    assert args.map_id == [171596]
    assert args.all_std_maps is False
    assert args.opening_book_prob_p1 is None


def test_build_arg_parser_all_std_maps_and_map_list() -> None:
    mod = _load_train_parallel()
    args = mod.build_arg_parser().parse_args(
        ["--checkpoint", "x.pt", "--map-id", "171596,123858", "--all-std-maps"]
    )
    assert args.map_id == [171596, 123858]
    assert args.all_std_maps is True


def test_build_arg_parser_opening_book_prob_p1() -> None:
    mod = _load_train_parallel()
    args = mod.build_arg_parser().parse_args(
        [
            "--checkpoint",
            "x.pt",
            "--opening-book-prob",
            "1.0",
            "--opening-book-prob-p1",
            "0.0",
        ]
    )
    assert args.opening_book_prob == 1.0
    assert args.opening_book_prob_p1 == 0.0


def test_filter_map_pool_single_and_multi() -> None:
    mod = _load_train_parallel()
    pool = _synthetic_pool()
    one = mod._filter_map_pool(pool, map_ids=[171596])
    assert [m["map_id"] for m in one] == [171596]
    two = mod._filter_map_pool(pool, map_ids=[123858, 171596])
    assert [m["map_id"] for m in two] == [123858, 171596]


def test_filter_map_pool_all_std_maps() -> None:
    mod = _load_train_parallel()
    pool = _synthetic_pool()
    std = mod._filter_map_pool(pool, all_std_maps=True)
    assert {m["map_id"] for m in std} == {171596, 123858}


def test_filter_map_pool_unknown_id_raises() -> None:
    mod = _load_train_parallel()
    with pytest.raises(ValueError, match="999999"):
        mod._filter_map_pool(_synthetic_pool(), map_ids=[999999])


def test_sample_training_matchup_uniform_over_filtered_pool() -> None:
    from rl.env import sample_training_matchup

    pool = _synthetic_pool()
    filtered = [m for m in pool if m["map_id"] in (171596, 123858)]
    seen: set[int] = set()
    for seed in range(200):
        mid, *_rest = sample_training_matchup(
            filtered, co_p0=14, co_p1=14, rng=__import__("random").Random(seed)
        )
        seen.add(int(mid))
    assert seen == {171596, 123858}


def test_opening_book_fallback_when_map_has_no_entries(tmp_path: Path) -> None:
    from rl.opening_book import TwoSidedOpeningBookManager

    book_path = tmp_path / "book.jsonl"
    book_path.write_text(
        json.dumps(
            {
                "book_id": "dd-p0",
                "map_id": 171596,
                "seat": 0,
                "action_indices": [1, 2, 3],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    mgr = TwoSidedOpeningBookManager(book_path, prob=1.0, seed=0)
    mgr.on_episode_start(episode_id=1, map_id=123858, co_ids=[14, 14])
    assert mgr.controllers[0].candidate_count == 0
    assert mgr.controllers[0].book_id is None
    assert mgr.controllers[0].episode_enabled is True
    import numpy as np

    mask = np.ones(64, dtype=bool)
    assert mgr.suggest_flat(seat=0, calendar_turn=1, action_mask=mask) is None


def test_opening_book_per_seat_prob_p1_zero() -> None:
    from rl.opening_book import TwoSidedOpeningBookManager

    book_path = ROOT / "data" / "designed_desires_opening_book.jsonl"
    if not book_path.is_file():
        pytest.skip("opening book fixture missing")
    mgr = TwoSidedOpeningBookManager(
        book_path, prob=1.0, prob_p1=0.0, seed=0, strict_co=False
    )
    mgr.on_episode_start(episode_id=1, map_id=171596, co_ids=[14, 14])
    assert mgr.controllers[0].episode_enabled is True
    assert mgr.controllers[0].book_id is not None
    assert mgr.controllers[1].episode_enabled is False
    assert mgr.controllers[1].book_id is None
