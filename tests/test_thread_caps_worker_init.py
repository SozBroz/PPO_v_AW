"""Phase 1c: SubprocVecEnv worker _init thread caps in rl.self_play._make_env_factory."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch

from rl.self_play import _make_env_factory

ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "data" / "gl_map_pool.json"
MAP_ID = 123858


def _single_map_pool() -> list[dict]:
    with open(POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    return [next(m for m in pool if m.get("map_id") == MAP_ID)]


def test_init_sets_thread_caps_when_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.delenv(key, raising=False)

    _init = _make_env_factory(
        map_pool=_single_map_pool(),
        checkpoint_dir=str(tmp_path),
        co_p0=1,
        co_p1=1,
        tier_name="T3",
    )
    try:
        _init()
    except BaseException:
        # Env or sb3 may be unavailable; caps are set at the start of _init.
        pass

    assert os.environ.get("OMP_NUM_THREADS") == "1"
    assert os.environ.get("MKL_NUM_THREADS") == "1"
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
    assert os.environ.get("NUMEXPR_NUM_THREADS") == "1"
    assert torch.get_num_threads() == 1


def test_init_respects_pre_existing_env_var_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "4", prepend=False)
    _init = _make_env_factory(
        map_pool=_single_map_pool(),
        checkpoint_dir=str(tmp_path),
        co_p0=1,
        co_p1=1,
        tier_name="T3",
    )
    try:
        _init()
    except BaseException:
        pass
    assert os.environ.get("OMP_NUM_THREADS") == "4"


def test_init_torch_set_num_threads_does_not_crash_when_torch_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("simulated")

    monkeypatch.setattr(torch, "set_num_threads", boom)
    _init = _make_env_factory(
        map_pool=_single_map_pool(),
        checkpoint_dir=str(tmp_path),
        co_p0=1,
        co_p1=1,
        tier_name="T3",
    )
    try:
        _init()
    except RuntimeError as e:
        if "simulated" in str(e):
            pytest.fail("set_num_threads RuntimeError should be caught inside _init")
        raise
    except BaseException:
        pass
