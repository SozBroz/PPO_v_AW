"""SelfPlayTrainer: train_reconfig_request.json in-process reconfiguration."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from gymnasium import spaces

from rl.fleet_env import REPO_ROOT, FleetConfig
from rl.self_play import SelfPlayTrainer


def _make_trainer(tmp_path: Path) -> SelfPlayTrainer:
    d = tmp_path / "ckpt"
    d.mkdir(parents=True, exist_ok=True)
    return SelfPlayTrainer(
        total_timesteps=1,
        n_envs=1,
        n_steps=32,
        batch_size=16,
        checkpoint_dir=d,
        fleet_cfg=FleetConfig(
            role="auxiliary",
            machine_id="m1",
            shared_root=tmp_path,
            repo_root=REPO_ROOT,
        ),
    )


def _write_reconfig(
    tmp_path: Path, *, request_id: str = "1", args: dict | None = None
) -> Path:
    if args is None:
        args = {
            "--n-envs": 1,
            "--n-steps": 32,
            "--batch-size": 16,
        }
    p = tmp_path / "fleet" / "m1" / "train_reconfig_request.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "request_id": request_id,
                "args": args,
                "reason": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return p


def test_reconfig_request_validation_rejects_oversized_batch(
    tmp_path: Path,
) -> None:
    t = _make_trainer(tmp_path)
    t.n_envs, t.n_steps, t.batch_size = 1, 8, 4
    _write_reconfig(
        tmp_path,
        args={"--n-envs": 1, "--n-steps": 8, "--batch-size": 64},
    )
    m = MagicMock()
    s = spaces.Box(0, 1, (2,), np.float32)
    ve = MagicMock()
    ve.observation_space = s
    m2, v2 = t._maybe_apply_train_reconfig_request(m, ve, 0)
    assert m2 is m
    assert v2 is ve
    failed = list((tmp_path / "fleet" / "m1").glob("train_reconfig_request.failed.*.json"))
    assert failed


def test_reconfig_request_observation_space_mismatch_falls_back(
    tmp_path: Path,
) -> None:
    t = _make_trainer(tmp_path)
    t.n_envs, t.n_steps, t.batch_size = 1, 32, 8
    _write_reconfig(tmp_path)
    m = MagicMock()
    s1 = spaces.Box(0, 1, (3,), np.float32)
    s2 = spaces.Box(0, 1, (4,), np.float32)
    ve = MagicMock()
    ve.observation_space = s1
    nv = MagicMock()
    nv.observation_space = s2
    t._build_vec_env = MagicMock(return_value=nv)  # type: ignore[method-assign]
    m2, v2 = t._maybe_apply_train_reconfig_request(m, ve, 0)
    assert m2 is m
    assert v2 is ve
    nv.close.assert_called_once()
    assert list((tmp_path / "fleet" / "m1").glob("train_reconfig_request.failed.*.json"))


def test_reconfig_request_happy_path_swaps_env_with_mocks(
    tmp_path: Path,
) -> None:
    t = _make_trainer(tmp_path)
    t.n_envs, t.n_steps, t.batch_size = 1, 32, 8
    _write_reconfig(
        tmp_path, args={"--n-envs": 1, "--n-steps": 32, "--batch-size": 8}
    )
    space = spaces.Box(0, 1, (5,), np.float32)
    ve = MagicMock()
    ve.close = MagicMock()
    ve.observation_space = space
    new_ve = MagicMock()
    new_ve.observation_space = space
    t._build_vec_env = MagicMock(return_value=new_ve)  # type: ignore[method-assign]
    m = MagicMock()
    nm = MagicMock()
    nm.tensorboard_log = None
    with patch("rl.ckpt_compat.load_maskable_ppo_compat", return_value=nm) as pload:
        nmod, nvec = t._maybe_apply_train_reconfig_request(m, ve, 100)
    m.save.assert_called()
    pload.assert_called_once()
    assert nmod is nm
    assert nvec is new_ve
    ve.close.assert_called_once()
    ap = list((tmp_path / "fleet" / "m1").glob("train_reconfig_request.applied.*.json"))
    assert ap


def test_reconfig_request_writes_failed_ack_on_save_exception(
    tmp_path: Path,
) -> None:
    t = _make_trainer(tmp_path)
    t.n_envs, t.n_steps, t.batch_size = 1, 32, 8
    _write_reconfig(tmp_path)
    space = spaces.Box(0, 1, (1,), np.float32)
    ve = MagicMock()
    ve.observation_space = space
    new_ve = MagicMock()
    new_ve.observation_space = space
    t._build_vec_env = MagicMock(return_value=new_ve)  # type: ignore[method-assign]
    m = MagicMock()
    m.save.side_effect = OSError("disk full")
    m2, v2 = t._maybe_apply_train_reconfig_request(m, ve, 0)
    assert m2 is m
    assert v2 is ve
    new_ve.close.assert_called()
    assert list((tmp_path / "fleet" / "m1").glob("train_reconfig_request.failed.*.json"))


def test_reconfig_no_fleet_config_noops(tmp_path: Path) -> None:
    t = _make_trainer(tmp_path)
    t.fleet_cfg = None
    m, v = MagicMock(), MagicMock()
    a, b = t._maybe_apply_train_reconfig_request(m, v, 0)
    assert a is m and b is v
