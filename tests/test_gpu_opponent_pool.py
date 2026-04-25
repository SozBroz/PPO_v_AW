"""GPU opponent semaphore pool (rl.self_play)."""

from __future__ import annotations

import pytest

from rl import self_play as sp


def test_gpu_opponent_pool_permits_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWBW_GPU_OPPONENT_POOL_SIZE", raising=False)
    assert sp.gpu_opponent_pool_permits() == sp.OPPONENT_CUDA_WORKERS_MAX


def test_gpu_opponent_pool_permits_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWBW_GPU_OPPONENT_POOL_SIZE", "99")
    assert sp.gpu_opponent_pool_permits() == 32
    monkeypatch.setenv("AWBW_GPU_OPPONENT_POOL_SIZE", "0")
    assert sp.gpu_opponent_pool_permits() == 1


def test_gpu_opponent_pool_enabled_requires_both_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWBW_ALLOW_CUDA_OPPONENT", raising=False)
    monkeypatch.delenv("AWBW_GPU_OPPONENT_POOL", raising=False)
    assert sp.gpu_opponent_pool_enabled() is False
    monkeypatch.setenv("AWBW_GPU_OPPONENT_POOL", "1")
    assert sp.gpu_opponent_pool_enabled() is False
    monkeypatch.setenv("AWBW_ALLOW_CUDA_OPPONENT", "1")
    assert sp.gpu_opponent_pool_enabled() is True


def test_opponent_inference_device_pool_forces_cpu_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWBW_ALLOW_CUDA_OPPONENT", "1")
    monkeypatch.setenv("AWBW_GPU_OPPONENT_POOL", "1")
    assert sp._opponent_inference_device_for_worker(99) == "cpu"


def test_opponent_inference_device_legacy_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AWBW_GPU_OPPONENT_POOL", raising=False)
    monkeypatch.setenv("AWBW_ALLOW_CUDA_OPPONENT", "1")
    monkeypatch.setenv("AWBW_OPPONENT_CUDA_WORKERS", "2")
    try:
        import torch
    except ImportError:
        pytest.skip("torch required")
    if not torch.cuda.is_available():
        assert sp._opponent_inference_device_for_worker(0) == "cpu"
        assert sp._opponent_inference_device_for_worker(1) == "cpu"
    else:
        assert sp._opponent_inference_device_for_worker(0) == "cuda"
        assert sp._opponent_inference_device_for_worker(3) == "cpu"
