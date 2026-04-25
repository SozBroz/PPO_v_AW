"""tools/propose_train_args — pc-b cap and heuristics."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_propose_mod():
    p = REPO / "tools" / "propose_train_args.py"
    name = "propose_train_args_test"
    spec = importlib.util.spec_from_file_location(name, p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _probe(
    machine_id: str, *, phys: int, ram_gb: float, probed_at: str = "2026-04-22T00:00:00Z"
) -> dict:
    return {
        "machine_id": machine_id,
        "probed_at": probed_at,
        "cpu": {"physical_cores": phys, "logical_processors": phys * 2, "model_name": "t"},
        "ram": {"total_gb": ram_gb, "free_gb_at_probe": 1.0},
        "gpu": {"available": True, "device_name": "x", "vram_total_gb": 8.0},
        "disk": {"checkpoint_root_writable": True, "checkpoint_root_path": "/x"},
        "platform": "linux",
    }


def test_pc_b_caps_n_envs_at_four_regardless_of_probe() -> None:
    m = _load_propose_mod()
    big = _probe("pc-b", phys=32, ram_gb=64.0)
    doc = m.propose_from_probe(big)
    assert doc["args"]["--n-envs"] == 4
    # RAM headroom + n_envs<=8 → extended rollout; batch follows PPO heuristic.
    assert doc["args"]["--n-steps"] == m._N_STEPS_EXTENDED
    assert doc["args"]["--batch-size"] == 1024
    assert m.PC_B_REASON in doc["reasoning"]
    assert "n_steps=" in doc["reasoning"]
    assert doc["auto_apply"] is False


def test_pc_b_low_ram_keeps_legacy_batch_and_512_steps() -> None:
    m = _load_propose_mod()
    doc = m.propose_from_probe(_probe("pc-b", phys=8, ram_gb=16.0))
    assert doc["args"]["--n-envs"] == 4
    assert doc["args"]["--n-steps"] == 512
    assert doc["args"]["--batch-size"] == 256
    assert doc["reasoning"] == m.PC_B_REASON


def test_pc_b_moderate_ram_uses_1024_steps_and_heuristic_batch() -> None:
    m = _load_propose_mod()
    doc = m.propose_from_probe(_probe("pc-b", phys=8, ram_gb=32.0))
    assert doc["args"]["--n-envs"] == 4
    assert doc["args"]["--n-steps"] == 1024
    assert doc["args"]["--batch-size"] == 1024
    assert "1024" in doc["reasoning"]
    assert m.PC_B_REASON in doc["reasoning"]


def test_non_pc_heuristic_big_machine() -> None:
    m = _load_propose_mod()
    doc = m.propose_from_probe(_probe("keras-aux", phys=32, ram_gb=64.0))
    assert doc["args"]["--n-envs"] == 12
    assert doc["args"]["--n-steps"] == 512
    assert doc["args"]["--batch-size"] == 1024
    assert "heuristic" in doc["reasoning"]
    # n_envs>8: skip long rollout (buffer would be too large).
    assert "n_steps=" not in doc["reasoning"]


def test_non_pc_heuristic_small_machine() -> None:
    m = _load_propose_mod()
    doc = m.propose_from_probe(_probe("edge-box", phys=8, ram_gb=16.0))
    assert doc["args"]["--n-envs"] == 4
    assert doc["args"]["--batch-size"] == 512
    assert doc["args"]["--n-steps"] == 512


def test_non_pc_moderate_n_envs_and_ram_bumps_n_steps_to_1024() -> None:
    m = _load_propose_mod()
    doc = m.propose_from_probe(_probe("aux-32g", phys=8, ram_gb=32.0))
    assert doc["args"]["--n-envs"] == 6
    assert doc["args"]["--n-steps"] == 1024
    assert doc["args"]["--batch-size"] == 1024
    assert "n_steps=1024" in doc["reasoning"]


def test_non_pc_extended_rollout_at_very_high_ram() -> None:
    m = _load_propose_mod()
    # max(1, phys-2)=8, RAM allows 16 envs by GiB rule but cap 12 → n_envs=8 keeps long rollout.
    doc = m.propose_from_probe(_probe("workstation", phys=10, ram_gb=64.0))
    assert doc["args"]["--n-envs"] == 8
    assert doc["args"]["--n-steps"] == m._N_STEPS_EXTENDED
    assert doc["args"]["--batch-size"] == 1024
    assert "n_steps=2048" in doc["reasoning"]


def test_max_safe_n_envs_phys_minus_two_and_caps() -> None:
    m = _load_propose_mod()
    assert m.max_safe_n_envs_from_probe(_probe("x", phys=16, ram_gb=64.0)) == 12
    assert m.max_safe_n_envs_from_probe(_probe("x", phys=16, ram_gb=64.0), absolute_cap=16) == 14
    assert m.max_safe_n_envs_from_probe(_probe("x", phys=4, ram_gb=64.0)) == 2
    assert m.max_safe_n_envs_from_probe(_probe("x", phys=2, ram_gb=64.0)) == 1
