"""
End-to-end smoke: real ``train.py`` subprocess with Phase 11c MCTS CLI args.

`train.py` only accepts ``--mcts-mode off | eval_only`` (Phase 11c). The operator
brief mentions ``alphazero``; that value is not on the public parser today — the
MCTS “on” case is ``eval_only`` (storage + symmetric-eval path; PPO still uses
the policy for rollouts). We therefore smoke-test ``eval_off`` and ``off`` with
low ``--mcts-sims`` so a future ``alphazero`` (or in-rollout MCTS) can extend the
same pattern once wired.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from rl.train_launch_env import environ_for_train_subprocess

REPO_ROOT = Path(__file__).resolve().parents[1]


def _base_train_cmd(tmp_ckpt: Path) -> list[str | Path]:
    return [
        sys.executable,
        str(REPO_ROOT / "train.py"),
        "--iters",
        "200",
        "--n-envs",
        "2",
        "--n-steps",
        "32",
        "--batch-size",
        "16",
        "--device",
        "cpu",
        "--save-every",
        "10000",
        "--checkpoint-dir",
        str(tmp_ckpt),
        "--map-id",
        "123858",
        "--tier",
        "T3",
        "--co-p0",
        "1",
        "--co-p1",
        "1",
        "--cold-opponent",
        "random",
        "--mcts-sims",
        "2",
        "--mcts-c-puct",
        "1.5",
        "--mcts-dirichlet-alpha",
        "0.3",
        "--mcts-dirichlet-epsilon",
        "0.0",
        "--mcts-temperature",
        "0.0",
        "--mcts-min-depth",
        "0",
        "--mcts-root-plans",
        "2",
        "--mcts-max-plan-actions",
        "64",
    ]


def _mcts_env() -> dict[str, str]:
    return {
        **environ_for_train_subprocess(),
        "AWBW_MACHINE_ID": f"e2e-mcts-smoke-{os.getpid()}",
        "AWBW_TRACK_PER_WORKER_TIMES": "0",
    }


def _assert_clean_train_exit(
    proc: subprocess.CompletedProcess[str], *, label: str
) -> None:
    last_err = (proc.stderr or "")[-4096:]
    last_out = (proc.stdout or "")[-2048:]
    assert proc.returncode == 0, (
        f"train.py ({label}) exited with {proc.returncode}\n"
        f"--- stderr (last 4KB) ---\n{last_err}\n"
        f"--- stdout (last 2KB) ---\n{last_out}"
    )
    assert "Traceback" not in (proc.stderr or ""), f"unexpected traceback:\n{last_err}"


@pytest.mark.slow
def test_train_py_with_mcts_eval_only_runs_to_clean_exit(tmp_path: Path) -> None:
    """MCTS *config* on (``eval_only``); short PPO run must exit 0, no tracebacks."""
    save_dir = tmp_path / "ckpt"
    save_dir.mkdir()
    cmd = _base_train_cmd(save_dir) + [
        "--mcts-mode",
        "eval_only",
    ]
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(REPO_ROOT),
        env=_mcts_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    _assert_clean_train_exit(proc, label="eval_only MCTS")


@pytest.mark.slow
def test_train_py_with_mcts_off_runs_to_clean_exit(tmp_path: Path) -> None:
    """Default ``mcts off`` with same hyperparameters; must not crash."""
    save_dir = tmp_path / "ckpt"
    save_dir.mkdir()
    cmd = _base_train_cmd(save_dir) + [
        "--mcts-mode",
        "off",
    ]
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(REPO_ROOT),
        env=_mcts_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    _assert_clean_train_exit(proc, label="mcts off")
