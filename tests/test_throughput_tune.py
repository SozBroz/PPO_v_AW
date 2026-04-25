"""Unit tests for tools/throughput_tune.choose_n_envs_throughput."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.throughput_tune import choose_n_envs_throughput


def _null_log() -> logging.Logger:
    log = logging.getLogger("test_throughput_tune")
    log.handlers.clear()
    log.addHandler(logging.NullHandler())
    return log


def test_choose_n_envs_uses_median_per_candidate(tmp_path: Path) -> None:
    root = tmp_path
    (root / "logs").mkdir(parents=True)
    (root / "fleet" / "m1").mkdir(parents=True)
    (root / "train.py").write_text("# stub\n", encoding="utf-8")
    diag = root / "logs" / "fps_diag.jsonl"

    proposed = {
        "machine_id": "m1",
        "args": {"--n-envs": 2, "--n-steps": 512, "--batch-size": 256},
    }

    def fake_popen(cmd: list[str], **kwargs: object) -> MagicMock:
        n = int(cmd[cmd.index("--n-envs") + 1])
        if n == 2:
            vals = [10.0, 30.0, 20.0]
        elif n == 3:
            vals = [25.0, 35.0, 24.0, 36.0]
        else:
            vals = [1.0]

        class Proc:
            pid = 4242

            def wait(self, timeout: float | None = None) -> int:
                with diag.open("a", encoding="utf-8") as f:
                    for v in vals:
                        f.write(json.dumps({"env_steps_per_s_total": v}) + "\n")
                return 0

            def kill(self) -> None:
                pass

        return Proc()  # type: ignore[return-value]

    with (
        patch("tools.throughput_tune.subprocess.Popen", side_effect=fake_popen),
        patch("tools.throughput_tune._wait_host_headroom", return_value=True),
        patch("tools.throughput_tune._lower_probe_priority"),
        patch("tools.throughput_tune._atomic_write_json"),
    ):
        winner, report = choose_n_envs_throughput(
            machine_id="m1",
            proposed=proposed,
            gids=[],
            max_envs=3,
            per_candidate_s=30.0,
            min_iters=1000,
            max_host_ram_pct=90.0,
            max_host_cpu_pct=90.0,
            host_wait_s=0.0,
            repo_root=root,
            fleet_dir=root / "fleet" / "m1",
            log=_null_log(),
            make_probe_argv=lambda n: [
                "--machine-id",
                "m1",
                "--n-envs",
                str(int(n)),
                "--n-steps",
                "512",
                "--fps-diag",
            ],
        )

    assert winner == 3
    assert report["winner_n_envs"] == 3
    by_n = {c["n_envs"]: c["median"] for c in report["candidates"]}
    assert by_n[2] == 20.0
    assert by_n[3] == 30.0


def test_choose_n_envs_tie_break_lower_n(tmp_path: Path) -> None:
    root = tmp_path
    (root / "logs").mkdir(parents=True)
    (root / "fleet" / "m1").mkdir(parents=True)
    (root / "train.py").write_text("# stub\n", encoding="utf-8")
    diag = root / "logs" / "fps_diag.jsonl"

    proposed = {
        "machine_id": "m1",
        "args": {"--n-envs": 2, "--n-steps": 512, "--batch-size": 256},
    }

    def fake_popen(cmd: list[str], **kwargs: object) -> MagicMock:
        n = int(cmd[cmd.index("--n-envs") + 1])
        rate = 100.0 if n in (3, 4) else float(n * 10)

        class Proc:
            pid = 4242

            def wait(self, timeout: float | None = None) -> int:
                with diag.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"env_steps_per_s_total": rate}) + "\n")
                return 0

            def kill(self) -> None:
                pass

        return Proc()  # type: ignore[return-value]

    with (
        patch("tools.throughput_tune.subprocess.Popen", side_effect=fake_popen),
        patch("tools.throughput_tune._wait_host_headroom", return_value=True),
        patch("tools.throughput_tune._lower_probe_priority"),
        patch("tools.throughput_tune._atomic_write_json"),
    ):
        winner, _report = choose_n_envs_throughput(
            machine_id="m1",
            proposed=proposed,
            gids=[],
            max_envs=4,
            per_candidate_s=30.0,
            min_iters=500,
            max_host_ram_pct=90.0,
            max_host_cpu_pct=90.0,
            host_wait_s=0.0,
            repo_root=root,
            fleet_dir=root / "fleet" / "m1",
            log=_null_log(),
            make_probe_argv=lambda n: [
                "--machine-id",
                "m1",
                "--n-envs",
                str(int(n)),
                "--n-steps",
                "512",
                "--fps-diag",
            ],
        )

    assert winner == 3


def test_host_headroom_abort_returns_baseline(tmp_path: Path) -> None:
    root = tmp_path
    (root / "logs").mkdir(parents=True)
    (root / "fleet" / "m1").mkdir(parents=True)
    (root / "train.py").write_text("# stub\n", encoding="utf-8")

    proposed = {
        "machine_id": "m1",
        "args": {"--n-envs": 5, "--n-steps": 512, "--batch-size": 256},
    }

    with (
        patch("tools.throughput_tune.subprocess.Popen") as popen,
        patch("tools.throughput_tune._wait_host_headroom", return_value=False),
        patch("tools.throughput_tune._atomic_write_json"),
    ):
        winner, report = choose_n_envs_throughput(
            machine_id="m1",
            proposed=proposed,
            gids=[],
            max_envs=8,
            per_candidate_s=30.0,
            min_iters=100,
            max_host_ram_pct=90.0,
            max_host_cpu_pct=90.0,
            host_wait_s=0.0,
            repo_root=root,
            fleet_dir=root / "fleet" / "m1",
            log=_null_log(),
            make_probe_argv=lambda n: ["--n-envs", str(int(n))],
        )

    popen.assert_not_called()
    assert winner == 5
    assert report.get("abort_reason") == "host_headroom_timeout_pre_sweep"


def test_min_iters_zero_uses_32768_floor(tmp_path: Path) -> None:
    root = tmp_path
    (root / "logs").mkdir(parents=True)
    (root / "fleet" / "m1").mkdir(parents=True)
    (root / "train.py").write_text("# stub\n", encoding="utf-8")
    diag = root / "logs" / "fps_diag.jsonl"

    proposed = {
        "machine_id": "m1",
        "args": {"--n-envs": 2, "--n-steps": 512, "--batch-size": 256},
    }
    seen: list[int] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> MagicMock:
        seen.append(int(cmd[cmd.index("--iters") + 1]))
        with diag.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"env_steps_per_s_total": 50.0}) + "\n")

        class Proc:
            pid = 4242

            def wait(self, timeout: float | None = None) -> int:
                return 0

            def kill(self) -> None:
                pass

        return Proc()  # type: ignore[return-value]

    with (
        patch("tools.throughput_tune.subprocess.Popen", side_effect=fake_popen),
        patch("tools.throughput_tune._wait_host_headroom", return_value=True),
        patch("tools.throughput_tune._lower_probe_priority"),
        patch("tools.throughput_tune._atomic_write_json"),
    ):
        choose_n_envs_throughput(
            machine_id="m1",
            proposed=proposed,
            gids=[],
            max_envs=2,
            per_candidate_s=30.0,
            min_iters=0,
            max_host_ram_pct=90.0,
            max_host_cpu_pct=90.0,
            host_wait_s=0.0,
            repo_root=root,
            fleet_dir=root / "fleet" / "m1",
            log=_null_log(),
            make_probe_argv=lambda n: [
                "--machine-id",
                "m1",
                "--n-envs",
                str(int(n)),
                "--n-steps",
                "512",
            ],
        )

    assert seen == [max(32768, 2 * 512 * 2)]

