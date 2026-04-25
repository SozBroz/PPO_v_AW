"""scripts/start_solo_training — bootstrap CLI and train argv wiring."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_sst():
    p = REPO / "scripts" / "start_solo_training.py"
    name = "start_solo_training_test"
    spec = importlib.util.spec_from_file_location(name, p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_build_train_argv_uses_proposed_args() -> None:
    m = _load_sst()
    proposed = {
        "machine_id": "pc-b",
        "args": {"--n-envs": 2, "--n-steps": 512, "--batch-size": 128},
    }
    argv = m._build_train_argv(
        proposed=proposed, machine_id="pc-b", train_extra=["--ent-coef", "0.02"]
    )
    assert argv[0] == sys.executable
    assert "train.py" in argv[1]
    i = argv.index("--n-envs")
    assert argv[i + 1] == "2"
    i = argv.index("--batch-size")
    assert argv[i + 1] == "128"
    assert "--ent-coef" in argv
    assert "--max-env-steps" in argv and argv[argv.index("--max-env-steps") + 1] == "10000"
    assert "--max-p1-microsteps" in argv and argv[argv.index("--max-p1-microsteps") + 1] == "4000"
    i_mid = argv.index("--machine-id")
    assert argv[i_mid + 1] == "pc-b"
    assert "--log-replay-frames" not in argv


def test_build_train_argv_log_replay_frames() -> None:
    m = _load_sst()
    proposed = {"machine_id": "pc-b", "args": {}}
    argv = m._build_train_argv(
        proposed=proposed,
        machine_id="pc-b",
        train_extra=[],
        log_replay_frames=True,
    )
    assert "--log-replay-frames" in argv


def test_initial_curriculum_merge_omits_stage_d_map_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = _load_sst()
    import logging

    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    mid = "pc-b"
    fleet = tmp_path / "fleet" / mid
    fleet.mkdir(parents=True)
    state = {
        "current_stage_name": "stage_d_gl_std_map_pool_t4",
        "entered_stage_at_ts": 1.0,
        "games_observed_in_stage": 10,
        "last_proposal_ts": 1.0,
        "last_seen_finished_games": 0,
    }
    (fleet / "curriculum_state.json").write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "game_log.jsonl").write_text("", encoding="utf-8")

    proposed = {
        "machine_id": mid,
        "args": {"--n-envs": 4, "--n-steps": 1024, "--batch-size": 1024},
    }
    merged = m._merge_curriculum_for_initial_launch(
        proposed,
        machine_id=mid,
        state_path=fleet / "curriculum_state.json",
        log=logging.getLogger("test_initial_curriculum_merge"),
        write_state=False,
    )

    assert merged["args"]["--map-id"] is None
    assert merged["args"]["--tier"] == "T4"
    assert merged["args"]["--co-p0"] == 14
    assert merged["args"]["--co-p1"] == 14
    argv = m._build_train_argv(proposed=merged, machine_id=mid, train_extra=[])
    assert "--map-id" not in argv
    assert "--curriculum-tag" in argv


def test_operator_train_args_override_merges_after_curriculum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = _load_sst()
    import logging

    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    mid = "wo-test"
    fleet = tmp_path / "fleet" / mid
    fleet.mkdir(parents=True)
    state = {
        "current_stage_name": "stage_d_gl_std_map_pool_t4",
        "entered_stage_at_ts": 1.0,
        "games_observed_in_stage": 0,
        "last_proposal_ts": 1.0,
        "last_seen_finished_games": 0,
    }
    (fleet / "curriculum_state.json").write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "game_log.jsonl").write_text("", encoding="utf-8")
    (fleet / "operator_train_args_override.json").write_text(
        json.dumps(
            {
                "args": {
                    "--n-envs": 99,
                    "--map-id": None,
                },
                "reasoning": "op test",
            }
        ),
        encoding="utf-8",
    )
    proposed = {
        "machine_id": mid,
        "args": {"--n-envs": 4, "--n-steps": 1024, "--batch-size": 1024},
    }
    merged = m._merge_curriculum_for_initial_launch(
        proposed,
        machine_id=mid,
        state_path=fleet / "curriculum_state.json",
        log=logging.getLogger("t"),
        write_state=False,
    )
    merged2 = m._merge_operator_train_args_override_into_proposed(
        merged, fleet_dir=fleet, log=logging.getLogger("t")
    )
    assert merged2["args"]["--n-envs"] == 99
    assert merged2["args"]["--map-id"] is None
    # Curriculum still set tier etc.
    assert merged2["args"]["--tier"] == "T4"


def test_launch_env_sets_machine_and_flags() -> None:
    m = _load_sst()
    e = m._launch_env(machine_id="pc-b")
    assert e["AWBW_MACHINE_ID"] == "pc-b"
    assert e["AWBW_REWARD_SHAPING"] == "phi"
    assert e["AWBW_TIME_COST"] == "0.00005"
    assert e["AWBW_TRUNCATION_PENALTY"] == "0.25"
    assert "AWBW_CAPTURE_MOVE_GATE" not in e
    assert e["AWBW_TRACK_PER_WORKER_TIMES"] == "1"
    assert "AWBW_LOG_REPLAY_FRAMES" not in e


def test_train_popen_environ_strips_cli_owned_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _load_sst()
    monkeypatch.setenv("AWBW_LEARNER_GREEDY_MIX", "0.9")
    monkeypatch.setenv("AWBW_CAPTURE_MOVE_GATE", "1")
    monkeypatch.setenv("AWBW_MACHINE_ID", "host-shell")
    merged = m._train_popen_environ(m._launch_env(machine_id="pc-b"))
    assert "AWBW_LEARNER_GREEDY_MIX" not in merged
    assert "AWBW_CAPTURE_MOVE_GATE" not in merged
    assert merged["AWBW_MACHINE_ID"] == "pc-b"


def test_launch_env_log_replay_frames() -> None:
    m = _load_sst()
    e = m._launch_env(machine_id="pc-b", log_replay_frames=True)
    assert e["AWBW_LOG_REPLAY_FRAMES"] == "1"


def test_launch_env_torch_compile() -> None:
    m = _load_sst()
    e = m._launch_env(machine_id="pc-b", torch_compile=True)
    assert e["AWBW_TORCH_COMPILE"] == "1"


def test_hybrid_gpu_cpu_opponent_env_cuda_split() -> None:
    import logging

    m = _load_sst()
    log = logging.getLogger("t")
    probe = {
        "cpu": {"physical_cores": 16},
        "gpu": {"available": True},
    }
    h = m._hybrid_gpu_cpu_opponent_env(
        n_envs=14,
        probe=probe,
        enabled=True,
        min_n_envs=8,
        cuda_opponent_workers=4,
        log=log,
    )
    assert h.get("AWBW_ALLOW_CUDA_OPPONENT") == "1"
    assert h.get("AWBW_GPU_OPPONENT_POOL") == "1"
    assert h.get("AWBW_GPU_OPPONENT_POOL_SIZE") == "4"
    assert h.get("AWBW_WORKER_OMP_THREADS") == "2"
    assert "AWBW_OPPONENT_CUDA_WORKERS" not in h


def test_hybrid_gpu_cpu_opponent_env_below_min_ne() -> None:
    import logging

    m = _load_sst()
    log = logging.getLogger("t")
    h = m._hybrid_gpu_cpu_opponent_env(
        n_envs=6,
        probe={"cpu": {"physical_cores": 16}, "gpu": {"available": True}},
        enabled=True,
        min_n_envs=8,
        cuda_opponent_workers=4,
        log=log,
    )
    assert h == {}


def test_hybrid_gpu_cpu_opponent_env_no_gpu_all_cpu_threads() -> None:
    import logging

    m = _load_sst()
    log = logging.getLogger("t")
    h = m._hybrid_gpu_cpu_opponent_env(
        n_envs=14,
        probe={"cpu": {"physical_cores": 16}, "gpu": {"available": False}},
        enabled=True,
        min_n_envs=8,
        cuda_opponent_workers=4,
        log=log,
    )
    assert "AWBW_ALLOW_CUDA_OPPONENT" not in h
    assert h.get("AWBW_WORKER_OMP_THREADS") == "2"


def test_dry_run_bootstrap_subprocess_zero(tmp_path: Path) -> None:
    pr = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "start_solo_training.py"),
            "--machine-id",
            "pc-b",
            "--dry-run-bootstrap",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert pr.returncode == 0
    out = pr.stdout + pr.stderr
    assert "dry-run" in out.lower()
    assert "train cmd:" in out or "train cmd" in out.lower()
    assert "orchestrator cmd:" in out or "orchestrator" in out.lower()


def test_machine_id_required() -> None:
    pr = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "start_solo_training.py")],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert pr.returncode != 0


def test_probe_failure_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = _load_sst()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "start_solo_training",
            "--machine-id",
            "pc-b",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )
    bad = MagicMock(returncode=1, stderr="probe failed", stdout="")
    with (
        patch.object(m.subprocess, "run", return_value=bad),
        patch.object(m, "_configure_logging"),
    ):
        assert m.main() == 1


def test_train_launch_cmd_json_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = _load_sst()
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    mid = "solo-t1"
    fleet = tmp_path / "fleet" / mid
    fleet.mkdir(parents=True, exist_ok=True)
    proposed = {
        "machine_id": mid,
        "args": {"--n-envs": 4, "--n-steps": 512, "--batch-size": 256},
    }
    argv = m._build_train_argv(proposed=proposed, machine_id=mid, train_extra=[])
    env_overlay = m._launch_env(machine_id=mid)
    doc = {"cmd": argv, "env": env_overlay, "cwd": str(tmp_path.resolve())}
    launch_path = fleet / "train_launch_cmd.json"
    m._atomic_write_json(launch_path, doc)
    raw = json.loads(launch_path.read_text(encoding="utf-8"))
    assert isinstance(raw["cmd"], list) and raw["cmd"][0] == sys.executable
    assert isinstance(raw["env"], dict)
    assert raw["env"]["AWBW_MACHINE_ID"] == mid
    assert Path(raw["cwd"]) == tmp_path.resolve()


def test_atomic_write_json_roundtrip(tmp_path: Path) -> None:
    m = _load_sst()
    p = tmp_path / "x" / "f.json"
    m._atomic_write_json(p, {"a": 1})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}


def test_argv_to_args_dict_accumulates_live_games_id() -> None:
    m = _load_sst()
    exe = str(Path(sys.executable).resolve())
    train_py = str(REPO / "train.py")
    argv = [
        exe,
        train_py,
        "--live-snapshot-dir",
        "C:\\snap",
        "--live-games-id",
        "10",
        "--live-games-id",
        "20",
    ]
    d = m._argv_to_args_dict(argv)
    assert d["--live-games-id"] == [10, 20]
    assert d["--live-snapshot-dir"] == "C:\\snap"


def test_build_train_argv_merges_train_extra_no_duplicate_n_envs() -> None:
    m = _load_sst()
    proposed = {
        "machine_id": "pc-b",
        "args": {"--n-envs": 4, "--n-steps": 1024, "--batch-size": 1024},
    }
    argv = m._build_train_argv(
        proposed=proposed,
        machine_id="pc-b",
        train_extra=["--n-envs", "14"],
    )
    assert argv.count("--n-envs") == 1
    i = argv.index("--n-envs")
    assert argv[i + 1] == "14"


def test_proposed_args_sync_preserves_map_id_null() -> None:
    m = _load_sst()
    proposed = {
        "machine_id": "pc-b",
        "args": {
            "--map-id": None,
            "--tier": "T3",
            "--n-envs": 8,
            "--n-steps": 1024,
            "--batch-size": 1024,
        },
    }
    argv = m._build_train_argv(
        proposed=proposed, machine_id="pc-b", train_extra=[]
    )
    assert "--map-id" not in argv
    synced = m._proposed_args_synced_from_train_argv(proposed, argv)
    assert synced["args"].get("--map-id") is None


def test_discover_live_games_subdirs(tmp_path: Path) -> None:
    m = _load_sst()
    base = tmp_path / "amarinner_my_games"
    (base / "111" / "live_replay.json").parent.mkdir(parents=True)
    (base / "111" / "live_replay.json").write_text("{}", encoding="utf-8")
    (base / "222" / "engine_snapshot.pkl").parent.mkdir(parents=True)
    (base / "222" / "engine_snapshot.pkl").write_bytes(b"x")
    (base / "notnum").mkdir()
    (base / "333").mkdir()
    assert m._discover_live_games_subdirs(base) == [111, 222]


def test_inject_live_games_appends_flags(tmp_path: Path) -> None:
    m = _load_sst()
    import logging

    log = logging.getLogger("test_inject")
    base = tmp_path / "g"
    (base / "9" / "live_replay.json").parent.mkdir(parents=True)
    (base / "9" / "live_replay.json").write_text("{}", encoding="utf-8")
    ex, gids = m._inject_live_games_train_extra(
        ["--n-envs", "24"], base, no_auto=False, log=log
    )
    assert gids == [9]
    assert "--live-snapshot-dir" in ex
    i = ex.index("--live-games-id")
    assert ex[i + 1] == "9"


def test_resolve_live_snapshot_pkl_nested(tmp_path: Path) -> None:
    from rl.self_play import _resolve_live_snapshot_pkl_path

    (tmp_path / "5" / "engine_snapshot.pkl").parent.mkdir(parents=True)
    (tmp_path / "5" / "engine_snapshot.pkl").write_bytes(b"x")
    p = _resolve_live_snapshot_pkl_path(tmp_path, 5)
    assert "engine_snapshot.pkl" in p


def test_resolve_live_snapshot_pkl_flat(tmp_path: Path) -> None:
    from rl.self_play import _resolve_live_snapshot_pkl_path

    (tmp_path / "7.pkl").write_bytes(b"y")
    p = _resolve_live_snapshot_pkl_path(tmp_path, 7)
    assert p.endswith("7.pkl")


def test_infer_live_learner_seats_iwinagain_pov(tmp_path: Path) -> None:
    m = _load_sst()
    import logging

    log = logging.getLogger("test_seats")
    base = tmp_path / "export"
    # games_id=1: co_p0=8 Sami, co_p1=10 Eagle; iwinagain is Eagle (seat 1)
    d1 = base / "1"
    d1.mkdir(parents=True)
    (d1 / "meta.json").write_text(
        json.dumps(
            {
                "games_id": 1,
                "map_id": 1,
                "co_p0_id": 8,
                "co_p1_id": 10,
                "tier": "T2",
            }
        ),
        encoding="utf-8",
    )
    (d1 / "live_replay.json").write_text(
        json.dumps(
            {
                "first_snap": {
                    "players": {
                        "100": {"id": 100, "order": 6, "co_id": 8},
                        "200": {"id": 200, "order": 19, "co_id": 10},
                    }
                },
                "game_state_turn0": {
                    "players": {
                        "100": {
                            "users_username": "other",
                            "players_id": 100,
                        },
                        "200": {
                            "users_username": "iwinagain",
                            "players_id": 200,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    seats = m._infer_live_learner_seats(base, [1], "iwinagain", log)
    assert seats == [1]


def test_bump_n_envs_in_proposed_for_live() -> None:
    m = _load_sst()
    import logging

    log = logging.getLogger("test_bump_n")
    p: dict = {"args": {"--n-envs": 2}}
    m._bump_n_envs_in_proposed_for_live(p, 3, log)
    assert p["args"]["--n-envs"] == 3
    p2: dict = {"args": {"--n-envs": 28}}
    m._bump_n_envs_in_proposed_for_live(p2, 3, log)
    assert p2["args"]["--n-envs"] == 28


def test_tune_n_envs_throughput_inplace_updates_proposed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = _load_sst()
    import logging

    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    fleet_mid = tmp_path / "fleet" / "pc-x"
    fleet_mid.mkdir(parents=True)

    log = logging.getLogger("test_throughput_tune_inplace")
    proposed: dict = {
        "machine_id": "pc-x",
        "args": {"--n-envs": 2, "--n-steps": 512, "--batch-size": 256},
        "reasoning": "seed",
    }
    with patch(
        "tools.throughput_tune.choose_n_envs_throughput",
        lambda **k: (7, {"mock": True, "winner_median": 42.5}),
    ):
        m._tune_n_envs_throughput_inplace(
            proposed,
            machine_id="pc-x",
            train_extra=[],
            live_gids=[],
            max_envs=8,
            per_candidate_s=1.0,
            min_iters=4096,
            max_host_ram_pct=90.0,
            max_host_cpu_pct=90.0,
            host_wait_s=0.0,
            log_replay_frames=False,
            log=log,
        )
    assert proposed["args"]["--n-envs"] == 7
    assert proposed["args"]["--batch-size"] == 256
    assert "throughput_tune: n_envs=7" in str(proposed.get("reasoning", ""))
    ovr = fleet_mid / "operator_train_args_override.json"
    assert ovr.is_file()
    odoc = json.loads(ovr.read_text(encoding="utf-8"))
    assert odoc["args"]["--n-envs"] == 7
    assert odoc["args"]["--batch-size"] == 256
    assert odoc["source"] == "throughput_tune"
    assert odoc["throughput_tune"]["winner_n_envs"] == 7
    assert odoc["throughput_tune"]["winner_median"] == 42.5


def test_tune_writes_clamped_batch_to_operator_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PPO requires batch_size <= n_steps * n_envs; override must match."""
    m = _load_sst()
    import logging

    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    fleet_mid = tmp_path / "fleet" / "pc-z"
    fleet_mid.mkdir(parents=True)
    log = logging.getLogger("test_tune_batch_clamp")
    proposed: dict = {
        "machine_id": "pc-z",
        "args": {"--n-envs": 4, "--n-steps": 512, "--batch-size": 9999},
    }
    with patch(
        "tools.throughput_tune.choose_n_envs_throughput",
        lambda **k: (2, {"winner_median": 1.0}),
    ):
        m._tune_n_envs_throughput_inplace(
            proposed,
            machine_id="pc-z",
            train_extra=[],
            live_gids=[],
            max_envs=8,
            per_candidate_s=1.0,
            min_iters=4096,
            max_host_ram_pct=90.0,
            max_host_cpu_pct=90.0,
            host_wait_s=0.0,
            log_replay_frames=False,
            log=log,
        )
    assert proposed["args"]["--batch-size"] == 1024
    odoc = json.loads(
        (fleet_mid / "operator_train_args_override.json").read_text(encoding="utf-8")
    )
    assert odoc["args"]["--batch-size"] == 1024
    assert odoc["args"]["--n-envs"] == 2


def test_ensure_train_argv_n_envs_for_live_appends() -> None:
    m = _load_sst()
    import logging

    log = logging.getLogger("test_ensure_n")
    av: list = [sys.executable, "train.py", "--n-envs", "2", "--x", "1"]
    m._ensure_train_argv_n_envs_for_live(av, 4, log)
    assert m._last_int_for_flag(av, "--n-envs") == 4

    av2 = [sys.executable, "train.py", "--n-envs", "30"]
    m._ensure_train_argv_n_envs_for_live(av2, 3, log)
    assert m._last_int_for_flag(av2, "--n-envs") == 30
    assert av2 == [sys.executable, "train.py", "--n-envs", "30"]


def test_cli_includes_no_orchestrator_auto_apply() -> None:
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "start_solo_training.py"), "--help"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert "no-orchestrator-auto-apply" in (r.stdout or "")
