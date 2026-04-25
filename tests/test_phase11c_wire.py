"""Phase 11c: MCTS CLI wiring (train.py, SelfPlayTrainer, symmetric_checkpoint_eval)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_symmetric_eval_module():
    p = ROOT / "scripts" / "symmetric_checkpoint_eval.py"
    spec = importlib.util.spec_from_file_location("symmetric_checkpoint_eval", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dummy_sb3_model():
    import numpy as np
    import torch
    from sb3_contrib.common.maskable.distributions import MaskableCategorical

    class _DummyPol:
        def obs_to_tensor(self, obs):
            if isinstance(obs, dict):
                return {
                    k: torch.as_tensor(v, dtype=torch.float32).unsqueeze(0)
                    for k, v in obs.items()
                }, None
            raise TypeError(obs)

        def predict_values(self, obs_t):
            if isinstance(obs_t, dict):
                b = next(iter(obs_t.values())).shape[0]
            else:
                b = obs_t.shape[0]
            return torch.zeros((b, 1), dtype=torch.float32)

        def get_distribution(self, obs_t, action_masks=None):
            if action_masks is None:
                raise ValueError("need masks")
            n = int(action_masks.shape[-1])
            logits = torch.zeros((1, n), dtype=torch.float32)
            return MaskableCategorical(logits=logits, masks=action_masks)

    class _DummyModel:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.policy = _DummyPol()

        def predict(self, obs, action_masks=None, deterministic=False):
            mask = action_masks
            legal = np.where(mask)[0]
            idx = int(legal[0]) if len(legal) > 0 else 0
            return np.array([idx], dtype=np.int64), None

    return _DummyModel()


def test_train_parser_mcts_flags() -> None:
    from train import build_train_argument_parser

    p = build_train_argument_parser()
    ns = p.parse_args(
        [
            "--mcts-mode",
            "eval_only",
            "--mcts-sims",
            "32",
            "--mcts-c-puct",
            "2.0",
            "--mcts-dirichlet-alpha",
            "0.2",
            "--mcts-dirichlet-epsilon",
            "0.1",
            "--mcts-temperature",
            "0.5",
            "--mcts-min-depth",
            "3",
            "--mcts-root-plans",
            "6",
            "--mcts-max-plan-actions",
            "128",
        ]
    )
    assert ns.mcts_mode == "eval_only"
    assert ns.mcts_sims == 32
    assert ns.mcts_c_puct == 2.0
    assert ns.mcts_dirichlet_alpha == 0.2
    assert ns.mcts_dirichlet_epsilon == 0.1
    assert ns.mcts_temperature == 0.5
    assert ns.mcts_min_depth == 3
    assert ns.mcts_root_plans == 6
    assert ns.mcts_max_plan_actions == 128


def test_train_parser_log_replay_frames_and_machine_id() -> None:
    from train import build_train_argument_parser

    p = build_train_argument_parser()
    ns = p.parse_args(["--log-replay-frames", "--machine-id", "pc-b"])
    assert ns.log_replay_frames is True
    assert ns.machine_id == "pc-b"


def test_self_play_trainer_stores_mcts_config(tmp_path: Path) -> None:
    from rl.fleet_env import REPO_ROOT, FleetConfig
    from rl.self_play import SelfPlayTrainer

    t = SelfPlayTrainer(
        total_timesteps=1,
        n_envs=1,
        checkpoint_dir=tmp_path / "ckpt",
        fleet_cfg=FleetConfig(
            role="auxiliary",
            machine_id="phase11c",
            shared_root=tmp_path,
            repo_root=REPO_ROOT,
        ),
        mcts_mode="eval_only",
        mcts_sims=9,
        mcts_c_puct=1.25,
        mcts_dirichlet_alpha=0.15,
        mcts_dirichlet_epsilon=0.05,
        mcts_temperature=0.75,
        mcts_min_depth=2,
        mcts_root_plans=5,
        mcts_max_plan_actions=200,
    )
    assert t.mcts_mode == "eval_only"
    assert t.mcts_sims == 9
    assert t.mcts_c_puct == 1.25
    assert t.mcts_dirichlet_alpha == 0.15
    assert t.mcts_dirichlet_epsilon == 0.05
    assert t.mcts_temperature == 0.75
    assert t.mcts_min_depth == 2
    assert t.mcts_root_plans == 5
    assert t.mcts_max_plan_actions == 200


def test_symmetric_eval_mcts_payload_off_matches_default(tmp_path: Path) -> None:
    mod = _load_symmetric_eval_module()
    p = mod.build_symmetric_checkpoint_eval_parser()
    z = tmp_path / "stub.zip"
    z.write_bytes(b"z")
    zp = str(z)
    base = [
        "--candidate",
        zp,
        "--baseline",
        zp,
        "--map-id",
        "123858",
        "--tier",
        "T3",
        "--co-p0",
        "1",
        "--co-p1",
        "1",
    ]
    a_explicit = p.parse_args(base + ["--mcts-mode", "off"])
    a_default = p.parse_args(base)
    assert json.dumps(mod._mcts_fields_from_args(a_explicit), sort_keys=True) == json.dumps(
        mod._mcts_fields_from_args(a_default), sort_keys=True
    )


def test_worker_game_mcts_eval_only_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_symmetric_eval_module()
    (tmp_path / "a.zip").write_bytes(b"a")
    (tmp_path / "b.zip").write_bytes(b"b")

    with (ROOT / "data" / "gl_map_pool.json").open(encoding="utf-8") as f:
        pool = [m for m in json.load(f) if m.get("map_id") == 123858]

    monkeypatch.setattr(
        "rl.ckpt_compat.load_maskable_ppo_compat",
        lambda *a, **k: _dummy_sb3_model(),
    )

    payload = {
        "game_i": 7,
        "seed": 42,
        "challenger": str(tmp_path / "a.zip"),
        "defender": str(tmp_path / "b.zip"),
        "map_pool_json": json.dumps(pool),
        "tier": "T2",
        "co_p0": 10,
        "co_p1": 23,
        "deterministic": True,
        "max_env_steps": 120,
        "max_p1_microsteps": None,
        "max_turns": None,
        "mcts_mode": "eval_only",
        "mcts_sims": 4,
        "mcts_c_puct": 1.5,
        "mcts_dirichlet_alpha": 0.3,
        "mcts_dirichlet_epsilon": 0.0,
        "mcts_temperature": 0.0,
        "mcts_min_depth": 1,
        "mcts_root_plans": 2,
        "mcts_max_plan_actions": 256,
    }
    gi, w, trunc, tel = mod._worker_game(payload)
    assert gi == 7
    assert isinstance(w, int)
    assert isinstance(trunc, bool)
    assert tel["mcts_total_decisions"] >= 1
    assert len(tel["mcts_decision_wall_s"]) == tel["mcts_total_decisions"]
    assert tel["mcts_failures"] == 0


def test_worker_game_mcts_off_no_telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_symmetric_eval_module()
    (tmp_path / "a.zip").write_bytes(b"a")
    (tmp_path / "b.zip").write_bytes(b"b")
    with (ROOT / "data" / "gl_map_pool.json").open(encoding="utf-8") as f:
        pool = [m for m in json.load(f) if m.get("map_id") == 123858]
    monkeypatch.setattr(
        "rl.ckpt_compat.load_maskable_ppo_compat",
        lambda *a, **k: _dummy_sb3_model(),
    )
    payload = {
        "game_i": 0,
        "seed": 0,
        "challenger": str(tmp_path / "a.zip"),
        "defender": str(tmp_path / "b.zip"),
        "map_pool_json": json.dumps(pool),
        "tier": "T2",
        "co_p0": 10,
        "co_p1": 23,
        "deterministic": True,
        "max_env_steps": 40,
        "max_p1_microsteps": None,
        "max_turns": None,
        "mcts_mode": "off",
        "mcts_sims": 16,
        "mcts_c_puct": 1.5,
        "mcts_dirichlet_alpha": 0.3,
        "mcts_dirichlet_epsilon": 0.25,
        "mcts_temperature": 1.0,
        "mcts_min_depth": 4,
        "mcts_root_plans": 8,
        "mcts_max_plan_actions": 256,
    }
    _gi, _w, _t, tel = mod._worker_game(payload)
    assert tel["mcts_total_decisions"] == 0
    assert tel["mcts_decision_wall_s"] == []
    assert tel["mcts_pv_depths"] == []
    assert tel["mcts_failures"] == 0


def test_worker_game_mcts_failure_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import rl.mcts as mcts_mod

    mod = _load_symmetric_eval_module()
    (tmp_path / "a.zip").write_bytes(b"a")
    (tmp_path / "b.zip").write_bytes(b"b")
    with (ROOT / "data" / "gl_map_pool.json").open(encoding="utf-8") as f:
        pool = [m for m in json.load(f) if m.get("map_id") == 123858]
    monkeypatch.setattr(
        "rl.ckpt_compat.load_maskable_ppo_compat",
        lambda *a, **k: _dummy_sb3_model(),
    )
    real = mcts_mod.run_mcts
    calls = {"n": 0}

    def _flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("forced mcts failure for test")
        return real(*a, **k)

    monkeypatch.setattr(mcts_mod, "run_mcts", _flaky)
    payload = {
        "game_i": 0,
        "seed": 0,
        "challenger": str(tmp_path / "a.zip"),
        "defender": str(tmp_path / "b.zip"),
        "map_pool_json": json.dumps(pool),
        "tier": "T2",
        "co_p0": 10,
        "co_p1": 23,
        "deterministic": True,
        "max_env_steps": 80,
        "max_p1_microsteps": None,
        "max_turns": None,
        "mcts_mode": "eval_only",
        "mcts_sims": 2,
        "mcts_c_puct": 1.5,
        "mcts_dirichlet_alpha": 0.3,
        "mcts_dirichlet_epsilon": 0.0,
        "mcts_temperature": 0.0,
        "mcts_min_depth": 0,
        "mcts_root_plans": 2,
        "mcts_max_plan_actions": 256,
    }
    _gi, _w, _trunc, tel = mod._worker_game(payload)
    assert tel["mcts_failures"] >= 1
    assert calls["n"] >= 2


def test_eval_summary_json_mcts_fields_contract() -> None:
    """Mirrors scripts/symmetric_checkpoint_eval.py json-out branch."""
    import statistics

    mod = _load_symmetric_eval_module()
    p = mod.build_symmetric_checkpoint_eval_parser()
    z = ROOT / "data" / "gl_map_pool.json"
    args_off = p.parse_args(
        [
            "--candidate",
            str(z),
            "--baseline",
            str(z),
            "--map-id",
            "123858",
            "--tier",
            "T3",
            "--co-p0",
            "1",
            "--co-p1",
            "1",
        ]
    )
    args_on = p.parse_args(
        [
            "--candidate",
            str(z),
            "--baseline",
            str(z),
            "--map-id",
            "123858",
            "--tier",
            "T3",
            "--co-p0",
            "1",
            "--co-p1",
            "1",
            "--mcts-mode",
            "eval_only",
        ]
    )
    agg_walls = [0.1, 0.3, 0.2]
    agg_pv = [2, 4, 3]
    summary_off: dict = {"stub": True}
    summary_on: dict = {"stub": True}
    if args_on.mcts_mode == "eval_only":
        summary_on["mcts_per_decision_wall_s_p50"] = float(statistics.median(agg_walls))
        summary_on["mcts_total_decisions"] = 3
        summary_on["mcts_total_wall_s"] = sum(agg_walls)
        summary_on["mcts_avg_principal_variation_depth"] = float(sum(agg_pv) / len(agg_pv))
        summary_on["mcts_failure_count"] = 0
    for k in (
        "mcts_per_decision_wall_s_p50",
        "mcts_total_decisions",
        "mcts_total_wall_s",
        "mcts_avg_principal_variation_depth",
        "mcts_failure_count",
    ):
        assert k not in summary_off
        assert k in summary_on
