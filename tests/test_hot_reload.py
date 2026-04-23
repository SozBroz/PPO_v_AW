"""Phase 10d: hot weight reload from fleet reload_request.json."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from rl.fleet_env import REPO_ROOT, FleetConfig
from rl.self_play import SelfPlayTrainer


def _make_trainer(
    tmp_path: Path,
    *,
    hot_reload_enabled: bool = True,
    hot_reload_min_steps_done: int = 0,
) -> SelfPlayTrainer:
    return SelfPlayTrainer(
        total_timesteps=1,
        n_envs=1,
        checkpoint_dir=tmp_path / "ckpt",
        fleet_cfg=FleetConfig(
            role="auxiliary",
            machine_id="m1",
            shared_root=tmp_path,
            repo_root=REPO_ROOT,
        ),
        hot_reload_enabled=hot_reload_enabled,
        hot_reload_min_steps_done=hot_reload_min_steps_done,
    )


def test_hot_reload_skips_when_no_request_file(tmp_path: Path) -> None:
    t = _make_trainer(tmp_path)
    model = MagicMock()
    t._maybe_apply_reload_request(model, 1000)
    model.set_parameters.assert_not_called()


def test_hot_reload_applies_when_request_present(tmp_path: Path) -> None:
    t = _make_trainer(tmp_path)
    z = tmp_path / "fake.zip"
    z.write_bytes(b"zip")
    fleet_dir = tmp_path / "fleet" / "m1"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    req_path = fleet_dir / "reload_request.json"
    req_path.write_text(
        json.dumps(
            {
                "target_zip": str(z),
                "issued_at": 12345,
                "reason": "test",
            }
        ),
        encoding="utf-8",
    )
    model = MagicMock()
    t._maybe_apply_reload_request(model, 1000)
    model.set_parameters.assert_called_once()
    call_kw = model.set_parameters.call_args
    assert call_kw[0][0] == str(z)
    assert call_kw[1].get("exact_match") is True
    assert not req_path.is_file()
    assert any(
        p.name.startswith("reload_request.applied.") and p.suffix == ".json"
        for p in fleet_dir.iterdir()
    )


def test_hot_reload_idempotent_in_same_process(tmp_path: Path) -> None:
    t = _make_trainer(tmp_path)
    z = tmp_path / "fake.zip"
    z.write_bytes(b"zip")
    fleet_dir = tmp_path / "fleet" / "m1"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    req_path = fleet_dir / "reload_request.json"
    payload = {
        "target_zip": str(z),
        "issued_at": 99,
        "reason": "test",
    }
    req_path.write_text(json.dumps(payload), encoding="utf-8")
    model = MagicMock()
    t._maybe_apply_reload_request(model, 1000)
    assert model.set_parameters.call_count == 1
    # Restore same request; trainer remembers applied key
    req_path.write_text(json.dumps(payload), encoding="utf-8")
    t._maybe_apply_reload_request(model, 2000)
    assert model.set_parameters.call_count == 1


def test_hot_reload_skips_when_steps_below_min(tmp_path: Path) -> None:
    t = _make_trainer(tmp_path)
    z = tmp_path / "fake.zip"
    z.write_bytes(b"zip")
    fleet_dir = tmp_path / "fleet" / "m1"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    req_path = fleet_dir / "reload_request.json"
    req_path.write_text(
        json.dumps(
            {
                "target_zip": str(z),
                "min_steps_done": 5000,
                "issued_at": 1,
            }
        ),
        encoding="utf-8",
    )
    model = MagicMock()
    t._maybe_apply_reload_request(model, 1000)
    model.set_parameters.assert_not_called()
    assert req_path.is_file()


def test_hot_reload_skips_when_target_zip_missing(tmp_path: Path) -> None:
    t = _make_trainer(tmp_path)
    fleet_dir = tmp_path / "fleet" / "m1"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    req_path = fleet_dir / "reload_request.json"
    req_path.write_text(
        json.dumps(
            {
                "target_zip": str(tmp_path / "nope.zip"),
                "issued_at": 2,
            }
        ),
        encoding="utf-8",
    )
    model = MagicMock()
    t._maybe_apply_reload_request(model, 10_000)
    model.set_parameters.assert_not_called()
    assert req_path.is_file()


def test_hot_reload_trainer_min_steps_greater_than_request(
    tmp_path: Path,
) -> None:
    t = _make_trainer(tmp_path, hot_reload_min_steps_done=5000)
    z = tmp_path / "fake.zip"
    z.write_bytes(b"zip")
    fleet_dir = tmp_path / "fleet" / "m1"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    req_path = fleet_dir / "reload_request.json"
    req_path.write_text(
        json.dumps({"target_zip": str(z), "min_steps_done": 0, "issued_at": 3}),
        encoding="utf-8",
    )
    model = MagicMock()
    t._maybe_apply_reload_request(model, 1000)
    model.set_parameters.assert_not_called()


def test_hot_reload_disabled_does_nothing(tmp_path: Path) -> None:
    t = _make_trainer(tmp_path, hot_reload_enabled=False)
    z = tmp_path / "fake.zip"
    z.write_bytes(b"zip")
    fleet_dir = tmp_path / "fleet" / "m1"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    (fleet_dir / "reload_request.json").write_text(
        json.dumps({"target_zip": str(z), "issued_at": 4}),
        encoding="utf-8",
    )
    stub_model = MagicMock()
    stub_model.env = None
    with patch.object(t, "_maybe_apply_reload_request") as m_apply:
        t._maybe_handle_rollout_boundary(stub_model, None, 1000)
    m_apply.assert_not_called()
