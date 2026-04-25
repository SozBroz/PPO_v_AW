"""train._sync_worker_inherited_env_flags — curriculum vs inherited shell env."""
from __future__ import annotations


def test_clears_stale_learner_greedy_mix_when_cli_zero(monkeypatch) -> None:
    import os

    monkeypatch.setenv("AWBW_LEARNER_GREEDY_MIX", "0.3")
    from train import _sync_worker_inherited_env_flags, build_train_argument_parser

    p = build_train_argument_parser()
    args = p.parse_args(["--learner-greedy-mix", "0"])
    _sync_worker_inherited_env_flags(args)
    assert "AWBW_LEARNER_GREEDY_MIX" not in os.environ


def test_sets_learner_greedy_mix_when_positive(monkeypatch) -> None:
    import os

    monkeypatch.delenv("AWBW_LEARNER_GREEDY_MIX", raising=False)
    from train import _sync_worker_inherited_env_flags, build_train_argument_parser

    p = build_train_argument_parser()
    args = p.parse_args(["--learner-greedy-mix", "0.15"])
    _sync_worker_inherited_env_flags(args)
    assert os.environ.get("AWBW_LEARNER_GREEDY_MIX") == "0.15"


def test_clears_capture_gate_when_flag_off(monkeypatch) -> None:
    import os

    monkeypatch.setenv("AWBW_CAPTURE_MOVE_GATE", "1")
    from train import _sync_worker_inherited_env_flags, build_train_argument_parser

    p = build_train_argument_parser()
    args = p.parse_args([])
    _sync_worker_inherited_env_flags(args)
    assert "AWBW_CAPTURE_MOVE_GATE" not in os.environ


def test_sets_capture_gate_when_flag_on(monkeypatch) -> None:
    import os

    monkeypatch.delenv("AWBW_CAPTURE_MOVE_GATE", raising=False)
    from train import _sync_worker_inherited_env_flags, build_train_argument_parser

    p = build_train_argument_parser()
    args = p.parse_args(["--capture-move-gate"])
    _sync_worker_inherited_env_flags(args)
    assert os.environ.get("AWBW_CAPTURE_MOVE_GATE") == "1"
