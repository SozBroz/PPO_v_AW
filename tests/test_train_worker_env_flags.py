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


def test_clears_stale_egocentric_episode_prob_when_cli_zero(monkeypatch) -> None:
    import os

    monkeypatch.setenv("AWBW_EGOCENTRIC_EPISODE_PROB", "0.4")
    from train import _sync_worker_inherited_env_flags, build_train_argument_parser

    p = build_train_argument_parser()
    args = p.parse_args(["--egocentric-episode-prob", "0"])
    _sync_worker_inherited_env_flags(args)
    assert "AWBW_EGOCENTRIC_EPISODE_PROB" not in os.environ


def test_sets_egocentric_episode_prob_when_positive(monkeypatch) -> None:
    import os

    monkeypatch.delenv("AWBW_EGOCENTRIC_EPISODE_PROB", raising=False)
    from train import _sync_worker_inherited_env_flags, build_train_argument_parser

    p = build_train_argument_parser()
    args = p.parse_args(["--egocentric-episode-prob", "0.25"])
    _sync_worker_inherited_env_flags(args)
    assert os.environ.get("AWBW_EGOCENTRIC_EPISODE_PROB") == "0.25"


def test_sets_capture_gate_when_flag_on(monkeypatch) -> None:
    import os

    monkeypatch.delenv("AWBW_CAPTURE_MOVE_GATE", raising=False)
    from train import _sync_worker_inherited_env_flags, build_train_argument_parser

    p = build_train_argument_parser()
    args = p.parse_args(["--capture-move-gate"])
    _sync_worker_inherited_env_flags(args)
    assert os.environ.get("AWBW_CAPTURE_MOVE_GATE") == "1.0"


def test_sets_capture_gate_fraction(monkeypatch) -> None:
    import os

    monkeypatch.delenv("AWBW_CAPTURE_MOVE_GATE", raising=False)
    from train import _sync_worker_inherited_env_flags, build_train_argument_parser

    p = build_train_argument_parser()
    args = p.parse_args(["--capture-move-gate", "0.75"])
    _sync_worker_inherited_env_flags(args)
    assert os.environ.get("AWBW_CAPTURE_MOVE_GATE") == "0.75"


def test_clears_deprecated_phi_enemy_property_capture_penalty(monkeypatch) -> None:
    import os

    monkeypatch.setenv("AWBW_PHI_ENEMY_PROPERTY_CAPTURE_PENALTY", "1.0")
    from train import _sync_worker_inherited_env_flags, build_train_argument_parser

    p = build_train_argument_parser()
    args = p.parse_args([])
    _sync_worker_inherited_env_flags(args)
    assert "AWBW_PHI_ENEMY_PROPERTY_CAPTURE_PENALTY" not in os.environ


def test_sets_pairwise_zero_sum_reward_when_flag_on(monkeypatch) -> None:
    import os

    monkeypatch.delenv("AWBW_PAIRWISE_ZERO_SUM_REWARD", raising=False)
    from train import _sync_worker_inherited_env_flags, build_train_argument_parser

    p = build_train_argument_parser()
    args = p.parse_args(["--pairwise-zero-sum-reward"])
    _sync_worker_inherited_env_flags(args)
    assert os.environ.get("AWBW_PAIRWISE_ZERO_SUM_REWARD") == "1"


def test_clears_pairwise_zero_sum_reward_when_flag_off(monkeypatch) -> None:
    import os

    monkeypatch.setenv("AWBW_PAIRWISE_ZERO_SUM_REWARD", "1")
    from train import _sync_worker_inherited_env_flags, build_train_argument_parser

    p = build_train_argument_parser()
    args = p.parse_args([])
    _sync_worker_inherited_env_flags(args)
    assert "AWBW_PAIRWISE_ZERO_SUM_REWARD" not in os.environ
