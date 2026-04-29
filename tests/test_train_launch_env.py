"""rl.train_launch_env — strip curriculum CLI mirrors from inherited env."""
from __future__ import annotations


def test_environ_for_train_subprocess_drops_cli_owned_keys(monkeypatch) -> None:
    import os

    monkeypatch.setenv("AWBW_LEARNER_GREEDY_MIX", "0.99")
    monkeypatch.setenv("AWBW_EGOCENTRIC_EPISODE_PROB", "0.5")
    monkeypatch.setenv("AWBW_CAPTURE_MOVE_GATE", "1")
    monkeypatch.setenv("AWBW_PHI_ENEMY_PROPERTY_CAPTURE_PENALTY", "1.0")
    monkeypatch.setenv("AWBW_PAIRWISE_ZERO_SUM_REWARD", "1")
    monkeypatch.setenv("AWBW_MACHINE_ID", "x")
    from rl.train_launch_env import TRAIN_CLI_OWNED_ENV_KEYS, environ_for_train_subprocess

    e = environ_for_train_subprocess()
    for k in TRAIN_CLI_OWNED_ENV_KEYS:
        assert k not in e
    assert e.get("AWBW_MACHINE_ID") == "x"


def test_pop_train_cli_owned_keys_from_os_environ(monkeypatch) -> None:
    import os

    monkeypatch.setenv("AWBW_LEARNER_GREEDY_MIX", "0.3")
    monkeypatch.setenv("AWBW_EGOCENTRIC_EPISODE_PROB", "0.2")
    monkeypatch.setenv("AWBW_CAPTURE_MOVE_GATE", "1")
    monkeypatch.setenv("AWBW_PHI_ENEMY_PROPERTY_CAPTURE_PENALTY", "0.8")
    monkeypatch.setenv("AWBW_PAIRWISE_ZERO_SUM_REWARD", "1")
    from rl.train_launch_env import pop_train_cli_owned_keys_from_os_environ

    pop_train_cli_owned_keys_from_os_environ()
    assert "AWBW_LEARNER_GREEDY_MIX" not in os.environ
    assert "AWBW_EGOCENTRIC_EPISODE_PROB" not in os.environ
    assert "AWBW_CAPTURE_MOVE_GATE" not in os.environ
    assert "AWBW_PHI_ENEMY_PROPERTY_CAPTURE_PENALTY" not in os.environ
    assert "AWBW_PAIRWISE_ZERO_SUM_REWARD" not in os.environ
