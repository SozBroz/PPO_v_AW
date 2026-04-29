"""
Environment hygiene for ``train.py`` subprocesses.

``train.py`` mirrors ``--learner-greedy-mix``, ``--egocentric-episode-prob``,
``--capture-move-gate``, and ``--pairwise-zero-sum-reward`` into ``os.environ``
so worker processes see the same knobs.  Those variables must not
be injected from PowerShell, repo ``.env``, or an orchestrator
``{**os.environ, **launch_overlay}`` merge ahead of argv — **CLI is canonical**.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

TRAIN_CLI_OWNED_ENV_KEYS: frozenset[str] = frozenset(
    {
        "AWBW_LEARNER_GREEDY_MIX",
        "AWBW_EGOCENTRIC_EPISODE_PROB",
        "AWBW_CAPTURE_MOVE_GATE",
        # Deprecated: strip stale shells/overlays so the extra property-loss
        # punishment stays off unless a developer intentionally wires a new path.
        "AWBW_PHI_ENEMY_PROPERTY_CAPTURE_PENALTY",
        "AWBW_PAIRWISE_ZERO_SUM_REWARD",
    }
)


def environ_for_train_subprocess(
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy *base* (default ``os.environ``) minus :data:`TRAIN_CLI_OWNED_ENV_KEYS`."""
    if base is None:
        base = os.environ
    return {k: v for k, v in base.items() if k not in TRAIN_CLI_OWNED_ENV_KEYS}


def pop_train_cli_owned_keys_from_os_environ() -> None:
    """Remove CLI-owned keys from ``os.environ`` (e.g. after ``load_dotenv``)."""
    for k in TRAIN_CLI_OWNED_ENV_KEYS:
        os.environ.pop(k, None)
