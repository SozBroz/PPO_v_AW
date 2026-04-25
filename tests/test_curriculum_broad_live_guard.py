"""curriculum_broad_prob must not apply to live snapshot workers (ladder / live PPO)."""
from __future__ import annotations

import pytest

_MINI_POOL = [
    {
        "map_id": 123001,
        "name": "mini",
        "type": "std",
        "tiers": [
            {"tier_name": "T3", "enabled": True, "co_ids": [1, 7]},
        ],
    },
]


@pytest.fixture
def capture_broad(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    captured: dict[str, float] = {}

    def _fake(
        sample_map_pool,
        *,
        co_p0=None,
        co_p1=None,
        tier_name=None,
        curriculum_broad_prob=0.0,
        rng=None,
    ):
        captured["curriculum_broad_prob"] = float(curriculum_broad_prob)
        meta = sample_map_pool[0]
        tier = meta["tiers"][0]
        return (
            int(meta["map_id"]),
            str(tier["tier_name"]),
            int(tier["co_ids"][0]),
            int(tier["co_ids"][1]),
            str(meta.get("name", "")),
        )

    monkeypatch.setattr("rl.env.sample_training_matchup", _fake)
    return captured


def test_sample_config_passes_full_broad_without_live(
    capture_broad: dict[str, float],
) -> None:
    from rl.env import AWBWEnv

    env = AWBWEnv(
        map_pool=list(_MINI_POOL),
        curriculum_broad_prob=1.0,
        live_snapshot_path=None,
    )
    env._sample_config()
    assert capture_broad["curriculum_broad_prob"] == 1.0


def test_sample_config_zeros_broad_when_live_snapshot_path_set(
    capture_broad: dict[str, float],
) -> None:
    from rl.env import AWBWEnv

    env = AWBWEnv(
        map_pool=list(_MINI_POOL),
        curriculum_broad_prob=1.0,
        live_snapshot_path="C:\\nope\\ladder.pkl",
    )
    env._sample_config()
    assert capture_broad["curriculum_broad_prob"] == 0.0
