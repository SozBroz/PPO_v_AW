"""Turn-level replay buffer for value-guided RHEA training.

This intentionally stores only value-learning inputs. It does not store
candidate_features, candidate_mask, action logprobs, PPO advantages, or other
policy-gradient baggage. One sample is one full acting-player turn transition:

    state_before_turn -> execute RHEA-selected full turn -> state_after_turn

The value learner trains on turn-level TD targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(slots=True)
class RheaTransition:
    spatial_before: np.ndarray
    scalars_before: np.ndarray
    reward_turn: float
    spatial_after: np.ndarray
    scalars_after: np.ndarray
    done: bool
    winner: Optional[int]
    acting_seat: int
    day: int
    phi_delta: float
    value_after_at_search_time: float
    search_score: float


def payload_to_transition(p: dict[str, Any]) -> RheaTransition:
    """Convert a JSON-deserialized dict into a RheaTransition.

    Used when ingesting remote transition files written by rhea_remote_actor.py.
    Handles both list-format arrays (from JSON) and numpy arrays (in-process).
    """
    def _to_float32_float(v):
        if isinstance(v, (list, tuple)):
            return np.array(v, dtype=np.float32)
        return v

    def _to_int64(v):
        if isinstance(v, (list, tuple)):
            return np.array(v, dtype=np.int64)
        return v

    return RheaTransition(
        spatial_before=np.array(p["spatial_before"], dtype=np.float32),
        scalars_before=np.array(p["scalars_before"], dtype=np.float32),
        reward_turn=float(p["reward_turn"]),
        spatial_after=np.array(p["spatial_after"], dtype=np.float32),
        scalars_after=np.array(p["scalars_after"], dtype=np.float32),
        done=bool(p["done"]),
        winner=p.get("winner"),
        acting_seat=int(p["acting_seat"]),
        day=int(p["day"]),
        phi_delta=float(p["phi_delta"]),
        value_after_at_search_time=float(p["value_after_at_search_time"]),
        search_score=float(p["search_score"]),
    )


class RheaReplayBuffer:
    def __init__(self, capacity: int, *, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._data: list[RheaTransition] = []
        self._pos = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._data)

    def add(self, t: RheaTransition) -> None:
        if len(self._data) < self.capacity:
            self._data.append(t)
        else:
            self._data[self._pos] = t
            self._pos = (self._pos + 1) % self.capacity

    def add_batch(self, transitions: list[RheaTransition]) -> int:
        """Add a batch of transitions. Returns number actually added (capped by capacity).

        Used by the multi-machine learner when ingesting remote transition files
        written by rhea_remote_actor.py on other machines.
        """
        added = 0
        for t in transitions:
            if len(self._data) < self.capacity:
                self._data.append(t)
            else:
                self._data[self._pos] = t
                self._pos = (self._pos + 1) % self.capacity
            added += 1
        return added

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if not self._data:
            raise RuntimeError("cannot sample an empty replay buffer")
        bs = min(int(batch_size), len(self._data))
        idx = self._rng.choice(len(self._data), size=bs, replace=False)
        batch = [self._data[int(i)] for i in idx]

        return {
            "spatial_before": np.stack([b.spatial_before for b in batch]).astype(np.float32),
            "scalars_before": np.stack([b.scalars_before for b in batch]).astype(np.float32),
            "reward_turn": np.asarray([b.reward_turn for b in batch], dtype=np.float32),
            "spatial_after": np.stack([b.spatial_after for b in batch]).astype(np.float32),
            "scalars_after": np.stack([b.scalars_after for b in batch]).astype(np.float32),
            "done": np.asarray([b.done for b in batch], dtype=np.float32),
            "winner": np.asarray([(-1 if b.winner is None else b.winner) for b in batch], dtype=np.int64),
            "acting_seat": np.asarray([b.acting_seat for b in batch], dtype=np.int64),
            "day": np.asarray([b.day for b in batch], dtype=np.int64),
            "phi_delta": np.asarray([b.phi_delta for b in batch], dtype=np.float32),
            "value_after_at_search_time": np.asarray([b.value_after_at_search_time for b in batch], dtype=np.float32),
            "search_score": np.asarray([b.search_score for b in batch], dtype=np.float32),
        }