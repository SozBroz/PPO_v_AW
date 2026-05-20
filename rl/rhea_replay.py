"""Turn-level replay buffer for value-guided RHEA training.

This intentionally stores only value-learning inputs. It does not store
candidate_features, candidate_mask, action logprobs, PPO advantages, or other
policy-gradient baggage. One sample is one full acting-player turn transition:

    state_before_turn -> execute RHEA-selected full turn -> state_after_turn

The value learner trains on turn-level TD targets.

Step-level schema (``RheaStepTransition``, payload ``kind="step"``) is defined
for future stepwise training; ingest accepts v1 turn payloads by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

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


@dataclass(slots=True)
class RheaStepTransition:
    """One executed engine micro-step on the real board (pre-step convention)."""

    spatial: np.ndarray
    scalars: np.ndarray
    spatial_next: np.ndarray
    scalars_next: np.ndarray
    phi_delta: float
    acting_seat: int
    day: int
    step_index: int
    turn_id: int
    done: bool
    turn_done: bool
    winner: Optional[int]
    phase: str
    action_type: str


def _array_f32(v: Any) -> np.ndarray:
    if isinstance(v, np.ndarray):
        return v.astype(np.float32, copy=False)
    return np.array(v, dtype=np.float32)


def is_step_payload(p: dict[str, Any]) -> bool:
    return p.get("kind") == "step" or int(p.get("schema_version", 1)) >= 2


def payload_to_turn_transition(p: dict[str, Any]) -> RheaTransition:
    """Convert a JSON-deserialized v1 turn dict into a RheaTransition."""
    return RheaTransition(
        spatial_before=_array_f32(p["spatial_before"]),
        scalars_before=_array_f32(p["scalars_before"]),
        reward_turn=float(p["reward_turn"]),
        spatial_after=_array_f32(p["spatial_after"]),
        scalars_after=_array_f32(p["scalars_after"]),
        done=bool(p["done"]),
        winner=p.get("winner"),
        acting_seat=int(p["acting_seat"]),
        day=int(p["day"]),
        phi_delta=float(p["phi_delta"]),
        value_after_at_search_time=float(p["value_after_at_search_time"]),
        search_score=float(p["search_score"]),
    )


def payload_to_step_transition(p: dict[str, Any]) -> RheaStepTransition:
    """Convert a JSON-deserialized v2 step dict into a RheaStepTransition."""
    return RheaStepTransition(
        spatial=_array_f32(p["spatial"]),
        scalars=_array_f32(p["scalars"]),
        spatial_next=_array_f32(p["spatial_next"]),
        scalars_next=_array_f32(p["scalars_next"]),
        phi_delta=float(p["phi_delta"]),
        acting_seat=int(p["acting_seat"]),
        day=int(p["day"]),
        step_index=int(p["step_index"]),
        turn_id=int(p.get("turn_id", 0)),
        done=bool(p["done"]),
        turn_done=bool(p["turn_done"]),
        winner=p.get("winner"),
        phase=str(p.get("phase", "")),
        action_type=str(p.get("action_type", "")),
    )


def payload_to_transition(
    p: dict[str, Any],
) -> Union[RheaTransition, RheaStepTransition]:
    """Dispatch turn (v1) vs step (v2) payloads."""
    if is_step_payload(p):
        return payload_to_step_transition(p)
    return payload_to_turn_transition(p)


def transition_to_payload(
    t: RheaTransition,
    *,
    json_safe: bool = False,
) -> dict[str, Any]:
    """Serialize a turn-level transition for IPC / remote actors."""

    def _arr(x: np.ndarray) -> Any:
        if json_safe:
            return x.tolist()
        return x

    return {
        "schema_version": 1,
        "spatial_before": _arr(t.spatial_before),
        "scalars_before": _arr(t.scalars_before),
        "reward_turn": float(t.reward_turn),
        "spatial_after": _arr(t.spatial_after),
        "scalars_after": _arr(t.scalars_after),
        "done": bool(t.done),
        "winner": t.winner,
        "acting_seat": int(t.acting_seat),
        "day": int(t.day),
        "phi_delta": float(t.phi_delta),
        "value_after_at_search_time": float(t.value_after_at_search_time),
        "search_score": float(t.search_score),
    }


def step_to_payload(
    t: RheaStepTransition,
    *,
    json_safe: bool = False,
) -> dict[str, Any]:
    """Serialize a step-level transition (schema v2)."""

    def _arr(x: np.ndarray) -> Any:
        if json_safe:
            return x.tolist()
        return x

    return {
        "schema_version": 2,
        "kind": "step",
        "spatial": _arr(t.spatial),
        "scalars": _arr(t.scalars),
        "spatial_next": _arr(t.spatial_next),
        "scalars_next": _arr(t.scalars_next),
        "phi_delta": float(t.phi_delta),
        "acting_seat": int(t.acting_seat),
        "day": int(t.day),
        "step_index": int(t.step_index),
        "turn_id": int(t.turn_id),
        "done": bool(t.done),
        "turn_done": bool(t.turn_done),
        "winner": t.winner,
        "phase": str(t.phase),
        "action_type": str(t.action_type),
    }


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
