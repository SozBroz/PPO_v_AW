"""Two-sided opening-book support for AWBW training.

The runtime unit is still a flat action-index book because that is what the
human-demo ingest currently emits.  This module deliberately treats CO IDs as
metadata by default: books are indexed by ``(map_id, seat)`` and only filtered by
CO if a caller explicitly opts in with ``strict_co=True``.

The important design point is PPO correctness: learner-side books should be
used by *forcing the legal-action mask* to the next book action, not by silently
replacing the sampled action after PPO has already recorded a different action.
Opponent-side books can be selected directly because opponent actions are not
stored in the learner rollout buffer.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class _Book:
    book_id: str
    map_id: int
    seat: int
    co_id: int | None
    horizon_days: int
    action_indices: list[int]
    source_game_id: int | None = None
    session_id: str | None = None


@dataclass
class OpeningBookIndex:
    """Index flat-action opening books by map and engine seat."""

    by_map_seat: dict[tuple[int, int], list[_Book]] = field(default_factory=dict)

    @classmethod
    def from_jsonl(cls, path: Path | str) -> "OpeningBookIndex":
        idx = cls()
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)

                # Accept either the legacy per-seat schema or a simple joint schema
                # with ``seats: {"0": {"action_indices": [...]}, "1": ...}``.
                if isinstance(o.get("seats"), dict):
                    for seat_s, payload in o["seats"].items():
                        if not isinstance(payload, dict):
                            continue
                        b = _book_from_obj(o, int(seat_s), payload, line_no)
                        if b is not None:
                            idx.by_map_seat.setdefault((b.map_id, b.seat), []).append(b)
                    continue

                b = _book_from_obj(o, int(o.get("seat", 0) or 0), o, line_no)
                if b is not None:
                    idx.by_map_seat.setdefault((b.map_id, b.seat), []).append(b)
        return idx


def _book_from_obj(
    root: dict[str, Any], seat: int, payload: dict[str, Any], line_no: int
) -> _Book | None:
    try:
        map_id = int(root.get("map_id", 0) or 0)
        action_indices = [int(x) for x in (payload.get("action_indices") or [])]
    except Exception:
        return None
    if map_id <= 0 or seat not in (0, 1) or not action_indices:
        return None
    co_id_raw = payload.get("co_id", root.get("co_id"))
    if co_id_raw in (None, "", 0, "0"):
        c0 = payload.get("co0", root.get("co0"))
        c1 = payload.get("co1", root.get("co1"))
        if c0 is not None and c1 is not None:
            co_id_raw = int(c0) if seat == 0 else int(c1)
    co_id = int(co_id_raw) if co_id_raw not in (None, "", 0, "0") else None
    base_book_id = str(root.get("joint_book_id") or root.get("book_id") or f"line{line_no}")
    if "joint_book_id" in root or isinstance(root.get("seats"), dict):
        book_id = f"{base_book_id}_s{seat}"
    else:
        book_id = base_book_id
    source_game = root.get("source_game_id")
    return _Book(
        book_id=book_id,
        map_id=map_id,
        seat=seat,
        co_id=co_id,
        horizon_days=int(root.get("horizon_days", payload.get("horizon_days", 0)) or 0),
        action_indices=action_indices,
        source_game_id=int(source_game) if source_game not in (None, "", 0, "0") else None,
        session_id=str(root.get("book_session_id") or root.get("session_id") or "") or None,
    )


class OpeningBookController:
    """Per-episode cursor for one engine seat."""

    def __init__(
        self,
        index: OpeningBookIndex,
        *,
        seat: int,
        strict_co: bool,
        rng: random.Random,
        max_calendar_turn: int | None,
    ) -> None:
        self._index = index
        self._seat = int(seat)
        self._strict_co = bool(strict_co)
        self._rng = rng
        mct = max_calendar_turn if max_calendar_turn is not None and int(max_calendar_turn) > 0 else None
        self._max_calendar_turn = mct
        self._book: _Book | None = None
        self._cursor = 0
        self._episode_token: int | None = None
        self.actions_used = 0
        self.fallbacks = 0
        self.desync = False
        self.desync_reason: str | None = None
        self.book_id: str | None = None
        self.suggest_calls = 0
        self.episode_enabled = False
        self.candidate_count = 0

    @property
    def seat(self) -> int:
        return self._seat

    def on_episode_start(
        self,
        *,
        episode_id: int,
        map_id: int,
        co_id_for_seat: int | None,
        enabled: bool,
    ) -> None:
        if self._episode_token == int(episode_id):
            return
        self._episode_token = int(episode_id)
        self._cursor = 0
        self.actions_used = 0
        self.fallbacks = 0
        self.desync = False
        self.desync_reason = None
        self.book_id = None
        self._book = None
        self.suggest_calls = 0
        self.episode_enabled = bool(enabled)
        cands = list(self._index.by_map_seat.get((int(map_id), int(self._seat)), ()))
        self.candidate_count = len(cands)
        if self._strict_co and co_id_for_seat is not None:
            cands = [b for b in cands if b.co_id is None or b.co_id == int(co_id_for_seat)]
        if not enabled or not cands:
            return
        self._book = self._rng.choice(cands)
        self.book_id = self._book.book_id

    def peek_flat(self, *, calendar_turn: int, action_mask: np.ndarray) -> int | None:
        return self._next_flat(calendar_turn=calendar_turn, action_mask=action_mask, advance=False)

    def peek_next_flat_safe(
        self, *, calendar_turn: int, action_mask: np.ndarray
    ) -> int | None:
        """Return the booked flat index **only if** it is legal under *action_mask*.

        Does **not** advance the cursor, increment counters, or call
        :meth:`_mark_desync`. Used by the env to decide whether capture-greedy
        teacher overrides must be suppressed while a joint opening book line is
        still active — applying the teacher after a book-aligned commit would
        execute a different move than the book cursor assumes and surfaces as
        ``action_not_legal`` on the next peek.
        """
        b = self._book
        if b is None or not b.action_indices or not self.episode_enabled:
            return None
        if self._max_calendar_turn is not None and int(calendar_turn) > int(
            self._max_calendar_turn
        ):
            return None
        if b.horizon_days and int(calendar_turn) > int(b.horizon_days):
            return None
        if self._cursor >= len(b.action_indices):
            return None
        ai = int(b.action_indices[self._cursor])
        if ai < 0 or ai >= int(action_mask.shape[0]):
            return None
        if not bool(action_mask[ai]):
            return None
        return ai

    def suggest_flat(self, *, calendar_turn: int, action_mask: np.ndarray) -> int | None:
        return self._next_flat(calendar_turn=calendar_turn, action_mask=action_mask, advance=True)

    def commit_flat(self, action_idx: int) -> None:
        b = self._book
        if b is None or self._cursor >= len(b.action_indices):
            return
        expected = int(b.action_indices[self._cursor])
        if int(action_idx) == expected:
            self._cursor += 1
            self.actions_used += 1
        else:
            self._mark_desync("learner_action_not_book")

    def _next_flat(self, *, calendar_turn: int, action_mask: np.ndarray, advance: bool) -> int | None:
        b = self._book
        if b is None or not b.action_indices or not self.episode_enabled:
            return None
        if advance:
            self.suggest_calls += 1
        if self._max_calendar_turn is not None and int(calendar_turn) > int(self._max_calendar_turn):
            return None
        if b.horizon_days and int(calendar_turn) > int(b.horizon_days):
            return None
        if self._cursor >= len(b.action_indices):
            return None
        ai = int(b.action_indices[self._cursor])
        if ai < 0 or ai >= int(action_mask.shape[0]):
            self._mark_desync("flat_out_of_range")
            return None
        if not bool(action_mask[ai]):
            self._mark_desync("action_not_legal")
            return None
        if advance:
            self._cursor += 1
            self.actions_used += 1
        return ai

    def _mark_desync(self, reason: str) -> None:
        self.desync = True
        self.desync_reason = reason
        self._book = None

    def log_fields(self) -> dict[str, Any]:
        p = "p1" if self._seat == 1 else "p0"
        return {
            f"opening_book_id_{p}": self.book_id,
            f"opening_book_used_{p}": bool(self.actions_used),
            f"opening_book_actions_{p}": int(self.actions_used),
            f"opening_book_desync_{p}": bool(self.desync),
            f"opening_book_fallback_reason_{p}": self.desync_reason,
            f"opening_book_episode_enabled_{p}": bool(self.episode_enabled),
            f"opening_book_suggest_calls_{p}": int(self.suggest_calls),
            f"opening_book_candidate_count_{p}": int(self.candidate_count),
        }


class TwoSidedOpeningBookManager:
    """Owns per-seat book controllers for an AWBWEnv episode."""

    def __init__(
        self,
        path: str | Path,
        *,
        seats: str | Iterable[int] = "both",
        prob: float = 1.0,
        strict_co: bool = False,
        max_day: int | None = None,
        seed: int = 0,
    ) -> None:
        self.index = OpeningBookIndex.from_jsonl(Path(path))
        self.prob = max(0.0, min(1.0, float(prob)))
        self.strict_co = bool(strict_co)
        self.rng = random.Random(int(seed))
        self.enabled_seats = _parse_seats(seats)
        self.controllers = {
            0: OpeningBookController(
                self.index,
                seat=0,
                strict_co=self.strict_co,
                rng=random.Random(int(seed) + 101),
                max_calendar_turn=max_day,
            ),
            1: OpeningBookController(
                self.index,
                seat=1,
                strict_co=self.strict_co,
                rng=random.Random(int(seed) + 202),
                max_calendar_turn=max_day,
            ),
        }

    def on_episode_start(
        self,
        *,
        episode_id: int,
        map_id: int,
        co_ids: list[int | None],
    ) -> None:
        use_episode = self.prob > 0.0 and self.rng.random() < self.prob
        for seat, ctl in self.controllers.items():
            ctl.on_episode_start(
                episode_id=episode_id,
                map_id=map_id,
                co_id_for_seat=co_ids[seat] if 0 <= seat < len(co_ids) else None,
                enabled=bool(use_episode and seat in self.enabled_seats),
            )

    def peek_flat(
        self, *, seat: int, calendar_turn: int, action_mask: np.ndarray
    ) -> int | None:
        ctl = self.controllers.get(int(seat))
        if ctl is None:
            return None
        return ctl.peek_flat(calendar_turn=calendar_turn, action_mask=action_mask)

    def peek_book_candidate_flat_safe(
        self, *, seat: int, calendar_turn: int, action_mask: np.ndarray
    ) -> int | None:
        ctl = self.controllers.get(int(seat))
        if ctl is None:
            return None
        return ctl.peek_next_flat_safe(calendar_turn=calendar_turn, action_mask=action_mask)

    def suggest_flat(
        self, *, seat: int, calendar_turn: int, action_mask: np.ndarray
    ) -> int | None:
        ctl = self.controllers.get(int(seat))
        if ctl is None:
            return None
        return ctl.suggest_flat(calendar_turn=calendar_turn, action_mask=action_mask)

    def commit_flat(self, *, seat: int, action_idx: int) -> None:
        ctl = self.controllers.get(int(seat))
        if ctl is not None:
            ctl.commit_flat(int(action_idx))

    def log_fields(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for ctl in self.controllers.values():
            out.update(ctl.log_fields())
        return out


def _parse_seats(seats: str | Iterable[int]) -> set[int]:
    if isinstance(seats, str):
        raw = seats.strip().lower()
        if raw in ("both", "all", "0,1", "1,0"):
            return {0, 1}
        if raw in ("p0", "0"):
            return {0}
        if raw in ("p1", "1"):
            return {1}
        if raw in ("none", "off", "false", "0.0"):
            return set()
        vals: set[int] = set()
        for part in raw.replace(";", ",").split(","):
            part = part.strip()
            if part in ("0", "p0"):
                vals.add(0)
            elif part in ("1", "p1"):
                vals.add(1)
        return vals or {0, 1}
    return {int(x) for x in seats if int(x) in (0, 1)}


class OpeningBookCheckpointOpponent:
    """Backward-compatible opponent wrapper.

    New training should pass opening-book config to AWBWEnv instead.  This class
    remains so old launch commands do not crash; it still controls only the
    wrapped non-learner opponent.
    """

    def __init__(
        self,
        inner: object,
        book_path: str | Path,
        *,
        book_seat: int = 1,
        book_prob: float = 1.0,
        strict_co: bool = False,
        max_day: int | None = None,
        seed: int = 0,
    ) -> None:
        import weakref

        self._inner = inner
        self._book_seat = int(book_seat)
        self._mgr = TwoSidedOpeningBookManager(
            book_path,
            seats=str(self._book_seat),
            prob=book_prob,
            strict_co=strict_co,
            max_day=max_day,
            seed=seed,
        )
        self._env_ref: Any = None
        self._last_episode_id: int | None = None
        self._wref = weakref

    def reload_pool(self, zip_paths: list[str] | None = None) -> int | None:
        fn = getattr(self._inner, "reload_pool", None)
        if fn is None:
            return None
        return fn(zip_paths)

    def attach_env(self, env: object) -> None:
        if hasattr(self._inner, "attach_env"):
            self._inner.attach_env(env)
        self._env_ref = self._wref.ref(env)

    @property
    def reload_count(self) -> int:
        return int(getattr(self._inner, "reload_count", 0) or 0)

    @property
    def _model(self) -> Any:
        """Same checkpoint as P1 after book lines; used for spirit / heuristic value diag."""
        return getattr(self._inner, "_model", None)

    def needs_observation(self) -> bool:
        fn = getattr(self._inner, "needs_observation", None)
        return True if fn is None else bool(fn())

    def mode(self) -> str:
        im = getattr(self._inner, "mode", None)
        inner_label = str(im()) if callable(im) else "checkpoint"
        return f"opening_book+{inner_label}"

    def __call__(self, obs: object, mask: object) -> int:
        env = self._env_ref() if self._env_ref is not None else None
        st = getattr(env, "state", None) if env is not None else None
        m = np.asarray(mask, dtype=bool)
        if st is not None and env is not None:
            eid = int(getattr(env, "_episode_id", 0) or 0)
            if eid != self._last_episode_id:
                self._last_episode_id = eid
                co_ids = [None, None]
                try:
                    co_ids = [int(st.co_states[0].co_id), int(st.co_states[1].co_id)]
                except Exception:
                    pass
                self._mgr.on_episode_start(
                    episode_id=eid,
                    map_id=int(st.map_data.map_id),
                    co_ids=co_ids,
                )
            if int(st.active_player) == int(self._book_seat):
                a = self._mgr.suggest_flat(
                    seat=self._book_seat,
                    calendar_turn=int(getattr(st, "turn", 0) or 0),
                    action_mask=m,
                )
                if a is not None:
                    self._sync_log(env)
                    return int(a)
        act = int(self._inner(obs, m))
        self._sync_log(env)
        return act

    def _sync_log(self, env: object | None) -> None:
        if env is None:
            return
        d = getattr(env, "_opening_book_log", None)
        if isinstance(d, dict):
            d.update(self._mgr.log_fields())
