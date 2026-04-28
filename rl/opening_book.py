"""
Load JSONL opening books and suggest legal flat actions for the opponent seat.

Each line is an object with ``map_id``, ``seat``, ``horizon_days``, and
``action_indices`` (ordered flat indices from human demo ingest).
``horizon_days`` of 0 means no per-day cap; length is the list of
``action_indices`` (typical for books built from truncated pro replays).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class _Book:
    book_id: str
    map_id: int
    seat: int
    co_id: int | None
    horizon_days: int
    action_indices: list[int]


@dataclass
class OpeningBookIndex:
    """Index books by (map_id, seat)."""

    by_map_seat: dict[tuple[int, int], list[_Book]] = field(default_factory=dict)

    @classmethod
    def from_jsonl(cls, path: Path) -> "OpeningBookIndex":
        idx = cls()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                cid = o.get("co_id")
                if cid is None:
                    seat = int(o.get("seat", 0) or 0)
                    c0, c1 = o.get("co0"), o.get("co1")
                    if c0 is not None and c1 is not None:
                        cid = int(c0) if seat == 0 else int(c1)
                    else:
                        cid = None
                b = _Book(
                    book_id=str(o.get("book_id", "")),
                    map_id=int(o.get("map_id", 0) or 0),
                    seat=int(o.get("seat", 0) or 0),
                    co_id=int(cid) if cid is not None else None,
                    horizon_days=int(o.get("horizon_days", 0) or 0),
                    action_indices=[int(x) for x in (o.get("action_indices") or [])],
                )
                key = (b.map_id, b.seat)
                idx.by_map_seat.setdefault(key, []).append(b)
        return idx


class OpeningBookController:
    """Per-episode: pick a book, step through ``action_indices`` if still legal."""

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
        mct = max_calendar_turn
        if mct is not None and int(mct) <= 0:
            mct = None
        self._max_calendar_turn: int | None = mct
        self._book: _Book | None = None
        self._cursor = 0
        self._episode_token: int | None = None
        self.actions_used = 0
        self.fallbacks = 0
        self.desync = False
        self.desync_reason: str | None = None
        self.book_id: str | None = None
        self.suggest_calls: int = 0

    def on_episode_start(
        self,
        *,
        episode_id: int,
        map_id: int,
        co_id_for_seat: int | None,
    ) -> None:
        if self._episode_token == episode_id:
            return
        self._episode_token = episode_id
        self._cursor = 0
        self.actions_used = 0
        self.fallbacks = 0
        self.desync = False
        self.desync_reason = None
        self.book_id = None
        self._book = None
        self.suggest_calls = 0
        cands = list(self._index.by_map_seat.get((int(map_id), int(self._seat)), ()))
        if self._strict_co and co_id_for_seat is not None:
            cands = [b for b in cands if b.co_id is None or b.co_id == int(co_id_for_seat)]
        if not cands:
            return
        self._book = self._rng.choice(cands)
        self.book_id = self._book.book_id

    def suggest_flat(
        self,
        *,
        calendar_turn: int,
        action_mask: np.ndarray,
    ) -> int | None:
        """Next legal flat action from the selected book line, or ``None`` if unavailable."""
        b = self._book
        if b is None or not b.action_indices:
            return None
        self.suggest_calls += 1
        if self._max_calendar_turn is not None and int(calendar_turn) > int(
            self._max_calendar_turn
        ):
            return None
        if b.horizon_days and int(calendar_turn) > int(b.horizon_days):
            return None
        if self._cursor >= len(b.action_indices):
            return None
        ai = int(b.action_indices[self._cursor])
        if ai < 0 or ai >= action_mask.shape[0]:
            self._mark_desync("flat_out_of_range")
            return None
        if not bool(action_mask[ai]):
            self._mark_desync("action_not_legal")
            return None
        self._cursor += 1
        self.actions_used += 1
        return ai

    def _mark_desync(self, reason: str) -> None:
        self.desync = True
        self.desync_reason = reason
        self._book = None


class OpeningBookCheckpointOpponent:
    """Try opening book lines first, then delegate to a :class:`_CheckpointOpponent`."""

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
        self._index = OpeningBookIndex.from_jsonl(Path(book_path))
        self._book_seat = int(book_seat)
        self._book_prob = float(max(0.0, min(1.0, book_prob)))
        md = max_day
        if md is not None and int(md) <= 0:
            md = None
        self._ctl = OpeningBookController(
            self._index,
            seat=self._book_seat,
            strict_co=bool(strict_co),
            rng=random.Random(int(seed) + 17),
            max_calendar_turn=md,
        )
        self._prob_rng = random.Random(int(seed))
        self._env_ref: Any = None
        self._last_episode_id: int | None = None
        self._episode_use_book: bool = False
        self._wref = weakref

    def reload_pool(self, zip_paths: list[str] | None = None) -> int | None:
        """Delegate Phase 10c opponent pool refresh to the inner checkpoint opponent."""
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
        if fn is None:
            return True
        return bool(fn())

    def mode(self) -> str:
        if self._ctl.book_id:
            im = getattr(self._inner, "mode", None)
            inner_label = str(im()) if callable(im) else "checkpoint"
            return f"opening_book+{inner_label}"
        im = getattr(self._inner, "mode", None)
        return str(im()) if callable(im) else "checkpoint"

    def __call__(self, obs: object, mask: object) -> int:
        import numpy as np

        env = self._env_ref() if self._env_ref is not None else None
        st = getattr(env, "state", None) if env is not None else None
        m = np.asarray(mask, dtype=bool)
        if st is not None and env is not None:
            eid = int(getattr(env, "_episode_id", 0) or 0)
            if eid != self._last_episode_id:
                self._last_episode_id = eid
                map_id = int(st.map_data.map_id)
                my_seat = int(self._book_seat)
                co_s = st.co_states[my_seat] if 0 <= my_seat < len(st.co_states) else None
                co_id = int(co_s.co_id) if co_s is not None else None
                self._ctl.on_episode_start(
                    episode_id=eid, map_id=map_id, co_id_for_seat=co_id
                )
                self._episode_use_book = (
                    self._book_prob > 0.0
                    and self._ctl.book_id is not None
                    and self._prob_rng.random() < self._book_prob
                )
        seat_ok = True
        if st is not None and env is not None:
            enemy_seat = int(getattr(env, "_enemy_seat", self._book_seat))
            active = int(st.active_player)
            seat_ok = (
                int(self._book_seat) == enemy_seat and active == int(self._book_seat)
            )
        use_book = (
            seat_ok
            and self._episode_use_book
            and bool(self._ctl._index.by_map_seat)
            and st is not None
        )
        if use_book:
            t = int(getattr(st, "turn", 0) or 0)
            a = self._ctl.suggest_flat(calendar_turn=t, action_mask=m)
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
        if not isinstance(d, dict):
            return
        p = "p1" if int(self._book_seat) == 1 else "p0"
        d[f"opening_book_id_{p}"] = self._ctl.book_id
        d[f"opening_book_used_{p}"] = bool(self._ctl.actions_used)
        d[f"opening_book_actions_{p}"] = int(self._ctl.actions_used)
        d[f"opening_book_desync_{p}"] = bool(self._ctl.desync)
        d[f"opening_book_fallback_reason_{p}"] = self._ctl.desync_reason
        d[f"opening_book_episode_enabled_{p}"] = bool(self._episode_use_book)
        d[f"opening_book_suggest_calls_{p}"] = int(self._ctl.suggest_calls)
