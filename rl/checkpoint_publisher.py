"""
Background async publisher for trainer checkpoints — Phase 10a.

When --local-checkpoint-mirror is set, the trainer writes checkpoints to a
fast local directory (one fsync) and a daemon thread copies them to the
slow shared root in the background. Default: feature off; falls back to
direct write semantics identical to pre-Phase-10a behavior.

Risks documented in .cursor/plans/train.py_fps_campaign_c26ce6d4.plan.md
sections 10a + "Critical risks (Phase 10)".

Publish order contract: for each training save chunk, callers must enqueue
a timestamped ``checkpoint_*.zip`` first, then ``latest.zip`` (e.g. two
``save_and_publish`` calls in that order). The worker copies jobs in queue
order, so a racing aux re-globs the new ``checkpoint_*.zip`` on the shared
root before ``latest.zip`` is replaced.
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = ("CheckpointPublisher",)


@dataclass
class _PublishJob:
    """Internal job: copy *local_path* to *shared_dir* / <stem>.zip."""

    stem: str
    local_path: Path


def _oldest_index_with_prefix(pending: deque[_PublishJob], prefix: str) -> Optional[int]:
    for i, j in enumerate(pending):
        if j.stem.startswith(prefix):
            return i
    return None


class CheckpointPublisher:
    """
    Local-first checkpoint write with async copy to a shared (slow) directory.

    The only public entry points are :meth:`save_and_publish`,
    :meth:`publish_existing`, :meth:`drain`, and :meth:`close`.
    """

    def __init__(
        self,
        *,
        local_mirror_dir: Path,
        shared_dir: Path,
        queue_max: int = 4,
        drain_timeout_s: float = 60.0,
        logger=None,
    ) -> None:
        self._local = Path(local_mirror_dir).resolve()
        self._shared = Path(shared_dir).resolve()
        if not self._shared.is_dir():
            raise FileNotFoundError(f"shared_dir does not exist: {self._shared}")
        self._local.mkdir(parents=True, exist_ok=True)

        if queue_max < 1:
            raise ValueError("queue_max must be >= 1")
        self._queue_max = int(queue_max)
        self.drain_timeout_s = float(drain_timeout_s)
        self._logger = logger

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending: deque[_PublishJob] = deque()
        self._closed = False
        self._inflight = 0
        # Default-set: wait() is a no-op. Tests clear to pause the worker.
        self._internal_pause_event = threading.Event()
        self._internal_pause_event.set()

        self.queue_depth = 0
        self.publishes_completed = 0
        self.publish_errors = 0
        self.last_publish_wall_s: float = 0.0

        self._worker = threading.Thread(target=self._run, name="CheckpointPublisher", daemon=True)
        self._worker.start()

    def _log_warning(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.warning(msg)
        else:
            print(msg, file=sys.stderr)

    def _enqueue(self, job: _PublishJob) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("CheckpointPublisher is closed")
            # Coalesce: only the newest *latest* is worth publishing.
            if job.stem == "latest":
                self._pending = deque(j for j in self._pending if j.stem != "latest")
            new_is_ckpt = job.stem.startswith("checkpoint_")
            while len(self._pending) >= self._queue_max:
                if new_is_ckpt:
                    oi = _oldest_index_with_prefix(self._pending, "checkpoint_")
                    if oi is not None:
                        del self._pending[oi]
                        continue
                self._pending.popleft()
            self._pending.append(job)
            self.queue_depth = len(self._pending)
            self._cond.notify()

    def save_and_publish(self, model, stem: str) -> Path:
        """Save *model* to the local mirror and enqueue copy to the shared root."""
        from rl.self_play import _atomic_model_save

        _atomic_model_save(model, self._local / stem)
        local_zip = self._local / f"{stem}.zip"
        self._enqueue(_PublishJob(stem=stem, local_path=local_zip))
        return local_zip

    def publish_existing(self, local_zip_path: Path) -> None:
        """Enqueue a copy of an on-disk local zip to the shared root."""
        p = Path(local_zip_path)
        stem = p.stem
        if not p.is_file():
            raise FileNotFoundError(f"local checkpoint zip not found: {p}")
        self._enqueue(_PublishJob(stem=stem, local_path=p.resolve()))

    def _publish_one(self, job: _PublishJob) -> None:
        final_zip = self._shared / f"{job.stem}.zip"
        tmp_pub = self._shared / f"{job.stem}.zip.publishing"
        try:
            if not job.local_path.is_file():
                raise OSError(f"missing local file: {job.local_path}")
            shutil.copy2(str(job.local_path), str(tmp_pub))
            os.replace(str(tmp_pub), str(final_zip))
        except OSError as exc:
            self.publish_errors += 1
            self._log_warning(
                f"[CheckpointPublisher] publish failed: src={job.local_path} "
                f"dst={tmp_pub} err={exc}"
            )
            try:
                if tmp_pub.exists():
                    tmp_pub.unlink()
            except OSError:
                pass
        else:
            self.publishes_completed += 1
            self.last_publish_wall_s = time.time()

    def _run(self) -> None:
        while True:
            with self._lock:
                while not self._pending:
                    if self._closed and self._inflight == 0:
                        return
                    self._cond.wait(timeout=0.2)
                job = self._pending.popleft()
                self.queue_depth = len(self._pending)
                self._inflight = 1
            try:
                self._internal_pause_event.wait()
                self._publish_one(job)
            finally:
                with self._lock:
                    self._inflight = 0
                    self._cond.notify_all()

    def drain(self, timeout_s: Optional[float] = None) -> int:
        """Block until the queue and in-flight work are done (or *timeout_s*).

        Returns the number of **successful** publishes that completed
        after this call started (the delta in :attr:`publishes_completed`).
        """
        t0 = time.time()
        with self._lock:
            start_done = self.publishes_completed
        while True:
            with self._lock:
                if not self._pending and self._inflight == 0:
                    return self.publishes_completed - start_done
            if timeout_s is not None and (time.time() - t0) >= timeout_s:
                with self._lock:
                    return self.publishes_completed - start_done
            with self._lock:
                self._cond.wait(timeout=0.05)

    def close(self) -> None:
        """Request shutdown, then join the worker thread (bounded wait)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cond.notify_all()
        self._worker.join(timeout=self.drain_timeout_s)

    def __enter__(self) -> "CheckpointPublisher":
        return self

    def __exit__(self, *exc) -> None:
        self.drain()
        self.close()
