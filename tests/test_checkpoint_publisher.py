"""
Tests for ``rl.checkpoint_publisher.CheckpointPublisher`` (Phase 10a).

The fake :meth:`Model.save` used here is intentionally not SB3 — the publisher
only consumes the produced ``.zip`` bytes; no algorithm introspection.
"""
from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path
from unittest import mock

# Repo root (tests/ lives under it)
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.checkpoint_publisher import CheckpointPublisher  # noqa: E402


class _FakeSb3Model:
    """Tiny stand-in: ``save`` writes a ``.zip`` next to the temp base, like SB3."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def save(self, path: str) -> None:
        p = Path(path)
        final_zip = p.parent / f"{p.name}.zip"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("payload.bin", self._payload)
        final_zip.write_bytes(buf.getvalue())


def test_publisher_creates_local_then_shared_after_drain(tmp_path: Path) -> None:
    local = tmp_path / "loc"
    shared = tmp_path / "sh"
    shared.mkdir()
    m = _FakeSb3Model(b"alpha")
    pub = CheckpointPublisher(
        local_mirror_dir=local,
        shared_dir=shared,
        queue_max=4,
    )
    try:
        p = pub.save_and_publish(m, "checkpoint_test")
        assert p == local / "checkpoint_test.zip"
        assert p.is_file()
        assert not (shared / "checkpoint_test.zip").is_file()
        pub.drain(timeout_s=5.0)
        assert (shared / "checkpoint_test.zip").is_file()
        assert (shared / "checkpoint_test.zip").read_bytes() == p.read_bytes()
    finally:
        pub.close()


def test_publisher_default_off_is_a_no_op_in_self_play_trainer(tmp_path: Path) -> None:
    from rl.self_play import SelfPlayTrainer

    t = SelfPlayTrainer(
        total_timesteps=1,
        save_every=10_000_000,
        checkpoint_dir=tmp_path,
        local_checkpoint_mirror=None,
    )
    assert t._publisher is None
    assert t.local_checkpoint_mirror is None


def test_publisher_orders_timestamped_before_latest(tmp_path: Path) -> None:
    local = tmp_path / "loc"
    shared = tmp_path / "sh"
    shared.mkdir()
    pub = CheckpointPublisher(local_mirror_dir=local, shared_dir=shared, queue_max=4)
    try:
        pub.save_and_publish(_FakeSb3Model(b"ts"), "checkpoint_aaa")
        pub.save_and_publish(_FakeSb3Model(b"lat"), "latest")
        pub.drain(timeout_s=5.0)
        t_zip = shared / "checkpoint_aaa.zip"
        l_zip = shared / "latest.zip"
        assert t_zip.is_file() and l_zip.is_file()
        assert os.path.getmtime(t_zip) <= os.path.getmtime(l_zip) + 1e-3
    finally:
        pub.close()


def test_publisher_drop_oldest_when_queue_full(tmp_path: Path) -> None:
    local = tmp_path / "loc"
    shared = tmp_path / "sh"
    shared.mkdir()
    pub = CheckpointPublisher(local_mirror_dir=local, shared_dir=shared, queue_max=2)
    try:
        pub._internal_pause_event.clear()
        for i in range(5):
            m = _FakeSb3Model(bytes([i]))
            pub.save_and_publish(m, "latest")
        with pub._lock:  # type: ignore[attr-defined]
            assert pub.queue_depth == 1
        pub._internal_pause_event.set()
        pub.drain(timeout_s=5.0)
        z = shared / "latest.zip"
        assert z.is_file()
        with zipfile.ZipFile(z) as zf:
            assert zf.read("payload.bin") == bytes([4])
    finally:
        pub.close()


def test_publisher_drain_returns_count_within_timeout(tmp_path: Path) -> None:
    local = tmp_path / "loc"
    shared = tmp_path / "sh"
    shared.mkdir()
    pub = CheckpointPublisher(local_mirror_dir=local, shared_dir=shared, queue_max=4)
    try:
        pub._internal_pause_event.clear()
        for i, b in enumerate([b"aa", b"bb", b"cc"]):
            pub.save_and_publish(_FakeSb3Model(b), f"chunk_{i}")
        pub._internal_pause_event.set()
        n = pub.drain(timeout_s=5.0)
        assert n == 3
        assert pub.queue_depth == 0
    finally:
        pub.close()


def test_publisher_publish_error_does_not_kill_thread(tmp_path: Path) -> None:
    local = tmp_path / "loc"
    shared = tmp_path / "sh"
    shared.mkdir()
    pub = CheckpointPublisher(local_mirror_dir=local, shared_dir=shared, queue_max=4)
    real_replace = os.replace
    n_publishing = 0

    def _flaky(a: "str | os.PathLike[str]", b: "str | os.PathLike[str]") -> None:
        nonlocal n_publishing
        if ".publishing" in str(a):
            n_publishing += 1
            if n_publishing == 1:
                raise OSError("simulated replace failure (publisher path only)")
        return real_replace(a, b)

    try:
        m1 = _FakeSb3Model(b"one")
        m2 = _FakeSb3Model(b"two")
        with mock.patch("os.replace", side_effect=_flaky):
            pub.save_and_publish(m1, "errz")
            pub.drain(timeout_s=5.0)
            assert pub.publish_errors == 1
            assert pub.publishes_completed == 0
            pub.save_and_publish(m2, "errz")
            pub.drain(timeout_s=5.0)
        assert pub.publishes_completed == 1
        assert (shared / "errz.zip").is_file()
    finally:
        pub.close()


def test_publisher_close_is_idempotent(tmp_path: Path) -> None:
    shared = tmp_path / "s"
    shared.mkdir()
    pub = CheckpointPublisher(local_mirror_dir=tmp_path / "l", shared_dir=shared, queue_max=2)
    pub.close()
    pub.close()
    assert not pub._worker.is_alive()  # type: ignore[attr-defined]
