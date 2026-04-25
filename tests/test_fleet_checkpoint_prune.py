"""Disk retention and ordering for `checkpoint_*.zip` PPO snapshots."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from rl.fleet_env import (
    new_checkpoint_stem_utc,
    prune_checkpoint_zip_snapshots,
    sorted_checkpoint_zip_paths,
)


def test_prune_keeps_newest_hundred() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for i in range(105):
            p = d / f"checkpoint_20000101T000000_{i:09d}Z.zip"
            p.write_bytes(b"x")
            t = 1_000.0 + float(i)
            os.utime(p, (t, t))
        removed = prune_checkpoint_zip_snapshots(d, 100)
        assert removed == 5
        kept = sorted_checkpoint_zip_paths(d)
        assert len(kept) == 100
        # Oldest five (i=0..4) pruned; remaining start at i=5
        assert kept[0].name == "checkpoint_20000101T000000_000000005Z.zip"
        assert kept[-1].name == "checkpoint_20000101T000000_000000104Z.zip"
        # Uniquely named stems (merge-safe); two saves never share the same stem
        a = new_checkpoint_stem_utc()
        b = new_checkpoint_stem_utc()
        assert a != b
        assert a.startswith("checkpoint_")
        assert a.endswith("Z")
        assert ":" not in a


def test_prune_disabled_when_cap_zero() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for i in range(3):
            p = d / f"checkpoint_20000101T000000_{i:09d}Z.zip"
            p.write_bytes(b"x")
            os.utime(p, (float(i), float(i)))
        assert prune_checkpoint_zip_snapshots(d, 0) == 0
        assert len(sorted_checkpoint_zip_paths(d)) == 3


def test_sorted_checkpoint_paths_orders_by_mtime_then_name() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        p1 = d / "checkpoint_0001.zip"
        p2 = d / "checkpoint_0002.zip"
        p3 = d / "checkpoint_0010.zip"
        p1.write_bytes(b"c")
        p2.write_bytes(b"a")
        p3.write_bytes(b"b")
        os.utime(p1, (30.0, 30.0))
        os.utime(p2, (10.0, 10.0))
        os.utime(p3, (20.0, 20.0))
        names = [p.name for p in sorted_checkpoint_zip_paths(d)]
        assert names == ["checkpoint_0002.zip", "checkpoint_0010.zip", "checkpoint_0001.zip"]
