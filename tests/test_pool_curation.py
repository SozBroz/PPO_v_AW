"""Unit tests for Phase 10b ``prune_checkpoint_zip_curated`` (fleet pool)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from rl.fleet_env import prune_checkpoint_zip_curated, sorted_checkpoint_zip_paths


def _touch_zip(parent: Path, name: str, *, mtime: float | None = None) -> Path:
    p = parent / name
    p.write_bytes(b"")
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def _verdict_json(parent: Path, zip_stem: str, winrate: float) -> None:
    """Matches ``fleet/<id>/eval/<stem>.json`` + daemon field shape."""
    payload = {
        "schema_version": 1,
        "candidate_wins": int(winrate * 10),
        "baseline_wins": 10 - int(winrate * 10),
        "games_decided": 10,
        "winrate": float(winrate),
        "ckpt": f"{zip_stem}.zip",
        "timestamp": time.time(),
    }
    parent.mkdir(parents=True, exist_ok=True)
    (parent / f"{zip_stem}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_curator_falls_back_to_fifo_when_no_verdicts(tmp_path: Path) -> None:
    t0 = time.time() - 86_400.0
    for i in range(20):
        _touch_zip(
            tmp_path,
            f"checkpoint_fifo_{i:02d}.zip",
            mtime=t0 + float(i),
        )
    s = prune_checkpoint_zip_curated(
        tmp_path,
        k_newest=4,
        m_top_winrate=4,
        d_diversity=2,
        verdicts_root=None,
    )
    assert s["fallback_used"] is True
    assert len(s["removed"]) == 20 - (4 + 4 + 2)
    assert len(sorted_checkpoint_zip_paths(tmp_path)) == 4 + 4 + 2


def test_curator_keeps_k_newest_by_mtime(tmp_path: Path) -> None:
    t0 = time.time() - 10_000.0
    paths = [
        _touch_zip(tmp_path, f"checkpoint_k_{i}.zip", mtime=t0 + float(i) * 10.0)
        for i in range(10)
    ]
    s = prune_checkpoint_zip_curated(
        tmp_path,
        k_newest=3,
        m_top_winrate=0,
        d_diversity=0,
        verdicts_root=None,
    )
    assert s["fallback_used"] is True
    kept = {p.name for p in sorted_checkpoint_zip_paths(tmp_path)}
    want = {paths[-1].name, paths[-2].name, paths[-3].name}
    assert kept == want


def test_curator_keeps_m_top_by_winrate(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet" / "test-machine" / "eval"
    t_old = time.time() - 3600.0
    for i in range(10):
        st = f"checkpoint_m_{i}"
        _touch_zip(tmp_path, f"{st}.zip", mtime=t_old)
        _verdict_json(fleet, st, winrate=float(i) / 10.0)
    s = prune_checkpoint_zip_curated(
        tmp_path,
        k_newest=0,
        m_top_winrate=3,
        d_diversity=0,
        verdicts_root=tmp_path / "fleet",
        min_age_minutes=0.0,
    )
    assert s["fallback_used"] is False
    kept = {p.stem for p in sorted_checkpoint_zip_paths(tmp_path)}
    assert kept == {"checkpoint_m_7", "checkpoint_m_8", "checkpoint_m_9"}


def test_curator_union_of_k_m_d(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet" / "m1" / "eval"
    base = time.time() - 50_000.0
    names = [f"checkpoint_u{i}" for i in range(6)]
    for i, st in enumerate(names):
        _touch_zip(tmp_path, f"{st}.zip", mtime=base + float(i) * 100.0)
        _verdict_json(fleet, st, winrate=float(i) * 0.1)
    s = prune_checkpoint_zip_curated(
        tmp_path,
        k_newest=2,
        m_top_winrate=2,
        d_diversity=2,
        verdicts_root=tmp_path / "fleet",
        min_age_minutes=0.0,
    )
    assert s["fallback_used"] is False
    cap = 2 + 2 + 2
    assert s["kept_total"] <= cap
    u = set(s["kept_by_recency"]) | set(s["kept_by_winrate"]) | set(s["kept_by_diversity"])
    assert s["kept_total"] == len(u)


def test_curator_min_age_protects_fresh_zips(tmp_path: Path) -> None:
    now = time.time()
    _touch_zip(tmp_path, "checkpoint_old_a.zip", mtime=now - 600.0)
    _touch_zip(tmp_path, "checkpoint_old_b.zip", mtime=now - 500.0)
    for n in ("checkpoint_f0", "checkpoint_f1", "checkpoint_f2"):
        _touch_zip(tmp_path, f"{n}.zip", mtime=now - 10.0)
    # One verdict for a **non-pool** stem forces the curated code path; no M picks.
    only_eval = tmp_path / "fleet" / "m" / "eval"
    only_eval.mkdir(parents=True, exist_ok=True)
    (only_eval / "not_a_pool_ckpt_stem.json").write_text(
        json.dumps(
            {
                "winrate": 0.0,
                "ckpt": "other_repo.zip",
                "games_decided": 0,
            }
        ),
        encoding="utf-8",
    )
    s2 = prune_checkpoint_zip_curated(
        tmp_path,
        k_newest=1,
        m_top_winrate=0,
        d_diversity=0,
        verdicts_root=tmp_path / "fleet",
        min_age_minutes=5.0,
    )
    assert s2["fallback_used"] is False
    for st in ("checkpoint_f0", "checkpoint_f1", "checkpoint_f2"):
        assert any(p.stem == st for p in sorted_checkpoint_zip_paths(tmp_path))
    assert "checkpoint_old_a" in s2["removed"] and "checkpoint_old_b" in s2["removed"]
    assert s2["kept_total"] == 3


def test_curator_dry_run_deletes_nothing(tmp_path: Path) -> None:
    t0 = time.time() - 400.0
    for i in range(10):
        _touch_zip(tmp_path, f"checkpoint_dr_{i}.zip", mtime=t0 + float(i))
    before = {p.name for p in tmp_path.iterdir()}
    s = prune_checkpoint_zip_curated(
        tmp_path,
        k_newest=1,
        m_top_winrate=0,
        d_diversity=0,
        verdicts_root=None,
        dry_run=True,
    )
    after = {p.name for p in tmp_path.iterdir()}
    assert before == after
    assert len(s["removed"]) == 9


def test_curator_skips_malformed_verdict_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fleet = tmp_path / "fleet" / "z" / "eval"
    fleet.mkdir(parents=True, exist_ok=True)
    (fleet / "bad.json").write_text("not json {{{", encoding="utf-8")
    _verdict_json(fleet, "checkpoint_ok", winrate=0.99)
    _touch_zip(tmp_path, "checkpoint_ok.zip", mtime=time.time() - 100.0)
    _touch_zip(tmp_path, "checkpoint_other.zip", mtime=time.time() - 200.0)
    s = prune_checkpoint_zip_curated(
        tmp_path,
        k_newest=0,
        m_top_winrate=1,
        d_diversity=0,
        verdicts_root=tmp_path / "fleet",
        min_age_minutes=0.0,
    )
    assert s["fallback_used"] is False
    err = capsys.readouterr().err
    assert "[curator] skipping malformed verdict" in err
    assert "bad.json" in err
    assert (tmp_path / "checkpoint_ok.zip").is_file()
    out = {p.stem for p in sorted_checkpoint_zip_paths(tmp_path)}
    assert "checkpoint_ok" in out


def test_curator_returns_summary_shape(tmp_path: Path) -> None:
    _touch_zip(tmp_path, "checkpoint_a.zip", mtime=time.time() - 60.0)
    s = prune_checkpoint_zip_curated(tmp_path, verdicts_root=None)
    assert set(s.keys()) == {
        "kept_total",
        "kept_by_recency",
        "kept_by_winrate",
        "kept_by_diversity",
        "removed",
        "fallback_used",
        "reason",
    }
    assert isinstance(s["kept_total"], int)
    assert isinstance(s["fallback_used"], bool)
    assert isinstance(s["reason"], str)
    for k in ("kept_by_recency", "kept_by_winrate", "kept_by_diversity", "removed"):
        assert isinstance(s[k], list)
        for x in s[k]:
            assert isinstance(x, str)
