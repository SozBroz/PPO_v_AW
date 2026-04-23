"""Tests for the writer-root assert in rl.paths."""
from __future__ import annotations

from pathlib import Path

import pytest

from rl.paths import LOGS_DIR, REPO_ROOT, _assert_logs_dir_is_writer_root


def test_assert_passes_for_canonical_layout(tmp_path: Path):
    """The helper accepts any ``<root>/logs`` pair, not just the real repo's."""
    (tmp_path / "logs").mkdir()
    _assert_logs_dir_is_writer_root(tmp_path / "logs", tmp_path)


def test_assert_passes_for_real_repo():
    _assert_logs_dir_is_writer_root(LOGS_DIR, REPO_ROOT)


def test_assert_rejects_main_mirror_subpath(tmp_path: Path):
    (tmp_path / "logs" / "logs").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="LOGS_DIR must be REPO_ROOT/logs exactly"):
        _assert_logs_dir_is_writer_root(tmp_path / "logs" / "logs", tmp_path)


def test_assert_rejects_aux_subpath(tmp_path: Path):
    (tmp_path / "logs" / "keras-aux").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="LOGS_DIR must be REPO_ROOT/logs exactly"):
        _assert_logs_dir_is_writer_root(tmp_path / "logs" / "keras-aux", tmp_path)


def test_assert_rejects_unrelated_path(tmp_path: Path):
    (tmp_path / "elsewhere").mkdir()
    with pytest.raises(RuntimeError, match="LOGS_DIR must be REPO_ROOT/logs exactly"):
        _assert_logs_dir_is_writer_root(tmp_path / "elsewhere", tmp_path)


def test_assert_rejects_sibling_named_logs(tmp_path: Path):
    """A ``logs`` dir that's a sibling of the repo root, not under it, must fail."""
    repo = tmp_path / "repo"
    repo.mkdir()
    sibling_logs = tmp_path / "logs"
    sibling_logs.mkdir()
    with pytest.raises(RuntimeError, match="LOGS_DIR must be REPO_ROOT/logs exactly"):
        _assert_logs_dir_is_writer_root(sibling_logs, repo)
