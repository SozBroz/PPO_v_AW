"""``rl.ai_vs_ai`` CLI argv stripping — must not pass ``ai_vs_ai.py`` to argparse."""
from __future__ import annotations

import sys
from unittest import mock

import pytest


@pytest.fixture()
def ai_vs_ai_mod():
    import rl.ai_vs_ai as m

    return m


def test_argv_u_then_script_path_strips_script(ai_vs_ai_mod) -> None:
    with mock.patch.object(
        sys,
        "argv",
        [
            r"C:\Python\python.exe",
            "-u",
            r"C:\Users\me\AWBW\rl\ai_vs_ai.py",
            "--random",
            "--no-follow-train",
        ],
    ):
        assert ai_vs_ai_mod._argv_for_this_module() == ["--random", "--no-follow-train"]


def test_argv_m_module_only(ai_vs_ai_mod) -> None:
    with mock.patch.object(
        sys,
        "argv",
        [r"C:\Python\python.exe", "-m", "rl.ai_vs_ai"],
    ):
        assert ai_vs_ai_mod._argv_for_this_module() == []


def test_argv_m_module_with_flags(ai_vs_ai_mod) -> None:
    with mock.patch.object(
        sys,
        "argv",
        [r"C:\Python\python.exe", "-X", "utf8", "-m", "rl.ai_vs_ai", "--seed", "3"],
    ):
        assert ai_vs_ai_mod._argv_for_this_module() == ["--seed", "3"]


def test_argv_relative_script(ai_vs_ai_mod) -> None:
    with mock.patch.object(
        sys,
        "argv",
        ["python", "rl/ai_vs_ai.py", "--map-id", "98"],
    ):
        assert ai_vs_ai_mod._argv_for_this_module() == ["--map-id", "98"]


def test_argv0_resolved_same_as_module_file_returns_flags(ai_vs_ai_mod) -> None:
    """``python -m rl.ai_vs_ai``: no ``-m`` in argv; flags-only tail must not become []."""
    mod_path = str(ai_vs_ai_mod.__file__)
    with mock.patch.object(
        sys,
        "argv",
        [
            mod_path,
            "--no-follow-train",
            "--random",
            "--max-turns",
            "2",
            "--map-id",
            "98",
        ],
    ):
        assert ai_vs_ai_mod._argv_for_this_module() == [
            "--no-follow-train",
            "--random",
            "--max-turns",
            "2",
            "--map-id",
            "98",
        ]
