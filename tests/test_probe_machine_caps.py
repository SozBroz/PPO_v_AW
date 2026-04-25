"""tools/probe_machine_caps — JSON shape, CLI, graceful GPU/torch paths."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_probe_module():
    p = REPO / "tools" / "probe_machine_caps.py"
    name = "probe_machine_caps_test"
    spec = importlib.util.spec_from_file_location(name, p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_build_probe_payload_required_fields_and_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWBW_MACHINE_ID", "pytest-m1")
    mod = _load_probe_module()
    d = mod.build_probe_payload()
    assert d["machine_id"] == "pytest-m1"
    assert isinstance(d["probed_at"], str) and "T" in d["probed_at"]
    assert d["cpu"]["physical_cores"] >= 0
    assert d["cpu"]["logical_processors"] >= 1
    assert isinstance(d["cpu"]["model_name"], str)
    assert isinstance(d["ram"]["total_gb"], float)
    assert isinstance(d["ram"]["free_gb_at_probe"], float)
    assert isinstance(d["gpu"]["available"], bool)
    assert "device_name" in d["gpu"]
    assert d["gpu"]["vram_total_gb"] is None or isinstance(d["gpu"]["vram_total_gb"], (int, float))
    assert isinstance(d["disk"]["checkpoint_root_writable"], bool)
    assert isinstance(d["disk"]["checkpoint_root_path"], str)
    assert d["platform"] == sys.platform


def test_build_probe_payload_machine_id_unknown_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AWBW_MACHINE_ID", raising=False)
    mod = _load_probe_module()
    d = mod.build_probe_payload()
    assert d["machine_id"] == "unknown"


def test_build_probe_payload_machine_id_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWBW_MACHINE_ID", "ignored-for-override")
    mod = _load_probe_module()
    d = mod.build_probe_payload(machine_id_override="pc-b")
    assert d["machine_id"] == "pc-b"


def test_try_cuda_props_import_error_subprocess() -> None:
    """Isolated process so blocking ``torch`` does not corrupt the main test VM."""
    code = r"""
import importlib.util
import sys
import builtins
from pathlib import Path
real = builtins.__import__
def di(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or (isinstance(name, str) and name.startswith("torch.")):
        raise ImportError("torch blocked")
    return real(name, globals, locals, fromlist, level)
builtins.__import__ = di
repo = Path(sys.argv[1])
p = repo / "tools" / "probe_machine_caps.py"
spec = importlib.util.spec_from_file_location("pmc_iso", p)
m = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(m)
assert m._try_cuda_props() == (False, None, None)
"""
    pr = subprocess.run(
        [sys.executable, "-c", code, str(REPO)],
        capture_output=True,
        text=True,
    )
    assert pr.returncode == 0, pr.stdout + pr.stderr


def test_probe_script_print_only() -> None:
    env = os.environ.copy()
    env["AWBW_MACHINE_ID"] = "subprocess-probe"
    pr = subprocess.run(
        [sys.executable, str(REPO / "tools" / "probe_machine_caps.py"), "--print-only"],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert pr.returncode == 0, pr.stderr
    d = json.loads(pr.stdout)
    assert d["machine_id"] == "subprocess-probe"
    assert "cpu" in d and "gpu" in d
