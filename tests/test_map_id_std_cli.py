"""train.py --map-id std / gl-std → GL std pool (map_id None)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _fo():
    p = REPO / "scripts" / "fleet_orchestrator.py"
    spec = importlib.util.spec_from_file_location("fo_map_std", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["fo_map_std"] = m
    spec.loader.exec_module(m)
    return m


def test_parse_map_id_std_aliases() -> None:
    from train import _parse_map_id_cli, build_train_argument_parser

    assert _parse_map_id_cli("std") is None
    assert _parse_map_id_cli("STD") is None
    assert _parse_map_id_cli("gl-std") is None
    assert _parse_map_id_cli("123858") == [123858]
    assert _parse_map_id_cli("123858, 133665") == [123858, 133665]
    with pytest.raises(SystemExit):
        build_train_argument_parser().parse_args(["--map-id", "not-a-map"])


def test_parse_co_csv_cli() -> None:
    from train import _parse_co_csv_cli, build_train_argument_parser

    assert _parse_co_csv_cli("1") == [1]
    assert _parse_co_csv_cli("14,12") == [14, 12]
    assert _parse_co_csv_cli("1, 1, 7") == [1, 7]
    p = build_train_argument_parser()
    ns = p.parse_args(["--co-p0", "1,14", "--co-p1", "12"])
    assert ns.co_p0 == [1, 14]
    assert ns.co_p1 == [12]


def test_build_train_argv_emits_map_id_std_string() -> None:
    fo = _fo()
    doc = {
        "args": {
            "--n-envs": 4,
            "--map-id": "std",
            "--tier": "T3",
        }
    }
    argv = fo.build_train_argv_from_proposed_args(doc, repo_root=REPO)
    i = argv.index("--map-id")
    assert argv[i + 1] == "std"


def test_cli_map_id_std_roundtrip() -> None:
    from train import build_train_argument_parser

    p = build_train_argument_parser()
    args = p.parse_args(["--map-id", "std", "--tier", "T3"])
    assert args.map_id is None
    assert args.tier == "T3"
