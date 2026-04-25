"""Phase 11d: MCTS health gate (tools/mcts_health)."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n\n")


def _g(
    mid: str = "pc-b",
    *,
    cap: float = 2.0,
    terr: float = 0.6,
    losses: tuple = (5, 25),
    winner: int = 0,
    turns: float = 30.0,
) -> dict:
    return {
        "machine_id": mid,
        "turns": turns,
        "winner": winner,
        "captures_completed_p0": cap,
        "terrain_usage_p0": terr,
        "losses_hp": [losses[0], losses[1]],
    }


def _bad_rows(n: int) -> list[dict]:
    return [
        _g(
            cap=0.0,
            terr=0.0,
            losses=(30, 5),
            winner=1,
            turns=10.0,
        )
        for _ in range(n)
    ]


def _solid_200_40win() -> list[dict]:
    rows: list[dict] = []
    for i in range(200):
        win = i < 80
        rows.append(
            _g(
                cap=2.0,
                terr=0.6,
                losses=(5, 25) if win else (4, 20),
                winner=0 if win else 1,
                turns=30.0,
            )
        )
    return rows


def test_all_bad_games_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWBW_MCTS_HEALTH_MIN_GAMES", "50")
    p = tmp_path / "game_log.jsonl"
    _write_jsonl(p, _bad_rows(200))
    from tools.mcts_health import compute_health

    v = compute_health("pc-b", tmp_path, window=200)
    assert v.pass_overall is False
    assert v.proposed_mcts_mode == "off"
    assert v.proposed_mcts_sims == 0


def test_mixed_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWBW_MCTS_HEALTH_MIN_GAMES", "50")
    rows: list[dict] = []
    for i in range(200):
        # Strong capture but terrain always zero (fails pass_terrain)
        rows.append(
            _g(
                cap=3.0,
                terr=0.0,
                losses=(5, 25),
                winner=0,
                turns=30.0,
            )
        )
    from tools.mcts_health import compute_health

    v = compute_health("pc-b", tmp_path, window=200)
    assert v.pass_terrain is False
    assert v.pass_overall is False


def test_solid_passes_sims8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWBW_MCTS_HEALTH_MIN_GAMES", "50")
    _write_jsonl(tmp_path / "game_log.jsonl", _solid_200_40win())
    from tools.mcts_health import compute_health

    v = compute_health("pc-b", tmp_path, window=200)
    assert v.pass_overall is True
    assert v.proposed_mcts_mode == "eval_only"
    assert v.proposed_mcts_sims == 8
    assert v.metrics.win_rate == pytest.approx(0.4, abs=0.01)


def test_sims_tier_16(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWBW_MCTS_HEALTH_MIN_GAMES", "50")
    # 0.5 win, 2.5+ captures, all else strong
    rows: list[dict] = []
    for i in range(200):
        win = i < 100
        rows.append(
            _g(
                cap=3.0,
                terr=0.6,
                losses=(4, 22),
                winner=0 if win else 1,
                turns=30.0,
            )
        )
    _write_jsonl(tmp_path / "game_log.jsonl", rows)
    from tools.mcts_health import compute_health

    v = compute_health("pc-b", tmp_path, window=200)
    assert v.metrics.win_rate == pytest.approx(0.5, abs=0.01)
    assert v.metrics.avg_capture_completions_per_game >= 2.5
    assert v.pass_overall is True
    assert v.proposed_mcts_sims == 16


def test_sims_tier_32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWBW_MCTS_HEALTH_MIN_GAMES", "50")
    rows: list[dict] = []
    for i in range(200):
        win = i < 110
        rows.append(
            _g(
                cap=2.0,
                terr=0.55,
                losses=(3, 30),
                winner=0 if win else 1,
                turns=32.0,
            )
        )
    _write_jsonl(tmp_path / "game_log.jsonl", rows)
    from tools.mcts_health import compute_health

    v = compute_health("pc-b", tmp_path, window=200)
    assert v.metrics.win_rate >= 0.55
    # army: all good trades
    assert v.metrics.army_value_lead_pos_rate >= 0.55
    assert v.pass_overall is True
    assert v.proposed_mcts_sims == 32


def test_insufficient_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWBW_MCTS_HEALTH_MIN_GAMES", "50")
    _write_jsonl(tmp_path / "game_log.jsonl", _solid_200_40win()[:49])
    from tools.mcts_health import compute_health

    v = compute_health("pc-b", tmp_path, window=200)
    assert v.games_in_window == 49
    assert v.pass_overall is False
    assert v.proposed_mcts_mode == "off"
    assert v.reasoning == "insufficient data"


def test_env_threshold_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Borderline: barely fails default capture, passes when min lowered
    rows = []
    for i in range(50):
        rows.append(
            _g(
                cap=0.3,
                terr=0.6,
                losses=(4, 20),
                winner=0,
                turns=30.0,
            )
        )
    _write_jsonl(tmp_path / "game_log.jsonl", rows)
    monkeypatch.setenv("AWBW_MCTS_HEALTH_MIN_GAMES", "30")
    monkeypatch.setenv("AWBW_MCTS_HEALTH_CAPTURE_SENSE_MIN", "0.4")
    monkeypatch.setenv("AWBW_MCTS_HEALTH_CAPTURE_COMPLETIONS_MIN", "0.2")

    from tools.mcts_health import compute_health

    v_fail = compute_health("pc-b", tmp_path, window=200)
    assert v_fail.pass_capture is False

    monkeypatch.setenv("AWBW_MCTS_HEALTH_CAPTURE_SENSE_MIN", "0.0")
    monkeypatch.setenv("AWBW_MCTS_HEALTH_CAPTURE_COMPLETIONS_MIN", "0.0")
    v_ok = compute_health("pc-b", tmp_path, window=200)
    assert v_ok.pass_capture is True


def test_write_health_json_uses_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, Path]] = []
    real = os.replace

    def _track(a: str, b: str) -> None:
        calls.append((Path(a), Path(b)))
        real(a, b)

    monkeypatch.setenv("AWBW_MCTS_HEALTH_MIN_GAMES", "1")
    rows = [
        _g(cap=2.0, terr=0.6, winner=0, turns=30.0, losses=(3, 20)),
    ]
    _write_jsonl(tmp_path / "l" / "game_log.jsonl", rows)
    from tools.mcts_health import compute_health, write_health_json

    v = compute_health("pc-b", tmp_path / "l", window=5)
    dest = tmp_path / "fleet" / "pc-b"
    with mock.patch("os.replace", _track):
        p = write_health_json(v, dest)
    assert p == dest / "mcts_health.json"
    assert p.is_file()
    assert any(
        t.suffixes == [".json"] and t.name == "mcts_health.json" for _, t in calls
    )
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert raw["proposed_mcts_mode"] in ("off", "eval_only")


def test_mismatched_machine_id_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWBW_MCTS_HEALTH_MIN_GAMES", "1")
    rows = [
        {**_g(), "machine_id": "other"},
    ]
    _write_jsonl(tmp_path / "game_log.jsonl", rows)
    from tools.mcts_health import compute_health

    v = compute_health("pc-b", tmp_path, window=5)
    assert v.games_in_window == 0
    assert v.reasoning == "insufficient data"
