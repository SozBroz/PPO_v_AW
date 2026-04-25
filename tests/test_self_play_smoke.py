"""CI smoke: small random self-play fuzzer run (Phase 4)."""

from __future__ import annotations

from pathlib import Path

from tools.self_play_fuzzer import run_fuzzer


def test_self_play_fuzzer_smoke():
    root = Path(__file__).resolve().parent.parent
    pool = root / "data" / "gl_map_pool.json"

    summary = run_fuzzer(
        games=10,
        seed=1,
        map_pool_path=pool,
        max_days=15,
        map_sample=5,
        maps_dir=root / "data" / "maps",
        pool_path=pool,
        out_path=None,
        quiet=True,
        # Keep CI smoke under ~30s wall time even if a map spins.
        game_timeout_sec=12.0,
    )

    soft: list[str] = []
    for gr in summary["results"]:
        for d in gr.defects:
            if d.type in ("game_timeout", "empty_legal_set"):
                soft.append(f"game {gr.game_index}: {d.type} {d.detail}")

    if soft:
        print("warnings (non-fatal):", *soft, sep="\n  ")

    by = summary["defects_by_type"]
    assert by.get("mask_step_disagree", 0) == 0, by
    assert by.get("invariant_violation", 0) == 0, by
    assert by.get("uncaught_exception", 0) == 0, by
    assert by.get("step_timeout", 0) == 0, by
