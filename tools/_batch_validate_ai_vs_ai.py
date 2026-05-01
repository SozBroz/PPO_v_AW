"""Run N ai_vs_ai games and oracle-replay each zip (viewer-equivalent action stream)."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rl.ai_vs_ai import run_game  # noqa: E402
from tools.oracle_zip_replay import replay_oracle_zip  # noqa: E402

_POOL = _REPO / "data" / "gl_map_pool.json"
_MAPS = _REPO / "data" / "maps"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=10_000)
    ap.add_argument("--game-id-base", type=int, default=880_000)
    ap.add_argument("--max-days", "--max-turns", dest="max_days", type=int, default=40)
    ap.add_argument("--tier", type=str, default="T2")
    args = ap.parse_args()

    out = Path(tempfile.mkdtemp(prefix="ai_vs_ai_val_"))
    failures: list[tuple[int, BaseException]] = []
    for i in range(args.runs):
        gid = args.game_id_base + i
        seed = args.seed_base + i
        try:
            zp = run_game(
                map_id=None,
                ckpt_path=None,
                co0=None,
                co1=None,
                tier=args.tier,
                seed=seed,
                max_days=args.max_days,
                force_random=True,
                open_viewer=False,
                output_dir=out,
                game_id=gid,
                capture_move_gate=False,
            )
            tr_path = zp.with_suffix(".trace.json")
            tr = json.loads(tr_path.read_text(encoding="utf-8"))
            _ls = tr.get("luck_seed")
            r = replay_oracle_zip(
                zp,
                map_pool=_POOL,
                maps_dir=_MAPS,
                map_id=int(tr["map_id"]),
                co0=int(tr["co0"]),
                co1=int(tr["co1"]),
                tier_name=str(tr.get("tier") or "T2"),
                luck_seed=int(_ls) if _ls is not None else None,
            )
            print(
                f"OK seed={seed} gid={gid} map={tr['map_id']} "
                f"trace_actions={tr['n_actions_full_trace']} oracle_actions={r.actions_applied}"
            )
        except BaseException as e:
            print(f"FAIL seed={seed} gid={gid}: {e!r}")
            failures.append((i, e))

    print(f"work dir: {out}")
    if failures:
        raise SystemExit(f"{len(failures)} run(s) failed")


if __name__ == "__main__":
    main()
