#!/usr/bin/env python3
"""
Build engine state for an in-progress game via live ``load_replay`` and write
``{games_id}.pkl`` for training (see ``rl.live_snapshot`` / ``--live-games-id``).

Uses ``secrets.txt`` (line 1 user, line 2 password).  Resolves map/CO
metadata with :func:`tools.amarriner_live_meta.resolve_games_meta` if you do
not pass a catalog JSON with ``--meta-json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.amarriner_live_meta import resolve_games_meta  # noqa: E402
from tools.desync_audit_amarriner_live import SECRETS, _login, build_live_engine_state  # noqa: E402
from rl.live_snapshot import write_live_snapshot  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("games_id", type=int, help="Amarriner games_id")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .pkl path (default: .tmp/awbw_live_snapshot/<id>.pkl)",
    )
    ap.add_argument(
        "--learner-seat",
        type=int,
        default=0,
        choices=(0, 1),
        help="Engine seat you control (0 or 1); stored in the pickle",
    )
    ap.add_argument(
        "--meta-json",
        type=Path,
        default=None,
        help="JSON file with a single catalog row, or {games: {id: row}} (optional)",
    )
    ap.add_argument(
        "--map-pool", type=Path, default=ROOT / "data/gl_map_pool.json"
    )
    ap.add_argument("--maps-dir", type=Path, default=ROOT / "data/maps")
    ap.add_argument("--sleep", type=float, default=0.35)
    args = ap.parse_args()

    if not SECRETS.is_file():
        print(f"[live_snap] missing {SECRETS}", file=sys.stderr)
        return 1
    if not args.map_pool.is_file():
        print(f"[live_snap] missing --map-pool {args.map_pool}", file=sys.stderr)
        return 1

    meta: dict
    if args.meta_json and args.meta_json.is_file():
        raw = json.loads(args.meta_json.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "games" in raw and isinstance(raw["games"], dict):
            g = raw["games"].get(str(args.games_id))
            if g is None:
                print(
                    f"[live_snap] games_id {args.games_id} not in {args.meta_json}",
                    file=sys.stderr,
                )
                return 1
            meta = g
        else:
            meta = raw
        meta.setdefault("games_id", int(args.games_id))
    else:
        got = resolve_games_meta(int(args.games_id), repo_root=ROOT)
        if got is None:
            print(
                f"[live_snap] no catalog row for games_id={args.games_id}; "
                "use --meta-json or refresh data/*.json",
                file=sys.stderr,
            )
            return 1
        meta = got

    lines = [ln.strip() for ln in SECRETS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 2:
        print("[live_snap] secrets.txt: line1 user, line2 password", file=sys.stderr)
        return 1
    sess = requests.Session()
    if not _login(sess, lines[0], lines[1]):
        print("[live_snap] login failed", file=sys.stderr)
        return 1

    state, awbw = build_live_engine_state(
        sess,
        meta,
        map_pool=args.map_pool,
        maps_dir=args.maps_dir,
        sleep_s=float(args.sleep),
    )
    out = args.out
    if out is None:
        out = ROOT / ".tmp" / "awbw_live_snapshot" / f"{int(args.games_id)}.pkl"
    out = Path(out)
    ap_seat = int(state.active_player)
    ls = int(args.learner_seat) & 1
    if ap_seat != ls:
        print(
            f"[live_snap] warning: engine active_player={ap_seat} but --learner-seat={ls}. "
            "Train with matching --live-learner-seats or refresh when it is your turn.",
            file=sys.stderr,
        )

    write_live_snapshot(
        out,
        state,
        games_id=int(args.games_id),
        learner_seat=ls,
        awbw_to_engine=awbw,
    )
    print(f"[live_snap] wrote {out} (active_player={ap_seat}, learner_seat={ls})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
