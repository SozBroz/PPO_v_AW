#!/usr/bin/env python3
"""
Export **your** Amarriner games to a local folder without using finished-game zip
(``replay_download.php`` does not serve in-progress games).

For each ``games_id`` from ``yourgames.php`` (default listing:
``yourgames.php?yourTurn=0``, same as in-browser “your games”):

- ``live_replay.json`` — full ``load_replay.php`` envelope stream (rebuildable offline
  with the same code path as ``desync_audit_amarriner_live``).
- ``engine_snapshot.pkl`` — when ``map_id`` + both COs are known (catalog or
  ``game.php`` scrape), same binary as ``rl.live_snapshot.write_live_snapshot``.

Credentials: repo ``secrets.txt`` (line 1 user, line 2 password).

This output is for the **Python engine** (live replay audit, training snapshots). It is
**not** the AWBW Replay Player ``.zip`` / PHP snapshot format; the desktop viewer does not
need to consume these files.

Example::

  python tools/amarriner_export_my_games_replays.py --out replays/amarinner_my_games
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.live_snapshot import write_live_snapshot  # noqa: E402
from tools.amarriner_list_your_games import (  # noqa: E402
    HEADERS,
    _login,
    list_your_games_ids,
)
from tools.amarriner_live_meta import (  # noqa: E402
    infer_meta_from_game_page_html,
    meta_from_first_snap_and_map_id,
    resolve_games_meta,
)
from tools.desync_audit_amarriner_live import (  # noqa: E402
    SECRETS,
    _fetch_live_envelopes,
    _http_get_text,
    build_live_engine_state_from_fetched,
)
from tools.amarriner_list_your_games import BASE_URL  # noqa: E402


def _envelopes_to_jsonable(
    envelopes: list[tuple[int, int, list[dict]]],
) -> list[list]:
    return [[pid, day, acts] for pid, day, acts in envelopes]


def _map_id_from_page(html: str) -> int | None:
    import re

    m = re.search(r"prevmaps\.php\?maps_id=(\d+)", html, re.I)
    if m:
        return int(m.group(1))
    m2 = re.search(r'"tiles_maps_id"\s*:\s*(\d+)', html)
    if m2:
        return int(m2.group(1))
    return None


def _resolve_meta_for_export(
    html: str,
    first_snap: dict,
    games_id: int,
) -> dict | None:
    m = resolve_games_meta(int(games_id), repo_root=ROOT)
    if m is not None and m.get("map_id") is not None and m.get("co_p0_id") is not None:
        return m
    m_html = infer_meta_from_game_page_html(html, int(games_id))
    if m_html is not None:
        return m_html
    mid = _map_id_from_page(html)
    if mid is not None:
        m_snap = meta_from_first_snap_and_map_id(first_snap, mid, int(games_id))
        if m_snap is not None:
            return m_snap
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "replays" / "amarinner_my_games",
        help="Output directory (created if missing)",
    )
    ap.add_argument(
        "--map-pool", type=Path, default=ROOT / "data" / "gl_map_pool.json"
    )
    ap.add_argument("--maps-dir", type=Path, default=ROOT / "data" / "maps")
    ap.add_argument("--sleep", type=float, default=0.35)
    ap.add_argument(
        "--games-id",
        type=int,
        action="append",
        default=None,
        help="Restrict to these ids (default: all from yourgames.php)",
    )
    ap.add_argument(
        "--your-turn",
        type=int,
        default=0,
        help="When discovering ids, pass yourgames.php?yourTurn= (default 0).",
    )
    ap.add_argument(
        "--no-your-turn-param",
        action="store_true",
        help="Discover ids from plain yourgames.php (no yourTurn= query).",
    )
    args = ap.parse_args()

    if not SECRETS.is_file():
        print(f"[export_my_games] missing {SECRETS}", file=sys.stderr)
        return 1
    if not args.map_pool.is_file():
        print(f"[export_my_games] missing {args.map_pool}", file=sys.stderr)
        return 1

    lines = [ln.strip() for ln in SECRETS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 2:
        print("[export_my_games] secrets: line1 user, line2 password", file=sys.stderr)
        return 1
    user, password = lines[0], lines[1]

    sess = requests.Session()
    if not _login(sess, user, password):
        print("[export_my_games] login failed", file=sys.stderr)
        return 1

    if args.games_id:
        gids = args.games_id
    else:
        yt = None if args.no_your_turn_param else int(args.your_turn)
        gids = list_your_games_ids(sess, your_turn=yt)
    if not gids:
        print("[export_my_games] no game ids (empty your games list / parse miss)")
        return 0

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for gid in gids:
        sub = out_dir / str(int(gid))
        sub.mkdir(parents=True, exist_ok=True)
        game_url = f"{BASE_URL}/game.php?games_id={int(gid)}"
        one: dict = {"games_id": int(gid), "ok": True, "engine_snapshot": None, "error": None}
        try:
            html = _http_get_text(sess, game_url)
            (Path(sub) / "game_page.html").write_text(html, encoding="utf-8")

            envelopes, first_snap, gs0, per_turn_units = _fetch_live_envelopes(
                sess, games_id=int(gid), sleep_s=float(args.sleep)
            )
            payload = {
                "games_id": int(gid),
                "source": "load_replay.php",
                "envelopes": _envelopes_to_jsonable(envelopes),
                "first_snap": first_snap,
                "game_state_turn0": gs0,
                "per_turn_units": per_turn_units,
            }
            (sub / "live_replay.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )

            meta = _resolve_meta_for_export(html, first_snap, int(gid))
            for stale in ("meta_unresolved.txt", "export_error.txt"):
                p = sub / stale
                if p.is_file():
                    p.unlink()
            if meta is not None:
                (sub / "meta.json").write_text(
                    json.dumps(meta, indent=2), encoding="utf-8"
                )
            else:
                (sub / "meta_unresolved.txt").write_text(
                    "Could not resolve map_id + both co ids (catalog + game.php HTML + "
                    "first_snap). live_replay.json is still enough to inspect the stream.\n",
                    encoding="utf-8",
                )

            if meta is not None:
                state, awbw = build_live_engine_state_from_fetched(
                    meta,
                    envelopes,
                    first_snap,
                    gs0,
                    per_turn_units,
                    map_pool=args.map_pool,
                    maps_dir=args.maps_dir,
                )
                pkl = sub / "engine_snapshot.pkl"
                write_live_snapshot(
                    pkl,
                    state,
                    games_id=int(gid),
                    learner_seat=0,
                    awbw_to_engine=awbw,
                )
                one["engine_snapshot"] = str(pkl)
        except Exception as exc:  # noqa: BLE001
            one["ok"] = False
            one["error"] = f"{type(exc).__name__}: {exc}"
            (sub / "export_error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        summary.append(one)

    (out_dir / "manifest.json").write_text(
        json.dumps({"games": summary}, indent=2), encoding="utf-8"
    )
    n_ok = sum(1 for s in summary if s.get("ok"))
    n_snap = sum(1 for s in summary if s.get("engine_snapshot"))
    print(
        f"[export_my_games] wrote {out_dir} — {len(summary)} game(s), "
        f"ok={n_ok}, engine_snapshot={n_snap}"
    )
    for s in summary:
        st = "ok" if s.get("ok") else f"ERR {s.get('error')}"
        ex = s.get("engine_snapshot") or "-"
        print(f"  games_id={s['games_id']}: {st}  pkl={ex}")
    return 0 if n_ok == len(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
