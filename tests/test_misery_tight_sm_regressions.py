"""
Tight-zip (``#frames == #envelopes``) replays: post-envelope HP resync in
``tools.desync_audit._run_replay_instrumented`` must mirror PHP or
``--enable-state-mismatch`` drifts (map 123858 Misery — gids 1632825, 1634267,
1635164; 2026-04-22).
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPLAYS = ROOT / "replays" / "amarriner_gl"
MAP_POOL = ROOT / "data" / "gl_map_pool.json"
MAPS_DIR = ROOT / "data" / "maps"
_CATALOGS = (
    ROOT / "data" / "amarriner_gl_std_catalog.json",
    ROOT / "data" / "amarriner_gl_extras_catalog.json",
)


def _merge_catalog_games() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in _CATALOGS:
        d = json.loads(p.read_text(encoding="utf-8"))
        g = d.get("games") or d
        for k, v in g.items():
            if isinstance(v, dict) and k not in out:
                out[k] = v
    return out


def _row(gid: int) -> dict:
    g = _merge_catalog_games()
    r = g.get(str(gid))
    if r is None:
        raise KeyError(gid)
    return r


_GIDS = (1632825, 1634267, 1635164)
_ZIPS = tuple(REPLAYS / f"{g}.zip" for g in _GIDS)
_HAVE = all(p.is_file() for p in _ZIPS)


@unittest.skipUnless(_HAVE, "requires Misery amarriner_gl zips for gids 1632825/7/4")
class TestMiseryTightZipStateMismatchClean(unittest.TestCase):
    def test_state_mismatch_audit_ok(self) -> None:
        from tools.desync_audit import CLS_OK, _audit_one

        for gid in _GIDS:
            meta = _row(gid)
            z = REPLAYS / f"{gid}.zip"
            with self.subTest(games_id=gid):
                r = _audit_one(
                    games_id=gid,
                    zip_path=z,
                    meta=meta,
                    map_pool=MAP_POOL,
                    maps_dir=MAPS_DIR,
                    seed=1,
                    enable_state_mismatch=True,
                )
                self.assertEqual(r.cls, CLS_OK, msg=r.message)


if __name__ == "__main__":
    unittest.main()
