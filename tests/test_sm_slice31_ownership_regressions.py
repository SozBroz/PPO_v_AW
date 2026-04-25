"""
Phase 11 SM ownership slice: games flagged for follow-up on ``state_mismatch_units``
(2026-04-22 register retune) must stay ``ok`` with ``--enable-state-mismatch``.

Uses the same per-envelope PHP treasury + post-frame HP hygiene as the full
936 SM audit. Skips if any required zip is absent (e.g. minimal checkout).
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

# Imperator's 31-gid state_mismatch_units ownership batch (incl. 1630341 Sonja).
_GIDS = (
    1615143,
    1617442,
    1620585,
    1621434,
    1622104,
    1624082,
    1624721,
    1624764,
    1625804,
    1625905,
    1627004,
    1627066,
    1627323,
    1627403,
    1627495,
    1627563,
    1627696,
    1627885,
    1628220,
    1628276,
    1628357,
    1628446,
    1628722,
    1628824,
    1629120,
    1629157,
    1629383,
    1629539,
    1629757,
    1630082,
    1630341,
)

_ZIPS = tuple(REPLAYS / f"{g}.zip" for g in _GIDS)
_HAVE = all(p.is_file() for p in _ZIPS)


def _merge_last_wins() -> dict[str, dict]:
    by_gid: dict[str, dict] = {}
    for p in _CATALOGS:
        d = json.loads(p.read_text(encoding="utf-8"))
        for k, g in (d.get("games") or {}).items():
            if isinstance(g, dict) and "games_id" in g:
                by_gid[k] = g
    return by_gid


@unittest.skipUnless(_HAVE, "requires all 31 amarriner_gl zips for SM slice-31 regression")
class TestStateMismatchSlice31OwnershipOk(unittest.TestCase):
    def test_state_mismatch_audit_all_ok(self) -> None:
        from tools.desync_audit import CLS_OK, _audit_one

        g = _merge_last_wins()
        for games_id in _GIDS:
            meta = g.get(str(games_id))
            self.assertIsNotNone(
                meta, f"missing catalog row for {games_id}"
            )
        for games_id in _GIDS:
            z = REPLAYS / f"{games_id}.zip"
            with self.subTest(games_id=games_id):
                r = _audit_one(
                    games_id=games_id,
                    zip_path=z,
                    meta=g[str(games_id)],
                    map_pool=MAP_POOL,
                    maps_dir=MAPS_DIR,
                    seed=1,
                    enable_state_mismatch=True,
                )
                self.assertEqual(r.cls, CLS_OK, msg=f"{r.message!r}")


if __name__ == "__main__":
    unittest.main()
