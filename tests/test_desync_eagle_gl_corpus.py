"""
Global League replays where **Eagle** (``co_id`` 10) is P0 and/or P1: every
local ``replays/amarriner_gl/<games_id>.zip`` must pass ``tools.desync_audit``
**canonical** and **state-mismatch** lanes.

Rationale: Eagle D2D / COP / SCOP (especially Lightning Strike) touch combat
modifiers, ``moved`` refresh, and build/unload carve-outs. Oracle replay uses
``oracle_mode=True`` and does **not** exercise ``get_legal_actions`` — see
``test_co_eagle_lightning_strike`` for RL legality. This module catches
**zip-level** regressions (``oracle_gap``, ``state_mismatch_*``) on the full
Eagle cohort so they cannot slip through only on non-Eagle games.

Run: ``pytest tests/test_desync_eagle_gl_corpus.py`` (adds ~2–4 min vs unit-only).
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

EAGLE_CO_ID = 10
# Match CLI default ``--state-mismatch-hp-tolerance 10`` (Phase 11J retune).
_DEFAULT_SM_HP_TOL = 10
_SEED = 1


def _merge_catalog_games() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in _CATALOGS:
        d = json.loads(p.read_text(encoding="utf-8"))
        g = d.get("games") or {}
        for k, v in g.items():
            if isinstance(v, dict) and k not in out:
                out[k] = v
    return out


def _eagle_games_with_zip() -> list[tuple[int, dict]]:
    g = _merge_catalog_games()
    rows: list[tuple[int, dict]] = []
    for _k, meta in sorted(g.items(), key=lambda kv: int(kv[0])):
        if not isinstance(meta, dict):
            continue
        if meta.get("co_p0_id") != EAGLE_CO_ID and meta.get("co_p1_id") != EAGLE_CO_ID:
            continue
        gid = int(meta["games_id"])
        z = REPLAYS / f"{gid}.zip"
        if not z.is_file():
            continue
        rows.append((gid, meta))
    return rows


_EAGLE_ROWS = _eagle_games_with_zip()
_HAVE_ANY = len(_EAGLE_ROWS) > 0


@unittest.skipUnless(_HAVE_ANY, "requires at least one Eagle GL zip under replays/amarriner_gl")
class TestEagleGlCorpusDesyncAudit(unittest.TestCase):
    """Merged std+extras catalog ∩ local zips where either seat is Eagle."""

    def test_canonical_lane_ok(self) -> None:
        from tools.desync_audit import CLS_OK, _audit_one

        for gid, meta in _EAGLE_ROWS:
            z = REPLAYS / f"{gid}.zip"
            with self.subTest(games_id=gid, lane="canonical"):
                r = _audit_one(
                    games_id=gid,
                    zip_path=z,
                    meta=meta,
                    map_pool=MAP_POOL,
                    maps_dir=MAPS_DIR,
                    seed=_SEED,
                    enable_state_mismatch=False,
                )
                self.assertEqual(r.cls, CLS_OK, msg=r.message)

    def test_state_mismatch_lane_ok(self) -> None:
        from tools.desync_audit import CLS_OK, _audit_one

        for gid, meta in _EAGLE_ROWS:
            z = REPLAYS / f"{gid}.zip"
            with self.subTest(games_id=gid, lane="state_mismatch"):
                r = _audit_one(
                    games_id=gid,
                    zip_path=z,
                    meta=meta,
                    map_pool=MAP_POOL,
                    maps_dir=MAPS_DIR,
                    seed=_SEED,
                    enable_state_mismatch=True,
                    state_mismatch_hp_tolerance=_DEFAULT_SM_HP_TOL,
                )
                self.assertEqual(r.cls, CLS_OK, msg=r.message)


if __name__ == "__main__":
    unittest.main()
