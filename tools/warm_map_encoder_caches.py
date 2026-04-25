"""
Pre-fill ``MapData`` terrain one-hot + defense-star encoder caches for every map
in ``data/gl_map_pool.json``.

Caches are normally built lazily on first ``encode_state``; warming after ``load_map``
avoids the first-hit Python loops when training starts.

Usage (repo root)::

    python tools/warm_map_encoder_caches.py

Exit code 0 on success; prints maps warmed / skipped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.map_loader import load_map  # noqa: E402
from rl.encoder import warm_map_static_encoder_cache  # noqa: E402


def main() -> int:
    pool_path = ROOT / "data" / "gl_map_pool.json"
    maps_dir = ROOT / "data" / "maps"
    if not pool_path.is_file():
        print(f"Missing pool: {pool_path}", file=sys.stderr)
        return 1
    with open(pool_path, encoding="utf-8") as f:
        pool: list[dict] = json.load(f)
    ok = 0
    skipped = 0
    for meta in pool:
        mid = int(meta["map_id"])
        try:
            md = load_map(mid, pool_path, maps_dir)
        except (OSError, ValueError, FileNotFoundError) as e:
            print(f"skip map_id={mid}: {e}")
            skipped += 1
            continue
        warm_map_static_encoder_cache(md)
        assert getattr(md, "_encoded_defense_stars", None) is not None
        ok += 1
    print(f"warmed {ok} maps ({skipped} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
