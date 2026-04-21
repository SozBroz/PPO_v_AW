"""Re-audit the 124 IF-class games (curated set frozen for this debugging cycle)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tools.desync_audit import (  # noqa: E402
    CATALOG_DEFAULT, MAP_POOL_DEFAULT, MAPS_DIR_DEFAULT, ZIPS_DEFAULT,
    _audit_one, _load_catalog,
)

# Frozen list of 124 game IDs that originally failed insufficient_funds /
# build-economy checks before the engine income & build-strict fixes landed.
GAMES = [
    1607045, 1609533, 1611213, 1613840, 1614281, 1614506, 1615143, 1616284,
    1617442, 1619117, 1619504, 1621898, 1621963, 1622443, 1622610, 1623057,
    1624082, 1624421, 1624515, 1625290, 1625633, 1625784, 1625906, 1627245,
    1627324, 1627523, 1627530, 1627552, 1628008, 1628163, 1628220, 1628322,
    1628953, 1629104, 1629512, 1630038, 1630151, 1630406, 1630712, 1631039,
    1631214, 1631333, 1631767, 1632124, 1632283, 1632289, 1632403, 1632458,
    1632504, 1633120, 1633303, 1633481, 1633562, 1633673, 1633907, 1634030,
    1634284, 1634366, 1634492, 1634571, 1634699, 1634757, 1634864, 1634889,
    1634893, 1634961, 1635001, 1635025, 1635146, 1635242, 1635383, 1635534,
    1635659, 1635708, 1635836, 1636011, 1636063, 1636157, 1636217, 1636275,
    1636384,
]


def main() -> int:
    print(f"# auditing {len(GAMES)} games (originally insufficient_funds)")
    catalog = _load_catalog(CATALOG_DEFAULT)
    games_meta = catalog.get("games") or {}
    by_id = {int(g["games_id"]): g for g in games_meta.values() if isinstance(g, dict)}

    rows = []
    counts: dict[str, int] = {}
    for i, gid in enumerate(GAMES, 1):
        meta = by_id.get(gid, {"games_id": gid, "map_id": -1})
        zpath = ZIPS_DEFAULT / f"{gid}.zip"
        try:
            row = _audit_one(
                games_id=gid,
                zip_path=zpath,
                meta=meta,
                map_pool=MAP_POOL_DEFAULT,
                maps_dir=MAPS_DIR_DEFAULT,
            )
        except Exception as e:
            print(f"  ! audit raised on {gid}: {e}")
            continue
        rows.append(row)
        counts.setdefault(row.cls, 0)
        counts[row.cls] += 1
        if i % 25 == 0:
            print(f"  ... {i}/{len(GAMES)} done")
    print()
    print("=== class counts ===")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<30} {v:>5}")
    print()
    print("=== ALL non-ok games ===")
    for r in rows:
        if r.cls == "ok":
            continue
        print(
            f"  {r.games_id}  {r.cls}\t{r.approx_action_kind or '?'}  "
            f"day~{r.approx_day}  P0={r.co_p0_id} P1={r.co_p1_id}  "
            f"{(r.message or '')[:140]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
