"""Phase 11J-DAMAGE-CANON diff builder.

Compares engine data/damage_table.json to the AWBW canonical chart at
https://awbw.amarriner.com/damage.php (parsed/cached inline below) and
emits three lists:
  - wave1_changes:        both sides numeric, values differ
  - wave2_changes:        engine null -> PHP numeric (cautious; may unlock targeting)
  - engine_keeps_non_null: engine numeric, PHP null  (kept; document in _notes)

One-shot alignment (rip the band-aid):

  python tools/_phase11j_canon_diff.py --apply-full

writes every PHP-covered cell (625) from the embedded PHP_RAW snapshot into
``data/damage_table.json`` — including nulling cells where PHP has ``-``.

Engine units NOT in PHP (Gunboat=22, Oozium=26) are excluded.
Piperunner (=25) cells are reported separately so we can be careful.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# PHP order (alphabetical) -> engine unit name
PHP_TO_ENGINE = {
    "AntiAir": "AntiAir",
    "APC": "APC",
    "Artillery": "Artillery",
    "BCopter": "BCopter",
    "Battleship": "Battleship",
    "BlackBoat": "BlackBoat",
    "BlackBomb": "BlackBomb",
    "Bomber": "Bomber",
    "Carrier": "Carrier",
    "Cruiser": "Cruiser",
    "Fighter": "Fighter",
    "Infantry": "Infantry",
    "Lander": "Lander",
    "MdTank": "MedTank",
    "Mech": "Mech",
    "MegaTank": "MegaTank",
    "Missile": "Missiles",
    "NeoTank": "NeoTank",
    "Piperunner": "Piperunner",
    "Recon": "Recon",
    "Rocket": "Rocket",
    "Stealth": "Stealth",
    "Sub": "Submarine",
    "TCopter": "TCopter",
    "Tank": "Tank",
}

PHP_ORDER = [
    "AntiAir", "APC", "Artillery", "BCopter", "Battleship", "BlackBoat",
    "BlackBomb", "Bomber", "Carrier", "Cruiser", "Fighter", "Infantry",
    "Lander", "MdTank", "Mech", "MegaTank", "Missile", "NeoTank",
    "Piperunner", "Recon", "Rocket", "Stealth", "Sub", "TCopter", "Tank",
]

# PHP table fetched 2026-04-21 from https://awbw.amarriner.com/damage.php
# rows = attackers (alphabetical), cols = defenders (alphabetical), '-' = null
PHP_RAW = """\
45 50 50 120 - - 120 75 - - 65 105 - 10 105 1 55 5 25 60 55 75 - 120 25
- - - - - - - - - - - - - - - - - - - - - - - - -
75 70 75 - 40 55 - - 45 65 - 90 55 45 85 15 80 40 70 80 80 - 60 - 70
25 60 65 65 25 25 - - 25 55 - 75 25 25 75 10 65 20 55 55 65 - 25 95 55
85 80 80 - 50 95 - - 60 95 - 95 95 55 90 25 90 50 80 90 85 - 95 - 80
- - - - - - - - - - - - - - - - - - - - - - - - -
- - - - - - - - - - - - - - - - - - - - - - - - -
95 105 105 - 75 95 - - 75 85 - 110 95 95 110 35 105 90 105 105 105 - 95 - 105
- - - 115 - - 120 100 - - 100 - - - - - - - - - - 100 - 115 -
- - - 115 - 25 120 65 5 - 55 - - - - - - - - - - 100 90 115 -
- - - 100 - - 120 100 - - 55 - - - - - - - - - - 85 - 100 -
5 14 15 7 - - - - - - - 55 - 1 45 1 25 1 5 12 25 - - 30 5
- - - - - - - - - - - - - - - - - - - - - - - - -
105 105 105 12 10 35 - - 10 45 - 105 35 55 95 25 105 45 85 105 105 - 10 45 85
65 75 70 9 - - - - - - - 65 - 15 55 5 85 15 55 85 85 - - 35 55
195 195 195 22 45 105 - - 45 65 - 135 75 125 125 65 195 115 180 195 195 - 45 55 180
- - - 120 - - 120 100 - - 100 - - - - - - - - - - 100 - 120 -
115 125 115 22 15 40 - - 15 50 - 125 40 75 115 35 125 55 105 125 125 - 15 55 105
85 80 80 105 55 60 120 75 60 60 65 95 60 55 90 25 90 50 80 90 85 75 85 105 80
4 45 45 10 - - - - - - - 70 - 1 65 1 28 1 6 35 55 - - 35 6
85 80 80 - 55 60 - - 60 85 - 95 60 55 90 25 90 50 80 90 85 - 85 - 80
50 85 75 85 45 65 120 70 45 35 45 90 65 70 90 15 85 60 80 85 85 55 55 95 75
- - - - 55 95 - - 75 25 - - 95 - - - - - - - - - 55 - -
- - - - - - - - - - - - - - - - - - - - - - - - -
65 75 70 10 1 10 - - 1 5 - 75 10 15 70 10 85 15 55 85 85 - 1 40 55
"""


def parse_php() -> dict[tuple[str, str], int | None]:
    out: dict[tuple[str, str], int | None] = {}
    rows = [r.strip() for r in PHP_RAW.strip().splitlines() if r.strip()]
    assert len(rows) == 25, f"expected 25 PHP rows, got {len(rows)}"
    for ri, row in enumerate(rows):
        cells = row.split()
        assert len(cells) == 25, f"row {ri} has {len(cells)} cells"
        att = PHP_ORDER[ri]
        for ci, cell in enumerate(cells):
            dfn = PHP_ORDER[ci]
            v: int | None = None if cell == "-" else int(cell)
            out[(att, dfn)] = v
    return out


def apply_php_overlap_full(repo: Path | None = None) -> dict[str, int]:
    """Set every cell in the PHP 25×25 matrix to the embedded snapshot (incl. null)."""
    repo = repo or Path(__file__).resolve().parents[1]
    table_path = repo / "data" / "damage_table.json"
    with open(table_path, "r", encoding="utf-8") as fh:
        engine = json.load(fh)
    php = parse_php()
    name_to_idx = {u: i for i, u in enumerate(engine["unit_order"])}
    rows = engine["table"]
    changed = 0
    for (att_php, dfn_php), php_v in php.items():
        att = PHP_TO_ENGINE[att_php]
        dfn = PHP_TO_ENGINE[dfn_php]
        ai, di = name_to_idx[att], name_to_idx[dfn]
        if rows[ai][di] != php_v:
            rows[ai][di] = php_v
            changed += 1
    notes = engine.setdefault("_notes", [])
    notes.insert(
        0,
        "2026-04-21 (Phase 11J-DAMAGE-TABLE-CANON --apply-full): One-shot sync of all "
        "625 AWBW PHP matrix cells from https://awbw.amarriner.com/damage.php into "
        "data/damage_table.json (PHP snapshot: PHP_RAW in this module). "
        f"{changed} cell(s) changed (numeric fills + PHP '-' → null). "
        "Matchups involving only Gunboat/Oozium outside the PHP grid are unchanged.",
    )
    with open(table_path, "w", encoding="utf-8") as fh:
        json.dump(engine, fh, indent=2)
        fh.write("\n")
    return {"cells_changed": changed}


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--apply-full":
        result = apply_php_overlap_full()
        print(f"apply-full: {result}")
        return

    repo = Path(__file__).resolve().parents[1]
    table_path = repo / "data" / "damage_table.json"
    with open(table_path, "r", encoding="utf-8") as fh:
        engine = json.load(fh)
    unit_order = engine["unit_order"]
    table = engine["table"]
    name_to_idx = {u: i for i, u in enumerate(unit_order)}

    php = parse_php()

    wave1: list[dict] = []
    wave2: list[dict] = []
    engine_keeps: list[dict] = []
    php_keeps: list[dict] = []  # both null - no-op
    matches = 0

    for (att_php, dfn_php), php_v in php.items():
        att = PHP_TO_ENGINE[att_php]
        dfn = PHP_TO_ENGINE[dfn_php]
        ai = name_to_idx[att]
        di = name_to_idx[dfn]
        eng_v = table[ai][di]
        if eng_v == php_v:
            matches += 1
            continue
        rec = {"att": att, "def": dfn, "old": eng_v, "new": php_v,
               "row": ai, "col": di,
               "is_piperunner": (att == "Piperunner" or dfn == "Piperunner")}
        if eng_v is not None and php_v is not None:
            wave1.append(rec)
        elif eng_v is None and php_v is not None:
            wave2.append(rec)
        elif eng_v is not None and php_v is None:
            engine_keeps.append(rec)
        else:
            php_keeps.append(rec)

    summary = {
        "total_php_cells": len(php),
        "matches": matches,
        "wave1_count": len(wave1),
        "wave2_count": len(wave2),
        "engine_keeps_non_null_count": len(engine_keeps),
        "wave1_changes": wave1,
        "wave2_changes": wave2,
        "engine_keeps_non_null": engine_keeps,
    }

    out_path = repo / "logs" / "_phase11j_canon_diff.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"PHP cells:        {len(php)}")
    print(f"matches:          {matches}")
    print(f"wave1 (num/num):  {len(wave1)}")
    print(f"wave2 (null/num): {len(wave2)}")
    print(f"engine_keeps:     {len(engine_keeps)}")
    print(f"-> {out_path.relative_to(repo)}")


if __name__ == "__main__":
    main()
