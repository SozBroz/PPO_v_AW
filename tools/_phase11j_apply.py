"""Apply / revert a list of (att, def, new) cell changes to data/damage_table.json.

Usage:
  python tools/_phase11j_apply.py apply  <changes.json>   # changes is a list of {att,def,new}
  python tools/_phase11j_apply.py revert <changes.json>   # reverts to {att,def,old}

Preserves the json.dumps(indent=2) layout the file already uses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_table(repo: Path):
    with open(repo / "data" / "damage_table.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_table(repo: Path, obj) -> None:
    with open(repo / "data" / "damage_table.json", "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")


def main() -> None:
    mode = sys.argv[1]
    changes_path = Path(sys.argv[2])
    repo = Path(__file__).resolve().parents[1]

    table = load_table(repo)
    name_to_idx = {u: i for i, u in enumerate(table["unit_order"])}
    rows = table["table"]

    with open(changes_path, "r", encoding="utf-8") as fh:
        changes = json.load(fh)

    applied = 0
    for ch in changes:
        ai = name_to_idx[ch["att"]]
        di = name_to_idx[ch["def"]]
        if mode == "apply":
            target = ch["new"]
        elif mode == "revert":
            target = ch["old"]
        else:
            raise SystemExit(f"unknown mode {mode}")
        if rows[ai][di] != target:
            rows[ai][di] = target
            applied += 1

    save_table(repo, table)
    print(f"{mode}: {applied}/{len(changes)} cells touched")


if __name__ == "__main__":
    main()
