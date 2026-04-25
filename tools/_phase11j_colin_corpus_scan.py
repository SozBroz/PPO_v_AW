"""Phase 11J-COLIN-IMPL-SHIP — corpus scan helper.

Counts Colin-related rows in the canonical 936-zip desync register and
reports the size of the (non-std) Colin batch catalog.
"""
import json
from pathlib import Path

REG = Path("logs/desync_register_post_phase11j_v2_936.jsonl")
BATCH = Path("data/amarriner_gl_colin_batch.json")


def main() -> None:
    hits = 0
    total = 0
    if REG.exists():
        with REG.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                row = json.loads(line)
                blob = json.dumps(row).lower()
                if "colin" in blob:
                    hits += 1
                    print(
                        "  HIT: gid=", row.get("games_id"),
                        "class=", row.get("class"),
                    )
    print(f"Register {REG}: rows={total} colin_matches={hits}")

    if BATCH.exists():
        cb = json.loads(BATCH.read_text(encoding="utf-8"))
        games = cb.get("games") if isinstance(cb, dict) else cb
        if isinstance(games, list):
            print(f"Colin batch catalog: {len(games)} zips (non-std maps)")
            ids = [g.get("games_id") for g in games][:15]
            print(f"  First {len(ids)} games_ids: {ids}")


if __name__ == "__main__":
    main()
