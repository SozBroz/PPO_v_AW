"""Compare two desync registers and report class flips."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return {int(json.loads(l)["games_id"]): json.loads(l)
            for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def main():
    base = load(ROOT / "logs" / "desync_register_post_phase11j_fu_100.jsonl")
    new = load(ROOT / "logs" / "desync_register_post_phase11j_f2_fu_funds_100.jsonl")

    targets = {1621434, 1621898, 1622328, 1624082}

    flips_better = []  # gap -> ok
    flips_worse = []   # ok -> gap
    flips_other = []
    same = 0
    for gid in sorted(set(base) & set(new)):
        a = base[gid]["class"]
        b = new[gid]["class"]
        if a == b:
            same += 1
            continue
        delta = (a, b)
        if a == "ok" and b == "oracle_gap":
            flips_worse.append((gid, a, b))
        elif a == "oracle_gap" and b == "ok":
            flips_better.append((gid, a, b))
        else:
            flips_other.append((gid, a, b))

    print(f"Total compared: {len(base & new.keys() if False else (set(base) & set(new)))}")
    print(f"Same class:     {same}")
    print(f"\nGAINED (gap -> ok): {len(flips_better)}")
    for gid, a, b in flips_better:
        marker = " <TARGET>" if gid in targets else ""
        print(f"  {gid}: {a} -> {b}{marker}")
    print(f"\nLOST (ok -> gap):   {len(flips_worse)}")
    for gid, a, b in flips_worse:
        msg = new[gid].get("message", "")[:90]
        marker = " <TARGET>" if gid in targets else ""
        print(f"  {gid}: {a} -> {b}{marker} | {msg}")
    print(f"\nOTHER flips:        {len(flips_other)}")
    for gid, a, b in flips_other:
        print(f"  {gid}: {a} -> {b}")

    print("\nTarget GIDs status:")
    for gid in sorted(targets):
        if gid in new:
            print(f"  {gid}: base={base.get(gid, {}).get('class', 'MISSING')} -> new={new[gid]['class']}")
        else:
            print(f"  {gid}: NOT IN 100-GAME SAMPLE")


if __name__ == "__main__":
    main()
