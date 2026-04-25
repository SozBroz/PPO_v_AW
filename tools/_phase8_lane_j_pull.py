"""Phase 8 Lane J — pull oracle_gap shapes (analysis utility, not production)."""
import json
import collections
import re
import sys

def main():
    rows = []
    path = "logs/desync_register_post_phase6.jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("class") == "oracle_gap":
                rows.append(r)
    print(f"oracle_gap rows: {len(rows)}")

    shapes = collections.Counter()

    def normalize(msg):
        if not msg:
            return "<empty>"
        s = msg
        s = re.sub(r"\b\d+\b", "N", s)
        s = re.sub(r"\([^)]*\)", "(...)", s)
        s = re.sub(r"'[^']*'", "'X'", s)
        s = s[:160]
        return s

    for r in rows:
        shapes[normalize(r.get("message", ""))] += 1

    out_path = "logs/phase8_lane_j_oracle_gap_shapes.json"
    with open(out_path, "w", encoding="utf-8") as out:
        json.dump({"total": len(rows), "shapes": shapes.most_common()}, out, indent=2)

    print("top 30 shapes:")
    for s, c in shapes.most_common(30):
        print(f"  {c:4d}  {s[:130]}")

    # Also emit full rows per shape for drill (smallest gid)
    by_shape = collections.defaultdict(list)
    for r in rows:
        by_shape[normalize(r.get("message", ""))].append(r)
    drill = {}
    for shape, rs in by_shape.items():
        gids = sorted(int(x["games_id"]) for x in rs if x.get("games_id") is not None)
        drill[shape] = {
            "count": len(rs),
            "min_gid": min(gids) if gids else None,
            "sample_gids": gids[:5],
        }
    with open("logs/phase8_lane_j_shape_drill.json", "w", encoding="utf-8") as f:
        json.dump(drill, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
