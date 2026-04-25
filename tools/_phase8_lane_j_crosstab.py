"""Phase 8 Lane J — crosstab oracle_gap vs clusters and action kinds."""
import json
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    rows = []
    with open(ROOT / "logs/desync_register_post_phase6.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("class") == "oracle_gap":
                rows.append(r)

    by_kind = collections.Counter(r.get("approx_action_kind") or "?" for r in rows)
    print("approx_action_kind (oracle_gap):")
    for k, v in by_kind.most_common():
        print(f"  {v:4d}  {k}")

    clusters = json.loads((ROOT / "logs/desync_clusters_post_phase6.json").read_text(encoding="utf-8"))
    gap_ids = {int(r["games_id"]) for r in rows}
    shadow = []
    for name, gids in clusters.items():
        if name == "ok":
            continue
        gset = set(gids)
        overlap = gap_ids & gset
        if overlap:
            shadow.append((name, len(overlap), len(gset), sorted(overlap)[:15]))
    shadow.sort(key=lambda x: -x[1])
    out = {
        "oracle_gap_by_action_kind": dict(by_kind),
        "cluster_shadow": [
            {"cluster": n, "oracle_gap_in_cluster": oc, "cluster_size": cs, "sample_gids": sm}
            for n, oc, cs, sm in shadow[:40]
        ],
    }
    (ROOT / "logs/phase8_lane_j_cluster_crosstab.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("\nTop cluster shadows (oracle_gap rows in cluster):")
    for n, oc, cs, sm in shadow[:15]:
        print(f"  {oc:3d} / {cs:4d} in {n!r}  sample: {sm[:5]}")

    # engine_other cluster: how many of 162?
    eo = set(clusters.get("engine_other", []))
    print(f"\noracle_gap in engine_other: {len(gap_ids & eo)} / {len(gap_ids)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
