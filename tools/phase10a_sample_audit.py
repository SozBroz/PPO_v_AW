"""Phase 10A sample audit: re-run the 47 B_COPTER engine_bug target gids
through ``tools.desync_audit._audit_one`` after applying the damage-table
fixes (data/damage_table.json) and the MG-secondary ammo fix
(engine/game.py::_apply_attack).

Compares the new verdict (status / cls / exception_type) against the prior
``engine_bug`` verdict and writes a per-gid result table plus a summary.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tools.desync_audit import (
    CATALOG_DEFAULT,
    MAP_POOL_DEFAULT,
    MAPS_DIR_DEFAULT,
    ZIPS_DEFAULT,
    _audit_one,
)

TARGETS = Path("logs/phase10a_b_copter_targets.jsonl")
OTHER_TARGETS = Path("logs/phase10a_other_unit_targets.jsonl")
OUT_DIR = Path("logs")
OUT_DIR.mkdir(exist_ok=True)


def _load_catalog() -> dict[int, dict]:
    data = json.loads(Path(CATALOG_DEFAULT).read_text(encoding="utf-8"))
    games = data["games"] if isinstance(data, dict) and "games" in data else data
    if isinstance(games, dict):
        out = {}
        for k, v in games.items():
            row = dict(v)
            row.setdefault("games_id", int(k))
            out[int(k)] = row
        return out
    return {int(r["games_id"]): r for r in games}


def _audit(targets_path: Path, label: str) -> dict:
    rows = [
        json.loads(line)
        for line in targets_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    catalog = _load_catalog()
    results = []
    cls_before = Counter()
    cls_after = Counter()
    status_after = Counter()
    flips_to_ok = 0
    still_bug = 0
    new_oracle_gap = 0

    for r in rows:
        gid = int(r["games_id"])
        meta = catalog.get(gid)
        if meta is None:
            continue
        zip_path = ZIPS_DEFAULT / f"{gid}.zip"
        if not zip_path.exists():
            results.append({
                "games_id": gid, "before_cls": r.get("cls"),
                "after_cls": "missing_zip", "after_status": "missing_zip",
                "exc": "", "msg": "",
            })
            continue
        try:
            audited = _audit_one(
                games_id=gid,
                zip_path=zip_path,
                meta=meta,
                map_pool=MAP_POOL_DEFAULT,
                maps_dir=MAPS_DIR_DEFAULT,
                seed=1,
            )
        except Exception as exc:
            results.append({
                "games_id": gid, "before_cls": r.get("cls"),
                "after_cls": "audit_crash",
                "after_status": "audit_crash",
                "exc": type(exc).__name__,
                "msg": str(exc)[:200],
            })
            continue

        before = r.get("cls", "")
        after = audited.cls
        cls_before[before] += 1
        cls_after[after] += 1
        status_after[audited.status] += 1
        if after == "ok":
            flips_to_ok += 1
        elif after == "engine_bug":
            still_bug += 1
        elif after == "oracle_gap":
            new_oracle_gap += 1

        results.append({
            "games_id": gid,
            "before_cls": before,
            "after_cls": after,
            "after_status": audited.status,
            "exc": audited.exception_type,
            "msg": (audited.message or "")[:200],
        })

    summary = {
        "label": label,
        "total": len(results),
        "flips_to_ok": flips_to_ok,
        "still_engine_bug": still_bug,
        "flips_to_oracle_gap": new_oracle_gap,
        "cls_before": dict(cls_before),
        "cls_after": dict(cls_after),
        "status_after": dict(status_after),
    }
    return {"summary": summary, "results": results}


def main() -> None:
    out = {}
    for label, path in (("b_copter", TARGETS), ("other_units", OTHER_TARGETS)):
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        print(f"=== Auditing {label} ({path}) ===")
        out[label] = _audit(path, label)
        s = out[label]["summary"]
        print(f"  total={s['total']}  ok={s['flips_to_ok']}  "
              f"still_bug={s['still_engine_bug']}  "
              f"oracle_gap={s['flips_to_oracle_gap']}")
        print(f"  after_cls: {s['cls_after']}")

    out_path = OUT_DIR / "phase10a_sample_audit_results.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for label, payload in out.items():
            fh.write(json.dumps({"summary": payload["summary"]}) + "\n")
            for row in payload["results"]:
                fh.write(json.dumps({"label": label, **row}) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
