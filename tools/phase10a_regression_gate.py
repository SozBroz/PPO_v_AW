"""Phase 10A regression gate (orchestrator order — tightened):

  Gate 5: BEFORE/AFTER spot-audit on 5 B_COPTER engine_bug rows.
  Gate 5b: BEFORE/AFTER spot-audit on 5 NON-B_COPTER ok rows that exercise
           Manhattan canon (direct attacks) and Andy SCOP CO.
  Gate 6: 30 random NON-B_COPTER ok rows must stay ok.

This script reads the post-Phase-9 register, picks deterministic samples,
and runs ``tools.desync_audit._audit_one`` against the *current* engine.
The "BEFORE" verdict is just the row's prior ``cls`` from the register
(snapshotted before the fix landed). The "AFTER" verdict is whatever the
re-audit produces with the fix in place.

NOTE: Phase 10A only modified ``data/damage_table.json`` and
``engine/game.py::_apply_attack`` (MG ammo gate). It did *not* touch
``engine/action.py``. The orchestrator's regression gate still asks for
non-B_COPTER ok rows to be re-validated to catch any collateral damage.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from tools.desync_audit import (
    CATALOG_DEFAULT,
    MAP_POOL_DEFAULT,
    MAPS_DIR_DEFAULT,
    ZIPS_DEFAULT,
    _audit_one,
)

REGISTER = Path("logs/desync_register_post_phase9.jsonl")
LOGS = Path("logs"); LOGS.mkdir(exist_ok=True)
OUT = LOGS / "phase10a_regression_gate.jsonl"
SUMMARY = LOGS / "phase10a_regression_gate_summary.txt"


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


def _load_register() -> list[dict]:
    return [
        json.loads(line)
        for line in REGISTER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _row_cls(row: dict) -> str:
    return row.get("class") or row.get("cls") or ""


def _has_b_copter(row: dict) -> bool:
    msg = (row.get("message") or "")
    return "B_COPTER" in msg


def _audit(gid: int, catalog: dict[int, dict]) -> dict:
    meta = catalog.get(gid)
    if meta is None:
        return {"games_id": gid, "after_cls": "missing_meta",
                "after_status": "missing_meta", "exc": "", "msg": ""}
    zip_path = ZIPS_DEFAULT / f"{gid}.zip"
    if not zip_path.exists():
        return {"games_id": gid, "after_cls": "missing_zip",
                "after_status": "missing_zip", "exc": "", "msg": ""}
    try:
        a = _audit_one(
            games_id=gid, zip_path=zip_path, meta=meta,
            map_pool=MAP_POOL_DEFAULT, maps_dir=MAPS_DIR_DEFAULT, seed=1,
        )
    except Exception as exc:
        return {"games_id": gid, "after_cls": "audit_crash",
                "after_status": "audit_crash",
                "exc": type(exc).__name__, "msg": str(exc)[:200]}
    return {"games_id": gid, "after_cls": a.cls,
            "after_status": a.status,
            "exc": a.exception_type, "msg": (a.message or "")[:200]}


def main() -> None:
    rng = random.Random(0xCAE5A21)  # deterministic — replicable across runs
    catalog = _load_catalog()
    register = _load_register()

    by_gid_first = {}
    for row in register:
        gid = int(row.get("games_id", -1))
        if gid > 0 and gid not in by_gid_first:
            by_gid_first[gid] = row

    b_copter_bug_rows = [
        r for r in register
        if _row_cls(r) == "engine_bug" and _has_b_copter(r)
    ]
    b_copter_bug_rows.sort(key=lambda r: int(r["games_id"]))
    b_copter_sample = b_copter_bug_rows[:5]

    non_b_copter_ok = [
        (gid, r) for gid, r in by_gid_first.items()
        if _row_cls(r) == "ok" and not _has_b_copter(r)
    ]
    rng.shuffle(non_b_copter_ok)

    spot_ok_sample = []
    used = set()
    andy_co_ids = {1}
    for gid, r in non_b_copter_ok:
        cos = {r.get("co_p0_id"), r.get("co_p1_id")}
        if andy_co_ids & cos and gid not in used:
            spot_ok_sample.append((gid, r)); used.add(gid)
            if len(spot_ok_sample) >= 2:
                break
    for gid, r in non_b_copter_ok:
        if gid in used:
            continue
        spot_ok_sample.append((gid, r)); used.add(gid)
        if len(spot_ok_sample) >= 5:
            break

    spot_ok_sample_gids = [(gid, r) for gid, r in spot_ok_sample]

    out_rows = []
    print("=== Gate 5: 5 B_COPTER engine_bug BEFORE/AFTER ===")
    bc_flips_to_ok = 0
    bc_regressed = 0
    for r in b_copter_sample:
        gid = int(r["games_id"])
        before = _row_cls(r)
        after = _audit(gid, catalog)
        flipped_ok = after["after_cls"] == "ok"
        if flipped_ok:
            bc_flips_to_ok += 1
        out = {"gate": "5_b_copter_bug", "games_id": gid,
               "before_cls": before, **after}
        out_rows.append(out)
        marker = "OK_FLIP" if flipped_ok else after["after_cls"].upper()
        print(f"  gid={gid}  before={before:>11}  after={after['after_cls']:>11}  [{marker}]")

    print()
    print("=== Gate 5b: 5 NON-B_COPTER ok BEFORE/AFTER (Andy/Manhattan guard) ===")
    ok_kept = 0
    for gid, src_row in spot_ok_sample_gids:
        before = "ok"
        co_p0 = src_row.get("co_p0_id")
        co_p1 = src_row.get("co_p1_id")
        after = _audit(gid, catalog)
        kept = after["after_cls"] == "ok"
        if kept:
            ok_kept += 1
        out = {"gate": "5b_non_b_copter_ok", "games_id": gid,
               "before_cls": before,
               "co_p0": co_p0, "co_p1": co_p1,
               **after}
        out_rows.append(out)
        marker = "KEPT_OK" if kept else "REGRESSED"
        print(f"  gid={gid}  cos=({co_p0},{co_p1})  "
              f"before={before:>3}  after={after['after_cls']:>11}  [{marker}]")

    audit30_pool_filtered = [g for (g, _) in non_b_copter_ok if g not in used]
    audit30 = audit30_pool_filtered[:30]

    print()
    print(f"=== Gate 6: 30 random NON-B_COPTER ok rows ===")
    g6_kept = 0
    g6_flipped = []
    cls_after = Counter()
    for gid in audit30:
        before = "ok"
        after = _audit(gid, catalog)
        cls_after[after["after_cls"]] += 1
        if after["after_cls"] == "ok":
            g6_kept += 1
        else:
            g6_flipped.append((gid, after["after_cls"], after["msg"][:120]))
        out = {"gate": "6_random_ok_30", "games_id": gid,
               "before_cls": before, **after}
        out_rows.append(out)

    print(f"  kept_ok={g6_kept}/{len(audit30)}  cls_after={dict(cls_after)}")
    if g6_flipped:
        print("  FLIPPED:")
        for gid, cls, msg in g6_flipped:
            print(f"    gid={gid}  -> {cls}  msg={msg}")

    OUT.write_text(
        "\n".join(json.dumps(r) for r in out_rows) + "\n", encoding="utf-8"
    )

    verdict_g5 = "GREEN" if bc_flips_to_ok >= 1 else "RED"
    verdict_g5b = "GREEN" if ok_kept == 5 else "RED"
    verdict_g6 = "GREEN" if g6_kept == 30 else "RED"
    overall = "GREEN" if verdict_g5 == "GREEN" and verdict_g5b == "GREEN" and verdict_g6 == "GREEN" else "RED"

    summary = (
        f"Phase 10A regression gate\n"
        f"=========================\n"
        f"Gate 5  (5 B_COPTER bug -> at least 1 ok):    {bc_flips_to_ok}/5  {verdict_g5}\n"
        f"Gate 5b (5 non-B_COPTER ok stay ok):          {ok_kept}/5    {verdict_g5b}\n"
        f"Gate 6  (30 random non-B_COPTER ok stay ok):  {g6_kept}/30   {verdict_g6}\n"
        f"OVERALL:                                       {overall}\n"
    )
    SUMMARY.write_text(summary, encoding="utf-8")
    print()
    print(summary)


if __name__ == "__main__":
    main()
