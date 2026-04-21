#!/usr/bin/env python3
"""
Regenerate live-vs-zip overlap analysis and cleanliness summary.

Reads:
  logs/desync_register_amarriner_live.jsonl  (full live audit, one row per game)
  logs/desync_register.jsonl                 (zip pool audit)

Writes:
  logs/amarriner_gl_current_cleanliness.json
  logs/live_pool_overlap_analysis.md

Run after: python tools/desync_audit_amarriner_live.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.live_desync_buckets import bucket_live_failure, overlap_bucket_for_md  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _bucket_live_failure(msg: str, cls: str, action_kind: str | None) -> str:
    """Map a live first-divergence row to an overlap bucket (stable ids for tables)."""
    return overlap_bucket_for_md(bucket_live_failure(msg, cls, action_kind))


def _zip_oracle_gap_shapes(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Rough shape counts for non-ok zip oracle_gap rows."""
    c: Counter[str] = Counter()
    for r in rows:
        if r.get("class") != "oracle_gap":
            continue
        msg = r.get("message") or ""
        if "oracle_fire" in msg or "Fire (no path)" in msg:
            c["fire"] += 1
        elif "Capt (no path)" in msg:
            c["capt"] += 1
        elif "no repair-eligible" in msg.lower() or (
            "Repair" in msg and "ally" in msg
        ):
            c["repair_eligibility"] += 1
        elif "no unit" in msg.lower() and "Move" in msg:
            c["move_no_unit"] += 1
        elif "AttackSeam" in msg or "seam" in msg.lower():
            c["attackseam"] += 1
        elif "Load" in msg:
            c["load"] += 1
        else:
            c["other"] += 1
    return dict(c)


def _co_names(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for cid, row in (data.get("cos") or {}).items():
        if isinstance(row, dict) and "name" in row:
            out[str(cid)] = str(row["name"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--live-register",
        type=Path,
        default=ROOT / "logs" / "desync_register_amarriner_live.jsonl",
    )
    ap.add_argument(
        "--zip-register",
        type=Path,
        default=ROOT / "logs" / "desync_register.jsonl",
    )
    ap.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "data" / "amarriner_gl_current_list_p1.json",
    )
    ap.add_argument(
        "--co-data",
        type=Path,
        default=ROOT / "data" / "co_data.json",
    )
    ap.add_argument(
        "--out-cleanliness",
        type=Path,
        default=ROOT / "logs" / "amarriner_gl_current_cleanliness.json",
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "logs" / "live_pool_overlap_analysis.md",
    )
    args = ap.parse_args()

    def _posix(p: Path) -> str:
        try:
            return p.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return str(p)

    live = _load_jsonl(args.live_register)
    zrows = _load_jsonl(args.zip_register)
    co_names = _co_names(args.co_data)

    class_counts: Counter[str] = Counter()
    for r in live:
        class_counts[r.get("class") or "unknown"] += 1

    failures = [
        r
        for r in live
        if (r.get("class") or "") != "ok" and r.get("status") == "first_divergence"
    ]
    n_fail = len(failures)
    n_ok = class_counts.get("ok", 0)

    bucket_rows: list[tuple[int, str, str, str]] = []
    for r in failures:
        gid = int(r["games_id"])
        msg = r.get("message") or ""
        ak = r.get("approx_action_kind") or "?"
        cls = r.get("class") or ""
        b = _bucket_live_failure(msg, cls, ak)
        head = msg.replace("\n", " ")[:100]
        bucket_rows.append((gid, ak, b, head))

    bucket_counts: Counter[str] = Counter()
    for _gid, _ak, b, _h in bucket_rows:
        bucket_counts[b] += 1

    zip_class = Counter(r.get("class") for r in zrows)
    zip_gap_shapes = _zip_oracle_gap_shapes(zrows)

    cat_meta: dict[str, Any] = {}
    if args.catalog.is_file():
        cat = json.loads(args.catalog.read_text(encoding="utf-8"))
        cat_meta = cat.get("meta") or {}

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat(timespec="seconds")
    cleanliness = {
        "as_of_utc": now_iso,
        "source": "tools/desync_audit_amarriner_live.py + tools/analyze_live_pool_overlap.py",
        "catalog": _posix(args.catalog),
        "register": _posix(args.live_register),
        "games_audited": len(live),
        "class_counts": {k: class_counts[k] for k in sorted(class_counts)},
        "first_divergence_total": n_fail,
        "engine_bug_count": class_counts.get("engine_bug", 0),
        "interpretation": (
            "Live Amarriner load_replay.php JSON is replayed through apply_oracle_action_json. "
            "ok = stepped full half-turn stream without engine exception. "
            "oracle_gap = adapter/site-shape gaps (UnsupportedOracleAction) or oracle heuristics, "
            "not classified engine_bug. loader_error = harness parse issue (e.g. unexpected JSON keys)."
        ),
    }
    args.out_cleanliness.parent.mkdir(parents=True, exist_ok=True)
    args.out_cleanliness.write_text(
        json.dumps(cleanliness, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    covered_fire = bucket_counts.get("covered_by_fire_lane", 0)
    not_cov = n_fail - covered_fire

    # CO appearances on oracle_gap failures only (excludes loader_error)
    co_appear: Counter[str] = Counter()
    for r in failures:
        if r.get("class") != "oracle_gap":
            continue
        for key in ("co_p0_id", "co_p1_id"):
            v = r.get(key)
            if v is not None:
                co_appear[str(int(v))] += 1

    lines: list[str] = []
    lines.append("# Live GL pool vs zip oracle backlog — overlap forecast")
    lines.append("")
    if now_iso.endswith("+00:00"):
        stamp = now_iso[:-6].replace("T", " ", 1) + " UTC"
    else:
        stamp = now_iso.replace("T", " ", 1)
    lines.append(
        f"**As of:** {stamp} "
        f"(live: `{_posix(args.live_register)}`, zip: `{_posix(args.zip_register)}`)"
    )
    if cat_meta.get("scraped_at"):
        lines.append(
            f"**Catalog scraped:** {cat_meta.get('scraped_at')} "
            f"(`n_games={cat_meta.get('n_games', '?')}`)"
        )
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(
        f"Live audit: **{len(live)}** in-progress GL std games (no zip export). "
        f"**{n_ok}** replay clean through the live harness; **{n_fail}** first divergences "
        f"(`{class_counts.get('oracle_gap', 0)}` oracle_gap, `{class_counts.get('loader_error', 0)}` loader_error)."
    )
    lines.append("")
    lines.append(
        f"**Overlap with zip-pool fixes:** **{covered_fire}** live rows match the "
        f"`oracle_fire` / `drift_range_los_or_unmapped_co` cluster targeted by the zip backlog. "
        f"**{not_cov}** rows are other shapes (Load pairing, Join synthesis, Supply JSON, AttackSeam keys, etc.) "
        f"and do not automatically clear when fire-only fixes land."
    )
    lines.append("")
    lines.append(
        f"Zip register (`desync_register.jsonl`): **{zip_class.get('ok', 0)}** ok, "
        f"**{zip_class.get('oracle_gap', 0)}** oracle_gap, **{zip_class.get('engine_bug', 0)}** engine_bug "
        f"(`{len(zrows)}` rows total). "
        f"Among zip `oracle_gap` rows, dominant message shapes: {json.dumps(zip_gap_shapes)}."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    THEME_TITLE: dict[str, str] = {
        "covered_by_fire_lane": "A — Zip overlap: `oracle_fire` / range–LOS–CO drift",
        "covered_by_move_lane": "B — Zip overlap: move-no-unit pattern",
        "covered_by_repair_lane": "C — Zip overlap: repair-eligibility pattern",
        "covered_by_capt_lane": "D — Zip overlap: capture drift pattern",
        "covered_by_jess_refuel": "E — Zip overlap: Jess refuel (if any)",
        "covered_by_naval_idle_drain": "F — Zip overlap: naval idle drain (if any)",
        "not_covered_live_load_pairing": "G — Live-only: Load cargo/transport resolution",
        "not_covered_capt_position_drift": "H — Live-only: Capt position drift",
        "not_covered_supply_shape": "I — Live-only: Supply JSON shape",
        "not_covered_power_json_shape": "I2 — Live-only: Power JSON (missing playerID)",
        "not_covered_attackseam_loader": "J — Live-only: AttackSeam loader / key mapping",
        "not_covered_join_synth_path": "K — Live-only: Join → Move synthesis / no path",
        "not_covered_repair_json_shape": "L — Live-only: Repair dict shape",
        "not_covered_other": "M — Uncategorized",
    }

    lines.append("## 1. Regrouped by theme (live failures)")
    lines.append("")
    themes: dict[str, list[tuple[int, str, str]]] = {}
    for gid, ak, b, head in sorted(bucket_rows, key=lambda t: t[0]):
        title = THEME_TITLE.get(b, f"— `{b}`")
        themes.setdefault(title, []).append((gid, ak, head))

    def _sort_key(title: str) -> tuple[int, str]:
        if title and title[0].isalpha() and title[1:3] == " —":
            return (ord(title[0]), title)
        return (999, title)

    order = sorted(themes.keys(), key=_sort_key)
    for th in order:
        lines.append(f"### {th}")
        lines.append("")
        lines.append("| gid | action | message (head) |")
        lines.append("|-----|--------|----------------|")
        for gid, ak, head in themes[th]:
            lines.append(f"| {gid} | {ak} | {head} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 2. Overlap buckets (full taxonomy)")
    lines.append("")
    lines.append("| bucket | count |")
    lines.append("|--------|------:|")
    for name in sorted(bucket_counts.keys()):
        lines.append(f"| `{name}` | {bucket_counts[name]} |")
    lines.append("")
    cov_keys = [k for k in sorted(bucket_counts) if k.startswith("covered_by_")]
    not_keys = [k for k in sorted(bucket_counts) if k.startswith("not_covered_")]
    lines.append(
        f"**Covered-by-zip-lane subtotal:** {sum(bucket_counts[k] for k in cov_keys)}. "
        f"**Not-covered (live-only / cross-cutting) subtotal:** {sum(bucket_counts[k] for k in not_keys)}."
    )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 3. Per-row table (all failures)")
    lines.append("")
    lines.append("| gid | action | bucket | message (head) |")
    lines.append("|-----|--------|--------|----------------|")
    for gid, ak, b, head in sorted(bucket_rows, key=lambda t: t[0]):
        lines.append(f"| {gid} | {ak} | `{b}` | {head} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Appendix A — CO id frequency (oracle_gap failures only)")
    lines.append("")
    lines.append(
        "Per-`co_id` appearance count (each of `co_p0_id` and `co_p1_id` counted once per failing game). "
        "Names from `data/co_data.json`."
    )
    lines.append("")
    lines.append("| co_id | name | appearances |")
    lines.append("|------:|------|------------:|")
    for cid, cnt in sorted(co_appear.items(), key=lambda x: (-x[1], int(x[0]))):
        nm = co_names.get(cid, "?")
        lines.append(f"| {cid} | {nm} | {cnt} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Appendix B — Compact failure JSON (live)")
    lines.append("")
    lines.append("```json")
    for r in sorted(failures, key=lambda x: int(x["games_id"])):
        slim = {
            "games_id": r.get("games_id"),
            "map_id": r.get("map_id"),
            "co_p0_id": r.get("co_p0_id"),
            "co_p1_id": r.get("co_p1_id"),
            "class": r.get("class"),
            "message": r.get("message"),
            "approx_envelope_index": r.get("approx_envelope_index"),
            "approx_day": r.get("approx_day"),
            "actions_applied": r.get("actions_applied"),
        }
        lines.append(json.dumps(slim, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Method notes")
    lines.append("")
    lines.append(
        "- **Zip pool** rows are from batch `desync_audit.py` over downloaded replay zips; "
        "message-shape counts are heuristic substrings, not formal taxonomy IDs."
    )
    lines.append(
        "- **Live** rows are from `desync_audit_amarriner_live.py` (authenticated `load_replay.php` stream)."
    )
    lines.append(
        "- Regenerate this file: `python tools/analyze_live_pool_overlap.py` after a full live audit."
    )
    lines.append("")

    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[overlap] wrote {args.out_cleanliness}")
    print(f"[overlap] wrote {args.out_md}")
    print(f"[overlap] live: {len(live)} games, ok={n_ok}, fail={n_fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
