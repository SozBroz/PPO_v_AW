"""
Cluster desync register rows into operational subtypes (message-shape buckets).

Usage (repo root)::

  python tools/cluster_desync_register.py --register logs/desync_register_regen_20260420.jsonl
  python tools/cluster_desync_register.py --register logs/desync_register.jsonl --markdown docs/desync_bug_tracker.md

Subtypes are stable prefixes for triage dashboards; they are not a second ``class``
column in ``desync_audit.py`` — see ``docs/desync_audit.md``. For audit + JSON + optional
markdown in one invocation, use ``tools/run_desync_cluster.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def desync_subtype(row: dict[str, Any]) -> str:
    cls = str(row.get("class") or "")
    msg = row.get("message") or ""
    mlow = msg.lower()
    if cls == "ok":
        return "ok"
    if cls == "catalog_incomplete":
        return "catalog_incomplete"
    if cls == "replay_no_action_stream":
        # Same subtype label as legacy loader_error+ReplaySnapshotOnly (stable dashboards).
        return "loader_rv1_no_action_stream"
    if cls == "loader_error":
        if "empty replay" in mlow:
            return "loader_empty_replay"
        et = str(row.get("exception_type") or "")
        if et == "ReplaySnapshotOnly" or "snapshots only" in mlow:
            # Zip has ``<games_id>`` PHP lines but no ``a<games_id>`` action gzip (RV1).
            return "loader_rv1_no_action_stream"
        return "loader_snapshot_or_zip"
    if cls == "engine_bug":
        if "illegal move" in mlow:
            return "engine_illegal_move"
        return "engine_other"
    if cls == "oracle_gap":
        if msg.startswith("Move: no unit"):
            return "oracle_move_no_unit"
        if "active_player" in msg:
            return "oracle_turn_active_player"
        if "Move resolved to ACTION" in msg or "no legal terminator" in msg:
            return "oracle_move_terminator"
        if msg.startswith("Capt") or (len(msg) > 4 and msg[:120].strip().startswith("Capt")):
            return "oracle_capture_path"
        if msg.startswith("Fire") or "Fire " in msg[:50]:
            return "oracle_fire"
        if msg.startswith("Repair"):
            return "oracle_repair"
        if msg.startswith("Supply"):
            # APC resupply / no-path Supply (distinct from Repair / Unload).
            return "oracle_supply"
        if msg.startswith("Unload"):
            return "oracle_unload"
        if "AttackSeam" in msg or msg.startswith("Seam"):
            return "oracle_seam"
        if msg.startswith("Build"):
            # Strict BUILD no-op / envelope drift (engine refused BUILD).
            return "oracle_build"
        if msg.startswith("Power"):
            return "oracle_power"
        if msg.startswith("Join"):
            return "oracle_join"
        if msg.startswith("Load"):
            return "oracle_load"
        if msg.startswith("Hide"):
            return "oracle_hide"
        if "unknown awbw unit name" in mlow:
            return "oracle_unknown_unit"
        if "unsupported oracle action" in mlow:
            return "oracle_unsupported_kind"
        return "oracle_other"
    if cls == "replay_aborted":
        return "legacy_replay_aborted"
    return cls or "unknown"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def cluster(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        out[desync_subtype(r)].append(int(r["games_id"]))
    for k in out:
        out[k].sort()
    return dict(out)


def render_markdown(
    *,
    register_path: Path,
    rows: list[dict[str, Any]],
    clusters: dict[str, list[int]],
    baseline_path: Path | None,
    baseline_rows: list[dict[str, Any]] | None,
) -> str:
    lines: list[str] = []
    lines.append("# Desync bug tracker (clustered)")
    lines.append("")
    lines.append("Auto-generated summary of `tools/desync_audit.py` register rows, grouped by **subtype**")
    lines.append("(message-shape buckets). Official taxonomy remains `class` in each JSONL row; subtypes are for triage.")
    lines.append("")
    lines.append(f"- **Source register:** `{_repo_rel(register_path)}`")
    lines.append(f"- **Games in register:** {len(rows)}")
    lines.append("")

    cls_counts = Counter(r.get("class") for r in rows)
    lines.append("## Summary by `class`")
    lines.append("")
    lines.append("| `class` | Count |")
    lines.append("|---------|------:|")
    for k in sorted(cls_counts.keys(), key=lambda x: (-cls_counts[x], str(x))):
        lines.append(f"| `{k}` | {cls_counts[k]} |")
    lines.append("")

    if baseline_rows is not None and baseline_path is not None:
        baseline_clusters = cluster(baseline_rows)
        all_subs = set(baseline_clusters.keys()) | set(clusters.keys())
        lines.append("## Subtype counts vs baseline")
        lines.append("")
        lines.append(
            f"Cluster sizes (subtype buckets) compared to `{_repo_rel(baseline_path)}`."
        )
        lines.append("")
        lines.append("| Subtype | Baseline | Current | Δ |")
        lines.append("|---------|---------:|--------:|--:|")
        for sub in sorted(
            all_subs,
            key=lambda s: (
                -abs(len(clusters.get(s, [])) - len(baseline_clusters.get(s, []))),
                s,
            ),
        ):
            b_ct = len(baseline_clusters.get(sub, []))
            c_ct = len(clusters.get(sub, []))
            delta = c_ct - b_ct
            lines.append(f"| `{sub}` | {b_ct} | {c_ct} | {delta:+d} |")
        lines.append("")

        old_cls = Counter(r.get("class") for r in baseline_rows)
        new_cls = Counter(r.get("class") for r in rows)
        lines.append("## Progress vs baseline snapshot")
        lines.append("")
        lines.append(
            f"Compared to `{_repo_rel(baseline_path)}` (same catalog + zip set when both runs used full GL-std scope)."
        )
        lines.append("")
        old_by_gid = {int(r["games_id"]): r for r in baseline_rows}
        new_by_gid = {int(r["games_id"]): r for r in rows}
        if set(old_by_gid) == set(new_by_gid):
            became_ok = [
                g
                for g in old_by_gid
                if old_by_gid[g].get("class") != "ok" and new_by_gid[g].get("class") == "ok"
            ]
            lost_ok = [
                g
                for g in old_by_gid
                if old_by_gid[g].get("class") == "ok" and new_by_gid[g].get("class") != "ok"
            ]
            lines.append(
                f"- **Net `ok` games:** {new_cls.get('ok', 0)} now vs {old_cls.get('ok', 0)} in baseline "
                f"({new_cls.get('ok', 0) - old_cls.get('ok', 0):+d})."
            )
            lines.append(
                f"- **Flipped to `ok`:** {len(became_ok)} (from non-ok in baseline). "
                f"**Flipped from `ok`:** {len(lost_ok)}."
            )
            if became_ok:
                c_prev = Counter(old_by_gid[g].get("class") for g in became_ok)
                lines.append(f"- **Previous class of flipped-to-ok games:** {dict(c_prev)}")
            lines.append("")
            lines.append("Documented engine-side fixes (see also `docs/desync_audit.md` § Resolved):")
            lines.append("")
            lines.append("- **Port naval movement (2026-04):** neutral Port tiles use port cost table for Lander/Black Boat; removes a large `engine_illegal_move` cluster.")
            lines.append("- **Resign / terminal replay:** games that previously registered as `replay_aborted` often complete as **`ok`** once resign is applied without raising.")
            lines.append("")
        else:
            lines.append(
                f"- **Per-game `ok` flips:** not computed — `games_id` set differs "
                f"(baseline {len(old_by_gid)} games, current {len(new_by_gid)})."
            )
            lines.append("")

    lines.append("## Subtypes (grouped backlog)")
    lines.append("")
    lines.append("| Subtype | Count | Typical cause | Where to fix |")
    lines.append("|---------|------:|---------------|--------------|")
    blurbs: dict[str, tuple[str, str]] = {
        "ok": ("Clean oracle replay.", "—"),
        "catalog_incomplete": ("Catalog missing `co_p0_id` / `co_p1_id`.", "`data/amarriner_gl_std_catalog.json` + scrape"),
        "loader_empty_replay": ("Zip has no PHP snapshot frames.", "Zip download / parse"),
        "loader_rv1_no_action_stream": (
            "Zip has snapshots only — no `a<game_id>` gzip with `p:` lines (ReplayVersion 1). Register `class` is `replay_no_action_stream`.",
            "Not fixable in oracle; mirror may never ship `a<games_id>` for these games",
        ),
        "loader_snapshot_or_zip": ("Other CO/zip layout or missing member (not RV1-only).", "`tools/diff_replay_zips.py`, catalog drift"),
        "engine_illegal_move": ("Engine rejected reachability / move legality.", "`engine/` movement + `docs/desync_audit.md` triage"),
        "engine_other": ("Non–illegal-move engine exception under mapped action.", "`engine/` + oracle path"),
        "oracle_move_no_unit": ("Oracle cannot resolve AWBW unit id to engine tile.", "`tools/oracle_zip_replay.py` Move + `units_id`"),
        "oracle_turn_active_player": ("Envelope player vs `active_player`.", "`tools/oracle_zip_replay.py` turn advance"),
        "oracle_move_terminator": ("Move ended on property/combat tile; JOIN/LOAD/CAPTURE/WAIT choice.", "`tools/oracle_zip_replay.py` terminator resolution"),
        "oracle_capture_path": ("Capt / no-path capture shapes.", "`tools/oracle_zip_replay.py` Capt"),
        "oracle_fire": ("Fire without full Move path, indirects, attacker resolution.", "`tools/oracle_zip_replay.py` Fire"),
        "oracle_repair": ("Black Boat repair target/eligibility.", "`tools/oracle_zip_replay.py` Repair"),
        "oracle_supply": ("APC Supply / nested Supply.unit / WAIT at path end.", "`tools/oracle_zip_replay.py` Supply"),
        "oracle_unload": ("Transport unload / cargo resolution.", "`tools/oracle_zip_replay.py` Unload"),
        "oracle_seam": ("Seam attack mapping.", "`tools/oracle_zip_replay.py`"),
        "oracle_build": ("BUILD refused or strict no-op (funds/owner/tile); not `active_player` rows.", "`tools/oracle_zip_replay.py` Build"),
        "oracle_power": ("COP/SCOP envelope or power-stage resolution.", "`tools/oracle_zip_replay.py` Power"),
        "oracle_join": ("Join nested Move / join resolution.", "`tools/oracle_zip_replay.py` Join"),
        "oracle_load": ("Load nested Move / load resolution.", "`tools/oracle_zip_replay.py` Load"),
        "oracle_hide": ("Sub dive/hide path or no-path hide.", "`tools/oracle_zip_replay.py` Hide"),
        "oracle_unknown_unit": ("`units_name` not mapped to `UnitType`.", "`tools/oracle_zip_replay.py` naming"),
        "oracle_unsupported_kind": ("Action kind not implemented in oracle.", "`tools/oracle_zip_replay.py`"),
        "oracle_other": ("Other `UnsupportedOracleAction` message shape (after prefix rules).", "`tools/oracle_zip_replay.py`"),
        "legacy_replay_aborted": ("Legacy audit only — resign / early abort classification.", "Re-run audit; expect `ok` if engine handles resign"),
    }
    for sub in sorted(clusters.keys(), key=lambda s: (-len(clusters[s]), s)):
        if sub == "ok":
            continue
        n = len(clusters[sub])
        cause, fix = blurbs.get(sub, ("See `message` in register.", "`tools/oracle_zip_replay.py` / `engine/`"))
        lines.append(f"| `{sub}` | {n} | {cause} | {fix} |")
    lines.append("")

    lines.append("### `ok` games")
    lines.append("")
    ok_ids = clusters.get("ok", [])
    lines.append(f"- **Count:** {len(ok_ids)}")
    if len(ok_ids) <= 40:
        lines.append(f"- **games_id:** {', '.join(str(g) for g in ok_ids)}")
    else:
        lines.append(f"- **games_id (first 40):** {', '.join(str(g) for g in ok_ids[:40])} … *({len(ok_ids) - 40} more)*")
    lines.append("")

    for sub in sorted(clusters.keys(), key=lambda s: (-len(clusters[s]), s)):
        if sub == "ok":
            continue
        gids = clusters[sub]
        sample_msg = ""
        for r in rows:
            if desync_subtype(r) == sub and r.get("message"):
                sample_msg = (r["message"] or "").replace("\n", " ")[:160]
                break
        lines.append(f"### `{sub}` ({len(gids)} games)")
        lines.append("")
        if sample_msg:
            lines.append(f"*Example message:* `{sample_msg}`")
            lines.append("")
        if len(gids) <= 60:
            lines.append("**games_id:** " + ", ".join(str(g) for g in gids))
        else:
            lines.append("**games_id (first 60):** " + ", ".join(str(g) for g in gids[:60]))
            lines.append("")
            lines.append(f"*…and {len(gids) - 60} more (see JSON export).*")
        lines.append("")

    lines.append("## Machine-readable export")
    lines.append("")
    lines.append(
        f"Run `python tools/cluster_desync_register.py --register {_repo_rel(register_path)} --json "
        f"logs/desync_clusters.json` to emit `{len(clusters)}` keys with sorted `games_id` lists."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--register", type=Path, required=True, help="Path to desync register JSONL")
    ap.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional older register JSONL for progress section (same games_id set)",
    )
    ap.add_argument("--markdown", type=Path, default=None, help="Write markdown report")
    ap.add_argument("--json", type=Path, default=None, help="Write {\"subtype\": [games_id, ...]} JSON")
    args = ap.parse_args()

    if not args.register.is_file():
        print(f"missing {args.register}", file=sys.stderr)
        return 1

    rows = load_jsonl(args.register)
    clusters = cluster(rows)
    baseline_rows: list[dict[str, Any]] | None = None
    baseline_path = args.baseline
    if args.baseline:
        if not args.baseline.is_file():
            print(f"missing baseline {args.baseline}", file=sys.stderr)
            return 1
        baseline_rows = load_jsonl(args.baseline)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(clusters, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[cluster_desync_register] wrote {args.json}")

    if args.markdown:
        md = render_markdown(
            register_path=args.register.resolve(),
            rows=rows,
            clusters=clusters,
            baseline_path=baseline_path.resolve() if baseline_path else None,
            baseline_rows=baseline_rows,
        )
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(md, encoding="utf-8")
        print(f"[cluster_desync_register] wrote {args.markdown}")

    if not args.json and not args.markdown:
        for sub in sorted(clusters.keys(), key=lambda s: (-len(clusters[s]), s)):
            print(f"{sub:40} {len(clusters[sub]):5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
