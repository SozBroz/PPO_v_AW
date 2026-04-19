#!/usr/bin/env python3
"""
Compare two AWBW replay .zip files (official vs homebrew): structure, inner encoding,
turn counts, presence of action stream, and top-level PHP field keys on turn 0.

Usage:
  python tools/compare_awbw_replays.py path/to/official.zip path/to/homebrew.zip
  python tools/compare_awbw_replays.py --json official.zip homebrew.zip
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any


def _load_inner(raw: bytes) -> tuple[str, str]:
    """Return (label, text). Tries gzip first, then utf-8."""
    try:
        with gzip.open(io.BytesIO(raw)) as gz:
            return "gzip", gz.read().decode("utf-8")
    except OSError:
        return "raw", raw.decode("utf-8", errors="replace")


def _analyze_zip(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "entries": []}
    if not path.is_file():
        out["error"] = "file not found"
        return out

    with zipfile.ZipFile(path) as z:
        out["zip_comment"] = (z.comment or b"").decode("utf-8", errors="replace")
        for info in z.infolist():
            raw = z.read(info.filename)
            enc, text = _load_inner(raw)
            entry: dict[str, Any] = {
                "name": info.filename,
                "compressed_size": info.compress_size,
                "compress_type": info.compress_type,
                "inner_encoding": enc,
                "decompressed_bytes": len(text.encode("utf-8")),
            }
            first = text[:120].replace("\n", "\\n")
            entry["prefix"] = first

            if text.startswith("p:"):
                entry["kind"] = "actions"
                entry["action_stream_chars"] = len(text)
            elif 'O:8:"awbwGame"' in text:
                entry["kind"] = "game_state"
                lines = text.split("\n")
                entry["line_count"] = len(lines)
                entry["nonempty_lines"] = sum(1 for ln in lines if ln.strip())
                line0 = lines[0] if lines else ""
                # String keys after PHP s:N:"... pattern (length >= 2 avoids noise from nested fragments)
                keys = re.findall(r's:\d+:"([a-zA-Z_][a-zA-Z0-9_]*)";', line0[:200_000])
                seen: list[str] = []
                for k in keys:
                    if len(k) < 2:
                        continue
                    if k not in seen:
                        seen.append(k)
                entry["php_string_keys_turn0_ordered"] = seen
                entry["php_key_count_unique"] = len(seen)
            else:
                entry["kind"] = "unknown"

            out["entries"].append(entry)

    return out


def _diff_lists(a: list[str], b: list[str]) -> dict[str, list[str]]:
    sa, sb = set(a), set(b)
    return {
        "only_in_first": sorted(sa - sb),
        "only_in_second": sorted(sb - sa),
        "same_order": a == b,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Compare AWBW replay zip structures.")
    p.add_argument("reference", type=Path, help="Working / official replay (e.g. replay_1630459_*.zip)")
    p.add_argument("candidate", type=Path, help="Homebrew replay (e.g. 126694.zip)")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    args = p.parse_args()

    ref = _analyze_zip(args.reference)
    cand = _analyze_zip(args.candidate)

    if args.json:
        overlap = {}
        ref_gs = next((e for e in ref.get("entries", []) if e.get("kind") == "game_state"), None)
        cand_gs = next((e for e in cand.get("entries", []) if e.get("kind") == "game_state"), None)
        if ref_gs and cand_gs:
            kr = ref_gs.get("php_string_keys_turn0_ordered") or []
            kc = cand_gs.get("php_string_keys_turn0_ordered") or []
            overlap["php_keys"] = _diff_lists(kr, kc)
        print(json.dumps({"reference": ref, "candidate": cand, "diff": overlap}, indent=2))
        return 0

    print("=== Paths ===")
    print(f"  Reference:  {ref['path']}")
    print(f"  Candidate:  {cand['path']}")

    def summarize(label: str, data: dict[str, Any]) -> None:
        print(f"\n=== {label} ===")
        if "error" in data:
            print(f"  ERROR: {data['error']}")
            return
        for e in data["entries"]:
            print(f"  [{e['name']}] kind={e.get('kind')} inner={e.get('inner_encoding')}")
            if e.get("kind") == "game_state":
                print(f"    lines (snapshots): {e.get('line_count')}")
                print(f"    unique PHP string keys (turn 0, ordered): {e.get('php_key_count_unique')}")
            if e.get("kind") == "actions":
                print(f"    action blob length: {e.get('action_stream_chars')}")

    summarize("Reference", ref)
    summarize("Candidate", cand)

    ref_gs = next((e for e in ref.get("entries", []) if e.get("kind") == "game_state"), None)
    cand_gs = next((e for e in cand.get("entries", []) if e.get("kind") == "game_state"), None)
    ref_act = any(e.get("kind") == "actions" for e in ref.get("entries", []))
    cand_act = any(e.get("kind") == "actions" for e in cand.get("entries", []))

    print("\n=== Quick deltas ===")
    print(f"  Action stream present: reference={ref_act}  candidate={cand_act}")
    if ref_gs and cand_gs:
        print(f"  Snapshot lines: reference={ref_gs.get('line_count')}  candidate={cand_gs.get('line_count')}")
        kr = ref_gs.get("php_string_keys_turn0_ordered") or []
        kc = cand_gs.get("php_string_keys_turn0_ordered") or []
        d = _diff_lists(kr, kc)
        if d["only_in_first"]:
            print(f"  Keys only in reference: {d['only_in_first'][:40]}{' ...' if len(d['only_in_first']) > 40 else ''}")
        if d["only_in_second"]:
            print(f"  Keys only in candidate: {d['only_in_second'][:40]}{' ...' if len(d['only_in_second']) > 40 else ''}")
        print(f"  Same key order: {d['same_order']}")

    print("\n=== Suggested workflow ===")
    print("  1. Align zip entry names: official uses <game_id> for state and a<game_id> for actions.")
    print("  2. Diff first snapshot line: compare maps_id, weather_*, turn/day/funds/capture_win, buildings count.")
    print("  3. If viewer shows 'Desync': state snapshots disagree with action replay - add action stream or accept snapshot-only mode.")
    print("  4. Re-run with --json and diff the output, or gzip -dc < entry | head -c 5000 for manual inspection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
