"""Diff funds-row gid sets between two desync_audit registers."""
from __future__ import annotations
import json, sys
from pathlib import Path

def load_funds(p: str) -> set[int]:
    out: set[int] = set()
    for ln in Path(p).read_text(encoding="utf-8", errors="replace").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("class") == "state_mismatch_funds":
            out.add(r["games_id"])
    return out

def load_class(p: str, cls: str) -> set[int]:
    out: set[int] = set()
    for ln in Path(p).read_text(encoding="utf-8", errors="replace").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("class") == cls:
            out.add(r["games_id"])
    return out

if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    for cls in ("state_mismatch_funds", "state_mismatch_units", "state_mismatch_multi"):
        sa = load_class(a, cls)
        sb = load_class(b, cls)
        print(f"--- {cls} ---")
        print(f"baseline_only ({a}): {sorted(sa - sb)}")
        print(f"new_only      ({b}): {sorted(sb - sa)}")
        print(f"common: {len(sa & sb)}")
