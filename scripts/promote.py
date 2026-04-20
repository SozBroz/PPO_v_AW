#!/usr/bin/env python3
"""
Operator gate for ``checkpoints/promoted/best.zip``.

Manual (default): list ``promoted/candidate_*.zip``, pick one, confirm, atomically replace ``best.zip``.

``--auto-promote``: promote the newest ``candidate_*.zip`` that has a fleet verdict JSON
(``fleet/*/eval/*.json``) with ``schema_version`` 1, ``promotion_threshold_met`` true, and
``promoted_candidate_zip`` pointing at that candidate. Thresholds are enforced when the
eval daemon writes the verdict (see ``scripts/fleet_eval_daemon.py``).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _iter_verdicts(repo: Path):
    fleet = repo / "fleet"
    if not fleet.is_dir():
        return
    for box in sorted(fleet.iterdir()):
        if not box.is_dir():
            continue
        ev = box / "eval"
        if not ev.is_dir():
            continue
        for p in sorted(ev.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            yield p


def _pick_auto_candidate(repo: Path, promoted: Path) -> Path | None:
    """Newest candidate zip referenced by a qualifying verdict."""
    candidates = {p.resolve() for p in promoted.glob("candidate_*.zip")}
    best: Path | None = None
    best_mtime = -1.0
    for vpath in _iter_verdicts(repo):
        try:
            data = json.loads(vpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if int(data.get("schema_version", 0)) != 1:
            continue
        if not data.get("promotion_threshold_met"):
            continue
        rel = data.get("promoted_candidate_zip")
        if not rel:
            continue
        cand = Path(rel)
        cand = cand.resolve() if cand.is_absolute() else (repo / cand).resolve()
        if cand not in candidates:
            continue
        m = cand.stat().st_mtime
        if m > best_mtime:
            best_mtime = m
            best = cand
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=_ROOT, help="Repository root")
    ap.add_argument(
        "--auto-promote",
        action="store_true",
        help="Pick newest candidate that a fleet verdict marks as threshold-met",
    )
    ap.add_argument("--yes", "-y", action="store_true", help="Skip confirmation (manual mode)")
    args = ap.parse_args()

    repo = args.repo.resolve()
    promoted = repo / "checkpoints" / "promoted"
    promoted.mkdir(parents=True, exist_ok=True)
    best = promoted / "best.zip"

    if args.auto_promote:
        chosen = _pick_auto_candidate(repo, promoted)
        if chosen is None:
            print("[promote] auto: no qualifying candidate + verdict pair found")
            return 2
        print(f"[promote] auto: {chosen.name}")
    else:
        cands = sorted(promoted.glob("candidate_*.zip"), key=lambda p: p.stat().st_mtime)
        if not cands:
            print("[promote] No candidate_*.zip in promoted/")
            return 1
        print("[promote] Candidates (oldest first):")
        for i, c in enumerate(cands):
            print(f"  [{i}] {c.name}")
        raw = input("Index to promote (empty = abort): ").strip()
        if raw == "":
            print("[promote] Aborted.")
            return 0
        chosen = cands[int(raw)]

    if not chosen.is_file():
        print(f"[promote] Missing file: {chosen}")
        return 3

    if not args.auto_promote and not args.yes:
        print(f"This will overwrite: {best}")
        if input("Type YES: ").strip() != "YES":
            print("[promote] Aborted.")
            return 0

    tmp = promoted / f".best_new_{int(time.time())}.zip"
    shutil.copy2(chosen, tmp)
    tmp.replace(best)
    print(f"[promote] best.zip <- {chosen.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
