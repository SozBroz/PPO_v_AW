"""Dump residual engine_bug + flipped oracle_gap rows after Phase 10A audit."""
from __future__ import annotations

import json
from pathlib import Path

P = Path("logs/phase10a_sample_audit_results.jsonl")
rows = [json.loads(l) for l in P.read_text(encoding="utf-8").splitlines() if l.strip()]

print("--- still engine_bug ---")
for r in rows:
    if r.get("after_cls") == "engine_bug":
        msg = (r.get("msg") or "")[:200]
        print(f"  gid={r.get('games_id')}  label={r.get('label')}  "
              f"exc={r.get('exc')}  msg={msg}")

print("\n--- flipped to oracle_gap ---")
for r in rows:
    if r.get("after_cls") == "oracle_gap":
        msg = (r.get("msg") or "")[:200]
        print(f"  gid={r.get('games_id')}  label={r.get('label')}  "
              f"exc={r.get('exc')}  msg={msg}")
