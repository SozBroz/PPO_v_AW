"""Wave driver for Phase 11J damage canon.

Applies changes one batch at a time:
  1. Snapshot current state
  2. Apply N changes
  3. Run 936-game audit
  4. Compare to baseline (ok >= baseline_ok AND engine_bug == 0)
  5. If pass -> commit, move on
     If fail -> bisect: split batch in half, recurse

Outputs per-batch register paths under logs/_canon_wave1_batch<N>.jsonl
and a final summary JSON at logs/_phase11j_wave_results.json.

Run:
  python tools/_phase11j_wave_driver.py wave1
  python tools/_phase11j_wave_driver.py wave2
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIFF_JSON = REPO / "logs" / "_phase11j_canon_diff.json"
BASELINE_OK = 928       # current floor (was 927; we run hotter now)
BASELINE_GAP = 8        # current oracle_gap count
BATCH_SIZE = 20

AUDIT_CMD = [
    sys.executable, "tools/desync_audit.py",
    "--catalog", "data/amarriner_gl_std_catalog.json",
    "--catalog", "data/amarriner_gl_extras_catalog.json",
]


def load_table():
    return json.loads((REPO / "data" / "damage_table.json").read_text(encoding="utf-8"))


def save_table(obj) -> None:
    (REPO / "data" / "damage_table.json").write_text(
        json.dumps(obj, indent=2) + "\n", encoding="utf-8"
    )


def apply_changes(changes, *, direction: str) -> None:
    """direction='apply' sets to new value; 'revert' sets to old value."""
    table = load_table()
    name_to_idx = {u: i for i, u in enumerate(table["unit_order"])}
    rows = table["table"]
    for ch in changes:
        ai = name_to_idx[ch["att"]]
        di = name_to_idx[ch["def"]]
        if direction == "apply":
            rows[ai][di] = ch["new"]
        elif direction == "revert":
            rows[ai][di] = ch["old"]
        else:
            raise ValueError(direction)
    save_table(table)


def run_audit(register_path: Path) -> dict:
    cmd = AUDIT_CMD + ["--register", str(register_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    out = proc.stdout
    # Parse the trailer counts
    counts = {"ok": 0, "engine_bug": 0, "oracle_gap": 0, "other": {}}
    in_summary = False
    for line in out.splitlines():
        line = line.rstrip()
        if "[desync_audit]" in line and "audited" in line:
            in_summary = True
            continue
        if in_summary:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) >= 2 and parts[-1].isdigit():
                key = " ".join(parts[:-1])
                counts.setdefault("other", {})[key] = int(parts[-1])
                if key == "ok":
                    counts["ok"] = int(parts[-1])
                elif key == "engine_bug":
                    counts["engine_bug"] = int(parts[-1])
                elif key == "oracle_gap":
                    counts["oracle_gap"] = int(parts[-1])
            else:
                in_summary = False
    if proc.returncode != 0:
        print(f"  AUDIT NONZERO RC={proc.returncode}; tail of stderr:")
        print(proc.stderr[-2000:])
    return counts


def is_pass(counts: dict) -> bool:
    return (
        counts.get("engine_bug", 0) == 0
        and counts.get("ok", 0) >= BASELINE_OK
    )


def try_apply_batch(changes: list, label: str, depth: int = 0) -> tuple[list, list]:
    """Return (applied, reverted)."""
    indent = "  " * depth
    print(f"{indent}[{label}] trying {len(changes)} change(s)")
    apply_changes(changes, direction="apply")
    reg = REPO / "logs" / f"_canon_{label}.jsonl"
    counts = run_audit(reg)
    print(f"{indent}  -> ok={counts.get('ok')}  engine_bug={counts.get('engine_bug')}  "
          f"oracle_gap={counts.get('oracle_gap')}")
    if is_pass(counts):
        return list(changes), []
    if len(changes) == 1:
        print(f"{indent}  REVERT single cell {changes[0]['att']} vs {changes[0]['def']}: "
              f"{changes[0]['old']} -> {changes[0]['new']} caused regression")
        apply_changes(changes, direction="revert")
        # Re-audit to confirm we're back at baseline
        counts2 = run_audit(reg)
        print(f"{indent}  post-revert: ok={counts2.get('ok')} engine_bug={counts2.get('engine_bug')}")
        return [], list(changes)
    # Bisect
    print(f"{indent}  BISECT: regression found, splitting batch")
    apply_changes(changes, direction="revert")  # roll back whole batch
    # Re-audit to ensure baseline restored before bisecting
    counts_revert = run_audit(REPO / "logs" / f"_canon_{label}_revert_check.jsonl")
    if counts_revert.get("ok", 0) < BASELINE_OK or counts_revert.get("engine_bug", 0) > 0:
        print(f"{indent}  WARNING: revert did not restore baseline! "
              f"ok={counts_revert.get('ok')} bug={counts_revert.get('engine_bug')}")
    half = len(changes) // 2
    left = changes[:half]
    right = changes[half:]
    a1, r1 = try_apply_batch(left, f"{label}_L", depth + 1)
    a2, r2 = try_apply_batch(right, f"{label}_R", depth + 1)
    return a1 + a2, r1 + r2


def run_wave(wave_key: str, batch_size: int) -> dict:
    diff = json.loads(DIFF_JSON.read_text(encoding="utf-8"))
    changes = diff[wave_key]
    print(f"=== {wave_key}: {len(changes)} cell(s) targeted ===")

    applied_all: list = []
    reverted_all: list = []
    batches = [changes[i:i + batch_size] for i in range(0, len(changes), batch_size)]
    for bi, batch in enumerate(batches, start=1):
        label = f"{wave_key}_batch{bi}"
        applied, reverted = try_apply_batch(batch, label)
        applied_all.extend(applied)
        reverted_all.extend(reverted)
        if len(reverted) > 5:
            # Stop condition per orders
            print(f"STOP: batch {bi} had {len(reverted)} reverts; aborting wave")
            break
    summary = {
        "wave": wave_key,
        "applied_count": len(applied_all),
        "reverted_count": len(reverted_all),
        "applied": applied_all,
        "reverted": reverted_all,
    }
    out = REPO / "logs" / f"_phase11j_{wave_key}_results.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    print(f"applied:  {len(applied_all)}")
    print(f"reverted: {len(reverted_all)}")
    return summary


def main() -> None:
    wave = sys.argv[1] if len(sys.argv) > 1 else "wave1"
    if wave == "wave1":
        run_wave("wave1_changes", BATCH_SIZE)
    elif wave == "wave2":
        # wave2: one cell at a time per orders
        run_wave("wave2_changes", 1)
    else:
        raise SystemExit(f"unknown wave {wave}")


if __name__ == "__main__":
    main()
