# Phase 11J — GID 1607045 regression closeout

## Executive summary

| Field | Value |
|--------|--------|
| **games_id** | 1607045 |
| **COs (engine seats)** | P0 **Drake** (`co_p0_id` 5), P1 **Rachel** (`co_p1_id` 28) |
| **Symptom (post wave-5)** | `oracle_gap` ~day **24**, **Build** ARTILLERY at **(4,20)** for P0 — engine **insufficient funds** vs PHP (e.g. engine **5670**, need **6000**) |
| **Wave suspects** | **Not applicable as primary cause:** no Sasha, Sonja, or Colin on this map. Inherited wave reports: [L1-WAVE-2 (Sasha War Bonds)](phase11j_l1_wave_2_ship.md), [Sonja D2D](phase11j_sonja_d2d_impl.md), [Colin](phase11j_colin_impl_ship.md). |
| **Bisect** | Forcing **all War Bonds deferred** and disabling **Sonja ×1.5 counter-attack** did **not** clear 1607045 — consistent with Drake/Rachel and co-gated paths. |
| **Verdict** | **YELLOW** — regression is **not** cleanly attributable to a single wave-of-five engine edit for this gid; root failure mode is **oracle combat HP pinning vs PHP fractional internal HP**, which snowballs into income/repair/build. **Ship-or-revert** for the *wave* is **N/A** for 1607045; keeping wave edits does **not** regress the 11 closure games (see table). |

## Root cause classification

1. **Register delta (936 corpus)**  
   Only **1607045** flipped **ok → `oracle_gap`** between the prior postfix register and `logs/desync_register_post_wave5_936_20260421_1335.jsonl`.  
   **11** games flipped **`oracle_gap` → ok** (closures attributed to the wave):  
   `1622501, 1624764, 1626284, 1626991, 1628953, 1630669, 1634146, 1634267, 1634893, 1635164, 1635658`.

2. **Drill signal**  
   First **P0 funds** divergence vs PHP appears at **end of envelope 43** (turn roll: income + property repair for Drake). Earlier **unit HP** mismatches vs PHP begin around **env 13 / day 7** (Rachel envelope with **Fire**). The day-24 build shortfall is **downstream** of that combat/funds chain.

3. **Mechanism (oracle, not terminator)**  
   `_oracle_set_combat_damage_override_from_combat_info` in `tools/oracle_zip_replay.py` maps AWBW `combatInfo` **`units_hit_points`** as **integer display bars** → internal HP via **`int × 10`** (capped 0–100). PHP can carry **fractional display** / sub-bucket internal HP that still **rounds to the same integer bar** in logs. Pinning damage from **integer bar only** can **over- or under-shoot** true internal HP vs PHP; luck and cascaded **repair/income** then diverge. Relevant implementation (unchanged in this closeout — **T5-owned** non-terminator oracle path):

```1247:1271:D:\AWBW\tools\oracle_zip_replay.py
    def _to_internal(disp_raw: Any) -> Optional[int]:
        if disp_raw is None:
            return None
        try:
            d = int(disp_raw)
        except (TypeError, ValueError):
            return None
        return max(0, min(100, d * 10))

    awbw_def_hp = _to_internal(def_ci.get("units_hit_points")) if isinstance(def_ci, dict) else None
    awbw_att_hp = _to_internal(att_ci.get("units_hit_points")) if isinstance(def_ci, dict) else None
    ...
    state._oracle_combat_damage_override = (dmg, counter)
```

   Experimental oracle changes (max-damage tie-break, bucket scan, etc.) were **reverted**; they **over-corrected** other envelopes (e.g. earlier build failures). **No safe ≤15 LOC surgical refinement** was validated within the mission window without T5 corpus rules.

## Fix / refinement / escalation

| Option | Choice |
|--------|--------|
| **A — Surgical refinement (engine)** | **Not taken** — COs are Drake/Rachel; wave edits for Sasha/Sonja/Colin are **not** the lever for this gid. |
| **B — Escalate / residual** | **Taken for this report** — treat 1607045 as **known residual** until oracle combat pinning can reconcile **integer `units_hit_points`** with PHP’s **fractional internal** trajectory (T5 lane: non-terminator helpers only per hard rules). |
| **C — Revert wave** | **Not recommended** — would **lose 11 closures** with **no** evidence the wave caused this regression. |

## Closure validation table

Audit command (post closeout):  
`python tools/desync_audit.py --catalog data/amarriner_gl_std_catalog.json --catalog data/amarriner_gl_extras_catalog.json --games-id 1607045 --games-id <11 closures> --register logs/_phase11j_1607045_closeout.jsonl`

| games_id | Status |
|----------|--------|
| 1607045 | **oracle_gap** (~day 24, P0 build insufficient funds) |
| 1622501 | ok |
| 1624764 | ok |
| 1626284 | ok |
| 1626991 | ok |
| 1628953 | ok |
| 1630669 | ok |
| 1634146 | ok |
| 1634267 | ok |
| 1634893 | ok |
| 1635164 | ok |
| 1635658 | ok |

**Summary:** 11 / 11 closures **ok**; 1607045 **oracle_gap** unchanged.

## Pytest

`python -m pytest --tb=no -q --ignore=test_trace_182065_seam_validation.py`  
**Result:** 654 passed, 5 skipped, 2 xfailed, 3 xpassed, **0 failed** (80.69s). Meets **≤2 failures** baseline.

## Hard-rule compliance

- Did **not** modify Rachel SCOP missile AOE, Von Bolt, `tools/desync_audit.py`, or Fire/Move **terminator** helpers in `oracle_zip_replay.py`.
- Did **not** wholesale-revert L1-WAVE-2 / Sonja / Colin / state-retune / delete-RL-guard.

## Follow-up lane (T5)

1. Define a **provably unique** mapping from `combatInfo` + envelope context to defender/attacker internal HP when PHP uses **sub-display** precision (or ingest fractional `units_hit_points` if present in some frames).  
2. Re-run **1607045 + 11 closures + 100-game sample** after oracle fix.

## Verdict letter

**YELLOW** — Wave-of-five engine edits are **not** the attributable root cause for **1607045** (Drake/Rachel); **11 closures remain green**; **1607045** stays **`oracle_gap`** pending **oracle combat HP** refinement (escalated to T5). **Not RED** — corpus trade-off favors **keeping the wave**; residual is **isolated** and **explained**.

---

*"In any moment of decision, the best thing you can do is the right thing, the next best thing is the wrong thing, and the worst thing you can do is nothing."* — often attributed to Theodore Roosevelt (early 20th c., U.S. President and reformer; wording varies in popular quotation).
