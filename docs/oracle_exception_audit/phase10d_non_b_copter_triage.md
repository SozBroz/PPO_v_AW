# Phase 10D — MECH / RECON / MEGA_TANK / BLACK_BOAT Fire-drift triage (non–B_COPTER)

**Campaign:** `desync_purge_engine_harden`  
**Mode:** read-only recon (no engine/oracle edits)  
**Inputs:** `logs/desync_register_post_phase9.jsonl`, prior phase write-ups, `engine/action.py`, `logs/desync_regression_log.md` (Phase 6 / orchestrator footnote)  
**Artifacts:** `logs/phase10d_targets.jsonl`, `logs/phase10d_classification.json`, this document.

## Executive summary

| Unit class | Count | Smallest-drift `games_id` | Cause class (A–F) | Shared with Lane 10A air-pathing hypothesis? |
|------------|------:|--------------------------|---------------------|-----------------------------------------------|
| MECH | 9 | 1622104 (drift 1) | **E** | **No** — `unit_class` is infantry; not air/copter reachability |
| RECON | 3 | 1626655 (drift 2) | **E** | **No** — vehicle / ground movement |
| MEGA_TANK | 2 | 1634561 (drift 2) | **E** | **No** — vehicle |
| BLACK_BOAT | 1 | 1626642 (drift 0) | **F** | **No** — failure is **unarmed** attacker / ATTACK illegality, not tile drift |

**Lane 10A (B_COPTER):** Treat these **15** rows as **mostly orthogonal** to a fix scoped to **air** `compute_reachable_costs` behavior. **14/15** match the **Phase 7 Bucket A** narrative (`unit_pos` ≠ replay firing tile / nested **Fire** move not reflected in engine state before `_apply_attack`). **1/15** (**BLACK_BOAT**) is **drift 0** and is **not** a movement parity problem at all.

**Phase 11 queue:** Continue **Fire nested Move + board sync** work (Lane G / Lane L family), not MECH-specific terrain charts, unless a drill proves a **B** terrain row.

## Classification legend (A–F)

| Class | Meaning |
|-------|---------|
| **A** | Same root as B_COPTER air / air `compute_reachable_costs` issue |
| **B** | Terrain cost mismatch in `engine/units.py` / movement tables vs [AWBW movement rules](https://awbw.fandom.com/wiki/Movement) |
| **C** | CO movement bonus parity (canonical example: Lane M Andy SCOP +1 in `compute_reachable_costs`) |
| **D** | Load/unload / cargo state desync |
| **E** | **Fire-time board drift** — engine tile lags AWBW nested-move path end (Phase 7 **Bucket A**; Phase 8 Lane G context) |
| **F** | **Oracle / replay** — engine rule set rejects the action (e.g. attack with unarmed unit) |

## Per-row table (15 rows)

| `games_id` | Unit | Drift | Class |
|------------|------|------:|-------|
| 1617442 | MECH | 2 | E |
| 1622104 | MECH | 1 | E |
| 1625804 | MECH | 2 | E |
| 1626385 | MECH | 1 | E |
| 1626642 | BLACK_BOAT | 0 | **F** |
| 1626655 | RECON | 2 | E |
| 1629539 | RECON | 3 | E |
| 1629951 | MECH | 1 | E |
| 1630188 | MECH | 2 | E |
| 1630983 | MECH | 2 | E |
| 1631289 | MECH | 2 | E |
| 1631494 | MEGA_TANK | 5 | E |
| 1634256 | RECON | 4 | E |
| 1634561 | MEGA_TANK | 2 | E |
| 1634717 | MECH | 2 | E |

Full machine-readable copy: `logs/phase10d_classification.json`.

## Case studies (smallest drift per class)

### MECH — `1622104`

- **Failure:** `_apply_attack` — target `(6, 16)` not in range for MECH **from** `(6, 17)` while **unit_pos** `(7, 17)` (drift 1).
- **Failing stream:** `approx_envelope_index` 43, day 22, kind **Fire**.
- **Nested path:** `paths.global` length 2; tail ends at AWBW `{x:17, y:6}` → engine coordinates **(row 6, col 17) = (6, 17)**, matching oracle **from**; engine still holds **(7, 17)**.
- **CO powers at failure:** `cop_active` / `scop_active` **false** for both players (`co_p0_id` 22 Jake, `co_p1_id` 11 Adder).
- **Interpretation:** Same **shape** as Phase 7 case study **1618770** (tank): replay path end is the firing stance; **engine did not commit** that tile before combat. **Not** an air-movement cost table issue.

### RECON — `1626655`

- **Failure:** from `(13, 21)` vs unit_pos `(14, 20)` (drift 2); env 15, day 8, Fire; path tail matches **from**.
- **CO powers:** off (Andy vs Andy).
- **Interpretation:** **E** — chassis irrelevant; still Bucket A / nested Fire.

### MEGA_TANK — `1634561`

- **Failure:** from `(2, 12)` vs unit_pos `(2, 14)` (drift 2); env 57, day 29, Fire; path tail lands on `(2, 12)`.
- **Overlap:** Phase 9 Lane M audit already notes this gid as **engine_bug** at **MEGA_TANK** Fire **after** Capt path was fixed — consistent with **deeper Fire drift**, not Capt-only.
- **Interpretation:** **E**.

### BLACK_BOAT — `1626642` (**Class F**)

- **Failure:** **Drift 0** — `from` and `unit_pos` both `(1, 3)`; target `(2, 3)` is orthogonal (Manhattan 1). Under Phase 6 canon, direct fire uses **Manhattan-1** only ([AWBW combat / adjacency](https://awbw.fandom.com/wiki/Combat); orchestrator footnote in `logs/desync_regression_log.md` rejects diagonal direct fire).
- **Engine:** `data/damage_table.json` row for `UnitType.BLACK_BOAT` (index 23) has **no** non-null attack entries — `get_attack_targets` will never list enemy units for Black Boat **ATTACK**.
- **AWBW primary source:** [Black Boat](https://awbw.fandom.com/wiki/Black_Boat) — transport / repair role; not a direct-combat unit in the main-series sense. The live replay still carries **`action: Fire`**; the engine correctly treats **ATTACK** as impossible → **`_apply_attack` range** error is a **semantic mismatch** (Fire vs **Repair** / export), **Lane 10B-style oracle/replay**, not `compute_reachable_costs`.

## Cross-row root-cause grouping

1. **MECH + RECON + MEGA_TANK (14 rows):** **One shared failure family (E)** — **Fire** envelope: AWBW `paths.global` end matches oracle **from**, **engine `unit_pos` does not** (or move not applied). This is the same **Bucket A** class documented in `docs/oracle_exception_audit/phase7_drift_triage.md` and the **nested Fire / path-end** story in `docs/oracle_exception_audit/phase8_lane_g_drift_fix.md`. It is **not** the Lane 10A hypothesis (“air-unit pathing parity in `compute_reachable_costs`”) unless that lane deliberately broadens to **all** unit classes and **all** movement modes.

2. **BLACK_BOAT (1 row):** **Separate (F)** — no position drift; **no weapon** in damage matrix → **oracle / replay classification** issue, not terrain or CO MP.

3. **Classes B, C, D:** Not adopted as **primary** labels for any row on this slice. Sampled failures had **no SCOP** (e.g. Grimm mirror **1634717** at failure). **1630188** has **Adder COP** active on P1 at failure; could warrant a **C** spot-check if MP math is ever suspect, but the visible signature remains **`unit_pos` vs `from`** like other **E** rows.

## Recommended fix scope (for planners)

| Class | Scope |
|-------|--------|
| **E (14 rows)** | **Phase 11 / Fire lane:** nested **Move** commit before `_apply_attack`, reachability vs zip, and continuation of Lane G / Lane L themes — **not** B_COPTER-only air tiles. Optional: prove **B** on a failing tile only if `compute_reachable_costs` omits a waypoint that AWBW allows ([terrain costs](https://awbw.fandom.com/wiki/Terrain)). |
| **F (1626642)** | **Lane 10B / oracle:** map AWBW **Black Boat** “Fire” or combat payloads to **REPAIR** / non-attack handling when the unit has **no** base damage vs the defender; do **not** spend air-pathing budget here. |

## References

- `docs/oracle_exception_audit/phase7_drift_triage.md` — Bucket A framework  
- `docs/oracle_exception_audit/phase8_lane_g_drift_fix.md` — nested Fire / path-end  
- `docs/oracle_exception_audit/phase9_lane_m_capt_join_supply.md` — CO +1 movement parity example; **1634561** cited  
- `logs/desync_regression_log.md` — Phase 6 Manhattan / **orchestrator footnote** (do not revert Chebyshev)  
- `engine/action.py` — `compute_reachable_costs`, `get_attack_targets` (Manhattan range; Black Boat **REPAIR** branch)

---

*Campaign closed for Phase 10D recon; fixes deferred to Lane 10A / 10B / Phase 11.*
