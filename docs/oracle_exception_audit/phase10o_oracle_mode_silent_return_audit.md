# Phase 10O — Engine `game.py` `_apply_*` silent return audit (`oracle_mode=True` path)

**Campaign:** `desync_purge_engine_harden`  
**Mode:** read-only investigation (no edits to `engine/game.py` or other source).  
**Scope:** `engine/game.py` only — every `def _apply_*` reachable from `GameState.step`.

---

## Executive summary

| Metric | Value |
|--------|------:|
| **`_apply_*` functions in `game.py`** | **12** |
| **Functions with no silent early-exit** | **3** (`_apply_weather_from_power`, `_apply_power_effects`, `_apply_capture`) |
| **Distinct SUSPECT guard branches** | **24** (see `logs/phase10o_oracle_mode_audit.json`) |
| **JUSTIFIED defensive / documented behaviors** | **4** (seam damage coercion ×2, seam routing, repair resupply semantics) |

### How `oracle_mode` interacts with `_apply_*`

`oracle_mode` is **only** read in `GameState.step()` (lines 248–256) to **skip** the `IllegalActionError` STEP-GATE. **No `_apply_*` method takes `oracle_mode`.** When the gate is bypassed, any constructed `Action` reaches the same `_apply_*` code paths as a hand-crafted action. **Silent `return` branches therefore matter most when the legal-action mask was not consulted** — exactly the oracle / export scenario Phase 10G flagged as HIGH-risk #3.

**Oracle callsite:** `tools/oracle_zip_replay.py::_engine_step` → `state.step(act, oracle_mode=True)` (lines 91–101).

---

## Per-function summary table

| Function | Line range | Silent returns / no-op exits | JUSTIFIED | SUSPECT |
|----------|------------|------------------------------|----------:|--------:|
| `_apply_weather_from_power` | 498–511 | 0 | 0 | 0 |
| `_apply_power_effects` | 513–602 | 0 | 0 | 0 |
| `_apply_attack` | 608–764 | 1 semantic no-damage path | 0 | 1 |
| `_apply_capture` | 787–866 | 0 (failures **raise**) | 0 | 0 |
| `_apply_wait` | 872–928 | 1 | 0 | 1 |
| `_apply_dive_hide` | 930–977 | 2 | 0 | 2 |
| `_apply_repair` | 1002–1076 | 5 | 0 | 5 |
| `_apply_seam_attack` | 1082–1168 | 0 skips (coercions only) | 2 | 0 |
| `_apply_load` | 1174–1212 | 1 | 0 | 1 |
| `_apply_join` | 1218–1265 | 1 | 0 | 1 |
| `_apply_unload` | 1271–1345 | 7 | 0 | 7 |
| `_apply_build` | 1351–1412 | 6 | 0 | 6 |
| **Totals** | | **24** | **2** | **24** |

Notes:

- **`_apply_attack`:** Missing attacker always **raises** (`ValueError`). The only listed “silent” concern is **empty defender, seam not applied** → `_finish_action` without combat (642–650) — still consumes the unit’s turn after `_move_unit`.
- **`_apply_repair`:** One branch (`1021–1023`) **`return`s without `_finish_action`** (worst case for stage drift). Other branches call `_finish_action` but skip heal/resupply — still **SUSPECT** vs AWBW.
- **`_apply_seam_attack`:** `dmg is None` → `0` (1118–1119, 1141–1142) rated **JUSTIFIED** per Phase 10G S22 — not a full action skip.

Machine-readable detail: `logs/phase10o_oracle_mode_audit.json`.

---

## Classification criteria (from Lane K + this lane)

- **JUSTIFIED:** Behavior matches documented replay/engine policy (defensive coercion, intentional wiki-aligned resupply), or the path still commits a consistent state change with logging where required.
- **SUSPECT:** Early exit **without raise** that can **hide** engine↔AWBW disagreement when STEP-GATE is off — especially **missing units**, **funds/build**, **occupancy**, **adjacency**, **cargo state**.

Subtypes used in the JSON: **A** position drift, **B** missing unit, **C** funds mismatch, **D** action stage / type mismatch.

---

## All SUSPECT entries (abbreviated)

Full rows: `logs/phase10o_oracle_mode_audit.json` → `suspect_patterns[]`.

| ID | Function | Lines | Subtype | What it masks |
|----|----------|-------|---------|----------------|
| S10O-01 | `_apply_wait` | 874–875 | B | Missing unit at `unit_pos` — full no-op |
| S10O-02 | `_apply_dive_hide` | 933–934 | B | Missing unit |
| S10O-03 | `_apply_dive_hide` | 935–936 | D | Unit cannot dive — wrong type resolution |
| S10O-04 | `_apply_repair` | 1021–1023 | B | Not a Black Boat / no unit — **no `_finish_action`** |
| S10O-05 | `_apply_repair` | 1029–1031 | D | No `target_pos` — ends turn without repair |
| S10O-06 | `_apply_repair` | 1033–1036 | B | Bad/missing repair target |
| S10O-07 | `_apply_repair` | 1037–1042 | D | Self-target guard |
| S10O-08 | `_apply_repair` | 1047–1049 | A | Not Manhattan-adjacent |
| S10O-09 | `_apply_load` | 1177–1178 | B | Missing mover or transport |
| S10O-10 | `_apply_join` | 1229–1230 | B/C | No merge — **gold overflow not applied** |
| S10O-11..17 | `_apply_unload` | 1287–1323 | A/B/D | Transport/cargo/drop/terrain guards |
| S10O-18..23 | `_apply_build` | 1354–1382 | A/C/D | Illegal factory, producibility, **funds**, occupancy |
| S10O-24 | `_apply_attack` | 642–650 | B | Fire into empty non-seam — no damage |

---

## Phase 10F funds-drift cross-reference (39 / 50 sample)

For each SUSPECT row, expected causal link to **Phase 10F** silent snapshot drift (`phase10f_silent_drift_recon.md`):

| Pattern group | drift `Y` / `N` / `MAYBE` | Comment |
|---------------|---------------------------|---------|
| `_apply_build` insufficient funds / occupancy / factory (S10O-19–23) | **Y** | Directly skips treasury spend; aligns with “funds-first” mismatch and `Build no-op` narrative in engine comments (`_grant_income`, `_apply_build`). |
| `_apply_join` silent (S10O-10) | **Y** | Merge **credits funds**; skip loses +gold vs PHP. |
| `_apply_repair` partial skips (S10O-04–08) | **MAYBE** | Heal costs and resupply affect funds/HP lines on same step as 10F rows. |
| `_apply_wait` / `_apply_load` / `_apply_unload` / `_apply_attack` empty | **MAYBE** | Mostly position/stack; can cascade into economy on later envelopes. |
| `_apply_dive_hide` wrong-type return | **N** | Unlikely primary driver of funds-first drift. |

**Bottom line:** Tightening **`_apply_build`** and **`_apply_join`** silent paths is the strongest lever to **surface** the kind of drift Phase 10F measured — not because every silent return *caused* drift, but because **skipped gold mutations** line up with **funds** and **merge overflow** semantics.

---

## Top 10 Phase 11 tightenings (prioritized)

| Rank | Target | Impact | Tighten risk | Recommended patch shape |
|------|--------|--------|--------------|-------------------------|
| 1 | `_apply_build` (all `return` 1354–1382) | **Highest** — direct funds/build parity | **Medium** — may break replays already diverged | `oracle_strict` or audit-only: `ValueError` with reason enum (`insufficient_funds`, `occupied`, …). |
| 2 | `_apply_join` (1229–1230) | **High** — funds from merge | **Low** | Raise if mover/partner missing under strict oracle. |
| 3 | `_apply_repair` (1021–1023) | **High** — full no-op without `_finish_action` | **Low** | Raise if `ActionType.REPAIR` but unit is not Black Boat / missing. |
| 4 | `_apply_wait` (874–875) | **Medium** | **Medium** | Raise on missing unit in strict mode; risk: drift-recovery no-ops. |
| 5 | `_apply_load` (1177–1178) | **Medium** | **Medium** | Raise with both positions. |
| 6 | `_apply_repair` (1033–1049) | **Medium** | **Medium** | Differentiate “illegal” vs “finish anyway” for oracle audit. |
| 7 | `_apply_unload` (collective) | **Medium** | **High** | Per-branch typed errors; unload staging is fragile. |
| 8 | `_apply_dive_hide` (933–936) | **Low–Medium** | **Medium** | Raise missing/wrong-type for strict oracle. |
| 9 | `_apply_attack` empty non-seam (642–650) | **Medium** (HP) | **High** | Strict raise if no seam and no defender — verify AWBW never emits. |
| 10 | **Meta:** `step(..., oracle_strict=True)` | **Cross-cutting** | **Low** | Thread flag into `_apply_*` only for audit/export — keep RL default unchanged. |

### Risk notes (top 10)

- **Build:** Legitimate “skip” might exist if engine is already wrong — tightening **surfaces** failure earlier; may require **oracle funds sync** before BUILD envelopes.
- **Join/Unload:** Multi-step AWBW sequences may differ in staging; **raises** need golden replay verification.
- **Attack empty tile:** Could reject obscure but legal AWBW outcomes — **verify** before hard raise.

---

## Relation to Phase 10G (`engine/game.py` 42 / 14)

Lane G’s static sweep mixed **exception handlers** and **silent control flow** across `engine/game.py`. This lane **narrows** to `_apply_*` only and **grounds** each branch in line-level semantics. Counts will not match 10G’s “42 patterns” 1:1 — this report is the **authoritative `_apply_*` catalog** for Phase 11 cleanup.

---

## Artifacts

| Artifact | Purpose |
|----------|---------|
| `logs/phase10o_oracle_mode_audit.json` | Structured per-pattern data, Phase 11 top-10, JSON refs |
| This file | Human briefing + tables |

---

## Return summary (for parent agent)

- **`_apply_*` functions audited:** **12**
- **JUSTIFIED (defensive / documented):** **3–4** (seam coercion ×2 + seam terminator + repair resupply note)
- **SUSPECT guard branches:** **24**
- **Top Phase 11 targets:** **`_apply_build`**, **`_apply_join`**, **`_apply_repair` (1021–1023)**, then missing-unit paths (`WAIT`/`LOAD`), then **`_apply_unload`**, **`_apply_attack`** empty tile
- **Phase 10F funds drift (39/50):** Tightening **BUILD** and **JOIN** silent returns is **most likely** to **catch or explain** funds-class drift; other SUSPECT paths are **MAYBE** (cascade) or **N** (niche)

---

*“The price of greatness is responsibility.”* — Winston Churchill, speech (1940s)  
*Churchill: British Prime Minister during the Second World War; often quoted on leadership under accountability.*
