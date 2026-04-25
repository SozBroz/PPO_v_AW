# Phase 11J-FIRE-DRIFT — Closeout (read-only validation)

**Lane:** ENGINE (`engine/game.py::_apply_attack`) + ORACLE (`tools/oracle_zip_replay.py`) — P-COLO-ATTACKER, P-AMMO override-bypass, P-DRIFT-DEFENDER  
**Date:** 2026-04-21  
**Mode:** READ-ONLY validation + report only (no source edits in this session)  
**Hypothesis / spec:** `docs/oracle_exception_audit/phase11j_fire_drift_hypothesis.md`  
**Report shape reference:** `docs/oracle_exception_audit/phase11j_f2_koal_fix.md`

---

## Section 1 — Confirmation of three edits (Step 1)

Searched the working tree for the agreed comment markers (equivalent to grepping after `git diff HEAD`).

| Marker | Location | Edit |
|--------|----------|------|
| `Phase 11J P-COLO-ATTACKER` | `engine/game.py` (~L631–638 region before attacker resolution) | **A** — prefer `selected_unit` when it matches `action.unit_pos` |
| `Phase 11J P-AMMO` | `engine/game.py` (~L666–672, `oracle_pinned` / range-check bypass) | **B** — skip defense-in-depth range check when `_oracle_combat_damage_override` is set |
| `Fire: oracle resolved defender type` | `tools/oracle_zip_replay.py` inside `_oracle_fire_assert_attacker_can_damage_defender` (~L1139–1143) | **C** — `UnsupportedOracleAction` when `get_base_damage` is `None` |

**Result:** **3 / 3** markers present. **No escalation** — the FIRE-DRIFT bundle is present in the tree.

---

## Section 2 — Files changed (`git diff HEAD --stat`)

```
 engine/game.py             |  212 +++++-
 tools/oracle_zip_replay.py | 1651 +++++++++++---------------------------------
 2 files changed, 601 insertions(+), 1262 deletions(-)
```

**Counsel:** The `oracle_zip_replay.py` stat reflects a **large** working-tree delta vs `HEAD`, not just the surgical P-DRIFT-DEFENDER helper. The **Edit C** surface is localized (marker + `_oracle_fire_assert_attacker_can_damage_defender` and call sites). Any full merge review should treat the zip replay file as potentially carrying additional refactors beyond this lane’s three bullets.

---

## Section 3 — Code edit summaries (by marker)

### Edit A — P-COLO-ATTACKER (`engine/game.py::_apply_attack`)

After the `Phase 11J P-COLO-ATTACKER` block: resolve `attacker` from `selected_unit` when alive and `sel.pos == action.unit_pos`, else `get_unit_at(action.unit_pos)`. Addresses co-occupied attacker tile (e.g. gid **1634664** hypothesis).

### Edit B — P-AMMO override-bypass (`engine/game.py::_apply_attack`)

After the `Phase 11J P-AMMO` comment: `oracle_pinned = self._oracle_combat_damage_override is not None`; the `get_attack_targets` defense-in-depth check runs only when `defender_pre is not None and not oracle_pinned`. Intended for MG / primary-ammo=0 / oracle-pinned damage paths (gids **1622104**, **1625784**, **1630983**, **1635025**, **1635846** per hypothesis).

### Edit C — P-DRIFT-DEFENDER (`tools/oracle_zip_replay.py`)

`_oracle_fire_assert_attacker_can_damage_defender` raises `UnsupportedOracleAction` with message containing `Fire: oracle resolved defender type …` when the resolved engine defender type has no `get_base_damage` entry (gid **1631494** → `oracle_gap`).

---

## Section 4 — Seven-target audit (`tools/desync_audit.py`, seed **1**)

**Register:** `logs/desync_register_post_phase11j_fire_drift_targeted.jsonl`  
**Pre (Phase 10Q baseline):** all seven rows were `class: engine_bug` in `logs/desync_register_post_phase10q.jsonl`.

| games_id | Pre (10Q) | Post (FIRE-DRIFT validation) | §5 hypothesis expectation |
|----------|-----------|------------------------------|---------------------------|
| 1622104 | `engine_bug` | **`oracle_gap`** — Move path truncation (env ~47) | Edit B → engine fix / full replay |
| 1625784 | `engine_bug` | **`ok`** | Edit B |
| 1630983 | `engine_bug` | **`ok`** | Edit B |
| 1631494 | `engine_bug` | **`oracle_gap`** — Edit C message (FIGHTER vs TANK resolver-miss) | Edit C → `oracle_gap` |
| 1634664 | `engine_bug` | **`oracle_gap`** — Move path truncation (env ~23; envelope still Fire-heavy) | Edit A → engine fix / full replay |
| 1635025 | `engine_bug` | **`ok`** | Edit B |
| 1635846 | `engine_bug` | **`ok`** | Edit B |

**Counts:** `engine_bug` **0** / 7; **`ok`** **4**; **`oracle_gap`** **3**.

**Match to hypothesis §5:** **1631494** matches exactly (oracle re-bucket). **1625784, 1630983, 1635025, 1635846** match (full `ok`). **1622104** and **1634664** **do not** match the “six silent full replays” story: both cleared the original `_apply_attack` / friendly-fire `engine_bug` but now hit **downstream** `oracle_gap` on **Move** (path truncation). That is **not** a regression to `engine_bug`; it is **progress** with remaining oracle/path debt.

---

## Section 5 — Nine regression gates + Edit B safety

### Gates 1–6 (pytest via `python -m pytest`)

| # | Gate | Floor | Result |
|---|------|-------|--------|
| 1 | `tests/test_engine_negative_legality.py -v --tb=no` | 44p / 3xp / 0f | **44 passed, 3 xpassed, 0 failed** |
| 2 | `tests/test_andy_scop_movement_bonus.py` + `tests/test_co_movement_koal_cop.py --tb=no` | 7 passed | **7 passed** |
| 3 | `tests/test_engine_legal_actions_equivalence.py::test_legal_actions_step_equivalence --tb=no` | 1 passed | **1 passed** (~28.6s) |
| 4 | `tests/test_co_build_cost_hachi.py` + `tests/test_co_income_kindle.py` + `tests/test_oracle_strict_apply_invariants.py --tb=no` | 15 passed | **15 passed** |
| 5 | `test_oracle_zip_replay.py -v --tb=no` | (report count) | **62 passed** |
| 6 | `--tb=no -q` (full suite) | ≤2 failures (deferred trace) | **1 failed**, 493 passed, 5 skipped, 2 xfailed, 3 xpassed — failure is **`test_trace_182065_seam_validation.py::TestTrace182065SeamValidation::test_full_trace_replays_without_error`** (same deferred Sami seam lane as Koal report) |

### Gate 7 — Targeted seven-game re-audit

Done in Section 4: **0** `engine_bug`.

### Gate 8 — 100-game sample (existing register)

**File:** `logs/desync_register_post_phase11j_sample.jsonl` (100 lines).  
**`"class": "engine_bug"` count:** **0** (grep).

### Gate 9 — Fresh 50-game sample (seed **1**)

**Command:** `python tools/desync_audit.py --max-games 50 --seed 1 --register logs/desync_register_post_phase11j_fire_drift_50.jsonl`  
**Summary line:** `ok=45`, `oracle_gap=5`, **`engine_bug=0`** — meets floor **`engine_bug ≤ 0`**.

### Edit B safety (Step 4) — non-oracle / RL play

- `_oracle_combat_damage_override` is documented on `GameState` as an oracle channel, default **`None`**, cleared after consumption in `_apply_attack` (set to `None` after read).
- Production assignment located in `tools/oracle_zip_replay.py` (`_oracle_set_combat_damage_override_from_combat_info` → `state._oracle_combat_damage_override = (dmg, counter)`).
- Therefore **`oracle_pinned`** is **False** on normal `step()` ATTACK actions unless tests or oracle explicitly set the override. The range check **still runs** on non-oracle paths.

**Escalation:** None — **no leak** of Edit B into normal legality unless the override is set.

---

## Section 6 — KOAL cross-check (Step 5)

**Register:** `logs/desync_register_post_phase11j_fire_drift_koal_crosscheck.jsonl`  
**Command:** `--games-id 1605367 --games-id 1622104 --games-id 1630794 --seed 1`

| games_id | Expected | Actual | Notes |
|----------|----------|--------|-------|
| 1605367 | Still `oracle_gap` (Koal post-fix) | **`oracle_gap`** (Move truncation) | **Not regressed** to `engine_bug` |
| 1622104 | FIRE-DRIFT target | **`oracle_gap`** (Move truncation) | **Not** `engine_bug`; compatible with Koal doc note that this gid advances past old MP/fire failures |
| 1630794 | Still `engine_bug` (FU-ORACLE pending) | **`engine_bug`** (same Illegal move / Load envelope) | **Not regressed**; FIRE-DRIFT does not touch Capt seat-switch |

**Verdict:** FIRE-DRIFT and **11J-F2-KOAL** fixes are **compatible**; no KOal regression signals in this slice.

---

## Section 7 — `engine_bug` residual count

| Slice | Phase 10Q baseline | Post FIRE-DRIFT (this validation) |
|-------|-------------------|-----------------------------------|
| Full catalog row count in `desync_register_post_phase10q.jsonl` with `"class": "engine_bug"` | **10** | (unchanged file; not re-scanned full 800) |
| Seven FIRE-DRIFT targets | **7** × `engine_bug` | **0** × `engine_bug` |
| 100-game sample `post_phase11j_sample.jsonl` | — | **`engine_bug` = 0** |
| Fresh 50-game sample `post_phase11j_fire_drift_50.jsonl` | — | **`engine_bug` = 0** |

**vs 11J-F2-Koal floor:** Koal report claimed first-100 `engine_bug=0` post-Koal; the **100-row sample** here still shows **`engine_bug=0`** — **no upward movement** in that slice.

---

## Section 8 — Carry-forward debt (from hypothesis §6)

- **Ammo drift (B-COPTER):** **1625784** / **1635846** now **`ok`** in this audit; hypothesis still warns that **underlying** primary-ammo state may diverge from AWBW — monitor future **F2** state-diff audits (Phase **11K** or later).
- **F5 1626642:** explicitly **out of lane** per hypothesis; not re-run here.
- **1622104 / 1634664:** **Move** oracle_gap / path truncation — separate from the original `_apply_attack` failures; triage under Phase-9-class move/path lanes.

---

## Section 9 — Verdict

**YELLOW**

- **GREEN** criteria (“all seven targets **closed** as full **`ok`** replays, no regressions”) is **not** met: **three** targets end in **`oracle_gap`** (one intentional **1631494**, two **Move** truncations).
- **RED** criteria are **not** met: **no** `engine_bug` on the seven targets; **no** new failures in gates 1–5; gate 6 still **one** deferred trace failure only; KOal cross-check **clean**; samples show **`engine_bug=0`**.

**Net:** FIRE-DRIFT **eliminates the Phase 10Q `engine_bug` cluster on all seven gids** and passes the regression floor, but **two** gids trade an attack-classification bug for **downstream oracle move debt** instead of a clean **`ok`**.

---

## Return summary (executive)

| Metric | Value |
|--------|--------|
| FIRE-DRIFT edits confirmed | **3 / 3** |
| Seven targets with post status **`ok`** or **`oracle_gap`** (none `engine_bug`) | **7 / 7** |
| Seven targets with post status **`ok`** only | **4 / 7** |
| Nine gates | **8 PASS** at floor + **Gate 7 PASS** (0 `engine_bug` on targets); **Gate 6** **PASS** vs ≤2 failures (1 known deferred) |
| `engine_bug` in 100-row sample | **0** |
| `engine_bug` in fresh 50-game sample | **0** |
| Verdict | **YELLOW** |

---

*Phase 11J-FIRE-DRIFT closeout complete (validation-only session).*

*"The first principle is that you must not fool yourself — and you are the easiest person to fool."* — Richard Feynman, 1974 Caltech commencement address  
*Feynman: American physicist; the line is his warning about self-deception in scientific judgment.*
