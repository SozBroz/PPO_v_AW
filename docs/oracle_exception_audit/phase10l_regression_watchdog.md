# REGRESSION DETECTED — full pytest — caused by mixed / workspace drift + Phase 10A ammo semantics — recommend triage per failure cluster (fix tests vs revert 10A MG patch for oracle test only)

**Phase 10L — Regression Watchdog** (delayed omnibus). Run date: **2026-04-21**.

---

## ESCALATION (orchestrator-first)

| Field | Value |
|--------|--------|
| **Verdict** | **RED** |
| **Failing suite** | **Full `pytest`** — **11 failed**, 455 passed (vs Phase 9 floor **261 passed, 0 failed**; collection is larger now, but **any failure vs 0-fail floor is a regression**). |
| **Suspected lanes** | **Phase 10A** (`engine/game.py` MG ammo gate — directly implicates `test_oracle_zip_replay.py::...test_picks_nearest_attacker_to_zip_anchor_when_ambiguous`). **STEP-GATE / legality surface** failures cluster across prune, build guards, Black Boat repair, trace replay — may be **pre-existing branch drift** or interaction with uncommitted multi-lane edits; **not** all attributable to a single lane without bisect. |
| **Recommendation** | 1) **Bisect** from Phase 9–green commit to HEAD on the 11 tests. 2) For **tank vs infantry** oracle test: update expectation to **MG = no primary ammo decrement** (10A canon) **or** narrow 10A ammo gate if test intent is primary-fire-only. 3) For **STEP-GATE** failures: confirm whether `get_legal_actions` vs `step` contract changed; fix tests to use `oracle_mode=True` where they craft illegal-mask actions, or restore mask behavior if engine regressed. 4) **Append `logs/desync_regression_log.md`** with Phase 10A + 10L gate results so the canonical log matches shipped lanes. |
| **Fuzzer** | **Skipped** (N=1000) because full pytest already RED — per Phase 10L charter. Re-run after pytest is green. |

---

## Pre-flight inventory

| Check | Result |
|--------|--------|
| `phase10a_b_copter_pathing.md` | **Present** after wait-protocol poll (~10 min; file appeared during poll). |
| `logs/desync_regression_log.md` Phase 10A entry | **Absent** — log still ends at Phase 9 / Phase 10 queue; **documentation gap**. |
| Phase 10 lane reports present (`docs/oracle_exception_audit/phase10*.md`) | **10A, 10B, 10C, 10D, 10E, 10F, 10G, 10H, 10I** (no **10J** doc in tree at run time). |
| `git status` | Dirty `main` vs `origin/main`: extensive modified/untracked files (`engine/game.py`, `engine/action.py`, `data/damage_table.json`, `tools/oracle_zip_replay.py`, many tests, etc.). Watchdog is **read-only** on source; results reflect **current working tree**. |

**10A lane summary (from doc):** damage-table fills (B_COPTER vs LANDER/BLACK_BOAT, RECON vs copters); **`_apply_attack`** skips primary ammo decrement when secondary MG applies (Infantry/Mech defenders); **no** `oracle_zip_replay.py` change. Targeted pytest in doc reports full sweep **225 passed / 1 failed** on their machine; this workspace **full** run shows **11** failures — **worse than 10A’s reported single failure**, so additional drift is present outside 10A’s documented scope.

---

## Per-suite delta table

| Suite | Phase 9 floor | Phase 10L result | Delta | Verdict |
|--------|----------------|------------------|-------|---------|
| Full pytest | 261 passed, 0 failed | 455 passed, **11 failed**, 5 skipped, 2 xfailed, 3 xpassed | **+11 failures** | **RED** |
| Negative legality | 44 passed, 3 xpassed, 0 failed | 44 passed, 3 xpassed, 0 failed | None | **GREEN** |
| Property equivalence | 1 passed | 1 passed | None | **GREEN** |
| Andy SCOP | 2 passed (Lane M) | 2 passed | None | **GREEN** |
| Fuzzer N=1000 | 0 defects | *Skipped* (pytest RED) | — | **SKIPPED** |
| Desync audit (sample) | 25 ok stay ok; 25 non-BC `engine_bug` no worse | 24/24 **ok→ok** (1 ok sample gid **1635245** missing zip on disk); 16/16 **engine_bug→engine_bug** | No class worsening on audited rows | **GREEN** (sample only) |

**Campaign audit headline (741 games)** was **not** re-run; sample does **not** prove 627/51/63 floor — only that sampled **ok** rows did not flip on re-audit and non–B_COPTER **engine_bug** rows did not worsen.

---

## RED — failure clusters (traceback pointers)

Full tracebacks: **`logs/phase10l_pytest.log`**.

| Test | Mechanism | Suspected attribution |
|------|-----------|------------------------|
| `test_picks_nearest_attacker_to_zip_anchor_when_ambiguous` | `AssertionError: 9 not less than 9` — **tank ammo not decremented** after Fire vs Infantry | **Phase 10A** MG secondary ammo accounting |
| `test_cargo_dies_with_lander` | `IllegalActionError` STEP-GATE — battleship attack at Manhattan 1 | **Known** from 10A doc as fixture/STEP-GATE issue; still **fails** here |
| `test_step_accepts_*_wait_on_*_city` (×2) | `IllegalActionError` — handcrafted `WAIT` not in mask | STEP-GATE / action–mask contract |
| `test_black_boat_repair` (×2) | REPAIR offered or not in legal set vs test expectation | Legality / repair rules |
| `test_build_guard` (×2), `test_naval_build_guard` | `IllegalActionError` on crafted `BUILD` | STEP-GATE / build legality |
| `test_trace_182065_seam_validation` (×2) | `IllegalActionError` on `BUILD` / `END_TURN` during trace replay | Export/replay vs current legality |

---

## Audit sample — flip table

**Seeds:** `random.seed(0x504831304C)` for gid sampling from `logs/desync_register_post_phase9.jsonl`.

**OK cohort (25 drawn):**

- **1635245:** not in `logs/phase10l_audit_sample.jsonl` — **`replays/amarriner_gl/1635245.zip` missing** on disk → not a class flip, **infrastructure gap**.
- **Remaining 24:** **ok → ok**.

**Non–B_COPTER `engine_bug` cohort:** only **16** rows exist in Phase 9 register (not 25). All **16** re-audited as **`engine_bug`** (no improvement to `ok` in this sample; **no** worsening to `oracle_gap`).

---

## Artifacts

| Artifact | Path |
|----------|------|
| Pytest | `logs/phase10l_pytest.log` |
| Negative | `logs/phase10l_neg.log` |
| Property | `logs/phase10l_property_equiv.log` |
| Andy SCOP | `logs/phase10l_andy_scop.log` |
| Audit sample register | `logs/phase10l_audit_sample.jsonl` |
| Audit sample run | `logs/phase10l_audit_sample_run.log` |
| Structured stats | `logs/phase10l_regression.json` |

---

## Final verdict

**RED.** Manhattan / negative-legality / property-equivalence / Andy SCOP / sampled re-audit **held**, but **full pytest does not** — **11** failures. Phase 10 is **not** regression-clean until full pytest is restored to **0 failures** (and the regression log is brought current).

**Next watchdog:** after fixes, re-run **full pytest**, then **N=1000 fuzzer** (seed **1**, `--max-days 30` to match `logs/fuzzer_run_n1000_post_phase6.jsonl`), and optionally full **741-game** `desync_audit` to confirm **627/51/63** floor or document new baseline.

---

*"In any moment of decision, the best thing you can do is the right thing, the next best thing is the wrong thing, and the worst thing you can do is nothing."* — attributed to Theodore Roosevelt (26th U.S. President, early 20th c.)  
*Roosevelt: U.S. President and reformer; quote often cited on executive judgment under uncertainty.*
