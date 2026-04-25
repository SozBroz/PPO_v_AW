# Phase 11K-C2-DRILL — End-turn funds cluster (read-only)

Drill window: **Phase 11K-DRIFT-CLUSTER** cluster **C2** (“funds-primary”, not bilateral C1). Goal: per-game boundary facts, sub-cluster labels, fix-lane ranking, and cross-references to **11Y / 11M / 11STATE** without editing engine, oracle, tests, or production data.

**Harness:** `tools/_phase11k_c2_drill.py` (writes `logs/phase11k_c2_gids.txt`, `logs/phase11k_c2_drill.jsonl`, `logs/phase11k_c2_drill.csv`).

---

## Section 1 — C2 definition + drill methodology

**C2 (from Phase 11K):** `classify → funds`, and **not** C1. C1 is defined when the first combined mismatch lines include **both** `P0 funds` and `P1 funds`. C2 is the remaining funds-primary bucket (**105 / 200** in the stratified sample).

**Drill methodology (this phase):**

1. **Gid list:** All rows in `logs/phase11k_drift_summary.csv` with `cluster` starting with `C2` → **105** games (`logs/phase11k_c2_gids.txt`).
2. **First drift step:** `first_step` from the summary CSV, with JSONL fallback `first_snapshot_mismatch_step` from `logs/phase11k_drift_data.jsonl`.
3. **Oracle replay:** Same seed contract as Phase 11K (`tools.desync_audit.CANONICAL_SEED`, `tools.desync_audit._seed_for_game`), `make_initial_state`, `apply_oracle_action_json` for envelopes **0 … first_step** inclusive (matches `tools/_phase11k_drill.py::_drill_one_clean`).
4. **Boundary instrumentation** at the first drift envelope:
   - **PHP funds** immediately before that envelope: gzip frame index `first_step` (same index `_phase11k_drill` uses for “before” in its event record; post-mismatch compare uses `frames[first_step+1]`).
   - **Engine funds** before applying envelope `first_step` vs after completing it; **PHP funds** after from `frames[first_step+1]`.
   - **Active CO** at boundary: `state.co_states[state.active_player].co_id` after the envelope (matches 11K `events.first_full_snapshot.active_co_id` intent).
   - **Last meaningful action:** last JSON action in the envelope that is not `End` (if only `End`, recorded as `End`).
5. **Drift sign / magnitude:** Parsed from `first_mismatch_lines[0]` in the JSONL (`engine` vs `php_snapshot`); magnitude bucket: small **&lt;100**, medium **100–1000**, large **&gt;1000** G absolute delta on that line.

**Coverage:** Full **105 / 105** C2 games drilled successfully (no oracle abort in this harness on these rows).

---

## Section 2 — Per-game drill table (first 30 games + full CSV)

**First 30 C2 games (CSV row order = `games_id` sort):** see `logs/phase11k_c2_drill.csv` lines 2–31 (reproduced below for quick reading).

| games_id | tier | map_id | first_step | active_CO | day_field | turn_after | last_non_End | drift_sign | mag |
|---------:|------|-------:|-----------:|-----------|----------:|-----------:|--------------|------------|-----|
| 1609626 | T4 | 77060 | 19 | Jake (22) | 10 | 11 | Build | engine_gt_php | medium |
| 1615231 | T2 | 173170 | 10 | Olaf (9) | 6 | 6 | Build | engine_gt_php | medium |
| 1618984 | T3 | 180298 | 5 | Andy (1) | 3 | 3 | Capt | engine_lt_php | large |
| 1619108 | T3 | 77060 | 20 | Drake (5) | 11 | 11 | Build | engine_gt_php | medium |
| 1619695 | T1 | 146797 | 14 | Hawke (12) | 8 | 8 | Build | engine_gt_php | medium |
| 1619803 | T3 | 133665 | 18 | Andy (1) | 10 | 10 | Move | engine_gt_php | medium |
| 1620188 | T3 | 170934 | 13 | Lash (16) | 7 | 8 | Build | engine_gt_php | medium |
| 1620301 | T1 | 123858 | 14 | Sami (8) | 8 | 8 | Move | engine_gt_php | large |
| 1621507 | T4 | 140000 | 18 | Jake (22) | 10 | 10 | Capt | engine_gt_php | large |
| 1621707 | T1 | 69201 | 17 | Hawke (12) | 9 | 10 | Move | engine_gt_php | medium |
| 1621898 | T1 | 69201 | 20 | Javier (27) | 11 | 11 | Build | engine_gt_php | large |
| 1621999 | T3 | 166877 | 18 | Andy (1) | 10 | 10 | Build | engine_gt_php | large |
| 1622501 | T3 | 133665 | 13 | Rachel (28) | 7 | 8 | Build | engine_gt_php | medium |
| 1623014 | T1 | 166877 | 16 | Von Bolt (30) | 9 | 9 | Move | engine_gt_php | medium |
| 1623193 | T4 | 146797 | 19 | Grimm (20) | 10 | 11 | Build | engine_gt_php | large |
| 1623484 | T4 | 123858 | 14 | Jake (22) | 8 | 8 | Move | engine_gt_php | medium |
| 1623588 | T4 | 171596 | 25 | Grimm (20) | 13 | 14 | Move | engine_gt_php | medium |
| 1624026 | T4 | 140000 | 23 | Jake (22) | 12 | 13 | Move | engine_gt_php | medium |
| 1624141 | T2 | 134930 | 23 | Sami (8) | 12 | 13 | Build | engine_gt_php | large |
| 1624316 | T3 | 134930 | 19 | Andy (1) | 10 | 11 | Build | engine_gt_php | large |
| 1624721 | T3 | 140000 | 17 | Rachel (28) | 9 | 10 | Build | engine_gt_php | medium |
| 1624783 | T3 | 77060 | 19 | Lash (16) | 10 | 11 | Build | engine_gt_php | medium |
| 1624953 | T4 | 146797 | 18 | Grimm (20) | 10 | 10 | Build | engine_gt_php | medium |
| 1624973 | T4 | 159501 | 11 | Jess (14) | 6 | 7 | Build | engine_gt_php | medium |
| 1625211 | T3 | 171596 | 14 | Rachel (28) | 8 | 8 | Fire | engine_gt_php | medium |
| 1625657 | T2 | 133665 | 14 | Olaf (9) | 8 | 8 | Build | engine_gt_php | large |
| 1626529 | T3 | 170934 | 17 | Andy (1) | 9 | 10 | Build | engine_gt_php | large |
| 1626628 | T1 | 170934 | 16 | Hawke (12) | 9 | 9 | Build | engine_gt_php | large |
| 1626658 | T3 | 140000 | 12 | Andy (1) | 7 | 7 | Build | engine_gt_php | medium |
| 1626763 | T1 | 173170 | 19 | Von Bolt (30) | 10 | 11 | Build | engine_gt_php | medium |

**Full 105 rows:** `logs/phase11k_c2_drill.csv` (columns include per-seat PHP/engine deltas across the boundary and `first_mismatch_line0`).

---

## Section 3 — Sub-cluster definitions + sizes (105-game aggregate)

Counts below are **exclusive labels for reporting**; in reality **income timing**, **heal/repair cost**, and **oracle slack** often **co-occur** on the same envelope (Phase 10N already noted HP+funds on the same step).

| Sub-cluster | Rule (heuristic) | Size (105) | Notes |
|-------------|------------------|------------:|-------|
| **C2.1 — Engine-rich vs PHP (`engine_gt_php`)** | First mismatch line: engine funds &gt; PHP | **104** | Dominant sign; consistent with “engine already applied next income / heal charge / build” more than PHP line at compare index (see 10N **1628609**, **1620188** narratives). |
| **C2.2 — Engine-poor vs PHP (`engine_lt_php`)** | engine &lt; PHP on first funds line | **1** | **`1618984`** only: last non-`End` action **Capt**; large gap (see §4). |
| **C2.3 — Last meaningful action = Build** | | **62** | Largest action tail; aligns with spend-side + property-heal-after-build stories. |
| **C2.4 — Last meaningful action = Move** | | **32** | Second-largest; often paired with combat/position changes before `End`. |
| **C2.5 — Last meaningful action = Fire** | | **7** | Combat-adjacent treasury (Sasha War Bonds, repair coupling, etc.). |
| **C2.6 — Capt / Repair** | **Capt** or **Repair** | **2 + 2** | Rare in C2 tail but high signal when present. |
| **Magnitude** | small / medium / large | **0 / 72 / 33** | No “small” in this sample’s first-line parser (many deltas land 100–420+). |

**Active CO at boundary (top 8, 105 games):** Andy **14**, Jake **12**, Rachel **11**, Olaf **10**, Sami **9**, Adder **8**, Lash **6**, Hawke **6** (ids **1, 22, 28, 9, 8, 11, 16, 12**). Aligns with 11K heatmap (Andy / Jake / Sami / Adder / Rachel).

---

## Section 4 — Per-sub-cluster citations (chart, `co_data.json`, code, status)

**Primary rules source:** [AWBW CO Chart — co.php](https://awbw.amarriner.com/co.php).

| Sub-cluster | Chart / canon | `data/co_data.json` | Engine path | Status |
|-------------|---------------|---------------------|-------------|--------|
| **Income timing (C2.1, overlaps C1 mechanism)** | Export cadence: when PHP line records funds vs start-of-turn income | N/A | `engine/game.py` — `_end_turn` switches `active_player`, then `_resupply_on_properties`, then `_grant_income` for the new active player (see ~358–493) | **Comparator / pairing issue** as much as engine: 10N **1628609** showed pre-grant match then full-day grant vs PHP line; **implement** only after export ground truth (Replay Player / PHP) is fixed — else risk masking real economy bugs (`phase10n_funds_drift_recon.md`). |
| **Colin / Sasha D2D income** | Colin: +100 G per property; Sasha: +100 G per property (chart) | Colin/Sasha entries describe powers; D2D text may lag chart | `_grant_income`: branches for `co_id` **15** and **19** (~487–491) | **Partially implemented** for per-property +100; Sasha SCOP “War Bonds” damage→funds **not** in `_grant_income` — **11Y-CO-WAVE-2** quantifies SCOP gap. |
| **Kindle** | Chart silent on +50% city income (10T / 11A) | Older text mentioned city income; **do not** treat JSON alone as canon | `_grant_income` explicitly **does not** branch Kindle (comment ~465–477) | **Evidence-gated** — 11A rollback; live PHP vs engine reconciled against **chart**, not wiki-only. |
| **Rachel repair (+1 HP, liable for costs)** | Chart: “Units repair +1 additional HP (note: liable for costs).” | `co_data.json` Rachel block **omits** D2D repair line (only luck in `day_to_day`) | Property heal: `_resupply_on_properties` / `_property_day_repair_gold` (see 11Y §4) | **Mis-implemented vs chart** for D2D repair amount/cost — **11Y-RACHEL-IMPL** in flight. |
| **Hachi (build discount)** | Chart | **`co_data.json`:** `id` **17** = **Hachi**; `id` **22** = **Jake** (do not confuse) | `_apply_build` / unit cost | **Shipped** in campaign (Phase 11A); C2 still drifts — shows **Hachi discount alone does not clear C2**. |
| **Oracle slack (build/repair no-op)** | N/A | N/A | `_apply_build` / `_apply_repair` with insufficient funds; oracle may still advance | **Documented** 10F / 10N **1620188** | **Mis-match**: engine no-op vs zip recorded. |
| **Capture income (C2.2 single game)** | AWBW: capture completes → property ownership; income timing for **new** property | N/A | Capture path + next `_grant_income` | **1618984**: PHP **9000** vs engine **1000** on P0 at first drift — consistent with **same-day capture credit vs next-day engine income** *hypothesis*; needs a **dedicated** capture-day micro-drill (not proven here beyond one row). |

---

## Section 5 — Cross-reference: 11Y-CO-WAVE-2, 11M-WAVE2, 11STATE, 11A

| Lane | Relevance to C2 |
|------|-----------------|
| **Phase 11Y-CO-WAVE-2** (`docs/oracle_exception_audit/phase11y_co_wave_2.md`) | **Sasha** SCOP War Bonds funds gap; **Rachel** D2D repair drills on live zips; **Colin** absent from catalog. C2 heat includes **Rachel 11** games at boundary — overlaps **11Y-RACHEL-IMPL**. |
| **Phase 11M SUSPECT Wave 2** (`docs/oracle_exception_audit/phase11m_suspect_wave_2.md`) | Silent returns on **Repair** / **Unload** / **Fire** can mask treasury + HP; **indirect** driver for C2 when envelopes disagree on legality. |
| **Phase 11A (Kindle rollback / Hachi shipped)** | `engine/game.py::_grant_income` Kindle comment references 11A; C2 still **74.5%** in 11K — confirms **single-CO build/income** fixes do not clear the pool. |
| **11STATE-MISMATCH-IMPL** (`docs/oracle_exception_audit/phase11state_mismatch_design.md`) | Proposes wiring snapshot diff into `desync_audit` — **does not** replace need for **income boundary** semantics; Option B (per-envelope) matches this drill. |

---

## Section 6 — Top 5 fix lane backlog (ranked by estimated games closed)

Estimates are **order-of-magnitude** for the **C2 pool (~105)** and overlap **C1**/**export** work from Phase 11K §6.

| Rank | Lane name | Sub-cluster | Est. # games closed | Files affected | Est. LOC | In flight? |
|-----:|-----------|-------------|--------------------:|----------------|--------:|------------|
| 1 | **11Y-INCOME-BOUNDARY** + export pairing spec | C2.1 (104× `engine_gt_php`) + C1 family | **25–55** | `tools/replay_snapshot_compare.py`, `tools/replay_state_diff.py`, possibly compare hook; `engine/game.py` only after spec locked | 80–200 | **11STATE** design; comparator work |
| 2 | **11Y-FUNDS-ORACLE-SLACK** (strict / diff-only) | C2.3 Build-heavy (**62**) | **15–35** | `tools/oracle_zip_replay.py`, `engine/game.py` `_apply_build` / `_apply_repair` | 40–120 | Not in flight as default oracle |
| 3 | **11Y-CO-INCOME-CANON** + **Sasha War Bonds** | CO heat (Andy/Rachel/Sami/Kindle/Olaf…) | **10–25** | `engine/game.py` `_grant_income`, `_apply_power_effects`, combat | 40–100 | **11Y** partial (Sasha SCOP quantified) |
| 4 | **11Y-RACHEL-IMPL** (D2D +1 HP / cost) | Rachel-boundary rows (**11** in 105) + HP coupling | **5–15** of C2 (upper bound if repair is dominant in those rows) | `engine/game.py` resupply/repair | 30–70 | **In flight** per 11Y |
| 5 | **11M oracle_strict** (Repair/Unload/Fire) | C2.5 Fire (**7**), Repair (**2**), indirect | **3–10** | `engine/game.py` guarded paths | 40–80 | **11M** design ready |

---

## Section 7 — Verdict

**YELLOW**

- **GREEN:** C2 is **operationally defined**, **105/105** games drilled with structured boundary CSV; **dominant sign** (`engine_gt_php`) and **action tail** (Build/Move) are stable; primary sources (**co.php**, `co_data.json`, `game.py::_end_turn` / `_grant_income`) are citable.
- **YELLOW:** Sub-clusters **overlap** (income cadence vs heal vs oracle slack vs CO D2D); **one** `engine_lt_php` row is not enough to prove capture-day income globally; Rachel’s chart rule is **not** fully reflected in JSON.
- **RED:** Not applied — we did **not** prove a **single** numeric root cause for all 104 `engine_gt_php` rows.

---

*“In preparing for battle I have always found that plans are useless, but planning is indispensable.”* — Dwight D. Eisenhower, National Defense Executive Reserve Conference, **1957**  
*Eisenhower: Supreme Allied Commander in World War II and 34th U.S. President.*
