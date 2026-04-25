# Phase 11W-EXEC — Extras audit triage (`desync_register_extras_baseline.jsonl`)

**Mode:** read-only triage (no engine/oracle/tests/data/zip edits).  
**Input:** `logs/desync_register_extras_baseline.jsonl` from  
`python tools/desync_audit.py --catalog data/amarriner_gl_extras_catalog.json --register logs/desync_register_extras_baseline.jsonl --seed 1`  
**Context:** `docs/oracle_exception_audit/phase11w_extras_catalog_recon.md` (195 games, 14 maps, disjoint from std catalog).

---

## Section 1 — Executive summary

- **195-row class split:** **179** ok, **15** oracle_gap, **1** engine_bug; **0** loader_error, **0** replay_no_action_stream, **0** catalog_incomplete — same taxonomy as Phase 10Q; no ingest-class rows.
- **Novelty:** **0** new oracle_gap message families vs std pool (still only **Move:** truncation + **Build no-op**); **1** engine_bug, **F4** (friendly fire), same exception bucket as 10Q (`ValueError`). **0** NEW (non–F1–F5) engine shapes.
- **Fold-in:** **Recommendation A** — treat **741 + 195 = 936** as the expanded audit universe for forward-looking registers; extras behavior is in-family with 10Q (ok% within noise; oracle_gap slightly higher; engine_bug lower).

---

## Section 2 — Per-class table

Register rows use the field **`class`** (triage brief `cls`).

| Class | Count | % of 195 | Phase 10Q % (741 games) |
|-------|------:|---------:|------------------------:|
| ok | 179 | 91.8% | 91.8% |
| oracle_gap | 15 | 7.7% | 6.9% |
| engine_bug | 1 | 0.5% | 1.3% |
| loader_error | 0 | 0% | 0% |
| replay_no_action_stream | 0 | 0% | 0% |
| catalog_incomplete | 0 | 0% | 0% |

**Combined pool (informational):** 680+179 = **859** ok, 51+15 = **66** oracle_gap, 10+1 = **11** engine_bug on **936** games — matches the arithmetic of merging `logs/desync_register_post_phase10q.jsonl` with this extras register.

---

## Section 3 — `engine_bug` per-row classification (Phase 11D taxonomy)

| games_id | Unit (from message) | Message head | Classification |
|----------|----------------------|--------------|----------------|
| 1636707 | MED_TANK | `_apply_attack: friendly fire from player 0 on MED_TANK at (15, 6)` | **F4** — friendly fire / `_apply_attack` self-target family (Phase 11D) |

**F1–F5 rollup (extras, primary label):** F1 **0**, F2 **0**, F3 **0**, F4 **1**, F5 **0**, **NEW** **0**.

---

## Section 4 — `oracle_gap` message-shape histogram + comparison

**Buckets** (normalized prefixes):

| Shape | Extras (n=15) | Phase 10Q (n=51) |
|-------|---------------|------------------|
| **Move:** (`Move: engine truncated path…`) | 12 (80.0%) | 41 (80.4%) |
| **Build no-op** (`Build no-op at tile …`) | 3 (20.0%) | 10 (19.6%) |
| Other | 0 | 0 |

**Comparison to std pool:** Extras reproduce the **same two families** seen in Phase 10Q (predominantly **Move:** / upstream drift; minority **Build no-op** / funds or occupancy refusal). **No new prefix** (no `Fire:`, `Join:`, `AttackSeam:`, etc. as first-class gap lines in this slice).

**Extras `Build no-op` rows (detail):**

| games_id | map_id | gist |
|----------|--------|------|
| 1628446 | 166877 | NEO_TANK — insufficient funds |
| 1634045 | 170934 | TANK — insufficient funds |
| 1634464 | 133665 | INFANTRY — tile occupied |

---

## Section 5 — CO matchup heatmap (failures only)

Non-ok rows: **16** (15 oracle_gap + 1 engine_bug). Each row contributes one **ordered** matchup `(co_p0_id, co_p1_id)`.

**Top matchups (ties sorted by `games_id`):**

| Rank | co_p0_id | co_p1_id | Failures |
|------|----------|----------|----------|
| 1 (tie) | 30 | 30 | 2 |
| 1 (tie) | 18 | 11 | 2 |

(Remaining 12 failing matchups have count **1** each.)

**Per-CO failure involvement** (failures count a CO if it is P0 or P1 on a non-ok row; each game = two CO “half-slots”):

| co_id | Fail involvements | Appearances (half-games) | Rate |
|------:|------------------:|-------------------------:|-----:|
| 18 | 6 | 11 | 54.5% |
| 30 | 6 | 14 | 42.9% |
| 16 | 3 | 25 | 12.0% |
| 21 (Koal) | 1 | 9 | 11.1% |
| … | … | … | … |

**Global mean** involvement per CO half-slot ≈ **8.2%** (= 16 failures / 195 games, counting two COs per game as slots: 32/390).

**>2× mean (disproportionate):** **CO 18**, **CO 30** — **caveat:** low appearance counts (11 and 14); high rates may be **small-N noise** but they merit awareness in stretch-pool triage.

**Koal (CO 21):** **9** extras games include Koal; **1** failure (**1628849**, `oracle_gap` / **Move:** truncation, matchup **11 / 21**). Koal is **not** an outlier vs the ~8% baseline at this N.

**Vs recon CO frequency:** Recon top half-game counts were **11, 23, 1, 8, 14, …** — failures are **not** concentrated on the most frequent COs **11 / 23 / 1**; the hottest failure *rates* are on less frequent COs **18** and **30**.

---

## Section 6 — Map heatmap (failures only)

Per-map counts are from the **195-row register** (ground truth for this audit).

| map_id | Fails | Games | Fail rate | Notes |
|--------|------:|------:|----------:|-------|
| 69201 | 3 | 11 | **27.3%** | >2× global (~8.2%) |
| 140000 | 2 | 11 | **18.2%** | >2× global |
| 166877 | 2 | 14 | 14.3% | |
| 180298 | 2 | 15 | 13.3% | |
| 133665 | 3 | 23 | 13.0% | high fail *count*, moderate rate |
| 170934 | 1 | 13 | 7.7% | includes 1 `oracle_gap` |
| 159501 | 1 | 13 | 7.7% | |
| 146797 | 1 | 15 | 6.7% | |
| 171596 | 1 | 20 | 5.0% | |
| *others* | 0 | … | 0% | 123858, 134930, 126428, 173170, 77060 |

**Global fail rate:** 16/195 ≈ **8.2%**. **Maps >2× global:** **69201**, **140000**.

**Concentration check:** No single map holds **>50%** of all 16 failures (max share **3/16 ≈ 19%** on maps **69201** and **133665**).

---

## Section 7 — Novel findings

| Check | Result |
|--------|--------|
| `engine_bug` exception type outside {ValueError, KeyError, FileNotFoundError, IndexError} | **No** — single row is `ValueError`. |
| `engine_bug` message outside F1–F5 | **No** — **F4** friendly fire. |
| `oracle_gap` shape not seen in std pool | **No** — only **Move:** + **Build no-op**, same as Phase 10Q. |
| One CO or map **>50%** of bugs | **No**. |
| `loader_error` / `replay_no_action_stream` | **None** — ingest pipeline clean on this slice. |

**Watch items (not “novel taxonomy” but operational):** **CO 18 / 30** high failure *rates* at small N; **map 69201** and **140000** elevated failure *rates*. These are **YELLOW-bookkeeping** signals, not a new defect class.

---

## Section 8 — Combined audit recommendation

**Choice: A — Fold 195 extras into the canonical 936-game audit universe**

**Rationale:**

- Class mix matches Phase 10Q families; **no** new loader/catalog/action-stream classes.
- Oracle gaps are the **same two message species** as std pool; proportions (~80% Move, ~20% Build no-op) align with Phase 10Q.
- Single `engine_bug` is **F4**, already in the Phase 11D residual set — not a new engine species.
- **B** (separate stretch baseline) is reasonable if a team wants **std-only** CI gates forever — but analytically the campaign already treats oracle_gap as upstream/state-sync debt; extras do not change that story.
- **C** (defer) is **not** justified: no systemic new failure mode surfaced.

---

## Section 9 — Verdict

**GREEN** — Extras audit is **in-family** with Phase 10Q: same gap shapes, subsampled engine_bug consistent with known F4, no ingest failures. Fold-in **A** recommended; keep **CO 18 / 30** and **maps 69201 / 140000** on the radar for future targeted triage (small-N / rate hotspots).

---

*Phase 11W-EXEC-TRIAGE complete.*
