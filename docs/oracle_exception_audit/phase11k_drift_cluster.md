# Phase 11K-DRIFT-CLUSTER — 200-game silent snapshot drift

## Section 1 — Sample methodology and breakdown

| Item | Value |
|------|--------|
| Source register | `logs/desync_register_post_phase10q.jsonl` |
| Filter | `class == ok` (680 rows) |
| Sample size | 200 |
| RNG | Stratified sampling with `random.Random(42)` per cell |
| Stratification | Tier × envelope length (short ≤21 / mid 22–29 / long ≥30 envelopes), proportional allocation + largest-remainder to hit exactly 200 |
| Comparator | Same as Phase 10F: `compare_snapshot_to_engine` on gzipped PHP lines after each `p:` envelope (`tools/replay_state_diff` / `tools/replay_snapshot_compare`), per-game `random.seed(_seed_for_game(CANONICAL_SEED, games_id))` |
| Drill harness | `tools/_phase11k_drill.py` → `_drill_one_clean` |

**Tier note:** The post-10Q `ok` pool has **no T0** rows (only T1–T4). Stratification used T1–T4 × three length buckets.

**Envelope tertiles (global, for bucket boundaries):** low ≤21, mid 22–29, long ≥30 (`logs/phase11k_sample_gids.summary.json`).

**Sample splits:**

| Tier | Count (200) |
|------|-------------:|
| T1 | 31 |
| T2 | 53 |
| T3 | 60 |
| T4 | 56 |

| Length bucket (envelopes_total) | Count |
|---------------------------------|------:|
| short (≤21) | 71 |
| mid (22–29) | 62 |
| long (≥30) | 67 |

**Artifacts**

- Gid list: `logs/phase11k_sample_gids.txt`
- Stratification metadata: `logs/phase11k_sample_gids.summary.json`
- Per-game drill: `logs/phase11k_drift_data.jsonl` (200 lines, one JSON object per game)
- Drill meta (thresholds, stratification): `logs/phase11k_drift_data.meta.json`

**HP rule (drill):** internal HP delta ≥10 vs PHP `hit_points`×10 on tile-matched, type-aligned units (`HP_INTERNAL_THRESHOLD = 10` in `_phase11k_drill.py`). Independent “first HP” indices are only meaningful while the replay still applies oracle actions; the primary Phase 10F parity metric is **first combined snapshot mismatch** (`silent_drift`).

---

## Section 2 — Per-game drift summary

**Definitions**

- **`silent_drift`:** First `compare_snapshot_to_engine` mismatch (same stopping criterion as `tools/replay_state_diff.run_zip` / Phase 10F). **149 / 200 (74.5%)** games show drift.
- **`clean_snapshot_through_stop`:** No snapshot mismatch on any compared frame before the replay loop stopped — **51 / 200**.
- **`replay_truncated`:** Engine reached terminal state (`state.done`) before all envelopes were applied — common on long exports; **not** treated as an oracle exception for this audit.

**Compact table — first 30 games (sorted by `games_id` in CSV; table order matches CSV sort)**

| games_id | tier | map_id | silent_drift | first_step | class | cluster | active_CO_id | last_action |
|----------|------|--------|--------------|------------|-------|---------|--------------|-------------|
| 1609626 | T4 | 77060 | Y | 19 | funds | C2 | 22 | End |
| 1615231 | T2 | 173170 | Y | 10 | funds | C2 | 9 | End |
| 1618984 | T3 | 180298 | Y | 5 | funds | C2 | 1 | Capt |
| 1619108 | T3 | 77060 | Y | 20 | funds | C2 | 5 | End |
| 1619695 | T1 | 146797 | Y | 14 | funds | C2 | 12 | End |
| 1619803 | T3 | 133665 | Y | 18 | funds | C2 | 1 | End |
| 1620039 | T4 | 123858 | Y | 12 | funds | C1 | 22 | End |
| 1620188 | T3 | 170934 | Y | 13 | funds | C2 | 16 | End |
| 1620301 | T1 | 123858 | Y | 14 | funds | C2 | 8 | End |
| 1620558 | T2 | 133665 | Y | 16 | funds | C1 | 5 | End |
| 1621299 | T3 | 126428 | N | — | none | — | — | — |
| 1621507 | T4 | 140000 | Y | 18 | funds | C2 | 22 | End |
| 1621707 | T1 | 69201 | Y | 17 | funds | C2 | 12 | End |
| 1621898 | T1 | 69201 | Y | 20 | funds | C2 | 27 | End |
| 1621999 | T3 | 166877 | Y | 18 | funds | C2 | 1 | End |
| 1622452 | T2 | 171596 | Y | 31 | funds | C1 | 7 | End |
| 1622501 | T3 | 133665 | Y | 13 | funds | C2 | 28 | End |
| 1622528 | T1 | 166877 | Y | 21 | funds | C1 | 30 | End |
| 1623014 | T1 | 166877 | Y | 16 | funds | C2 | 30 | End |
| 1623067 | T4 | 170934 | N | — | none | — | — | — |
| 1623121 | T4 | 77060 | N | — | none | — | — | — |
| 1623193 | T4 | 146797 | Y | 19 | funds | C2 | 20 | End |
| 1623484 | T4 | 123858 | Y | 14 | funds | C2 | 22 | End |
| 1623588 | T4 | 171596 | Y | 25 | funds | C2 | 20 | End |
| 1624026 | T4 | 140000 | Y | 23 | funds | C2 | 22 | End |
| 1624141 | T2 | 134930 | Y | 23 | funds | C2 | 8 | End |
| 1624316 | T3 | 134930 | Y | 19 | funds | C2 | 1 | End |
| 1624489 | T4 | 171596 | N | — | none | — | — | — |
| 1624635 | T4 | 166877 | Y | 15 | hp | C3 | 11 | End |
| 1624668 | T2 | 180298 | N | — | none | — | — | — |

**Full 200 rows:** `logs/phase11k_drift_summary.csv` (same columns + `co_p0_id`, `co_p1_id`, `replay_truncated`).

**Raw JSONL:** `logs/phase11k_drift_data.jsonl` (includes `first_mismatch_lines`, `events`, `first_funds_step`, `first_hp10_step`, `first_count_step`).

---

## Section 3 — Cluster definitions and sizes

Clusters are **rule labels** from the first combined mismatch lines (`first_mismatch_lines`), not an ML clustering.

| ID | Name | Rule | Approx. size (of 200) |
|----|------|------|----------------------:|
| **C1** | Income-boundary / bilateral treasury | First mismatch lines include **both** `P0 funds` and `P1 funds` | **33** |
| **C2** | Funds-primary (unilateral or funds+hp on same frame) | `classify` → `funds`, not bilateral by rule above | **105** |
| **C3** | HP-primary | First combined mismatch classified as **hp** (funds lines absent in classifier’s fund regex pass — typically HP-first or HP-only in combined compare order) | **11** |
| **C4** | Position / unit-count | `unit tile set mismatch` or structural unit set issues as **first** line | **0** in this 200-draw (still tracked via `first_count_step` in JSONL) |
| **C5** | Comparator / naming | `type engine=` / Missile alias class | **1** (overlaps C2/C3 if funds also present; rare) |

**Cross-check:** 33 + 105 + 11 = **149** drifting games. **138** of 149 have **funds** as the primary classifier bucket; **11** are **hp**-primary.

**Action tail:** At the first mismatch envelope, **`End`** (end-of-half-turn) dominates last-action counts — consistent with snapshot compare running **after** the full envelope is applied. Use mismatch lines + CO/map for root-cause work, not last action alone.

---

## Section 4 — CO and map heatmap

Among games with **`silent_drift == true`** (149 games), **active CO** = `events.first_full_snapshot.active_co_id` (engine seat’s CO after the envelope).

**Top active CO ids (counts)**

| co_id | name (from `data/co_data.json`) | count |
|------:|---------------------------------|------:|
| 1 | Andy | 20 |
| 22 | Jake | 14 |
| 8 | Sami | 14 |
| 11 | Adder | 14 |
| 28 | Rachel | 12 |
| 9 | Olaf | 11 |
| 12 | Hawke | 8 |
| 16 | Lash | 8 |

**Top map_id (counts)**

| map_id | count |
|-------:|------:|
| 171596 | 17 |
| 166877 | 14 |
| 180298 | 13 |
| 133665 | 13 |
| 173170 | 12 |
| 146797 | 12 |
| 170934 | 12 |
| 140000 | 12 |

Map **171596** (“Designed Desires” in `data/gl_map_pool.json`) is the single hottest map in this draw; **Andy (1)** is the hottest active CO at first mismatch.

---

## Section 5 — Phase 10F 50-game before / after

Phase 10F used `logs/desync_register_post_phase9.jsonl`, seed **1**, `n=50` (`docs/oracle_exception_audit/phase10f_silent_drift_recon.md`).

This phase re-ran **`_drill_one_clean`** on the **same 50 games** as `logs/phase10f_silent_drift.jsonl` and compared to stored `snapshot_diff_ok` / `first_step_mismatch`.

**Result:** **39 / 50** had drift in 10F and **39 / 50** have `silent_drift` now; **11 / 50** clean in both. **Zero regressions:** no game flipped from drift→clean or clean→drift on this metric.

| Metric | Phase 10F | Phase 11K (same 50 gids) |
|--------|-----------|---------------------------|
| Drift count | 39 | 39 |
| Clean count | 11 | 11 |
| First-step agreement | — | all 50 match `first_step_mismatch` where both defined |

Detailed rows: `logs/phase11k_phase10f_before_after.json`.

**Interpretation:** Phase 11A (Hachi 90% shipped, Kindle rolled back) did **not** change the PHP-vs-engine snapshot classification on the frozen 10F sample; the dominant remaining gap is still **economy + combat coupling**, not a single CO build-cost fix.

---

## Section 6 — Top 5 fix lanes (ranked backlog)

| Rank | Lane ID | Cluster | Est. games closed (campaign order-of-magnitude) | Files / areas | Complexity | Est. LOC |
|------|---------|---------|-----------------------------------------------------|---------------|------------|----------|
| 1 | **11Y-INCOME-BOUNDARY** | C1 bilateral funds vs gzip cadence | **15–35** (of ~33 in C1) | `tools/replay_snapshot_compare.py`, `tools/replay_state_diff.py`, possibly `engine/game.py` turn/income ordering *after* export spec is fixed | **HIGH** | 80–200 |
| 2 | **11Y-FUNDS-ORACLE-SLACK** | C2 funds (oracle no-op build/repair vs zip) | **20–50** (subset of ~105) | `tools/oracle_zip_replay.py`, `engine/game.py` `_apply_build` / repair, `oracle_strict` paths | **MED** | 40–120 |
| 3 | **11Y-HP-COMBAT** | C3 HP-primary | **~11** (upper bound) | `engine` combat / luck, `oracle_state_sync`, `tools/oracle_zip_replay.py` combat | **MED** | 60–150 |
| 4 | **11Y-CO-INCOME-CANON** | C2/C1 (residual D2D income gaps) | **10–25** (overlap with C1/C2) | `engine/game.py::_grant_income`, `data/co_data.json` vs live PHP evidence | **MED** | 30–80 |
| 5 | **11Y-COMPARATOR-ALIAS** | Missiles / naming | **1–3** | `tools/replay_snapshot_compare.py` | **LOW** | 5–20 |

---

## Section 7 — Verdict

| Metric | Phase 10F (50, seed 1) | Phase 11K (200, seed 42 stratified) |
|--------|------------------------|-------------------------------------|
| Silent snapshot drift rate | **78%** (39/50) | **74.5%** (149/200) |

**Conclusion:** The 200-game draw is **statistically aligned** with the 10F baseline (difference within sampling noise). **`ok` still does not imply PHP snapshot parity.**

**Closable games (rough):** With a **perfect** income-boundary story plus **oracle slack** hardening and **combat HP** alignment, a **40–90** game band of this 200 might flip to clean — **not** the full 149 without addressing bilateral cadence (C1) and combat coupling (C3).

**Next orders:** (1) lock export semantics for gzip line vs engine income with one ground-truth doc (Replay Player / PHP). (2) treat C1 as **HIGH** risk — wrong fix masks real economy bugs. (3) keep Hachi regression tests; Kindle stays **evidence-gated** per Phase 11A.

---

*“In God we trust; all others bring data.”* — W. Edwards Deming (attributed; management-quality aphorism, late 20th c.)  
*Deming: American statistician and management thinker known for quality-control and continuous improvement.*
