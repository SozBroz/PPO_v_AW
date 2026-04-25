# Phase 8 — Lane J: `oracle_gap` classification (post–Phase 6 register)

**Campaign:** `desync_purge_engine_harden`  
**Lane:** J — read-only classification (no engine/oracle edits in this phase)  
**Inputs:** `logs/desync_register_post_phase6.jsonl`, `logs/desync_clusters_post_phase6.json`, `tools/oracle_zip_replay.py`, `logs/desync_regression_log.md` (Phase 7 orchestrator footnote on Manhattan canon)

**Artifacts:** `logs/phase8_lane_j_pull.log`, `logs/phase8_lane_j_oracle_gap_shapes.json`, `logs/phase8_lane_j_shape_drill.json`, `logs/phase8_lane_j_cluster_crosstab.json`, `tools/_phase8_lane_j_pull.py`, `tools/_phase8_lane_j_crosstab.py`

---

## Headline counts

| Metric | Value |
|--------|------:|
| Total `oracle_gap` rows | **162** |
| **GENUINE LIMITATION** | **0** |
| **MASKED BUG (oracle)** | **0** |
| **MASKED BUG (engine)** | **8** |
| **DOWNSTREAM DESYNC** | **154** |

**Note:** `desync_audit.py` maps **every** `UnsupportedOracleAction` to `oracle_gap` (no sub-class). All 162 rows are `exception_type: UnsupportedOracleAction`.

### Top message shapes (normalized, digits → `N`)

| Rank | Count | Shape (normalized prefix) |
|------|------:|---------------------------|
| 1 | **154** | `Move: engine truncated path vs AWBW path end; upstream drift` |
| 2–9 | **1 each** | `Build no-op at tile (...) unit=<UNIT> for engine P0/P1: engine refused BUILD (...)` (8 distinct shapes by unit / player / optional `funds_after`) |

**Coverage:** The single “Move truncated” shape covers **95.1%** (154/162) of rows; **100%** of rows are covered by two **semantic** families (move-path truncation + build refusal).

### `approx_action_kind` breakdown (154 “Move truncated” rows only)

| Kind | Count |
|------|------:|
| Move | 112 |
| Fire | 32 |
| Capt | 7 |
| Join | 2 |
| Supply | 1 |

These kinds are **where the register last saw an action** before failure; the **shared** message is still the same string. The three code paths that emit it (see below) explain Fire vs Move vs other.

---

## Cross-reference with `desync_clusters_post_phase6.json`

**Clustering model** (`tools/cluster_desync_register.py`): each game appears in **exactly one** cluster bucket derived from `class` + message shape (`desync_subtype`). There is **no** overlap between `engine_bug` and `oracle_gap` on the same `games_id`.

| Cluster | `oracle_gap` rows | Notes |
|---------|------------------:|--------|
| `oracle_other` | **154** | `Move: engine truncated…` (no more specific prefix in `desync_subtype`) |
| `oracle_build` | **8** | `Build no-op…` |
| `engine_other` | **0** | — |

**Interpretation:** `oracle_gap` rows are **not** “shadowed” inside `engine_other` (149 ids). They are a **disjoint** 162-game set that fails first under `UnsupportedOracleAction` instead of surfacing as `ValueError` / `engine_bug`. **Fixing upstream drift** in the engine/oracle pipeline can move games **between** these first-failure classes on a future audit.

---

## Engine bug clusters vs `oracle_gap` “shadow”

**`post_phase6` clusters** in this file: `ok`, `engine_other`, `oracle_other`, `oracle_build` only (no `engine_illegal_move` bucket in this run).

| “Top” `engine_bug` cluster | Size | `oracle_gap` ids also in this cluster |
|-----------------------------|-----:|--------------------------------------|
| `engine_other` | 149 | **0** (mutually exclusive by construction) |

**N/A — top 5:** There is only **one** `engine_bug` cluster present in `desync_clusters_post_phase6.json`. There is **no** row-level overlap to report; the useful relationship is **complementary** (same catalog, different first-failure surface).

---

## Per-family deep dive (families with ≥3 rows)

### Family A — `Move: engine truncated path vs AWBW path end; upstream drift`

| Field | Value |
|------|--------|
| **Row count** | **154** |
| **Normalized shape** | `Move: engine truncated path vs AWBW path end; upstream drift` |
| **Raise sites** (`tools/oracle_zip_replay.py`) | **(1)** `_apply_move_paths_then_terminator` — **3897–3900** after `SELECT_UNIT` + `move_pos` commit: `u.pos` ≠ JSON path end `(er, ec)`. **(2)** `Fire` → post-kill duplicate early-return — **5617–5620** (defender post-kill branch: mover alive but not at path end). **(3)** `Fire` with nested `Move.paths` — **5791–5794** after `ATTACK` step: `u.pos` still ≠ path end. |
| **Code intent** | Explicit **fail-fast** once the engine cannot place the unit at the AWBW path tail after the move/attack sequence. Comment **“upstream drift”** in the user-facing string is intentional: the oracle is **not** claiming a missing AWBW feature; it is **refusing** to pretend the move completed when the committed state disagrees with the zip. |
| **Classification** | **DOWNSTREAM DESYNC (154)** — symptom of **prior** state divergence (path geometry, nested Move, post-kill envelope ordering, or reachability). Same strategic line as Phase 7 Lane D **primary** hypothesis (nested Move / board drift), **not** the rejected “loosen Manhattan” diagonal hypothesis (`logs/desync_regression_log.md` — Phase 7 orchestrator footnote). |
| **Representative `games_id`** | **1605367** (smallest in cluster); **1615143** (early `approx_envelope_index` 7 — fails with fewer applied envelopes). |
| **Recommended action** | **TRACE_UPSTREAM** + **RECONCILE** — trace the **first** point where `unit.pos` / `funds` / turn order diverges; prioritize **Fire (32)** + **Move (112)** sub-streams separately in Phase 9. |
| **Estimated impact if remediated** | **Up to ~154** rows (entire family); overlap with future `engine_bug` surfacing depends on whether fixes land before or after other asserts. |

### Family B — `Build no-op at tile (...) … engine refused BUILD`

| Field | Value |
|------|--------|
| **Row count** | **8** (one row per distinct normalized shape; unit/player text differs) |
| **Normalized shape** | `Build no-op at tile (...) unit=<TYPE> for engine P<N>: engine refused BUILD (...)` |
| **Raise site** | **4591–4596** — after `ActionType.BUILD`, if `funds` and alive-unit count unchanged, `_oracle_diagnose_build_refusal` feeds `UnsupportedOracleAction` with the refusal detail. |
| **Code intent** | Not a deferred-feature gate: **BUILD** was applied **or** no-op’d; oracle detects **no** observable build. |
| **Classification** | **MASKED BUG (engine) (8)** — **candidate** real engine/build-rule or precondition mismatch vs AWBW; **alternatively** stale funds/tile from **upstream drift** (same catalog mechanics as other gaps). **Requires per-replay drill** to split. |
| **Representative `games_id`** | **1625178** (smallest); full set: **`1625178`, `1627563`, `1628287`, `1630064`, `1632006`, `1632778`, `1634587`, `1634961`** |
| **Recommended action** | **RECONCILE** — compare `_oracle_diagnose_build_refusal` reason, `funds`, `active_player`, and tile occupancy vs AWBW snapshot at that envelope; **KEEP** classification only if refusal matches AWBW rules. |
| **Estimated impact if remediated** | **Up to ~8** rows. |

---

## Families with &lt;3 rows

**None** — all 162 rows belong to the two families above (154 + 8).

---

## `REQUIRES_HUMAN_REVIEW`

| Family | Question |
|--------|----------|
| **Build no-op (8)** | For each `games_id`, does **BUILD** fail because **(a)** engine `can_build` / property rules are wrong vs AWBW, or **(b)** **funds** or **tile occupancy** are already wrong because of **earlier** drift? **(b)** would reclassify the row as **DOWNSTREAM DESYNC** for campaign bookkeeping even though the message is the same. |

---

## Phase 9 recommendations (ordered by estimated row impact)

1. **Move / Fire path truncation** — **TRACE_UPSTREAM** — **~154 rows** — Align engine state with zip **before** Move/Fire commits; sub-priorities: **112** plain `Move`, **32** `Fire` nested move, **7** `Capt`, **2** `Join`, **1** `Supply`. **Do not** loosen Phase 6 Manhattan attacker filter; path truncation is orthogonal to that footnote (see regression log Phase 7).

2. **Build refusal** — **RECONCILE** — **~8 rows** — Drill `_oracle_diagnose_build_refusal` output per gid; fix engine build eligibility or **precondition** drift.

3. **Metrics / regression** — After (1), re-run `desync_audit` and expect **migration** between `oracle_gap` and `engine_bug` for games that previously failed later; track **first-divergence** class shifts.

---

## Files of record

- Register: `logs/desync_register_post_phase6.jsonl`
- Clusters: `logs/desync_clusters_post_phase6.json`
- Oracle: `tools/oracle_zip_replay.py` (raise sites cited above)
- Phase 7 Manhattan footnote: `logs/desync_regression_log.md` — `## 2026-04-20 — Phase 7` — ORCHESTRATOR FOOTNOTE (Lane D diagonal hypothesis wrong; **do not** revert Manhattan correction)
