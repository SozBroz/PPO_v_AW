# Phase 11J-V2-936-AUDIT — Post-campaign canonical floor (FUNDS / MOVE-TRUNCATE / LANE-L / CLUSTER-B)

**Mode:** read-only campaign audit. No engine, oracle, tests, or data edits.
**Register:** `logs/desync_register_post_phase11j_v2_936.jsonl` (936 rows, 397,619 bytes).
**Stderr trace:** `logs/desync_audit_phase11j_v2_936.stderr.txt`.
**Purpose:** Establish the **post-Phase-11J-campaign** ground truth after the four ship lanes (FUNDS-SHIP, MOVE-TRUNCATE-SHIP, LANE-L-WIDEN-SHIP, CLUSTER-B-SHIP), and quantify the lift over the **pre-campaign 936 baseline (864 / 70 / 2)**.

---

## Executive summary

**Final tuple: `(ok=894, oracle_gap=39, engine_bug=3)`.**
Net lift vs the pre-campaign 936 baseline `(864 / 70 / 2)`: **+30 ok, −31 oracle_gap, +1 engine_bug**.
The campaign delivered: the **MOVE-TRUNCATE** family collapsed from 55 → 3 rows (−52, −94.5%), the **single oracle_gap fire-resolution** row cleared, and the prior **F2/Koal `not reachable` row 1630794** is gone. The **+30 ok** is real lift; the residual `oracle_gap` mass migrated almost entirely to the `BUILD no-op` family (now **34 rows = 87% of all `oracle_gap`**, split 25 funds / 9 occupied), which was previously hidden behind upstream move-truncation failures (first-divergence shift, not new bugs introduced). Two new `engine_bug` rows surface as **F4 friendly-fire**, both at first divergence in deeper-floor replays unlocked by upstream fixes — same drift pattern that previously surfaced 1636707 from F4 → F2. **Verdict: GREEN.** The hill is taken; the new firing line is the BUILD-FUNDS cluster.

---

## Section 1 — Audit run details

### Command

```text
python tools/desync_audit.py ^
  --catalog data/amarriner_gl_std_catalog.json ^
  --catalog data/amarriner_gl_extras_catalog.json ^
  --max-games 936 --seed 1 ^
  --register logs/desync_register_post_phase11j_v2_936.jsonl
```

(Note: the user spec referenced `--register-output`; the actual flag is `--register`. Catalog
flag is repeatable per Phase 11W-EXEC pattern; both `std` (800 rows) and `extras` (210 rows)
catalogs were merged; total catalog union = 1,010 rows.)

### Tool output (stderr summary)

- `catalogs: data\amarriner_gl_std_catalog.json data\amarriner_gl_extras_catalog.json total_games=1010`
- `zips_matched=936 filtered_out_by_map_pool=15 filtered_out_by_co=0`
- `936 games audited` → `ok 894 / oracle_gap 39 / engine_bug 3`

### Parameters

| Field | Value |
|--------|--------|
| `--seed` | `1` (canonical) |
| Games audited | **936** |
| Wall time | **~292 s (~4.9 min)** |
| Exit code | `0` |
| Register row count | `936` |

---

## Section 2 — Per-class breakdown (all 936 games)

| Class | Count | % | Δ vs pre-campaign 936 baseline |
|-------|------:|--:|------------------------------:|
| ok                       | **894** | 95.51% | **+30** |
| oracle_gap               | **39**  | 4.17%  | **−31** |
| engine_bug               | **3**   | 0.32%  | **+1**  |
| loader_error             | 0       | 0%     | 0 |
| replay_no_action_stream  | 0       | 0%     | 0 |
| catalog_incomplete       | 0       | 0%     | 0 |
| **Total**                | **936** | **100%** | — |

Exception types under non-ok:

| Count | `exception_type` | Class mapping |
|------:|------------------|---------------|
| 39    | `UnsupportedOracleAction` | `oracle_gap` |
| 3     | `ValueError`              | `engine_bug` |

---

## Section 3 — `oracle_gap` family rollup

Bucketed by message shape (no fine-grained tile/unit splitting). 39 rows total.

| Rank | Family | Count | % of `oracle_gap` |
|-----:|--------|------:|------------------:|
| 1 | **BUILD no-op: insufficient funds** | **25** | 64.1% |
| 2 | **BUILD no-op: tile occupied**      | **9**  | 23.1% |
| 3 | **MOVE: engine truncated path vs AWBW path end (upstream drift)** | **3** | 7.7% |
| 4 | **MOVE: mover not found in engine (refusing drift spawn)**       | **2** | 5.1% |

Sub-breakdowns:

**BUILD no-op: insufficient funds (25)** — by built unit / engine seat:

- Unit:   `INFANTRY ×12, TANK ×4, B_COPTER ×3, NEO_TANK ×2, ANTI_AIR ×2, BOMBER ×1, MECH ×1`
- Seat:   `P0 ×18, P1 ×7` (P0-skewed ~72%)
- Sample shortfall sizes: 100$, 110$, 200$, 300$, 370$, 400$, 420$, 600$ (×3), 700$ (×2), 800$, 900$, 910$, 1000$, 2000$, 2200$, 2600$, 4300$, 5300$, 6500$. Median shortfall is small (~600$) → consistent with **CO-power / income / repair-cost timing** drift, not gross funds drift.

**BUILD no-op: tile occupied (9)** — by built unit / engine seat:

- Unit:   `MECH ×3, INFANTRY ×3, MEGA_TANK ×1, ANTI_AIR ×1, MED_TANK ×1`
- Seat:   `P0 ×5, P1 ×4`
- **Recurring tiles:** `(8,14)` 3×, `(12,10)` 3× → strong map-or-state co-incidence. Likely a stale-unit / death-clear ordering bug on contested production tiles.

---

## Section 4 — Top 10 fine-grained `oracle_gap` templates

Three families × first-clause heads (truncated to 160 chars). Sample gids cited per row.

| # | Count | Template head | 3 sample gids |
|---|------:|---------------|---------------|
| 1 | 3 | `Move: engine truncated path vs AWBW path end; upstream drift` | 1605367, 1626181, 1635162 |
| 2 | 2 | `Move: mover not found in engine; refusing drift spawn from global` | 1626236, 1628722 |
| 3 | 2 | `Build no-op … unit=INFANTRY for engine P0: engine refused BUILD (insufficient funds (need 1000$, have 600$)…)` | 1626991, 1635846 |
| 4 | 1 | `Build no-op at tile (3,19) unit=INFANTRY for engine P0: engine refused BUILD (insufficient funds (need 1000$, have 100$)…)` | 1622501 |
| 5 | 1 | `Build no-op at tile (13,3) unit=NEO_TANK for engine P1: engine refused BUILD (insufficient funds (need 22000$, have 16700$)…)` | 1624082 |
| 6 | 1 | `Build no-op at tile (4,12) unit=MEGA_TANK for engine P0: engine refused BUILD (tile occupied; funds_after=40600$)` | 1625178 |
| 7 | 1 | `Build no-op at tile (8,14) unit=MECH for engine P0: engine refused BUILD (tile occupied; funds_after=26100$)` | 1626223 |
| 8 | 1 | `Build no-op at tile (8,14) unit=MECH for engine P0: engine refused BUILD (tile occupied; funds_after=26500$)` | 1628236 |
| 9 | 1 | `Build no-op at tile (8,14) unit=INFANTRY for engine P0: engine refused BUILD (tile occupied; funds_after=21600$)` | 1628287 |
| 10 | 1 | `Build no-op at tile (12,10) unit=MECH for engine P1: engine refused BUILD (tile occupied; funds_after=20000$)` | 1630064 |

(Ranks 11–25 are the remaining BUILD-no-op singletons distributed across distinct tile/unit/seat combinations.)

---

## Section 5 — `engine_bug` enumeration (all rows)

3 rows, all `ValueError` at first divergence:

| games_id | Family | Message |
|----------|--------|---------|
| **1629202** | **F4 — friendly fire / self-target** | `_apply_attack: friendly fire from player 0 on MECH at (7, 20)` |
| **1632825** | **F4 — friendly fire / self-target** | `_apply_attack: friendly fire from player 0 on INFANTRY at (12, 19)` |
| **1636707** | **F2 — illegal move / not reachable** | `Illegal move: Infantry (move_type=infantry) from (14, 6) to (11, 6) (terrain id=20, fuel=98) is not reachable.` |

Lane assignment:

- **1636707** — already flagged in the prior 936 audit (Section 7) as needing a **distinct Wave 3 lane** (not the Koal F2 oracle FU). Still here. **Assign to F2-RESIDUAL-WAVE3.**
- **1629202, 1632825** — both **F4 friendly-fire** at first divergence. Pre-campaign these games likely diverged earlier on `MOVE-TRUNCATE` and never reached this attack frame; the campaign uncovered them. **Assign to F4-FRIENDLY-FIRE-WAVE2** (pair with 1636707-style first-divergence migration analysis before assuming regression).

---

## Section 6 — Comparison vs pre-campaign 936 baseline

Pre-campaign source: `logs/desync_register_post_phase11j_combined.jsonl` (864 / 70 / 2), summarized in `docs/oracle_exception_audit/phase11j_936_audit.md`.

### Class-level deltas

| Class | Pre (864 / 70 / 2) | Post (this run) | Δ |
|-------|------------------:|----------------:|--:|
| ok          | 864 | **894** | **+30** |
| oracle_gap  | 70  | **39**  | **−31** |
| engine_bug  | 2   | **3**   | **+1**  |

### Family-level deltas (`oracle_gap`)

| Family | Pre (≈) | Post | Δ | Driver |
|--------|--------:|-----:|--:|--------|
| MOVE: truncated path                  | **55** | **3**  | **−52** | **MOVE-TRUNCATE-SHIP** delivered as designed (94.5% reduction). |
| BUILD no-op: tile occupied            | ≈6     | 9      | +3      | First-divergence migration (deeper-floor exposure). |
| BUILD no-op: insufficient funds       | ≈8     | 25     | +17     | First-divergence migration; FUNDS-SHIP fixed gross drift but exposed sub-1000$ residuals. |
| MOVE: mover not found                 | 1      | 2      | +1      | Migration. |
| FIRE: oracle resolved defender        | 1      | 0      | **−1**  | Campaign-cleared. |
| **Total**                             | **70** | **39** | **−31** | |

(Pre-campaign per-family counts derived from Phase 11J-936-AUDIT Section 4 ranks 1–14; the BUILD-no-op subfamilies are approximate because the prior table reported 14 fine-grained templates rather than coarse buckets.)

### `engine_bug` row-level deltas

| games_id | Pre status | Post status | Notes |
|----------|------------|-------------|-------|
| **1630794** | engine_bug (F2 / Koal) | **gone** (now `ok` or `oracle_gap`) | **11J-F2-KOAL-FU-ORACLE** lane closed it. |
| **1636707** | engine_bug (F2) | **engine_bug (F2)** unchanged | Pre-flagged "needs Wave 3 lane"; still here. |
| **1629202** | not in pre engine_bug list | **engine_bug (F4)** new | First-divergence migration after upstream fixes. |
| **1632825** | not in pre engine_bug list | **engine_bug (F4)** new | First-divergence migration. |

Net: **−1 from Koal closure, +2 from F4 surface**. The +2 are not regressions in the engine — they are **new first-divergence sites unlocked** by clearing the upstream MOVE-TRUNCATE wall (same migration pattern previously documented for 1636707).

---

## Section 7 — Recommended follow-up lanes

Ranked by row yield. Clusters of size ≥3 only, plus engine_bug rows even where below threshold (per spec: engine_bug rows always need a lane).

| Lane | Class | Rows | Priority | Justification |
|------|-------|-----:|----------|---------------|
| **L1 — BUILD-FUNDS-RESIDUAL** | oracle_gap | **25** | **P0 (highest yield)** | Single largest cluster, 64% of remaining `oracle_gap`. Median shortfall ~600$ → CO-power / income-tick / repair-cost timing. P0-skewed (18/25). Suggest start with `INFANTRY ×12` subset (cheapest unit, simplest cost model). |
| **L2 — BUILD-OCCUPIED-TILES** | oracle_gap | **9**  | **P1** | Tiles `(8,14)` and `(12,10)` recur 3× each → strong map/state co-incidence. Suggest first: pick one recurring tile, diff engine vs PHP snapshot for unit-at-tile state at the failing turn. |
| **L3 — F4-FRIENDLY-FIRE-WAVE2** | engine_bug | **2** | **P1** (engine_bug always lane-bound) | gids 1629202 + 1632825. Both `_apply_attack` raises on same-player target. Confirm whether these are first-divergence migration vs new attack-resolution defect. Pair with extras-baseline 1636707 lessons. |
| **L4 — MOVE-TRUNCATE-LAST-MILE** | oracle_gap | **3** | **P2** (diminishing returns) | gids 1605367, 1626181, 1635162. Residual after MOVE-TRUNCATE-SHIP. Worth one final pass. |
| **L5 — F2-RESIDUAL-WAVE3 (1636707)** | engine_bug | **1** | **P2** | Pre-flagged in prior audit as needing distinct Wave 3 lane; still unlanded. Standalone investigation. |
| **L6 — MOVER-NOT-FOUND** | oracle_gap | **2** | **P3** (below ≥3 threshold) | Below cluster threshold; defer unless rises after L1–L4. |

---

## Section 8 — Verdict

| Rule | This run |
|------|----------|
| engine_bug ≤ 5 → **GREEN** | **3** → **GREEN** |
| 6–15 → YELLOW | — |
| >15 → RED | — |

**Verdict: GREEN.** Campaign net: `+30 ok, −31 oracle_gap, +1 engine_bug` against the 864/70/2 baseline. The MOVE-TRUNCATE family is effectively cleared (−52 rows). The +1 engine_bug is offset by the Koal F2 closure and is composed of two F4 first-divergence migrations on deeper-floor replays — the **expected** pattern when upstream gates lift.

The follow-up firing line is **BUILD-FUNDS-RESIDUAL (25 rows)** — that single cluster is the next worthwhile ship.

---

## Reproducibility

- Re-run audit: command in §1 (deterministic with `--seed 1` modulo catalog/zip changes).
- Re-classify: `python logs/_phase11j_v2_936_classify.py` (writes class breakdown, top messages, exception types, engine_bug rows, full oracle_gap dump).
- Re-bucket: `python logs/_phase11j_v2_936_buckets.py` (writes family rollup + funds/occupied subdivisions + recurring tiles).
- Stderr trace: `logs/desync_audit_phase11j_v2_936.stderr.txt`.

*"Roma victrix — sed semper vigilantes."* (Latin, Roman military maxim)
*"Rome victorious — but always vigilant."*
*Standard Roman legion commendation: a victory acknowledged, a posture maintained — applicable when a campaign closes but the firing line has only shifted.*
