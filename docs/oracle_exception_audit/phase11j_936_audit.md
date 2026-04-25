# Phase 11J-936-AUDIT — Canonical post–Phase 11 floor

**Mode:** read-only campaign audit (no engine, oracle, tests, or data edits).  
**Register:** `logs/desync_register_post_phase11j_combined.jsonl` (936 rows).  
**Purpose:** Establish the **new baseline** after Phase 11A/B/C/J-FIRE-DRIFT/J-F2-KOAL work; compare to Phase 10Q (741 std) and Phase 11W extras (195).

---

## Section 1 — Audit run details

### Command

```text
python tools/desync_audit.py ^
  --catalog data/amarriner_gl_std_catalog.json ^
  --catalog data/amarriner_gl_extras_catalog.json ^
  --register logs/desync_register_post_phase11j_combined.jsonl ^
  --seed 1
```

Multi-catalog merge is supported (`tools/desync_audit.py` merges `games` blocks; duplicate `games_id` uses the last file — here std and extras sets are disjoint).

### Tool output (stderr summary)

- `catalogs: … total_games=995` (union of catalog rows before zip intersection).
- `zips_matched=936` `filtered_out_by_map_pool=0` `filtered_out_by_co=0` (741 std-only zips + 195 extras-only zips **on std-map-pool** maps).

### Parameters

| Field | Value |
|--------|--------|
| `--seed` | `1` |
| Games audited | **936** |
| Wall time (process exit) | **~205 s** (~3.4 min) on the host that ran this audit |
| Exit code | `0` |

---

## Section 2 — Per-class counts (all 936 games)

| Class | Count | % |
|-------|------:|--:|
| ok | 864 | 92.31% |
| oracle_gap | 70 | 7.48% |
| engine_bug | 2 | 0.21% |
| loader_error | 0 | 0% |
| replay_no_action_stream | 0 | 0% |
| catalog_incomplete | 0 | 0% |
| **Total** | **936** | **100%** |

---

## Section 3 — `engine_bug` family classification (Phase 11D F1–F5)

| Family | Count | Notes |
|--------|------:|--------|
| **F1** — Bucket A / attack-position drift | 0 | — |
| **F2** — Illegal move / reachability (`not reachable`) | **2** | Both rows match Phase 11D row-shape for **F2** (no friendly-fire / Black Boat signature). |
| **F3** — CO power / B_COPTER sub-lexicon | 0 | — |
| **F4** — Friendly fire / self-target | 0 | **1636707** no longer fails as F4 at first divergence (contrast Phase 11W extras baseline — see Section 7). |
| **F5** — Black Boat / unarmed / occupancy | 0 | **1626642** not among `engine_bug` rows — consistent with F5 lane work. |
| **OTHER** | 0 | — |

---

## Section 4 — `oracle_gap` message families (top templates)

Buckets by dominant keyword / shape (human-readable rollup):

| Rank | Template / family | Count | % of `oracle_gap` |
|------|-------------------|------:|------------------:|
| 1 | Move: engine truncated path vs AWBW path end; upstream drift | 55 | 78.6% |
| 2 | Build no-op: engine refused BUILD (tile occupied / insufficient funds / etc.) | 13 | 18.6% |
| 3 | Move: mover not found in engine | 1 | 1.4% |
| 4 | Fire: oracle resolved defender … (no damage entry vs defender type) | 1 | 1.4% |

Fine-grained unique prefixes in this run: **14** distinct oracle_gap message heads (Build no-ops split across tiles/units).

### Top 10 fine-grained templates (first clause before `;`, truncated to ~160 chars)

| # | Count | Template head |
|---|------:|----------------|
| 1 | 55 | `Move: engine truncated path vs AWBW path end` |
| 2 | 2 | `Build no-op at tile (8,14) unit=MECH for engine P0: engine refused BUILD (tile occupied…` |
| 3 | 2 | `Build no-op at tile (12,10) unit=INFANTRY for engine P1: engine refused BUILD (tile occupied…` |
| 4 | 1 | `Build no-op at tile (4,12) unit=MEGA_TANK for engine P0: engine refused BUILD (tile occupied…` |
| 5 | 1 | `Move: mover not found in engine` |
| 6 | 1 | `Build no-op at tile (1,4) unit=INFANTRY for engine P1: engine refused BUILD (insufficient funds…` |
| 7 | 1 | `Build no-op at tile (8,14) unit=INFANTRY for engine P0: engine refused BUILD (tile occupied…` |
| 8 | 1 | `Build no-op at tile (12,8) unit=NEO_TANK for engine P1: engine refused BUILD (insufficient funds…` |
| 9 | 1 | `Build no-op at tile (12,10) unit=MECH for engine P1: engine refused BUILD (tile occupied…` |
| 10 | 1 | `Fire: oracle resolved defender type TANK at (14, 13) but FIGHTER has no damage entry against it …` |

Ranks 11–14 are four additional **Build no-op** singletons (one row each).

---

## Section 5 — Per-tier × class

Classes shown: main outcome classes + zeros for infrastructure classes.

| Tier | ok | oracle_gap | engine_bug | loader_error | replay_no_action_stream | catalog_incomplete | **n** |
|------|---:|-----------:|-------------:|-------------:|------------------------:|-------------------:|------:|
| T1 | 129 | 10 | 0 | 0 | 0 | 0 | 139 |
| T2 | 239 | 18 | 0 | 0 | 0 | 0 | 257 |
| T3 | 252 | 15 | 1 | 0 | 0 | 0 | 268 |
| T4 | 244 | 27 | 1 | 0 | 0 | 0 | 272 |

### Disproportion notes

- **`oracle_gap` rate** rises toward **T4** (**~9.9%** of T4 games vs **~5.6–7.2%** on T1–T3). Engine_bug counts are **1 each** on T3 and T4 only (too rare for strong tier conclusions).
- **T1/T2** have **zero** `engine_bug` in this run.

---

## Section 6 — Comparison tables

### vs Phase 10Q — 741 std games

| Class | Phase 10Q | Phase 11J (std subset) | Δ |
|-------|----------:|----------------------:|--:|
| ok | 680 | **685** | **+5** |
| oracle_gap | 51 | **55** | **+4** |
| engine_bug | 10 | **1** | **−9** |

Phase 10Q source: `logs/desync_register_post_phase10q.jsonl`.

### vs Phase 11W — 195 extras

| Class | Phase 11W (extras baseline) | Phase 11J (extras subset) | Δ |
|-------|----------------------------:|--------------------------:|--:|
| ok | 179 | **179** | **0** |
| oracle_gap | 15 | **15** | **0** |
| engine_bug | 1 | **1** | **0** |

Phase 11W source: `logs/desync_register_extras_baseline.jsonl`  
*(See `docs/oracle_exception_audit/phase11w_exec_extras_audit.md`.)*

### Combined 936 — pre vs post Phase 11J

Pre–Phase-11 arithmetic from Phase 10Q + 11W registers: **859** ok / **66** oracle_gap / **11** engine_bug.

| Class | Pre-Phase-11 (859 / 66 / 11) | Phase 11J (936) | Δ |
|-------|-----------------------------:|----------------:|--:|
| ok | 859 | **864** | **+5** |
| oracle_gap | 66 | **70** | **+4** |
| engine_bug | 11 | **2** | **−9** |

Interpretation: **large(engine_bug)** drop is almost entirely from **std** (−9); **oracle_gap** increased slightly (**+4**, all in std); **extras** fingerprint is **unchanged** on this seed.

---

## Section 7 — Surviving `engine_bug` enumeration + backlog

| games_id | Tier | `approx_day` | `approx_envelope_index` | `approx_action_kind` | Family | Message (abridged) |
|----------|------|-------------:|------------------------:|------------------------|--------|----------------------|
| **1630794** | T4 | 19 | 37 | Load | **F2** | `Illegal move: Infantry … from (2, 7) to (1, 10) … is not reachable.` |
| **1636707** | T3 | 12 | 23 | Move | **F2** | `Illegal move: Infantry … from (14, 6) to (11, 6) … is not reachable.` |

### Follow-up lane coverage

| Item | Assessment |
|------|------------|
| **11J-F2-KOAL-FU-ORACLE (1630794)** | **Covered** — same gid as Phase 11D F2 / Koal-class residual; still `engine_bug` with Load + `not reachable`. |
| **11J-F5-OCCUPANCY-IMPL (1626642)** | **Not listed** — **1626642** does not appear as `engine_bug` in this register (F5 lane objective consistent with closure on this floor). |
| **1636707** | **Not covered** by the two lanes above. In Phase 11W extras baseline this gid was **F4 (friendly fire)**; on Phase 11J it diverges first on **`Illegal move … not reachable`**. Treat as **first-divergence migration** after upstream fixes — **not** a proven regression without C# viewer diff, but **requires an explicit Wave 3 lane** (reachability / map 180298 / INF MP path) distinct from the Koal oracle follow-up. |

**Counts:** **1** row aligned with an in-flight F2-oracle lane (**1630794**); **1** row needs a **new or separate** lane (**1636707**).

---

## Section 8 — `loader_error` / infrastructure

| Metric | Value |
|--------|------:|
| `loader_error` rows | **0** |
| Gid clusters | *n/a* |

Per Phase 10G concern (harness `except`-path masking): **no** loader_error rows in this run — nothing to cluster.

---

## Section 9 — Verdict

| Rule | This run |
|------|----------|
| engine_bug ≤ 5 → **GREEN** | **2** → **GREEN** |
| 6–15 → YELLOW | — |
| >15 → RED | — |

**Verdict: GREEN** — engine_bug count is **2**, with **no** loader_error masking and **F1/F4/F5** residuals cleared at first-divergence for this gate.

---

## Reproducibility

- Re-summary: `python tools/_phase11j_936_analyze.py` (parses `logs/desync_register_post_phase11j_combined.jsonl`).
- Re-run audit: same command as Section 1; expect deterministic classification with `--seed 1` aside from any future catalog/zip changes.
