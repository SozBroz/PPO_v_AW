# Phase 11W — Extras catalog audit-expansion recon

**Scope:** Read-only recon for onboarding **195** games in `data/amarriner_gl_extras_catalog.json` (936 on-disk zips − 741 covered by default `desync_audit` run). **No** edits to `engine/game.py` (Phase 11A/11B). **Artifacts:** this file, `logs/phase11w_extras_inventory.json`.

---

## Section 1 — Gap quantification

### 1.1 Catalog totals

| Metric | Value |
|--------|------:|
| Games in extras catalog (`meta.n_games` / counted) | **195** |
| Distinct `map_id` | **14** |
| Extras vs std catalog `games_id` intersection | **0** (disjoint sets) |
| Zips under `replays/amarriner_gl/` for those 195 ids | **195** (all present) |

### 1.2 Tier distribution (game-level)

| Tier | Games |
|------|------:|
| T1 | 28 |
| T2 | 60 |
| T3 | 52 |
| T4 | 55 |
| T0 | 0 |

*(No TL/T0 rows appear in extras; naming matches catalog `tier` strings.)*

### 1.3 CO frequency (top 10, counting each of `co_p0_id` and `co_p1_id` per game)

| `co_id` | Uses (half-games) |
|--------:|------------------:|
| 11 | 43 |
| 23 | 37 |
| 1 | 37 |
| 8 | 28 |
| 14 | 25 |
| 7 | 25 |
| 16 | 25 |
| 22 | 24 |
| 9 | 21 |
| 28 | 19 |

### 1.4 CO coverage vs std catalog

All CO ids that appear in extras also appear in at least one row of `data/amarriner_gl_std_catalog.json` (**no** “extras-only” CO id in this scrape).

### 1.5 Timestamps

All rows share the same `scraped_at`: `2026-04-20T18:48:53.599062+00:00` (single build of the extras catalog). No meaningful in-catalog date spread.

### 1.6 Per-map summary (distinct `map_id`)

Full machine-readable rows: `logs/phase11w_extras_inventory.json` → `maps`.

| `map_id` | Games | Name (catalog) | Size (CSV) | Pool `type` | `p0_country_id` |
|---------:|------:|----------------|------------|-------------|-----------------|
| 69201 | 11 | Beads on a String | 21×16 | std | 1 |
| 77060 | 7 | A Hope Forlorn | 23×19 | std | 1 |
| 123858 | 22 | Misery | 21×17 | std | 1 |
| 126428 | 15 | Ft. Fantasy | 19×19 | std | 1 |
| 133665 | 8 | Walls Are Closing In | 21×17 | std | 1 |
| 134930 | 8 | Swiss Banking | 19×19 | std | 1 |
| 140000 | 12 | 140000 Ways to Die | 19×19 | std | 1 |
| 146797 | 12 | Inland Viking | 19×19 | std | 1 |
| 159501 | 12 | A Dance With Magnums | 19×19 | std | 1 |
| 166877 | 12 | Throne Of Skulls | 19×19 | std | 1 |
| 170934 | 12 | The Final Dance | 19×19 | std | 1 |
| 171596 | 12 | Designed Desires | 19×19 | std | 1 |
| 173170 | 12 | The Great Sea | 21×17 | std | 1 |
| 180298 | 12 | Eternity, Served Cold | 19×19 | std | 1 |

---

## Section 2 — Normalization status

### 2.1 `gl_map_pool.json` tagging

- **Std vs other:** `tools/gl_std_maps.py::gl_std_map_ids` includes only entries with `"type": "std"`.
- **Extras maps:** All **14** distinct ids have `"type": "std"` in `data/gl_map_pool.json` (same tagging as current audit targets).
- **Normalization marker:** All **14** have `p0_country_id == 1` (OS/BM convention per ingest skill).

### 2.2 Files on disk (`data/maps/`)

For every distinct extras `map_id`:

- **`{map_id}.csv`:** present (**14** / 14).
- **`{map_id}_units.json`:** present (**14** / 14).

### 2.3 Category counts (recon taxonomy)

| Category | Maps |
|----------|-----:|
| **READY** (CSV + pool + `p0_country_id == 1`) | **14** |
| **NEEDS_NORMALIZE** | **0** |
| **MISSING_CSV** | **0** |
| **MISSING_POOL** | **0** |
| **MISSING_PREDEPLOY** (no applicable gap) | **0** — sidecars exist for all 14; `load_map` loads predeploy where defined. |

### 2.4 OS/BM pipeline (reference)

- **Order:** `awbw-replay-ingest` skill — download → `run_normalize_map_to_os_bm` (per zip or batch `--from-catalog`) → reconcile `*_units.json` via `load_map` → `desync_audit`.
- **What normalization touches:** `engine/map_country_normalize.py` remaps **property** terrain to Orange Star / Blue Moon; `normalize_map_to_os_bm.py` writes CSV, sets pool `p0_country_id` to **1**, reconciles `*_units.json` if present.
- **Batch tool default catalog:** `normalize_map_to_os_bm.py --from-catalog` defaults to `amarriner_gl_std_catalog.json`; for extras-only maps use `--catalog data/amarriner_gl_extras_catalog.json` (function `catalog_map_ids` is catalog-agnostic).

### 2.5 Engine loader requirements (`engine/map_loader.py::load_map`)

- Map id must exist in `gl_map_pool.json`.
- `{map_id}.csv` must exist under `maps_dir`.
- Optional `{map_id}_units.json` loaded via `load_predeployed_units_file` (may be empty).
- `p0_country_id` in pool drives seating remap when set.

---

## Section 3 — Code / filter changes inventory

### 3.1 Actual exclusion for the current 195 games

Default `desync_audit` uses `data/amarriner_gl_std_catalog.json`. In `_iter_zip_targets`, a zip is skipped when there is **no catalog row** for its `games_id`:

```263:268:D:\AWBW\tools\desync_audit.py
        meta = by_id.get(gid)
        if meta is None:
            continue  # zip without catalog metadata — cannot pick map_id/COs
        mid = _meta_int(meta, "map_id", -1)
        if mid not in std_map_ids:
            continue
```

Because extras and std catalogs are **disjoint** (0 overlapping `games_id`), the **first** `continue` is what drops all 195 extras under default CLI. The **second** (`mid not in std_map_ids`) is **not** triggered for this dataset: every extras `map_id` is in `gl_std_map_ids(...)`.

### 3.2 Minimal onboarding **without** code changes

CLI already supports a catalog path and **repeatable** `--catalog`:

- `--catalog PATH` with `action="append"` → `catalog_paths` merged by `_merge_catalog_files` when length > 1 (`tools/desync_audit.py` approx. lines 475–536).

**Examples (future execution — not run in this recon):**

```text
python tools/desync_audit.py --catalog data/amarriner_gl_extras_catalog.json --register logs/desync_register_extras_baseline.jsonl
```

Merge std + extras (995 unique games; last wins on duplicate ids — here duplicates are impossible):

```text
python tools/desync_audit.py --catalog data/amarriner_gl_std_catalog.json --catalog data/amarriner_gl_extras_catalog.json --register logs/desync_register_merged.jsonl
```

### 3.3 Optional / future-proof code changes

| Item | Location | Purpose |
|------|----------|---------|
| `--no-map-pool-filter` (or `--include-non-std-maps`) | `_iter_zip_targets`, `_count_zip_filter_stats` | Parity with `amarriner_download_replays.py --allow-non-gl-std-maps` if future extras include `type != "std"` pool entries. |
| `_count_zip_filter_stats` | Same guard as iterator | Keep stderr telemetry meaningful when filter is optional. |

**Scope:** Small, localized (conditional around lines 231–232 and 267–268; mirror in `_count_zip_filter_stats` 231–232). Not required for **this** 195-game set.

### 3.4 Download path alignment

`tools/amarriner_download_replays.py` documents std-only scheduling by default and `--allow-non-gl-std-maps` for other pool types (lines 9–12). Extras catalog games are “completed on std maps but not listed in std catalog scrape” — download tooling is separate from audit catalog selection.

---

## Section 4 — Risk inventory

| Category | Level | Rationale |
|----------|-------|-----------|
| **Engine compat** | **LOW** | Same **14** std maps already in pool; tiers T1–T4 match normal GL play. No evidence of fog/weather-only maps in this slice. |
| **Oracle compat** | **MED** | New replays can still hit `UnsupportedOracleAction` shapes not seen in the 741-game sample; CO ids are a subset of std-catalog CO exposure. |
| **OS/BM normalize side effects** | **LOW** | No extras map needs normalization today; distinct `map_id`s — no accidental overwrite of unrelated CSVs when future normalize runs are scoped by catalog. |
| **Audit runtime** | **LOW–MED** | ~26% more games than 741 → rough linear extrapolation **~15–25 min** if per-game cost matches std; **MED** if these replays skew longer (more envelopes). |
| **Result interpretation** | **MED** | Maps overlap std rotation, but **game sample** is disjoint; `oracle_gap` / `engine_bug` rates are **directionally** comparable, not guaranteed identical to std-only baseline thresholds. |

**Highest risk category:** **Oracle compat** / **Result interpretation** (tied — both **MED**).

---

## Section 5 — Onboarding plan (11W-EXEC lanes)

### 11W-EXEC-1 — Catalog-only audit enablement (documentation + smoke)

- **Owner:** Composer 2  
- **Files touched:** None required; optionally `docs/oracle_exception_audit/phase11w_extras_catalog_recon.md` (already) + regression log pointer.  
- **Acceptance:** One command documented; dry `--max-games 3` run completes (optional smoke, not part of this recon).  
- **Time:** 0.5 h  
- **Depends:** None  

### 11W-EXEC-2 — Optional `desync_audit` flag for non-std futures

- **Owner:** Composer 2  
- **Files touched:** `tools/desync_audit.py` (`_iter_zip_targets`, `_count_zip_filter_stats`, argparse).  
- **Acceptance:** With flag, iterator includes zips whose `map_id` ∉ `gl_std_map_ids`; stats line reflects new counts; small test or golden stderr snippet.  
- **Time:** 1–2 h  
- **Depends:** 11W-EXEC-1 only if product wants single PR; can parallelize  

### 11W-EXEC-3 — Sample audit (50 games)

- **Owner:** Composer 2  
- **Files touched:** `logs/` outputs only.  
- **Acceptance:** `desync_register_extras_sample.jsonl` with 50 rows; cluster pass or manual triage notes.  
- **Time:** 0.5 h compute + 1 h triage  
- **Depends:** 11W-EXEC-1  

### 11W-EXEC-4 — Pattern remediation

- **Owner:** Opus or Composer 2 (by finding)  
- **Files touched:** `tools/oracle_zip_replay.py`, engine modules, or oracle exceptions — **not** `engine/game.py` until 11A/11B complete unless explicitly unblocked.  
- **Acceptance:** Classified failures have tickets or fixes; no orphan `loader_error` without locator.  
- **Depends:** 11W-EXEC-3  

### 11W-EXEC-5 — Full 195-game baseline + regression hygiene

- **Owner:** Composer 2  
- **Files touched:** `logs/desync_register_extras_baseline.jsonl`, `logs/desync_regression_log.md` (or campaign summary).  
- **Acceptance:** Full register; `cluster_desync_register` output; baseline committed per project norms.  
- **Depends:** 11W-EXEC-3 (11W-EXEC-4 if blockers found)  

### 11W-EXEC-6 — Normalize batch tooling (only if new maps appear)

- **Owner:** Composer 2  
- **Files touched:** None today; run `normalize_map_to_os_bm.py --from-catalog --catalog data/amarriner_gl_extras_catalog.json` when CSV/pool drift appears.  
- **Acceptance:** All targeted maps `p0_country_id == 1` and CSV OS/BM.  
- **Depends:** New maps missing readiness  

---

## Section 6 — Pre-flight `load_map` results (3 spot-checks)

Random seed **42**; `games_id` → `map_id`:

| `games_id` | `map_id` | Zip on disk | `load_map` |
|------------|----------|-------------|------------|
| 1636530 | 123858 | Yes | OK — Misery, **17×21**, predeploy **2** |
| 1629373 | 171596 | Yes | OK — Designed Desires, **19×19**, predeploy **1** |
| 1627885 | 126428 | Yes | OK — Ft. Fantasy, **19×19**, predeploy **5** |

---

## Section 7 — Bottom line

| Question | Verdict |
|----------|---------|
| **Go / no-go** | **GO** — data plane is ready (pool, CSV, predeploy, OS/BM marker). |
| **Blocker** | Operational: pass **`--catalog data/amarriner_gl_extras_catalog.json`** (or merge). **Not** map-pool filtering for this cohort. |
| **Code change** | **Optional** for future non-std maps; **zero** lines strictly required to audit the current 195. |
| **First lane** | **11W-EXEC-1** (~0.5 h): document + first small audit / full 195 when approved. |

---

*If you know the enemy and know yourself, you need not fear the result of a hundred battles.* — Sun Tzu, *The Art of War* (classical Chinese military treatise; common English rendering)

*Sun Tzu: Chinese strategist, traditionally credited as author of military classics.*
