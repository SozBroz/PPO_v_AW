---
name: awbw-replay-ingest
description: >-
  After downloading AWBW Global League replays, normalize map faction colors to Orange Star / Blue Moon, then run desync_audit and cluster_desync_register. Use when adding replays, refreshing amarriner_gl zips, fixing map seating, regenerating logs/desync_register.jsonl, or the user mentions OS/BM normalization, normalize_map_to_os_bm, or post-download map pipeline.
---

# AWBW replay ingest — download → OS/BM → audit

## Order of operations (non-negotiable)

1. **Download** — `python tools/amarriner_download_replays.py …`  
   By default this **already runs** `run_normalize_map_to_os_bm` for each successfully saved zip’s `map_id` (before any other local processing).  
   Escape hatch: `--skip-os-bm-normalize` (only for rare debugging).

2. **Batch-normalize existing maps** (one-time or after pool/csv edits):  
   `python tools/normalize_map_to_os_bm.py --from-catalog`  
   Iterates **unique `map_id`** values in `data/amarriner_gl_std_catalog.json`; skips missing CSV / pool entries.

3. **Reconcile predeploy** — If `data/maps/<map_id>_units.json` exists, re-check `force_engine_player` after terrain IDs change (see `engine/map_loader.py` + `engine/map_country_normalize.py`).

4. **Desync register** — `python tools/desync_audit.py` (writes `logs/desync_register.jsonl` by default).

5. **Categorize** — `python tools/cluster_desync_register.py --register logs/desync_register.jsonl --markdown logs/desync_bug_tracker.md --json logs/desync_clusters.json`

## What OS/BM normalization does

- Rewrites `data/maps/<map_id>.csv` so the two competitive factions use **Orange Star** and **Blue Moon** terrain IDs.
- Sets `p0_country_id` to **`1`** in `data/gl_map_pool.json` for that map.
- Implementation: `tools/normalize_map_to_os_bm.py` (`run_normalize_map_to_os_bm`), logic in `engine/map_country_normalize.py`.

## Do not

- Run `desync_audit` on a newly downloaded zip **before** normalization for that `map_id` if the map was not already OS/BM (seating drift vs GL).
- Assume PHP snapshot zip terrain matches the local CSV — engine oracle uses **local** `load_map`; viewer art may differ until a separate snapshot rewriter exists.

## Related skills

- `awbw-engine` — engine/oracle overview.
- `awbw-replay-system` — zip layout, export, viewer format.
- `desync-triage-viewer` — C# viewer triage after register rows exist.
