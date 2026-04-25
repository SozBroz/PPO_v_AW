# Phase 9 — Lane L validation slice L-VAL-3 (Q3 gids)

**Campaign:** `desync_purge_engine_harden`  
**Lane:** L-VAL-3 — re-audit Q3 slice after Lane L `oracle_zip_replay` Move path-end fix (`tools/oracle_zip_replay.py`).

## Slice definition

| Field | Value |
|--------|--------|
| **Quartile** | Q3: `1630006 < games_id ≤ 1632479` |
| **Rows** | **45** (from `logs/phase9_lane_l_move_targets.jsonl`) |
| **games_id range** | **1630037** .. **1632479** |

**Note:** The user-provided audit script called `_audit_one(gid, zip_path, seed=1)`; current `tools/desync_audit._audit_one` is keyword-only and requires `meta`, `map_pool`, and `maps_dir`. The run used catalog fields from each slice row plus `data/gl_map_pool.json` / `data/maps/`, `seed=1`, matching the CLI harness.

## Verdict tally (45 games)

| Verdict | Count |
|---------|------:|
| `FLIPPED_OK` | **36** |
| `STUCK_SAME_FAMILY` | **5** |
| `ESCALATED_TO_ENGINE_BUG` | **4** |
| `ZIP_MISSING` | 0 |
| `CRASH` | 0 |

**Interpretation:** The Move path-end fix cleared the original truncated-path `oracle_gap` for **36/45** games in this quartile. **Five** still raise the same `Move: engine truncated path vs AWBW path end; upstream drift` message (Join / Load / Move / nested Fire shapes — see windows file). **Four** now fail earlier with **`engine_bug`** at **`Fire`** (range / `unit_pos` mismatch — consistent with Lane L doc’s “nested Fire / drift” note).

## ESCALATED — `engine_bug` (4)

First divergence is no longer the Move oracle message; `class == engine_bug`.

| games_id | approx_envelope_index | new_message (truncated) |
|----------|----------------------:|-------------------------|
| 1630038 | 24 | `_apply_attack: target (5, 3) not in attack range for B_COPTER from (5, 4) (unit_pos=(2, 5))` |
| 1630308 | 31 | `_apply_attack: target (17, 19) not in attack range for B_COPTER from (18, 19) (unit_pos=(18, 18))` |
| 1631289 | 30 | `_apply_attack: target (16, 4) not in attack range for MECH from (16, 3) (unit_pos=(15, 2))` |
| 1631621 | 26 | `_apply_attack: target (15, 14) not in attack range for B_COPTER from (16, 14) (unit_pos=(16, 14))` |

## STUCK — same `oracle_gap` family (5)

Still `oracle_gap` with substring `engine truncated path` in the message.

| games_id | approx_envelope_index | approx_action_kind |
|----------|----------------------:|--------------------|
| 1630784 | 32 | Join |
| 1630794 | 37 | Load |
| 1631257 | 22 | Move |
| 1632283 | 29 | Fire |
| 1632447 | 29 | Fire |

## Output paths

| Artifact | Path |
|----------|------|
| Q3 slice (input rows) | `logs/phase9_lane_l_val3_slice.jsonl` |
| Per-game results | `logs/phase9_lane_l_val3_results.jsonl` |
| Audit console log | `logs/phase9_lane_l_val3_audit.log` |
| Last-3-action windows (ESCALATED + STUCK) | `logs/phase9_lane_l_val3_windows.json` |

Full action JSON in the windows file is large; use it for envelope/day/action-kind context at first divergence.

## Regression log cross-check

Phase 7 **ORCHESTRATOR FOOTNOTE** (`logs/desync_regression_log.md`): Manhattan direct-fire canon and Lane D diagonal hypothesis — **not** the driver for these four `Fire` escalations; the failures match **position / range drift** at `_apply_attack`, i.e. downstream of the Move-path reconciliation work.

---

*“In preparing for battle I have always found that plans are useless, but planning is indispensable.”* — Dwight D. Eisenhower, remarks at NATO Headquarters, Paris (1957)  
*Eisenhower: Supreme Allied Commander in Europe in WWII, later 34th U.S. President.*
