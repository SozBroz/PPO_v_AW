# Phase 9 — Lane L-VAL-4 (Q4 validation)

**Campaign:** Move oracle path-end fix validation (`tools/oracle_zip_replay.py`, Lane L).  
**Slice:** `games_id > 1632479` from `logs/phase9_lane_l_move_targets.jsonl` (Q4 quartile).

## Slice

| Field | Value |
|--------|--------|
| Row count | **45** |
| `games_id` range | **1632558** .. **1636387** |

## Verdict tally

| Verdict | Count |
|---------|------:|
| `FLIPPED_OK` | 39 |
| `STUCK_SAME_FAMILY` | 3 |
| `ESCALATED_TO_ENGINE_BUG` | 3 |
| `PROGRESSED_NEW_GAP` | 0 |
| `ZIP_MISSING` / `CRASH` | 0 |

Re-audit used `tools.desync_audit._audit_one` with metadata from the slice rows, `MAP_POOL_DEFAULT`, `MAPS_DIR_DEFAULT`, and `seed=1` (current keyword-only API).

## `ESCALATED_TO_ENGINE_BUG` (new `engine_bug` surface)

| `games_id` | `approx_envelope_index` | `approx_action_kind` | Message (truncated) |
|------------|---------------------------|----------------------|---------------------|
| 1634717 | 34 | Fire | `_apply_attack: target (2, 13) not in attack range for MECH from (3, 13) (unit_pos=(5, 13))` |
| 1635025 | 36 | Fire | `_apply_attack: target (15, 19) not in attack range for B_COPTER from (14, 19) (unit_pos=(16, 15))` |
| 1635846 | 29 | Fire | `_apply_attack: target (8, 10) not in attack range for B_COPTER from (7, 10) (unit_pos=(2, 9))` |

## `STUCK_SAME_FAMILY` (still `oracle_gap`, truncated-path message)

| `games_id` | `approx_envelope_index` | `approx_action_kind` |
|------------|---------------------------|----------------------|
| 1634328 | 20 | Move |
| 1634809 | 22 | Fire |
| 1635119 | 48 | Move |

## Outputs

| Artifact | Path |
|----------|------|
| Input slice | `logs/phase9_lane_l_val4_slice.jsonl` |
| Per-game results | `logs/phase9_lane_l_val4_results.jsonl` |
| Console audit | `logs/phase9_lane_l_val4_audit.log` |
| 3-action windows (STUCK + ESCALATED) | `logs/phase9_lane_l_val4_windows.json` |

## Required reading

- `docs/oracle_exception_audit/phase9_lane_l_move_fix.md` — Lane L fix and known follow-ups.
- `logs/desync_regression_log.md` — Phase 7 orchestrator footnote (Manhattan vs diagonal; do not revert Phase 6 Manhattan correction).
