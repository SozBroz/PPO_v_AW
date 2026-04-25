# Phase 9 — Lane L-VAL-2 (Q2 validation)

**Campaign:** Move oracle path-end fix validation (`tools/oracle_zip_replay.py`, Lane L).  
**Slice:** `1627004 < games_id ≤ 1630006` from `logs/phase9_lane_l_move_targets.jsonl`.

## Slice

| Field | Value |
|--------|--------|
| Row count | **46** |
| `games_id` range | **1627034** .. **1630006** |

## Verdict tally

| Verdict | Count |
|---------|------:|
| `FLIPPED_OK` | 36 |
| `STUCK_SAME_FAMILY` | 5 |
| `ESCALATED_TO_ENGINE_BUG` | 4 |
| `PROGRESSED_NEW_GAP` | 1 |
| `ZIP_MISSING` / `CRASH` | 0 |

**Note:** Re-audit used `tools.desync_audit._audit_one` with catalog metadata, `MAP_POOL_DEFAULT`, `MAPS_DIR_DEFAULT`, and `seed=1` (current API).

## `ESCALATED_TO_ENGINE_BUG` (new `engine_bug` surface)

All four are `_apply_attack` range mismatches (unit position vs attack-from hex).

| `games_id` | `approx_envelope_index` | `approx_action_kind` | Message (truncated) |
|------------|-------------------------|----------------------|---------------------|
| 1628868 | 26 | Fire | `_apply_attack: target (0, 8) not in attack range for B_COPTER from (0, 9) (unit_pos=(0, 14))` |
| 1629034 | 31 | Fire | `_apply_attack: target (6, 2) not in attack range for B_COPTER from (6, 1) (unit_pos=(10, 2))` |
| 1629120 | 40 | Fire | `_apply_attack: target (6, 7) not in attack range for B_COPTER from (5, 7) (unit_pos=(4, 8))` |
| 1629951 | 26 | Fire | `_apply_attack: target (11, 6) not in attack range for MECH from (10, 6) (unit_pos=(9, 6))` |

## `STUCK_SAME_FAMILY` (still `oracle_gap`, truncated-path message)

| `games_id` | `approx_envelope_index` | `approx_action_kind` |
|------------|-------------------------|----------------------|
| 1627557 | 32 | Capt |
| 1627696 | 22 | Fire |
| 1628722 | 31 | Fire |
| 1628985 | 17 | Move |
| 1629722 | 33 | Move |

## `PROGRESSED_NEW_GAP` (for traceability)

| `games_id` | `approx_envelope_index` | First failure |
|------------|-------------------------|----------------|
| 1628236 | 26 | `Build` — build no-op / tile occupied (`MECH` at (8,14)) |

## Outputs

| Artifact | Path |
|----------|------|
| Input slice | `logs/phase9_lane_l_val2_slice.jsonl` |
| Per-game results | `logs/phase9_lane_l_val2_results.jsonl` |
| Console audit | `logs/phase9_lane_l_val2_audit.log` |
| 3-action windows (STUCK + ESCALATED) | `logs/phase9_lane_l_val2_windows.json` |

## Required reading — footnote

`logs/desync_regression_log.md` (Phase 7 orchestrator footnote) was **not present** in this workspace at validation time; Lane L fix context is in `docs/oracle_exception_audit/phase9_lane_l_move_fix.md`.
