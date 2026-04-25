# Phase 9 — Lane L validation (Move fix)

Consolidated reports from validation lanes L-VAL-1 … L-VAL-4 (quartiles of `logs/phase9_lane_l_move_targets.jsonl`). Each lane appends under its own heading.

---

## Lane L-VAL-1 (Q1)

**Slice:** `games_id` **1607045 … 1627004** (inclusive), **46** rows (`logs/phase9_lane_l_val1_slice.jsonl`).

**Method:** Per-game `tools.desync_audit._audit_one` with `seed=1`, `data/gl_map_pool.json`, `data/maps`, catalog `data/amarriner_gl_std_catalog.json` (same defaults as `desync_audit.py`). Parent instructions used `_audit_one(gid, zip, seed)`; current API is keyword-only and requires catalog `meta`.

### Verdict tally

| Verdict | Count |
|--------|------:|
| FLIPPED_OK | 36 |
| PROGRESSED_NEW_GAP | 2 |
| STUCK_SAME_FAMILY | 3 |
| ESCALATED_TO_ENGINE_BUG | 5 |
| CRASH | 0 |
| ZIP_MISSING | 0 |

### ESCALATED_TO_ENGINE_BUG (new `engine_bug` — priority for Phase 10)

| games_id | New message (truncated) |
|----------|-------------------------|
| 1614665 | `_apply_attack: target (12, 7) not in attack range for B_COPTER from (11, 7) (unit_pos=(13, 3))` |
| 1622104 | `_apply_attack: target (6, 16) not in attack range for MECH from (6, 17) (unit_pos=(7, 17))` |
| 1624590 | `_apply_attack: target (14, 22) not in attack range for B_COPTER from (13, 22) (unit_pos=(10, 25))` |
| 1624758 | `_apply_attack: target (12, 5) not in attack range for B_COPTER from (11, 5) (unit_pos=(6, 5))` |
| 1625784 | `_apply_attack: target (6, 5) not in attack range for B_COPTER from (6, 6) (unit_pos=(7, 10))` |

All five surfaced on **`Fire`** envelopes after the Move fix; messages match the Phase 6/7 pattern (attacker `from` vs `unit_pos` drift under Manhattan-range checks).

### STUCK_SAME_FAMILY (still `oracle_gap` with “engine truncated path …”)

| games_id | `approx_envelope_index` | Notes |
|----------|-------------------------|--------|
| 1607045 | 46 | First failure on nested **`Join`** (day 24); same class of join/seat issues called out in `phase9_lane_l_move_fix.md` for this gid. |
| 1626437 | 25 | Fails on **`Fire`** with truncated-path message — nested Fire / path-end edge still not fully reconciled. |
| 1626991 | 26 | Nested **`Join`** (day 14 envelope window); same family as 1607045. |

### PROGRESSED_NEW_GAP (no longer the truncated-path Move message)

- **1623866** — `AttackSeam: no ATTACK to seam (11, 14) from (12, 13); legal=['WAIT', 'ATTACK']`
- **1626236** — `Move: mover not found in engine; refusing drift spawn from global`

### Artifacts (this lane)

| File | Purpose |
|------|---------|
| `logs/phase9_lane_l_val1_slice.jsonl` | Q1 target rows |
| `logs/phase9_lane_l_val1_results.jsonl` | Per-gid verdict + class/message |
| `logs/phase9_lane_l_val1_audit.log` | Console copy of the sweep |
| `logs/phase9_lane_l_val1_windows.json` | Three-envelope windows (envelope index ±1) for ESCALATED + STUCK; **compact** `action_kinds` per envelope. Amarriner zips do not contain `actions.json`; windows were built via `parse_p_envelopes_from_zip` (gzip `a{games_id}` stream). |

---
