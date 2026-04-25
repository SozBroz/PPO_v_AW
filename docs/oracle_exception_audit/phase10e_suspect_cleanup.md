# Phase 10E — Lane K SUSPECT cleanup (oracle slack tightening)

**Campaign:** `desync_purge_engine_harden`  
**Scope:** `tools/oracle_zip_replay.py` only (no `engine/`). **Cap:** 6 APPLY edits (under the 8-edit budget).  
**Forbidden:** Lane 10B move-terminator / nested-Move-in-Fire internals; Phase 6 Manhattan direct-fire candidate filter; Lane O / Lane I already-tightened sites.

**Canon:** `logs/desync_regression_log.md` Phase 7 ORCHESTRATOR FOOTNOTE — do not loosen post–Phase 6 Manhattan tightening or reintroduce diagonal direct-fire slack.

---

## Lane K inventory note

`docs/oracle_exception_audit/phase8_lane_k_slack_inventory.md` rates **32 SUSPECT** occurrences by **pattern cluster** (not always 32 uniquely line-identified rows). This document triages **every SUSPECT cluster** and records **six** concrete APPLY edits executed in Phase 10E.

---

## Triage tally

| Disposition | Count | Meaning |
|-------------|------:|---------|
| **APPLY** | **6** | Tightened in Phase 10E (this lane) |
| **DEFER** | 16 | Collision with Lane 10B, behavior-risk, or Phase 11 queue |
| **SKIP** | 10 | Already tightened in Lane O, Lane K JUSTIFIED, or Phase 6 hands-off |

**Total clusters accounted for:** 32

---

## Triage table (32 SUSPECT clusters)

| # | Area (Lane K ref.) | Disposition | Reason |
|---|-------------------|-------------|--------|
| 1 | Supply `eng_hint` `try/except pass` (~4703) | **SKIP** | Lane O tightened (`UnsupportedOracleAction` + `from e`) |
| 2 | `_guess_unmoved_mover_from_site_unit_name` swallow (~3202) | **SKIP** | Lane O: re-raise when non-empty raw name |
| 3 | `_oracle_ensure_envelope_seat` silent early exit (~311) | **SKIP** | Lane O: raise on bad / unmapped pid |
| 4 | `_resolve_fire_or_seam_attacker` terminal `None` (~2465) | **SKIP** | Lane O: `OracleFireSeamNoAttackerCandidate` |
| 5 | `Fire Move: could not resolve unit.global for mover` (~5700) | **DEFER** | Nested-Move-in-Fire branch — Lane 10B forbidden zone |
| 6 | Fire mover / attacker resolution f-strings (~5843+) | **DEFER** | Same nested-Fire flank as #5 |
| 7 | `_oracle_fire_attack_move_pos_candidates` (Phase 6 block) | **SKIP** | Phase 6 hard hands-off |
| 8 | `_oracle_fire_resolve_defender_target_pos` Chebyshev-1 / fog | **DEFER** | Geometric slack; wrong-foe risk needs dedicated audit |
| 9 | Fire no-path silent `return` (obsolete combat rows) | **DEFER** | Lane K: high churn; instrumentation vs strict product call |
| 10 | `_after_attack_seam` “without paths” raise (~5963) | **DEFER** | Callback passed to `_apply_move_paths_then_terminator` — Lane 10B |
| 11 | `_resolve_fire_or_seam_attacker` weak caller errors (cluster) | **SKIP** | Lane O terminal path; call sites use subclass |
| 12 | `_oracle_resolve_nested_hide_unhide_units_id` `int` pass (~1062) | **SKIP** | Lane K JUSTIFIED: best-effort optional `units_id` |
| 13 | `_optional_declared_unit_type_from_move_gu` return None (~3337) | **SKIP** | Lane K JUSTIFIED: explicitly optional type |
| 14 | `replay_first_mover_from_snapshot_turn` int parse (~3036) | **DEFER** | Wide bootstrap impact; fail-soft by design |
| 15 | Repair boat `_repair_boat_awbw_id` swallow in tile disambiguation (~1883, ~1942) | **DEFER** | Optional narrowing hint; strict variant needs pairing tests |
| 16 | `_oracle_resolve_move_unit_from_global_and_path` `want_t` swallow (~3859) | **APPLY** | Lane K DELETE-style: non-empty unmapped name → fail loud (mirrors Lane O guesser) |
| 17 | Move waypoint `except: continue` loops (many lines) | **SKIP** | Lane K JUSTIFIED: malformed JSON fragments |
| 18 | `extract_json_action_strings_from_envelope_line` PHP recovery | **SKIP** | Lane K JUSTIFIED |
| 19 | `Power` `int(raw_pid)` could throw bare `ValueError` (~4706) | **APPLY** | Chained `UnsupportedOracleAction` for diagnostics |
| 20 | Fire no-path “Fire without Move.paths.global” (~5510) | **APPLY** | Message lists attacker keys when `att` empty (no behavior change) |
| 21 | AttackSeam no-path missing `combatInfo` (~5997) | **APPLY** | Richer context: `gu` / `unit` wrap / `AttackSeam` keys |
| 22 | `Unload` `transportID` `int()` (~6092) | **APPLY** | Chained `UnsupportedOracleAction` on non-numeric id |
| 23 | `Unload` cargo snapshot unresolved (~6103) | **APPLY** | Message lists `unit` wrap + merged + flat key sets |
| 24 | `_oracle_snap_active_player_to_engine` | **SKIP** | Lane K JUSTIFIED (high-leverage, documented) |
| 25 | `_oracle_nudge_eng_occupier_off_production_build_tile` | **SKIP** | Lane K JUSTIFIED |
| 26 | `_oracle_move_med_tank_label_engine_tank_drift` heuristics | **DEFER** | Naming-drift helper; tighten only with replay corpus |
| 27 | Short literal `UnsupportedOracleAction` elsewhere (AST “short msg” hunt) | **DEFER** | Batch message pass → Phase 11 |
| 28 | Capt optional building helpers `except → None` | **SKIP** | Lane K JUSTIFIED: optional fields |
| 29 | Combat damage `_to_internal` non-numeric → None (~1148) | **SKIP** | Lane K JUSTIFIED: outer raise if both None |
| 30 | AttackSeam / Fire no-path `units_players_id` parse → None | **DEFER** | Changing seat-sync behavior needs targeted replays |
| 31 | Unload resolver message (“drift recovery disabled…”) | **DEFER** | Already chains `__cause__`; further split → Phase 11 |
| 32 | Indirect-fire / seam geometry helpers (non–Phase 6) | **DEFER** | Orthogonal to Phase 10E message tightenings |

---

## Per-APPLY edits

### 1. `want_t` unmapped name — `_oracle_resolve_move_unit_from_global_and_path`

- **Lane K:** DELETE table row “3797–3798 `except UnsupportedOracleAction: want_t = None`” (SUSPECT family).
- **Location:** `tools/oracle_zip_replay.py` (~3859–3866).
- **Before:** `except UnsupportedOracleAction: want_t = None` for all failures from `_name_to_unit_type`.
- **After:** Re-raise when `raw_nm` is non-empty after strip (same rule as Lane O guesser); keep `want_t = None` only for empty/whitespace names.
- **Rationale:** Non-empty but unknown labels are zip corruption or label drift — surface early instead of falling through to weaker heuristics.

### 2. `Power` — non-int `playerID`

- **Lane K:** Implicit (numeric field parse without diagnostic).
- **Location:** ~4702–4712.
- **After:** `try/except (TypeError, ValueError) as e: raise UnsupportedOracleAction(...) from e` around `int(raw_pid)`.
- **Rationale:** Malformed `playerID` now classifies as oracle-visible gap with chained cause.

### 3. Fire (no path) — missing `Move.paths.global` with attacker present

- **Lane K:** Short message cluster; “Fire without Move.paths.global” when `att` empty was ambiguous.
- **Location:** ~5510–5514.
- **After:** Message includes `attacker_keys=...` for the empty-attacker case (still raises; clearer triage).
- **Rationale:** Distinguishes “no attacker dict” vs “empty attacker dict” vs other failures.

### 4. AttackSeam (no path) — missing `combatInfo`

- **Lane K:** Vague message cluster (“without Move.paths.global or combatInfo”).
- **Location:** ~6003–6011.
- **After:** Message includes sorted keys from resolved `gu`, `uwrap`, and `aseam`.
- **Rationale:** Faster diagnosis of which bucket lost `combatInfo` when paths are empty.

### 5. `Unload` — `transportID` parse

- **Location:** ~6093–6100.
- **After:** `int(raw_tid)` wrapped; `UnsupportedOracleAction` with `from e` on failure.
- **Rationale:** Non-numeric transport id is always corrupt JSON for this action kind.

### 6. `Unload` — unresolved cargo snapshot

- **Lane K:** SUSPECT vague message (“could not resolve cargo snapshot…”).
- **Location:** ~6108–6118.
- **After:** Message includes types/keys for `unit` wrap, merged `gu`, and flat global.
- **Rationale:** Pinpoints whether envelope merge, per-seat, or flat global failed.

---

## Pytest

**Command:**

`python -m pytest tests/ test_oracle_zip_replay.py --tb=short -q` (tee: `logs/phase10e_pytest.log`)

**Result:** `265 passed`, `2 xfailed`, `3 xpassed` (no test file changes).

**Note:** An initial run in this session reported 2 failures in `test_oracle_terminator_snap.py` against a **longer** test module revision; the current workspace test file is shorter and the **final** full sweep is green.

---

## Sample audit (first 30 catalog `games_id`, ascending)

**Script:** In-process `_audit_one` with `seed=CANONICAL_SEED`, `map_pool` / `maps_dir` defaults, std-tier map filter; output: `logs/phase10e_sample_audit.log`.

**Result:** No `CRASH` rows (no uncaught exceptions escaping the audit harness). Rows are `ok`, `oracle_gap`, `engine_bug`, or `SKIP` (map not in std pool for that catalog slice). **1607045** reports `oracle_gap` (“Move: engine truncated path…”) — expected classification, not a harness crash.

---

## Rollbacks

None. All APPLY edits remained after pytest.

---

## Artifacts

- `logs/phase10e_pytest.log`
- `logs/phase10e_sample_audit.log`
- `docs/oracle_exception_audit/phase10e_suspect_cleanup.md` (this file)
