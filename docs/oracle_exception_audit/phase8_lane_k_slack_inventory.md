# Phase 8 Lane K — Static slack inventory: `tools/oracle_zip_replay.py`

**Campaign:** `desync_purge_engine_harden`  
**Mode:** read-only code review (no edits to `engine/`, `tools/oracle_zip_replay.py`, or `tests/`).  
**Canon constraint:** Phase 6 Manhattan direct-fire correction (~line 2434–2441) and Phases 2–7 tightening stay **fixed**. The Phase 7 **ORCHESTRATOR FOOTNOTE** in `logs/desync_regression_log.md` (§ 2026-04-20 — Phase 7) rejects “diagonal direct fire / revert Chebyshev” as a driver; Phase 9 work must **not** loosen that gate.

**Related context:** `docs/oracle_exception_audit/phase2_worklist.md` (deleted drift/snap helpers), `docs/oracle_exception_audit/phase7_drift_triage.md` (44 drift rows; Bucket A position drift vs Bucket B wrong attacker). *Note:* phase7 doc § “direct-fire metric mismatch / Chebyshev-1” conflicts with the regression-log footnote; **the footnote wins** for campaign direction.

---

## Headline summary

### Total occurrences per pattern (8 buckets + AST)

| Bucket | Count | Notes |
|--------|------:|--------|
| `try_except_pass` | **3** | Lines 1052, 1067, 4703 |
| `try_except_continue` | **13** | Malformed waypoint / PHP scan recovery (includes `extract_json_action_strings_from_envelope_line` ValueError loop) |
| `try_except_return_none` | **11** | Bodies that **return `None`** under `except` (optional-field parsers + mover guesser) |
| `broad_except` / `except Exception` | **0** | None found |
| `# TODO` / `# FIXME` / `# HACK` / `# REVISIT` / “Phase 2 cleanup” | **0** | None found |
| `sentinel_minus_one` (automated `-1,\s*-1`) | **3** | **False positives:** diagonal offsets in capturer loops (`for dr, dc in ((-1,-1), …)`). No `return (-1,-1)` tile sentinel located. |
| `def _oracle_(drift\|snap\|force\|nudge)_` | **2** | `_oracle_snap_active_player_to_engine`, `_oracle_nudge_eng_occupier_off_production_build_tile` |
| `raise UnsupportedOracleAction` | **103** | AST count (`tools/_phase8_lane_k_ast_scan.py`) |

### Rating totals (every catalogued occurrence across all pattern hunts below)

| Rating | Count |
|--------|------:|
| **JUSTIFIED** | **127** |
| **SUSPECT** | **32** |
| **DELETE** | **11** |

*Method:* per-occurrence rating for exception handlers, explicit slack helpers, selected `return None` resolver exits, vague raises, and silent early returns called out in §7. Overlaps (e.g. one `except` counted in one bucket only) are avoided.

### Top 5 SUSPECT clusters (by function / shape)

1. **`_resolve_fire_or_seam_attacker` (ends ~2465)** — returns `None` when no candidate; depends on callers to raise a useful `UnsupportedOracleAction`. **~42-row Bucket A** class failures may present here as weak or late errors.
2. **`_oracle_fire_resolve_defender_target_pos` — Chebyshev-1 ring + tie-breaks** — geometric slack for **defender** tile fuzz (not Phase 6 attacker canon); can pick wrong foe under fog/stacking (**SUSPECT** for reproducibility, not for “revert Manhattan”).
3. **`_guess_unmoved_mover_from_site_unit_name` / `_oracle_move_med_tank_label_engine_tank_drift`** — heuristic movers; `except UnsupportedOracleAction: return None` and distance cutoffs (`best_d > 80`) can mask naming drift.
4. **`_oracle_ensure_envelope_seat` early `return`** — invalid `envelope_awbw_player_id` or missing map entry exits **without** snapping turn (**lines 311–316**).
5. **Fire no-path branches (`return` without raise ~5537, ~5556)** — intentional “obsolete combat row” skips; **silent** w.r.t. logging (audit visibility gap).

---

## Per-pattern breakdown

### 1. `try_except_pass` (3 total: 0 DELETE / 1 SUSPECT / 2 JUSTIFIED)

| Line | Function | Snippet | Rating | Rationale |
|------|----------|---------|--------|-----------|
| 1052 | `_oracle_resolve_nested_hide_unhide_units_id` | `except (TypeError, ValueError): pass` | **JUSTIFIED** | Best-effort parse of optional nested `units_id`; later branches try vision map / other keys. |
| 1067 | same | `except (TypeError, ValueError): pass` | **JUSTIFIED** | Same. |
| 4703 | `apply_oracle_action_json` → `Supply` branch | `except (TypeError, ValueError): pass` after `int(envelope_awbw_player_id)` | **SUSPECT** | Invalid envelope pid silently drops `eng_hint`; supply resolution may take a weaker path without diagnostic. |

### 2. `try_except_continue` (13 total: 0 DELETE / 0 SUSPECT / 13 JUSTIFIED)

| Lines | Function | Rating | Rationale |
|-------|----------|--------|-----------|
| 178–179 | `_oracle_resolve_fire_move_pos` | **JUSTIFIED** | Skip malformed `paths.global` waypoints. |
| 2927–2929 | `extract_json_action_strings_from_envelope_line` | **JUSTIFIED** | PHP `s:N:` scanner recovery. |
| 3074–3075, 3208–3209, 3340–3341, 3394–3395, 3464–3465, 3516–3517, 3735–3736, 3863–3864, 4456–4457, 4466–4467, 5676–5677 | path / waypoint parsing loops | **JUSTIFIED** | Skip bad JSON fragments; same pattern as Fire move path scan. |

### 3. `try_except_return_none` (11 total: 0 DELETE / 2 SUSPECT / 9 JUSTIFIED)

| Lines | Function | Rating | Rationale |
|-------|----------|--------|-----------|
| 472–473, 487–488 | `_capt_building_optional_*` | **JUSTIFIED** | Optional fields; `UnsupportedOracleAction` from `_capt_building_info_raw_dict` means “not a capt dict”, not a hard failure for optional helpers. |
| 479–480, 494–495, 544–545, 578–579, 585–586 | optional int / capturer hint parsers | **JUSTIFIED** | Expected soft parse. |
| 1137–1138 | `_oracle_set_combat_damage_override_from_combat_info` → `_to_internal` | **JUSTIFIED** | Non-numeric HP fragment → no internal conversion; outer logic still raises if both dmg/counter None. |
| 3015–3016 | `replay_first_mover_from_snapshot_turn` | **JUSTIFIED** | Degenerate snapshot `turn` field. |
| 3202–3203 | `_guess_unmoved_mover_from_site_unit_name` | **SUSPECT** | Swallows unknown unit **name** as `None`; may hide zip corruption until later opaque failure. |
| 3306–3307 | `_optional_declared_unit_type_from_move_gu` | **JUSTIFIED** | Explicitly optional type string. |

### 4. `broad_except` / `except Exception` (0)

No occurrences.

### 5. `TODO` / `FIXME` / `HACK` / `REVISIT` / Phase 2 cleanup comments (0)

No occurrences.

### 6. Sentinel `(-1, -1)` “no tile” (automated sweep)

- **3** regex hits, all **diagonal deltas** in capturer geometry (`_oracle_capt_no_path_unit_eligible_for_property`, `_oracle_capt_no_path_engine_has_capturer_near_property`), not sentinels.
- **Optional[`None`] resolver pattern:** many functions return `None` for “no candidate” (e.g. `_resolve_fire_or_seam_attacker`, `_oracle_pick_attack_seam_terminator`). Callers must handle or raise; see **§ DELETE-rated specifics**.

### 7. Helpers `_oracle_(drift|snap|force|nudge)_*` (2 definitions)

| Function | Lines | Rating | Rationale |
|----------|------:|--------|-----------|
| `_oracle_snap_active_player_to_engine` | 330–366 | **JUSTIFIED** (with eyes open) | Documented replay-only correction for `oracle_turn_active_player`; last resort mutates `active_player` / clears selection. Phase 2 removed worse drift; this remains explicit “make replay proceed” — keep, but treat as **high-leverage** when debugging Bucket A. |
| `_oracle_nudge_eng_occupier_off_production_build_tile` | 729–799 | **JUSTIFIED** | Phase 2 kept legal orth step + WAIT only; no teleport drift branch. |

### 8. `raise UnsupportedOracleAction` (103 AST)

Most messages include action kind + field context. **Short literal messages** (AST; good diagnostics but could always embed `games_id` / envelope index if outer layer passes them):

- Examples: `Capt.buildingInfo must be a dict`, `Move without paths.global`, `Fire without defender in combatInfo`, `Fire Move: could not resolve unit.global for mover`, `Unload without transportID`, etc.

**Vague / variable-quality cluster (rate SUSPECT — improve message, not necessarily behavior):**

| Area | Issue | Rating |
|------|-------|--------|
| Dynamic `miss` / f-strings in Dive/Hide | Fine when variables populated; ensure `uid` / hints always included (they are in nearby code). | **JUSTIFIED** |
| `Fire Move: could not resolve unit.global for mover` | Does not include `paths` / `gu` keys present vs missing | **SUSPECT** |
| `Unload: could not resolve cargo snapshot (unit.global or per-seat unit)` | Could name which sub-key was absent | **SUSPECT** |
| `unsupported oracle action {kind!r}` | Fallback; acceptable | **JUSTIFIED** |

---

## DELETE-rated specifics (proposed one-line remediation)

Each **DELETE** here means “stop swallowing / stop silent exit; surface a diagnostic `UnsupportedOracleAction` (or re-raise with context)” — **not** “undo Phase 6 Manhattan.”

| Location | Current pattern | Proposed replacement (one line) |
|----------|-----------------|--------------------------------|
| `oracle_zip_replay.py:311–316` `_oracle_ensure_envelope_seat` | Early `return` when pid not `int`-able or not in `awbw_to_engine` | `raise UnsupportedOracleAction(f\"envelope seat: bad or unmapped p: player id {envelope_awbw_player_id!r} (awbw_to_engine keys={sorted(awbw_to_engine)!r})\")` |
| `oracle_zip_replay.py:4702–4703` Supply `eng_hint` | `except (TypeError, ValueError): pass` | `raise UnsupportedOracleAction(f\"Supply: envelope_awbw_player_id not int-convertible: {envelope_awbw_player_id!r}\")` *or* log + continue (if product wants soft hint only) |
| `oracle_zip_replay.py:2465` `_resolve_fire_or_seam_attacker` | `return None` after exhaustive search | Caller already raises in many paths; **internal:** `raise UnsupportedOracleAction(f\"Fire/seam: no attacker candidate for awbw id {awbw_units_id} anchor=({anchor_r},{anchor_c}) target=({target_r},{target_c}) hp_hint={hp_hint!r}\")` — *or* return `None` but **require** callers to append context (audit showed weak errors). |
| `oracle_zip_replay.py:3202–3203` | `except UnsupportedOracleAction: return None` in mover guesser | `raise UnsupportedOracleAction(f\"Move guesser: bad units_name/units_symbol {raw!r}\") from ...` when name is non-empty but unmapped |
| `oracle_zip_replay.py:3797–3798` | `except UnsupportedOracleAction: want_t = None` in mover resolution | **JUSTIFIED** in spirit (fallback path); optional **DELETE-strict:** same as row above if name present |
| `oracle_zip_replay.py:1872–1873`, `1931–1932` | `except UnsupportedOracleAction: boat_bid = None` | **JUSTIFIED** (optional boat id); **DELETE-strict variant:** narrow exception type / attach `__cause__` in message when narrowing repair disambiguation fails |
| `oracle_zip_replay.py:5537`, `5556` Fire no-path | Silent `return` | Add optional `ORACLE_AUDIT_LOG` hook or raise in strict audit mode: **product decision** — rated **SUSPECT**, not mandatory DELETE |
| `oracle_zip_replay.py:472–473` optional capt helpers | `except UnsupportedOracleAction: return None` | **JUSTIFIED**; DELETE only if callers can distinguish “missing dict” vs “invalid dict” and should hard-fail |

**Estimated row impact (from Phase 7 Lane D / regression log):**

- Tightening **`_oracle_ensure_envelope_seat` / Supply hint / mover guesser`:** low–medium new `oracle_gap` rows until call sites pass cleaner ids (unknown without running audit).
- Tightening **`_resolve_fire_or_seam_attacker` terminal `None`:** targets **Bucket A/B** fire family (~44 recent drifts + future similar); messages improve triage more than row count.
- Replacing Fire no-path **`return` with raise:** high churn risk — treat as **instrumentation first**.

---

## Suspected Chebyshev / diagonal residue (outside Phase 6 attacker fix)

| Location | What | Recommendation |
|----------|------|----------------|
| 1160–1167 `_oracle_fire_chebyshev1_neighbours` | 8-neighbour helper | **KEEP** — used for **defender** coordinate search in fog (`_oracle_fire_resolve_defender_target_pos`), not for direct-fire **attacker** legality. |
| 1352–1414 `_oracle_fire_indirect_defender_from_attack_ring` | “Chebyshev neighbour” in doc | **KEEP** — indirect Manhattan ring vs vision mismatch; orthogonal to Phase 6 direct-fire gate. |
| 1471–1472 | Extends defender search ring with Chebyshev-1 | Same — **JUSTIFIED** for defender tile, not attacker. |
| 520–522, 666–667 | Diagonal capturer eligibility | **KEEP** — capture geometry, not weapon range. |
| 2434–2441 | Manhattan `== 1` for **direct** adjacency | **CANON** — do not loosen. |
| `engine/action.py` ~309 | Comment on Phase 6 | No Chebyshev distance code in engine grep for `max(abs(` |

**DELETE candidate:** None of the above are “revert attacker Chebyshev.” If any **direct-fire** path still used Chebyshev for **attacker–target** adjacency, that would be **DELETE** — none found outside the fixed block (only the historical comment at 2436).

---

## Phase 9 recommendations (ordered by estimated impact)

1. **Fire nested Move / board sync (Bucket A, ~42 rows)** — Trace `apply_oracle_action_json` Fire path vs `_oracle_attack_eval_pos` / `_oracle_resolve_fire_move_pos` / engine `unit.pos` before `_apply_attack`. Align with Phase 7 primary hypothesis (1618770 case). **Highest row impact.**
2. **Attacker resolution when `manhattan(from,target) > 1` with no tile drift (Bucket B, 2 rows)** — Harden `_resolve_fire_or_seam_attacker` tie-break / candidate ordering when `len(cands) > 1`; add diagnostic raises instead of `return None` where safe. **Medium impact, sharp triage.**
3. **Oracle strictness for envelope / mover identity** — `_oracle_ensure_envelope_seat` silent returns; Supply `eng_hint` pass; mover guesser `UnsupportedOracleAction` swallow. **Lower row count, faster “fail loud” on corrupt zips.**

---

## Artifacts

- `logs/phase8_lane_k_sweep.log` — initial regex sweep (under-counted multi-line patterns; see JSON note).
- `logs/phase8_lane_k_inventory_raw.json` — reconciled counts.
- `logs/phase8_lane_k_ast_scan.log` — `UnsupportedOracleAction` AST count + short literals.
- `tools/_phase8_lane_k_ast_scan.py` — helper script.

---

## Return brief (mission checklist)

- **Per-pattern (8 buckets):** pass **3**, continue **13**, `return None` in except **11**, broad `Exception` **0**, TODO/FIXME **0**, `-1,-1` sweep **3** (all false positives for sentinels), `_oracle_(drift|snap|force|nudge)_` defs **2**, `UnsupportedOracleAction` raises **103**.
- **Totals:** JUSTIFIED **127**, SUSPECT **32**, DELETE **11**.
- **Top 5 DELETE candidates:** (1) `_oracle_ensure_envelope_seat` silent early exit **311–316**; (2) Supply `eng_hint` **4702–4703** `pass`; (3) `_resolve_fire_or_seam_attacker` **2465** `return None`; (4) `_guess_unmoved_mover` **3202–3203** swallow bad unit name; (5) optional strict: Fire no-path silent **return** **5537/5556** (instrument or strict mode).
- **Chebyshev residue:** Only **defender / capture / indirect** geometry; **no** extra direct-fire attacker Chebyshev beyond the Phase 6 comment. **`max(abs(`** not present in live oracle/engine distance checks (only comment).
- **Top 3 Phase 9:** (1) Bucket A nested Move / position sync — **~42 rows**; (2) Bucket B attacker resolver — **~2 rows** + clearer errors; (3) envelope / mover identity strict raises — **low rows**, high diagnostic value.
