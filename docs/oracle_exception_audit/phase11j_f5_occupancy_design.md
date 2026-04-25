# Phase 11J-F5-OCCUPANCY-DESIGN — Forced fire stance + stacked tile (`games_id` 1626642)

**Mode:** design only (no code changes in this phase)  
**Prerequisite recon:** `docs/oracle_exception_audit/phase11d_f5_recon.md`  
**Coordination:** Phase **11J-FIRE-DRIFT** (in flight) owns `tools/oracle_zip_replay.py` Fire envelope handling.

---

## Section 1 — Function inventory: `_move_unit_forced`

**Definition**

| File | Line | Role |
|------|------|------|
| `engine/game.py` | 1499 | `GameState._move_unit_forced(unit, new_pos)` — teleports `unit.pos` without reachability checks. |

**Call sites (production Python)**

| # | File | Line | Calling context | Destination may be occupied? |
|---|------|------|-----------------|------------------------------|
| 1 | `tools/oracle_zip_replay.py` | 4018 | **`_apply_move_paths_then_terminator`**: after a Move step, when `json_path_was_unreachable` and `_oracle_path_tail_occupant_allows_forced_snap` (empty / self / legal JOIN only — not LOAD). | **Sometimes yes** — JOIN partner or “self” edge cases; explicitly **excludes** arbitrary friendly occupant (see `_oracle_path_tail_occupant_allows_forced_snap`). |
| 2 | `tools/oracle_zip_replay.py` | 5771 | **Fire envelope, post-kill duplicate path**: snap `mover_pk` to JSON path tail `tail_pk` when tail not in `compute_reachable_costs` and occupant rules allow (mirrors Lane G/L). | **Sometimes yes** — gated by same occupant helper family as (1). |
| 3 | `tools/oracle_zip_replay.py` | 5922 | **Fire envelope with `Move.paths` (Lane G / Bucket A)**: `fire_pos` ∉ `costs_fire` but `get_attack_targets(state, u, fire_pos)` still contains defender → `_move_unit_forced(u, fire_pos)` before `SELECT_UNIT` + `ATTACK`. | **Yes — this is the 1626642 failure class:** tail can be a legal *firing* hex in range math while blocked in reachability; incumbent friendly can remain on that hex in engine state. |
| 4 | `tools/oracle_zip_replay.py` | 5977 | **Post-`ATTACK` nested move reconciliation**: when `json_fire_path_was_unreachable` and tail snap is allowed, force mover to `json_fire_path_end` and align `selected_move_pos`. | **Sometimes yes** — same gating as (1). |
| 5 | `tools/export_awbw_replay_actions.py` | 525 | **Export / trace recovery**: on `ValueError` from `state.step(action)` (diverged replay), force `moving_unit` to `end` and `_finish_action`. | **Typically expects** cleared firing tile; if state is already wrong, could stack — out of scope for 1626642 but same helper. |

**Non-call references**

- `tests/test_oracle_move_resolve.py` (~109): documents policy “do not force onto another live unit” for plain Move truncation.
- Docstrings / campaign markdown reference the helper; not additional call sites.

**Count:** **4** oracle/export call sites + **1** definition (5 executable call sites total).

---

## Section 2 — Current behavior trace: `_move_unit_forced`

**Verbatim behavior** (`engine/game.py` ~1499–1512):

1. If `new_pos == unit.pos`: **return** (no-op).
2. If the unit’s **old** tile has a capturable property with `capture_points < 20`, reset capture to **20** (abort partial capture when leaving).
3. Set **`unit.pos = new_pos`**.
4. **Does not** adjust fuel, **does not** call `_move_unit`, **does not** call `_finish_action`.
5. **Does not** read or write `selected_unit`, `selected_move_pos`, or `action_stage`.

**Implication:** Any other alive unit that already had `pos == new_pos` remains there — the engine has **no occupancy grid**; `get_unit_at(row, col)` scans `state.units` and returns the **first** alive match.

---

## Section 3 — Failure mode walkthrough (`games_id` 1626642, envelope 22, blob 6)

Aligned with recon §3.4; engine row/col = AWBW `(y, x)`.

1. **Before blob 6 of envelope 22:** MED_TANK at **(2, 6)**, friendly **BLACK_BOAT** at **(1, 3)**, enemy **TANK** at **(2, 3)**.
2. **`_oracle_resolve_fire_move_pos`** returns **`fire_pos = (1, 3)`** — ZIP tail / stance where Manhattan direct fire hits `(2, 3)` even though **`(1, 3) ∉ compute_reachable_costs`** for the MED_TANK (blocked / drift vs AWBW path).
3. **`fire_pos not in costs_fire`** and **`get_attack_targets(state, u, fire_pos)`** contains the defender → oracle calls **`state._move_unit_forced(u, (1, 3))`** (`oracle_zip_replay.py` ~5915–5922).
4. **After forced move:** **Both** MED_TANK and BLACK_BOAT have **`pos == (1, 3)`** (illegal stacking in real AWBW; possible in oracle state).
5. Oracle sets `start = u.pos` → **(1, 3)**, builds **`SELECT_UNIT`** with **`select_unit_id=su_id`** (MED_TANK’s `unit_id`) — selection is **correct**.
6. **`ActionType.ATTACK`** is built with **`unit_pos=start=(1, 3)`**, **`move_pos=fire_pos=(1, 3)`**, **`target_pos=(2, 3)`**, and **no `select_unit_id`** (`oracle_zip_replay.py` ~5949–5956).
7. **`_apply_attack`** does **`attacker = self.get_unit_at(*action.unit_pos)`** (~632) — **not** `get_unit_at_oracle_id`. Scan order returns the **BLACK_BOAT** first.
8. Range check uses BLACK_BOAT vs TANK → **`get_attack_targets`** empty → **`ValueError`** with misleading “BLACK_BOAT … not in attack range” text.

**Root symptom:** ATTACK addressing uses **tile-only** lookup while Fire pipeline already proved **`select_unit_id`** is necessary for `SELECT_UNIT` on stacks (`Action` docstring in `engine/action.py`).

---

## Section 4 — Fix options

### Option A — Oracle + small engine: pin ATTACK to `select_unit_id`

**Engine**

- In **`GameState._apply_attack`**, resolve attacker with  
  **`get_unit_at_oracle_id(*action.unit_pos, action.select_unit_id)`**  
  instead of **`get_unit_at`** only.
- **`get_unit_at_oracle_id`** already exists (~185–200): if `select_unit_id` is set, match alive unit at tile **and** `unit_id`; else fall back to `get_unit_at`. **Backward compatible** for all existing ATTACK actions with `select_unit_id=None`.

**Oracle**

- Where Fire (and any other) oracle path builds **`Action(ActionType.ATTACK, ...)`** with a known striker `u`, set **`select_unit_id=int(u.unit_id)`**.
- **Primary site:** Fire-with-path block (~5923–5957) already defines **`su_id = int(u.unit_id)`** for `SELECT_UNIT` — **thread the same into `ATTACK`**.
- **Audit:** Other `ActionType.ATTACK` constructions in `oracle_zip_replay.py` (e.g. Fire **no-path** ~5710–5718, AttackSeam no-path ~6107–6116) should pass **`select_unit_id`** whenever the striker `u` is resolved — same class of bug if a stack ever appears there.

**Pros**

- **Localized**, reuses existing API; **no** new displacement heuristics.
- **Does not** change `_move_unit_forced` semantics for Move/Join/export paths.
- **~1 line** engine change + **1 field** per ATTACK construction that needs it.

**Cons**

- **Does not remove duplicate `pos`:** two units can still share a tile until something else moves them. Risk is **residual** if later code assumes unique positions (less common than `get_unit_at` on ATTACK). Longer-term **F1 / board sync** (recon §6) still reduces how often the boat remains on the tank’s hex.

### Option B — Engine: `_move_unit_forced` resolves incumbent

**Behavior sketch**

- Before `unit.pos = new_pos`, inspect **`incumbent = get_unit_at(new_pos)`** (excluding `unit` if `incumbent is unit` already handled).
- **Same `unit_id`:** no-op branch already covered by `new_pos == unit.pos` / identity.
- **Friendly incumbent:** displace to a chosen adjacent legal tile (terrain + occupancy), or under **`oracle_strict`** raise **`IllegalActionError`** when no safe nudge exists.
- **Enemy incumbent:** should be rare for forced fire stance; likely **raise** or strict fail.

**Pros**

- Restores **single-occupant invariant** at the forced hex; **`get_unit_at`** becomes unambiguous again for that tile.

**Cons**

- **Displacement policy is under-specified** for naval tiles (1626642: where does the BLACK_BOAT go?) — wrong choice **amplifies drift**.
- Touches **all five** call sites’ semantics (Move tail snap, duplicate Fire, export recovery).
- Higher regression surface; more LOC and tests.

---

## Section 5 — Recommended option

**Recommend Option A** (`select_unit_id` on **`ATTACK`** + **`_apply_attack`** uses **`get_unit_at_oracle_id`**).

**Rationale**

1. **`get_unit_at_oracle_id` + `Action.select_unit_id` already exist** for exactly this “AWBW stack / drawable ambiguity” case (`engine/action.py` ~85–87, `game.py` ~185–200). **`SELECT_UNIT` already uses it; `ATTACK` is the inconsistent outlier.**
2. **Call-site count:** forcing `_move_unit_forced` to displace incumbents affects Move/Join/Fire/export uniformly; Option A **only** tightens attack dispatch.
3. **1626642 is proven** to be wrong **`get_unit_at`** on ATTACK after correct **`SELECT_UNIT`** — the smallest fix matches the failure.
4. **Option B** should stay in reserve for a future “unique pos invariant” campaign once displacement rules are nailed down with viewer/site evidence.

**Counsel:** If after Option A a **later** step in the same envelope still assumes a unique occupant at `(1, 3)`, triage that call site the same way (pin `unit_id` or fix upstream F1 drift), rather than guessing boat slides.

---

## Section 6 — Test design (3–5 tests)

1. **`test_fire_forced_stance_duplicate_tile_pins_attacker_gl_1626642`** (or name with `select_unit_id`): Minimal reproduction: friendly unit A and striker B on same tile after `_move_unit_forced` (or direct assignment), `ATTACK` with `select_unit_id=B.unit_id` → combat proceeds as B; without pin, expect legacy `ValueError` or assert wrong attacker type. Prefer **synthetic** layout matching recon coordinates/types if full zip is heavy.

2. **`test_fire_forced_stance_empty_tile_regression`**: Striker alone on firing hex, `select_unit_id` set → **identical** outcome to pre-change (damage, positions). Guards against regressions in `_apply_attack` / `get_unit_at_oracle_id` fallback.

3. **`test_apply_attack_select_unit_id_none_falls_back_to_get_unit_at`**: No duplicate at `unit_pos`, `select_unit_id=None` → behavior unchanged (existing tests likely cover; add explicit if missing).

4. **(Option B only)** **`test_move_unit_forced_oracle_strict_raises_on_friendly_incumbent`**: If Option B is ever chosen, strict mode rejects ambiguous stack instead of silent displacement.

5. **`test_replay_1626642_envelope_22_no_engine_bug`** (integration): Run oracle zip replay through **envelope ≥ 22**; expect **no** `engine_bug` row for the BLACK_BOAT range string (or `desync_audit` equivalent passes). May be slower — gate as integration / optional in CI.

---

## Section 7 — File partition + 11J-FIRE-DRIFT conflict assessment

**Option A touches**

| Area | File | Approximate region |
|------|------|-------------------|
| Attacker resolution | `engine/game.py` | `_apply_attack` ~631–636 |
| Fire ATTACK construction | `tools/oracle_zip_replay.py` | ~5949–5956 (with-path Fire); also audit ~5710–5718, ~6107–6116 |
| (Optional) tests | `tests/test_oracle_fire_resolve.py` or new module | New cases |

**11J-FIRE-DRIFT** (expected scope): `apply_oracle_action_json` **Fire** branch, `_oracle_resolve_fire_move_pos`, nested Fire / path handling — **same file and overlapping line band (~5700–5985)** as Option A’s oracle edits.

**Conflict with 11J-FIRE-DRIFT:** **Yes** (high probability of concurrent edit on `tools/oracle_zip_replay.py` Fire pipeline).

**Recommendation:** **(a) Wait for 11J-FIRE-DRIFT to land**, then implement Option A on top. **(b)** Option B (`engine/game.py` only) **avoids oracle merge conflict** but is **not** the preferred fix for reasons in Section 5.

---

## Section 8 — Implementation checklist (cold pickup)

1. **Land / rebase on** Phase 11J-FIRE-DRIFT’s `tools/oracle_zip_replay.py` changes.
2. **`engine/game.py` — `_apply_attack`:** Replace  
   `attacker = self.get_unit_at(*action.unit_pos)`  
   with  
   `attacker = self.get_unit_at_oracle_id(*action.unit_pos, action.select_unit_id)`.
3. **`tools/oracle_zip_replay.py`:** For every `Action(ActionType.ATTACK, ...)` where striker `u` is known, add **`select_unit_id=int(u.unit_id)`**. Minimum: Fire-with-path block (~5951–5956). **Grep** `ActionType.ATTACK` in this file and patch any other oracle-emitted ATTACK that can run after a forced stance or on a stack.
4. **Run** `python -m pytest` on affected tests (`test_oracle_fire_resolve`, `test_engine_*`, any zip replay test).
5. **Replay validation:** `1626642.zip` through envelope 22+; confirm register / audit no longer reports the misleading BLACK_BOAT range `engine_bug` for this gid.
6. **Document** in campaign notes: misleading register string = **stacked tile + `get_unit_at`**, not AWBW attributing strike to Black Boat.

---

## Section 9 — Complexity + LOC estimate

| Item | Estimate |
|------|----------|
| **Complexity** | **LOW** (Option A) — **MED** (Option B) |
| **LOC (Option A)** | **~5–20** (1 engine line + 1–3 `Action` kwargs + tests) |
| **LOC (Option B)** | **~30–60** + policy tests + risk of AWBW-mismatch |

**Verdict letter:** **GREEN** — clear forward path for Option A after FIRE-DRIFT lands; no fundamental blocker.

---

*End of Phase 11J-F5-OCCUPANCY-DESIGN.*
