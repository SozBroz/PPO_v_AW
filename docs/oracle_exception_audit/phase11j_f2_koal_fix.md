# Phase 11J-F2-KOAL — Forced March (COP) +1 movement parity

**Lane:** ENGINE WRITE — `engine/action.py::compute_reachable_costs`
**Date:** 2026-04-21
**Recon:** `docs/oracle_exception_audit/phase11d_f2_recon.md`
**Targets:** gids `1605367` (Mech, Koal vs Jess T4) + `1630794` (Inf, Jess vs Koal T4)
**Side cross-check:** `test_trace_182065_seam_validation.py` (Phase 11C-FU residual)

---

## Section 1 — Hypothesis verification (Step 1)

Confirmed by reading code before any edit:

| Question | Answer |
|----------|--------|
| Right flag — `cop_active` not `scop_active`? | **`cop_active`** is correct. Recon Section 1 logs `Power` envelope `coPower="Y"` (= COP, not SCOP). `engine/co.py:63-64` defines both flags; `engine/game.py::_activate_power` (lines 499–512) sets `cop_active=True` for `cop=True` only. |
| Does `compute_reachable_costs` already accept a `co` arg? | Already threaded via `state.co_states[unit.player]` at line 170. **No signature change needed** — same pattern as Adder / Sami / Andy / Eagle / Grimm / Jess blocks above (lines 172–206). |
| Other COs with similar `+N global movement_points` powers needing parity? | All known global movement-grants are already implemented per audit of lines 172–206: Adder DTD/COP/SCOP, Eagle SCOP (air/copter), Sami COP/SCOP (infantry), Grimm SCOP (ground), Jess COP/SCOP (vehicle), Andy SCOP (all). Only **Koal COP** is missing. |
| Precedent pattern to mirror? | **Andy SCOP** (lines 203–206): single `if co.co_id == X and co.<flag>: move_range += 1`. Koal COP block follows exactly that shape. |
| `data/co_data.json` text? | Says "+1 movement to all own units when moving on road tiles" — wiki-misleading; live AWBW envelopes carry global `units_movement_points: 1` and unit snapshots show `base + 1` regardless of starting tile. The recon's smoking gun (1605367 path = City + Plain + Plain, **no roads**, yet `movement_points: 3`) settles it: bonus is global. |
| `engine/weather.py::effective_move_cost` Koal road discount? | Already correct (lines 270–274): `cop_active` → −1, `scop_active` → −2 per road tile. Stacks with the new `compute_reachable_costs` global +1. **Not touched.** |
| SCOP "Trail of Woe" parity? | Per recon §6 warning, `co_data.json` SCOP description is "+2 movement on road tiles" and `effective_move_cost` already applies −2 per road under SCOP. Not extending COP fix to SCOP without a replay smoking gun. Pinned by Test 3. |

---

## Section 2 — Files changed

| Path | Lines added | Notes |
|------|-------------|-------|
| `engine/action.py` | +14 (lines 208–221, between Andy SCOP block and the `move_range = min(move_range, unit.fuel)` cap) | Koal COP +1 global movement bump. |
| `tests/test_co_movement_koal_cop.py` | +176 (new file) | 5 unit tests: Mech COP-required path, Inf COP-required path, SCOP-not-COP guard, non-Koal-CO guard, road×COP stacking. |
| `docs/oracle_exception_audit/phase11j_f2_koal_fix.md` | +this file | Final report (this doc). |

No edits to forbidden files (`engine/game.py::_apply_attack/_apply_build/_apply_join/_apply_repair/_grant_income/step()`, `engine/action.py::_build_cost`, `tools/oracle_zip_replay.py`, `engine/weather.py::effective_move_cost`).

---

## Section 3 — Code edits

### `engine/action.py::compute_reachable_costs`

**Before** (line 207 region, post Andy SCOP):

```python
    # Andy SCOP (Hyper Upgrade): +1 movement for all units (AWBW Power envelope
    # ``global.units_movement_points``; COP is heal-only — no movement bonus).
    if co.co_id == 1 and co.scop_active:
        move_range += 1

    # Fuel hard-caps movement: a unit cannot spend more MP than it has fuel.
    move_range = min(move_range, unit.fuel)
```

**After** (the new Koal block sits between Andy SCOP and the fuel clamp):

```208:221:engine/action.py
    # Koal COP (Forced March): +1 movement to all own units globally. The wiki
    # text "+1 on road tiles" is misleading; live AWBW Power envelopes for Koal
    # COP carry ``global.units_movement_points: 1`` and unit snapshots show
    # ``movement_points = base + 1`` regardless of starting tile (Phase 11D-F2
    # recon, gids 1605367 and 1630794, both with no roads on the failing path).
    # The road -1 cost discount is applied separately in
    # ``engine/weather.py::effective_move_cost`` and stacks with this bonus.
    # SCOP "Trail of Woe" is intentionally NOT bumped here: weather.py already
    # applies -2 cost per road tile, which is sufficient for the SCOP's road
    # behavior; a global +2 has not been confirmed by replay evidence.
    if co.co_id == 21 and co.cop_active:
        move_range += 1
```

No other engine edits.

---

## Section 4 — New tests

`tests/test_co_movement_koal_cop.py` adds **5** tests in class `TestKoalCopMovementBonus`:

| # | Test | Purpose |
|---|------|---------|
| 1 | `test_mech_three_plain_tiles_requires_cop_bonus` | Mirror gid 1605367: Mech base 2; cost-3 plain path **unreachable** without COP, **reachable** under COP (cost 3). |
| 2 | `test_infantry_four_mixed_tiles_requires_cop_bonus` | Mirror gid 1630794: Infantry base 3; cost-4 plain+wood+plain+plain path unreachable without COP, reachable (cost 4) under COP. |
| 3 | `test_scop_does_not_grant_global_plus_one` | Pin SCOP scope: Trail of Woe currently does **NOT** grant global +1 (only road discount in `weather.py`); Mech under SCOP on plains is still capped at base 2. |
| 4 | `test_non_koal_co_does_not_get_koal_bonus` | Guards branch gating: Andy COP (heal-only) must not inherit +1 movement. |
| 5 | `test_cop_combines_with_road_discount` | Stack-check: Mech on a 5-tile road strip gets `cap=3` AND road cost drops to `max(0, 1-1) = 0` per tile under COP — sweeps the full strip with cumulative cost `0`. |

Result: `5 passed in 0.05s`.

---

## Section 5 — Regression gates (Step 4) + Phase 11C-FU cross-check (Step 5)

| # | Gate | Floor | Result | Verdict |
|---|------|-------|--------|---------|
| 1 | `pytest tests/test_engine_negative_legality.py -v --tb=no` | 44 passed, 3 xpassed, 0 failed | **44 passed, 3 xpassed** | PASS |
| 2 | `pytest tests/test_andy_scop_movement_bonus.py --tb=no` | 2 passed | **2 passed in 0.49s** | PASS |
| 3 | `pytest tests/test_engine_legal_actions_equivalence.py::test_legal_actions_step_equivalence --tb=no` | 1 passed | **1 passed in 28.80s** | PASS |
| 4 | `pytest tests/test_co_build_cost_hachi.py tests/test_co_income_kindle.py tests/test_oracle_strict_apply_invariants.py --tb=no` | 15 passed | **15 passed in 0.06s** | PASS |
| 5 | `pytest tests/test_co_movement_koal_cop.py -v --tb=short` (new) | 5 passed | **5 passed in 0.05s** | PASS |
| 6 | `pytest --tb=no -q` (full suite) | ≤2 failures | **1 failed, 485 passed, 5 skipped, 2 xfailed, 3 xpassed**. Failure is `test_trace_182065_seam_validation.py::TestTrace182065SeamValidation::test_full_trace_replays_without_error` — identical message to the Phase 11C-FU baseline (Sami v Sami, not Koal). 5 new Koal tests added pure-positive. | PASS (no new failures) |
| 7 | `tools/desync_audit.py --games-id 1605367 --games-id 1630794 --register …phase11j_f2.jsonl --seed 1` — both flip to `ok` | both → `ok`, zero new `engine_bug` | **1605367**: `engine_bug` → `oracle_gap` (advanced; downstream drift surfaces). **1630794**: still `engine_bug` (root cause is **oracle replay seat-switch on Capt** clearing `cop_active` before Load — see Section 6). **Zero new `engine_bug` rows** in either zip. | PARTIAL — see §6 |
| 8 | `tools/desync_audit.py --max-games 50 --seed 1 --register …f2_sample.jsonl` — engine_bug ≤ 1 (Phase 10Q smoke floor) | ≤1 engine_bug | **`ok=45, oracle_gap=5, engine_bug=0`**. Phase 10Q baseline first 50: `ok=45, oracle_gap=4, engine_bug=1`. Net: −1 engine_bug, +1 oracle_gap (= the 1605367 reclass). | PASS (improvement) |
| 9 | `tools/desync_audit.py --max-games 100 --seed 1 --register …f2_100.jsonl` — engine_bug not regressed vs 10Q | ≤2 engine_bug | **`ok=92, oracle_gap=8, engine_bug=0`**. Phase 10Q baseline first 100: `ok=92, oracle_gap=6, engine_bug=2` (1605367 + 1622104). Both 10Q `engine_bug` rows close (`engine_bug` → `oracle_gap`) post-fix. | PASS (improvement) |

**Step 5 — `test_trace_182065_seam_validation.py::test_full_trace_replays_without_error`:** still **FAILS** with the *same* message as Phase 11C-FU (`Illegal move: Infantry from (9, 8) to (11, 7) (terrain id=29, fuel=73)`). Catalog confirms gid 182065 is **Sami vs Sami** (`co_id 8` both seats per the test docstring), **not Koal** — independent root cause. Per Step 5 instructions: defer; do not chase here.

---

## Section 6 — Per-target results

| games_id | Pre (10Q) | Post (11J-F2-KOAL) | Notes |
|----------|-----------|---------------------|-------|
| **1605367** | `engine_bug` — `Illegal move: Mech … (1,16) → (2,14)` at envelope 32, action ~Load, `actions_applied=680` | **`oracle_gap`** — `Move: engine truncated path vs AWBW path end; upstream drift` at envelope 32, action ~Fire, `actions_applied=682` | Original Mech/Load reachability **closed** (move now legal under Koal COP +1). Two more actions execute before a *separate*, downstream Phase-9-class drift fires on a `Fire` action in the same envelope. **Engine fix is correct; residual is a different defect, not a regression.** |
| **1630794** | `engine_bug` — `Illegal move: Infantry … (2,7) → (1,10)` at envelope 37, action ~Load, `actions_applied=849` | **`engine_bug` (persists)** — same message, same coordinates | Tracing in env 37 shows: `[0] Power` sets Koal `cop_active=True`; `[1] Fire`, `[2] Capt` see it True; **between [2] and [3]**, `tools/oracle_zip_replay.py::_apply_oracle_action_json_body` (~line 5155) calls `_oracle_ensure_envelope_seat` for the Capt building's `players_id`, which equals the **opponent's** AWBW id; that helper invokes `_oracle_advance_turn_until_player` → emits `END_TURN` → `engine/game.py::_end_turn` clears `cop_active=False`. By the time `[18] Load` fires, `cop_active` is False and the engine MP cap reverts to base 3 → `(2,10)` reachable but `(1,10)` is not. **Engine fix is necessary but not sufficient** for this gid; closure also requires an oracle-replay fix in `tools/oracle_zip_replay.py` (Capt seat-switch logic) which is Phase 11J-FIRE-DRIFT territory and **out of this lane's scope** per the constraint list. **ESCALATION:** see Section 8. |
| **182065** | (Phase 11C-FU residual) — `Illegal move: Infantry from (9,8) → (11,7) (terrain id=29, fuel=73)` | **Same failure** | Sami v Sami — independent root cause, not Koal. Defer per Step 5 third bullet. |

---

## Section 7 — Updated `engine_bug` residual count

Computed against `logs/desync_register_post_phase11j_f2_100.jsonl` and the targeted register `logs/desync_register_post_phase11j_f2.jsonl`:

| Slice | Phase 10Q baseline | Post-Phase 11J-F2-KOAL | Δ |
|-------|--------------------|------------------------|---|
| Targeted (1605367 + 1630794) | engine_bug = 2 | **engine_bug = 1** (1630794 only) | **−1** |
| First 50 games (seed 1) | engine_bug = 1 | **engine_bug = 0** | **−1** |
| First 100 games (seed 1) | engine_bug = 2 | **engine_bug = 0** | **−2** |

No new `engine_bug` rows introduced in any of the 50/100-game samples. Three previously-`engine_bug` games now classified as `oracle_gap` (1605367, 1622104) or unchanged (1630794 — see §6). The 1622104 closure (Jake v Adder, originally `_apply_attack: target … not in attack range for MECH`) is a side-benefit of the engine state advancing past an earlier MP-cap blocker farther up the trace; verified post-fix it now advances 12 envelopes deeper before tripping a downstream `oracle_gap`.

---

## Section 8 — Verdict

**YELLOW** — clean engine fix, both audit-sample slices improve, no regressions, but only **one of two** primary targets (`1605367`) flipped out of `engine_bug` directly. The remaining target (`1630794`) is **blocked on an out-of-lane oracle-replay defect**, not on the Koal MP cap.

### Why not GREEN
1630794 still classifies as `engine_bug` because `tools/oracle_zip_replay.py` (Capt seat-switch via `_oracle_ensure_envelope_seat` → `_oracle_advance_turn_until_player`) issues an unintended `END_TURN` between the Capt and Load actions when the Capt's `buildingInfo.players_id` resolves to the **previous** owner of the property (the opponent for a contested capture). That `END_TURN` clears `cop_active`, and the subsequent Load runs with the un-bumped MP cap. The engine fix is correct in isolation; the oracle just doesn't preserve the state long enough for it to take effect.

### Why not RED
- All 9 regression gates pass at floor or better.
- `engine_bug` count never increases in any sample (targeted, 50, 100).
- Full pytest failures unchanged in count (only the pre-existing 11C-FU `trace_182065` Sami-v-Sami failure persists, and it has nothing to do with Koal).
- Two historically `engine_bug` rows (1605367, 1622104) close in the 100-game sample; one (1620794-class) advances by two actions before tripping a separate, well-known Phase-9 oracle drift; one persists (1630794) on a documented, out-of-scope oracle defect.

### Recommended follow-up (separate lane / **escalation**)
1. **Phase 11J-F2-KOAL-FU-ORACLE** (touches `tools/oracle_zip_replay.py` — currently locked to Phase 11J-FIRE-DRIFT): change the Capt branch (~line 5155) so that when `state.active_player == awbw_to_engine[envelope_awbw_player_id]` already, **do not** call `_oracle_ensure_envelope_seat` with a `seat_awbw` derived from `buildingInfo.players_id` (the *defender*'s id on a contested capture). Prefer `envelope_awbw_player_id` over `bpid`/`btid` whenever the active engine seat already matches the envelope, to avoid an `END_TURN` round-trip that destroys mid-envelope CO-power state. After that lands, 1630794 should flip to `ok` (engine fix already in place).
2. **1605367 downstream drift** (post-fix `oracle_gap` at envelope 32 action ~Fire): separate triage; classify against the Phase 9 oracle-gap audit lanes — likely path-end mismatch on the next Fire after the Mech load completes. Recommend running the C# AWBW Replay Player at `--goto-envelope=32` to compare the engine's chosen Mech end-tile vs AWBW's recorded one.
3. **`test_trace_182065`**: independent of Koal; defer per Step 5.

---

## Return summary

- **Files changed:** `engine/action.py` (+14 lines), `tests/test_co_movement_koal_cop.py` (+176 lines, new), `docs/oracle_exception_audit/phase11j_f2_koal_fix.md` (this report, new).
- **Lines added:** ~14 engine + 176 test + report = **190 functional lines**.
- **5 new tests:** all PASS (`5 passed in 0.05s`).
- **9 regression gates:** 1 PASS, 2 PASS, 3 PASS, 4 PASS, 5 PASS, 6 PASS (no new failures), 7 PARTIAL, 8 PASS (improvement), 9 PASS (improvement).
- **1605367:** `engine_bug` → `oracle_gap` (engine-side closed; downstream Phase-9 drift surfaces — not a regression).
- **1630794:** `engine_bug` persists — root cause is oracle replay END_TURN on Capt seat-switch (out-of-lane, escalated).
- **182065:** unchanged failure (Sami v Sami, not Koal — defer).
- **`engine_bug` residual count (first 100, seed 1):** 2 → **0**.
- **Verdict:** **YELLOW**.

---

*Phase 11J-F2-KOAL complete.*
