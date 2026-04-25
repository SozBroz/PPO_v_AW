# Phase 11J-F2-KOAL-FU-ORACLE — Closeout (read-only validation)

**Lane:** Oracle replay — `tools/oracle_zip_replay.py` (Capt contested-capture + envelope seat)  
**Date:** 2026-04-21  
**Design refs:** `docs/oracle_exception_audit/phase11j_f2_koal_fix.md` §6–8 (escalation), `docs/oracle_exception_audit/phase11d_f2_recon.md`  
**Mode:** Validation only — no source edits in this pass.

---

## Section 1 — Edit confirmation (Step 1)

**Verdict: YES** — FU-ORACLE intent is present in two places:

1. **`_oracle_ensure_envelope_seat`** — After mapping `envelope_awbw_player_id` → engine seat `want`, the helper returns immediately when `state.active_player` already equals `want`, avoiding `_oracle_advance_turn_until_player` / spurious `END_TURN` that would clear mid-envelope `cop_active` / `scop_active`.

2. **`kind == "Capt"` (nested Capt, no Move list)** — Comment block **Phase 11J-F2-KOAL-FU-ORACLE** references **gid 1630794** and documents contested vs neutral `buildingInfo` semantics. Logic sets `envelope_already_aligned` when the envelope’s `p:` seat matches `active_player` *and* `bpid`/`btid` maps to the **property defender** (previous owner); in that case `_oracle_ensure_envelope_seat` is **skipped** (`not envelope_already_aligned` guard).

`git diff HEAD -- tools/oracle_zip_replay.py` includes the FU-ORACLE hunk (line numbers shift with other edits in the same file):

```diff
@@ -294,10 +317,14 @@ def _oracle_ensure_envelope_seat(
     if pid not in awbw_to_engine:
-        return
+        raise UnsupportedOracleAction(
+            f"envelope seat: unmapped p: player id {pid} (awbw_to_engine keys={sorted(awbw_to_engine)!r})"
+        )
     want = int(awbw_to_engine[pid])
     if int(state.active_player) == want:
         return
```

Capt-branch hunk (abbreviated): adds `prop_pre` / `_maps_to_property_defender`, `envelope_already_aligned`, and only calls `_oracle_ensure_envelope_seat` when `seat_awbw is not None and not envelope_already_aligned`.

**Scope note:** The same `git diff` vs `HEAD` is **large** (~1.7k lines touched per `git diff --stat`): it bundles FU-ORACLE with broader oracle replay changes (`oracle_mode` stepping, fire-path resolution, repair/Capt strictness, removals, etc.). This closeout treats **FU-ORACLE** as the Koal/Capt/cop_active preservation slice above; full diff is not reproduced here.

---

## Section 2 — Files changed (git)

```text
 tools/oracle_zip_replay.py | 1718 ++++++++++++--------------------------------
 1 file changed, 462 insertions(+), 1256 deletions(-)
```

*(from `git diff --stat HEAD -- tools/oracle_zip_replay.py` at validation time)*

---

## Section 3 — Code edit summary

**Active-player short-circuit (`_oracle_ensure_envelope_seat`):**  
For any caller that passes the correct AWBW seat id for the envelope, if the engine is already on that seat, the function no longer runs finish/advance-turn machinery. That prevents an extra `END_TURN` → `_end_turn` from wiping **Koal COP** (or other) power flags before later actions in the same envelope (e.g. **Load**).

**Capt contested-capture guard:**  
When PHP `buildingInfo` names the **defender** on a property that is still enemy-owned at capture-start, blindly using that id for seat alignment used to advance the wrong half-turn. If `active_player` already matches the **`p:`** envelope owner and the building reference maps to the defender, the replay **trusts the envelope** and skips `_oracle_ensure_envelope_seat`, preserving `cop_active` through the Capt → subsequent Load sequence (**1630794** / env 37).

---

## Section 4 — 1630794 status

**`ok`** — Full replay completes (`actions_applied=1001`, no first divergence).  
Register: `logs/desync_register_post_phase11j_f2_fu_targeted.jsonl` (seed 1).

This satisfies the lane **primary win condition** (Koal COP +1 preserved through contested Capt → Load).

---

## Section 5 — Cross-check 1605367 + 1622104

| games_id | class (targeted audit, seed 1) | Notes |
|----------|-------------------------------|--------|
| **1605367** | `oracle_gap` | Move truncation / upstream drift at envelope 32 (~682 actions) — consistent with post–11J-F2-KOAL downstream drift; not `engine_bug`. |
| **1622104** | `oracle_gap` | Move truncation at envelope depth; not `engine_bug`. |

Both remain in **good shape** relative to design expectations (`ok` or `oracle_gap`, not hard engine illegality on the original Koal MP issue).

---

## Section 6 — Nine gates (pass/fail)

| # | Gate | Floor | Result |
|---|------|-------|--------|
| 1 | `python -m pytest tests/test_engine_negative_legality.py -v --tb=no` | 44p / 3xp | **PASS** — 44 passed, 3 xpassed |
| 2 | `python -m pytest tests/test_andy_scop_movement_bonus.py tests/test_co_movement_koal_cop.py --tb=no` | 7 passed | **PASS** — 7 passed |
| 3 | `python -m pytest tests/test_engine_legal_actions_equivalence.py::test_legal_actions_step_equivalence --tb=no` | 1 passed | **PASS** — 1 passed |
| 4 | `python -m pytest tests/test_co_build_cost_hachi.py tests/test_co_income_kindle.py tests/test_oracle_strict_apply_invariants.py --tb=no` | 15 passed | **PASS** — **25 passed** (same files; more tests collected than historical 15 floor) |
| 5 | `python -m pytest test_oracle_zip_replay.py -v --tb=no` | record count | **PASS** — **62 passed** |
| 6 | `python -m pytest --tb=no -q` | ≤2 failures | **PASS** — 1 failed (`test_trace_182065_seam_validation` — known Phase 11C-FU residual), 507 passed |
| 7 | Targeted re-audit (Step 2) | 1630794 → `ok` | **PASS** |
| 8 | `python tools/desync_audit.py --max-games 50 --seed 1 --register logs/desync_register_post_phase11j_fu_50.jsonl` | `engine_bug` ≤ 0 | **PASS** — ok=45, oracle_gap=5, **engine_bug=0** |
| 9 | `python tools/desync_audit.py --max-games 100 --seed 1 --register logs/desync_register_post_phase11j_fu_100.jsonl` | `engine_bug` ≤ 0 | **PASS** — ok=89, oracle_gap=11, **engine_bug=0** |

---

## Section 7 — 100-game cross-check vs FIRE-DRIFT baseline

**Baseline:** `logs/desync_register_post_phase11j_sample.jsonl`  
**Post-FU (this run):** `logs/desync_register_post_phase11j_fu_100.jsonl`  
Same 100 catalog games (seed 1).

| Metric | FIRE sample | Post-FU | Δ |
|--------|-------------|---------|---|
| `ok` | 92 | 89 | **−3** |
| `oracle_gap` | 8 | 11 | +3 |
| `engine_bug` | 0 | 0 | 0 |

**Per-gid class changes (100-game intersection):**

| games_id | FIRE sample → Post-FU | Note |
|----------|----------------------|------|
| 1607045 | `oracle_gap` → `ok` | **Improvement** |
| 1621434 | `ok` → `oracle_gap` | **Regression candidate** (Build no-op / insufficient funds) |
| 1621898 | `ok` → `oracle_gap` | **Regression candidate** (Build no-op) |
| 1622328 | `ok` → `oracle_gap` | **Regression candidate** (Build no-op) |
| 1624082 | `ok` → `oracle_gap` | **Regression candidate** (Build no-op) |

Strict interpretation of the campaign rule (“no fewer `ok` rows”) is **not met**: net −3 `ok` vs FIRE-DRIFT sample, with no new `engine_bug`. Treat the four `ok` → `oracle_gap` rows as **oracle strictness / drift surfacing** follow-ups, not Koal-lane blockers.

---

## Section 8 — Updated `engine_bug` residual count

| Slice | Count |
|-------|--------|
| Targeted (1605367, 1622104, 1630794), seed 1 | **0** `engine_bug` |
| First 50 games (`…_fu_50.jsonl`) | **0** `engine_bug` |
| First 100 games (`…_fu_100.jsonl`) | **0** `engine_bug` |
| vs FIRE 100-game sample | **0** `engine_bug` both |

Phase 11J-F2-KOAL doc cited first-100 post-KOAL engine fix as `engine_bug=0`; **that holds** on this oracle revision.

---

## Section 9 — Verdict

**YELLOW**

- **1630794 closes as `ok`** — primary FU-ORACLE objective achieved (`cop_active` preserved through contested Capt; Koal Load reaches AWBW path).
- **All nine gates meet numeric floors** (including `engine_bug` ≤ 0 on 50/100 samples; full pytest within ≤2 failures).
- **Caveat:** 100-game sample vs `desync_register_post_phase11j_sample.jsonl` shows **fewer `ok` (−3)** and four **`ok` → `oracle_gap`** flips (mostly Build no-op cluster), so the strict “no regression in `ok` count” bar is **not** cleared — recommend a short **oracle_gap triage** lane on those gids if parity with FIRE-DRIFT sample is required for GREEN.

Not **RED**: no return of `engine_bug` on lane targets or samples; 1630794 does not persist as `engine_bug`.

---

## Return summary (commander brief)

| Item | Value |
|------|--------|
| Edit confirmation | **YES** — `_oracle_ensure_envelope_seat` early return + Capt **Phase 11J-F2-KOAL-FU-ORACLE** contested path; comment cites **1630794**. |
| 1630794 final status | **`ok`** |
| 9 gates pass/fail | **9/9 pass** (gate 4: 25 tests; gate 5: 62 tests; gate 6: 1 known deferred failure) |
| `engine_bug` residual | **0** on targeted + 50 + 100; unchanged vs FIRE 100 baseline |
| Verdict letter | **YELLOW** (lane win + clean `engine_bug`; sample `ok` count regressed vs FIRE-DRIFT 100) |

---

*Phase 11J-F2-KOAL-FU-ORACLE closeout complete.*
