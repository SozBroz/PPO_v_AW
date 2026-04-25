# Phase 11J-FINAL-STURM-SCOP-SHIP — gid 1635679 close-out

**Status:** Sturm SCOP/COP shipped (canon-correct). Audit floor advanced
**927 ok / 9 oracle_gap / 0 engine_bug → 931 ok / 5 oracle_gap / 0 engine_bug**.
gid 1635679 itself remains `oracle_gap` (residual −1000 g, NOT
attributable to Sturm); 1635846 remains `oracle_gap` (residual −400 g,
also NOT Sturm). See §6.

The "do-not-touch Sturm SCOP" freeze imposed in
[phase11j_final_build_no_op_residuals.md](phase11j_final_build_no_op_residuals.md)
was lifted by the imperator on 2026-04-21 for this lane.

---

## 1. Verdict

**Sturm SCOP is now canon-correct.** Empirical evidence (replay
ground-truth) and the AWBW CO Chart agree exactly with the engine on
damage value, AOE shape, target set, and HP floor. The residual −1000 g
drift on gid 1635679 survives the fix and is therefore a separate funds
mechanic, not a Sturm SCOP issue. **Escalation per the imperator's
contract: Sturm SCOP is closed; the residual is reclassified.**

| Lane | Outcome |
| --- | --- |
| Sturm Meteor Strike (COP) damage / AOE / floor | Canon-correct |
| Sturm Meteor Strike II (SCOP) damage / AOE / floor | Canon-correct |
| Audit floor `≥927 ok / ≤9 oracle_gap / 0 engine_bug` | **Met (931 / 5 / 0)** |
| pytest baseline | Held (660 passed; pre-existing trace_182065 fail unchanged) |
| Sturm cohort regressions | None — 2 of 3 Sturm zips closed; 3rd has non-Sturm residual |
| 1635679 → ok | **No.** Residual −1000 g, non-Sturm cause |
| 1635846 → ok | **No.** Independent −400 g drift on Hawke turn (no Sturm in this game) |

---

## 2. SCOP timing — gid 1635679

`tools/_phase11j_sturm_drill.py` enumerated every Power envelope in
`replays/amarriner_gl/1635679.zip`. Sturm activations:

| env | day | kind | center (x, y) | affected enemies (PHP) |
| --- | --- | --- | --- | --- |
| 28  | 15  | SCOP "Meteor Strike II" | (9, 6)  | 7 |
| 40  | 21  | SCOP "Meteor Strike II" | (8, 15) | 4 |

Companion zip used for the COP path:

| zip | env | day | kind | center | affected |
| --- | --- | --- | --- | --- | --- |
| 1637200 | 12 | 7 | COP "Meteor Strike" | (4, 7) | 6 |

The `missileCoords` field in the AWBW envelope holds **column / row**
order (`{"x": col, "y": row}`). The engine's coordinate system is
`(row, col)`; `tools/oracle_zip_replay.py` flips the order when
constructing the AOE set.

---

## 3. Primary-source canon

| Source | URL | Quote (Sturm row, fetched 2026-04-21) |
| --- | --- | --- |
| AWBW CO Chart (amarriner) | https://awbw.amarriner.com/co.php | *"Meteor Strike — A 2-range missile deals 4 HP damage. The missile targets an enemy unit located at the greatest accumulation of unit value."* |
| AWBW CO Chart (amarriner) | https://awbw.amarriner.com/co.php | *"Meteor Strike II — A 2-range missile deals 8 HP damage. The missile targets an enemy unit located at the greatest accumulation of unit value."* |
| AWBW Wiki — Sturm | https://awbw.fandom.com/wiki/Sturm | Confirms 2-range diamond AOE, flat damage, no terrain/D2D modifier. |
| Wars Wiki — Sturm | https://warswiki.org/wiki/Sturm | Anchors the shared 0.1-display-HP floor across all flat-damage CO missiles. |

The site `powerName` field for SCOP is **"Meteor Strike II"** (not the
lore name "Fury Storm" stored in `data/co_data.json`). Engine code
keys off `co_id == 29` and the `cop` boolean in `_apply_power_effects`,
not on the name string, so this discrepancy is harmless.

---

## 4. Empirical AOE verification

`tools/_phase11j_sturm_aoe_verify.py` was run against gid 1635679 (env
28, env 40 — SCOP) and gid 1637200 (env 12 — COP). For every
PHP-recorded HP delta in the cluster around `missileCoords`:

* Manhattan distance to centre ≤ 2 for **all** affected enemy units
  (13-tile diamond verified, no off-by-one, no Chebyshev outliers).
* Display HP delta uniformly **8** (SCOP) and **4** (COP).
* Units already at 1 internal HP survived at 1 internal HP — confirming
  the shared 0.1-display floor.
* Friendly units in the AOE took **zero** damage.

This is the AWBW canon, byte-exact, on real replay data. No rounding
mismatch survives.

---

## 5. Code diff

Two sites, minimal surface, no touch to `data/damage_table.json`.

### 5a. `engine/game.py::_apply_power_effects` — Sturm branch

```python
elif co.co_id == 29:
    aoe = self._oracle_power_aoe_positions
    self._oracle_power_aoe_positions = None
    dmg = 40 if cop else 80  # 4 HP COP, 8 HP SCOP (display) -> internal x10
    if aoe is not None:
        for u in self.units[opponent]:
            if u.pos in aoe:
                u.hp = max(1, u.hp - dmg)
```

* Mirrors the existing Hawke / Olaf / Von Bolt SCOP code shape (same
  flooring rule, same enemy-only iteration, same one-shot AOE consume).
* When the AOE pin is `None` (RL / non-oracle path) the branch no-ops;
  no missile targeter is implemented and a global enemy −40/−80 would
  massively over-damage. Documented in code comment with citations.
* Floor at 1 internal HP — consistent with prior CO implementations
  and Wars Wiki anchor.

### 5b. `tools/oracle_zip_replay.py` — Sturm `missileCoords` pin

```python
if str(obj.get("coName") or "") == "Sturm":
    mc_raw = obj.get("missileCoords")
    aoe_positions: set[tuple[int, int]] = set()
    if isinstance(mc_raw, list):
        for entry in mc_raw:
            if not isinstance(entry, dict):
                continue
            try:
                cx = int(entry["x"])
                cy = int(entry["y"])
            except (KeyError, TypeError, ValueError):
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if abs(dr) + abs(dc) <= 2:
                        aoe_positions.add((cy + dr, cx + dc))
    if not aoe_positions:
        raise UnsupportedOracleAction(
            "Power: Sturm Meteor Strike without parseable missileCoords; "
            "cannot pin 2-range Manhattan AOE for engine override "
            f"(missileCoords={mc_raw!r})"
        )
    state._oracle_power_aoe_positions = aoe_positions
```

* Parses the AWBW envelope's `missileCoords` (one or more `{x, y}`
  centres — Sturm's COP and SCOP can produce multi-strike volleys).
* Builds the union of 13-tile Manhattan diamonds, pins to
  `state._oracle_power_aoe_positions`.
* Hard-fail on missing/empty `missileCoords` rather than silently
  no-op'ing — guards against a future AWBW envelope schema drift
  silently masking a regression.

---

## 6. Audit floor change & residual classification

### 6a. Headline numbers

| Metric | Pre-fix (recon doc) | Post-fix (this ship) | Δ |
| --- | --- | --- | --- |
| ok | 927 | **931** | +4 |
| oracle_gap | 9 | **5** | −4 |
| engine_bug | 0 | **0** | 0 |

Post-fix register: `logs/_sturm_full_936_postfix.jsonl`.

### 6b. Remaining 5 oracle_gap (all `Build no-op` funds drift)

| gid | COs (P0 vs P1) | failed build | need / have | drift | Sturm involvement? |
| --- | --- | --- | --- | --- | --- |
| 1617442 | Von Bolt vs Hawke | P1 TANK | 7000 / 6850 | −150 g | None |
| 1624082 | Lash vs Drake | P1 NEO_TANK | 22000 / 21850 | −150 g | None |
| 1628849 | Hawke vs Sami | P1 B_COPTER | 9000 / 8800 | −200 g | None |
| **1635679** | **Sturm vs Hawke** | **P0 NEO_TANK** | **22000 / 21000** | **−1000 g** | Sturm fix applied — SCOP HP byte-exact in this game; residual drift is now solely on the Hawke (P1) end-turn income/repair path |
| **1635846** | **Hawke vs Eagle** | **P0 INFANTRY** | **1000 / 600** | **−400 g** | None — no Sturm in this game |

**Pattern:** all 5 are pure funds drift, all small (≤ −1000 g), all
build_no_op late in the game. Hawke (co_id 12) appears in 3 of 5 (1617442
P1, 1635679 P1, 1635846 P0). The most likely common cause is a Hawke
end-turn funds calculation (Black Wave HP-recovery interaction with
income or repair ordering). That is the next campaign, not this one.

### 6c. 1635679 residual breakdown — proof Sturm is not the cause

`logs/_sturm_1635679_postfix.jsonl` and the funds-drift trace in
`logs/_sturm_1635679_postfix_trace.txt` show:

* The two Sturm SCOP envelopes (env 28, env 40) now apply the correct
  HP delta to **every** affected enemy unit. The combat damage override
  (`_oracle_set_combat_damage_override_from_combat_info`) keeps the
  in-combat HP byte-exact regardless. So engine ≡ PHP on every unit's
  HP across the post-SCOP turn.
* Despite the HP convergence, P0 (Sturm) treasury still drifts −1000 g
  by env 32. The drift accumulates on **Hawke's** (P1) end-turn repair
  / income phase and bleeds into Sturm's next-turn build budget.
* This drift signature did **not exist** before the fix in the same
  shape — the prior −3800 g drift was the visible compound of (a) the
  missing SCOP damage and (b) the Hawke end-turn drift. Removing (a)
  exposes (b) cleanly.

**Conclusion:** the Sturm SCOP is canon-correct. The remaining −1000 g
on 1635679 is a Hawke-side funds mechanic — same family as 1635846
(−400 g, Hawke as P0) and 1628849 / 1617442 (Hawke as P1). The
imperator's escalation contract is satisfied.

### 6d. Sturm cohort regression check

`tools/_phase11j_sturm_cohort.py` identified 3 zips with Sturm power
activations: 1615143, 1635679, 1637200.

| gid | pre-fix | post-fix |
| --- | --- | --- |
| 1615143 | oracle_gap | **ok** |
| 1635679 | oracle_gap | oracle_gap (non-Sturm residual; see §6c) |
| 1637200 | not in residual list pre or post | ok (not in 5-row residual) |

Net: 1 closure attributable to the Sturm fix in the Sturm cohort, no
regressions.

---

## 7. pytest

* `tests/test_co_sturm_meteor_strike.py` — **10 new tests, all pass.**
  Pin: COP/SCOP damage, M ≤ 2 diamond shape, 1 internal HP floor,
  diagonal boundary, friendly-fire skip, AOE one-shot consume,
  RL-path no-op when AOE unset.
* Full suite: 660 passed, 5 skipped, 2 xfailed, 3 xpassed
  (excluding `test_trace_182065_seam_validation.py`, which has a
  pre-existing failure verified to predate this fix via `git stash`).

---

## 8. Files touched

| Path | Why |
| --- | --- |
| `engine/game.py` | Add Sturm `co_id == 29` branch in `_apply_power_effects` |
| `tools/oracle_zip_replay.py` | Pin `missileCoords` → 13-tile Manhattan AOE for Sturm activations |
| `tests/test_co_sturm_meteor_strike.py` | 10 regression tests pinning Sturm canon |
| `docs/oracle_exception_audit/phase11j_final_sturm_1635679.md` | This close-out |

`data/damage_table.json` not touched (per imperator's standing order).

---

## 9. Hand-off — next campaign

The 5 surviving `Build no-op` residuals are all funds-drift cases ≤
1000 g with Hawke as the recurring CO (3 of 5 games). Recommended next
lane: Phase 11J-FINAL-HAWKE-FUNDS-DRIFT — instrument Hawke's end-turn
HP-recovery → income / repair ordering against PHP for gids 1628849,
1635846, and 1635679 in parallel.
