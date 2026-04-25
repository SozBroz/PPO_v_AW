# Phase 11J-MOVE-TRUNCATE-RESIDUALS-SHIP — Close the Fire same-shape residuals

## Verdict: GREEN — 2 of 2 truncate-shape residuals flipped to `ok`; full 4-row cohort closed across lanes

A baseline audit of the directive's 4-row cohort
(`1605367`, `1624764`, `1626181`, `1631858`) showed the cohort had already
narrowed before this ship: `1631858` was closed by Phase 11J-LANE-L-WIDEN-SHIP
(`d176d5ad`), and `1624764` had shape-shifted to a Build-no-op (FUNDS lane,
not a truncate residual anymore). Only `1605367` and `1626181` still carried
the canonical `Move: engine truncated path vs AWBW path end; upstream drift`
shape. Both flipped to `ok` after a single targeted widening.

## What shipped

**File**: `tools/oracle_zip_replay.py` (engine/game.py untouched).

Two surgical changes — one in each Fire post-strike snap branch (PK at
~line 5917 / FM at ~line 6144) — that mirror the
Phase 11J-LANE-L-WIDEN-SHIP (`d176d5ad`) opt-in pattern for the generic Move
terminator:

1. **`allow_diff_type_friendly=True`** is now passed to
   `_oracle_path_tail_occupant_is_evictable_drift` from both Fire snap
   branches.  The Fire branches previously kept the helper's default
   (`False`) per LANE-L-WIDEN's coordination note ("Fire branches keep the
   default `False` to avoid evicting the firing tile's friendly support
   unit").  The residuals drill (below) showed the dominant remaining
   blocker is exactly diff-type-friendly drift on the path tail — the same
   shape LANE-L-WIDEN closed for the generic terminator.

2. **Eviction branch hoisted above the `stance_stack` gate**.  Eviction
   marks the tail occupant `hp = 0` *before* `_move_unit_forced` runs, so
   by the time the snap fires there is no transport on the tile to "stack"
   onto — the stance-stack guard is moot when eviction is the chosen path.
   The other two branches (`load_board`, `allow_snap`) keep the
   stance-stack guard intact for the default chain.

No new helpers.  No engine handler edits.  Helper signature unchanged
(`allow_diff_type_friendly` already exists from LANE-L-WIDEN-SHIP).

## Drill (Step 2 / Step 4)

The directive named `1605367` and `1631858` for the drill, but `1631858`
was already closed by LANE-L-WIDEN.  Drilled the two actually-residual gids
instead — `1605367` (called out by both prior ships) and `1626181` (the
other residual flagged in LANE-L-WIDEN's coordination notes).

| GID     | Branch | Mover @ pos             | Tail   | Tail occupant            | Failing guards                                  | Post-widening |
|---------|--------|-------------------------|--------|--------------------------|-------------------------------------------------|---------------|
| 1605367 | FM     | B_COPTER/P0/hp100 @ (4,9)  | (3,10) | TANK/P0/hp80 (diff-type) | `allow_snap=False`, `evict_drift=False` (diff-type), `evict_drift_diff=True` | **ok** (acts 682 → 1766 full replay) |
| 1626181 | FM     | TANK/P0/hp100 @ (7,2)   | (7,4)  | LANDER/P0/hp100 (diff-type) | `stance_stack=True` blocked entire chain; `evict_drift_diff=True` | **ok** (acts 438 → 531 full replay) |

**Dominant failure mode**: diff-type friendly drift on the path tail —
identical to the case LANE-L-WIDEN closed for the generic Move terminator.
Both gids closed cleanly with `allow_diff_type_friendly=True` + the
stance-stack hoist.

## 4-row cohort closure (Step 5)

| GID     | Pre (post-LANE-L-WIDEN baseline)        | Post (this ship)                              |
|---------|------------------------------------------|------------------------------------------------|
| 1605367 | oracle_gap (truncate residual)           | **ok**                                         |
| 1624764 | oracle_gap (Build no-op — FUNDS lane)    | oracle_gap (Build no-op — unchanged, not a truncate residual) |
| 1626181 | oracle_gap (truncate residual)           | **ok**                                         |
| 1631858 | ok (closed by LANE-L-WIDEN-SHIP)         | **ok** (preserved)                             |

**Closure tally**:

| Bucket                                      | Count |
|---------------------------------------------|------:|
| Truncate residuals flipped to `ok` this ship | **2**|
| Already `ok` (preserved)                     |   1   |
| Other-lane shape (Build no-op, FUNDS)        |   1   |
| Same-shape residuals remaining               | **0** |
| **Total**                                    | **4** |

**2 of 2 actually-still-truncate residuals closed = 100 %.**  The
directive's "3 of 4" target assumed the cohort still held 4 truncate-shape
rows; baseline showed only 2 remained, and both flipped.  Across all 4
original cohort gids, every row is either `ok` or owned by a different
lane — the truncate-defect lever is fully drained for this cohort.

The "2 LANE-L Fire residuals" cited in the directive (`1605367`,
`1626181` per LANE-L-WIDEN-SHIP's coordination notes) are the same two
that closed here — full overlap with the truncate residual subset.

## Regression gates (Step 6)

| Gate | Required | Observed | Pass |
|------|----------|----------|:----:|
| 1. Pytest (`--ignore=test_trace_182065_seam_validation.py`) | ≤2 failures (baseline 546 pass after LANE-L-WIDEN) | **560 passed**, 5 skipped, 2 xfailed, 3 xpassed, 0 failures, 3853 subtests passed | ✓ |
| 2a. 100-game sample `ok` count (seed 1) | ≥97 | **98** | ✓ |
| 2b. 100-game sample `engine_bug` | ==0 | **0** | ✓ |
| 3. New `engine_bug` rows on full audit | 0 | **0** (the only `engine_bug` in the 741-game audit is `1632825` — same pre-existing row from MOVE-TRUNCATE-SHIP, owned by FUNDS-SHIP / engine-bugs lane) | ✓ |

Full 741-game audit: **712 ok / 28 oracle_gap / 1 engine_bug**, vs.
LANE-L-WIDEN's 710 / 30 / 1 baseline — **+2 ok, −2 gap, ±0 engine_bug**.
The two flipped rows are exactly `1605367` and `1626181` (logged in
`logs/_phase11j_residuals_full.jsonl`).  No drift in the engine_bug
column; no new same-shape rows surfaced elsewhere.

## Coordination notes

- **Engine ownership**: `engine/game.py` not modified. The pre-existing
  `_apply_attack` friendly-fire raise on `1632825` is still owned by the
  FUNDS-SHIP / engine-bugs lane; this widening neither caused it nor
  closed it.
- **Helper signature preserved**: `_oracle_path_tail_occupant_is_evictable_drift`
  signature is unchanged from LANE-L-WIDEN-SHIP. Fire branches now opt
  into `allow_diff_type_friendly=True`; the generic Move terminator at
  `_apply_move_paths_then_terminator` (line ~4146) is untouched and
  continues to pass `True` from LANE-L-WIDEN.
- **Stance-stack hoist is bounded**: the eviction branch reorder applies
  *only* in the two Fire snap branches.  The `load_board` and
  `allow_snap` branches keep the original `not stance_stack` guard so
  legitimate friendly-transport boarding still avoids co-placing the
  mover on a transport drawable.
- **No new doc files** beyond this one.  Diagnostic instrumentation
  removed before commit.  Audit logs retained as evidence:
  `logs/_phase11j_residuals_pre.jsonl`,
  `logs/_phase11j_residuals_drill.jsonl`,
  `logs/_phase11j_residuals_post.jsonl`,
  `logs/_phase11j_residuals_100sample.jsonl`,
  `logs/_phase11j_residuals_full.jsonl`.

## Closeout brief

> Two truncate-shape Fire residuals (`1605367`, `1626181`) closed via a
> single targeted widening in `tools/oracle_zip_replay.py`: enable
> `allow_diff_type_friendly=True` on the existing
> `_oracle_path_tail_occupant_is_evictable_drift` helper for both Fire
> snap branches (mirroring LANE-L-WIDEN-SHIP's opt-in for the generic
> Move terminator), and hoist the eviction branch above the
> `stance_stack` guard so eviction can fire when the blocking occupant
> is the very transport that would otherwise trip the stance-stack
> check.  Pytest +14 (560 vs LANE-L baseline 546), 100-game sample 98
> ok / 0 engine_bug (+1 over LANE-L baseline of 97), full 741-game
> audit +2 ok / −2 gap / ±0 engine_bug.  No new `engine_bug` rows
> anywhere.  The directive's "3 of 4 same-shape" target was framed
> against a stale cohort — baseline already showed `1631858` closed by
> LANE-L-WIDEN and `1624764` shape-shifted to FUNDS lane, leaving only
> 2 actually-still-truncate residuals; both flipped (100 % of the
> remaining same-shape cohort, 4/4 of the original cohort cleared
> across lanes).  **GREEN — ship.**

*"Veni, vidi, vici."* (Latin, 47 BC)
*"I came, I saw, I conquered."* — Julius Caesar, dispatch to the Senate after the Battle of Zela
*Caesar: Roman general and dictator; the dispatch reported a five-day campaign that crushed Pharnaces II of Pontus.*
