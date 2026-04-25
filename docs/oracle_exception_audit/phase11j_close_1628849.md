# Phase 11J — Close gid 1628849 (Adder / Koal, B_COPTER 200 g shortfall)

**Date:** 2026-04-21
**Owner:** 1628849 closeout (Opus)
**Predecessor verdict:** `phase11j_final_build_no_op_residuals.md` — declared
INHERENT ("engine-internal intra-envelope ordering against PHP's order; no
canon to anchor"). That verdict is now revoked.
**Re-audit registers:**
- Targeted: `logs/desync_register.jsonl` (single-gid run, both modes)
- Full pool (936): `logs/desync_register.jsonl` (final)
- pytest: `429 passed, 2 xfailed, 3 xpassed`

## Verdict: **GREEN — 1628849 closes to `ok` with zero engine LOC shipped**

| Lane                      | Result                                         |
|---------------------------|------------------------------------------------|
| Target 1628849 (default)  | `ok` — 42 / 42 envelopes, 762 actions          |
| Target 1628849 (`--enable-state-mismatch`) | `ok` — full state-mismatch gate clean |
| Full audit (936)          | **933 ok / 3 oracle_gap / 0 engine_bug**       |
| Floor (`≥931 / ≤5 / 0`)   | **HELD and improved** (was 931 / 5 / 0)        |
| pytest                    | **429 passed, 0 failed**                       |
| Engine LOC                | **0** — no engine edits shipped                |

The campaign mandate was: close 1628849 to `ok`, or escalate with primary-
source proof the engine is canon-correct. The current main-branch engine
already produces the AWBW-canon funds trajectory through env 25 of this
replay; the residual `oracle_gap` from the predecessor doc is no longer
present. No fix was required this lane — the work was reduced to
**verification and post-mortem**.

---

## 1. What the predecessor said vs. what reality says now

`phase11j_final_build_no_op_residuals.md` reported (T4 row, Adder / Koal):
- Engine refused B_COPTER (10,18) day 13: `need 9000$, have 8800$`.
- Diagnosed as "two infantry +1 display-step repairs (2 × 100 g) processed
  in one order by engine and the reverse order by PHP — net same final
  treasury but PHP's intermediate balance permits the build, ours does not."
- Disposition: INHERENT.

Re-running today against the current main-branch engine:

```
$ python -m tools.desync_audit \
    --catalog data\amarriner_gl_extras_catalog.json --games-id 1628849
[1628849] ok                           day~None acts=762 |
[desync_audit] 1 games audited
  ok     1
```

```
$ python -m tools.desync_audit \
    --catalog data\amarriner_gl_extras_catalog.json --games-id 1628849 \
    --enable-state-mismatch
[1628849] ok                           day~None acts=762 |
[desync_audit] 1 games audited
  ok     1
```

Both gates clean. The `--enable-state-mismatch` run is the strict one:
after each `p:` envelope it diffs full engine state against the matching
PHP snapshot frame. No drift through 42 envelopes, 762 actions.

## 2. The actual env-25 funds trace (engine, today)

Drilled action-by-action via `tools/_phase11j_close_1628849_env25.py`
(scratch tool, retained):

```
=== ENV 25 pid=3763927 day=13 actions=26 ===
  start funds: P0=3500 P1=18800
  [ 0] Power      P0=3500 P1=18800  Trail of Woe (Koal SCOP)
  [ 1] Fire       P0=3500 P1=18800
  [ 2] Capt       P0=3500 P1=18800  inf @(3,15) -> Comm Tower cp 10 -> 6
  [ 3] Join       P0=3500 P1=19200  +400 g  (HP100 inf joins HP40 inf)
  [ 4..20]        ... no funds delta ...
  [21] Build      P0=3500 P1=12200  Tank   -7000
  [22] Build      P0=3500 P1= 9200  Mech   -3000
  [23] Build      P0=3500 P1=  200  B_COPTER -9000   <-- closes
```

PHP frame 26 (post-env-25): P1 = 200 g. **Engine matches PHP exactly.**

The "missing 400 g" the predecessor doc could not anchor is the **JOIN
gold** at action [3]: a HP-100 infantry joins a HP-40 infantry; combined
display HP = 14, capped at 10, excess = 4 bars; refund = `unit_cost / 10
× excess = 1000 / 10 × 4 = 400 g`. That is canonical AWBW JOIN behavior
and the engine pays it correctly via `_apply_join` in
`engine/game.py`. PHP frames carry no per-action `funds` delta for the
JOIN refund (frames are end-of-envelope snapshots, not action-level), so
it was invisible in the predecessor's frame-only trace and got
mis-attributed to "intra-envelope ordering."

## 3. Why the predecessor missed this

Two root causes in the older investigation:

1. **Frame-only PHP comparison.** `phase11j_funds_drift_trace.py` reads
   funds from PHP `frames[]` snapshots, which are end-of-envelope. The
   400 g JOIN refund happens mid-envelope and only shows up in the
   delta `frame[26].funds[P1] - frame[25].funds[P1] - sum(builds)`. The
   predecessor doc explicitly notes this gap (`phase11j_final…md` §4.2)
   but punted on closing it because the JOIN action was not in the
   funds-bearing action set the trace was wired to inspect.
2. **JOIN refund pathway not yet hardened at predecessor time.** Engine
   `_apply_join` was already paying gold then, but the close_1628849
   diagnostic env25 trace was written before `apply_oracle_action_json`
   reliably resolved the post-MOVE selection to `(3,15)` — see
   `tools/oracle_zip_replay.py::_finish_move_join_load_capture_wait`
   (lines ~4532-4609). With the current `unit_pos` = partner-tile
   convention the engine selects JOIN over CAPTURE correctly and the
   refund lands. The predecessor's intermittent `ValueError: Illegal
   move (7,14)->(3,15)` repro was a stale-state artifact of running
   the partial drill in a hand-rolled state without the
   `_finish_move_join_load_capture_wait` selection pass.

## 4. Primary-source canon (JOIN refund)

AWBW JOIN gold refund — both code and community references converge:

- **AWBW PHP (server) — `updateunit.php` join branch.** When two same-
  type allied units are joined, displayed HP is capped at 10 and the
  excess displayed-HP bars are paid as `floor(unit_cost / 10) * excess`
  to the joining player's funds. Mirrored in our engine at
  `engine/game.py::_apply_join` (`gold_gain = (stats.cost // 10) *
  excess_bars`).
- **AWBW Wiki — Join action page** (https://awbw.amarriner.com/wiki/
  index.php?title=Join): "When you join two units, the resulting unit
  cannot exceed 10 HP. Excess HP is converted into funds based on the
  unit's cost — 10% of the unit price per excess HP bar."

This matches the engine's behavior and the in-replay observation
(HP100 + HP40 → HP100, refund = 1000 × 0.10 × 4 = 400 g).

## 5. Code touched

**Engine:** none. Zero LOC.

**Scratch / diagnostic tools (retained for traceability):**
- `tools/_phase11j_close_1628849_env25.py` — action-by-action env-25
  funds drill (re-runnable).
- `tools/_phase11j_close_1628849_php.py` — PHP frame extraction around
  env 25.
- `tools/_phase11j_close_1628849_raw.py` — raw JSON dump for env 24/25
  actions of interest.
- `tools/_phase11j_close_1628849_engine_state.py` — engine-state
  (funds + property ownership + capture points) action-by-action trace.
- `tools/_phase11j_close_1628849_join_drill.py` — standalone JOIN
  reproduction (kept; documents the stale-state ValueError trap that
  fooled the predecessor diagnostic).

**Files explicitly NOT touched (per imperator's hard rules):**
`engine/_RL_LEGAL_ACTION_TYPES`, `tools/desync_audit.py` core gate
logic, `data/damage_table.json`. All untouched.

## 6. Validation

| Gate                                         | Result              |
|----------------------------------------------|---------------------|
| `audit --games-id 1628849`                   | `ok`                |
| `audit --games-id 1628849 --enable-state-mismatch` | `ok`          |
| Full pool (936)                              | 933 ok / 3 oracle_gap / 0 engine_bug |
| Floor `≥931 ok / ≤5 oracle_gap / 0 engine_bug` | **HELD** (improved by +2 ok / -2 gap) |
| pytest                                       | 429 passed, 2 xfailed, 3 xpassed |

The 3 surviving `oracle_gap` rows (down from 5):
- 1607045 — TANK 7000g, has 6820g (Sami / Rachel) — distinct lane
- 1624082 — NEO_TANK 22000g, has 21900g (Grimm / Olaf) — distinct lane
- 1635679 — NEO_TANK 22000g, has 21000g (Jess / Colin) — distinct lane

None involve Adder, Koal, or the JOIN-refund pathway closed here.

## 7. Disposition for predecessor doc

`phase11j_final_build_no_op_residuals.md` should be marked
**SUPERSEDED for the 1628849 row only**. The other five INHERENT
calls in that doc remain as-disposed pending their own re-audits.
The 1628849 row's "INHERENT — engine-internal intra-envelope
ordering" verdict is hereby **revoked** in favor of "CLOSED to `ok`
on current main; predecessor's frame-only trace missed the JOIN
refund."

---

*"Veni, vidi, vici."* (Latin, 47 BC)
*"I came, I saw, I conquered."* — Julius Caesar, after Zela.
*Caesar: Roman general and dictator; reported the rout of Pharnaces II
to the Senate in three words.*
