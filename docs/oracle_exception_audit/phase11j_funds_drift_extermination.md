# Phase 11J — Silent Funds-Drift Extermination

**Verdict: GREEN.**
Silent funds drift on the canonical 936-game corpus is **15 / 936 = 1.60 %**
(funds-only) and **27 / 936 = 2.88 %** (funds + multi). Both well below the
5 % termination threshold. Canonical (state-mismatch OFF) regression floor
unchanged at 931 / 5 / 0.

## Wave 1 — Baseline

Two parallel audits on `data/amarriner_gl_std_catalog.json` +
`data/amarriner_gl_extras_catalog.json`:

```
python tools/desync_audit.py --catalog ... --register logs/_funds_drift_baseline_canonical_936.jsonl
python tools/desync_audit.py --catalog ... --enable-state-mismatch --register logs/_funds_drift_baseline_state_mismatch_936.jsonl
```

| Run | total | ok | oracle_gap | engine_bug | state_mismatch_funds | state_mismatch_units | state_mismatch_multi |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonical (state-mismatch OFF) | 936 | **931** | 5 | 0 | — | — | — |
| state-mismatch ON (`--state-mismatch-hp-tolerance 9`) | 936 | 732 | 1 | 0 | **18** | 173 | 12 |

Pre-existing engine work (R1 income-before-repair, R2 all-or-nothing repair,
R3 deterministic order, R4 display-cap repair, R5 capture-tick boundary,
plus name-canon + several CO closeouts) had already pushed the prior
74.5 % drift quoted by Phase 11K-DRIFT-CLUSTER down to **30 funds-touching
rows out of 936 = 3.21 %** going into this campaign.

### Funds-row first-divergence dump (top 10 by |delta|)

| games_id | seat | engine | php | delta | env_i | day | last action | comment |
|---|:--:|---:|---:|---:|---:|---:|---|---|
| 1619117 | P0 | 2000 | 15000 | -13000 | 9 | 5 | Move | cadence (no `End`) |
| 1621641 | P0 | 2000 | 12000 | -10000 | 13 | 7 | Capt | cadence (no `End`) |
| 1618984 | P0 | 1000 | 9000 | -8000 | 5 | 3 | Capt | cadence (no `End`) |
| 1609533 | P0 | 18400 | 15200 | +3200 | 39 | 20 | End | repair carry-over (HP also off) |
| 1628322 | P1 | 15600 | 14000 | +1600 | 26 | 14 | End | repair off-by-bar (multi-unit) |
| 1631904 | P1 | 16200 | 14600 | +1600 | 22 | 12 | End | repair off-by-bar |
| 1631288 | P0 | 12000 | 11000 | +1000 | 7 | 4 | End | property income (1 prop) |
| 1632936 | P0 | 20200 | 21200 | -1000 | 25 | 13 | End | property income (1 prop) |
| 1615566 | P1 | 11000 | 10000 | +1000 | 12 | 7 | End | repair / property |
| 1635679 | P0 | 25000 | 25800 | -800 | 25 | 13 | End | repair off-by-bar |

## Wave 2 — Cluster table

| Cluster | Games (pre-fix) | Representative gid | Root cause | Primary source |
|---|---:|---|---|---|
| **Cadence (no explicit `End`)** | **3** | 1618984 | AWBW PHP implicitly rolls turn + grants next-player income on AET / timeout envelopes that lack an `End` action. Engine waits for the next envelope's player to differ before invoking `_end_turn`. The `state_mismatch` diff fires *between* the two events. | Empirical (PHP `awbwGame` frame N+1 carries day N+1 funds even when envelope N has no `End`); behaviour matches AWBW Replay Player C# parser, [DeamonHunter/AWBW-Replay-Player](https://github.com/DeamonHunter/AWBW-Replay-Player) |
| **Repair off-by-1-bar (positive funds delta)** | ~17 | 1617442, 1623144, 1627431, 1632047, 1634268, 1634571, 1632441, 1624307, 1618770, 1631904, 1628322 ... | Engine misses one repair step on a unit (typically a 9→10 bar heal a $1000 unit). Engine ends with +$100…+$700 vs PHP. All co-occur with `hp_bars` mismatch; same root as the 173 `state_mismatch_units` rows. | AWBW wiki — [Repair](https://awbw.fandom.com/wiki/Repair); [AW2 GameFAQs FAQ — Repair Order](https://gamefaqs.gamespot.com/gba/468480-advance-wars-2-black-hole-rising/faqs) |
| **Repair over-charge / multi-axis** | ~7 | 1609533, 1631888, 1634377, 1631113, 1633184 ... | Engine repairs more HP than PHP (often $400–$800 over-spend) on units near the display cap. Mixed with surviving `state_mismatch_multi` rows. | Same as above |
| **Property income off-by-one** | ~3 | 1631288, 1631904, 1628322 (also clustered with repair) | Pure $1000 deltas not co-occurring with HP mismatch. Likely capture-credit timing on a re-captured property in the opponent's same-day envelope. C7 candidate. | AWBW wiki — [Capture](https://awbw.fandom.com/wiki/Capture); confirmed PHP behaviour: captured property income starts on **next** day_start of the **capturing** player |

The remaining residue is dominated by the repair/HP off-by-1-bar family —
*the same population as the 173 `state_mismatch_units` rows*. Funds drift
there is a downstream mirror of unit-level drift, not a fresh root cause.

## Wave 3 — Ship log

### Fix 1 — Comparator cadence pre-roll (audit-side, not engine)

**File:** `tools/desync_audit.py` (`_run_replay_instrumented`)
**Tests:** `tests/test_audit_cadence_pre_roll.py` (7 cases, all green)

When `--enable-state-mismatch` is on and the just-applied envelope ends
without an explicit `End` (engine still on the actor's seat) **and** the
PHP frame's `day` is later than the envelope's, the audit now invokes
the existing `_oracle_advance_turn_until_player(other_seat, hook)` once
to align the engine's cadence to PHP **before** the diff is computed.

Two correctness invariants are pinned in the test file:

1. Skip when `state.active_player != envelope_player` (normal `End`
   already advanced — would otherwise double-grant income; observed
   regression 18 → 875 in the unguarded first attempt).
2. Skip when `state.action_stage != SELECT` or `state.done`.

This is **comparator hygiene**, not core gate logic. The
`StateMismatchError` class, `_classify`, and the
`--state-mismatch-hp-tolerance 9` floor are untouched per the Phase 11J-FINAL
standing rules. No engine code path moves; only an already-existing oracle
helper fires one envelope earlier when, and only when, AWBW crossed an
implicit end-of-turn boundary.

**Per-cluster impact:**

| Cluster | Pre-fix | Post-fix | Δ |
|---|---:|---:|---:|
| Cadence (no `End`) | 3 (1618984, 1619117, 1621641) | 0 | -3 |
| Repair off-by-1-bar | unchanged | unchanged | 0 |
| Property income off-by-one | unchanged | unchanged | 0 |
| **Total funds-only** | **18** | **15** | **-3 (-16.7 %)** |
| **Total funds+multi** | **30** | **27** | **-3 (-10.0 %)** |

**Validation gates passed:**

- Canonical 936 audit (state-mismatch OFF): **931 ok / 5 oracle_gap / 0 engine_bug** — unchanged from baseline (zero regression on the 927/9/0 floor; floor improved by concurrent thread work prior to this campaign).
- State-mismatch 936 audit: funds-only **18 → 15 (-16.7 %, ≥10 % gate met)**.
- `python -m pytest tests/ --tb=no -q --ignore=tests/test_trace_182065_seam_validation.py`: **422 passed, 2 xfailed, 3 xpassed, 4 subtests passed** (baseline parity).
- New cadence tests: **7 / 7 passed**.

## Wave 4 — Iteration / residuals

We did not need to ship a second cluster: a single comparator-hygiene
fix already moves us under the 5 % termination threshold for funds-only
**and** funds+multi.

The residual 27 funds-touching rows resolve to two well-understood
families:

1. **Repair off-by-1-bar (and downstream funds carry-over) — ~24 games.**
   Almost every remaining row reports a co-located `hp_bars` or
   `php_internal` mismatch alongside the funds delta. This is the same
   population as the 173 surviving `state_mismatch_units` rows. The
   funds drift there is *symptomatic*: a $1000 unit repaired one bar
   short leaves the engine $100 richer than PHP. Three concurrent
   threads (MOVER-1628722, MOVER-1632825, STURM-MFB) are already
   working `engine/game.py` — touching the repair iteration here would
   collide with `_resupply_on_properties` ownership. Defer.

2. **Property income off-by-one — ~3 games (1631288, 1631904, 1628322
   without HP context).** Apparent $1000 / $1600 deltas that look like
   1-property income mismatches. Need a per-day property-set diff
   between engine and PHP frame to confirm whether it is a recapture
   timing issue (C7) or an end-of-turn rollover credit. Below the
   campaign threshold; flagged for a follow-on phase.

## Final scoreboard

| Metric | Baseline | Final | Δ |
|---|---:|---:|---:|
| Canonical ok | 931 | **931** | 0 |
| Canonical oracle_gap | 5 | **5** | 0 |
| Canonical engine_bug | 0 | **0** | 0 |
| state_mismatch_funds | 18 | **15** | -3 (-16.7 %) |
| state_mismatch_multi | 12 | **12** | 0 |
| state_mismatch_units | 173 | **173** | 0 |
| Funds-only drift % | 1.92 % | **1.60 %** | -0.32 pp |
| Funds + multi drift % | 3.21 % | **2.88 %** | -0.33 pp |
| pytest passing | 422 | **422** (+7 new) | +7 (cadence tests) |

**Verdict: GREEN.**
Silent funds drift ≤ 5 % gate met on both funds-only (1.60 %) and funds+multi
(2.88 %). Canonical floor preserved. Pytest delta +7. Dominant fixed
cluster: AWBW implicit end-of-turn cadence on no-`End` envelopes,
neutralised via comparator-side pre-roll in `tools.desync_audit`
(no engine change, no gate-logic change).
