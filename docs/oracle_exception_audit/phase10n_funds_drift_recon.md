# Phase 10N — Funds drift root-cause recon

## Purpose

Phase 10F showed that many `class: ok` oracle replays still **silently drift** from gzipped PHP `awbwGame` snapshots (mostly **funds**, often with **HP** on the same step). This lane isolates **why** treasuries diverge so Phase 11 can patch **surgically** (no engine edits here).

## Methodology

1. **Seed table:** `docs/oracle_exception_audit/phase10f_silent_drift_recon.md` + `logs/phase10f_silent_drift.jsonl`.
2. **Representative cases** (confirmed against the table):
   - **1628546** — smallest first-mismatch envelope index (**11**), **P0-only** funds + HP; `co_p0_id=23` (**Kindle**), `co_p1_id=7` (Max).
   - **1620188** — **P0-only** funds at envelope **13**; `co_p0_id=16` (**Lash**), `co_p1_id=1` (Andy). Tight drill: no CO income D2D on either side.
   - **1628609** — **large bilateral** funds at envelope **13**; `co_p0_id=1` (Andy), `co_p1_id=16` (Lash); same map **180298** family as a clean 10F control (1629276).
3. **Harness:** `tools/_phase10n_drilldown.py` — `random.seed(_seed_for_game(CANONICAL_SEED, games_id))` (matches `tools/_phase10f_recon.py` / `desync_audit`), replays oracle envelopes, **monkey-patches** `GameState._grant_income` to log grants, and steps the same snapshot pairing as `tools/replay_state_diff.py`.
4. **Artifact:** structured dump `logs/phase10n_funds_drilldown.json`.

## How `replay_state_diff` compares funds

After each `p:` envelope, the engine is compared to **gzip line** `frame[step_i + 1]` when it exists (`tools/replay_snapshot_compare.compare_funds`): for each PHP player row, `funds` is compared to `state.funds[engine_seat]` via `awbw_players_id → {0,1}` from `frame[0]`.

## Per-case findings

### 1628546 (Kindle vs Max) — first mismatch step_i **11**

| step_i | envelope day (AWBW field) | engine turn | Notes |
|--------|----------------------------|------------|--------|
| 10 | 6 | 6 | Funds: **match** (P0 0, P1 10 000). |
| **11** | 6 | 7 | **First drift:** P0 engine **9 000** vs PHP **8 800** (Δ **+200** engine); same step lists **hp_bars** mismatch at `(0,6,5)`. |

**Income instrumentation (turn 7, P0 / Kindle):** immediately before the grant that ends P1’s half-turn and starts P0’s day, the patch saw **9 income properties**, **5 “city-only” tiles** (HQ/base/airport/port/lab/tower excluded), and a flat grant of **+9 000** (= **9 × 1 000**) — i.e. **no Kindle +50% on city tiles** in `engine/game.py` `_grant_income` (only Colin **15** and Sasha **19** have special-cases today).

**Interpretation:** The **+200** engine *advantage* is **not** explained by “missing Kindle bonus” alone (that would usually **lower** engine vs PHP on income-heavy turns). The same step’s **HP drift** points to **repair / build / combat-side** treasury effects (different HP ⇒ different heal cost or different build legality). Kindle’s missing D2D is still a **real** gap for long-run parity on city-heavy positions.

---

### 1620188 (Lash vs Andy) — first mismatch step_i **13**

| step_i | Action tail (abbrev.) | P0 funds (engine vs PHP) |
|--------|----------------------|---------------------------|
| 12 | Fire, Fire, Build, Capt, Build, End | **match** (800 / 12 000) |
| **13** | Move, Move, **Fire**, Build, Build, End | engine **12 030** vs PHP **11 400** (Δ **+630**, **P1 unchanged**) |

**Income instrumentation:** On the **turn 8** grant to P0 (Lash), **12 properties** ⇒ **+12 000**; treasury after grant **12 030** — envelope actions then **do not reduce** P0 below 12 030 in the engine trace, while PHP ends **11 400**. Net: PHP behaved as if **~630 G** more was spent or **less** was kept than in the engine after the same JSON.

**Interpretation:** With **no** income D2D on these COs, this is strong evidence for **spend-side** divergence: `_apply_build` **no-ops** when `funds < cost` (`engine/game.py`), while AWBW still recorded the build in the replay — **oracle slack** already called out in Phase 10F. **Fire** in the envelope ties the step to **combat state**, which can change **repair** eligibility/cost on the following frames.

---

### 1628609 (Andy vs Lash) — first mismatch step_i **13** (bilateral)

| step_i | P0 funds (eng / php) | P1 funds (eng / php) |
|--------|----------------------|----------------------|
| 12 | 0 / 0 | 18 000 / 18 000 |
| **13** | **37 800 / 18 800** | **23 000 / 5 000** |

**Critical trace:** The monkey-patch’s **last** P0 grant shows `funds_before_grant: [18 800, 23 000]` → after **+19 000**, **37 800 / 23 000**. PHP’s snapshot P0 **18 800** equals the engine’s **pre-grant** P0 balance **exactly** — then the engine applies **one full day’s income** (19 props × 1 000) that **PHP’s serialized line does not yet include** at the same compare index.

**Interpretation:** At least this zip exposes an **income cadence vs snapshot boundary** problem: engine and PHP agree **until** a boundary where the engine has **already** granted the next day’s income but the compared gzip frame has **not**. This can present as a **~1× daily income** bilateral jump and must be reconciled against **when** AWBW materializes “start of turn” funds in exported lines (vs our `_end_turn` → `_grant_income(opponent)` ordering). It is **not** explained by Kindle alone (neither CO is 23).

---

## Root-cause taxonomy (ranked)

1. **Incomplete CO daily income in `_grant_income`:** **Kindle (23)** — AWBW: **+50% income from owned *cities*** (see primary sources below). Engine only adds flat **n × 1 000** (+ Colin/Sasha branches). **~7 / 39** Phase‑10F drift rows include CO **23** (see `logs/phase10f_silent_drift.jsonl` grep).
2. **Spend / oracle slack:** **Build** and **repair** paths can diverge when the engine **silently no-ops** (insufficient funds, legality) but the zip still advances — funds then diverge without an oracle exception (**1620188**-shaped).
3. **Income timing vs gzip pairing:** **1628609** shows **pre‑grant** funds match PHP, then an **extra** grant vs the snapshot — investigate **half-turn boundary** alignment between `apply_oracle_action_json`/`_end_turn` and the **semantic moment** each PHP line encodes (may interact with **tight** `n_frames == n_envelopes` exports).
4. **Combat → economy coupling:** **HP** mismatches on the **same** step as funds (**1628546**) — fixing combat/repair may be **necessary** for funds parity even if `_grant_income` is perfected.

## Primary sources (AWBW canon)

- **Kindle / CO income text:** [AWBW CO Chart](https://awbw.amarriner.com/co.php) — official site list (Kindle: bonus funds from **cities**; **Sasha**: **+100 funds per income property** you own — matches the engine’s Sasha stub better than the narrative in `data/co_data.json`).
- **Weather (if extending beyond these three cases):** [Weather | AWBW Wiki](https://awbw.fandom.com/wiki/Weather) — confirm whether rain/snow alter **income** in AWBW (engine `GameState._grant_income` currently ignores `state.weather`).

## Phase 11 fix sketch (do not apply in 10N)

| Item | Target | Patch shape |
|------|--------|-------------|
| Kindle D2D | `engine/game.py` — `GameState._grant_income` | For `co_id == 23`, add **+50%** on income from **city** tiles only (properties that are income tiles but **not** HQ/base/airport/port/lab/tower — mirror `PropertyState` flags / terrain “City” rows in `engine/terrain.py`). |
| Build / repair slack | `engine/game.py` — `_apply_build`, `_apply_repair` **and/or** `tools/oracle_zip_replay.py` | When oracle replay applies a **Build** that the engine would no-op, **raise** or **hard-fail** in diff harnesses (stricter than RL `step`); or reconcile AWBW “insufficient funds” semantics with site logs. |
| Snapshot / income boundary | `tools/replay_state_diff.py` **or** `_end_turn` ordering | If canonical rule is “funds in line L are **before** next player’s income”, insert a **compare hook** that strips or delays the last `_grant_income` for tight pairings — **requires** one definitive AWBW export spec (Replay Player / PHP). |

## Regression test sketch

1. **Unit:** `make_initial_state` on a tiny fixture map with **only** neutral cities → assign to Kindle → `_grant_income(0)` must equal **1 500** per owned city (and **1 000** for non-city income tiles if present).
2. **Replay:** Add `replay_state_diff` on a **minimal** Kindle zip (or synthetic export) expecting **0** funds mismatch through **N** envelopes once Kindle is fixed.
3. **Anti-regression:** Re-run `tools/_phase10n_drilldown.py` on **1628546 / 1620188 / 1628609**; expect **1628609** to remain red until boundary semantics are decided (document expected outcome).

## Risk assessment

- **Kindle fix:** Low risk to non-Kindle games if gated on `co_id == 23`. **Risk:** misclassifying “city” vs neutral tile IDs on exotic maps — use the same predicate as terrain tables.
- **Stricter oracle build:** High churn on `desync_audit` if many zips relied on no-op builds — gate behind a **diff-only** flag first.
- **Income boundary:** Wrong fix could **mask** real economy bugs or break trailing/tight pairing — needs **C# Replay Player** / PHP export ground truth.

## Bottom line (one sentence)

**Treasury drift is driven mainly by incomplete CO income rules in `_grant_income` (notably Kindle’s city bonus), compounded by oracle spend slack and, in some zips, a mismatch between when `_grant_income` runs and when PHP snapshots encode post-income funds.**

---

*“In preparing for battle I have always found that plans are useless, but planning is indispensable.”* — Dwight D. Eisenhower, speech to the National Defense Executive Reserve Conference, 1957  
*Eisenhower: Supreme Allied Commander Europe in World War II and 34th U.S. President.*
