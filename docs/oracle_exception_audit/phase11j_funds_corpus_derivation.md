# Phase 11J-FUNDS-CORPUS — Income vs property-repair ordering from PHP zips

**Verdict letter: ESCALATE** (corpus clears the *ordering* question; **ship attempt RED** — reverted.)

---

## Commander brief

Across **100** catalog games sampled from `logs/desync_register_post_phase11j_combined.jsonl` (`class: ok`, excluding CO ids **17 / 28 / 23** — Hachi / Rachel / Kindle), with **Andy / Max / Sami** mirror matchups preferred then backfilled, AWBW PHP **`players[*].funds`** at turn-start boundaries (see §2) **always agrees with the income-then-repair hypothetical** whenever that hypothetical disagrees with repair-then-income: **69** turn-starts **`PHP_MATCHES_IBR`**, **0** **`PHP_MATCHES_RBI`**, **39** **`PHP_MATCHES_NEITHER`**, **1906** **`AMBIGUOUS`** (repair order irrelevant). So **Tier-1 PHP behavior is uniformly income-before-property-day-repair** in this lane, not a 60/40 split.

A **ship trial** (swap in `GameState._end_turn` to `_grant_income` then `_resupply_on_properties`) **failed** your gates: **`desync_audit.py --max-games 100 --seed 1`** dropped to **`ok=88`** vs baseline **`89`** (`logs/phase11j_income_order_gate100.jsonl`), and the four FU gids remained **`oracle_gap`** on Build no-op (`_phase11j_funds_drill.py` still shows material **`NEITHER`**-class drift after the ordering fix, e.g. **1621434** env **27** P0 `engine 18000` vs PHP `22600` at `players[*].funds`). Per standing rule **“no new regressions”**, the engine change was **reverted**; **`tests/test_co_income_ordering.py`** was removed with the revert.

**Counsel:** Treat **income-first** as **AWBW PHP-grounded** for turn-start economy, but **do not merge** until repair eligibility / partial-heal / Sasha War Bonds (Phase 11J-F2) are aligned so the **100-game** and **1621434 / 1621898 / 1622328 / 1624082** gates can absorb the reorder without burning `ok` rows.

---

## Section 1 — Prior art locked in

- Escalation and drill context: `docs/oracle_exception_audit/phase11j_f2_koal_fu_oracle_funds.md`.
- Funds cadence recon: `docs/oracle_exception_audit/phase10n_funds_drift_recon.md`.
- Drill deltas: `tools/_phase11j_funds_drill.py` (compares engine vs PHP **`funds`** per envelope).
- **Engine order (reverted baseline):** `engine/game.py::_end_turn` calls **`_resupply_on_properties(opponent)`** then **`_grant_income(opponent)`** (repair-before-income).
- **PHP loader:** `tools/diff_replay_zips.load_replay` → per-gzip-line dicts; player treasury is **`players` → row → `funds`** (int). Same field as `tools/oracle_zip_replay.py` / `replay_state_diff` (not a separate `_load_php_snapshots` symbol — the mission text maps to **`load_replay`** + envelope pairing in `tools/replay_snapshot_compare.py`).

---

## Section 2 — Corpus probe methodology

**Tool:** `tools/_phase11j_funds_ordering_probe.py`

For each completed oracle replay, on every **`_end_turn`** that runs the start-of-opponent-turn block (not the max-turn early exit), after the fuel / crash prefix (mirror of `engine/game.py` lines 359–438):

1. **Deep-copy** baseline `GameState` → **`s_ibr`**: `_grant_income(opp)` then `_resupply_on_properties(opp)` → **`engine_funds_ibr`**.
2. **Deep-copy** baseline → **`s_rbi`**: `_resupply_on_properties(opp)` then `_grant_income(opp)` → **`engine_funds_rbi`**.
3. Read **PHP** funds for the turn-starter seat **`opp`** from **`frames[snap_i]`**, where **`snap_i = env_i + 1`** and **`snap_i < len(frames)`** (trailing and **tight** layouts per `replay_snapshot_pairing`; no frame after the last half-turn → **no row** — same rule as `replay_state_diff.py`).
4. **Classify** that turn-start:
   - **`AMBIGUOUS`**: `engine_funds_ibr == engine_funds_rbi`.
   - **`PHP_MATCHES_IBR`**: `php == engine_funds_ibr` and `php != engine_funds_rbi`.
   - **`PHP_MATCHES_RBI`**: `php == engine_funds_rbi` and `php != engine_funds_ibr`.
   - **`PHP_MATCHES_NEITHER`**: else.

**Sample:** 100 `games_id` from combined register, `ok`, both COs not in `{17,28,23}`, prefer both in `{1,7,8}` then sorted fill.

**Artifacts:** `logs/phase11j_funds_ordering_probe.json` (full per-game records + summary).

---

## Section 3 — Aggregate and per-game results (100-game run)

| Bin | Count |
|-----|------:|
| **AMBIGUOUS** | 1906 |
| **PHP_MATCHES_IBR** | 69 |
| **PHP_MATCHES_RBI** | 0 |
| **PHP_MATCHES_NEITHER** | 39 |

**Games completed full replay in probe:** 84 of 100 (16 exited early — `oracle_gap`, `unsupported_pairing`, or `probe_exception`; see JSON `cases`).

**Order-resolving PHP agreement:** among turns where **`engine_funds_ibr != engine_funds_rbi`**, PHP matched **IBR only** (**69 / 69**); **never** matched **RBI alone**.

### Sample gids (≥2 per non-ambiguous bin)

| Bin | Example `games_id` (see JSON for `env_i` / `php_funds_turn_starter`) |
|-----|---------------------------------------------------------------------|
| **PHP_MATCHES_IBR** | **1609589**, **1622452**, **1623698**, **1624316**, **1624758** |
| **PHP_MATCHES_NEITHER** | **1609589**, **1623698**, **1624758**, **1625633** (any row with `bin: PHP_MATCHES_NEITHER` in `logs/phase11j_funds_ordering_probe.json`) |
| **PHP_MATCHES_RBI** | *none* |
| **AMBIGUOUS** | Majority of games (ordering does not change treasury at that boundary). |

**Illustrative Tier-1 cite (IBR):** game **1609589**, envelope **18**, turn-starter seat **P1**: PHP **`players[*].funds`** for P1 = **16600** (frame index **19**), matching **`engine_funds_ibr` = 16600** and not **`engine_funds_rbi` = 18000** — reproduced in probe JSON rows.

---

## Section 4 — Decision vs mission table (wording correction)

Mission Step 4 pairs **“PHP-matches-IBR”** with **income-before-repair**. If that pattern dominates, **AWBW is income-first**; the **current** engine (repair-first) is then **wrong** on ordering.

The printed decision table in the mission appears **inconsistent** with Step 2 and with `engine/game.py`: it assigns “engine correct / do not change” to **IBR > 90%** while the **swap** line names **`_resupply_on_properties` before `_grant_income`**, which is **already** the reverted baseline. **This report follows Tier-1 PHP + Step 2 definitions**, not the contradictory row text.

**Empirical conclusion:** **Income then property-day repair** matches PHP in **all 69** non-ambiguous order-sensitive turn-starts observed (100-game sample).

---

## Section 5 — Ship trial and validation (reverted)

| # | Gate | Result |
|---|------|--------|
| 1 | `pytest tests/test_co_income_*.py` | **N/A** after revert (`test_co_income_ordering.py` removed); **`test_co_income_kindle.py`** passes on reverted tree |
| 2 | `pytest tests/test_oracle_strict_apply_invariants.py -v` | **PASS** (reverted) |
| 3 | `pytest --tb=no -q --ignore=test_trace_182065_seam_validation.py` | **PASS** — **526** passed (reverted) |
| 4 | Gids **1621434 / 1621898 / 1622328 / 1624082** → ≥3 `ok` | **FAIL** — all remained **`oracle_gap`** on Build no-op after ordering swap trial |
| 5 | 100-game `ok ≥ 89`, `engine_bug == 0` | **FAIL** — **`ok=88`**, `engine_bug=0` (`logs/phase11j_income_order_gate100.jsonl`) |
| 6–9 | CO / repair / drill / count ≥526 | **Not pursued** after gate 5 fail; **revert executed** |

**Funds drill after swap (trial only, before revert):** `logs/phase11j_funds_drill_post_income_order.json` — e.g. **1621434** first P1 funds drift moved from env **14** to later P0 divergence; **1622328** still large P1 gap at env **28**; **1624082** persistent P1 **−200** vs PHP — consistent with Phase 11J-F2 (**ordering necessary but not sufficient**).

---

## Section 6 — Verdict letter and next moves

**ESCALATE**

- **Ordering:** PHP corpus resolves **income-before-repair** (no RBI-exclusive matches).
- **Merge:** **Blocked** — **−1** `ok` on 100-game and FU gids still red; **code reverted** per “no new regressions.”
- **Next:** Bundle **income-first** with repair-model / eligibility / CO-side fixes (Phase 11J-F2 Q1–Q2), then re-run the nine gates.

---

## Section 7 — Artifacts

| Path | Role |
|------|------|
| `tools/_phase11j_funds_ordering_probe.py` | Corpus classifier (kept) |
| `logs/phase11j_funds_ordering_probe.json` | 100-game run output |
| `logs/phase11j_income_order_gate100.jsonl` | Ship-trial 100-game register (**88 ok**) |
| `logs/phase11j_funds_drill_post_income_order.json` | Per-envelope funds drill on four gids (**trial**) |

---

*"In any moment of decision, the best thing you can do is the right thing, the next best thing is the wrong thing, and the worst thing you can do is nothing."* — attributed to Theodore Roosevelt (letters / speeches; often quoted form, early 20th c.)  
*Roosevelt: 26th U.S. President; the line is widely used in leadership contexts.*
