# Logging & statistics — iteration plan

This document extends the main `PLAN.md` with a concrete roadmap for **training-time game logging**, **learning diagnostics**, **CO head-to-head statistics**, and **fair CO coverage**.

---

## Quick answers (current code)

### Are CO matchups randomized? Mirror or not?

In `rl/env.py`, `_sample_config()` picks **two independent** draws from the tier’s `co_ids`:

```python
random.choice(co_ids),  # P0
random.choice(co_ids),  # P1
```

So:

- **Yes — randomized** (uniform over that tier’s list, independently for each side).
- **Not forced mirror.** A **mirror** (same CO vs same CO) **can** happen whenever both draws land on the same ID — probability \(1/n\) if there are \(n\) COs in the tier and draws are uniform i.i.d., higher if the list repeats IDs.
- **Not forced non-mirror** either. If you want **no mirrors**, or **only mirrors**, that requires an explicit rule (see below).

### Is every CO explored equally?

**No guarantee today.** Independent uniform draws converge to **equal long-run frequency** only asymptotically; over any finite training window some COs will appear more often by chance. Tiers with **different** `co_ids` list lengths also change how often each global CO shows up unless you **stratify** sampling.

---

## Goals

1. **Finished-game stats** visible while training (or offline from the same log) so you can tell whether the policy is improving vs baselines and over time.
2. **Per–CO win rate**, broken down by **map**, **tier**, and **opponent CO** (full **matchup matrix** / head-to-head), not only marginal “CO wins on this map.”
3. **Controlled CO coverage**: move from “i.i.d. uniform” toward **measurable balance** (e.g. target equal games per CO per stratum, or explicit pair schedule).

---

## Phase A — Emit one record per finished episode (training)

**Problem today:** `data/game_log.jsonl` is only written from `watch_game()` smoke mode, not from `AWBWEnv` during PPO rollouts.

**Deliverable:**

- On **episode end** (terminal `step` or `reset` after done), append **one JSON line** per finished game. Each record must be interpretable standalone for analysis and dashboards.

### Per-game log record (required / target fields)

| Field | Description |
|--------|-------------|
| **Outcome & matchup** | **`winner`**: which player won (`0`, `1`, or draw / `-1` if your engine uses that). **`p0_co`**, **`p1_co`**: the two CO IDs — together this is the **matchup** (order matters for first-player effects; analysis can also derive unordered pair). |
| **Where it was played** | **`map_id`** (and optionally **`map_name`** if cheaply available from pool metadata). **`tier`** (tier name string). |
| **Economy** | **`funds_end`**: `[p0, p1]` ending treasury (already in smoke `log_game`). **`gold_spent_p0`**, **`gold_spent_p1`** (or `gold_spent: [x, y]`): **total funds paid** over the game for builds, repairs, etc. — *not* currently on `GameState`; **add cumulative spend counters** in the engine (increment on every deduction) or define an agreed proxy (e.g. derived from replay) before logging. |
| **Losses** | **`losses_p0`**, **`losses_p1`** (or `losses: [x, y]`): per-team **combat/economic losses** in a single agreed definition, e.g. **total HP lost on own units**, **number of units eliminated**, and/or **aggregate value** (unit cost × lost HP) — *requires* explicit counters or post-game walk of `game_log` / unit state. Pick one primary metric for dashboards and optional secondary fields. |
| **Length & scale** | **`turns`**, **`n_actions`** (or step count) as today. |
| **Training context** | **`agent_plays`**: which player index the trained policy controls (e.g. always `0`). **`opponent_type`**: `random` \| `checkpoint` \| … Optional **`training_steps`** (global env steps) at episode end. |

### Timestamps (required on every game row)

Each finished-game record should carry **when** the game ended (and optionally when it started) so you can sort, correlate with training runs, and plot learning over real time.

| Field | Description |
|--------|-------------|
| **`timestamp`** | **Wall-clock when the game finished** — keep existing behavior: Unix epoch **seconds** as a float (e.g. `time.time()`), easy to sort and diff. |
| **`timestamp_iso`** *(recommended)* | Same instant as **RFC 3339 / ISO 8601** string in **UTC** (e.g. `2026-04-16T02:15:30.123456+00:00`) for human-readable logs and spreadsheets. |
| **`episode_started_at`** *(optional)* | Epoch float or ISO string when **`reset()`** ran — lets you compute **realtime episode duration** independent of turn count. |
| **`timezone_note`** | Document in code: epoch is always **UTC** for ISO; local wall time is not required in the JSON row if ISO is UTC-only. |

**Training / console logs:** any periodic summary line (Phase D) should also include a **timestamp** (ISO or epoch) on each flush so tailing a file proves progress is live.

**Principle:** every row answers **who won**, **what the matchup was**, **which map (and tier)**, **when it finished** (and optionally started), plus **economic and loss pressure** per side when the engine can supply them. If a metric is not yet implemented, log **`null`** or omit with a **version field** (`log_schema_version`) so parsers don’t break.

- **Concurrency:** SubprocVecEnv = multiple workers writing concurrently → use **one writer process**, **per-worker files** merged periodically, or **file locks** / `logging` `QueueHandler` — pick one pattern and document it.

**Learning signal (minimal):**

- Rolling **win rate of the learning agent (P0)** vs opponent type, optionally per map/tier.
- Compare to **fixed baselines** (e.g. random-opponent win rate floor) on the same slice.

Optional: small **TensorBoard scalars** (mean episode reward, win rate last N games) so you see a curve without opening JSONL.

---

## Phase B — Analysis: marginal + head-to-head CO stats

**Inputs:** same JSONL as Phase A.

**B1 — Marginal (already close to `analysis/co_ranker.py`):**

- For each `(map_id, tier, co_id)`: wins / games **from that CO’s perspective** (both P0 and P1 contribute, as today’s ranker does).

**B2 — Head-to-head (new):**

- For each `(map_id, tier, co_a, co_b)` with `co_a < co_b` or directed `(co_a → co_b)`:
  - **Games** where that ordered pair (or unordered pair) occurred.
  - **Wins for CO_a** when CO_a is P0 vs CO_b as P1, and **swapped** assignment — or store **ordered** outcomes and aggregate both directions into an **undirected** summary plus **first-player advantage** if desired.

**Output artifacts (examples):**

- `data/co_rankings.json` — extend or add `data/co_matchups.json` / `data/co_h2h_{map}_{tier}.json` to avoid breaking existing consumers.
- Optional: CSV pivot tables for spreadsheet use.

**CLI:** extend `python train.py --rank` or add `python -m analysis.co_h2h` with filters (map, tier, min games).

---

## Phase C — Equal CO exploration (sampling policy)

Pick one or combine:

| Strategy | Idea | Pros | Cons |
|----------|------|------|------|
| **Stratified uniform** | Cycle or sample so each CO in tier gets equal target count per map/tier window | Even exposure | Slightly more bookkeeping |
| **Uniform over pairs** | Sample unordered `{co_a, co_b}` then assign P0/P1 at random (or alternate) | Even **pair** exposure | Need list of legal pairs; mirror optional |
| **Latin square / round-robin** | Deterministic schedule over COs | Maximum balance | Less randomness |
| **Importance monitoring only** | Keep i.i.d. but log **empirical counts** and alert if min/max ratio crosses threshold | Simple | Does not fix imbalance |

**Mirror behavior (explicit product decision):**

- **Allow mirrors:** keep independent draws or include self-pairs in pair-uniform sampling.
- **Forbid mirrors:** resample P1 until `p1_co != p0_co` (or sample from `co_ids \ {p0_co}`).
- **Only mirrors:** only for diagnostics — rarely useful for training diversity.

Document the chosen rule next to `_sample_config()` so analysis and training stay aligned.

---

## Phase D — Dashboard / ops (optional)

- Tail-friendly **summary every K episodes**: win rate, mean length, CO count histogram; optional means of **gold spent** and **losses** per side when those fields exist. **Prefix each summary block with a timestamp** (epoch or ISO UTC) so background training logs show *when* the block was emitted.
- If a web server exists: read-only endpoint for last-N-games summary (no need for full replay storage).

---

## Phase E — Fix “watch games” / Flask server launch (`server/app.py`)

**Observed failure** when running:

```text
python server/app.py
```

```text
ModuleNotFoundError: No module named 'server'
```

at `from server.routes.watch import ...` inside `create_app()`.

**Cause:** With `python server/app.py`, Python puts **`server/`** on `sys.path` as the script directory, not the **repo root**. The package name `server` is then the *current* folder’s parent conceptually, but imports expect **`server` to be a package under the project root** — so absolute imports `server.routes.*` fail.

**Fix (choose one and document in README / dev notes):**

1. **Preferred — run as a module from repo root** (same as `AWBW`):

   ```bash
   cd C:\Users\phili\AWBW
   python -m server.app
   ```

   This keeps `sys.path[0]` as the project root so `server` resolves as a package.

2. **Alternative:** Set **`PYTHONPATH`** to the project root before `python server/app.py` (e.g. `$env:PYTHONPATH="C:\Users\phili\AWBW"` in PowerShell).

3. **Code change (optional):** At the very top of `server/app.py`, insert the repo root into `sys.path` if missing (guards `python server/app.py`), *or* switch blueprints to relative imports — less clean than (1).

**Goal:** Anyone can start **watch / replay / analysis** UIs without import errors; document the canonical command in the main README so the “watching games” path is not broken by habit.

---

## Suggested implementation order

1. Phase A (log every episode from env/trainer, safe concurrent writes).
2. Phase B2 + minimal CLI export (head-to-head tables you asked for).
3. Phase C (sampling + counters to verify equal exploration).
4. Phase D as needed.
5. **Phase E** (document + optionally harden `app.py` launch) — quick win, unblocks browser-based watching.

---

## Open questions (resolve when implementing)

- Should head-to-head stats treat **P0 vs P1** as symmetric (combine) or report **first-player** separately?
- Minimum games per cell before showing win rate (Wilson / Beta priors already used for marginal ranks).
- Whether checkpoint opponents need their **policy id** in the log row for stratified analysis.
- Exact definition of **losses** (HP lost vs units destroyed vs lost value) and whether to log **multiple** loss metrics.
- How to validate **gold spent** (ledger in engine vs reconciliation from `game_log` replay).
- Whether to duplicate **`timestamp`** as both epoch and **`timestamp_iso`** in every row (recommended for tooling) or only one canonical form.

---

*This file is an iteration plan; implementation details live in code and tests when you build each phase.*
