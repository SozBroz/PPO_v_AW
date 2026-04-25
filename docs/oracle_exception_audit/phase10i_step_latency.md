# Phase 10I — `GameState.step()` latency (STEP-GATE)

## Methodology

- **Harness:** `tools/_phase10i_step_benchmark.py`
- **Timer:** `time.perf_counter_ns()` (no `cProfile`)
- **Instrumentation:** Monkeypatch `engine.game.GameState.step` (total per-call time) and `engine.game.get_legal_actions` (only while `step` is running — STEP-GATE path). Mask sampling for the random policy uses **`engine.action.get_legal_actions` (unpatched)**, matching how `step()` resolves the gate: a second full enumeration when the caller already built a legal list for action selection.
- **Map:** `map_id=133665` (“Walls Are Closing In”), first entry in `data/gl_map_pool.json`; same pool / `data/maps` layout as the Phase 4 fuzzer.
- **Scenario:** `make_initial_state(..., co_id=1, co_id=1, tier_name="T2")` (Andy vs Andy); random uniform legal action; **10 000** successful `step()` calls; new games started when an episode ends until the step budget is met.
- **Machine (this run):** see `logs/phase10i_step_latency.json` for `platform` and `python` (example: Windows, Python 3.12.10, Intel — fields `platform.node` / `processor`).

## Results

**Authoritative numbers:** `logs/phase10i_step_latency.json` (regenerating the harness may shift means slightly with OS scheduling).

| Metric | Mean (µs) | Median (µs) | p95 (µs) | p99 (µs) |
|--------|-----------|-------------|----------|----------|
| `step()`, `oracle_mode=False` (STEP-GATE **on**) | 129.78 | 71.7 | 439.8 | 670.0 |
| `step()`, `oracle_mode=True` (STEP-GATE **off**) | 53.46 | 7.9 | 288.3 | 423.5 |
| `get_legal_actions()` **inside STEP-GATE only** | 67.34 | 11.9 | 347.4 | 496.2 |
| Residual: step − gate legal time (trace + dispatch + `_apply_*` + win check) | 62.44 | 11.8 | 324.6 | 478.2 |

- **Mean(legal_gate) / mean(step)** with gate: **~0.52** — on this workload, roughly half of mean `step()` time is the STEP-GATE enumeration alone.
- **STEP-GATE overhead vs oracle:**  
  \((\text{mean step gate on} - \text{mean step oracle}) / \text{mean step oracle} \times 100 \approx \mathbf{142.7\%}\) (from the JSON artifact).

## Verdict: **RED**

Thresholds from the campaign brief: GREEN &lt; 5%, YELLOW 5–15%, RED &gt; 15%. **~74%** incremental cost on `step()` (relative to `oracle_mode=True`) is far above RED.

**Interpretation:** With the current contract, every `step()` that is not in oracle mode performs **`get_legal_actions(self)` again** after the policy / env has typically **already** computed a legal mask to choose `action`. This benchmark isolates that *second* enumeration inside `step()`. It does **not** measure mask generation for the policy alone; it measures the duplicate work the gate adds on top of a sampling loop that already calls `get_legal_actions` once per step — which matches uniform-random play and many RL env wiring patterns.

## Phase 11 — cached-legal-set fast path (sketch)

If we need bot-scale throughput (MCTS / search / RL rollouts):

1. **Reuse the mask:** Allow callers that already proved `action ∈ legal` to pass a flag or precomputed digest (narrow API, high risk if misused) — usually worse than structural caching.
2. **Cache on state (preferred sketch):** After each successful `step`, store `last_legal: frozenset[Action]` (or a canonical fingerprint) on `GameState`, invalidated on any mutation path (single place: end of `step` / internal helpers). `step()` checks membership against cache first; on miss, compute and refresh.
3. **Lazy invalidation:** Track a monotonic `state_revision` bumped on mutation; cache stores `(revision, legal_set)`. Cheap equality before full recompute.

Until one of these exists, STEP-GATE remains correct but **expensive** for high-volume `step()` loops that already pay for one full enumeration per action.

## Artifacts

| File | Role |
|------|------|
| `tools/_phase10i_step_benchmark.py` | Harness |
| `logs/phase10i_step_latency.json` | Raw stats + metadata |
