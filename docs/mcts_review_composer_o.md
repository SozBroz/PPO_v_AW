# MCTS review — `rl/mcts.py` Phase 11b PUCT + design addendum

**Part A — checklist (2026-04-23):** Phase 11b audit scope; `train.py` MCTS remains `off` / `eval_only` (no in-rollout MCTS in `SelfPlayTrainer` yet; knobs are for orchestration / symmetric eval).

| Item | Verdict | Notes |
| --- | --- | --- |
| Determinism (same seed + same root state → same action) | **PASS** (with conditions) | `np.random.default_rng(config.rng_seed)` drives plan sampling, Dirichlet, and root softmax when `temperature > 0`. `temperature=0` uses argmax on visit counts; deterministic. With `test_mcts.py::test_mcts_determinism_temperature_zero`, the pure-numpy/CPU path is covered. A SB3 `make_callables` policy is still stochastic if `model.predict(..., deterministic=False)`; determinism of full MCTS+NN is a caller contract. |
| `total_value / visit_count` divide-by-zero; UCB at `visit_count=0` | **PASS** | `_child_q_for_parent` returns `0.0` when `child.visit_count <= 0` before dividing. PUCT U term uses `1.0 + ch.visit_count` in the denominator. |
| Memory: tree between calls | **PASS** | `run_mcts` allocates a new `TurnNode` tree each call; return value is `(plan, stats)` only. No static/global retention; SB3 callables do not keep tree state. |
| Dirichlet: root-only; training vs eval | **INDETERMINATE in `mcts.py`** / **call-site contract** | Root first `_expand_node` uses `apply_dirichlet=…`; deeper expansions set `apply_dirichlet=False` (**root only**: yes). `run_mcts` does not know train vs eval: callers must set `dirichlet_epsilon=0` for eval-style runs. `stats["dirichlet_applied"]` is true when `config.dirichlet_epsilon > 0` (i.e. “enabled by config,” not a separate “mixed this rollout” bit). |
| P0 / P1 value backprop sign | **PASS** | `_backup` flips the backed-up value at actor boundaries: `if node.actor != parent.actor: v = -v`. Child Q for parent inverts when actors differ. |
| Max plan actions / max depth / termination | **PASS** (bounded) | Turn rollouts cap via `max_plan_actions` on `apply_full_turn`. `run_mcts` has no global sim-depth cap, but a single call is limited by `num_sims` and tree growth from those sims. Pathological games are a domain concern. |

**Bug found (Part A)?** N — no high-confidence defect identified in the original checklist pass.

**Part A follow-ups (historical)**  
- Wire `alphazero` (or in-rollout MCTS) in `train.py` / `SelfPlayTrainer` if/when curriculum enables it; e2e smoke will then add that mode.  
- Perf: `mcts_sims=64` + SubprocVecEnv / GPU telemetry.  
- If strict eval MCTS: ensure `mcts_dirichlet_epsilon=0` from the caller.

---

## Part B — Design addendum: turn-level search, luck traces, edge stats, risk control

**Status:** specification and audit trail aligned with MASTERPLAN §4 (stochastic combat), `.cursor/plans/mcts_optimization_campaign.plan.md` (RNG / correctness before scale), and Phase 11a `GameState.apply_full_turn`. **Not fully implemented in code** as of this revision; implementers should treat this section as the contract.

### B.1 Keep turn-level MCTS

- **Tree nodes = full player turns** after `SELECT`; children are **distinct full-turn outcomes** (policy-sampled plans), not sub-step masks.  
- This stays the spine: `apply_full_turn` in `engine/game.py` + `rl/mcts.py` PUCT. Do not expand raw legal sub-step trees.

### B.2 Normal simulations sample damage RNG

- **Default:** each simulation path draws combat luck from the engine’s stochastic rules via `state.luck_rng` (see `apply_full_turn`’s `rng_seed` behavior: per-call seeding replaces `luck_rng` for that rollout).  
- **Intent:** MCTS backpropagates values that are **expectations under sampled luck**, not a single determinization, unless an explicit research mode adds averaging or chance nodes (out of scope until designed and tested; see MASTERPLAN fog / chance-node notes).  
- **Campaign alignment:** before transposition tables or aggressive caching, prove **branch-consistent RNG** per simulation (see `mcts_optimization_campaign` RNG audit todo).

### B.3 Metadata on `apply_full_turn` traces

Extend turn rollouts so each **sub-step** (each `state.step`) can emit structured metadata alongside the existing optional `on_step(state, action, reward, done)` hook. Proposed fields (names stable for JSON / logs):

| Field | Type (conceptual) | Meaning |
| --- | --- | --- |
| `attack_damage_rolls` | `list[int]` or `null` | Realized damage rolls for attack(s) in this step (empty if no attack). |
| `killed_unit` | `bool` or `int \| null` | Whether a unit was removed by this step; optional engine unit id. |
| `survived_at_hp` | `int \| null` | Defender HP after combat if applicable. |
| `capture_interrupted` | `bool` | True if a capture was in progress or intended but failed (e.g. unit died, illegal interruption). |
| `critical_threshold_event` | `bool` or small enum | **Narrow definition (implementer-chosen):** e.g. combat outcome within ε of lethal, funds/unit count crossing a scripted cliff, or “one-roll flipped outcome class.” Used to flag edges that need extra local resampling (B.4). |

**Implementation sketch:** collect metadata inside `GameState.step` (or a thin wrapper used only by `apply_full_turn`) so oracle parity with live stepping can be tested. Either extend `on_step` to pass a `trace_step: dict` / dataclass, or accumulate a `trace: list[TurnStepTrace]` on the side during `apply_full_turn` and return it as a fifth return value or out-parameter. Prefer **one** canonical struct in `engine/` or `rl/` to avoid drift.

### B.4 Critical threshold events → extra local resamples

- When building or scoring a **candidate turn plan** (a child edge), if the aggregated trace for that plan contains **`critical_threshold_event == true`** on any sub-step:  
  - Mark the edge as **luck-sensitive**.  
  - After the main MCTS budget, run **additional local rollouts** that **replay the same action list** (`list[Action]`) with **independent `luck_rng` / `rng_seed` draws** (fixed plan, resampled luck).  
- **Goal:** stabilize estimates of **kill probability** and **value tail** (B.5) for high-variance military/capture lines without exploding the global tree width.  
- **Count:** operator-tunable `mcts_luck_resamples` (e.g. 8–64) per flagged edge; cap total wall time.

### B.5 Edge statistics to store (per tree edge / child)

Today `TurnNode` stores `visit_count` and `total_value` (scalar sum). The design adds **sufficient statistics** for risk-aware selection:

| Stat | Role |
| --- | --- |
| `visits` | Same as `visit_count`. |
| `mean_value` | `total_value / visits` (aligned with current Q for the active actor). |
| `value_variance` | Online **Welford** variance over backed-up leaf values **as seen at this edge** (or over explicit resample returns in B.4). |
| `worst_p10_value` | Empirical 10th percentile of those returns (maintain a small reservoir sample or histogram per edge for bounded memory). |
| `kill_probability` | Fraction of resamples (or sub-steps) where `killed_unit` indicates the **targeted** friendly/enemy unit died per spec (define per CO/unit class). |

**Backprop contract:** either extend `_backup` to pass a list of per-sim leaf values into each node on the path, or maintain running stats only on **explicit luck resamples** for flagged edges; document which definition training vs eval uses.

### B.6 Selection: EV plus production risk control

- **Training / analysis / symmetric eval (default):** keep **mean-value (EV) + PUCT** as today: exploration uses visit counts and priors; final move can remain argmax visits or temperature-sampled per campaign.  
- **Production (live / high-stakes):** add a **risk layer** on root children only, after search completes:  
  - **Do not** replace PUCT during tree growth with a fragile heuristic; apply risk **only** when choosing among root’s concrete turn plans.  
  - Examples (pick one or combine; all require B.5 stats):  
    - **Hard constraint:** drop candidates with `kill_probability > τ` and `mean_value` not sufficiently above the next best.  
    - **Tail penalty:** score `α * mean_value + (1 - α) * worst_p10_value` (or CVaR-like blend) and **argmax** that score instead of raw mean.  
    - **Variance guard:** require `visits >= v_min` on an edge before it is eligible for production pick.  
  - **Logging:** emit chosen rule + per-root-child stats to eval JSON / TB for postmortems.

### B.7 Tests and gates (when implemented)

- Unit: fixed plan + two seeds → different `attack_damage_rolls` / `killed_unit`; trace non-empty.  
- Unit: plan without combat → `critical_threshold_event` false; no mandatory resample path.  
- Integration: small tree, flagged edge → resample count matches config; `kill_probability` in `[0,1]`.  
- Regression: turn-level invariants and oracle desync discipline unchanged.

### B.8 Related files

- `engine/game.py` — `apply_full_turn`, `step`, combat RNG.  
- `rl/mcts.py` — `TurnNode`, `_backup`, `run_mcts`, root selection.  
- `docs/mcts_review_composer_o.md` (this file).  
- `.cursor/plans/mcts_optimization_campaign.plan.md` — profiling, RNG audit, batched NN (orthogonal to B.3–B.6 but same program).

---

*Revision: Part B added 2026-04-25 — integrates stochastic trace metadata, local luck resamples, edge statistics, and production risk control without abandoning turn-level MCTS.*
