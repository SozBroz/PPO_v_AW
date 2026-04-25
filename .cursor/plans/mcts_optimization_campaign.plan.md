---
name: MCTS optimization campaign
overview: >-
  When Phase 1 value quality and throughput allow, scale turn-level PUCT from
  prototype (tens–low hundreds of sims) toward 800–1600+ sims per P0 decision
  (training/eval) and toward time-budgeted anytime search for production AWBW
  play. Prioritize correctness (combat RNG), compute (batched NN + rollout cost),
  and search geometry (adaptive branching, exploration tuning) before raw sim
  inflation. AWBW is turn-based: production is gated by clock time more than a
  fixed iteration cap, but combinatorial breadth still demands policy-guided
  narrowing (turn-level C-MCTS-style aggregation is already the spine).
todos:
  - id: mcts-opt-0-gates
    content: >-
      Preconditions: confirm MASTERPLAN §4 gates — explained_variance and eval
      mix match search distribution; symmetric eval harness stable; document
      baseline winrate vs pool at current sims (e.g. 16–128) in a fixed eval matrix.
    status: pending
  - id: mcts-opt-1-profile
    content: >-
      Profile one full MCTS decision: wall time split — apply_full_turn rollouts
      vs SB3 policy.predict / predict_values / get_distribution; record at
      num_sims 256/512/800/1600. Decide whether first optimization is engine,
      NN batching, or fewer expands.
    status: pending
  - id: mcts-opt-2-rng-audit
    content: >-
      Audit luck_rng / combat randomness per simulation path: each sim must use
      a reproducible branch RNG or explicit policy; document whether transposition
      caching is safe. Add tests if gaps (determinism vs seed, no cross-sim bleed).
    status: pending
  - id: mcts-opt-3-adaptive-k
    content: >-
      Implement adaptive root_plans (higher K at root, lower K deeper) and/or
      progressive widening (expand top-M priors until visit threshold). Tune
      max_plan_actions vs empirical turn length from game_log.
    status: pending
  - id: mcts-opt-4-puct-sweep
    content: >-
      Offline grid on symmetric_checkpoint_eval: c_puct, dirichlet (eps, alpha),
      min_depth, temperature; fixed wall budget per game; metrics — winrate,
      mcts_decision_wall p50/p95, PV depth, root visit entropy.
    status: pending
  - id: mcts-opt-5-batch-nn
    content: >-
      Batch leaf/value (and optional policy) forwards — queue states per MCTS
      wave, single GPU forward; consider virtual loss for parallel expand. Target
      linear scaling of sims with sub-linear NN cost.
    status: pending
  - id: mcts-opt-6-escalator-data
    content: >-
      Wire escalator / mcts_eval_summary to record marginal winrate vs sim count
      and DROP rules; only promote sim doubling when curve shows ROI (align with
      tools/mcts_escalator.py and fleet audit).
    status: pending
  - id: mcts-opt-7-doc-masterplan
    content: >-
      MASTERPLAN §4 now includes Production MCTS (clock budget, breadth vs depth,
      training vs production table). Keep in sync when time-based MCTS lands in code.
    status: completed
  - id: mcts-opt-8-time-budget
    content: >-
      Production path: add optional wall-clock budget per P0 root (anytime loop
      until deadline) alongside num_sims cap; log sims_used + wall_s in telemetry;
      wire symmetric_checkpoint_eval / live play entrypoint when defined.
    status: pending
  - id: mcts-opt-9-search-control
    content: >-
      Heuristic time multiplier from cheap complexity signals (e.g. unit count,
      frontline proxy); cap multipliers; log factor per turn for analysis.
    status: pending
---

# MCTS optimization campaign

## Goal

Make **turn-level PUCT** in this repo ([`rl/mcts.py`](c:\Users\phili\AWBW\rl\mcts.py)) as **useful per wall-clock** as possible when you move from “prototype eval” to **800–1600+ simulations per P0 root decision** for **reproducible training/eval**, and toward **time-budgeted anytime search** for **production AWBW** (competitive / live), while keeping **promotion metrics** trustworthy.

**Non-goals for this campaign (unless explicitly reopened):**

- `train_advisor` / in-rollout MCTS inside PPO (`model.learn`) — wrong off-policy geometry; MASTERPLAN and [`train.py`](c:\Users\phili\AWBW\train.py) keep **`eval_only`** for search.
- Fog / information-set MCTS — needs belief-state search; out of scope until MASTERPLAN fog lane.

## Current architecture (truth on disk)

| Piece | Role |
|--------|------|
| [`engine/game.py`](c:\Users\phili\AWBW\engine\game.py) `apply_full_turn` | Rollout primitive: one **full player turn** from SELECT. Optional **`return_trace=True`** → 5-tuple with per-step MCTS trace; default 4-tuple unchanged. |
| [`rl/mcts.py`](c:\Users\phili\AWBW\rl\mcts.py) | PUCT at **turn nodes**; **`root_plans`** children; Dirichlet / `min_depth`; `make_callables_from_sb3_policy`. **Root risk layer:** `luck_resamples`, `risk_mode`, `EdgeStats`, optional JSONL decision log; default selection remains visit-based. |
| [`scripts/symmetric_checkpoint_eval.py`](c:\Users\phili\AWBW\scripts\symmetric_checkpoint_eval.py) | **`eval_only`** MCTS; telemetry JSON; CLI for luck/risk; **`mcts_root_entropy`**, **`mcts_chosen_risk`**. |
| [`tools/mcts_health.py`](c:\Users\phili\AWBW\tools\mcts_health.py) | Competence gate for turning MCTS on per machine. |
| [`tools/mcts_escalator.py`](c:\Users\phili\AWBW\tools\mcts_escalator.py) | Sim budget escalator; pair with eval summaries. |
| [`MASTERPLAN.md`](c:\Users\phili\AWBW\MASTERPLAN.md) §4, **§14** | Phase 2 narrative; **staged MCTS-0…4** rollout ladder + risk-layer implementation table. |

**Rollout stages** are in **MASTERPLAN §14** (MCTS-0 … MCTS-4). This campaign file is **scale-up / perf / tuning**, not a separate stage ladder.

**Horizon intuition (your “2 days = 4 turns”):** four **player-turns** from root ≈ **4 turn-nodes** along a line (P0→P1→P0→P1). Deeper “what-if” lines to ~4 **game days** ≈ **8 turn-nodes**. Search should **allocate** visits via PUCT; do not assume every sim reaches max depth — value head handles early leaves.

## Production vs training (AWBW is not RTS)

**Clock-first:** Unlike StarCraft-style RTS, AWBW does not force sub-second reactions. **Production** agents should treat MCTS as **anytime**: a **wall-time budget per P0 turn** (e.g. 30–60s, rule- and hardware-dependent) drives how many expansions you complete; **doubling search time** usually helps **if** V(s) is well-calibrated on searched states (same scaling intuition as AlphaZero-class systems, with AWBW-specific rollout cost).

**Training / fleet eval:** Keep **fixed `num_sims`** (or capped budgets) for **reproducible** sweeps, escalator curves, and A/B until time-based mode is validated.

**Iteration scale (order-of-magnitude targets, not promises):**

| Mode | Typical sim / rollouts per root (indicative) |
|------|-----------------------------------------------|
| Low-difficulty / fast eval | ~800–2,000 |
| Strong production (heavy hardware) | **10,000–100,000+** if NN + rollouts are batched and fast enough |
| Literature anchor | AlphaZero chess often cited ~**1,600 sims** in **~0.4s**; AWBW **turn-level** sims are **not** directly comparable — expect **higher** rollout counts for similar “thinking depth” if each sim runs a full turn.

**Breadth vs depth:**

- **Breadth (bottleneck):** Combinatorial **turn** structure (unit A / unit B / …). Mitigation: **policy-guided** children only; in-repo that is **K distinct full-turn plans** per node + PUCT — align with **combinatorial MCTS (C-MCTS)** ideas: a **turn** is one macro-decision, not one mask click.
- **Depth:** Useful lookahead hits **diminishing returns** after on the order of **~10–12 player-turns** in noisy positions; prefer **extra width** at tactical horizons before chasing very deep speculative lines.

**Exploration vs exploitation:**

| Phase | Exploration | Final selection |
|-------|-------------|----------------|
| Training / analysis | Dirichlet at root, temperature > 0 where useful | May be stochastic for data gen |
| Production play | Minimal noise (exploitation-forward) | **Deterministic** from visit counts (temperature 0 / argmax) |

**Policy “move pruning”:** In production, **never** enumerate all legal micro-moves. Use **policy mass** to limit branching (top-N or sampled rollouts). Here, **turn-level** search already avoids sub-step trees; extend with **adaptive K**, **progressive widening**, and (if needed) **per-stage** top-k legal filtering **without** breaking engine legality.

**Search-control heuristic:** Allocate **more wall time** on **high-density** turns (many units, contested front); **less** on simpler phases — implement as **logged multiplier** on time budget or dynamic K.

**External references (general literature / community):** AlphaZero / MuZero lines; tabletop & arXiv surveys on large branching and combinatorial MCTS; CEUR-WS and related work on policy-guided narrowing — use for intuition; **this repo’s contract** is still `apply_full_turn` + `rl/mcts.py` + tests.

## Strategy pillars

### 1. Correctness before scale

- **Combat luck:** `GameState.luck_rng` must be **branch-consistent** per simulation path (derived seed per sim + path, or documented expectimax). Otherwise backprop is wrong and 1600 sims **amplify noise**.
- **Transposition table:** only if state key encodes **everything** that affects future payoffs (including RNG state if luck not fixed per branch). If in doubt, **no TT** in v1 of scale-up.

### 2. Branching geometry (biggest quality lever per sim)

- Today each expanded node samples **`root_plans` (default 8)** distinct full-turn lines — effective branching cap.
- **Adaptive K:** raise **K at root** (e.g. 16–32) for diversity where it matters; **lower K** deeper or use **progressive widening** (expand top priors first, add siblings as visits grow).
- **`max_plan_actions`:** tune from **real turn lengths** in logs; prevents runaway micro-step rollouts.

### 3. Exploration / “DeepMind-style” stochasticism

- **PUCT:** grid-tune **`c_puct`** against fixed eval matrix and wall budget.
- **Dirichlet:** root-only; **ε > 0** for exploratory runs; **ε = 0** for strict promotion (see [`docs/mcts_review_composer_o.md`](c:\Users\phili\AWBW\docs\mcts_review_composer_o.md)).
- **Final selection temperature:** `0` = argmax visits for eval; `> 0` only if generating training targets or intentional stochastic play.
- **Plan diversity:** `_sample_plans` relies on **stochastic** policy rollouts — if diversity collapses, fix **policy entropy / rollout temperature** before raising sim count.

### 4. Compute: batched NN

- Serial `predict` / `predict_values` per leaf will **dominate** at 800–1600 sims.
- **Batch** leaf evaluations (and optionally batched policy logits for priors) per MCTS “wave”; add **virtual loss** or equivalent if parallelizing expand.

### 5. Engine throughput

- If profiling shows **`apply_full_turn`** dominates, prioritize **hot-path** work in engine / copy semantics **after** NN batching is on the table (don’t optimize rollouts in a vacuum).

### 6. Tuning loop (disciplined)

- Fixed **eval matrix**: maps, seats, seeds, checkpoint pair.
- Sweep: `num_sims` × `c_puct` × `root_plans` / adaptive-K schedule × Dirichlet × `min_depth`.
- Report: winrate vs pool, **mcts_decision_wall** p50/p95, **PV depth**, **visit entropy** at root (collapse = bad exploration).
- **Production mirror:** repeat critical sweeps under a **fixed wall-time** cap (anytime) to ensure training-knob winners transfer when iteration count is not fixed.

### 7. Escalator discipline

- Double sim budget only when **marginal winrate** (or agreed metric) justifies cost; log curves to `mcts_escalator` / eval JSON so you **see** when search stops paying.

## Risks / anti-patterns

- **High sims + weak V(s)** on the search distribution → confident wrong lines (MASTERPLAN already warns).
- **Promotion eval with Dirichlet + temperature** → non-reproducible “wins.”
- **Caching states** without RNG discipline → silent wrong MCTS.
- **train_advisor** → PPO importance ratio trap; refuse unless research track with explicit mitigation.

## Success criteria (campaign complete = “ready to lean on search”)

1. At target sim budget (e.g. 800–1600), **median** decision time acceptable for your eval cadence; p95 bounded.
2. **RNG / determinism** documented and tested for one root state + seed.
3. **Adaptive branching** landed; ablation shows winrate or PV quality gain vs fixed K=8 at same wall time.
4. **Batched NN** or proven bottleneck is rollouts (with next concrete engine task listed).
5. Escalator / logs show **ROI curve** for sim doubling.

## Beyond AlphaZero / DeepMind classics (research pointers)

If your mental model is **AlphaGo / AlphaZero / AlphaStar-style** MCTS + policy–value nets, these are the usual **next layers** people read when scaling or modernizing search (not a shopping list for this repo—pick what matches AWBW: **long turns, stochastic combat, simultaneous-ish resolution**).

**Model-based RL (learned dynamics)**

- **MuZero** and follow-ons (**Sampled MuZero**, **EfficientZero**, **ReAnalyze**): plan in a **learned model** when a perfect simulator is expensive or you want imagination rollouts; trade-offs are **model error** vs **engine fidelity** (for oracle-locked AWBW, the engine is the ground truth—learned models are a research fork, not a drop-in).

**Search algorithm / policy at the root**

- **Gumbel AlphaZero**: root selection / policy improvement without needing the full visit-count distribution in some setups; relevant when **visit budgets are small** or you want sharper action selection.
- **PUCT / UCB variants** and **progressive widening** (widen the tree as visits grow): classic when **branching is huge** (AWBW macro-actions already compress this; still useful if you re-expand the action space).
- **Levin / policy-guided tree search** (broad family): bias expansion toward **promising shallow prefixes** when depth is costly.

**Stochasticity and imperfect information**

- **Chance nodes** or **explicit luck modeling** in MCTS when transitions are stochastic (combat RNG): some pipelines use **determinization**, **averaging over samples**, or **chance-node MCTS**; mis-handling RNG is a common desync / wrong-posterior source—**keep oracle RNG discipline** if you experiment here.
- **ISMCTS / information-set MCTS** (card games, fog): less central if AWBW eval is **full observability**, but relevant if you ever search under **hidden information** or human-style partial maps.

**Systems: what actually wins wall clock**

- **Virtual loss** + **root-parallel / tree-parallel** MCTS: scale CPU workers without duplicating the same children forever.
- **Batched GPU inference** at the leaves (you already have this as a campaign item): often dominates once sims/sec is high enough.
- **Anytime / time-budgeted** search (campaign todo): AlphaStar and production engines almost always think in **ms per decision**, not a fixed visit count—same idea as your production MCTS notes in MASTERPLAN §4.

**Adjacent RL (not pure MCTS)**

- **MCTS-as-policy-improvement** vs **pure policy-gradient** ablations; **Go-Explore**-style exploration for sparse rewards; **R2D2**-style off-policy LSTM stacks—these are **not required** for a strong AZ baseline but explain some post-2018 “why doesn’t my tree search dominate” conversations.

When you dig into papers, skim for **(a)** branching assumptions, **(b)** deterministic vs stochastic simulators, **(c)** single-agent vs two-player zero-sum, and **(d)** whether results assume **full observability**—then map to AWBW’s **turn-wide** search and **oracle-locked** engine.

## Related reading

- [`docs/mcts_review_composer_o.md`](c:\Users\phili\AWBW\docs\mcts_review_composer_o.md) — Phase 11b audit.
- [`.cursor/plans/train.py_fps_campaign_c26ce6d4.plan.md`](c:\Users\phili\AWBW\.cursor\plans\train.py_fps_campaign_c26ce6d4.plan.md) — Phase 11 MCTS spine (historical).
- [`docs/SOLO_TRAINING.md`](c:\Users\phili\AWBW\docs\SOLO_TRAINING.md) — fleet MCTS gate behavior.

---

## Plan file

`c:\Users\phili\AWBW\.cursor\plans\mcts_optimization_campaign.plan.md`
