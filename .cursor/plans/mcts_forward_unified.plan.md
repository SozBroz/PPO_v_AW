---
name: MCTS forward unified
overview: >-
  Single execution plan for all MCTS work we proceed with: turn-level PUCT scale-up
  (correctness, geometry, compute, escalator, anytime search), Phase 11f sim ladder
  + ROI logging. Part B audit items (turn traces, luck resamples, edge stats,
  production risk control) are implemented in engine + rl/mcts + symmetric eval
  (2026-04); remaining work is validation, profiling, adaptive K, batched NN,
  escalator, and time-budget search. Non-goals: train_advisor / in-rollout MCTS,
  fog ISMCTS. North-star thresholds remain MASTERPLAN §4 (Phase 1 Full, EV>0.6 on
  search mix, rollout perf). Staged rollout ladder: MASTERPLAN §14 (MCTS-0…4).
todos:
  - id: mcts-fwd-01-gates-baseline
    content: >-
      Preconditions: MASTERPLAN §4 — eval distribution matches intended search mix;
      explained_variance credible on that mix; symmetric eval harness + fixed matrix
      (maps/seats/seeds/checkpoints); document baseline winrate vs pool at sims
      16–128 before changing search.
    status: pending
  - id: mcts-fwd-02-profile
    content: >-
      Profile one full MCTS decision: wall split apply_full_turn vs SB3 predict /
      value / distribution; repeat at num_sims 256/512/800/1600; decide first lever:
      engine, NN batching, or branching.
    status: pending
  - id: mcts-fwd-03-rng-audit
    content: >-
      Audit luck_rng / combat RNG per simulation path (reproducible branch seeds,
      no cross-sim bleed); document TT/cache safety; add tests where gaps.
    status: pending
  - id: mcts-fwd-04-turn-trace-metadata
    content: >-
      Implement apply_full_turn step trace per docs/mcts_review_composer_o.md Part
      B.3: attack_damage_rolls, killed_unit, survived_at_hp, capture_interrupted,
      critical_threshold_event (extend on_step or return trace list; engine tests).
    status: completed
  - id: mcts-fwd-05-luck-resamples
    content: >-
      For plans with critical_threshold_event, run extra local resamples (replay
      same Action list, independent luck seeds); tunable mcts_luck_resamples; cap
      wall time.
    status: completed
  - id: mcts-fwd-06-edge-stats
    content: >-
      Extend TurnNode (or parallel struct) with value_variance (Welford),
      worst_p10_value reservoir, kill_probability from traces/resamples; extend
      _backup or resample path to feed stats.
    status: completed
  - id: mcts-fwd-07-production-risk-selection
    content: >-
      Root-only risk layer for production: constraints and/or tail-penalty score
      (e.g. blend mean + p10); log per-child stats; training/eval keep EV+PUCT default.
    status: completed
  - id: mcts-fwd-08-adaptive-k
    content: >-
      Adaptive root_plans / progressive widening; tune max_plan_actions from
      game_log turn lengths.
    status: pending
  - id: mcts-fwd-09-puct-sweep
    content: >-
      Grid on symmetric_checkpoint_eval: c_puct, dirichlet, min_depth, temperature;
      fixed wall budget; metrics: winrate, mcts_decision_wall p50/p95, PV depth,
      root visit entropy.
    status: pending
  - id: mcts-fwd-10-batch-nn
    content: >-
      Batch leaf/value (optional policy) forwards per MCTS wave; virtual loss if
      parallel expand.
    status: pending
  - id: mcts-fwd-11-escalator-11f
    content: >-
      Phase 11f: eval_only sim ladder 16→32→64→128 gated on EV + winrate vs pool +
      no desync; log sim-vs-winrate curve to logs/mcts_escalator.jsonl; align
      tools/mcts_escalator.py + fleet audit (marginal ROI before doubling).
    status: pending
  - id: mcts-fwd-12-time-budget
    content: >-
      Optional wall-clock budget per P0 root (anytime loop); log sims_used +
      wall_s; wire symmetric_checkpoint_eval / live play when entrypoint defined.
    status: pending
  - id: mcts-fwd-13-search-control
    content: >-
      Cheap complexity signal → time multiplier or dynamic K at root; log factor
      per turn.
    status: pending
  - id: mcts-fwd-14-doc-sync
    content: >-
      On milestone landings: update MASTERPLAN §4 pointer text if time-budget or
      risk-selection ships; keep docs/mcts_review_composer_o.md Part B in sync with
      code truth.
    status: completed
---

# MCTS forward — unified execution plan

This file **consolidates forward MCTS work** from:

| Source (archive / reference) | Role |
| ------------------------------ | ---- |
| [`.cursor/plans/mcts_optimization_campaign.plan.md`](c:\Users\phili\AWBW\.cursor\plans\mcts_optimization_campaign.plan.md) | Scale-up strategy, pillars, success criteria, research pointers |
| [`.cursor/plans/train.py_fps_campaign_c26ce6d4.plan.md`](c:\Users\phili\AWBW\.cursor\plans\train.py_fps_campaign_c26ce6d4.plan.md) § Phase 11 / **11f** | Shipped 11a–11e spine; **pending 11f** escalator narrative |
| [`MASTERPLAN.md`](c:\Users\phili\AWBW\MASTERPLAN.md) §4, §14 | Phase 2 thresholds, staged MCTS-0…4, production vs prototype |
| [`docs/mcts_review_composer_o.md`](c:\Users\phili\AWBW\docs\mcts_review_composer_o.md) Part A + **Part B** | Checklist audit + **trace metadata, resamples, edge stats, risk control** |

**Proceed from this file** for todos and ordering; the sources above remain historical detail.

## Non-goals (unless explicitly reopened)

- **`train_advisor`** / MCTS inside PPO `model.learn` — off-policy trap; orchestrator refuses.
- **Fog / ISMCTS** — belief-state search; out of scope until MASTERPLAN fog lane.

## Already shipped (do not duplicate)

- **11a** [`engine/game.py`](c:\Users\phili\AWBW\engine\game.py) `apply_full_turn`
- **11b** [`rl/mcts.py`](c:\Users\phili\AWBW\rl\mcts.py) turn-level PUCT
- **11c** `--mcts-mode eval_only` in [`train.py`](c:\Users\phili\AWBW\train.py), [`scripts/symmetric_checkpoint_eval.py`](c:\Users\phili\AWBW\scripts\symmetric_checkpoint_eval.py)
- **11d** [`tools/mcts_health.py`](c:\Users\phili\AWBW\tools\mcts_health.py) + orchestrator merge path
- **11e** `terrain_usage_p0` + schema for gate inputs
- **11f / Part B (stochastic root risk, 2026-04)** — `apply_full_turn(..., return_trace=True)` (optional 5-tuple); [`rl/mcts.py`](c:\Users\phili\AWBW\rl\mcts.py) `EdgeStats`, `luck_resamples`, `risk_mode`, root JSONL log; [`scripts/symmetric_checkpoint_eval.py`](c:\Users\phili\AWBW\scripts\symmetric_checkpoint_eval.py) CLI + telemetry (`mcts_root_entropy`, `mcts_chosen_risk`); tests in [`tests/test_mcts.py`](c:\Users\phili\AWBW\tests\test_mcts.py). See [`docs/mcts_review_composer_o.md`](c:\Users\phili\AWBW\docs\mcts_review_composer_o.md) Part B, [`MASTERPLAN.md`](c:\Users\phili\AWBW\MASTERPLAN.md) §14.

## Recommended work order (high level)

1. **mcts-fwd-01** — gates + baseline matrix (nothing else is interpretable without this).
2. **mcts-fwd-02** + **mcts-fwd-03** — profile + RNG proof (correctness before scale).
3. ~~**mcts-fwd-04** → **mcts-fwd-07**~~ — **done (2026-04)** — trace → resamples → edge stats → production risk (stochastic combat story). Remaining: prove value-head frame + A/B `risk_mode` on target mix.
4. **mcts-fwd-08** → **mcts-fwd-10** — geometry + sweeps + batched NN (quality and wall-clock).
5. **mcts-fwd-11** — escalator / Phase 11f + ROI logs (prove each sim doubling pays).
6. **mcts-fwd-12** → **mcts-fwd-13** — anytime search + search-control heuristics (production-shaped).
7. **mcts-fwd-14** — documentation sync after each major landing.

Parallelism: **04–07** can overlap **08–09** after **03** is underway, but do not skip **01–03**.

## Architecture truth (unchanged)

| Piece | Role |
| ----- | ---- |
| `apply_full_turn` | One full player turn from SELECT; rollout primitive for MCTS |
| `rl/mcts.py` | PUCT at turn nodes; `root_plans` children; Dirichlet at root |
| `symmetric_checkpoint_eval.py` | Primary `eval_only` MCTS consumer + telemetry |
| `mcts_health.py` / `mcts_escalator.py` | Competence + sim budget discipline |

## Risks (carried forward)

- Weak **V(s)** on searched states + high sims → confidently wrong lines.
- **Promotion** with Dirichlet / temperature → non-reproducible metrics.
- **Caching / TT** without RNG discipline → silent wrong search.
- **Sub-step trees** — do not build; stay turn-level.

## Success criteria (program-level)

1. Target sim budget (e.g. 800–1600): median decision time acceptable; p95 bounded.
2. RNG story documented + tested for fixed root + seed.
3. Adaptive branching ablation wins vs fixed K at same wall time.
4. Batched NN landed **or** documented proof that rollouts are the bottleneck + next engine task.
5. Escalator / JSONL shows **marginal ROI** for sim doubling.
6. Part B: traces + resamples + edge stats + production risk **implemented and tested** per audit doc — **met** (see § Already shipped 11f/Part B).

---

## Plan file (absolute path)

`c:\Users\phili\AWBW\.cursor\plans\mcts_forward_unified.plan.md`
