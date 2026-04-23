---
name: Phase 1 foundation validation (curriculum + gates)
overview: >-
  Operationalize MASTERPLAN §3 and the Phase 1 curriculum ladder (Stages 0–4):
  attributable runs, narrow bootstrap on Misery + mirror CO/tier, slice metrics vs
  global Phase 1 Full Go gates, and telemetry (TensorBoard + game_log) so phase
  transitions are evidence-based — without skipping distribution rungs.
todos:
  - id: stage0-instrumentation
    content: >-
      Ensure every training/eval run is attributable — curriculum_tag or env name in
      logs; confirm game_log rows carry map_id, tier, p0_co_id, p1_co_id per schema;
      document how to slice TensorBoard and JSONL by tag.
    status: pending
  - id: stage1-narrow-bootstrap
    content: >-
      Configure Stage 1 narrow bootstrap — Misery ([data/gl_map_pool.json](c:\Users\phili\AWBW\data\gl_map_pool.json)) map_id 123858,
      Andy vs Andy (CO id 1), fixed T3 — in [train.py](c:\Users\phili\AWBW\train.py) / env factory; verify no accidental tier drift.
    status: pending
  - id: slice-metrics-scripts
    content: >-
      Add or tighten scripts/notebooks to compute slice-specific explained_variance
      proxy (from TB exports if needed), win rate, turns, property differential from
      [data/game_log.jsonl](c:\Users\phili\AWBW\data\game_log.jsonl); label outputs as slice vs global.
    status: pending
  - id: phase1-narrow-exit
    content: >-
      Satisfy Phase 1 Narrow exit — qualitative replay bar on Stage 1–2 map/mirror
      (per MASTERPLAN §6 reference) + stated slice metrics; explicitly do not treat as Phase 1 Full.
    status: pending
  - id: stage2-geometry
    content: >-
      Expand to Stage 2 — 2–5 structurally diverse Std maps, mirror or small CO set;
      verify transfer signals (property contest) before CO entropy.
    status: pending
  - id: stage3-co-generalization
    content: >-
      Stage 3 — stratified / full co_ids from enabled tiers on Stage 2 map set;
      monitor win rate vs checkpoint pool (52–62% healthy band per MASTERPLAN).
    status: pending
  - id: stage4-full-pool-gates
    content: >-
      Stage 4 / Phase 1 Full Go — all §3 gates on target distribution (explained_variance,
      win rates, game length trend, property differential); block Phase 2 production until met.
    status: pending
  - id: parallel-turn-level-api
    content: >-
      Calendar-risk parallel — implement/test turn-level rollout interface in engine
      (MASTERPLAN §4.2); does not substitute curriculum expansion.
    status: pending
  - id: masterplan-elo-league-backlog
    content: >-
      When ready (post Phase 1 Full / Phase 2 eval), paste [elo league plan](c:\Users\phili\AWBW\.cursor\plans\elo_league_masterplan.plan.md) draft
      into MASTERPLAN as §10.5 + optional §2/§7 cross-refs. Does not block Stages 0–4.
    status: pending
---

# Phase 1 foundation validation

Plan derived from [MASTERPLAN.md](c:\Users\phili\AWBW\MASTERPLAN.md) §3 (gates), curriculum ladder §3 “Curriculum and distribution”, and Phase 1 Narrow vs Full definitions. Small training changes and observability first; distribution expansion is sequential, not parallel guesswork.

## Non-goals

- Declaring Phase 1 complete on narrow-slice metrics alone.
- Competitive MCTS or hierarchical RL scope — only prerequisites where explicitly parallel (turn-level API).
- **Elo / league eval** — [draft + paste instructions for MASTERPLAN §10.5](c:\Users\phili\AWBW\.cursor\plans\elo_league_masterplan.plan.md) are deferred; not part of Phase 1 gates (optional Phase 2 eval/promotion layer).

## Repository truth

| Topic | Where |
|------|--------|
| Strategic thresholds and phase stack | [MASTERPLAN.md](c:\Users\phili\AWBW\MASTERPLAN.md) §2–§3 |
| Training entry / hyperparameters | [train.py](c:\Users\phili\AWBW\train.py) |
| Matchup and outcome telemetry | [data/game_log.jsonl](c:\Users\phili\AWBW\data\game_log.jsonl) |
| Map pool reference | [data/gl_map_pool.json](c:\Users\phili\AWBW\data\gl_map_pool.json) |
| Engine / replay cross-cutting work (separate track) | [awbw-engine-parity.plan.md](c:\Users\phili\AWBW\.cursor\plans\awbw-engine-parity.plan.md) |
| Elo league → MASTERPLAN doc (backlog) | [elo_league_masterplan.plan.md](c:\Users\phili\AWBW\.cursor\plans\elo_league_masterplan.plan.md) |

## Critical section — risks and wrong assumptions

- **Slice substitution:** Strong `explained_variance` on Misery + mirror does not imply strong V(s) on Stage 3–4; MCTS amplifies evaluator bias off-manifold (MASTERPLAN already warns).
- **Opponent pool health:** Win rate vs checkpoints >65% suggests weak pool, not genius agent — tune checkpoint cadence/size before restructuring the network.
- **Max-turn games:** Average length near cap often means non-decisive play — correlate with reward and replay quality before chasing policy entropy fixes alone.
- **What not to do:** Skip Stage 2 geometry before flooding CO entropy; treat prototype MCTS as validation of Phase 1 Full.

## Diagram — curriculum order

```mermaid
flowchart LR
  S0[Stage0_Instrumentation]
  S1[Stage1_NarrowBootstrap]
  S2[Stage2_GeometryMaps]
  S3[Stage3_COGeneralization]
  S4[Stage4_FullPool]
  S0 --> S1 --> S2 --> S3 --> S4
```

## After confirmation

Execute todos in dependency order; update `status` in this file’s frontmatter as items complete. Prefer merging duplicate observability work into Stage 0 rather than spreading one-off scripts.

---

## Plan file

```text
c:\Users\phili\AWBW\.cursor\plans\phase1-foundation-validation.plan.md
```
