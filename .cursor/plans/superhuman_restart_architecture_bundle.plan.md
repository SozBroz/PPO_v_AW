---
name: Superhuman restart — architecture, encoder, training-paradigm bundle
overview: "We have to restart training from scratch to fix the policy head bottleneck (256-d → 35k flat Linear after AdaptiveAvgPool destroys positional structure before action selection). Bundle every other shape-locked or contract-breaking change into the same restart so we only pay the checkpoint-invalidation cost once: deeper/wider residual tower with no spatial pooling, factored spatial policy head, threat/reachability influence channels in the encoder, ego-centric (\"me\" frame) encoder (MASTERPLAN §9 Tier 4), default reward flip to Φ-shaping (MASTERPLAN §11), seat-balanced PPO actor over both engine seats, and PFSP opponent sampling on top of the existing checkpoint pool. Defer fog (§8), native compilation (§12), and HRL (§5) — they remain off-bundle."
todos:
  - id: spec-and-compute-budget
    content: "Write the architectural contract before code — new spatial channel set, new factored action head layout, residual tower depth/width chosen against a measured compute envelope (target FPS at expected `n_envs` per machine, with the §10 multi-PC sync in scope). Land as a short design note under [docs/](D:\\AWBW\\docs) and link from this plan. Goal: nothing below this todo proceeds until the shape contract and FPS budget are written down."
    status: completed
  - id: encoder-equivalence-harness
    content: Add a regression harness in [tests/](D:\AWBW\tests) that encodes a fixed corpus of states and asserts byte-equal observation tensors and identical action-mask shape against a stored snapshot. This is the gate we will use forever after to know whether a future change reuses checkpoints or forces a restart (per MASTERPLAN §12.2 operational rule). Build this BEFORE we mutate [rl/encoder.py](D:\AWBW\rl\encoder.py) so the diff is auditable; snapshot today's 63-channel layout as the "pre-restart" baseline.
    status: completed
  - id: influence-channels
    content: "Extend [rl/encoder.py](D:\\AWBW\\rl\\encoder.py) with derived tactical channels per `docs/restart_arch/influence_channels_spec.md` (composer-authored): `threat_in_p0`, `threat_in_p1`, `reach_p0`, `reach_p1`, `turns_to_capture_p0`, `turns_to_capture_p1` (6 channels) plus a 7th `defense_stars` channel (per-tile `TerrainInfo.defense / 4`, ride existing terrain cache → zero runtime cost). New `N_SPATIAL_CHANNELS = 70` (63 + 7). Helpers live in `engine/threat.py` to keep the encoder thin. This is the largest sample-efficiency lever in the bundle and the most defensible call from the original critique table."
    status: completed
  - id: ego-centric-encoder
    content: MASTERPLAN §9 Tier 4 — swap P0/P1 channel blocks and the relevant scalars so "me" is always one block, with the engine seat preserved separately. This is the prerequisite for any "both seats" training (§9 doctrine) and for a single net to control either seat at inference. Land it now while we are already breaking the encoder contract.
    status: completed
  - id: log-schema-bump
    content: "Bump `log_schema_version` in [rl/env.py](D:\\AWBW\\rl\\env.py) and `game_log.jsonl` writers to record `me ∈ {0,1}` (engine seat the learner controlled), `opening_player`, `arch_version`, `reward_mode` (`level` vs `phi`), and `opponent_sampler` (`uniform` vs `pfsp`). Backward-compat: readers in [analysis/](D:\\AWBW\\analysis) and [scripts/eval_imitation.py](D:\\AWBW\\scripts\\eval_imitation.py) must default missing fields safely so old rows still parse."
    status: completed
  - id: residual-tower
    content: Replace the 3-block 64→128 trunk in [rl/network.py](D:\AWBW\rl\network.py) with an AlphaZero-style residual tower (target ~10–16 blocks × 128–192 channels — exact number set by the compute-budget todo). Drop the `AdaptiveAvgPool2d((8,8))` in both `AWBWFeaturesExtractor` and `AWBWNet`; keep stride-1 convolutions throughout so per-tile features survive into the head. Switch to GroupNorm or pre-activation ResBlocks if BatchNorm proves brittle under PPO's small minibatch updates.
    status: completed
  - id: move-encoding-redesign
    content: "**Wave-1 finding:** [rl/env.py](D:\\AWBW\\rl\\env.py) `_action_to_flat` for `SELECT_UNIT` uses `unit_pos` only and ignores `move_pos`. In MOVE stage every legal destination collapses to ONE flat index; the decoder picks the first match by sort order. The destination is therefore chosen by engine enumeration, not by the policy. A spatial policy head cannot raise the ceiling on the most frequent action type until this is fixed. **Approach:** in MOVE stage, encode `SELECT_UNIT` actions by `move_pos` at a new offset (recommended `_MOVE_OFFSET = 1818`, which sits in the existing unused range 1818–3499 → 900 destination indices, no `ACTION_SPACE_SIZE` bump). SELECT-stage `SELECT_UNIT` continues to encode by `unit_pos`. See `docs/restart_arch/move_encoding_redesign.md` (composer-authored)."
    status: completed
  - id: spatial-policy-head
    content: Factor the flat `Linear(256, 35_000)` policy into per-action-type `Conv2d(C, K_type, 1)` heads producing per-tile logits, with `_flat_to_action` updated to project (action_type, row, col, sub-arg) tuples back into the same legal index space. Depends on `move-encoding-redesign` so the spatial MOVE head actually drives destination choice. Action mask plumbing in [rl/env.py](D:\AWBW\rl\env.py) and `network.AWBWNet.forward` must masked-fill at the spatial logits directly. Keep the head outputs shape-stable across maps by always operating on the padded GRID_SIZE × GRID_SIZE.
    status: completed
  - id: value-head-rework
    content: Replace `Linear(256, 1)` with a small conv → 1×1 conv → global average pool → scalar value head (AlphaZero pattern). Init small (current `gain=0.01` is correct). Sanity-test that explained_variance under the new head reaches at least the level the old head hit on Stage 1 bootstrap before we trust the rest of the network.
    status: completed
  - id: phi-default-flip-and-learner-frame
    content: Flip default `AWBW_REWARD_SHAPING` to `phi` per MASTERPLAN §11.4 AND rewrite `_compute_phi` in [rl/env.py](D:\AWBW\rl\env.py) to compute Φ in the **learner frame** (me − enemy) instead of the hardcoded P0 frame (`p0_val − p1_val`, `cap_p0 − cap_p1`). Same for the legacy `level` reward path while it still exists. This is a shape-locked return-distribution change that V(s) will regress against, so it must ride the same restart. Wave-1 ego-centric refactor map (`docs/restart_arch/ego_centric_refactor_map.md` §7) flagged this as the dependency for any seat-balanced training. Pre-PPO, run [tools/phi_smoke.py](D:\AWBW\tools\phi_smoke.py) on Misery T3 mirror per §11.2 and record numbers under the new contract.
    status: completed
  - id: seat-balanced-actor
    content: Per MASTERPLAN §9 doctrine, alternate which engine seat the **learner** controls across [rl/self_play.py](D:\AWBW\rl\self_play.py) workers (half learner-as-P0 vs pool-as-P1, half learner-as-P1 vs pool-as-P0). Do **not** run policy gradients on actions from the frozen pool opponent — those are off-policy under PPO's clipped surrogate. Critic may train on both seats with sign-flipped returns under the zero-sum assumption; gate that behind a flag so we can A/B it.
    status: completed
  - id: pfsp-opponent-sampler
    content: Add prioritized fictitious self-play sampling on top of the existing `_CheckpointOpponent` rotation in [rl/self_play.py](D:\AWBW\rl\self_play.py). Weight historical checkpoints by `(1 − win_rate_against)^p` so the trainee gets the opponents it currently loses to. Default `p=2`; expose as `--pfsp-power`. This addresses the "counters all-in vs counters turtle" framing without adopting a full multi-population league yet.
    status: completed
  - id: async-vector-env
    content: MASTERPLAN §12.1 row 1 — swap `SubprocVecEnv` for shared-memory `AsyncVectorEnv` to claw back +10–30% of total FPS. This one is checkpoint-safe and could ship without the restart, but bundling it means we baseline the new architecture against the new throughput from day one and avoid re-baselining gates twice.
    status: completed
  - id: scratch-smoke-stage1
    content: "Short PPO scratch run on Stage 1 (Misery T3 Andy mirror per MASTERPLAN §3 ladder), under the bundle. Targets — episode rollouts complete, `explained_variance` moving off zero, replay viewer shows agent ending turns and contesting properties (§6 Phase 1 minimum bar). Fail fast: this is the moment to discover any contract bug (mask shape, action reprojection, value head dim) before committing to the full ladder."
    status: pending
  - id: phase1-ladder-rerun
    content: Re-run the MASTERPLAN §3 Stage 1 → Stage 4 curriculum ladder against the new architecture, all gates re-evaluated as fresh baselines (the old `explained_variance` plateau numbers are irrelevant to the new head). Phase 1 Full Go criteria from MASTERPLAN §3.5 still apply. [.cursor/plans/phase1-foundation-validation.plan.md](D:\AWBW\.cursor\plans\phase1-foundation-validation.plan.md) drives the per-stage execution; this todo just tracks "the ladder is now running on the new contract."
    status: pending
  - id: masterplan-update
    content: Update [MASTERPLAN.md](D:\AWBW\MASTERPLAN.md) — §1 architecture table (new tower, new head, ego-centric encoder, reward=phi default), §9 Tier 4 line ("funded as of this restart"), §11.4 default-flip checkbox ticked, and a new dated note that the pre-restart checkpoint line is archived but not deleted (kept for regression baselines and BC initialization experiments).
    status: completed
isProject: true
---

# Superhuman restart — architecture, encoder, training-paradigm bundle

## What this plan is

A single, atomic restart that bundles every change which would otherwise
each cost us a from-scratch training run. The triggering item is the
critique of [rl/network.py](D:\AWBW\rl\network.py): the current
`AdaptiveAvgPool2d((8,8)) → flatten → Linear(8192+17, 256) → Linear(256, 35_000)`
path is the wrong inductive bias for a positional 30×30 game and is
load-bearing on most of why we have not reached superhuman with the
existing trunk. Once we accept that fix, every other shape-locked
improvement (encoder channels, ego-centric frame, action layout, reward
default) becomes free to bundle, because the restart cost has already
been paid.

## Non-goals

- **Fog of war / POMDP work** — explicitly off-bundle per MASTERPLAN §8.
  Fog is a separate product line with its own checkpoint contract.
- **Native compilation (Cython / Numba / mypyc)** — MASTERPLAN §12 is
  shelved; the restart does not unblock it because the bundle does not
  change the engine internals it would target.
- **Hierarchical RL (Macro/Micro)** — Phase 3, MASTERPLAN §5. The new
  spatial head + influence channels may extend the flat-architecture
  plateau enough to make HRL deferrable; build, measure, then decide.
- **Full AlphaStar-style multi-population league** — PFSP sampling is in
  scope; persistent main-exploiter / league-exploiter populations are not.
- **MCTS production rollout** — Phase 2, MASTERPLAN §4. Prerequisite gates
  are unchanged; this bundle just makes the V(s) that MCTS will rely on
  a stronger evaluator.

## Repository truth

| Topic | Where |
|------|--------|
| Shipped policy/value net (restart stack) | [rl/network.py](D:\AWBW\rl\network.py) |
| Current encoder (channels and scalars) | [rl/encoder.py](D:\AWBW\rl\encoder.py) |
| Action space and mask construction | [rl/env.py](D:\AWBW\rl\env.py) |
| Self-play loop, opponent rotation, checkpoint pool | [rl/self_play.py](D:\AWBW\rl\self_play.py) |
| MaskablePPO trainer entry | [rl/ppo.py](D:\AWBW\rl\ppo.py), [train.py](D:\AWBW\train.py) |
| Reward shaping (Φ implementation) | [rl/env.py](D:\AWBW\rl\env.py), [tools/phi_smoke.py](D:\AWBW\tools\phi_smoke.py) |
| Phase / gate strategy | [MASTERPLAN.md](D:\AWBW\MASTERPLAN.md) §1, §3, §9, §11 |
| Curriculum execution plan | [.cursor/plans/phase1-foundation-validation.plan.md](D:\AWBW\.cursor\plans\phase1-foundation-validation.plan.md) |
| Reward-shaping plan (Φ origin) | [.cursor/plans/rl_capture-combat_recalibration_4ebf9d22.plan.md](D:\AWBW\.cursor\plans\rl_capture-combat_recalibration_4ebf9d22.plan.md) |
| Multi-PC sync (impacts FPS budget for tower sizing) | MASTERPLAN §10 |

## Bundle rationale — why now and why all at once

Restarting training is the binding cost. Every item in the table below
either changes the encoder shape, changes the policy/value head shape, or
changes the reward distribution that V(s) regresses against. Doing them
serially means N restarts; bundling means one.

| Change | Forces restart? | Bundled here? | Why |
|---|---|---|---|
| Spatial policy head | Yes (action head shape) | Yes | The trigger |
| **MOVE-encoding redesign** | **Yes (action layout, mask bits)** | **Yes (added post-wave-1 finding)** | **Without it, spatial head can't actually drive MOVE destination — wave-1 inventory composer proved `_action_to_flat` collapses all destinations to one flat index** |
| Wider/deeper residual tower | No (could be retrained from old encoder), but we are restarting anyway | Yes | Free with bundle |
| Drop AvgPool | Yes (changes feature shape into head) | Yes | Required by spatial head |
| Influence/threat channels (6) + defense_stars (1) | Yes (encoder channel count → 70) | Yes | Highest sample-efficiency lever |
| Ego-centric encoder | Yes (channel layout) | Yes | Unblocks §9 both-seats training |
| Φ reward as default + **learner-frame** | Yes (return distribution) | Yes | §11.4 default-flip moment + ego-centric §7 finding |
| Seat-balanced actor | No (data-collection only) | Yes | Free with ego-centric encoder; depends on it |
| PFSP opponent sampling | No (data-collection only) | Yes | Cheap; lets us validate league framing in the same eval window |
| AsyncVectorEnv swap | No (perf only) | Yes | Re-baselining gates twice is worse than bundling |
| Fog / POMDP | Yes | No | §8 — separate program |
| Native compilation | No (preserves obs) | No | §12 shelved |
| Hierarchical RL | Yes (architecture overhaul) | No | §5 — measure first |

## Architecture sketch (shipped; detail + param math in [docs/restart_arch/compute_budget.md](D:\AWBW\docs\restart_arch\compute_budget.md))

Trunk:

```text
spatial (B, H, W, C_new)  →  permute → (B, C_new, H, W)
  → Conv2d(C_new, F, 3, pad=1) + GroupNorm + ReLU
  → ResBlock × N         (stride 1 throughout, no AvgPool)
  → features (B, F, H, W)             # F ≈ 128–192, N ≈ 10–16
```

Heads:

```text
Spatial policy:
  per action-type k ∈ {move, attack, capture, build, ...}:
    Conv2d(F, K_k, 1) → (B, K_k, H, W) → flatten → reproject to flat index space
  scalar action-types (END_TURN, COP, SCOP):
    GAP(features) → Linear → logits

Value:
  Conv2d(F, F/4, 1) → ReLU → Conv2d(F/4, 1, 1) → GAP → tanh → scalar
```

Encoder additions (final list set by spec todo, not exhaustive):

- Threat-in (per-side): expected damage incoming to each tile next turn
- Reachability frontier (per-side): reuse engine BFS over move costs
- Turns-to-capture (per side, per property tile)
- Ego-centric channel ordering: own / enemy block instead of P0 / P1 block

## Critical section — risks and what not to do

- **Don't bundle fog.** It looks adjacent (encoder + value-head changes)
  but is its own checkpoint contract per MASTERPLAN §8.1. Bundling would
  silently force the same restart again the moment we want a non-fog model.
- **Don't expand the action space while changing its representation.**
  Keep the legal action set behaviourally identical; only the
  policy-head wiring changes. If we want new action types (e.g. higher-level
  build orders), do that as a separate bundle after the new architecture
  ships and the gates are met.
- **GroupNorm vs BatchNorm.** PPO collects small minibatches; BN can
  destabilize the value head when batch statistics drift between rollout
  and update. Default to GroupNorm or LayerNorm in the new tower; treat
  any decision to keep BN as needing an explicit justification.
- **Don't trust old `explained_variance` thresholds.** The MASTERPLAN §3
  gates are correct in spirit but the numerical baselines were measured
  against the old return distribution and the old head. After the
  restart, re-derive what "stable plateau" looks like before declaring
  Stage gates met.
- **Don't ship PFSP without the seat-balanced actor.** PFSP weights
  opponents by win rate; if the learner only ever plays as P0, the win
  rates are conditional on a seat the learner cannot escape, and the
  sampler will overweight asymmetric matchups for the wrong reason.
  Order matters — `seat-balanced-actor` before `pfsp-opponent-sampler`.
- **Don't delete the pre-restart checkpoint line.** Keep it on disk for
  (a) regression-tower comparisons, (b) optional behaviour cloning warm
  starts on the new architecture, and (c) honest "did the new arch
  actually beat the old one?" head-to-head evals via the existing pool /
  shared-latest gate.
- **Encoder equivalence harness comes first.** Without it, every future
  encoder edit risks silent shape drift; with it, future architectural
  iterations can be reasoned about without paying another restart.
  Build the harness against today's 63-channel encoder before we touch
  anything.
- **MCTS prereqs are unchanged.** This bundle does not satisfy
  MASTERPLAN §4.2 Phase 2 gates by itself — Phase 1 Full on the new
  architecture is still the prerequisite for production MCTS.

## Bundle vs incremental — items deliberately deferred

These are real improvements, but their cost-benefit is better outside
this bundle:

- **Macro/Micro hierarchical RL.** The new spatial head provides
  per-tile policy logits; that is materially closer to "tell each unit
  where to go" than the old flat head. Plausible the flat-architecture
  plateau moves out far enough that HRL becomes a Phase 3 decision, not
  a Phase 1 emergency. Decide after the new ladder runs.
- **Multi-population league (main-exploiter, league-exploiter).** PFSP
  on the existing pool gets us 80% of the "counters all-in *and*
  counters turtle" benefit at 10% of the engineering cost. Revisit if
  the bundle plateaus under PFSP.
- **Auto-regressive action decoder (AlphaStar-style).** The factored
  spatial head is sufficient for AWBW's action structure; an
  auto-regressive head only earns its complexity if we see specific
  evidence of conditional-action pathologies (e.g. picking the right
  unit but the wrong target tile despite a strong threat map).
- **Belief-conditioned value head.** Out of scope under the §8 fog
  separation; would only earn its keep on a fog product line.

## Diagram — bundle dependency order

```mermaid
flowchart TD
  A[spec_and_compute_budget]
  B[encoder_equivalence_harness]
  C[influence_channels]
  D[ego_centric_encoder]
  E[log_schema_bump]
  F[residual_tower]
  G[spatial_policy_head]
  H[value_head_rework]
  I[phi_default_flip]
  J[seat_balanced_actor]
  K[pfsp_opponent_sampler]
  L[async_vector_env]
  M[scratch_smoke_stage1]
  N[phase1_ladder_rerun]
  O[masterplan_update]

  A --> B
  B --> C
  B --> D
  C --> F
  D --> F
  D --> J
  F --> G
  F --> H
  G --> M
  H --> M
  I --> M
  E --> M
  J --> K
  K --> M
  L --> M
  M --> N
  N --> O
```

## After confirmation

Execute todos in dependency order — `spec-and-compute-budget` first
(nothing else proceeds without the shape contract and FPS budget on
disk), then `encoder-equivalence-harness` (so subsequent encoder diffs
are auditable), then encoder edits, then network, then training-paradigm
edits, then the scratch smoke. Phase 1 ladder rerun is governed by the
existing
[.cursor/plans/phase1-foundation-validation.plan.md](D:\AWBW\.cursor\plans\phase1-foundation-validation.plan.md);
this plan only tracks "the ladder runs on the new contract." Update todo
`status` in this file's frontmatter as items complete.

---

## Plan file

```text
D:\AWBW\.cursor\plans\superhuman_restart_architecture_bundle.plan.md
```
