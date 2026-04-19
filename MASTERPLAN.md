# AWBW-RL Master Plan

*Last updated: 2026-04-18*

This document is the strategic north star for the AWBW reinforcement learning project.
It records where we are, where we are going, and — critically — the concrete thresholds
that should gate each phase transition.

---

## 1. Where We Are Now

### Architecture (as of 2026-04-16)

| Component | Status | Detail |
|---|---|---|
| Feature extractor | **Active** | `AWBWFeaturesExtractor` — ResNet (3 blocks, 64→128ch) + scalar fusion → 256-dim |
| Policy | **Active** | SB3 `MultiInputPolicy` with `net_arch=[]` — Linear 256→35k + action mask |
| Value head | **Active** | Linear 256→1, step-level V(s) |
| Opponent | **Active** | `_CheckpointOpponent` — rotates through historical checkpoints, falls back to random |
| Reward | **Active** | Terminal ±1.0 + property delta ×0.005 + unit value differential ×2e-6 |
| Training | **Active** | MaskablePPO, n_steps=512, n_envs=4, batch=256, γ=0.99, λ=0.95 |

### What the Current Run Is Building

The network is learning three things simultaneously:

1. **Board reading** (`AWBWFeaturesExtractor` weights) — how to compress a 30×30×59 spatial
   grid into a 256-dim representation that captures unit positions, terrain relationships,
   property ownership, and threat geometry. This knowledge is **architecture-portable** and
   will transfer directly into any future model.

2. **Step-level tactics** (policy head weights) — which of the ~35k actions is best given
   the current board. This includes: which unit to move, where to move it, whether to attack
   or wait. The model acts one sub-step at a time (SELECT → MOVE → ACTION per unit) with
   no explicit memory across steps. Tactical competence — don't suicide, contest properties,
   use terrain cover — is learned here.

3. **State evaluation** (value head weights) — V(s), the expected future return from the
   current position. This is the single most important output for long-term intelligence.
   A strong V(s) is what allows the agent to recognize a winning position before the win
   materialises, and is the prerequisite for everything in Phase 2 and 3.

### Current Planning Horizon

The agent cannot plan strategically. It acts greedily against a learned value function.
The effective credit-assignment depth under γ=0.99, λ=0.95 is:

```
(0.99 × 0.95)^N = 0.9405^N

N=20  steps: 29% signal remaining   (~1 player turn of context)
N=50  steps: 5%  signal remaining   (~2 player turns)
N=100 steps: 0.3% signal remaining  (effectively invisible)
```

One full AWBW player turn ≈ 20–30 RL steps (8 units × 2–3 sub-steps each).
The agent is functionally optimising 1–2 turns ahead. Everything beyond that
depends entirely on how well V(s) has learned to evaluate board positions.

---

## 2. The Dependency Stack

Every phase below depends on the one beneath it being validated.
Do not skip rungs — **distribution** (maps / COs / tiers) is a rung, not only TensorBoard metrics.

```
Phase 3 — Hierarchical RL (Macro/Micro)
      ↑  requires: Phase 1 Full Go + qualitative replay bar on target mix + stable Micro + strong V(s)
Phase 2 — MCTS (production strength)
      ↑  requires: Phase 1 Full Go on Stage 3–4 distribution + turn-level sim + EV gate on that slice
Phase 2 — MCTS (prototype / plumbing)
      ↑  optional after: turn-level API + narrow curriculum (Stage 1–2) — validates wiring, not competitive eval
Phase 1 Full — validation on target distribution (Stage 3–4)
      ↑  requires: curriculum expanded past narrow bootstrap; gates in §3 apply
Phase 1b — Distribution expansion (maps, then CO/tier diversity)
      ↑  requires: narrow bootstrap showing signal (Phase 1 Narrow exit — §3)
Phase 1a — Curriculum bootstrap (fixed map / CO / tier + honest slice metrics)
      ↑  requires: AWBWFeaturesExtractor + dense rewards + checkpoint opponents + logged matchup tags
Phase 0 — Infrastructure                 ← COMPLETE
      (AWBWNet wired, rewards shaped, memory managed, self-play opponents active)
```

---

## 3. Phase 1 — Foundation Validation

**Goal:** Confirm the current architecture is learning meaningful representations
before investing in Phase 2 or 3.

### What to Monitor in TensorBoard

Launch TensorBoard with:
```
tensorboard --logdir logs/
```

#### 3.1 Value Loss — Primary Signal

The value loss measures how well V(s) predicts actual returns.

| Metric | Healthy | Warning | Abort |
|---|---|---|---|
| `train/value_loss` trend | Decreasing, then stable plateau | Oscillating without trend | Increasing after 500k steps |
| `train/explained_variance` | Growing toward 0.7–0.9 | Stuck below 0.2 after 1M steps | Negative (V(s) is worse than a constant) |

`explained_variance` is the most important single number in TensorBoard.
It measures what fraction of the variance in actual returns V(s) explains.
- **< 0.1**: V(s) is noise. Do not advance.
- **0.1–0.4**: V(s) is learning something but weak.
- **0.4–0.7**: V(s) has real signal. Phase 2 becomes viable.
- **> 0.7**: Strong V(s). Green light for Phase 2.

#### 3.2 Win Rate vs Opponent Pool

Track win rates from `data/game_log.jsonl`. Parse with:
```python
import json, pandas as pd
games = [json.loads(l) for l in open("data/game_log.jsonl")]
df = pd.DataFrame(games)
df["agent_won"] = df["winner"] == 0
df.groupby(pd.cut(df.index, 10))["agent_won"].mean()
```

| Win rate vs random | Interpretation |
|---|---|
| < 40% after 500k steps | Something is wrong — reward signal, action mask, or engine bug |
| 40–65% | Learning basic tactics, normal early phase |
| 65–80% | Solid tactical play — checkpoint opponents should now provide real signal |
| > 80% vs random | Random opponents no longer useful; checkpoint quality is what matters |

Win rate vs checkpoint pool should stabilise around **52–60%**. If it stays
near 50% indefinitely, the agent has plateaued and needs a structural change.
If it exceeds 65% consistently, your checkpoint pool is too weak (increase
`checkpoint_pool_size` or save checkpoints more frequently).

#### 3.3 Policy Entropy

Entropy measures how exploratory the policy is. With a 35k action space,
a uniform policy has entropy ≈ ln(35000) ≈ 10.5 nats. Your masked policy
will start lower since ~99% of actions are masked.

| `train/entropy_loss` | Interpretation |
|---|---|
| Decreasing slowly | Policy is specialising — healthy |
| Collapsing rapidly to near zero | Policy entropy collapse — increase `ent_coef` |
| Not moving at all | Learning signal not reaching policy — check rewards |

#### 3.4 Game Length

From `game_log.jsonl`, track `turns` distribution.

| Average game length | Interpretation |
|---|---|
| Near `MAX_TURNS` (100) | Agent is not learning to win — stuck or draws |
| Decreasing over time | Agent learning decisive play — good |
| Very short games (< 10 turns) | Possible reward exploit — inspect replays |

#### 3.5 Property and Unit Value Differential at Game End

From `game_log.jsonl`, track `property_count[0] - property_count[1]` for
winning games. A good agent should win with a clear property advantage, not
by luck. This differential should widen over training.

### Curriculum and distribution (Phase 1 ladder)

Train and evaluate on an explicit **curriculum** so narrow experiments (one map, mirror matchup) do not get confused with **full-pool** health. Log `map_id`, `tier`, `p0_co_id`, `p1_co_id` (already in `game_log` schema) and add a **run tag** (`curriculum_tag` / env name) so TensorBoard and logs can be sliced by stage.

Example ladder (tune with telemetry):

- **Stage 0 — Instrumentation:** ensure every run is attributable (tag + logged matchup fields). No change to learning objective beyond observability.
- **Stage 1 — Narrow bootstrap:** Misery (`map_id` **123858**, per `data/gl_map_pool.json`), **Andy vs Andy** (CO id **1**, `train.py` convention), **fixed tier T3** (Andy appears under T3 on Misery — avoids accidental T1/T2 sampling when the map is fixed).
- **Stage 2 — Geometry generalization:** add **2–5** structurally diverse Std maps (choke vs open, different property density), **still mirror or a small CO set**, to test transfer of property contesting without full CO entropy.
- **Stage 3 — CO generalization:** re-enable stratified / full `co_ids` from enabled tiers (or staged CO buckets) on the Stage 2 map set.
- **Stage 4 — Full pool:** return to “all Std maps / full tier sampling” for **Phase 1 Full Go** metrics.

**Parallel (calendar risk, not a substitute for curriculum):** implement the **turn-level rollout interface** in the engine early (Phase 2 prereq, §4.2). It is testable without a full MCTS loop and de-risks turn-level nodes vs RL sub-steps. It does **not** replace expanding the training distribution before relying on MCTS as main strength.

### Phase 1 Narrow (bootstrap) exit

Use this for **Stage 1–2** only — faster, slice-specific checks. Does **not** replace Phase 1 Full Go for Phase 3, deployment-scale Phase 2, or declaring “Phase 1 complete” globally.

- [ ] Qualitative replay bar from §6 on **that** map/mirror (or small map set).
- [ ] `explained_variance` and win-rate / EV trends **stated as slice-specific** (e.g. Misery + T3 + Andy mirror), explicitly **not** substituted for the global table in §3.2.

### Phase 1 Full Go (all must be met)

This is the **only** gate for declaring Phase 1 complete for **Phase 3** and for **deployment-scale / competitive** Phase 2. Interpret metrics on the **distribution you care about** (multi-map / multi-CO — Stage 3–4), not only a narrow slice.

- [ ] `explained_variance` > 0.4 and stable for 500k+ steps (on full-pool or Stage 3–4 training as applicable)
- [ ] Win rate vs random opponent > 70%
- [ ] Win rate vs checkpoint pool between 52–62%
- [ ] Average game length decreasing over time
- [ ] Property differential positive in > 75% of won games

---

## 4. Phase 2 — MCTS Integration

**Goal:** Add tree search so the agent can look 3–5 full turns ahead rather than
reacting greedily step-by-step.

### What MCTS Adds

MCTS uses the current policy (to prune the action space) and V(s) (to evaluate
leaf nodes) to build a search tree at inference time. Instead of sampling one
action and committing, it simulates K rollouts, aggregates the results, and picks
the action with the highest visit count.

In chess/Go (AlphaZero), each tree node = one full game position after one move.
In AWBW this is complicated by the sequential-move structure. **This is the core
engineering problem of Phase 2.**

### The Sequential-Move Problem and How to Solve It

AWBW turns are broken into 20–30 RL sub-steps. Building a tree over individual
sub-steps would branch at every SELECT/MOVE/ACTION, producing an intractably wide
tree. The solution is to operate the tree at the **turn level**:

1. **Add a turn-level rollout interface to the engine** — a function that takes
   a `GameState` and a complete turn plan (or a rollout policy) and returns the
   `GameState` after the full turn ends, without surfacing each sub-step as a
   separate RL decision.

2. **MCTS tree nodes = full-turn game states** — the tree branches once per
   player turn, not once per unit sub-step.

3. **Leaf evaluation = V(s)** from the current value head, evaluated at the
   post-turn state.

4. **Rollout policy = current policy** — used to sample plausible turn sequences
   quickly during simulation (not full greedy search).

### Weight Transfer for Phase 2

No architectural change required. MCTS wraps the existing model:
- Policy head → action prior (used to focus tree search on high-probability actions)
- Value head → leaf evaluator (used to score positions without playing to terminal)

The Phase 1 weights transfer **entirely and directly** to Phase 2.

MCTS leaf evaluation is **V(s) on the leaf state distribution**. A value head trained only on one layout/mirror can look strong **on that slice** while being **wrong off-manifold**; search then amplifies a biased evaluator. **Order:**

- **MCTS prototype** (small sim budget, one map) may run after **Stage 1–2** once the **turn-level API** exists — useful for plumbing and speed-of-light measurements.
- **MCTS as relied-on strength** (competitive eval, larger sim budget) waits for **Phase 1 Full Go** on the **Stage 3–4** distribution, including **EV > 0.6** on **that** slice (same spirit as the gate below). Prototype MCTS does not satisfy the Phase 2 “production” bar.

### Phase 2 Go Threshold

- [ ] **Phase 1 Full** thresholds met (§3), on the target training/eval distribution — not only narrow bootstrap
- [ ] Turn-level rollout interface implemented and tested in `engine/game.py`
- [ ] Engine can simulate a full turn in < 5ms (required for real-time MCTS)
- [ ] `explained_variance` > 0.6 (V(s) must be a strong evaluator for MCTS to work **on the states you will search**)

---

## 5. Phase 3 — Hierarchical RL (Macro/Micro)

**Goal:** Replace the single reactive policy with a two-level system. A Macro
network plans at the turn level; the Micro network (the current policy) executes
individual unit sub-steps within that plan.

### Architecture

```
Turn start
    ↓
Macro Network
  Input: board features (256-dim from AWBWFeaturesExtractor)
  Output: goal vector G (e.g. target tile per unit, or latent goal embedding)
  Acts: once per player turn
    ↓
Goal vector G
    ↓
Micro Network  (= current policy, goal-conditioned)
  Input: board features (256-dim) + G (concatenated)
  Output: SELECT/MOVE/ACTION sub-steps
  Acts: once per RL step, ~20–30 times per turn
    ↓
Turn ends → Macro receives turn-level reward → updates
```

### Goal Space Design Options

Three approaches, in order of complexity:

**Option A — Target tile assignment (simplest)**
Macro emits one (row, col) target tile per unit. Micro sees its assigned target
as part of its observation and learns to navigate toward it.
- Pros: interpretable, easy to debug
- Cons: Macro must learn which tiles matter without understanding why

**Option B — Latent goal vector (most flexible)**
Macro emits a K-dimensional continuous vector that Micro conditions on.
The goal representation is learned jointly.
- Pros: Macro can represent complex intentions
- Cons: hard to interpret, risk of goal collapse (Micro ignores G)

**Option C — Turn-end board state as goal (most grounded)**
Macro predicts what the board should look like at the end of this turn.
Micro is rewarded for achieving that board state.
- Pros: directly grounded, reward signal for Micro is clear
- Cons: requires predicting a full 30×30×59 board — expensive

**Recommendation:** Start with Option A. Target tile assignment is interpretable,
debuggable, and sufficient to test whether the Macro/Micro split is working.

### Weight Transfer Plan

The current training run produces three sets of weights that transfer directly:

| Weights | Transfers To | Fine-tune? |
|---|---|---|
| `AWBWFeaturesExtractor` | Both Macro and Micro feature extractors | Fine-tune slowly (low LR) |
| Policy head (Micro) | Micro policy head | Fine-tune; add goal concatenation input |
| Value head (step-level V(s)) | Micro critic | Fine-tune |
| — | **Macro network** | **Randomly initialised** |
| — | **Macro critic (turn-level)** | **Randomly initialised** |

Migration procedure:
1. Load trained Phase 1 checkpoint
2. Initialise Micro network from Phase 1 weights (extractor + heads)
3. Add goal input: concatenate goal vector G to the 256-dim feature vector before
   the existing policy/value heads. Re-initialise the first linear layer of each
   head to accept 256+K inputs (with the first 256 columns copied from Phase 1).
4. Initialise Macro network randomly (or with the same extractor, frozen).
5. Train Macro from scratch while fine-tuning Micro with a low learning rate (1e-5).
6. After Macro stabilises, unfreeze extractor and allow joint fine-tuning.

### Training Schedule

Two PPO loops running at different timescales:

```
For each Macro step (one full turn):
    Collect 20–30 Micro steps using current Macro goal G
    Update Micro policy on step-level rewards (property delta, unit value)
    Record turn-level outcome (turn-end board, turn reward)

Every N Macro steps:
    Update Macro policy on turn-level returns
    (terminal reward + turn-end property/unit differential)
```

### Phase 3 Go Threshold

- [ ] Phase 1 **Full** thresholds met (narrow bootstrap exit alone is insufficient)
- [ ] Micro policy demonstrates consistent tactical competence (watch replays — units
      should not suicide, should contest properties, should use terrain)
- [ ] `explained_variance` > 0.7 (V(s) must reliably evaluate positions)
- [ ] Win rate vs checkpoint pool has plateaued for 2M+ steps (architecture ceiling reached)
- [ ] You are prepared for 2–4 weeks of implementation and debugging

---

## 6. What to Watch in Replays (Qualitative Gates)

TensorBoard metrics can be misleading. Periodically run:
```
python -m rl.self_play watch
```
and ask:

**Phase 1 minimum bar:**
- [ ] Agent does not move units off the map or into sea (no-op masking working)
- [ ] Agent ends turns (not stuck in infinite SELECT loops)
- [ ] Agent attacks enemy units when adjacent
- [ ] Agent moves toward uncontested properties

**Phase 2 minimum bar:**
- [ ] Agent positions indirect-fire units (Artillery, Rockets) behind frontline
- [ ] Agent retreats low-HP units rather than leaving them to be destroyed
- [ ] Agent builds appropriate unit types for the map (ground vs naval vs air)

**Phase 3 minimum bar:**
- [ ] Agent executes multi-unit coordinated attacks (two units converge on one target)
- [ ] Agent builds units 4–5 turns before they are needed at the frontline
- [ ] Agent reads when it is winning and plays conservatively vs when losing and plays aggressively

---

## 7. Summary Timeline

| Phase | Entry Condition | Estimated Training Before Gate | Risk |
|---|---|---|---|
| Phase 0 (done) | — | — | Complete |
| Phase 1a (bootstrap) | Curriculum: fixed map/CO/tier + tags | Shorter runs; slice metrics only | Low |
| Phase 1b (expand) | Stage 2–3 ladder | Iterate until transfer looks real | Low–medium |
| Phase 1 Full | Stage 4 / full pool + §3 Full Go | 5–20M steps (days to weeks overnight) | Low |
| Phase 2 prototype (MCTS) | Turn-level API + optional narrow slice | Engineering time; not a strength milestone | Medium |
| Phase 2 production (MCTS) | Phase 1 **Full** + EV > 0.6 **on target mix** | After Phase 1 Full gate | Medium |
| Phase 3 (HRL) | Phase 1 Full + qualitative bar on target mix; EV > 0.7 | After Phase 1 Full (MCTS optional for HRL path) | High |

**Defer Phase 3 HRL** until Phase 1 **Full** gates plus a qualitative replay bar on the **target** map/CO mix; HRL remains high-risk (§5–6).

Phases 2 and 3 are not strictly sequential — MCTS is an inference-time upgrade
(no retraining required), Hierarchical RL is a full architecture change. You could
go Phase 1 Full → Phase 3 directly if the qualitative replay bar is met and you are
comfortable with the implementation risk. **Do not** treat MCTS prototype milestones as Phase 1 completion.

---

## 8. Future track — Fog of war (separate project)

This is **not** on the critical path for Phases 1–3 above, which assume **full observation** on Global League **Std** maps (no tile fog in current training). Treat fog as a **parallel, later programme** once the full-obs stack is proven—or if you explicitly prioritize ranked **Fog** maps as a product goal.

### 8.1 Policy: no shared weights with full-obs training

**Fog and non-fog do not share checkpoints.** Policy and value weights trained under perfect information optimize a different objective than under partial observability; merging training runs or “fine-tune std → fog” as if it were the same model invites gradient conflict and misleading value metrics. Plan for **separate training runs, separate checkpoints, and a separate eval ladder** for fog. Success on Phase 1 Full (std, full obs) **does not** certify a fog agent.

*(Optional later experiment: initialize fog training with random weights only, or small controlled studies with frozen low-level features—still **not** “the same weights” as production std.)*

### 8.2 What this track must include (engineering summary)

- **Engine:** Tile fog, vision from units and properties, terrain/weather per AWBW rules, and **transit visibility**—enemy moves visible tile-by-tile when any segment of the path enters your vision (humans track units through chokes between fog pockets).
- **Observations and masks:** Player-local tensors; `get_legal_actions` / masks must match what a human may do from that information set, not the true state. Opponent policies must receive **their** view, not P0’s oracle board.
- **POMDP:** A feedforward CNN on one snapshot is insufficient for corridor intel and last-known state unless a **belief state** is updated every relevant engine step (including **each opponent micro-step** during their turn—today’s env batches P1’s whole turn inside one P0 `step`, which can erase visibility events unless the contract emits traces or sub-steps).
- **Metrics:** Separate TensorBoard / `game_log` tags for fog; reinterpret or replace `explained_variance` if the critic sees hidden state (privileged critic) vs the policy.
- **Phase 2 / 3 (if fog ever gets search or HRL):** MCTS needs belief / information-set style search, not leaf `V(s)` on the true board; macro/micro inputs must be belief-consistent.

### 8.3 Relationship to the dependency stack

Fog does **not** sit under Phase 1 Full as a hidden dependency. It is a **second product line**: duplicate investment in engine correctness, encoding, env stepping, training, and gates—only after you choose to fund it.

---

## 9. P0-only training and seat / tempo asymmetry

**Priority (program sense): Tier 2 —** fits **Phase 1b** (distribution + curriculum + eval), **not** Phase 0 or narrow Phase 1a bootstrap. It does **not** block Stage 1 Misery mirror or turn-level API plumbing. It **does** belong on the path to **Phase 1 Full**: treat “agent only ever acts as engine **player 0**” as a known inductive bias and **measure** whether seat / opening order hurts you before claiming robust cross-context strength.

**Why it matters:** [`rl/env.py`](rl/env.py) documents that the **policy always controls player 0**; [`docs/player_seats.md`](docs/player_seats.md) ties that to human `/play/` and red/blue seats. [`rl/encoder.py`](rl/encoder.py) uses **fixed** P0 vs P1 channel blocks (not an ego-centric “always me” frame). The policy **never** selects actions on P1’s clock during training—so it does not learn **P1-style initiative** with the same action head, even though many decisions are **post-opponent** (reactive “second” timing on each P0 step). Geography also differs by [`p0_country_id`](data/gl_map_pool.json) vs “being blue on the same map.”

**What still works without code changes:** reactive play when `active_player` returns to P0 after P1’s micro-steps; a stable “I am always red in the tensor” mapping avoids label-switching bugs; stronger opponents and broader maps improve tempo defence without swapping seats.

### 9.1 Ordered mitigations (ROI vs effort)

| Order | Mitigation | Effort | When |
|------:|------------|--------|------|
| 1 | **Measure:** slice `game_log.jsonl` (or scripted seeds) by **opening player** (P0 first vs P1 first from `make_initial_state` rules) and report win rate / EV / game length | Low | As soon as Stage 2+ volume exists — **do this first** |
| 2 | **Curriculum:** bias sampling toward episodes where **P1 opens** (asymmetric predeploy maps, or controlled seeds) so P0’s first decision is “second on the clock” more often | Low–medium | Phase 1b, alongside map/CO distribution expansion |
| 3 | **Pool / faction exposure:** vary or randomize `p0_country_id` **across episodes** so the same policy sees different geography as P0 (still not P1 actions) | Medium | If metrics show overfitting to one corner / faction seat |
| 4 | **Ego-centric encoder** (swap P0/P1 channels + scalars so “me” is always one block; engine still 0/1) or **dual-role** training | High | Only if (1)–(3) show a **systematic** first-move / seat failure |

**Rule of thumb:** (1) and (2) are **default** Phase 1b hygiene. Pursue (3)–(4) only after metrics justify the engineering and checkpoint contract cost.

**Training doctrine once Tier 4 is funded (ego-centric “me” frame):**

- **Tier 4 is the prerequisite** for every “both sides” or “alternate seats” path below. Without ego-centric encoding, alternating the learner’s engine seat or logging opponent-side rows feeds P1-oriented state through a P0-oriented action head and teaches the wrong action bijection.
- **BC (supervised):** once the encoder is ego-centric, log and train on **both** seats’ human turns from the same games. BC has no on-policy constraint; this is the highest-ROI use of “both sides.”
- **PPO actor (on-policy):** alternate which engine seat the **learner** controls across parallel envs (e.g. half of [`rl/self_play.py`](rl/self_play.py) workers: learner P0 vs pool P1; half: learner P1 vs pool P0). Do **not** run policy gradients on actions taken by the **frozen pool** opponent — those trajectories are off-policy and are not correctable under PPO’s clipped surrogate without a different algorithm.
- **PPO critic:** the value head is regression, not PG. Train it on **both** seats’ states with **sign-flipped** returns under the zero-sum assumption (watch first-move / tempo asymmetry vs a strict −V symmetry).
- **Anti-pattern:** AlphaZero-style **pure** self-play (same weights on both sides, no pool) as the **primary** training mode. AlphaZero leans on MCTS as the regularizer; PPO without search tends toward exploit cycles and collapse when the only opponent is itself.
- **Correlation caveat:** “Log both sides of the **same** game” is not independent 2× data for the **actor** — shared seed, shared history, and GAE on correlated trajectories inflate batch size more than information. Prefer seat alternation across **independent** episodes over stuffing both seats from one game into the actor batch.

### 9.2 Checklist (optional Phase 1b / Full prep)

- [ ] `opening_player` field present in `game_log.jsonl` rows you analyze (`log_schema_version` ≥ **1.5**; see [`docs/seat_measurement.md`](docs/seat_measurement.md))
- [ ] Win rate (or EV) **conditional on opening player** logged or dashboarded (Tier 1 measurement report on ≥ Stage 2 volume)
- [ ] Curriculum includes a **meaningful fraction** of P1-opens games where the engine allows it
- [ ] If gate fails: document whether remediation is (3) pool remap or (4) representation change before declaring Phase 1 Full for “fair seat” claims
- [ ] **If Tier 4 is funded:** BC row format includes a `me ∈ {0, 1}` (or equivalent) flag and the encoder supports ego-centric channel swap; parallel envs are **seat-balanced** for the learner; critic targets use sign-flip on opponent-seat rows where applicable

---

*"He who fails to plan is planning to fail."*
*— Winston Churchill, British Prime Minister during World War II.*
