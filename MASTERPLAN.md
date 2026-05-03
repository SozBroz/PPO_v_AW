# AWBW-RL Master Plan

*Last updated: 2026-04-24*

This document is the strategic north star for the AWBW reinforcement learning project.
It records where we are, where we are going, and — critically — the concrete thresholds
that should gate each phase transition.

---

## 1. Where We Are Now

### Architecture (restart bundle — 2026-04)

| Component | Status | Detail |
|---|---|---|
| Encoder | **Active** | 70 spatial ch (ego-centric me/enemy + influence + defense stars); 17 scalars (`docs/restart_arch/MASTER_SPEC.md`) |
| Feature extractor | **Active** | `AWBWFeaturesExtractor` — stem 70→128, **10×** ResBlock@128, scalar→16 broadcast, GAP 144 → 256 MLP (matches `AWBWNet` trunk) |
| Policy | **Active** | Factored spatial head (`Conv2d` 1×1 on 144ch) + scatter to 35k flat; MOVE band 1818..2717; collision 900..902 |
| Value head | **Active** | GAP on 144 fused ch → MLP → scalar V(s) |
| Opponent | **Active** | `_CheckpointOpponent` + optional PFSP (`AWBW_PFSP`, `AWBW_PFSP_STATS`); optional `AsyncVectorEnv` (`AWBW_ASYNC_VEC`) |
| Reward | **Dual path** | **Default (`phi`):** learner-frame Φ-delta + terminal ±1.0 (sign-aware vs seat). **Fallback (`level`):** me−enemy property/value *levels*. `AWBW_REWARD_SHAPING` in `rl/env.py`. Seat balance: `AWBW_SEAT_BALANCE` / `AWBW_LEARNER_SEAT`. |
| Training | **Active** | MaskablePPO; defaults unchanged in `train.py` unless overridden — use **`--stage1-narrow`** for Phase 1a Misery preset |

**Pre-restart checkpoints** (`latest.zip` / dense Linear head / 62–63ch stem): kept on disk for BC experiments and regression baselines; **not** loadable into the new policy stem without a transplant. Prefer scratch or BC-init zips matched to the 70ch + factored head contract.

### What the Current Run Is Building

The network is learning three things simultaneously:

1. **Board reading** (`AWBWFeaturesExtractor` weights) — how to compress a 30×30×70 spatial
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

**Parallel (calendar risk, not a substitute for curriculum):** the **turn-level rollout interface** is in place (Phase 2 prereq, §4.2): `GameState.apply_full_turn` in `engine/game.py`, covered by `tests/test_apply_full_turn.py`, used as the rollout primitive in `rl/mcts.py`. It is testable without a full MCTS loop and de-risks turn-level nodes vs RL sub-steps. It does **not** replace expanding the training distribution before relying on MCTS as main strength.

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

1. **Turn-level rollout interface in the engine** — **done (Phase 11a):** `GameState.apply_full_turn` takes a `GameState` and a complete turn plan (`list[Action]`) or a rollout policy (`Callable[[GameState], Action]`) and returns the post-turn `GameState` (plus trace metadata), without requiring each sub-step to be a separate RL decision at the API boundary. See `engine/game.py`, `tests/test_apply_full_turn.py`, `rl/mcts.py`.

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

- **MCTS prototype** (small sim budget, one map) may run after **Stage 1–2** — the **turn-level API** (`apply_full_turn`) is available — useful for plumbing and speed-of-light measurements.
- **MCTS as relied-on strength** (competitive eval, larger sim budget) waits for **Phase 1 Full Go** on the **Stage 3–4** distribution, including **EV > 0.6** on **that** slice (same spirit as the gate below). Prototype MCTS does not satisfy the Phase 2 “production” bar.

### Phase 2 Go Threshold

- [ ] **Phase 1 Full** thresholds met (§3), on the target training/eval distribution — not only narrow bootstrap
- [x] Turn-level rollout interface implemented and tested in `engine/game.py` (`GameState.apply_full_turn`; `tests/test_apply_full_turn.py`)
- [ ] Engine can simulate a full turn in < 5ms (required for real-time MCTS)
- [ ] `explained_variance` > 0.6 (V(s) must be a strong evaluator for MCTS to work **on the states you will search**)

### Production MCTS (AWBW live / competitive play)

AWBW is **turn-based**, not RTS: the binding constraint for a **production** agent is usually **wall-clock per decision**, not a fixed iteration count inside a training loop. Branching is still combinatorial (many units × many legal sub-steps per turn), so **efficiency** matters even when you can search for tens of seconds.

**Anytime search:** In production, prefer a **time budget per P0 turn** (e.g. 30–60s, tunable by match rules and hardware) and run **best-effort MCTS until the budget expires**, rather than a single hard-coded sim count. Fleet **training / symmetric eval** may keep **fixed sim counts** for reproducibility until a time-based mode is validated. Scaling rule: **more compute generally helps** if the value head is accurate on the positions you search; doubling search time often yields measurable Elo gains **until** bias or diminishing returns dominate.

**Move / turn narrowing:** Do **not** expand raw full legal sub-step trees. Use the **policy** to concentrate search on high-prior lines (in-repo: turn-level children sampled via policy rollouts and `root_plans` / priors in `rl/mcts.py`). Literature and practice for large action spaces: cap expansion to **top-N policy mass** per decision point (analogous to “top 10–20” ideas **per branching point** — our tree already aggregates a **whole turn** into one edge; widen with **more distinct turn samples** and **progressive widening**, not exhaustive legal enumeration).

**Breadth vs depth:** **Breadth** (many plausible **variations of the current turn** and immediate replies) is usually the bottleneck in AWBW-style combinatorial turns. **Depth** helps but **diminishing returns** set in after roughly **10–12 player-turns** of lookahead in uncertain positions; extra wall time often buys more from **wider** search at tactical depths than from chasing very deep speculative lines. This aligns with **turn-level** nodes: each node already collapses one player’s full turn; “go wider” = more / smarter **turn-plan children**, not micro-step explosion.

**Training vs production (intent):**

| Aspect | Training / fleet eval (typical) | Production (standard play) |
|--------|----------------------------------|------------------------------|
| Gating | Fixed sim counts, reproducible seeds | **Clock time** (anytime algorithm) |
| Budget anchor | Hundreds–low thousands of sims / root for sweeps | **Seconds per turn**; may imply **10k–100k+** rollouts on strong hardware if each sim is cheap enough |
| Exploration | Dirichlet / temperature on policy or visits | **Exploitation-forward**: low noise, **deterministic** final pick from visit counts (or very low temperature) |
| Reference point | AlphaZero-class agents often cite **~1.6k sims** in **sub-second** chess settings; AWBW turn-level sims are **not** 1:1 comparable — expect **higher** effective rollouts for similar strength if rollouts are heavier. |

**Search-control heuristic (optional but valuable):** Spend **more wall time** on turns with **high unit density / contested frontlines**; spend **less** on phases where tactics are narrower (e.g. early property rhythm) **if** a cheap complexity signal agrees with telemetry. Implement as **multiplier on time budget** or **dynamic root K**, not silent behavior change without logging.

**Plan pointer:** Operational details, profiling order, and todos live in [`.cursor/plans/mcts_optimization_campaign.plan.md`](.cursor/plans/mcts_optimization_campaign.plan.md).

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
    Update Micro policy on step-level rewards (legacy: property + unit-value level terms; under **phi** mode: Φ-delta shaping from `rl/env.py`)
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

## 8. Future track — Fog, high funds, and ladder expansion (separate project)

This is **not** on the critical path for Phases 1–3 above, which assume **full observation** on Global League **Std** maps and “normal” economy defaults in training. Treat **fog of war** and **high-funds (HF) ranked** ladder play as a **parallel, later programme** once the full-obs / standard-economy stack is proven—or if you explicitly prioritize those **ranked** rulesets as a product goal.

### 8.1 Policy: no shared weights with full-obs training

**Fog and non-fog do not share checkpoints** as a single undifferentiated artifact. Policy and value weights trained under perfect information optimize a different objective than under partial observability; naively merging training runs or “fine-tune std → fog” **without** mode separation, masking, and metrics discipline invites gradient conflict and misleading value metrics. Plan for **separate training runs, separate checkpoints, and a separate eval ladder** per major ruleset (e.g. fog, high funds). Success on Phase 1 Full (std, full obs) **does not** certify a fog or HF agent.

**Transfer vs identity:** Structured transfer (§8.7–8.9)—frozen backbone, multi-task heads, or progressive columns—can **initialize** from a strong Standard network, but the **shipped** fog/HF policy should still be **versioned and evaluated** on its own ladder. “We froze the trunk” is not the same as “one checkpoint serves every mode in production” unless you deliberately commit to a unified MTL model (§8.8) and prove it does not regress Standard.

*(Optional later experiment: initialize specialized tracks from random weights only, for ablation—contrast with §8.7–8.9.)*

### 8.2 What this track must include (engineering summary)

- **Engine:** Tile fog, vision from units and properties, terrain/weather per AWBW rules, and **transit visibility**—enemy moves visible tile-by-tile when any segment of the path enters your vision (humans track units through chokes between fog pockets).
- **Observations and masks:** Player-local tensors; `get_legal_actions` / masks must match what a human may do from that information set, not the true state. Opponent policies must receive **their** view, not P0’s oracle board.
- **POMDP:** A feedforward CNN on one snapshot is insufficient for corridor intel and last-known state unless a **belief state** is updated every relevant engine step (including **each opponent micro-step** during their turn—today’s env batches P1’s whole turn inside one P0 `step`, which can erase visibility events unless the contract emits traces or sub-steps).
- **Metrics:** Separate TensorBoard / `game_log` tags for fog; reinterpret or replace `explained_variance` if the critic sees hidden state (privileged critic) vs the policy.
- **Phase 2 / 3 (if fog ever gets search or HRL):** MCTS needs belief / information-set style search, not leaf `V(s)` on the true board; macro/micro inputs must be belief-consistent.

### 8.3 The “Waypoints” hierarchical approach (HRL for FOG pathing)

Instead of having the RL agent pick **every** micro-step, plan for a **hierarchical** setup aligned with §5 (macro/micro), but with a dedicated pathing role:

- **High-level actor** — Chooses the **target destination** and the **intent** (e.g. attack, capture, wait). This is the “where and why,” not the full tile-by-tile path.
- **Low-level “pathing” head** — A smaller sub-network (or a specialized layer) that **only needs to be fully engaged under FOG**. It takes **(start, end)** coordinates (from the high level) and outputs either a **pathing mask** over reachable tiles or a **sequence of direction tokens** that realize the move.

**Why it helps:** In **Standard** (full-obs) mode, the pathing head can be **bypassed** in favour of a deterministic planner (e.g. A* shortest-path search to the destination). In **FOG**, the model can learn that a **curved route through woods** is higher value than a **straight line on a road** when exposure matters—behaviour that is awkward to get from a single flat action index per step with no structure.

This composes with §5’s macro/micro split: fog-specific training mainly stresses **consistency** between declared goals and **pathing under partial vision**, not a second unrelated policy.

### 8.4 Action masking and pointer / tile attention

**Pointer-style outputs (optional but strong fit for long moves):** Instead of a single monolithic action index, consider a **pointer network** or **transformer attention** that **“points to” a sequence of tiles** (or a compact representation of the path) so credit assignment for multi-tile movement is not fighting the 35k flat head alone.

- **Strict action masking (non-negotiable):** The legal set must be **player-local and engine-true**. If a unit has e.g. **6 movement**, the mask only allows destination tiles (or path segments) **within that reach**—no oracle leaks through extra logits.
- **Bounded path hypotheses:** For the pathing sub-problem, **do not** enumerate every permutation of N steps. **Discretize** to a small menu of **2–3** viable path classes (e.g. **“most direct”** vs **“most hidden / off-road”**), selected by the pathing head or by a post-process on top-`k` A* / BFS candidates. That keeps the **pointer / sequence** head trainable and interpretable.

MaskablePPO already assumes masks; a fog build must extend that contract to **fog-legal** actions and any new path head.

### 8.5 FOG as a “vision channel” (one network, two regimes)

To keep a **single** policy–value network (separate **checkpoints** per §8.1, but one **architecture**), treat **field-of-view** as an explicit **input channel** (e.g. a **bitmask** or multi-bit encoding aligned with the spatial grid):

- **Standard / full-obs** — The vision channel is **all ones** (or equivalent: “fully seen”).
- **FOG** — The channel is a **mix of** visible tiles, **unseen** (e.g. zeros), and **last-seen / memory** where rules permit (ties to the **belief / POMDP** bullet in §8.2).

**Joint or staged training:** If the policy is trained on **both** regimes (or curriculum from full → mixed), it can learn that when the **vision channel** is sparse (many zeros), the **pathing head** and tactical choices should be **more conservative**—without a hard-coded `if fog:` branch in code.

### 8.6 Relationship to the dependency stack

Fog and HF ladder expansion do **not** sit under Phase 1 Full as a hidden dependency. They are a **second product line**: duplicate investment in engine correctness, encoding, env stepping, training, and gates—only after you choose to fund them.

### 8.6a Std meta vs high-funds tech — Stealth bans and submarine rarity

**Known shortcoming:** Phase 1–3 training on Global League **Std** skews away from several **high-funds-relevant** unit lines:

- **Stealth** is **banned on many Std maps** (`unit_bans` in `data/gl_map_pool.json`). The policy therefore gets **little training signal** on hide/unhide, fuel curves while hidden, and **Fighter/Stealth-only** targeting rules—even though the engine implements them.
- **Submarines** are legal on many Std maps but **rarely built in human Std meta**; self-play and replay-derived behaviour may **under-sample** dive/surface, **submerged** interaction (only Cruiser and Sub may attack), and naval scouting that matters more when economies go long.

**High-funds (HF) ranked** and **high-treasury endgames** push toward tech (Stealth on allowed maps, subs, heavier naval/air). **Do not assume** Std-trained competence transfers to that regime without **HF-oriented curriculum, eval, and metrics** (see §8.1, §8.7–8.8). Separately, the encoder’s **14-way unit-type bucket** collapses many late tech types (`rl/encoder.py`); that tightens the case for **mode-specific** training or encoder upgrades when HF or rare units become a product target.

### 8.7 Frozen backbone (Standard feature extraction → FOG / HF heads)

A **superhuman Standard** (full-obs) bot’s **early and middle layers** have already learned a reusable “language” of AWBW: base placement, threat geometry, terrain value, and unit affordances. That representation is a natural **backbone** for new rulesets if you do not want to re-learn vision from scratch.

- **The move** — Load the Standard network and **freeze** the backbone (conv / ResNet blocks and shared stem): **no gradients** into those layers during the new training phase.
- **The addition** — Attach **new, randomly initialised** modules: task-specific **policy / value heads**, and where needed (§8.3) a **FOG pathing** head or, for **high funds**, an **HF-macro** head (economy tempo, tech pacing, and scale that differ from low-income training).
- **Transfer** — During FOG or HF training, **only the new parameters update**. The frozen backbone should **not** “forget” how Standard worked; the new heads learn to **reinterpret** the same spatial features under fog masks, partial observability, or different fund curves.

**Trade-off:** Simpler and stable for the backbone, but the new heads may be **underpowered** if the frozen features lack channels that only matter under fog (e.g. explicit belief memory)—you may need **thin adapter layers** (trained) between frozen trunk and new heads, or unfreeze the trunk later with a **very small** LR (see §8.8).

### 8.8 Multi-task learning (unified model across Standard, FOG, HF)

If the goal is a **single deployable model** that can play **multiple ladder modes**, train one **shared backbone** with **task-specific heads** (and/or task-specific small towers), instead of a permanently frozen std-only network.

- **The setup** — **Shared** feature extractor, then **separate** heads for Standard vs FOG vs high funds (or a head for “pathing / macro” that only receives gradients in the relevant mode).
- **The transition** — **Initialise** the shared backbone from your strongest **Standard** weights; initialise new heads from scratch (or from small pre-training).
- **Mode awareness** — Add a **one-hot** or **learned embedding** “game mode” vector **concatenated to** the flat feature vector (or added as bias to heads). The network then knows whether the episode is `standard`, `fog`, or `high_funds` (or GL ruleset id).
- **The result** — When `mode=standard`, the policy can **retain** behaviours that already work; when `mode=fog`, it **routes** spatial understanding through FOG-specific heads and the vision channel (§8.5). **Risk:** **interference** between modes is real—use **sampling** across modes per batch, **gradient balancing**, or **loss weighting** so HF or fog does not wash out Standard. Validate **each** mode on **its** ladder, not only aggregate loss.

This is the natural counterpart to a permanently frozen backbone (§8.7): more **unified capacity**, more **governance** (metrics, ablations, regression on Standard when training fog/HF).

### 8.9 Progressive neural networks (lateral transfer without overwriting Standard)

If the concern is **catastrophic forgetting**—FOG or HF training **degrading** Standard play even with care—a **ProgressiveNN**-style design keeps a **dedicated column** for the original task and **adds** capacity for the new one.

- **Column 1** — The **Standard** superhuman network, **frozen** (or updated only on Standard batches if using MTL in parallel).
- **Column 2+** — **New** weight columns for FOG, and optionally a third for **high funds** macro.
- **Lateral connections** — Later columns take **not only** the current input but also **activations** (or pre-specified feature slices) from earlier columns, so the new task can **reuse** “where units are and what matters” without **overwriting** Column 1’s weights.

**Use when:** you need **strong** reuse of Standard competence and a **provable** separation of trainable parameters for the new mode. **Cost:** more parameters, more engineering to wire lateral features and keep inference latency acceptable on the training stack.

---

## 9. P0-only training and seat / tempo asymmetry

**Priority (program sense): Tier 2 —** fits **Phase 1b** (distribution + curriculum + eval), **not** Phase 0 or narrow Phase 1a bootstrap. It does **not** block Stage 1 Misery mirror or turn-level API plumbing. It **does** belong on the path to **Phase 1 Full**: treat “agent only ever acts as engine **player 0**” as a known inductive bias and **measure** whether seat / opening order hurts you before claiming robust cross-context strength.

**Why it matters (historical vs restart):** Before the restart, the env trained **P0-only** with fixed P0/P1 tensor blocks. **Now:** [`rl/encoder.py`](rl/encoder.py) is **ego-centric** (me/enemy vs `observer`); optional **`AWBW_SEAT_BALANCE`** lets the learner act on either engine seat with the same flat action head. [`docs/player_seats.md`](docs/player_seats.md) still ties human `/play/` to red/blue. Geography still varies by [`p0_country_id`](data/gl_map_pool.json).

**What still works without code changes:** reactive play when `active_player` returns to P0 after P1’s micro-steps; a stable “I am always red in the tensor” mapping avoids label-switching bugs; stronger opponents and broader maps improve tempo defence without swapping seats.

### 9.1 Ordered mitigations (ROI vs effort)

| Order | Mitigation | Effort | When |
|------:|------------|--------|------|
| 1 | **Measure:** slice `game_log.jsonl` (or scripted seeds) by **opening player** (P0 first vs P1 first from `make_initial_state` rules) and report win rate / EV / game length | Low | As soon as Stage 2+ volume exists — **do this first** |
| 2 | **Curriculum:** bias sampling toward episodes where **P1 opens** (asymmetric predeploy maps, or controlled seeds) so P0’s first decision is “second on the clock” more often | Low–medium | Phase 1b, alongside map/CO distribution expansion |
| 3 | **Pool / faction exposure:** vary or randomize `p0_country_id` **across episodes** so the same policy sees different geography as P0 (still not P1 actions) | Medium | If metrics show overfitting to one corner / faction seat |
| 4 | **Ego-centric encoder** (swap P0/P1 channels + scalars so “me” is always one block; engine still 0/1) or **dual-role** training | High | Only if (1)–(3) show a **systematic** first-move / seat failure |

**Rule of thumb:** (1) and (2) are **default** Phase 1b hygiene. Pursue (3)–(4) only after metrics justify the engineering and checkpoint contract cost.

**Training doctrine — Tier 4 (ego-centric “me” frame) — funded in-repo (2026-04 restart):**

- Encoder + obs paths use **me/enemy** vs `observer`; enable **seat-balanced** rollouts with `AWBW_SEAT_BALANCE=1`. Human-vs-bot play uses the correct observer per seat in `server/play_human.py`.
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
- [x] **Tier 4 (encoder + logs):** ego-centric channel swap in `rl/encoder.py`; `learner_seat` / `reward_mode` in `game_log` (schema ≥1.9). **Seat-balanced envs** (`AWBW_SEAT_BALANCE`); BC row flag / critic sign-flip policy still tracked per run.

---

## 10. Multi-PC training (strategy + backlog)

**Canonical plan:** *Multi-machine checkpoint sync* — Cursor plan file `multi-machine_checkpoint_sync_32b867d3.plan.md` (async **weight** sync on a shared directory—not shared PPO rollout buffers).

### 10.1 Doctrine (second PC)

- **Primary ROI:** Two (or N) machines run `train.py` against a shared `AWBW_CHECKPOINT_DIR`; each process keeps **on-policy PPO** on **local** rollouts; they **publish / reload** `latest.zip` so the fleet shares one policy line and a richer `checkpoint_*.zip` opponent pool.
- **Do not:** Feed another machine’s trajectories into PPO as if on-policy; use **shared weights** or a **separate BC / offline RL** pass ([`scripts/train_bc.py`](scripts/train_bc.py), docs in [`docs/play_ui.md`](docs/play_ui.md)).
- **Do not:** “Shared pagefile” or mmap-as-RAM across PCs for weights—wrong layer, no coherency.
- **Checkpoint contract:** Same repo revision / encoder on all machines; see §8 for fog vs std separation.

### 10.1b Episode bounds and logging (train.py / `AWBWEnv`)

Long-run training assumes episodes eventually finish so PPO sees `dones`, TensorBoard episode diagnostics move, and `logs/game_log.jsonl` receives rows. A pathological policy can otherwise avoid `END_TURN` forever when caps are unset. **`train.py` now defaults `--max-env-steps 8000` and `--max-p1-microsteps 4000`** (disable with `0` or negative — not recommended for unattended runs). The env truncates when hit; **`game_log.jsonl` records both natural and truncated ends** (`log_schema_version` **1.8**: `terminated`, `truncated`, `truncation_reason` of `max_env_steps` / `max_p1_microsteps` / `null`). `scripts/start_solo_training.py` and `fleet_orchestrator.build_train_argv_from_proposed_args` always emit these flags so fleet restarts do not drop them.

### 10.2 Phase 1 — implementation backlog (code)

*Implement in Agent mode; paths reference repo root.*

| Deliverable | Detail |
|-------------|--------|
| [`rl/paths.py`](rl/paths.py) | `get_checkpoint_dir()`, `get_data_dir()`, `get_pool_path()`, `get_maps_dir()`, `get_game_log_path()`, `get_slow_games_log_path()` from `AWBW_CHECKPOINT_DIR` / `AWBW_DATA_DIR`. |
| [`rl/shared_training.py`](rl/shared_training.py) | `training_sync.sqlite`: `BEGIN IMMEDIATE` monotonic version; `publish_latest()` writes temp zip → `os.replace` → `latest.zip`, then manifest (`sha256`, `size`, `version`, `worker_id`, `num_timesteps`); `reload_model_if_newer()` validates then `MaskablePPO.load`; `prune_worker_snapshots(max_files)`; `sanitize_worker_id()`. |
| [`train.py`](train.py) | After `parse_args`, map `--checkpoint-dir` / `--data-dir` → `os.environ` **before** importing `rl.self_play`. Flags: `--shared-training`, `--worker-id`, `--publish-every` (default 1), `--checkpoint-retain` (0 = off). |
| [`rl/self_play.py`](rl/self_play.py) | Thread `checkpoint_dir`, `shared_training`, `worker_id`, `publish_every`, `checkpoint_retain`; opponent `refresh_every` lower when shared (e.g. 200); snapshot names `checkpoint_{worker}_{seq}.zip` when shared else legacy `checkpoint_{NNNN}.zip`; loop: reload-if-newer → `learn` → snapshot → conditional `publish_latest`; optional `shared_training.publish` on SIGINT. |
| [`rl/env.py`](rl/env.py) | Use `rl.paths` for game log, slow games, pool, maps (import after env-safe ordering—`paths` must not import `env`). |
| Consumers | [`scripts/eval_imitation.py`](scripts/eval_imitation.py), [`analysis/co_ranker.py`](analysis/co_ranker.py), [`analysis/co_h2h.py`](analysis/co_h2h.py): resolve log path via `get_game_log_path()` or env-aware helper. |
| Docs | [`README.md`](README.md) or [`docs/play_ui.md`](docs/play_ui.md): second-PC env vars, SMB tuning (`--publish-every`, `--save-every`), retention, subprocess opponent lag. |

### 10.3 Phase 2 — deferred (not required for basic two-PC sync)

- **BoN / Bo11 promotion:** alternating-seat series; promote winner to shared `latest` (wrap or extend [`scripts/bo3_checkpoint_playoff.py`](scripts/bo3_checkpoint_playoff.py) / [`scripts/symmetric_checkpoint_eval.py`](scripts/symmetric_checkpoint_eval.py)); high variance—document minimum games before automation.
- **`best.zip`:** promote only when periodic eval beats a baseline threshold (stops `latest` being purely last-writer in async mode).
- **Hostname / `curriculum_tag` in `game_log`:** when merging logs from multiple PCs for analysis.

### 10.4 Phase 3 — research (only if Phase 1–2 plateau)

- **Canonical symmetric / ego-centric observations** so one net can cover both seats without dual-policy training (ties to §9 Tier 4).
- **Opponent sampling policy** driven by measured win-rate band vs pool (MASTERPLAN healthy band ~52–62% vs checkpoints).
- **Ray / RLlib** or other true distributed PPO if disk-async remains insufficient.

---

## 11. Reward shaping — Φ rollout and validation (future work)

**Context:** Legacy dense rewards used **levels** (property count and army value) every P0 step, so cumulative shaping scaled with episode length and could **drown terminal ±1.0**; kills could read as negative on average because the value tax persisted. **Potential-based shaping** (`AWBW_REWARD_SHAPING=phi`) replaces that with `F(s,s') = Φ(s') − Φ(s)` plus a contested-capture term that **auto-refunds** chip progress when the engine resets `capture_points` (capturer dies / vacates). Implementation: `rl/env.py`, gated engine capture in `engine/game.py`, tests in `tests/test_env_shaping.py`, smoke tool `tools/phi_smoke.py`.

### 11.1 Training bootstrap (must-do before distributed workers)

`engine/game.py` reads `_PHI_SHAPING_ACTIVE` **once at import**. Set shaping mode **before** any import of `engine.game` (e.g. at the top of `train.py` after optional `.env` load, or in the shell that launches training):

- `AWBW_REWARD_SHAPING=phi` — enable Φ path + suppress engine `_CAPTURE_*` shaping.
- Optional tuning (defaults `2e-5` / `0.05` / `0.05`): `AWBW_PHI_ALPHA`, `AWBW_PHI_BETA`, `AWBW_PHI_KAPPA`.

SubprocVecEnv workers inherit env at spawn; changing mid-run is undefined — **restart the run** to switch modes or coefficients.

### 11.2 Pre-PPO validation (cheap, no GPU)

Run [`tools/phi_smoke.py`](tools/phi_smoke.py) on a fixed curriculum slice (e.g. Misery T3 mirror) to compare **phi vs level** on the same policy (greedy×greedy is a pessimistic stalemate stress test):

- **Trajectory shaping** `|ΣF|` should stay **≪** legacy level (order-of-magnitude gap is the health signal).
- **Per-step peak** `|F|`: defaults often land ~0.15–0.21 on rare large material swings; **halving** α/β/κ (~0.05–0.10 peak) if you want shaping strictly under ~20% of terminal per step.
- **Cap vs kill** (mean |shaping| on CAPTURE vs KILL steps): should stay **same order of magnitude** (no “revenge table”; favorable trades show as positive material Φ).

Document the command line and seed in the run notes so regressions are comparable.

### 11.3 PPO scratch smoke (after env validation)

- Short run (e.g. 50k–500k steps, small `n_envs` / `n_steps` acceptable) with **`AWBW_REWARD_SHAPING=phi`**, same curriculum tag as a recent **level** baseline for apples-to-apples.
- **Re-baseline TensorBoard:** `explained_variance` and value loss will shift vs level-trained checkpoints — do **not** interpret old thresholds literally until a new plateau forms.
- Watch **advantage / return variance**: if dominated by Φ spikes, prefer **halving** coefficients (§11.2) before architectural changes.
- **Checkpoint policy:** resuming a **level**-trained `latest.zip` under **phi** is allowed for smoke but expect **value-head drift**; a **scratch** run from random or BC init is the honest read for the new return distribution.

### 11.4 Default flip

- [x] Default shaping is **`phi`** in `rl/env.py` (`AWBW_REWARD_SHAPING` unset). Legacy **`level`** remains for ablations.
- [x] §1 table updated for phi-default + learner frame. The legacy level path stays until no run depends on it.

### 11.5 MASTERPLAN metric hygiene under phi

- Slice **win rate / game length / captures** by `curriculum_tag`, **`reward_mode`**, **`learner_seat`**, and `log_schema_version` (≥1.9) in `game_log.jsonl`. Use `tools/slice_game_log.py`.
- Phase 1 **Full Go** gates (§3) still apply; reward switch does **not** replace distribution or replay qualitative bars.

---

## 12. Native compilation & low-level perf (ACTIVE — NN bottleneck shifted)

**Status:** Active. Re-opened 2026-04-23 after Claude Code performance analysis identified that the restart architecture's larger NN (70ch encoder, 10× ResBlock@128, 256-dim trunk, 35k output head) has shifted the bottleneck profile from the cold-opponent baseline.

**Context shift:** The original §12 analysis was based on cold-opponent runs where inference cost was negligible. The current architecture (§1) includes:
- 70 spatial channels (vs ~59 before restart)
- 10× ResBlock@128 in AWBWFeaturesExtractor
- 256-dim trunk → 35k-dim policy head (8.96M params in a single Linear)
- Factored spatial head (already implemented) + MOVE band 1818..2717

This changes the cost profile: what was 77% non-engine overhead with a small NN now has a significant inference component.

### 12.0 Profile Before Fixing (MANDATORY first step)

The bottleneck location is now unknown. Run this to determine which fixes actually matter:

```python
# Add to train.py or self_play.py temporarily
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, with_stack=True) as prof:
    # Run ~100 steps
    trainer.model.learn(total_timesteps=100 * args.n_steps * args.n_envs)

prof.export_chrome_trace("perf_trace.json")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

This tells you whether the bottleneck is **inference** (forward pass), **PPO update** (backward pass), or **IPC** (env communication) — which determines which fixes below actually matter.

### 12.1 Tier 1: Free Wins (1–2 hours, no architecture risk)

| # | Fix | Expected Impact | Risk |
|---|-----|-----------------|------|
| 1a | `torch.compile(model.policy, mode="reduce-overhead")` | 20–40% speedup on forward pass | Low — fullgraph=False safe with MaskablePPO hooks |
| 1b | Mixed precision inference (`torch.amp.autocast` in forward) | 30–50% faster on modern GPUs for conv layers | Low — value/policy heads handled by SB3 |
| 1c | Tune `n_steps` × `n_envs` for new NN size | Amortize expensive PPO updates | Low — adjust to fit VRAM |

**torch.compile details:**
```python
# In rl/self_play.py, after model creation
import torch
if hasattr(torch, "compile"):
    model.policy = torch.compile(
        model.policy,
        mode="reduce-overhead",  # best for repeated same-shape calls
        fullgraph=False,          # False = safer with MaskablePPO hooks
    )
```
The `reduce-overhead` mode specifically targets GPU kernel launch overhead that hurts small-batch repeated inference.

**Mixed precision details:**
```python
# In AWBWFeaturesExtractor.forward() and policy forward
from torch.amp import autocast

def forward(self, obs):
    with autocast(device_type="cuda", dtype=torch.float16):
        spatial = obs["spatial"]
        # ... rest of forward
```

**n_steps tuning:** Current `n_steps=512, n_envs=4` gives 2048 rollout steps per update. With a larger NN, PPO update cost is higher relative to collection. Try:
```python
n_steps=1024, n_envs=8  # 8192 steps/update — same wall time per step,
                        # but 4× fewer expensive PPO updates
```

### 12.2 Tier 2: Medium Effort, High Impact

| # | Fix | Expected Impact | Risk |
|---|-----|-----------------|------|
| 2a | `AsyncVectorEnv` with `shared_memory=True` (replace SubprocVecEnv) | +10–30% IPC speedup | Medium — test pickle serialization |
| 2b | Actor-Learner threading split | Overlap GPU update with env stepping | High — SB3 integration complexity |

**AsyncVectorEnv details:**

The 30×30×63 spatial obs is 1.7× larger than before (63 vs ~37 effective channels). Every step serializes this over the pipe in SubprocVecEnv.

```python
# In rl/self_play.py
from stable_baselines3.common.vec_env import AsyncVectorEnv  # or gymnasium's

# Instead of:
env = SubprocVecEnv([make_env_fn(i) for i in range(n_envs)])

# Use:
env = AsyncVectorEnv([make_env_fn(i) for i in range(n_envs)],
                      shared_memory=True)
```

**Actor-Learner split (conceptual):**

This is the architectural fix that solves the large-NN problem at scale. The core issue: with SubprocVecEnv, GPU sits idle while workers step the engine, and CPU envs sit idle while GPU does PPO update.

```python
# Concept — wire into rl/self_play.py
import threading
import queue

rollout_queue = queue.Queue(maxsize=3)  # bounded so learner doesn't fall behind

def actor_thread():
    """Runs envs + inference on CPU, pushes rollout buffers."""
    while True:
        rollout = collect_rollout(env, policy_cpu_copy)
        rollout_queue.put(rollout)

def learner_loop():
    """Pulls rollouts, does GPU update, pushes weights back."""
    while True:
        rollout = rollout_queue.get()
        model.learn_from_rollout(rollout)
        # copy updated weights → actor's CPU policy
        actor_policy.load_state_dict(model.policy.state_dict())
```

Not trivial to wire into SB3, but it's the right architecture for large NNs. The GPU update and env stepping overlap instead of alternating.

### 12.3 Tier 3: NN Architecture — Don't Shrink, Restructure

If throughput is still insufficient after Tiers 1–2, restructure the network for efficiency rather than reducing capability:

| # | Fix | Expected Impact | Risk |
|---|-----|-----------------|------|
| 3a | Depthwise separable convs in ResNet blocks | ~8× fewer MACs on 3×3 convs | Medium — test equivalence |
| 3b | **Factored output head** (256→512→900+38 vs 256→35000) | 97% reduction: 9M→240k params | Low — lossless restructuring |

**3a: Depthwise Separable Convolutions:**

If the network went wider (e.g., 64→256 channels), most compute is in 3×3 convolutions. Replace with depthwise separable:

```python
# In rl/network.py AWBWFeaturesExtractor
class EfficientResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            # Depthwise: spatial mixing, cheap
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            # Pointwise: channel mixing
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.conv(x))
```

Same representational power for spatial + channel mixing, ~8× fewer multiply-adds.

**3b: Factored Output Head (HIGHEST LEVERAGE):**

The 256→35k output layer is almost certainly dominating inference time:
- 256-dim → 35,000-dim linear = **8.96M parameters** just in the policy head

This is already partially addressed in §1 (factored spatial head + scatter to 35k flat), but verify the implementation matches:

```python
# Instead of one 256→35000 linear:
# 35000 = 30*30*38 + small_actions ≈ tiles × actions_per_tile
# Factor as: 256 → 512 → [tile_logits(900) + action_logits(38)]
# Then combine: full_logit[tile, action] = tile_score[tile] + action_score[action]

class FactoredPolicyHead(nn.Module):
    def __init__(self, feat_dim=256, n_tiles=900, n_action_types=38):
        super().__init__()
        self.tile_head = nn.Linear(feat_dim, n_tiles)    # 230k params
        self.action_head = nn.Linear(feat_dim, n_action_types)  # 9.7k params
        # vs 8.96M params before

    def forward(self, features):
        tile_logits = self.tile_head(features)      # [B, 900]
        action_logits = self.action_head(features)  # [B, 38]
        # Outer sum to get [B, 900*38] ≈ [B, 34200], then pad/mask to 35000
        return (tile_logits.unsqueeze(2) + action_logits.unsqueeze(1)).view(B, -1)
```

This reduces the output head from 9M → 240k params while preserving expressivity for the tile-action decomposition.

**Verify channel count propagation:** The HP belief channels added 4 spatial channels (59→63). Ensure the first conv layer's input size reflects this change cleanly — a silent doubling would add ~40% to first-layer compute.

### 12.5 Options (ordered by ROI per week of work)

These remain relevant for engine-level optimizations AFTER the NN bottlenecks are addressed:

| Option | Effort | Expected total-fps gain | Training-data safe? |
|---|---|---|---|
| Shared-memory `AsyncVectorEnv` (replace `SubprocVecEnv`) | 1 week | +10-30% | ✅ Yes |
| `cProfile` train.py end-to-end (decision input only) | 2 hours | (informational) | ✅ Yes |
| Flat-array engine refactor (occupancy/terrain as `np.int8`) | 3-5 days | +5-15% on engine, **enables Numba** | ⚠️ See §12.6 |
| Numba `@njit` on `compute_reachable_costs` after flat-array | 1-2 days | 5-10× on engine, ~3-5% total | ✅ Yes |
| mypyc compile of typed engine module | 1 week | 1.5-3× on engine, ~5% total | ✅ Yes |
| Cython `cdef class` rewrite of `Unit`/`GameState` | 2-3 weeks | 3-10× on engine, ~10-15% total | ⚠️ See §12.6 |
| PyPy alternative interpreter | — | (incompatible with PyTorch + spawn) | N/A — skip |
| Nuitka whole-program compile | — | (poor fit for hot loops) | N/A — skip |

### 12.6 Training-data compatibility (CRITICAL)

**Most native-compilation options preserve trained weights.** Model weights (`AWBWFeaturesExtractor` + policy head + value head) depend only on the **observation tensor shape, action-space layout, and network architecture** — not on how the engine internally computes things.

**Safe (no weight invalidation, training data fully reusable):**
- Cython compilation of pure functions (preserves Python semantics)
- Numba `@njit` on numeric kernels (operates on the same arrays)
- mypyc compilation
- `AsyncVectorEnv` / shared-memory IPC swap
- BFS / pathfinding rewrites that produce byte-identical outputs (we have equivalence tests for this from Phases 2b/2c/2d)

**UNSAFE (would force a fresh training run from scratch):**
- Any change to the **observation channel layout** in `rl/encoder.py` (channel order, channel count, scalar vector shape) — `AWBWFeaturesExtractor` first conv layer is shape-locked
- Any change to the **35k action space layout** (`_flat_to_action` index mapping) — the policy head output dim is shape-locked
- Replacing `Unit` with a struct-of-arrays IF that change leaks into the encoder output — pure-internal refactors are safe

**Conditional (depends on implementation discipline):**
- Flat-array engine refactor (§12.1 row 3): SAFE if we keep `Unit` as a Python view object that the encoder consumes through the same interface; UNSAFE if we change what `encode_state` writes into the spatial buffer
- Cython `cdef class` rewrite of `Unit`/`GameState`: SAFE if encoder still receives the same numerical observation; the bigger risk is breaking pickle for `SubprocVecEnv` (would need `__reduce__`)

**Operational rule:** before any native-compilation work lands, run the encoder equivalence test:
```python
# quick regression: encode N states pre/post change, np.array_equal both
# obs tensors. If equal, weights survive.
```
If we cannot make that test pass, the change forces a fresh training run — quote the cost up front.

### 12.7 Re-entry conditions (legacy)

These conditions are now **superseded** by §12.0 (profile first). The bottleneck profile has shifted with the larger NN — engine cost dominance is no longer the prerequisite for optimization work.

**Original conditions (archived):**
- [x] Phase 6 `n_envs` sweep formalized
- [x] Profile shows non-engine bottleneck — **now known to be stale** (NN changed)
- [ ] Tier 1 free wins completed (torch.compile, mixed precision, n_steps tuning)
- [ ] Tier 2 medium-effort items evaluated (AsyncVectorEnv)
- [ ] If still insufficient: Tier 3 architectural changes

Until Tier 1 is exhausted: focus on NN-level optimizations, not engine-native compilation.

---

*"He who fails to plan is planning to fail."*
*— Winston Churchill, British Prime Minister during World War II.*


---

## 13. League System — Profiles, Sampling, and Training Integration (ACTIVE)

Goal:
Introduce a league-style training ecosystem that improves robustness and prevents overfitting on Standard maps.

Key principles:
- Single shared PPO
- Profiles are config-driven distributions
- No hardcoding CO lists into core logic
- Profiles differ via sampling, not architecture

Implementation stages:
- Phase 1: logging + tagging only
- Phase 2: sampling influence
- Phase 3: Elo-based matchmaking
- Phase 4: optional conditioning/adapters

Profiles (draft, configurable):
- Meta Optimizer
- Aggro / Tempo
- Control / Positional
- Density Punisher
- Econ / Scaling
- Explorer (low budget)
- Favorites (optional)

Profiles must differ in:
- CO sampling
- opponent sampling
- map weighting
- evaluation metrics

---

## 14. MCTS Rollout Strategy — Staged Integration (ACTIVE)
WARNING: BY THIS POINT WE SHOULD HAVE ALL MAPS/ ALL COs SEMI COMPETENT - NEED TO INTRODUCE SOME NEW TILE SPECIFIC CO INTERACTIONS BEFORE THIS POINT.  ergo kindle, lash, koal,jake need to have their tile dependent bonuses encoded 
Goal:
Introduce MCTS without destroying training throughput.

**Plan pointer:** Stages below are the program’s rollout ladder. **Code:** preset IDs `mcts_0` … `mcts_4`, merge helpers, and per-stage defaults live in [`rl/mcts_rollout_stages.py`](rl/mcts_rollout_stages.py) (wired in [`scripts/symmetric_checkpoint_eval.py`](scripts/symmetric_checkpoint_eval.py) via `--mcts-rollout-stage` and `mcts_work_payload_from_argparse`). Operational detail, remaining todos, and ordering live in [`.cursor/plans/mcts_forward_unified.plan.md`](.cursor/plans/mcts_forward_unified.plan.md) (execution) and [`.cursor/plans/mcts_optimization_campaign.plan.md`](.cursor/plans/mcts_optimization_campaign.plan.md) (scale-up / perf).

Stages (sim counts are indicative; training/eval often uses fixed `num_sims` until anytime mode is validated):

MCTS-0 (plumbing):
- 8–32 sims
- debug only
- **Code:** `rl/mcts.py` `run_mcts`, `engine/game.py` `apply_full_turn` — exercised in tests; not a strength milestone.

MCTS-1 (evaluation):
- 64–256 sims
- fixed seeds
- **Code:** `scripts/symmetric_checkpoint_eval.py` with `mcts_mode=eval_only`; pair with `tools/mcts_health.py` / fleet gates.

MCTS-2 (selective training assist):
- 128–512 sims
- used on subset of states
- **Not** full MCTS inside PPO `learn` — orchestrated / eval-only paths only (see Rules).

MCTS-3 (distillation):
- offline MCTS → supervised targets
- **Future:** not wired as a default training mode in-repo.

MCTS-4 (production):
- time-based anytime search
- deterministic output
- **Future:** wall-clock budget per P0 root (see unified plan todo `mcts-fwd-12`); current eval uses sim caps + optional root risk stats.

Rules:
- Do not use full MCTS in PPO loop
- Use MCTS as teacher, not replacement
- Require EV > 0.6 before relying on it

### MCTS stochastic risk layer (root-only) — **implemented**

**Shipped (2026-04):** root-only stochastic risk layer for advisor / production-style selection without changing the default PUCT training path.

| Piece | Role |
| ----- | ---- |
| `GameState.apply_full_turn(..., return_trace=True)` | Optional 5-tuple with per-step advisory trace (`critical_threshold_event`, combat deltas, capture flags). Default call still returns the original 4-tuple. `engine/game.py` |
| `rl/mcts.MCTSConfig` | `luck_resamples`, `luck_resample_critical_only`, `risk_mode` (`visit` \| `mean` \| `mean_minus_p10` \| `constrained`), `risk_lambda`, `catastrophe_value`, `max_catastrophe_prob`, `root_decision_log_path` |
| `rl/mcts` | `EdgeStats`, fixed-plan luck resampling after search, `_state_value_for_actor` (value head assumed **active-player frame**; flipped to root frame for risk scoring) |
| `scripts/symmetric_checkpoint_eval.py` | CLI / JSON payload for the above; telemetry includes `mcts_root_entropy`, `mcts_chosen_risk` |
| Tests | `tests/test_mcts.py` (luck resample schema, `return_trace` schema), `tests/test_apply_full_turn.py` (unchanged default API) |

**Operational default:** keep `risk_mode=visit` (legacy visit-count selection) for training / symmetric A/B unless you are explicitly ablating the risk policy. For ranked-advisor-style runs, see recommended flags in `docs/mcts_review_composer_o.md` Part B.

**Caveat:** If the PPO value head is not trained in **active-player** value frame, `_state_value_for_actor` will mis-score resampled children — validate value framing before trusting risk-ranked play.
