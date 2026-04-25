# Superhuman restart — master architecture spec

**Owner (coordinator):** Opus
**Plan file:** [.cursor/plans/superhuman_restart_architecture_bundle.plan.md](../../.cursor/plans/superhuman_restart_architecture_bundle.plan.md)
**Status:** Wave 1 specs complete and lead-approved. Wave 2 (implementation) is next.

This document is the **single contract** for the architectural restart bundle. Every detail lives in the per-topic specs in this folder; this file is the index, the locked numbers, and the wave-2 work breakdown. If a per-topic spec disagrees with this file on a locked number, this file wins and the per-topic spec gets a fixup PR.

---

## 1. Why a single bundled restart

The current network's policy head is `Linear(256, 35_000)` after `AdaptiveAvgPool2d((8,8))`, which destroys the spatial inductive bias before action selection and forces the policy to compress the entire 30×30 tactical decision through a 256-d bottleneck. Replacing that head requires retraining from scratch. We use the same restart to ship every other change that breaks the observation, action, or return-distribution contract, paying the checkpoint-invalidation cost exactly once.

**Bundled changes** (all detailed below):

1. Deeper/wider residual tower (10 blocks × 128 channels, no global pooling)
2. Factored spatial policy head (per-tile logits scattered into the existing 35,000-flat)
3. **MOVE-action encoding redesign** (newly added post wave-1; without it, change #2 cannot drive MOVE destinations)
4. New encoder channels: 6 influence + 1 defense_stars → `N_SPATIAL_CHANNELS = 70`
5. Ego-centric encoder reframe ("me" / "enemy" instead of P0/P1 fixed blocks)
6. `AWBW_REWARD_SHAPING=phi` as default, with Φ computed in the **learner frame**
7. Seat-balanced PPO actor (both engine seats roll out under the same policy)
8. PFSP opponent sampling on top of the existing snapshot pool
9. `AsyncVectorEnv` swap (perf, but bundled to avoid re-baselining gates twice)

**Out of scope** (deferred to MASTERPLAN, NOT in this bundle):

- Fog of war / POMDP (MASTERPLAN §8 — separate program)
- Native compilation / `torch.compile` (MASTERPLAN §12 — measured later)
- Hierarchical RL / option discovery (MASTERPLAN §5)
- Behavior cloning warmup (separate Phase-0 add-on if Phase-1 ladder stalls)

---

## 2. Locked numbers (the contract)

| Quantity | Value | Source |
|---|---|---|
| `N_SPATIAL_CHANNELS` | **70** | `influence_channels_spec.md` (63 baseline + 6 influence + 1 defense_stars) |
| `N_SCALARS` | unchanged at 17 | `rl/encoder.py` |
| Spatial grid | 30 × 30 (padded) | `rl/encoder.py` `GRID_SIZE` |
| Residual tower | **D=10, W=128** | `compute_budget.md` |
| Pooling before head | **none** (drop `AdaptiveAvgPool2d`) | `spatial_head_spec.md` §2 |
| `ACTION_SPACE_SIZE` | **35,000** (unchanged) | `rl/network.py` |
| `_MOVE_OFFSET` | **1818** (occupies 1818..2717, inside unused 1818..3499) | `move_encoding_redesign.md` §3.2 |
| Default `AWBW_REWARD_SHAPING` | **phi** | MASTERPLAN §11.4 |
| Φ frame | **learner** (me − enemy) | `ego_centric_refactor_map.md` §7 |

These numbers are frozen for wave 2. Any deviation requires a coordinator sign-off and an update to this table.

---

## 3. Per-topic spec index

| Topic | Spec | Wave-2 owner (composer role) |
|---|---|---|
| Action space inventory (status quo, used vs unused ranges) | `action_space_inventory.md` | Reference only — informs items below |
| Spatial policy head (per-tile logits → flat scatter) | `spatial_head_spec.md` | Network composer |
| **MOVE encoding redesign** | `move_encoding_redesign.md` | Env composer |
| Influence + defense-stars channels | `influence_channels_spec.md` (+ §6 below) | Encoder composer |
| Ego-centric encoder reframe | `ego_centric_refactor_map.md` | Encoder composer |
| Compute budget / tower sizing | `compute_budget.md` | Network composer (cite, do not re-derive) |

---

## 4. Tower & policy/value heads (consolidated)

**Trunk** (replaces `AWBWFeaturesExtractor` + `AWBWNet` trunk in `rl/network.py`):

```
stem:   Conv2d(70 → 128, 3×3, pad 1) → ReLU
trunk:  10 × ResBlock(128 → 128, 3×3, pad 1, BN, ReLU, residual add)
out:    keep spatial 30×30, channel 128 (NO AdaptiveAvgPool)
```

**Scalar fusion:** project the 17 scalars to a 30×30 broadcast plane (small MLP → `Linear(17 → 16)` → tile to `(B, 16, 30, 30)`) and concat onto the trunk output → `(B, 144, 30, 30)` for the heads. (See `spatial_head_spec.md` §3 for the exact broadcast — implement as that spec describes; this paragraph is just the contract.)

**Policy head:** factored, per `spatial_head_spec.md`:
- Per-tile spatial actions (SELECT, MOVE, ATTACK, CAPTURE/WAIT/LOAD/JOIN/DIVE_HIDE, REPAIR, BUILD-tile-of-build) emit `Conv2d(144 → K_type, 1×1)` → reshape & scatter into the flat 35,000-vector at the correct offsets.
- Pure scalar actions (END_TURN, ACTIVATE_COP, ACTIVATE_SCOP) come from a small `Linear` over a global-pooled feature, scattered into indices 0,1,2.
- The flat vector is masked-filled with `-inf` at illegal positions and the existing categorical sampler is reused unchanged. The shape contract `(B, ACTION_SPACE_SIZE)` is preserved.

**Value head:** `AdaptiveAvgPool2d((1,1))` → `Linear(144, 256) → ReLU → Linear(256, 1)`. (Pooling is fine for the value head; the issue was only the policy head losing per-tile structure.)

**Parameter count target:** ~5.2M trunk + ~0.7M heads = ~6M total per `compute_budget.md`. Confirm with `summary(model)` in the wave-2 smoke.

---

## 5. Action encoding (consolidated)

The full layout is in `move_encoding_redesign.md` §3.3. The two changes from today:

1. **MOVE-stage `SELECT_UNIT`** is encoded by `move_pos` at `_MOVE_OFFSET + r*30 + c` = 1818..2717. SELECT-stage `SELECT_UNIT` is unchanged at 3..902.
2. `_action_to_flat` gains a `state` argument so it can branch on `state.action_stage`. All call sites must update; wave-2 env composer audits the repo for any direct callers.

**Pre-existing collisions out of scope:** UNLOAD bit-packing and the SELECT/ATTACK overlap on 900..902 are documented in the inventory and remain. They are not action-destination collisions, so the spatial head is not blocked by them. Defer to a follow-up plan if they ever cause measurable harm.

---

## 6. Encoder channels (consolidated)

Total: **70 spatial channels** in this fixed order. The first 63 indices match today's encoder. Indices 63–69 are new.

| Idx range | Channels | Source |
|---|---|---|
| 0..27 | 28 unit-presence channels (14 unit types × 2 players) — but **reframed me/enemy** post-bundle | `ego_centric_refactor_map.md` |
| 28..29 | hp_lo, hp_hi (belief interval) | unchanged |
| 30..44 | 15 terrain one-hot | unchanged |
| 45..59 | 15 property channels (5 property types × 3 ownership states) — **reframed me/enemy/neutral** | `ego_centric_refactor_map.md` |
| 60..62 | 3 capture-extra channels — **reframed me/enemy/neutral-income mask** | `ego_centric_refactor_map.md` |
| **63** | `threat_in_me` (incoming damage to my units, normalized) | `influence_channels_spec.md` |
| **64** | `threat_in_enemy` (incoming damage to enemy units, normalized) | `influence_channels_spec.md` |
| **65** | `reach_me` (per-tile reachability frontier from my units) | `influence_channels_spec.md` |
| **66** | `reach_enemy` (per-tile reachability frontier from enemy units) | `influence_channels_spec.md` |
| **67** | `turns_to_capture_me` (per capturable property, my fastest cap ETA, clamped & normalized) | `influence_channels_spec.md` |
| **68** | `turns_to_capture_enemy` (mirror) | `influence_channels_spec.md` |
| **69** | `defense_stars` (per-tile `TerrainInfo.defense / 4`, ride existing terrain cache → ~0 runtime cost) | this spec (§6.1 below) |

The influence-channels spec was originally written assuming P0/P1 framing. Wave-2 encoder composer must reconcile it with the ego-centric refactor: every "_p0" / "_p1" suffix in that doc becomes "_me" / "_enemy" in the live encoder. The doc itself can stay as-is; the implementation interprets it through the ego-centric lens.

### 6.1 defense_stars channel (newly added post wave-1)

- **Definition:** for each tile `(r,c)` with terrain `t`, output `t.defense / 4.0` ∈ [0.0, 1.0].
- **Why:** defense stars are combat-critical (they multiply incoming damage in the AWBW formula) and the network can in principle derive them from the 15-channel terrain one-hot, but giving them as an explicit plane reduces the sample-efficiency burden and gives the policy a direct "stand on cover" prior. Cost is 1 channel out of 70; runtime is amortized by the existing `_encoded_terrain_channels` cache.
- **Implementation note:** add to the same cache that produces channels 30..44. The defense plane is constant per `MapData` and never depends on units, so caching it once per map is correct.

---

## 7. Reward shaping (consolidated)

**Default flip:** `AWBW_REWARD_SHAPING=phi`. The `level` reward path stays in the codebase as a fallback for ablation but is no longer the default.

**Frame fix:** `_compute_phi` in `rl/env.py` currently computes `Φ = α·(p0_val − p1_val) + β·(cap_p0 − cap_p1)` (hardcoded P0 frame). After the bundle, the function takes a `learner_seat` parameter (or reads `state.current_player_index`) and computes `Φ = α·(me_val − enemy_val) + β·(cap_me − cap_enemy)`. The same fix applies to the legacy `level` path while it still exists.

This is a return-distribution change, not an observation change, so V(s) regresses from scratch — which is fine because we are restarting anyway.

---

## 8. Training paradigm (consolidated)

**Seat-balanced actor:** the same network controls both engine seats; rollouts are split ~50/50 across seats so the policy generalizes across move-order. Depends on items 5 and 6 above (ego-centric encoder + Φ learner-frame).

**PFSP opponent sampling:** replace uniform-over-snapshots with PFSP weighting (probability proportional to `f(p_win_rate)` with `f` concave, e.g. `(1 − w)·w` or `(1 − w)^p`) so the actor spends more time fighting opponents it currently beats ~50/50 and less time stomping snapshots it already dominates. Implementation hook: `reload_opponent_pool` in `rl/self_play.py`.

**`AsyncVectorEnv` swap:** replace `SubprocVecEnv` with `gymnasium.vector.AsyncVectorEnv` for tighter step-overhead and easier per-env seeding. Bundled to consolidate baselining; not a contract change.

---

## 9. Wave 2 — implementation work breakdown

Five composers, dispatched in parallel. Each gets a single bounded scope and a clear acceptance test.

| ID | Scope | Files | Acceptance |
|---|---|---|---|
| `wave2-encoder` | New 70-channel encoder with influence + defense_stars + ego-centric reframe. Add `engine/threat.py` for influence helpers. Regenerate `tests/fixtures/encoder_equivalence_pre_restart.npz` → `_post_restart.npz`. | `rl/encoder.py`, `engine/threat.py`, `tests/fixtures/`, `tests/test_encoder_equivalence.py` | New encoder runs on a 100-state corpus without exception; equivalence test passes against newly snapshotted post-restart baseline; channel order matches §6 table exactly. |
| `wave2-env` | MOVE-encoding redesign per `move_encoding_redesign.md`. Thread `state` into `_action_to_flat`, audit all callers. Update `_get_action_mask`. Add `tests/test_action_encoding_equivalence.py` per spec §8–9. | `rl/env.py`, `tests/` | All 5 wave-2 unit tests in spec §9 pass; encoder-equivalence harness for actions establishes a frozen mask baseline. |
| `wave2-network` | New trunk (10×128, no AvgPool) + factored spatial policy head + value head. Preserve `(B, ACTION_SPACE_SIZE)` policy output shape. Update `summary(model)` log line. | `rl/network.py` | Forward pass on a synthetic batch of 16 returns correct shapes; param count is within 10% of `compute_budget.md` projection (~6M); masked-fill produces a valid categorical distribution on a synthetic mask. |
| `wave2-rewards` | Φ default flip + learner-frame rewrite of `_compute_phi`. Update `tools/phi_smoke.py` to report me/enemy framing. | `rl/env.py`, `tools/phi_smoke.py` | `phi_smoke` on Misery T3 mirror produces non-zero shaping signal under both seat assignments; sign of Φ flips when learner seat flips on a symmetric position. |
| `wave2-paradigm` | Seat-balanced rollouts + PFSP opponent sampling + `AsyncVectorEnv` swap. | `rl/self_play.py`, `rl/train.py` | Smoke training run for 10k env steps with 2 envs, both seats represented in the rollout buffer at ~45–55% each; PFSP weights are non-uniform after 5 snapshots; AsyncVectorEnv hands back observations of the right shape. |

**Coordinator integration step (after all five land):** I run a 5-minute scratch smoke (`train.py --total-timesteps 50000 --n-envs 4`) on pc-b, capture FPS, confirm all encoder/action/reward/network/paradigm changes work together, write the result into a wave-2 sign-off note, then dispatch wave 3.

**Wave 2 status (landed in-tree):** ego-centric 70-ch encoder with `engine/threat.py` influence planes; 10×128 trunk + scalar fusion + factored policy head (incl. MOVE band 1818..2717); Φ default + learner-frame shaping; seat-balanced episodes (`AWBW_SEAT_BALANCE`); PFSP (`AWBW_PFSP` / `AWBW_PFSP_STATS`); optional `AWBW_ASYNC_VEC` (falls back to `SubprocVecEnv`). Encoder baseline regenerated; full pytest green.

---

## 10. Wave 3 — Phase 1 ladder rerun

Out of scope for this spec. Owned by the existing Phase 1 plan (`.cursor/plans/phase1-foundation-validation.plan.md`); only the contract above is what wave 3 inherits.

---

## 11. Human-vs-bot encoder (fixed wave-2+)

`server/play_human.py` encodes human demos with `observer=HUMAN_PLAYER` (0) and bot inference with `observer=BOT_PLAYER` (1) so the ego-centric layout matches training.
