# Ego-centric encoder — refactor map (spec + status)

**Scope:** STD ranked play, no fog. **Non-goal:** changing engine `GameState` seat semantics (`active_player`, `units[0/1]`, `funds[0/1]`, …). Only the **observation** tensor (and any tooling that assumes fixed P0-blocks) is remapped to **me / enemy** relative to a caller-supplied `observer` (the learner or BC “human seat”).

**Prerequisite:** [MASTERPLAN.md](../../MASTERPLAN.md) §9 (Tier 4) and the §9.1 / training-doctrine table: ego-centric encoding is the gate for seat-balanced PPO and for BC on both seats from one human game.

**Related bundle:** [`.cursor/plans/superhuman_restart_architecture_bundle.plan.md`](../../.cursor/plans/superhuman_restart_architecture_bundle.plan.md) (`ego-centric-encoder`).

### Implementation status (2026-04)

- **Shipped:** ego-centric framing and the **77-channel** spatial encoder in [`rl/encoder.py`](../../rl/encoder.py) (`N_SPATIAL_CHANNELS = 77`, `N_SCALARS = 16`). The tail after the pre-influence stack is **six influence planes** (indices 63–68), **one defense-stars** plane (index 69), and **seven unit modifier** planes (indices 70–76); see module docstring and `N_INFLUENCE_CHANNEL_BASE` / `N_DEFENSE_STARS_CHANNEL` / `N_UNIT_MODIFIER_CHANNEL_BASE`.
- **Frozen baseline (byte-identical harness):** [`tests/fixtures/encoder_equivalence_pre_restart.npz`](../../tests/fixtures/encoder_equivalence_pre_restart.npz). **Regeneration policy** (env var, review expectations, 8-sample corpus): [`tests/fixtures/encoder_equivalence_README.md`](../../tests/fixtures/encoder_equivalence_README.md). [`tests/test_encoder_equivalence.py`](../../tests/test_encoder_equivalence.py) compares live `encode_state` output to that file.

| Repo truth (2026-04) | Value |
|----------------------|--------|
| `N_SPATIAL_CHANNELS` | **77** (includes **6** influence + **1** defense_stars + **7** unit modifier planes at the tail) |
| `N_SCALARS` | **16** |
| Encoder equivalence fixture | `tests/fixtures/encoder_equivalence_pre_restart.npz` |
| Regeneration / ops | `tests/fixtures/encoder_equivalence_README.md` |

---

## 1. Encoder API: before / after

### Current (fixed P0 / P1 blocks)

Implementation: [`rl/encoder.py`](../../rl/encoder.py). `N_SPATIAL_CHANNELS = 77`, `N_SCALARS = 16`. The **77** includes the **influence** block (6 channels), **defense_stars** (1 channel), and **unit modifier** block (7 channels) after terrain, property, capture, and neutral-income planes—see **Implementation status** above.

**Spatial (conceptual; indices are stable channel *positions*; shipped encoder uses me/enemy via `observer`):**

- **Unit presence:** `spatial[..., 0:14]` = **me** by type; `spatial[..., 14:28]` = **enemy** (`player_ch_offset = 14 * slot` with me=0, enemy=1 relative to `observer`; legacy prose below used P0/P1 when `observer=0`).
- **HP belief:** `hp_lo_ch = 28`, `hp_hi_ch = 29` — **already observer-aware:** when `belief` is set, the encoder uses exact HP for `unit.player == observer` and belief interval for the opponent (lines 292–312). When `belief is None` (STD legacy/debug), all units get exact `hp/100` in both channels (see module docstring lines 28–42).
- **Terrain:** 15 one-hot channels (offset after units + HP).
- **Property ownership:** for each property type, three slots `neutral / P0 / P1` at `prop_ch_offset + ptype*3 + ownership` with `ownership ∈ {0,1,2}` (lines 261–268).
- **Capture progress:** `cap_ch0` / `cap_ch1` = unit on tile with `player == 0` or `1` reducing `capture_points` (lines 242–280).
- **Neutral income mask:** one channel after capture pair (line 244).
- **Influence + defense stars + unit modifiers (tail of the 77):** six influence planes (`engine/threat.py` / `compute_influence_planes`), one map-static defense-stars channel (`TerrainInfo.defense / 4`), then seven occupied-cell unit modifier planes, indices **63–76** (`N_INFLUENCE_CHANNEL_BASE` … `N_UNIT_MODIFIER_CHANNEL_BASE + 6`).

**Scalars (index → meaning today):**

| Index | Current meaning |
|------:|-----------------|
| 0 | `funds[0] / 50_000` |
| 1 | `funds[1] / 50_000` |
| 2 | P0 power bar / SCOP threshold |
| 3 | P1 power bar / SCOP threshold |
| 4 | P0 COP active |
| 5 | P0 SCOP active |
| 6 | P1 COP active |
| 7 | P1 SCOP active |
| 8 | `turn / max_turns` |
| 9 | `float(state.active_player)` **raw 0/1 (not “my turn”)** |
| 10 | `co0.co_id / 30` |
| 11 | `co1.co_id / 30` |
| 12 | rain |
| 13 | snow |
| 14 | `co_weather_segments_remaining / 2` |
| 15 | income share — **me** share of contestable income tiles |

```python
def encode_state(state, *, observer=0, belief=None, out_spatial=None, out_scalars=None) -> (spatial, scalars):
    # spatial[..., 0:14] = P0 unit presence; [14:28] = P1 unit presence
    # property: +1 = owned by P0, +2 = owned by P1
    # cap_ch0 = P0 capturing; cap_ch1 = P1 capturing
    # scalars[0/1] = me/enemy funds; [9] = my_turn; [15] = me income share
```

### Proposed (ego-centric, same shapes)

`observer ∈ {0,1}` = engine seat of **“me”** (the policy being trained, the human in BC, or the player whose MCTS node is being evaluated). **Enemy** = `1 - observer` throughout the encoder.

- **Unit presence:** `spatial[..., 0:14]` = **me** units; `spatial[..., 14:28]` = **enemy** units.
- **Property:** `+1` = owned by me, `+2` = owned by enemy (still `0` = neutral).
- **Capture progress:** `cap_me` / `cap_enemy` (same two channel *slots* as today; rename semantics only).
- **HP:** unchanged *logic* — still exact HP for `unit.player == observer`, belief (or exact if `belief is None`) for the other side. Ego unit blocks now line up with “me” / “enemy” channel blocks.

**Scalars — shipped layout (`N_SCALARS = 16`, tier removed):**

| Index | Ego-centric meaning |
|------:|---------------------|
| 0 | `funds[observer] / 50_000` (**me** funds) |
| 1 | `funds[1-observer] / 50_000` (**enemy** funds) |
| 2 | **me** power bar / threshold |
| 3 | **enemy** power bar / threshold |
| 4 | **me** COP active |
| 5 | **me** SCOP active |
| 6 | **enemy** COP active |
| 7 | **enemy** SCOP active |
| 8 | `turn / max_turns` (unchanged) |
| 9 | **`1.0` if `state.active_player == observer` else `0.0`** (my turn — **replaces** raw `active_player` at index 9) |
| 10 | `co_states[observer].co_id / 30` |
| 11 | `co_states[1-observer].co_id / 30` |
| 12–14 | weather scalars (not seat-tied) |
| 15 | **me income share** — `count_income_properties(observer) / n_income_tiles` |

**Docstring / comments:** update module header in [`rl/encoder.py`](../../rl/encoder.py) (lines 1–48) to describe me/enemy; rename `_p0_income_share` to an observer-argument helper (e.g. `_income_share_for`) internally.

**`active_player` scalar (recommended):** use **`my_turn` binary** at index 9. **Do not** keep raw `state.active_player` in the vector — it breaks symmetry across seats (same as §7 risks). Engine `state.active_player` remains the source of truth in logs and `GameState`.

---

## 2. Refactor surface — every site that needs to change

Line numbers are from the **AWBW** repo as of the authoring pass; re-verify before implementing.

| File | Line(s) | Symbol / region | Current assumption | Required change |
|------|---------|-----------------|--------------------|-----------------|
| [`rl/encoder.py`](../../rl/encoder.py) | 1–26 | module doc | P0/P1 channel names in prose | Me/enemy + scalar table; `cap` naming |
| [`rl/encoder.py`](../../rl/encoder.py) | 57–90 | `N_*` constants | 77 spatial (incl. influence + defense_stars + unit modifiers) + 16 scalars | Locked in code; bumping channels is a restart event |
| [`rl/encoder.py`](../../rl/encoder.py) | 240–280 | property + capture | `ownership` 1=P0, 2=P1; `cap_ch0/ch1` by `player == 0/1` | Map to me/enemy using `observer` |
| [`rl/encoder.py`](../../rl/encoder.py) | 283–312 | unit loop + HP | P0 then P1 fixed offsets | Emit me then enemy; HP branch already uses `observer` |
| [`rl/encoder.py`](../../rl/encoder.py) | 314–370 | `encode_state` scalars | `funds[0/1]`, `co0/co1` order, `active_player` raw, `_p0_income_share` | Ego order + `my_turn` + me-income share |
| [`rl/encoder.py`](../../rl/encoder.py) | 374–380 | `_p0_income_share` | P0 only | Generalize to `observer` (rename) |
| [`rl/env.py`](../../rl/env.py) | 1–9 | module doc | “Agent = player 0” | After seat-balance: learner seat + ego obs |
| [`rl/env.py`](../../rl/env.py) | 431–438 | `AWBWEnv` class doc | Obs always P0 | Ego for `learner_seat` (new param / field) |
| [`rl/env.py`](../../rl/env.py) | 783–800 | `reset` | P1 opens: run opponent; “obs on P0’s clock” | If learner is P1, contract is “obs on **learner** clock” + ego `encode_state(observer=learner)` |
| [`rl/env.py`](../../rl/env.py) | 837–910 | `step` | “Decode & apply player-0 action”; `_get_obs()` default 0 | Pass `learner_seat` into `_get_obs(observer=…)`; step still applies **learner** engine actions only (seat-balance: swap who is wired to the policy) |
| [`rl/env.py`](../../rl/env.py) | 996–1046 | `_compute_phi` | Φ in **P0 frame** (p0_val − p1_val, cap_p0 − cap_p1) | If critic trains on P1 seat with sign flip per MASTERPLAN §9, or Φ in **learner** frame: express material/property/capture in me−enemy; coordinate with return sign |
| [`rl/env.py`](../../rl/env.py) | 1048–1073 | `_get_obs` | `observer: int = 0` | Default / call sites pass **learner** seat |
| [`rl/env.py`](../../rl/env.py) | 872–891 | `level` reward | `p0_props − p1_props`, P0 vs P1 army value | Learner-frame diff if reward stays asymmetric to seat (spec detail for implementing wave) |
| [`rl/env.py`](../../rl/env.py) | 1244–1303 | `_run_random_opponent` / `_run_policy_opponent` | Opponent = seat 1; `accumulated_reward -= r_opp` on P1 terminal | If learner can be P1, opponent loop and reward stitching **must** be generalized (major env change; not only encoder) |
| [`rl/env.py`](../../rl/env.py) | 1274–1279 | `_run_policy_opponent` | `_get_obs(observer=1)` for checkpoint bot | Opponent’s `observer` = opponent seat (always 1 today; becomes `1-learner` when learner seat varies) |
| [`rl/env.py`](../../rl/env.py) | 1367–1384 | `terrain_usage_p0` | P0 units only | If metrics stay “red seat” for MCTS gate, keep engine-seat field; if “learner,” rename + document |
| [`rl/env.py`](../../rl/env.py) | 1420–1492 | `_log_finished_game` | `agent_plays: 0` | Set `me` / `learner_seat` / `agent_plays` to actual engine seat; bump `log_schema_version` |
| [`rl/network.py`](../../rl/network.py) | 5–69, 115–155 | `AWBWFeaturesExtractor` / `AWBWNet` | `N_SPATIAL_CHANNELS`, `N_SCALARS` by shape only | **No semantic seat coupling** — **0 layout changes** if C stays **70** and scalars **17**; retrain when the tensor contract changes |
| [`rl/self_play.py`](../../rl/self_play.py) | 573–1188 | `_make_env_factory` / `_build_vec_env` | `AWBWEnv` with implicit P0 learner | Thread learner seat (per-env or per-worker), opponent seat, and logging |
| [`rl/ppo.py`](../../rl/ppo.py) | (file) | training loop | No P0 in loop | **Verify only** — PPO is seat-agnostic; asymmetry is in `AWBWEnv` + data |
| [`rl/mcts.py`](../../rl/mcts.py) | 374, 393, 415 | `policy_callable` / `value_callable` / `prior_callable` | `env._get_obs(observer=int(s.active_player))` | After ego: tensor is **me = active player**; consistent with my_turn scalar; **no layout hack** for “wrong seat” vs fixed P0 blocks |
| [`rl/ckpt_compat.py`](../../rl/ckpt_compat.py) | 24, 138–150 | obs space / loader | **70**ch + **17** scalars (current); older 62/63ch zips | **compat** expands legacy stems; another channel bump is a **new** restart line |
| [`server/play_human.py`](../../server/play_human.py) | 43–45 | `HUMAN_PLAYER` / comment | `encode_state` “always P0 view” | Human = me: `encode_state(..., observer=0)`; **bot** (lines 311–315) must use `observer=BOT_PLAYER` (today **bug/inconsistency**: `encode_state(state)` default 0 while bot acts as P1) |
| [`server/play_human.py`](../../server/play_human.py) | 333–358 | `_append_human_demo` | `encode_state(state)` P0 only | `encode_state(state, observer=0)` + add **`me: 0`**; for future two-human seats, `me` follows human side |
| [`tools/human_demo_rows.py`](../../tools/human_demo_rows.py) | 32–34, 36–47 | `build_demo_row_dict` | `encode_state` default | `observer=me`; include `me` in row |
| [`tools/human_demo_rows.py`](../../tools/human_demo_rows.py) | 59–60, 75–80 | `iter_demo_rows_from_trace_record` | “`active_player == 0`” P0 rows only | Emit rows for `active_player == me` with **`me` ∈ {0,1}** when ingesting both seats (plus MOVE filter policy) |
| [`tools/verify_observation_encoding.py`](../../tools/verify_observation_encoding.py) | 38–53, 93–110, 158–175 | `SCALAR_LABELS`, unit grid, `encode_state` | P0/P1 labels; upper=P0, lower=P1; only 13 scalar labels for 17 scalars (stale) | Ego labels; fix full 17 labels; unit grid legend me/enemy |
| [`rl/ai_vs_ai.py`](../../rl/ai_vs_ai.py) | 510–511 | obs dict | `encode_state(state)` default | Pass explicit `observer` per controlled side if both use policy |
| [`analysis/co_ranker.py`](../../analysis/co_ranker.py) | 83–103 | `compute_rankings` | `winner == 0/1` attributes wins to p0_co / p1_co IDs | **Keep** — engine seats; do not rewrite to me/enemy |
| [`analysis/co_h2h.py`](../../analysis/co_h2h.py) | 98–110 | H2H win counting | `winner` vs P0 identity | **Keep** (engine semantics) |
| [`scripts/eval_imitation.py`](../../scripts/eval_imitation.py) | 10, 64–86 | post-BC eval | `w == 0` = win; fixed P0 agent | If eval runs **ego + learner P0 only**, unchanged; if seat-balanced, compare `w == me` using logged `me` or fixed eval seat |
| [`scripts/train_bc.py`](../../scripts/train_bc.py) | 77–95 | training loop | Loads `row["spatial"]` / `row["scalars"]` as stored | Require **`me`** (or re-encode from replay with `encode_state(..., observer=me)`); see §4 |
| [`tools/phi_smoke.py`](../../tools/phi_smoke.py) | 79, 276 | episode outcome | `winner == 0` | P0-wins proxy; use **`winner == learner_seat`** if smoke runs with variable seat |
| [`MASTERPLAN.md`](../../MASTERPLAN.md) | 120–123 | metrics snippet | `agent_won = winner == 0` | With `me`: `agent_won = (winner == me)` **when `me` present**; else legacy `winner == 0` |
| [`docs/seat_measurement.md`](../../docs/seat_measurement.md) | 18–25 | example | `agent_won` vs `winner == 0` | Same as MASTERPLAN — document `me` field |
| [`docs/player_seats.md`](../../docs/player_seats.md) | 10–14 | table + bullets | “Policy always P0”; “friendly = P0” | Update to ego-centric + learner seat; link this doc |
| [`tests/test_encoder_equivalence.py`](../../tests/test_encoder_equivalence.py) | 1–367 | harness + `test_encoder_output_matches_frozen_baseline` | Byte-stable vs `encoder_equivalence_pre_restart.npz` | Any **intentional** encoder numeric change: follow [`tests/fixtures/encoder_equivalence_README.md`](../../tests/fixtures/encoder_equivalence_README.md) (e.g. `AWBW_REGEN_ENCODER_BASELINE=1` after review); checkpoint invalidation gate |
| [`tests/test_encoder_terrain_cache.py`](../../tests/test_encoder_terrain_cache.py) | 39–55 | `encode_state(st)` | No observer (default 0) | Still valid; may add ego smoke |
| [`test_weather.py`](../../test_weather.py) | 368–406 | `TestEncodeStateWeatherScalars` | Fixed indices 13–16 for weather + income | **Indices unchanged** if weather stays 13–15 and me_income 16; **recompute** expected values for scalar 9 (and any row using share 16) under ego if tests pin absolute floats |
| [`test_hp_belief.py`](../../test_hp_belief.py) | 300–410 | `encode_state(..., observer=0)` | P0 block tests | Add ego swap tests: same state, `observer=0` vs `1` places units in 0:14 vs 14:28 |
| [`tests/test_env_buffer_reuse_golden.py`](../../tests/test_env_buffer_reuse_golden.py) | 33–46 | obs digest | Reproducibility of obs bytes | Stays self-consistent; **no** frozen vectors — encoder change does not by itself break `pre == post` |
| [`tests/test_belief_diff_early_exit.py`](../../tests/test_belief_diff_early_exit.py) | 308 | `_get_obs(0)` | P0 | Align with test env’s learner seat |

**`MASKED_PLAYER`:** **not present** in [`rl/env.py`](../../rl/env.py); no constant by that name in repo search.

**Approximate table row count:** **33** site rows (file/symbol level); some rows bundle multiple line ranges in one file.

---

## 3. Game log schema implications

**Existing (engine) fields stay:** `winner ∈ {0,1}` = engine seat index; `p0_co` / `p1_co` (and `*_id`), `co_p0` / `co_p1` names, `property_count[0/1]`, `captures_*` by engine player, `opening_player`, etc.

**Add:**

- **`me` ∈ {0,1}** — engine seat the **learner** (trained policy) controlled for this episode.
- **`log_schema_version` bump** (e.g. 1.8 → 1.9) on the same write path as in [`rl/env.py`](../../rl/env.py) lines 1484–1491.

**Recommendation:** keep **all** existing fields in **engine-seat** semantics. **Do not** rewrite `winner` or CO columns to me/enemy — that would break [`analysis/co_ranker.py`](../../analysis/co_ranker.py), [`analysis/co_h2h.py`](../../analysis/co_h2h.py), and every notebook that already joins on `p0_co_id` / `winner`. Analysts use:

- `agent_won = (df["winner"] == df["me"])` when `me` is present;
- `agent_won = (df["winner"] == 0)` for legacy rows where the agent was always P0.

**Rationale:** engine seat is the stable key into `GameState` and the replay format; `me` is a single integer overlay for “who was learning/acting” without aliasing away `winner` meaning.

**Deprecate / clarify** `agent_plays` (today hardcoded `0` at line 1442): replace with `me` or set `agent_plays = me` for backward-friendly dashboards.

---

## 4. BC training data implications (MASTERPLAN §9)

**Row shape for ego-centric + both seats (same map):**

- **`me` ∈ {0,1}** for each row (human or learner seat for that decision).
- **`spatial` / `scalars`:** from `encode_state(state_before_action, observer=me, belief=…)` (STD: `belief=None` in offline; fog off-bundle).
- **`action_idx` / `action_mask`:** from the engine’s legal set for that **turn**; flat index is unchanged, but the **mask** is for `state.active_player == me` on that row.
- **P0 row (`me=0`):** same tuple family as today’s human_demos, but scalars/ spatial use me/enemy with observer 0.
- **P1 row (`me=1`):** same, with observer 1; **me** units sit in 0:14.

**Train on both rows of the same game?** **Yes** for BC (no on-policy issue). **Correlation caveat:** not independent; MASTERPLAN §9 recommends seat alternation across **independent** episodes for **actor** batches — for BC, document oversampling or batch construction if needed.

**One-line spec for [`scripts/train_bc.py`](../../scripts/train_bc.py):**  
Each JSONL row **must** include integer **`me`**, and the stored **`spatial`/`scalars`** must match **`encode_state(..., observer=me)`**; rows missing `me` default to `0` for legacy human_demos, or the script re-encodes from a replay with explicit seat.

**Ingestion:** update [`tools/human_demo_rows.py`](../../tools/human_demo_rows.py) filter from “only P0” to “both seats” when the trace supports it; [`server/play_human.py`](../../server/play_human.py) continues to log **only human seat 0** unless the UI is extended.

---

## 5. Test plan (proposed)

1. **Swap / asymmetry (channel ordering):** build a state with at least one **P0** unit and no P1 (or a known pattern). Assert `encode_state(..., observer=0)[spatial unit me block]` has signal in **0:14**; `encode_state(..., observer=1)` has that unit’s presence in **14:28** (enemy block).
2. **Mirror / paired states (invariance):** *Not* `encode(s)==encode(s')` for arbitrary “swap” without care — only holds if you **construct** `s_b` as `s_a` with P0↔P1 *and* you compare `encode_state(s_a, observer=0)` to `encode_state(s_b, observer=1)` with symmetric maps/COs. Document: **invariance test** = equality under that **full board swap** + same `luck_seed` / mirror symmetry; otherwise assert **expected permutation** of channels, not byte equality of unrelated states.
3. **Scalar 9 (my turn):** for fixed `state`, `scalars[9] == 1.0` iff `state.active_player == observer`.
4. **Scalar 16 (me income share):** spot-check against `state.count_income_properties(observer) / n_income` on a few boards.
5. **HP (STD, belief off):** unchanged numeric HP in lo/hi for all units; optional parity with pre-ego for `belief is None` paths.
6. **Regression:** [`tests/test_encoder_equivalence.py`](../../tests/test_encoder_equivalence.py) + [`tests/fixtures/encoder_equivalence_pre_restart.npz`](../../tests/fixtures/encoder_equivalence_pre_restart.npz) are the **restart gate** for any future encoder change; regen only as in [`tests/fixtures/encoder_equivalence_README.md`](../../tests/fixtures/encoder_equivalence_README.md) (see §7).

---

## 6. Migration ordering (implementing composer)

1. **Encoder** + unit tests (§5) + docstrings in [`rl/encoder.py`](../../rl/encoder.py) — **landed** (70ch ego-centric; see Implementation status).
2. **Encoder equivalence** harness: `encoder_equivalence_pre_restart.npz` is the **current** frozen tensor contract (name is historical); replace/regenerate only on intentional encoder edits per [`tests/fixtures/encoder_equivalence_README.md`](../../tests/fixtures/encoder_equivalence_README.md). Old **policy** zips are invalid across channel/restart boundaries regardless of filename.
3. **`AWBWEnv`:** learner seat + ` _get_obs(observer=learner)` + opponent loop / reward / Φ in learner frame (large change — may land in the same diff as seat-balanced-actor or immediately after).
4. **`log_schema_version` + `me` field** in [`rl/env.py`](../../rl/env.py) `_log_finished_game`.
5. **Tooling & demos:** `play_human`, `human_demo_rows`, `verify_observation_encoding`, `phi_smoke` / eval scripts as needed.
6. **BC** [`scripts/train_bc.py`](../../scripts/train_bc.py) + demo JSONL contract.
7. **Analysis docs** (MASTERPLAN / seat_measurement / player_seats) for `agent_won` with `me`.

Encoder-only PR without env seat plumbing produces **policy training inconsistency** (P0 step still, P0-ego obs) unless the env immediately passes `observer=0` (no visible change) — the **value** of ego is unlocked when `observer` follows the **learner**.

---

## 7. Risks / unknowns

- **Reward / Φ / terminal return** are **P0-learner-shaped** today ([`rl/env.py`](../../rl/env.py) P1 reward negation, `_compute_phi` in P0 frame). Ego **encoding** alone does not fix credit assignment; **seat-balanced** training needs an explicit spec for return signs and whether the critic uses raw `V` or flipped on opponent seat (MASTERPLAN §9).
- **Opponent as P1** is wired throughout (`_run_policy_opponent`, wall timers `wall_p0_s` / `wall_p1_s` naming). Generalizing to “learner may be P1” touches **more** than [`rl/encoder.py`](../../rl/encoder.py).
- **`play_human` bot** path ([`server/play_human.py`](../../server/play_human.py) line 314) uses `encode_state(state)` without `observer=1` — with ego fix, the bot should use `observer` = bot seat.
- **MCTS** ([`rl/mcts.py`](../../rl/mcts.py)) already uses `active_player` as observer; ego makes the **spatial** view match “I am the mover” in me/enemy form — **valuable** but changes tensor bytes vs any saved MCTS snapshot tests.
- **BC rows** in the wild: old rows are P0-ego (`observer=0` after refactor) or P0-fixed-blocks (before refactor). Re-ingest or version **`encoder_version`** / schema in each demo row.
- **Encoder equivalence harness** ([`tests/test_encoder_equivalence.py`](../../tests/test_encoder_equivalence.py)): fails when `encode_state` **output** drifts from [`tests/fixtures/encoder_equivalence_pre_restart.npz`](../../tests/fixtures/encoder_equivalence_pre_restart.npz) — **expected** after any intentional encoder change; treat as the **restart gate** (regen policy: [`tests/fixtures/encoder_equivalence_README.md`](../../tests/fixtures/encoder_equivalence_README.md)).

**STD / fog:** this map does not extend belief POMDP; HP remains as today for STD.

---

## Verification checklist (this spec)

- **Line numbers** were checked against `Read` / `Grep` of the files above; implementers should re-run before merge.
- **Scalar count** remains **17** — **no** `N_SCALARS` bump, only **semantic** remapping of indices 0–7, 9–11, 16; **8, 12–15** unchanged (turn, tier, weather).
- **Harness:** [`tests/test_encoder_equivalence.py`](../../tests/test_encoder_equivalence.py) pins **70**-channel ego-centric output to [`tests/fixtures/encoder_equivalence_pre_restart.npz`](../../tests/fixtures/encoder_equivalence_pre_restart.npz); a **new** frozen baseline (per README) is required when the tensor contract changes again, invalidating tied checkpoints.

---

## Report-back summary (for parent agent)

| Item | Value |
|------|--------|
| **Doc path** | [`docs/restart_arch/ego_centric_refactor_map.md`](ego_centric_refactor_map.md) (this file) |
| **Refactor table** | **~33** site rows (multi-line ranges grouped) |
| **Non-obvious / high impact** | (1) [`rl/mcts.py`](../../rl/mcts.py) already passes `active_player` as `observer` — ego aligns tensor with mover; (2) [`server/play_human.py`](../../server/play_human.py) bot used default `encode_state` (P0 view) while acting as P1; (3) full **seat-balance** needs env reward/opponent generalization, not encoder alone; (4) **Φ / level** shaping is P0-framed; (5) [`tools/human_demo_rows.py`](../../tools/human_demo_rows.py) filters P0-only today — BC both-seats needs filter + `me`. |
| **Biggest risk** | **Env** (`step`, P1 auto-loop, return sign, `_compute_phi`) — larger than `rl/encoder.py` in lines of effect. |
| **Encoder equivalence** | **Active** — `encoder_equivalence_pre_restart.npz` + README; any tensor change should fail tests until a reviewed regen. |

*This document includes 2026-04 implementation status for the encoder tensor; remaining rows cover env / logging / BC follow-ups. No Python changes in the doc-only edit that added that status.*
