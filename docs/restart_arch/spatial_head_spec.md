# Spatial (factored) policy head — design spec (STD, no fog)

**Status:** The factored spatial policy head is **implemented** in `rl/network.py` (`AWBWNet`: fused-trunk 1×1 convs, scatter into flat logits) with observation channels defined in `rl/encoder.py` (`N_SPATIAL_CHANNELS`; authoritative expression lives there). This document is the **design reference** for layout, index bands, and validation intent; behavior matches those modules.

**Goal:** (Historical motivation.) Replace the monolithic `Linear(hidden_size, ACTION_SPACE_SIZE)` policy head with AlphaZero-style **per-action-type 1×1 convolutions** on full-resolution trunk features, plus **scalar logits** for non-spatial flat indices, then **reproject** to a flat `(B, 35_000)` tensor so `MaskedPPO`, `rl/env.py` decoding, and masks stay byte-for-byte compatible.

**References (authoritative):**

- Flat encoding & mask: `rl/env.py` (`_action_to_flat`, `_get_action_mask`, offsets). **Duplicate flat constants** (for import hygiene): `rl/network.py` — must match `rl/env.py`.  
- `ACTION_SPACE_SIZE`, `_MOVE_OFFSET` (`1818`), `_BUILD_OFFSET` (`10_000`), `_REPAIR_OFFSET` (`3500`), collision merge `900…902`, scatter slices: `rl/network.py`.  
- `N_SPATIAL_CHANNELS` (**77**; sum of planes): `rl/encoder.py` (authoritative). Trunk: `rl/network.py` (`AWBWNet`).
- Engine action types: `engine/action.py`.  
- `UnitType` cardinality for BUILD: `engine/unit.py` (`UnitType`, `len(UnitType) == 27`).

---

## 1. Action space inventory (current)

Constants and helpers live in `rl/env.py` (e.g. `_ENC_W = 30`, `_ATTACK_OFFSET`, `_MOVE_OFFSET`, `_BUILD_OFFSET`, `_REPAIR_OFFSET`, `_UNLOAD_OFFSET`, `_N_UNIT_TYPES`) and duplicated flat layout constants in `rl/network.py` (`ACTION_SPACE_SIZE`, same offsets).

**Mask construction:** `_get_action_mask` iterates `get_legal_actions(state)`, computes `idx = _action_to_flat(action)`, and sets `mask[idx] = True` for in-range indices (`rl/env.py`, `_get_action_mask`). Optional curriculum: `_strip_non_infantry_builds` clears non-infantry BUILD slots in-place (`rl/env.py`).

| Action type | Index range (inclusive) | Per-tile? | Sub-args / encoding | Slot count | Mask construction reference |
|-------------|-------------------------|-----------|---------------------|------------|------------------------------|
| `END_TURN` | `0` | No (scalar) | — | 1 | `_get_action_mask` via `_action_to_flat` |
| `ACTIVATE_COP` | `1` | No | — | 1 | same |
| `ACTIVATE_SCOP` | `2` | No | — | 1 | same |
| `SELECT_UNIT` (SELECT stage) | `3` … `902` | Yes (tile) | `3 + r*_ENC_W + c` from **`action.unit_pos`** on `ActionStage.SELECT` | `900` | same |
| `ATTACK` | `900` … `1799` | Yes (target tile) | `_ATTACK_OFFSET + r*_ENC_W + c` | `900` | same |
| *(index collision)* | `900` … `902` | — | **Overlaps** last three SELECT-stage tiles `(29,27),(29,28),(29,29)` with first three `ATTACK` targets `(0,0),(0,1),(0,2)` — same flat integer means different actions in different stages. | 3 | Decoder disambiguates via **legal list**; policy must combine head outputs consistently (§3.4). |
| `CAPTURE` | `1800` | No | single index `_CAPTURE_IDX` | 1 | same |
| `WAIT` | `1801` | No | `_WAIT_IDX` | 1 | same |
| `LOAD` | `1802` | No | `_LOAD_IDX` | 1 | same |
| `JOIN` | `1803` | No | `_JOIN_IDX` | 1 | same |
| `DIVE_HIDE` | `1804` | No | `_DIVE_HIDE_IDX` | 1 | same |
| *(unused reservation)* | `1805` … `1809` | — | **No `_action_to_flat` branch** — these indices are never emitted by the encoder | 5 | Always false in any correct mask |
| `UNLOAD` | `1810` … `1817` | No (8-way discrete) | `_UNLOAD_OFFSET + slot*4 + direction`; `direction` ∈ {0..3} N/S/W/E; `slot` derived from `unit_type` (`int(unit_type) & 1`) | 8 | same |
| `SELECT_UNIT` (MOVE stage) | `1818` … `2717` | Yes (destination tile) | `_MOVE_OFFSET + r*_ENC_W + c` from **`move_pos`**; `_MOVE_OFFSET` = `1818` = `_UNLOAD_OFFSET + 8` (`rl/network.py`, `rl/env.py`) | `900` | same |
| `REPAIR` | `3500` … `4399` | Yes (ally tile) | `_REPAIR_OFFSET + r*_ENC_W + c` | `900` | same |
| `BUILD` | `10000` … `34299` | Yes × unit type | `_BUILD_OFFSET + (r*_ENC_W+c)*_N_UNIT_TYPES + int(UnitType)`; factory tile `(r,c)`; `_N_UNIT_TYPES = 27` | `900 × 27 = 24300` | same (+ optional `_strip_non_infantry_builds`) |
| *(padding / unused)* | `34300` … `34999` | — | `ACTION_SPACE_SIZE = 35_000` leaves tail unused (`10000 + 900*27 - 1 = 34299`) | `700` | Always false |

**Totals:** `35_000` flat slots; **used** primary bands = `3 + 900 + 900 + 5 + 8 + 900` (MOVE) `+ 900` (REPAIR) `+ 24300` (BUILD) = `28_016`, plus `5` never-used gap `1805…1809`, a large **inactive** span `2718…3499` before `_REPAIR_OFFSET`, and `700` tail padding `34300…34999`; remaining indices are permanently inactive.

**`ACTION_SPACE_SIZE`:** `35_000` (`rl/network.py`, `ACTION_SPACE_SIZE = 35_000`).

---

## 2. Factored head layout (implemented)

### 2.1 Trunk branch point

`AWBWNet` (`rl/network.py`) permutes the observation to `(B, N_SPATIAL_CHANNELS, 30, 30)` with **N_SPATIAL_CHANNELS = 77** (see `rl/encoder.py`), runs `stem` → **10×** `_ResBlock128` at full **30×30** resolution → trunk tensor **`x`**, shape `(B, 128, 30, 30)`. Scalars are projected to **16** planes (`scalar_to_plane`, `N_SCALARS = 16`) and broadcast, then concatenated: **`xf = cat([x, sp], dim=1)`** → `(B, 144, 30, 30)` where **144 = TRUNK_CHANNELS + SCALAR_PLANES** (`FUSED_CHANNELS`).

`adaptive_avg_pool2d(xf, (1,1))` yields **`g`**, shape `(B, 144)` — this vector drives **`value_head`** and **`linear_scalar_policy`**; the same **`xf`** feeds all policy 1×1 convs (`conv_select`, `conv_move`, `conv_attack`, `conv_repair`, `conv_build`).

(The SB3 `AWBWFeaturesExtractor` in the same file still uses a different **`fc` MLP** on pooled `xf`; that path is for legacy SB3 integration, not the `AWBWNet` policy/value just described.)

### 2.2 Per-head outputs (behavior-equivalent to current flat layout)

| Group | Head type | Conv / MLP | Output before reprojection | Channel / logit meaning |
|-------|-----------|------------|----------------------------|-------------------------|
| Powers + end turn | Scalar | `linear_scalar_policy` on **`g`**: `Linear(FUSED, 16)` then `[:, :3]` (see `AWBWNet`) | `(B, 3)` | maps to flat `0,1,2` in fixed order: `[END_TURN, COP, SCOP]` |
| SELECT stage | Spatial | `conv_select`: `Conv2d(FUSED, 1, 1)` | `(B, 1, 30, 30)` | logit at `(r,c)` → flat `3+r*30+c` |
| MOVE stage | Spatial | `conv_move`: `Conv2d(FUSED, 1, 1)` | `(B, 1, 30, 30)` | logit at `(r,c)` → flat `_MOVE_OFFSET + r*30+c` with `_MOVE_OFFSET = 1818` (`900` slots) |
| `ATTACK` target tile | Spatial | `conv_attack`: `Conv2d(FUSED, 1, 1)` | `(B, 1, 30, 30)` | → flat `900+r*30+c` |
| `BUILD` | Spatial | `conv_build`: `Conv2d(FUSED, 27, 1)` | `(B, 27, 30, 30)` | channel `k` at `(r,c)` → flat `10000+(r*30+c)*27+k` (`k` = `int(UnitType)`) |
| `REPAIR` target tile | Spatial | `conv_repair`: `Conv2d(FUSED, 1, 1)` | `(B, 1, 30, 30)` | → flat `3500+r*30+c` |
| CAPTURE / WAIT / LOAD / JOIN / DIVE_HIDE | Scalar | fused `Linear(FUSED, 16)` → `s_misc` slice | `(B, 5)` | scatter to `1800…1804` in fixed order |
| `UNLOAD` | Scalar | same `Linear` → `s_unl` | `(B, 8)` | maps to `1810…1817`: index `1810 + i` for `i ∈ [0,7]` matching `_UNLOAD_OFFSET + slot*4 + dir` |

**Channel ordering for BUILD:** channel index `k` must match `engine.unit.UnitType` integer values `0…26` (same as `_action_to_flat`).

**Spatial layout:** row-major `r ∈ [0,29]`, `c ∈ [0,29]`, consistent with `_action_to_flat` and `encode_state` (`rl/encoder.py`, `GRID_SIZE = 30`).

### 2.3 Parameter count vs. current

| Component | Formula | Approx. params |
|-----------|---------|----------------|
| **Dense baseline (for comparison)** | e.g. `Linear(256, 35000)` | order **~9M** |
| **Spatial 1×1 on `xf`** | `Conv2d(144,1,1)` ×4 (select, move, attack, repair) + `Conv2d(144,27,1)` (build) | `4×(144+1) + (144×27+27)` = **4,199** |
| **Scalar policy** | `Linear(144, 16)` = `3+5+8` logits | `144×16 + 16` = **2,320** |
| **Value head** | `Linear(144, 256)` + `Linear(256, 1)` (default `hidden_size`) | (listed separately from policy mapping) |

Notes:

- **144** = `FUSED_CHANNELS` (global-pooled `xf`, **not** `128+17` concat).
- Order-of-magnitude: policy mapping is **~6.5k** parameters vs. a single large `Linear` onto `35_000` flats.

---

## 3. Index reprojection scheme

### 3.1 Recommendation: flat-output factored head (ship-first)

**Chosen approach:** Compute factored logits, **`scatter` / indexed writes into `logits_flat[B, 35_000]`**, initialize with `-inf` (or a large negative constant), then apply the existing `masked_fill(~action_mask, -inf)` (`rl/network.py`, `forward`).

**Rationale:**

- `AWBWEnv.step`, `action_masks`, PPO rollout storage, and opponent policies all assume **integer actions in `[0, ACTION_SPACE_SIZE)`** (`rl/env.py`, `step`; `spaces.Discrete(ACTION_SPACE_SIZE)`).
- No change to `_flat_to_action` / `_get_action_mask` / `tests` that assert mask shape.
- Behavior-equivalent: same legal indices, same decoding.

**Non-choice (more invasive):** Structured action tuples/distributions would require refactoring SB3 MaskablePPO integration, rollout buffers, and env contract — out of scope for this restart wave.

### 3.2 Pseudocode — `forward`

```text
# Inputs: spatial (B,30,30,C), scalars (B,16), action_mask (B,35000) optional
# C = N_SPATIAL_CHANNELS = 77 (see rl/encoder.py for authoritative sum of planes)

x = permute(spatial)                     # (B, C, 30, 30)
x = stem(x); ... trunk blocks ...        # (B, 128, 30, 30)
sp = linear_scalars_to_16planes(scalars).view(B,16,1,1).expand(-1,-1,30,30)
xf = cat([x, sp], dim=1)                 # (B, FUSED_CHANNELS) = (B, 144, 30, 30)

# --- Policy branch (spatial), on xf (same as AWBWNet) ---
L_select = conv_1x1_select(xf)           # (B, 1, 30, 30)
L_move   = conv_1x1_move(xf)            # (B, 1, 30, 30)  -> _MOVE_OFFSET..+899 (1818..2717)
L_attack = conv_1x1_attack(xf)         # (B, 1, 30, 30)
L_repair = conv_1x1_repair(xf)         # (B, 1, 30, 30)
L_build  = conv_1x1_build(xf)         # (B, 27, 30, 30)

g = adaptive_avg_pool2d(xf, (1,1)).flatten(1)   # (B, 144)
s_all = linear_scalar_policy(g)         # (B, 16)  -> 3 + 5 + 8
s_pow = s_all[:, :3]                    # (B, 3)   -> 0,1,2
s_misc = s_all[:, 3:8]                  # (B, 5)   -> 1800..1804
s_unl  = s_all[:, 8:16]                 # (B, 8)   -> 1810..1817

# --- Value branch (AWBWNet): same g as above ---
v = value_head(g).squeeze(-1)            # value_head: Linear(144, hidden) -> ReLU -> Linear(hidden,1)

# --- Reproject to flat logits ---
logits = full((B, 35000), -inf)

# Build per-index contributions from SELECT (indices 3..902) and ATTACK (900..1799)
# Row-major flatten: flat_s[s] = 3 + s for s in 0..899  where s = r*30 + c
sel_contrib = L_select.view(B, 900)    # sel_contrib[b,s] feeds flat index 3+s
atk_contrib = L_attack.view(B, 900)    # atk_contrib[b,s] feeds flat index 900+s

# Default: overlap-free regions — SELECT-only 3..899, ATTACK-only 903..1799
logits[:, 3:900] = sel_contrib[:, 0:897]
logits[:, 903:1800] = atk_contrib[:, 3:900]

# Collision band 900..902: three flats each receive BOTH semantic contributions.
# Use sum so a single Linear-like scalar can be recovered when one branch is irrelevant;
# masking removes illegal mass; both branches may need to learn near-neutral logits off-stage.
logits[:, 900] = sel_contrib[:, 897] + atk_contrib[:, 0]
logits[:, 901] = sel_contrib[:, 898] + atk_contrib[:, 1]
logits[:, 902] = sel_contrib[:, 899] + atk_contrib[:, 2]

logits[:, 3500:4400] = L_repair.view(B, -1)

# MOVE: 900 destination tiles: [_MOVE_OFFSET, _MOVE_OFFSET + 900) = 1818..2718 (Python slice 1818:2718)
logits[:, _MOVE_OFFSET : _MOVE_OFFSET + 900] = L_move.view(B, -1)

# BUILD: vectorized scatter — logits[:, 10000 + s*27 + k] = L_build[b,k,r,c] with s=r*30+c
L_build_flat = L_build.permute(0, 2, 3, 1).reshape(B, 900, 27)
# implement with expand + add or index_put; _BUILD_OFFSET = 10000

# Scalar scatters (matches rl/network.py row-for-row)
logits[:, 0:3] = s_pow
logits[:, 1800] = s_misc[:,0]   # _CAPTURE_IDX
logits[:, 1801] = s_misc[:,1]   # _WAIT_IDX
logits[:, 1802] = s_misc[:,2]   # _LOAD_IDX
logits[:, 1803] = s_misc[:,3]   # _JOIN_IDX
logits[:, 1804] = s_misc[:,4]   # _DIVE_HIDE_IDX
logits[:, 1810:1818] = s_unl

# Indices 1805..1809, 2718..3499, 34300..34999 remain -inf

if action_mask is not None:
    logits = masked_fill(logits, ~action_mask, -inf)

return logits, value
```

**Ordering constraints (must match `rl/env.py`):**

- `s_misc` order must be `[CAPTURE, WAIT, LOAD, JOIN, DIVE_HIDE]` → `[1800,1801,1802,1803,1804]`.
- `s_unload[i]` → flat `1810 + i` where `i = slot*4 + direction` with the same slot/direction convention as `_action_to_flat`.

**Note on BUILD vectorization:** Implement tile-major scatter without Python loops over `k` in the hot path.

### 3.3 Equivalence intuition

For ATTACK-only flats `903..1799`, `logits[b, 900+r*30+c] == L_attack[b,0,r,c]` when `900+r*30+c >= 903`. For SELECT-only flats `3..899`, `logits[b, 3+r*30+c] == L_select[b,0,r,c]`. The mask enables a sparse legal subset; illegal tiles stay masked.

### 3.4 Collision band `900..902` and the dense baseline

The legacy `Linear(256, 35000)` assigns **one** weight row per flat index. For indices `900..902`, that row must serve **both** possible semantics (corner SELECT tiles vs. north-edge ATTACK targets) depending on game stage — already an ambiguous encoding resolved only by **which actions are legal**.

The factored head uses **two** spatial maps; the recommended merge is **`logits[f] = sel_part[f] + atk_part[f]`** on the collision band so the model can learn stage-appropriate logits (with masking suppressing illegal semantics). A tiny learned `Linear(2,1)` **per collision flat** (~9 extra params) is an alternative if sum proves unstable in training.

### 3.5 MOVE-stage `SELECT_UNIT` (flat geometry)

In MOVE stage, `_action_to_flat` encodes **`move_pos`** at **`_MOVE_OFFSET + r*_ENC_W + c`** with **`_MOVE_OFFSET = 1818`** (`rl/env.py`, `rl/network.py`). The MOVE band is **`900` consecutive indices** (`1818` … `2717`), distinct from SELECT-stage `3…902`. `AWBWNet` writes `conv_move` logits into `logits[:, _MOVE_OFFSET : _MOVE_OFFSET + 900]`. `_flat_to_action` in MOVE stage resolves `flat_idx` in that range to a legal `SELECT_UNIT` with matching `move_pos` by scanning the legal list. Per-destination policy mass is **not** collapsed to a single slot (unlike pre–move-encoding designs).

---

## 4. Mask plumbing

- **After reprojection:** `logits` has shape `(B, ACTION_SPACE_SIZE)` identical to today (`rl/network.py`). The existing line `logits = logits.masked_fill(~action_mask, float("-inf"))` remains valid.
- **`rl/env.py` mask construction:** `_get_action_mask` does **not** need changes for the factored head, because legality is still defined by enumerating engine actions and `_action_to_flat`.
- **`_strip_non_infantry_builds`:** Still operates on BUILD flat indices; **no change**.
- **Unused / padding indices:** Remain permanently masked false; keeping their logits at `-inf` is consistent.

**Pitfall (pre-existing, not introduced by this head):** If any code path assumed “dense” legal coverage of the whole flat range, it was already false (gap `1805–1809`, unused span `2718–3499` before `_REPAIR_OFFSET`, tail padding). The factored head must **not** fill those with finite logits in a way that could be sampled — leaving them at `-inf` is correct.

---

## 5. Validation tests (proposed)

1. **Scatter index sanity (random tensors):** Generate random `L_select, L_move, L_attack, L_repair, L_build` and scalar heads; run reprojection; assert **non-overlap** regions match (SELECT `3..899`, ATTACK `903..1799`, MOVE `1818..2717` vs `_REPAIR_OFFSET`/`_BUILD_OFFSET` bands, BUILD); assert **collision band** `logits[900]==sel[29,27]+atk[0,0]` (and analogs for 901, 902) per §3.2.

2. **Full flat span:** After reprojection, assert only the **documented used ranges** can differ from `-inf` when using controlled finite inputs; gap `1805–1809`, the **pre-REPAIR** gap `2718–3499`, and tail `34300–34999` stay `-inf`.

3. **Masked sampling:** With a random mask where only 1–5 entries are `True`, draw many categorical samples from `softmax(logits_masked)`; never observe an index outside legal positions.

4. **BUILD identity:** Set `L_build[b, ut, r, c] = K` (known constant) for one `(ut,r,c)`, all else `-inf` or very negative; reproject; assert `logits[b, 10000+(r*30+c)*27+ut] == K`.

5. **Parity with brute-force linear (optional regression):** For a **single fixed** small MLP that produces an equivalent full `Linear(256,35000)` on synthetic hidden vectors, compare against factored head **only if** someone constructs a weight tying proof — not required for training, but useful if porting old checkpoints (likely **not** feasible — expect **fresh init** for this restart).

---

## 6. Risks / unknowns

1. **`UNLOAD` slot encoding (`rl/env.py`, `_action_to_flat`):** `slot = int(action.unit_type) & 1` collapses cargo identity to two buckets. Multiple distinct legal `UNLOAD` actions **may share** the same flat index if they differ only by cargo not distinguished by that bit — **decoder** `_flat_to_action` resolves by scanning legals and returning the **first** match. This is a **pre-existing ambiguity**; the factored head does not worsen it, but **scalar `UNLOAD` logits cannot distinguish collisions** beyond what the flat space already could.

2. **MOVE vs SELECT:** The spatial heads use separate maps (`conv_select` vs `conv_move`); the env/mask only activates the band that matches `action_stage` — see §3.5 and `_MOVE_OFFSET`.

3. **SELECT / ATTACK overlap `900..902`:** See §3.4 — reprojection must **combine** contributions; naive “write SELECT block then overwrite with ATTACK block” is **wrong**.

4. **`ACTIVATE_COP` / `ACTIVATE_SCOP` without positions:** Scalar head is mandatory; pooling+scalars is appropriate.

5. **Memory at batch size `B`:** Reprojected `logits` remains `(B, 35000)` float32 ≈ **`B × 140 KiB`**. The observation’s spatial tensor is `(B, 30, 30, 77)`; activations through the **10** residual blocks are dominated by `(B, 128, 30, 30)` and fused **`xf`** `(B, 144, 30, 30)` in `AWBWNet` — the same as any full-resolution policy forward; plan VRAM for **large** `B` accordingly.

6. **Unused indices:** Gap `1805–1809` and span `2718–3499` (between MOVE band end and `_REPAIR_OFFSET`); no engine action maps there without an encoding change.

---

## Appendix: line references for “current” claims

| Claim | File:lines |
|-------|------------|
| Offsets `_ATTACK_OFFSET`, `_CAPTURE_IDX`, `_WAIT_IDX`, … `_UNLOAD_OFFSET`, `_BUILD_OFFSET`, `_REPAIR_OFFSET`, `_N_UNIT_TYPES` | `rl/env.py` approx. 161–177 |
| `_action_to_flat` cases | `rl/env.py` approx. 180–251 |
| `_get_action_mask` | `rl/env.py` approx. 284–306 |
| `ACTION_SPACE_SIZE`, `_MOVE_OFFSET`, scatter into flat logits, collision merge | `rl/network.py` (imports through forward) |
| `GRID_SIZE`, `N_SPATIAL_CHANNELS` (**77**), `N_SCALARS` (**16**) | `rl/encoder.py` (channel sum / definitions) |
| RL `ActionType` set | `engine/action.py` approx. 57–80, 93–111 |

---

*End of spec.*
