# Action space inventory (STD ranked, no fog)

Ground-truth reference for the RL **35,000**-wide discrete action space. Sources: `rl/env.py`, `engine/action.py`, `engine/game.py`, `engine/unit.py`, `rl/network.py`. **Inventory only — no design.**

Encoding grid width is fixed at **`_ENC_W = 30`** regardless of map size (`rl/env.py:161–163`). Tile linear index `tile_idx = r * 30 + c` for `r, c ∈ [0, 29]` (full 30×30 = **900** tiles).

`UnitType` is `engine/unit.py:19–46` (**27** members, values `0..26`). `rl/env.py` uses `_N_UNIT_TYPES = len(UnitType)` (`rl/env.py:176–177`).

---

## 1. Master table — every flat index range

| Index range (inclusive) | Action type | Per-tile? | Sub-args | Encode formula (`_action_to_flat`, `rl/env.py:180–251`) | Mask construction |
|---|---|---|---|---|---|
| **0** | `END_TURN` | No | — | `0` | `rl/env.py:284–306` |
| **1** | `ACTIVATE_COP` | No | — | `1` | same |
| **2** | `ACTIVATE_SCOP` | No | — | `2` | same |
| **3 – 902** | `SELECT_UNIT` | Yes (`unit_pos` tile) | `(r,c) = unit_pos` | `3 + r * 30 + c` | same |
| **900 – 1799** | `ATTACK` | Yes (defender / seam **target** tile) | `(r,c) = target_pos` | `900 + r * 30 + c` | same |
| **1800** | `CAPTURE` | No* | *terminator; destination tile is `move_pos` on the `Action`, not in the flat index* | `1800` | same |
| **1801** | `WAIT` | No | — | `1801` | same |
| **1802** | `LOAD` | No | — | `1802` | same |
| **1803** | `JOIN` | No | — | `1803` | same |
| **1804** | `DIVE_HIDE` | No | — | `1804` | same |
| **1805 – 1809** | **UNUSED** | — | — | *(no branch in `_action_to_flat`)* | never set by engine |
| **1810 – 1817** | `UNLOAD` | Cardinal × “slot” | See §6 | `_UNLOAD_OFFSET + slot * 4 + dir` | same |
| **1818 – 3499** | **UNUSED** | — | — | — | never set |
| **3500 – 4399** | `REPAIR` | Yes (`target_pos` tile) | Ally tile `(r,c)` | `3500 + r * 30 + c` (`_REPAIR_OFFSET = 3500`, `rl/env.py:174–175,244–248`) | same |
| **4400 – 9999** | **UNUSED** | — | — | — | never set |
| **10000 – 34299** | `BUILD` | Yes (factory **tile**) × unit type | `(r,c) = move_pos`, `unit_type` | `10000 + (r * 30 + c) * 27 + int(unit_type)` | same; optional strip `rl/env.py:308–321`, `984–986` |
| **34300 – 34999** | **UNUSED PADDING** | — | — | — | never set |

\*CAPTURE is “per-tile” only in the sense that the engine attaches `move_pos`; the flat index is a single scalar.

### 1.1 Numeric overlap (SELECT vs ATTACK)

- `SELECT_UNIT` occupies **3..902**; `ATTACK` occupies **900..1799**. Indices **900, 901, 902** are valid outputs of **both** formulas.
- At runtime, `get_legal_actions` is dispatched by `state.action_stage` (`engine/action.py:635–643`), so only one stage’s generator runs; **mask bits are never merged across stages for the same step**.
- **Interpretation of integers 900–902 is stage-dependent** (unit selection vs attack target).

### 1.2 MOVE stage — `move_pos` not encoded in the flat index (verified)

- In `ActionStage.MOVE`, legals are `Action(ActionType.SELECT_UNIT, unit_pos=unit.pos, move_pos=pos)` (`engine/action.py:775–777`).
- Engine applies the destination via `action.move_pos` when `action_stage == MOVE` (`engine/game.py:458–467`).
- **`_action_to_flat` for `SELECT_UNIT` uses only `action.unit_pos`**, not `move_pos` (`rl/env.py:191–194`).
- **Consequence:** for a fixed selected unit, **every legal MOVE destination shares the same flat index** (`3 + r0 * 30 + c0` for the unit’s start tile `(r0,c0)`).
- **`_flat_to_action`** resolves collisions by scanning `legal` and returning the **first** `Action` with matching flat (`rl/env.py:263–267`). Destination choice is therefore **order-dependent** (engine emits MOVE legals in **sorted `(row, col)`** order, `engine/action.py:775–777`).
- **Sanity check run (2026-04-23):** `AWBWEnv.reset(seed=0)` then stepped until `action_stage == MOVE` with 4 legal moves; `_action_to_flat` yielded **one unique value (187) repeated four times**.

---

## 2. Action mask construction

**Function:** `_get_action_mask(state, out=None, legal=None)` — `rl/env.py:284–306`.

**Steps:**

1. Allocate or zero a boolean array of shape **`(ACTION_SPACE_SIZE,)`** where `ACTION_SPACE_SIZE = 35_000` (`rl/network.py:20`, `rl/env.py:295–299`).
2. If `legal is None`, set `legal = get_legal_actions(state)` (`engine/action.py:635`; imported in `rl/env.py:30`).
3. For each `action` in `legal`, compute `idx = _action_to_flat(action)` and, if `0 <= idx < ACTION_SPACE_SIZE`, set `mask[idx] = True` (`rl/env.py:302–305`).

**Conversion engine → flat:** strictly **`_action_to_flat`** (`rl/env.py:180–251`); there is no separate scatter index.

**Incremental vs vectorized:** nested loop over Python `legal` list; boolean **scatter** into a pre-zeroed mask (or in-place fill on reused buffer).

**Call sites:**

- `AWBWEnv.action_masks` → `_get_action_mask` (`rl/env.py:974–987`).
- Opponent policy path also uses `_get_action_mask` (`rl/env.py:1281–1282`).
- Optional in-place buffer: `self._action_mask_buf` (`rl/env.py:546–547, 982–983`).

**Env-side mask changes vs raw `get_legal_actions`:**

- When `AWBW_BUILD_MASK_INFANTRY_ONLY` is truthy, `_strip_non_infantry_builds` clears non-infantry BUILD bits after the main loop (`rl/env.py:308–321`, `984–986`).
- When `AWBW_CAPTURE_MOVE_GATE` is set, **`get_legal_actions` itself** restricts MOVE legals (`engine/action.py:747–771`); the mask reflects that filtered list — not an extra post-pass in `_get_action_mask`.

**Network masking:** `AWBWNet.forward` applies `logits.masked_fill(~action_mask, float("-inf"))` when `action_mask` is provided (`rl/network.py:161–162`). Docstring: **`action_mask` shape `(batch, ACTION_SPACE_SIZE)` bool** (`rl/network.py:136–140`).

---

## 3. Sub-step structure (SELECT → MOVE → ACTION)

**Engine field:** `GameState.action_stage: ActionStage` (`engine/game.py:127`; enum `engine/action.py:39–41`: `SELECT`, `MOVE`, `ACTION`).

**What `get_legal_actions` emits per stage** (`engine/action.py:635–643`):

| Stage | Action types (from generators) |
|---|---|
| **SELECT** | `ACTIVATE_COP`, `ACTIVATE_SCOP`, `SELECT_UNIT` (unmoved units), `END_TURN` (conditional), `BUILD` (direct factory builds) — `_get_select_actions` `engine/action.py:656–731` |
| **MOVE** | `SELECT_UNIT` with `move_pos` set (destinations) — `_get_move_actions` `734–777` |
| **ACTION** | `WAIT`, `DIVE_HIDE`, `ATTACK`, `CAPTURE`, `UNLOAD`, `LOAD`, `JOIN`, `REPAIR` (Black Boat) — `_get_action_actions` `781–966` |

**Mask by stage:** `_get_action_mask` always mirrors **current** `get_legal_actions(self.state)`; only one stage’s action family is present at a time (plus the SELECT-only BUILD family when in SELECT).

**`AWBWEnv.step`:** decodes `action_idx` with `_flat_to_action(..., legal=self._get_legal())` (`rl/env.py:838`), then `GameState.step(action)` (`rl/env.py:856` → `_engine_step_with_belief`). Engine validates `action in get_legal_actions(self)` when not in oracle mode (`engine/game.py:402–410`).

---

## 4. Examples — round-trip decode

**Important:** `_flat_to_action` is **not** a closed-form decoder; it **searches the current legal list** (`rl/env.py:263–267`). Examples below are **semantic** descriptions; the concrete `Action` for a given integer **depends on `state` and legal ordering**.

| Flat | `_action_to_flat` rule | Typical meaning (when legal) |
|---:|---|---|
| **0** | `END_TURN` | End active player’s turn |
| **1** | `ACTIVATE_COP` | Fire COP |
| **2** | `ACTIVATE_SCOP` | Fire SCOP |
| **3** | `SELECT_UNIT` | `unit_pos = (0,0)` — in SELECT: choose unit there; in MOVE: **only first legal `move_pos` in sorted order** (see §1.2) |
| **187** | `SELECT_UNIT` | `unit_pos = (6,4)` since `3 + 6*30 + 4 = 187` |
| **900** | Ambiguous numeric | SELECT: unit at tile idx 897 / ATTACK: target `(0,0)` — whichever stage is active |
| **1800** | `CAPTURE` | Capture at committed `move_pos` |
| **1810** | `UNLOAD` | `_UNLOAD_OFFSET` + slot 0 + dir N |
| **3500** | `REPAIR` | Black Boat repair targeting tile `(0,0)` |
| **10000** | `BUILD` | Build `UnitType.INFANTRY` (`0`) at factory tile `(0,0)` |

Tile for flat **187**: `187 = 3 + 6*30 + 4` → **`(6,4)`** as `unit_pos` for `SELECT_UNIT`.

---

## 5. Per-tile action types — logits tally (interpretive)

The **flat space** mixes scalars, full 30×30 target grids, BUILD×27, and small UNLOAD blocks. The table below estimates **if** each family were laid out as **independent** per-tile channels on a 30×30 grid (for spatial-head planning). **This is not how the current flat head factors logits** (see §1.2 for MOVE).

| Kind | K (conceptual logits per tile, if spatialized) | Notes |
|---|---:|---|
| Unit pick / **MOVE destination** | **900** | Flat encoding **does not** separate destinations in MOVE stage (§1.2) |
| **ATTACK** target | **900** | Indices `900..1799` |
| **CAPTURE** | **1** global | Index `1800` |
| **WAIT / LOAD / JOIN / DIVE_HIDE** | **4×1** | Indices `1801..1804` |
| **UNLOAD** | **8** global | Indices `1810..1817` (not a 30×30 grid) |
| **REPAIR** target | **900** | Indices `3500..4399` |
| **BUILD** | **27 per tile** | **24,300** indices `10000..34299` |
| **END_TURN / COP / SCOP** | **3** global | `0..2` |

**Scalar / small-block globals in the encoded ranges:** `0..2` (3) + `1800..1804` (5) + `1810..1817` (8) → **16** indices that are not “one index per map cell” in the ATTACK/REPAIR/BUILD sense.

**Unused padding in `[0,35000)`:** `1805..1809` (5), `1818..3499` (1682), `4400..9999` (5600), `34300..34999` (700) → **7,987** indices never set by `_action_to_flat`.

**Indices that ever receive `True` from some legal game state** are a **subset** of the complement (27,013 “non-padding” slots in the static layout); **MOVE-stage collisions** mean fewer **semantically distinct** choices than distinct engine `Action` objects.

---

## 6. Findings the spatial-head composer needs

1. **MOVE / flat index collision (§1.2):** All MOVE destinations for the current unit share one flat index; decoding picks the **first** legal destination. Any spatial reprojection must reproduce **that** contract or intentionally diverge with a migration plan.
2. **`_flat_to_action` is legal-list lookup**, not bit-packing inversion (`rl/env.py:254–268`).
3. **`UNLOAD` encoding** (`rl/env.py:215–231`): uses **cardinal direction** from `target_pos - move_pos`; **`slot = int(unit_type) & 1`**, so only two slot bits — **collisions possible** when multiple cargo differ only in ways masked out (decoder again picks first legal match).
4. **`ATTACK` / `SELECT` numeric overlap** on **900..902** (§1.1).
5. **Engine-side pruning** changes the legal set (not the env mask loop): e.g. WAIT pruned when CAPTURE available (`engine/action.py:923–937`), APC WAIT pruning (`940–965`). Mask matches pruned list.
6. **RESIGN** exists in `ActionType` (`engine/action.py:81–82`) but is **not** in `_RL_LEGAL_ACTION_TYPES` / RL legals (`engine/action.py:93–111`).

---

## 7. Verification

- **Read:** full `rl/env.py` (lines **1–1550** in workspace; encoding/mask/step cited above).
- **Read:** `engine/action.py` through `get_legal_actions` and stage helpers (representative **1–120**, **635–777**, **781–966**).
- **Read:** `engine/game.py` `step` SELECT/MOVE handling **458–467**; STEP-GATE **402–410**.
- **Read:** `engine/unit.py` `UnitType` enum **19–46**.
- **Read:** `rl/network.py` **17–20**, **129–164** (mask shape and `masked_fill`).
- **Ran:** `python` confirming `_BUILD_OFFSET = 10000`, max BUILD index **34299**, max REPAIR index **4399**; **MOVE-stage duplicate-flat experiment** (4 legals → 1 unique flat).

---

## Account closure

| Question | Answer |
|---|---|
| All 35,000 indices accounted? | **Yes:** every integer in `[0,34999]` is either assigned above or listed as unused padding/gaps. |
| Mask shape `(35000,) bool` + `masked_fill`? | **Yes** (`rl/env.py:289–306`, `rl/network.py:161–162`). |
| Most confusing geometry | **MOVE stage:** `move_pos` varies per legal `Action` but **flat index ignores `move_pos`**, collapsing all destinations to one index with **order-dependent** decode. |
