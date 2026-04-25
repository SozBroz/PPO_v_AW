# Move-stage flat action encoding (design spec)

**Scope:** design only. Implementation is a later wave; do not treat this file as shipped behavior.

**Problem:** In `ActionStage.MOVE`, legal actions are `Action(SELECT_UNIT, unit_pos=…, move_pos=destination)`, but the RL flat encoder currently hashes only `unit_pos`, collapsing all move destinations to one index. The decoder disambiguates by scanning `get_legal_actions` in sorted destination order, so the policy cannot choose the tile.

**Bundle:** This encoding change ships in the same restart bundle as the new spatial policy head (lead-approved).

---

## 1. Status quo

### 1.1 Encoder / decoder (exact excerpts)

`_action_to_flat` — `SELECT_UNIT` uses `unit_pos` only:

```191:194:c:\Users\phili\AWBW\rl\env.py
    if at == ActionType.SELECT_UNIT:
        # SELECT: unit tile; MOVE stage also uses SELECT_UNIT with move_pos set (engine).
        r, c = action.unit_pos
        return 3 + r * _ENC_W + c
```

`_flat_to_action` — collision resolution is **first match in `legal` iteration order** (not a geometric inverse of `flat_idx`):

```254:268:c:\Users\phili\AWBW\rl\env.py
def _flat_to_action(
    flat_idx: int,
    state: GameState,
    legal: list[Action] | None = None,
) -> Optional[Action]:
    """
    Decode a flat integer back to a legal Action for the current state.
    Returns None if the index does not correspond to any legal action.
    """
    if legal is None:
        legal = get_legal_actions(state)
    for a in legal:
        if _action_to_flat(a) == flat_idx:
            return a
    return None
```

### 1.2 Collapse: many destinations, one flat index

In **MOVE** stage, the engine enumerates legal moves as `SELECT_UNIT` with `unit_pos` fixed to the selected unit and **varying** `move_pos` (`engine/action.py` builds `Action(ActionType.SELECT_UNIT, unit_pos=unit.pos, move_pos=pos)` in sorted `pos` order). The encoder ignores `move_pos`, so for a fixed unit at `(r₀, c₀)` every legal move shares:

`flat = 3 + r₀ * 30 + c₀`

**Documented in-repo check** (`docs/restart_arch/action_space_inventory.md` §1.2): `AWBWEnv.reset(seed=0)` stepped until `action_stage == MOVE` with **4** legal move actions produced **one** unique flat index, repeated four times.

**Larger D (same bug):** e.g. an infantry with `move_range == 3` on a real map might have **13** legal destination tiles after BFS; status quo still maps those **13** `Action` objects to **1** flat index. The engine then decodes the policy’s index as the **first** destination in sorted `(row, col)` order — not a learned preference across the 13.

**Infantry on roads (illustrative):** Reachable tile count depends on map, occupancy, and terrain costs; the failure mode is structural: if there are `D` legal destination tiles, **all `D` engine actions map to the same integer**, not `D` distinct integers.

**Policy-head expressivity loss:** The categorical policy assigns one logit per `flat` index. For the MOVE sub-stage, the **entire** move decision — up to one logit per reachable tile in principle — is forced into **a single** component of that 35,000-wide vector. The softmax therefore cannot assign distinct probabilities to two different legal destinations: whichever destination the implementation ties to that index (today: **the first in sorted `(row, col)` order** in `legal`) is the only one the action channel can “prefer” in expectation.

---

## 2. Why this blocks the spatial policy head

A spatial head is intended to emit **per-tile** preferences (e.g. one logit or two-stage logits per `(r, c)` on the 30×30 grid) and then **scatter** or gather them into the flat `ACTION_SPACE_SIZE` vector. For MOVE, the natural mapping is “destination tile `(r, c)` → logit for choosing that destination.”

With the current encoding, every legal destination for the selected unit maps to **the same flat index** `3 + r₀*30 + c₀`. Any aggregation that fills flat logits from a spatial map — sum, log-sum-exp, “take the logit at the source tile,” or max-pool over destinations — **fuses** what should be `D` separate decision dimensions into **one** scalar before the global softmax over 35,000 actions. The policy cannot learn “move North” vs “move South” as distinct high-level choices when both update the identical scalar logit. The spatial head’s per-tile output is not usable for move selection until each destination tile has **its own** flat index (or a separate channel), which is what the proposed layout fixes.

---

## 3. Proposed encoding

### 3.1 Rules

- **SELECT stage** (`action_stage == ActionStage.SELECT`, no unit selected yet): `SELECT_UNIT` encodes the **unit tile** to activate, **unchanged**:

  `flat = 3 + r*30 + c` with `(r, c) = action.unit_pos`  
  **Range:** 3..902 (900 slots).

- **MOVE stage** (`action_stage == ActionStage.MOVE`, `state.selected_unit` is set): `SELECT_UNIT` encodes the **destination tile** only:

  `flat = _MOVE_OFFSET + r*30 + c` with `(r, c) = action.move_pos`  
  **Recommended:** `_MOVE_OFFSET = 1818`.

- **ACTION stage:** unchanged; `SELECT_UNIT` does not appear in `get_legal_actions` for this stage.

### 3.2 Why `_MOVE_OFFSET = 1818`

`docs/restart_arch/action_space_inventory.md` (master table, §1) assigns:

- 1805–1809: unused (no branch)
- 1810–1817: `UNLOAD` (8 slots)
- **1818–3499: unused (1682 slots) — “never set” in the status quo**

A full 30×30 grid needs **900** consecutive indices. **1818 + 0 .. 1818 + 899 = 1818..2717**, which lies entirely inside 1818..3499 and does not overlap `UNLOAD` (ends 1817) or `_REPAIR_OFFSET` (3500). **1682 ≥ 900** — the block fits with room to spare.

`ACTION_SPACE_SIZE` remains **35,000** (`rl/network.py`); no growth past the 35k discrete head is required for this change.

### 3.3 Pre / post layout (side by side)

Only **MOVE-stage `SELECT_UNIT`** and the **previously always-unused 1818..2717** change semantics; other ranges are unchanged in numbering.

| Range (inclusive) | **Pre (status quo)** | **Post (proposed)** |
|-------------------|----------------------|----------------------|
| 0 | END_TURN | same |
| 1 | ACTIVATE_COP | same |
| 2 | ACTIVATE_SCOP | same |
| 3..902 | SELECT_UNIT by **unit** tile (SELECT) **and** collapsed MOVE (bug) | SELECT_UNIT by **unit** tile **only in SELECT**; **not** used for MOVE |
| 900..1799 | ATTACK | same |
| 1800..1804 | CAPTURE, WAIT, LOAD, JOIN, DIVE_HIDE | same |
| 1805..1809 | unused | same |
| 1810..1817 | UNLOAD | same |
| **1818..2717** | **unused** | **MOVE: SELECT_UNIT by `move_pos` (destination grid)** |
| 2718..3499 | unused | **unused** (781 slots; optional reserve or future use) |
| 3500..4399 | REPAIR | same |
| 4400..9999 | unused | same |
| 10000..34299 | BUILD | same |
| 34300..34999 | padding | same |

**Note:** SELECT and ATTACK still **numerically overlap** on 900..902 (inventory §1.1); that stage-dependent ambiguity is pre-existing and **out of scope** for this spec.

---

## 4. Encoder algorithm (pseudocode)

Parameters: `_ENC_W = 30`, `_MOVE_OFFSET = 1818`, and existing constants for other `ActionType`s.

```text
function _action_to_flat(action: Action, state: GameState) -> int:
    at ← action.action_type
    if at == END_TURN: return 0
    if at == ACTIVATE_COP: return 1
    if at == ACTIVATE_SCOP: return 2

    if at == SELECT_UNIT:
        if state.action_stage == SELECT:
            # move_pos is None on engine legals; pick friendly unit at unit_pos
            (r, c) ← action.unit_pos
            return 3 + r * _ENC_W + c
        if state.action_stage == MOVE:
            # Source is implicit in state.selected_unit; encode destination only.
            assert action.move_pos is not None
            (r, c) ← action.move_pos
            return _MOVE_OFFSET + r * _ENC_W + c
        # ACTION: SELECT_UNIT should not be legal; return 0 or assert in debug builds
        return 0

    # ATTACK, CAPTURE, WAIT, … same branches as today (use action.target_pos / move_pos / etc.)
    ...
```

**Edge cases**

- **SELECT stage, `SELECT_UNIT`:** Engine legals use `unit_pos` only; `move_pos` is unset (`None`). Encoding uses `3 + r*30 + c` — unchanged.
- **MOVE stage, `SELECT_UNIT`:** Engine sets **both** `unit_pos=unit.pos` and `move_pos=destination` (`engine/action.py` `_get_move_actions`). **Contract:** For any legal MOVE action, `action.unit_pos == state.selected_unit.pos`. The step handler only reads `action.move_pos` in MOVE stage (`engine/game.py` `selected_move_pos = action.move_pos`). Mismatching `unit_pos` is **not** a legal `get_legal_actions` result; the STEP-GATE rejects arbitrary tuples. The encoder may `assert` equality in debug mode; release code can follow `move_pos` only for the flat index.
- **End turn / powers / build / etc. in MOVE stage:** In MOVE, `get_legal_actions` **only** returns the move list (`_get_move_actions`); there is no `END_TURN` or `WAIT` in that stage. No special cases beyond **not** emitting scalar indices 0..2 for MOVE (they simply are not legal).

**`_get_action_mask`:** Still loops `for action in legal: mask[_action_to_flat(action, state)] = True` — the signature of `_action_to_flat` must gain `state` (or the caller passes `action_stage` + `selected_unit` explicitly). This is a wave-2 API detail.

---

## 5. Decoder algorithm (pseudocode)

**Goal:** For MOVE-stage indices in `1818..2717`, decode without relying on “first in sorted list” to pick among same-encoded actions (those collisions go away for MOVE).

```text
function _flat_to_action(flat_idx, state, legal) -> Action | None:
    if legal is None: legal ← get_legal_actions(state)

    # Fast path: MOVE-stage destination grid (no ambiguous collision)
    if _MOVE_OFFSET <= flat_idx < _MOVE_OFFSET + 900:
        r ← (flat_idx - _MOVE_OFFSET) // _ENC_W
        c ← (flat_idx - _MOVE_OFFSET) % _ENC_W
        u ← state.selected_unit
        if u is None: return None
        return Action(SELECT_UNIT, unit_pos=u.pos, move_pos=(r, c))  // then validate in legal list OR rely on step gate

    # Fallback: existing equality scan for all other action families
    for a in legal:
        if _action_to_flat(a, state) == flat_idx:
            return a
    return None
```

**Legal-action lookup for MOVE branch:** After constructing the candidate `Action`, implementation must still ensure it is in `legal` (or rely on `GameState.step` to reject illegal acts — the env already decodes before step). Recommended: **check membership in `legal`** or recompute `get_legal_actions` to match current behavior. The **old** failure mode (many legals same `_action_to_flat` → scan returns arbitrary first) **no longer applies** to MOVE indices, because each destination has a **unique** flat index. The linear scan in the generic branch is only needed for **non-MOVE** families where multiple `Action` objects could still hypothetically map to the same index (e.g. any remaining encoding collisions like UNLOAD slot collapse — pre-existing, out of scope).

---

## 6. Action mask plumbing and network

- **`_get_action_mask`:** Unchanged structurally: for each `action` in `get_legal_actions(state)`, set `mask[idx] = True` with `idx = _action_to_flat(action, state)`. In MOVE stage, each legal destination sets a **distinct** bit in `1818..2717` (one per `(r, c)`).

- **Network (`rl/network.py`):** `policy_head` output shape `(..., ACTION_SPACE_SIZE)` with `ACTION_SPACE_SIZE = 35_000`; `masked_fill` uses `action_mask` of shape `(batch, ACTION_SPACE_SIZE)` (`forward` at lines 129–163). **No shape change** if `ACTION_SPACE_SIZE` stays 35,000. Masked logits for indices 1818..2717 transition from “always -∞ in old checkpoints / unused” to “possibly unmasked and active” in new training (see §7).

---

## 7. Backward compatibility

**Old checkpoints and policies are incompatible** with the new scheme:

- Previously, indices `1818..2717` were **unused** (always masked `False` in real play — see inventory §1 unused range).
- After the change, those bits become **valid MOVE logits**; old networks have arbitrary weights in those dimensions. Loading an old weight tensor without retraining will assign **nonsense logits** to the new MOVE slice.

**Acceptable** per restart plan: the bundle is not checkpoint-compatible with pre-restart move semantics anyway (spatial head + training reset).

---

## 8. Encoder-equivalence harness coverage

`tests/test_encoder_equivalence.py` only verifies **`encode_state`** against a frozen `.npz` baseline; it does **not** cover `_action_to_flat` / mask shape.

**Wave 2:** Add a **sibling** test (e.g. `tests/test_action_encoding_equivalence.py`) or a dedicated section that:

- Imports `ACTION_SPACE_SIZE` from `rl.network` and asserts `len(mask) == ACTION_SPACE_SIZE` for `_get_action_mask` output.
- **Golden or table-driven** vectors: for a small set of synthetic `GameState` fixtures in SELECT vs MOVE, assert `_action_to_flat(a, s)` matches the documented formula (SELECT: 3..902; MOVE: 1818 + tile).
- **Regression pin:** `assert _action_to_flat(Action(SELECT_UNIT, ...), state_move) == _MOVE_OFFSET + r*30 + c` for known `(r, c)`.
- Optionally document expected bits set in a MOVE-stage mask (e.g. 13 ones in `1818..2717` for a constructed case).

The existing encoder equivalence test should **remain** observation-focused; do not overload it with action-space bytes unless the team wants a second frozen blob for mask fingerprints.

---

## 9. Validation (wave-2 unit tests)

1. **MOVE mask bit count:** Synthetic state: `action_stage=MOVE`, `selected_unit` at `(5,5)`, exactly `D` legal destinations (e.g. empty plain star, or mock `get_legal_actions` / minimal map). Assert `sum(mask[1818:2718]) == D` and no bits in that range outside the `D` destination coordinates.

2. **Round-trip:** For each legal destination `a`, `decode(encode(a))` yields the same `move_pos` (and `unit_pos` matches `selected_unit`).

3. **SELECT unchanged:** In SELECT, picking unit at `(r,c)` still sets exactly one bit at `3+r*30+c` and no bits in `1818..2717` unless also testing illegal crosses.

4. **Stage isolation:** In MOVE, `flat` for destinations never equals `3 + r0*30 + c0` of the **source** tile (unless by coincidence a destination coordinate equals a **different** scalar formula collision — use positions where `3+r*30+c != 1818+r'*30+c'` for a clear pass).

5. **Inventory regression:** One test file constant `_MOVE_OFFSET` / documented ranges vs `docs/restart_arch/action_space_inventory.md` so future refactors do not shift the block into `UNLOAD` or `REPAIR` ranges.

---

## 10. Open questions / risks (lead sign-off)

1. **API surface:** `_action_to_flat` today is `(action) -> int`. The clean fix threads `state: GameState` (or at least `action_stage` + `selected_unit` + `selected_move_pos`) for every call site in `rl/env.py` and any **tests/tools** that import it. **Confirm** exhaustive call-site list in wave 2 (search `_action_to_flat` across the repo, including server/training if any).

2. **Server / inference parity:** If any non-`rl/` code duplicates flat encoding (grep for `3 +` / `SELECT_UNIT` mapping), it must be updated in the same change — **or** the team centralizes on one helper. Spec is env-centric; a quick duplicate audit in wave 2 is recommended.

3. **UNLOAD / other legacy collisions:** This spec fixes MOVE *destination* ambiguity only. The inventory still flags **UNLOAD** slot bit-packing and other encode collisions; no change to those in this fix.

4. **Ambiguous 900..902 (SELECT vs ATTACK):** Pre-existing stage-dependent overlap remains; only MOVE vs SELECT partition is addressed here.

5. **PPO / experience buffer:** New indices become trainable; any logging that histograms `action_idx` by bucket should add a **MOVE-slice** bucket so dashboards do not misread 1818..2717 as “unused” noise.

6. **Opponent policy / distilled policies:** Any checkpointed opponent in self-play that maps flat→action must be retrained or wrapped with a compatibility shim; confirm whether opponent uses the same `env` decode path (it does: `_flat_to_action` in `rl/env.py`).

---

## Summary (implementation wave)

| Item | Value |
|------|--------|
| `_MOVE_OFFSET` | **1818** (900 slots **1818..2717** inside unused **1818..3499**) |
| `ACTION_SPACE_SIZE` | **35,000** (unchanged) |
| Engine `Action` / `get_legal_actions` | **No change** — encoding-only fix in RL layer |
