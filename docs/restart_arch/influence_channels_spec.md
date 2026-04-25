# Influence / threat spatial channels — design spec (STD, full obs, no fog)

**Status:** specification only (no implementation in this change).  
**Scope:** Standard ranked play, **no fog** — all features are computed from the engine’s full `GameState` (no belief-conditioned threat).

**Goal:** Add **derived tactical influence** planes to the spatial tensor in `rl/encoder.py` so the value/policy stack can learn exchange geometry, initiative, and capture races without rediscovering them from raw unit/terrain one-hots alone.

**References (existing code):**

| Area | Location |
|------|----------|
| Encoder layout & `encode_state` | `rl/encoder.py` |
| Movement BFS / reachability | `engine/action.py` — `compute_reachable_costs`, `get_reachable_tiles`, `reconstruct_shortest_move_path` |
| Attack range & legality | `engine/action.py` — `get_attack_targets` |
| Occupancy grid | `engine/action.py` — `_build_occupancy` |
| Damage formula, table, luck sweep | `engine/combat.py` — `get_base_damage`, `load_damage_table`, `calculate_damage`, `damage_range` |
| Unit stats (range, indirect, class) | `engine/unit.py` — `UNIT_STATS`, `UnitType` |
| Terrain defense stars & metadata | `engine/terrain.py` — `get_terrain` |
| Base damage matrix | `data/damage_table.json` (loaded via `engine/combat.load_damage_table`) |
| Full-turn MCTS driver (not on encode path) | `engine/game.py` — `GameState.apply_full_turn` |

**What does *not* exist today:** there is **no** `compute_threat_map` / influence helper in the engine; this spec assumes **new** pure functions (recommended home: `engine/threat.py` or `engine/obs_influence.py`) that call the symbols above.

---

## 1. Recommended new channels (with rationale)

Unless noted, **shape** is `(GRID_SIZE, GRID_SIZE)` with `GRID_SIZE = 30`, aligned with existing encoder padding (only `0..H-1, 0..W-1` from `state.map_data` are meaningful; outside map stays `0`).

### 1.1 `threat_in_p0`

| Field | Value |
|-------|--------|
| **Name** | `threat_in_p0` |
| **Shape** | `(30, 30)` |
| **Semantics** | For each tile `(r,c)`, **maximum single-strike HP damage** (engine internal `0–100` scale) that **P1 (seat 1)** could deal on the **next combat resolution** to a **synthetic P0 defender** placed on `(r,c)`, if that attack were legal **right now** under full-obs rules. Normalize to **`[0, 1]`** via `damage / 100`. Tiles where no attack is possible (out of range, null base damage, submerged/hidden rules, impassable defender placement) → `0`. |
| **Why tactically** | Gives the policy a direct read of **where standing next turn is lethal** against the current enemy posture, including post-move direct fire and stationary indirect. |
| **Source utility** | New helper (see §5) built from `compute_reachable_costs`, `get_attack_targets`, `get_terrain`, `calculate_damage` (or max of `damage_range` / fixed `luck_roll` — see §6). Attacker objects are **live** `Unit` instances from `state.units[1]`; defender is a **scratch** `Unit` (see §6). |
| **CPU cost** | **Expensive** — roughly **one BFS per enemy combat unit** that can move-and-fire or fire indirectly, plus O(range) target iteration; dominates encode time if implemented naïvely (see §3). |

### 1.2 `threat_in_p1`

| Field | Value |
|-------|--------|
| **Name** | `threat_in_p1` |
| **Shape** | `(30, 30)` |
| **Semantics** | Mirror of `threat_in_p0`: max one-strike damage **P0** could deal to a **synthetic P1** defender on `(r,c)`, normalized to `[0,1]`. |
| **Why tactically** | Symmetric **opponent threat** field; useful for value estimation and for the inactive player’s observation when `observer` matches. |
| **Source utility** | Same helper pattern; swap `state.units[0]` as attackers and `player=1` synthetic defender. |
| **CPU cost** | **Expensive** (same class as `threat_in_p0`; a second pass over friendly units). |

### 1.3 `reach_p0`

| Field | Value |
|-------|--------|
| **Name** | `reach_p0` |
| **Shape** | `(30, 30)` |
| **Semantics** | **Movement frontier** for the current turn: `1.0` if **any** alive P0 unit can legally **end movement** on `(r,c)` this turn (i.e. `(r,c)` is a key in `compute_reachable_costs` for that unit), else `0`. If multiple units can reach, still `1.0` (binary OR). Optional future: fractional `k/8` — **not** in v1. Normalization **`{0,1}`** (subset of `[0,1]`). |
| **Why tactically** | Summarizes **board control** and **immediate placement options** without the network diffusing reachability from per-unit channels. |
| **Source utility** | `compute_reachable_costs` (`engine/action.py`) per P0 unit; union of keys. |
| **CPU cost** | **Medium** — **one BFS per P0 unit** (typically ≲ 8–16 in STD). No extra work if threat pass already computed reach — but **threat** does not need full union unless reused; see §3 for cache sharing. |

### 1.4 `reach_p1`

| Field | Value |
|-------|--------|
| **Name** | `reach_p1` |
| **Shape** | `(30, 30)` |
| **Semantics** | Same as `reach_p0` for P1 units. |
| **Why tactically** | Symmetric control / counterplay prior. |
| **Source utility** | `compute_reachable_costs` for each P1 unit; union. |
| **CPU cost** | **Medium** — one BFS per P1 unit. |

### 1.5 `turns_to_capture_p0`

| Field | Value |
|-------|--------|
| **Name** | `turns_to_capture_p0` |
| **Shape** | `(30, 30)` |
| **Semantics** | **Property-focused** channel (non-property tiles: keep `0`). For each **capturable income property** tile (same notion as encoder’s property loop: exclude comm tower / lab per `rl/encoder.py` property handling), estimate **how soon** P0 could **finish** a capture, using a **cheap path proxy**: e.g. \(\min_u \text{MP path cost from foot unit } u \text{ with } `UNIT_STATS.can_capture`\) from `u.pos` to `(r,c)`, then map to **normalized turns** in `[0,1]`, where **`1.0` means unreachable this “planning horizon”** (per task request: clamp **unreachable → 1.0**). Exact AW capture timing (20 steps, Sami modifiers, mid-capture continuation) is **out of scope v1** — use **MP distance + optional `capture_points` slack** in the spec for implementers (see §6). |
| **Why tactically** | Encodes **race to income** without the network inferring graph search on the raw map. |
| **Source utility** | **No** single `GameState` method today; implement via repeated `compute_reachable_costs` from each **Infantry/Mech** (or `stats.can_capture`) **or** one new multi-source Dijkstra-like pass that reuses `effective_move_cost` rules from `compute_reachable_costs` (still “don’t re-derive movement rules” — factor shared kernel from `engine/action.py`). |
| **CPU cost** | **Medium** — one BFS per capturing unit (often 2–6), only **seeds** at property tiles when writing the grid. |

### 1.6 `turns_to_capture_p1`

| Field | Value |
|-------|--------|
| **Name** | `turns_to_capture_p1` |
| **Shape** | `(30, 30)` |
| **Semantics** | Mirror of `turns_to_capture_p0` for P1 capturers. |
| **Why tactically** | Symmetric race signal for contested lines. |
| **Source utility** | Same as P0 with `state.units[1]`. |
| **CPU cost** | **Medium**. |

### 1.7 `income_pressure` (optional)

| Field | Value |
|-------|--------|
| **Name** | `income_pressure` |
| **Shape** | `(30, 30)` |
| **Semantics** | **Optional v1** — recommend **omit** unless a non-redundant definition is locked. Candidate: **contestedness** = normalized `1000 * sign` of **per-turn swing** if ownership at `(r,c)` flips (income properties only; see `GameState._grant_income` in `engine/game.py` — baseline **1000g** per property before CO quirks). Normalized e.g. by `50000` to match scalar funds scale in `encode_state`. **Caveat:** global income is already partly captured by scalar `p0_income_share` and property one-hots; spatial “pressure” overlaps strongly with **`turns_to_capture_*` + `reach_*`**. If kept, treat as **experimental** and monitor training cost. |
| **Why tactically** | Highlights **high economic leverage** tiles when combined with reach/race channels. |
| **Source utility** | `state.properties` + ownership; no BFS if defined as static `±income/50000` on income tiles only. |
| **CPU cost** | **Cheap** if **no** pathing; **medium** if multiplied by “enemy can reach next turn”. |

### 1.8 Inclusion vs cost (summary)

| Channel | Sample-efficiency | Relative CPU | Verdict |
|---------|-------------------|----------------|---------|
| `threat_in_p0` / `threat_in_p1` | **Highest** — addresses the core critique (“where is dangerous?”) | **Highest** | **Slam-dunk** for architecture; need careful implementation budget (§3) |
| `reach_p0` / `reach_p1` | High — common in war games | Medium | **Slam-dunk** — reuses `compute_reachable_costs` directly |
| `turns_to_capture_p0` / `turns_to_capture_p1` | High for macro | Medium | **Strong** — needs new composition but same BFS primitive |
| `income_pressure` | Marginal / overlaps | Cheap–medium | **Marginal** — optional or defer to v2 |

Target: **~+5–15% wall time per `encode_state`** vs current 63-channel path. **Threat** is the only block likely to threaten that budget if `calculate_damage` is called millions of times; **reach** and **capture-turn** should stay inside the envelope with shared occupancy and tight inner loops (§3).

---

## 2. Final channel layout

Recommended **six** new spatial planes (omit optional `income_pressure` for the default count).

| Channel range | Block | Notes |
|---------------|-------|--------|
| 0–27 | Unit presence (unchanged) | 14 types × 2 players |
| 28–29 | HP belief lo/hi (unchanged) | **Keep** for future fog experiments even in STD |
| 30–44 | Terrain one-hot (unchanged) | 15 categories |
| 45–59 | Property ownership (unchanged) | 5 types × 3 ownerships |
| 60–62 | Capture / neutral income (unchanged) | P0 cap, P1 cap, neutral-income mask |
| **63** | **`threat_in_p0`** | NEW |
| **64** | **`threat_in_p1`** | NEW |
| **65** | **`reach_p0`** | NEW |
| **66** | **`reach_p1`** | NEW |
| **67** | **`turns_to_capture_p0`** | NEW |
| **68** | **`turns_to_capture_p1`** | NEW |
| **69** | *reserved or `income_pressure`* | Only if optional channel is adopted |

**Final `N_SPATIAL_CHANNELS`:**

- **Default (recommended):** `63 + 6 = 69`.
- **With optional `income_pressure`:** `70`.

---

## 3. Computation strategy

**Shared prep (once per `encode_state`):**

- `occupancy = _build_occupancy(state)` (`engine/action.py:195`) — use for all passes.
- Precompute `get_terrain` / `TerrainInfo` only where needed (tile under attacker move, defender tile) via `engine/terrain.get_terrain` from `state.map_data.terrain[r][c]`.

### 3.1 `threat_in_*` (expensive)

**Intended mechanics (match `get_attack_targets` / full-obs combat):**

- **Indirect units** (`UNIT_STATS.is_indirect`): only `move_pos == unit.pos` (`engine/action.py:496-498`). For each tile `(r,c)` in the **Manhattan ring** between `min_range` and `max_range` (with Grit/Jake range CO adjustments as in `get_attack_targets:502-513`), if an attack on a **defender occupying `(r,c)`** is legal, compute damage.
- **Direct units**: for each `move_pos` in `compute_reachable_costs(state, unit, occupancy=occ).keys()`, call `get_attack_targets(state, unit, move_pos, occupancy=occ)` — the engine already implements Manhattan range, ammo, submerged rules, seam targets, etc.
- **Damage value:** use **deterministic** mid or max **expected** strike:
  - **Recommended default:** `luck_roll = 5` single call to `calculate_damage` for speed; **or** `damage_range(...)[1] / 100` for **worst-case** threat (more conservative, 10× `calculate_damage` calls per pair).
  - **Documented alternative:** average of rolls `0..9` (10×) for **mean** expected damage under uniform luck digit — closer to “expected” wording; more expensive.

**Per-tile aggregation:** For tile `(r,c)`, take **`max` over all attacking units and legal (move, attack) pairs** of normalized damage \([0,1]\).

**Complexity (order-of-magnitude, 30×30 STD, ~16 units/side):**

- **Per attacker:** one `compute_reachable_costs` **O(visited cells)** (typically **≤ ~400** on open maps; ≤ `30×30` worst case with pipe/teleport edge cases).
- **Per reachable tile:** `get_attack_targets` scans **O(range²)** Manhattan box — small constant for `max_range ≤ 8`.
- **Per damage eval:** `calculate_damage` is **O(1)**; optional ×10 for luck sweep.

**Rough product:** 16 friendly attackers × (≈200–500 BFS expansions) × (≈10–50 range checks) × (1–10 damage evals) — **threat is the main CPU risk**. Keep inside **+15%** by: (1) **one** `calculate_damage` per (attacker, move, target tile) with fixed `luck_roll`, (2) **early exit** on zero base damage via `get_base_damage`, (3) **reuse** `costs` dict from BFS for direct units without recomputing path objects.

**Transports / cargo:** only the **top-level** unit on a tile attacks; **loaded units** do not contribute until unloaded — engine legality already matches this; no extra work.

**Pipe seams / empty-tile attack:** `get_attack_targets` can return seam targets without a unit. Threat **for a hypothetical unit on a seam tile** is still a valid “tile danger” question; include seam cases if they appear in `get_attack_targets` (defender is not a `Unit` — optional to **zero** seam cells for “unit threat” only; recommend **include** seam damage using `calculate_seam_damage` / seam path in `combat` where relevant — flag as **edge case** in tests).

### 3.2 `reach_*` (medium)

- For each unit on the side, `compute_reachable_costs`; mark union bitmap `1.0` on `spatial`.
- **Both sides** need separate loops — **cannot** reuse a single BFS for both players.

### 3.3 `turns_to_capture_*` (medium)

- For each **capturing** unit type (Infantry/Mech in AW), run `compute_reachable_costs`. For each **relevant property** tile, record **minimum** MP cost to **stand on** the property (same stop rules as in BFS `result`).

**Proxy for “turns” (v1):**

- Let `C = min_path_mp` (unreachable if tile not in `result` for any unit → store **`1.0`** normalized “bad”).
- Map `C` to `[0,1]`: e.g. `turn_norm = min(1.0, C / C_ref)` with `C_ref` ≈ **typical 6-tile range × 4 MP per plain** scale (~24–30) or **empirical max** on 30×30; **larger = harder/farther** so values near **1** mean “distant or unreachable”.

- **Improving fidelity later:** add `+ f(capture_points_remaining)` for units already on tile — read `Property.capture_points` and occupant from `state` (mirrors `rl/encoder.py` capture progress).

**Caching:** BFS for capture **overlaps** with `reach_*` for foot units — optionally merge loops so one `compute_reachable_costs` per **Infantry/Mech** seeds both `reach` (union) and `min_cost_to` arrays for property tiles only.

### 3.4 `income_pressure` (optional, cheap)

- If static: **no** BFS; fill from `state.properties` and **known per-tile income** (1000g baseline per `count_income_properties` logic; HQ included as income in engine).

### 3.5 Caching between encodes

- **Per-step `encode_state` only** — no cross-step cache in v1.
- **Within one encode:** build **`occupancy` once**; reuse for all BFS calls; **do not** cache `compute_reachable_costs` results across *different* units unless the helper explicitly batches (future).

---

## 4. Insertion point in `encode_state` (`rl/encoder.py`)

Current flow (line references from `c:\Users\phili\AWBW\rl\encoder.py` at time of writing):

- **57–65:** `N_SPATIAL_CHANNELS` and block size constants.
- **187–198:** `H`, `W`, spatial buffer allocation, `hp_lo_ch` / `hp_hi_ch` indices.
- **199–238:** **Terrain** block copy into `spatial[:, :, terrain_ch_offset : ...]`.
- **240–280:** **Property** ownership + capture + neutral mask loop.
- **282–312:** **Unit** presence + **HP** channels loop.

**Recommended insertion:** **after** the unit/HP loop **(after line 312, immediately before the scalar section at line 314)**. Rationale: influence features depend on the **full** unit set and property layout; terrain and property one-hots are already materialized; inserting here avoids an extra `GRID_SIZE×GRID_SIZE` pass over units before occupancy is conceptually final.

**Implementation pattern:**

- **Preferred:** a single call such as `fill_influence_channels(state, out=spatial, channel_base=63, observer=observer)` that writes **`spatial[:, :, 63:69]`** via **NumPy** slices (contiguous block copy), with **inner** loops in Python or **Numba**-eligible tight loops as future work.
- **Avoid** another full triple nested `for r in range(30): for c in range(30):` over *all* 69 channels in hot path for threat — build **`(H,W,6)` float32** in the helper, then one `np.copyto`.

**Constants to update in the same file** when implemented: module docstring **lines 4–11**, **`N_SPATIAL_CHANNELS` expression lines 62–65**, and any downstream **`rl/network.py`** / checkpoint compat that assumes `63` (out of scope for this spec — flag for migration).

---

## 5. Encoder API additions (proposed)

### 5.1 Core helper

```text
# Proposed location: engine/threat.py (or engine/obs_influence.py)

def compute_influence_spatial(
    state: GameState,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Return float32 array of shape (GRID_SIZE, GRID_SIZE, 6) in [0,1] for
    [threat_in_p0, threat_in_p1, reach_p0, reach_p1, turns_to_capture_p0, turns_to_capture_p1]
    (channel order matches Section 2).
    When ``out`` is provided, write in-place; otherwise allocate.
    """
```

**Rationale for living outside `rl/encoder.py`:** keeps the encoder thin and allows **unit tests** against the engine without importing PyTorch/RL. **Not** on `GameState` method to avoid bloating `engine/game.py` (≈2.7k+ lines).

**Lower-level factoring (internal):**

```text
def _max_threat_to_seat(
    state: GameState,
    aggressor_seat: int,
    victim_seat: int,
    occupancy: dict[tuple[int, int], Unit],
) -> np.ndarray: ...

def _reach_union(
    state: GameState,
    seat: int,
    occupancy: dict[tuple[int, int], Unit],
) -> np.ndarray: ...

def _capture_time_proxy(
    state: GameState,
    seat: int,
    occupancy: dict[tuple[int, int], Unit],
) -> np.ndarray: ...
```

### 5.2 Synthetic defender construction

- Allocate a **scratch** `Unit` (`engine/unit.py:368+`) with `player=victim_seat`, `pos=(r,c)`, `hp=100`, `ammo`/`fuel` from `UNIT_STATS[def_type]` max values as appropriate, `is_submerged` consistent with type if testing naval tiles.
- Reuse the same **scratch** in inner loops, **mutate** `pos` and `unit_type` / `hp` as needed, **do not** register in `state.units`.
- `calculate_damage` uses `defender.display_hp` and terrain — must match `get_attack_targets` visibility rules (sub/stealth).

### 5.3 Test strategy

- **Golden micro-maps** (small `GameState` fixtures): 1 direct tank, 1 indirect artillery — assert threat peaks at expected Manhattan loci; unreachable tiles `0`.
- **Regression vs engine:** for a few `(attacker, move, target)` triples, `calculate_damage` must match `GameState` combat outcomes already tested in `test_combat_anchor.py` (reference existing tests) for **luck_off** or fixed `luck_roll`.
- **Reach:** compare union keys to `get_reachable_tiles` manual OR on tick.
- **Capture proxy:** one neutral city + one infantry at measured distance — monotone decrease with distance.
- **Performance smoke:** microbenchmark `encode_state` p95 with/without 6 channels on a 30×30 sample — should stay **within +15%** of baseline.

---

## 6. Risks / unknowns

### 6.1 Engine symbols to add

- **Confirmed missing:** no **`compute_influence_spatial` / `compute_threat_map`** in repo — **new code required**, built from **real, existing** primitives listed in §0.

### 6.2 Threat model mismatches (simplifying assumptions)

| Topic | Assumption for v1 |
|-------|-------------------|
| **Hypothetical defender type** | Single **per tile class** (e.g. max over **Infantry, Tank, B-copter** where `effective_move_cost` allows standing) *or* **one** canonical type (e.g. full-HP **Mech**) with **0** on tiles that type **cannot** occupy — must be **fixed in implementation**; recommend **max over a small set (≤3 types)** to avoid 14× blow-up. |
| **COP/SCOP/weather/CO** | **Fully** reflected if using `calculate_damage` and `state.co_states` (matches engine). **Risk:** training only saw subset of COs — still truth-preserving. |
| **“Expected” damage** | True expectation needs **luck distribution**; using **`luck_roll=5` or `mean(damage_range)`** is an intentional approximation — document in training notes. |
| **Transports** | Unarmed/weak secondaries: respect `get_attack_targets` / ammo **0** rules. **Loaded units** ignored for outgoing threat. |
| **Dive / hide** | `is_submerged` on **real** attackers must match `Unit` state; synthetic defender for naval/air cases must be consistent or masked out. |

### 6.3 Ambiguous normalization

- **`threat_in_*`:** prefer **`[0,1]`** via `/100` on engine damage; clear semantics vs fixed **HP bucket** display.
- **`turns_to_capture_*`:** `C_ref` is **ambiguous** — choose **`C_ref = 60`** (≈2–3 turns of infantry move on sparse terrain) or **adaptive** from `map_data` max dimension; **rationale must be frozen** in implementation so checkpoints don’t silently drift.

### 6.4 If budget exceeded

- **Drop `turns_to_capture_*` first** (macro signal weaker than threat+reach) **or** compute **on downsampled** stride-2 / center 20×20 — **not** recommended for v1; better **simplify damage** to **`min(max_dmg/100,1)`** from `damage_range` (still 10×) only on **candidates after base-damage filter**.

---

## 7. Cost ceiling checklist

- [x] Reuse `compute_reachable_costs`, `get_attack_targets`, `calculate_damage`, `_build_occupancy` — **no** hand-rolled movement.
- [x] **Threat** inner loop minimized — `get_base_damage` gating, single luck sample or short-circuit.
- [x] **Reach + capture** can **share** per-unit BFS where both need foot reach.
- [x] No requirement to call `apply_full_turn` for encoding (snapshot-only).

---

*End of spec.*
