# Phase 11J-GID-1636707 — closeout (Snow hypothesis drill)

## Defect (register row)

- **gid:** 1636707  
- **Approx:** day 12, envelope index 23, first failing action inside that envelope (action index **17** in the 32-action chain).  
- **Message:** `Illegal move: Infantry (move_type=infantry) from (14, 6) to (11, 6) (terrain id=20, fuel=98) is not reachable.`

## Step 1 — Empirical: weather + path + terrain

**Replay:** `replays/amarriner_gl/1636707.zip` (extras catalog).  
**Envelope 23:** `p:3780539;d:12` (AWBW player `3780539` → **engine P1**, Andy).

After **22 full envelopes** (before any action in envelope 23):

- **`state.weather`:** `clear`  
- **`state.default_weather`:** `clear`  
- **`state.co_weather_segments_remaining`:** `0`  

**Tile path (engine row, col) for the recorded Move:** `paths.global` is AWBW x,y: (6,14)→(6,13)→(6,12)→(6,11) → engine **(14,6)→(13,6)→(12,6)→(11,6)** — **destination terrain id 20** (`WNRoad`), matching the error.

**Terrain IDs on that path:**

| Cell   | Terrain id | Name (terrain.py) |
|--------|------------|-------------------|
| (14,6) | 1          | Plain             |
| (13,6) | 3          | Wood              |
| (12,6) | 3          | Wood              |
| (11,6) | 20         | WNRoad            |

**Snow hypothesis:** **REJECTED** for this divergence frame — weather is **clear**, not snow. The prior “Snow + Wood = 2→4 MP” story does **not** explain this register row.

## Step 2 — Weather / `compute_reachable_costs` audit (canon)

**Code:** `engine/action.py::compute_reachable_costs` uses `terrain_id → effective_move_cost(...)` (`engine/weather.py`).

**AWBW Wiki (Snow):** The [Weather](https://awbw.fandom.com/wiki/Weather) page documents that in **Snow**, foot soldiers pay **double movement cost over Plains, Mountains, and Woods** (among other terrain effects). That matches the **intent** of `engine/weather.py::_SNOW` (infantry ×2 on plains/woods/mountains in the table).

**Conclusion for this gid:** **No** weather-table bug is in play at the failing frame — **clear** weather means `_SNOW` / `_RAIN` tables are not applied.

## Step 3 — Root cause classification (Case A/B/C)

| Case | Description | Applies? |
|------|-------------|----------|
| **A** | Engine too permissive vs PHP | **No** — error is engine rejection. |
| **B** | Engine too strict vs PHP | **No** — not a snow-cost math issue; weather is clear. |
| **C** | Other (fuel, FOW, etc.) | **Partial** — **wrong unit** is used for `_move_unit` after SELECT. |

### Actual root cause (extreme ownership)

**Smoking gun:** **Two different players’ units share the same tile `(14, 6)`** after the **22nd** envelope completes (before envelope 23):

- `Unit(INFANTRY, P0, …, pos=(14, 6), moved=False)`  
- `Unit(MED_TANK, P1, …, pos=(14, 6), moved=True)`  

That is **not** legal AWBW geometry (opposing players cannot co-occupy a ground tile). It is **upstream state drift** from replay stepping, but it explains the failure **without** snow.

**Failing move (action 17):** AWBW labels the mover **`Md.Tank`** (`units_name` / `Md.Tank` → `UnitType.MED_TANK`). Paths are correct for a **P1** medium tank marching **(14,6)→(11,6)**.

**Oracle resolution** (`tools/oracle_zip_replay.py::_apply_move_paths_then_terminator`, not edited here) resolves a **MED_TANK** mover and issues **SELECT_UNIT** with `select_unit_id` pinned to that unit’s **`unit_id`**.

**Commit path:** `GameState.step` → `SELECT_UNIT` uses **`get_unit_at_oracle_id`** (correct) → **`WAIT`** / move commit uses **`_apply_wait`**, which currently does:

```python
unit = self.get_unit_at(*action.unit_pos)
```

(`engine/game.py` around the `_apply_wait` entry).

**`get_unit_at`** returns the **first** alive unit at `(row,col)` in iteration order over `self.units.values()` — **P0 is iterated before P1`**, so it returns the **P0 Infantry** on `(14,6)`, **not** the oracle-pinned **P1 Medium Tank**.

Then `_move_unit` runs **reachability for Infantry** along the **same path** — blocked by **enemy** units on `(13,6)` / `(12,6)` — producing:

`Illegal move: Infantry … is not reachable.`

This matches the **Phase 11J-F5-OCCUPANCY** pattern already documented for **`_apply_attack`** (prefer `selected_unit` / `get_unit_at_oracle_id` on co-occupied oracle tiles), but **`_apply_wait` was not updated** to the same defense-in-depth.

**Bug class:** **Oracle co-occupancy / duplicate-position** + **WAIT commit using `get_unit_at` instead of `selected_unit` / `get_unit_at_oracle_id`**.

## Step 4 — Fix or escalate

**Intended fix (≤20 LOC, bounded):** In `engine/game.py::_apply_wait`, mirror **`_apply_attack`**:

- Prefer `self.selected_unit` when it matches `action.unit_pos`, else  
- `self.get_unit_at_oracle_id(*action.unit_pos, action.select_unit_id)`  

(and only then fall back to `get_unit_at` if needed for legacy callers).

**Hard rule conflict:** This session is **not** allowed to edit `engine/game.py` outside **weather-modifier helpers** (per campaign rules). **`_apply_wait` is not in that carve-out.**

**Escalation:** Hand **game.py `_apply_wait` / WAIT oracle pin** to the lane that owns **movement commit + Phase 11J-F5 occupancy** (same owners as the existing `_apply_attack` comment block). Optional follow-up: **investigate why P0 Infantry and P1 Med Tank share `(14,6)`** — that duplicate occupancy is the **upstream** anomaly; fixing WAIT alone **masks** the move for this gid but does not explain **how** the illegal stack was created.

**No change shipped** to `engine/weather.py`, `engine/action.py`, `tools/oracle_zip_replay.py`, or `tools/desync_audit.py` in this lane.

## Step 5 — Validation gates (as run)

| Gate | Result |
|------|--------|
| **Re-audit gid 1636707** | `python tools/desync_audit.py --games-id 1636707 --catalog data/amarriner_gl_extras_catalog.json --zips-dir replays/amarriner_gl` → still **`engine_bug`** (expected until `game.py` fix lands). |
| **100-game sample** | Not re-run (no engine change). |
| **pytest** `tests/test_weather*.py` | **No such files** in `tests/` (pattern absent). |
| **pytest** `tests/test_engine_negative_legality.py` + related | `python -m pytest tests/test_engine_negative_legality.py tests/test_co_movement_koal_cop.py tests/test_co_colin_mechanics.py -q` → **67 passed, 3 xpassed** (baseline). |

## Verdict letter

**E — Escalate (blocked by `game.py` edit policy).**

The **Snow** story is **not** the first-divergence mechanism for **1636707** at the audited frame. The actionable defect is **WAIT** move commit using **`get_unit_at`** under **duplicate-position oracle state**, not **Snow weather costs on wood**.

---

*Primary weather canon (for Snow generally, not the failing frame):* [AWBW Wiki — Weather](https://awbw.fandom.com/wiki/Weather) — Snow doubles foot movement cost on Plains, Mountains, and Woods (see page table).  
*Co.php / interface guide:* not re-fetched in this pass; escalation can add if a second-line citation is required.
