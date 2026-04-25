# Phase 11C-FU-DRILL — Trace 182065 `Illegal move … not reachable` root classification

**Date:** 2026-04-21  
**Mode:** Read-only drill (no engine/oracle/test/data edits).  
**Question:** Is the `Infantry from (9, 8) to (11, 7)` failure the same **Koal COP +1 MP** root as Phase 11D-F2, or an independent bug?

---

## Section 1 — Trace metadata

| Field | Value |
|--------|--------|
| **Artifact** | `replays/182065.trace.json` (no `182065.zip` in repo; catalog `data/amarriner_gl_std_catalog.json` has **no** entry for game id 182065) |
| **Map** | **126428** — *Ft. Fantasy* (`make_initial_state` / `MapData.name`) |
| **CO matchup** | **P0:** Sami (`co_id` **8**) · **P1:** Sami (`co_id` **8**) — mirror match |
| **Failing entry** | `full_trace` index **10526** — `WAIT`, `player` **1**, trace `turn` field **59** (calendar day per test module comment) |
| **Active player** | **1** (P1 / Blue) at failure |
| **CO power state (P1)** | `cop_active=False`, `scop_active=False` (sampled immediately before failing `WAIT`) |

**Trace window (indices 10524–10526):**

- `SELECT_UNIT` … `unit_pos` `[9, 8]`, `move_pos` `[11, 7]`
- `WAIT` … same — commits move to `(11, 7)` then wait.

---

## Section 2 — Reachability analysis

### Coordinate and message details

- Positions are **(row, col)** per `Action` / `_trace_to_action` (same as engine `unit.pos`).
- **Manhattan distance** from `(9, 8)` to `(11, 7)` is **|2| + |−1| = 3** (not 5).
- `fuel=73` does not cap movement (engine uses `min(move_range, fuel)`; infantry base `move_range` is 3).
- The error text’s **`terrain id=29`** is taken from **`new_pos`** in `_move_unit` (`terrain[new_pos[0]][new_pos[1]]`), i.e. **destination** tile `(11, 7)`, **not** the start tile `(9, 8)`.

### Terrain id 29

- In `engine/terrain.py`, **29** is **HShoal** (shoal family 29–32). `_shoal_costs()` gives **infantry move cost 1** per entry.

### MP budget vs F2 pattern

- With both COs Sami and **no** COP/SCOP active, infantry effective cap in `compute_reachable_costs` is **base 3** (Sami infantry +1 applies only when `cop_active` or `scop_active`).
- This is **not** the Phase 11D-F2 pattern (Koal **COP** global **+1** missing from `move_range` while a **cost-4** path requires **4** MP).

### Actual constraint (replayed to index 10525)

A local BFS matching `compute_reachable_costs` expansion rules shows:

- **`(11, 7)` is not in the raw visited set** before the end-tile filter.
- Expanding from `(11, 8)` toward `(11, 7)` would add **+1** MP (shoal → shoal), total **3**, within cap **3**.
- The expansion step **`(11, 8) → (11, 7)` is blocked** because `get_unit_at(11, 7)` returns an **enemy** unit:

  - **`Unit(BLACK_BOAT, P0, hp=100, pos=(11, 7), moved=False)`**

So the infantry cannot **enter** `(11, 7)`: enemy-occupied tiles are skipped in the BFS (`occupant is not None and occupant.player != unit.player` → `continue`). The failure is **occupancy / line-of-move**, not insufficient MP for a clear path.

---

## Section 3 — Root cause classification

| Classification | Applies? |
|----------------|----------|
| **KOAL_COP** | **No** — neither seat is Koal (`co_id` 21); P1 has no COP/SCOP active at the failure. |
| **OTHER_CO_POWER** | **No** — not a movement-budget mismatch from an unimplemented Sami (or other) power on this half-turn. |
| **DEEPER_DESYNC** | **Yes** — trace expects a legal **WAIT** after moving to `(11, 7)`, but the engine places a **P0 Black Boat** on `(11, 7)`, so the tile is **not** a valid infantry destination. Either AWBW had that tile empty (boat moved/sunk) or the recorded trace and full-step engine state diverged earlier. |
| **STALE_TRACE** | **Unproven here** — would require comparing to a fresh site export; the more direct evidence points to **state drift** vs trace, not necessarily a stale file format. |

**Primary label:** **DEEPER_DESYNC** (wrong unit occupancy on destination vs trace-expected legal end tile).

---

## Section 4 — Recommended next action

**Trace ownership:** Treat replay **182065** like other desync triage: find the **first** half-turn where **P0 Black Boat** position (or survival) **diverges** from what the trace implies for `(11, 7)` at day 59 — combat resolution, build order, unload, or move commit for naval tiles. Fixing **Koal COP MP** (Phase 11J-F2-KOAL) will **not** clear this test; the blocker is **enemy unit on the destination shoal**, not `move_range` cap.

---

## Section 5 — Verdict letter

**D** — **DEEPER_DESYNC** (enemy Black Boat on `(11, 7)`); **not** the same root as Phase 11D-F2 **KOAL_COP** MP gap.

---

*Phase 11C-FU-DRILL complete (read-only).*
