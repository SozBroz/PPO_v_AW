# Phase 11D-F2-RECON — F2 `Illegal move: … not reachable` (read-only)

**Scope:** Two Phase 10Q residual `engine_bug` rows (Phase 11D rows 1 and 5) with family **F2** — `compute_reachable_costs` / move commit rejects an AWBW-recorded **Move+Load** path.  
**Baseline:** `logs/desync_register_post_phase10q.jsonl`  
**Mode:** Read-only recon (no engine/oracle/data edits); this note is the deliverable.

---

## Section 1 — Per-game data table

| Field | 1605367 | 1630794 |
|-------|---------|---------|
| **Map id** | 77060 | 133665 |
| **Map name** | A Hope Forlorn | Walls Are Closing In |
| **Map size (rows×cols, local `data/maps/{id}.csv`)** | 19×23 | 21×25 |
| **Tier** | T4 | T4 |
| **CO P0 (id → name)** | 21 → Koal | 14 → Jess |
| **CO P1 (id → name)** | 14 → Jess | 21 → Koal |
| **Failing envelope index** | 32 | 37 |
| **Day (AWBW)** | 17 | 19 |
| **Actions in envelope** | 32 | 38 |
| **Full exception (10Q)** | `Illegal move: Mech (move_type=mech) from (1, 16) to (2, 14) (terrain id=1, fuel=68) is not reachable.` | `Illegal move: Infantry (move_type=infantry) from (2, 7) to (1, 10) (terrain id=46, fuel=99) is not reachable.` |
| **Phase 9 class** | `oracle_gap` — `Move: engine truncated path vs AWBW path end; upstream drift` | Same |
| **Unit / action** | Mech / Load | Infantry / Load |
| **AWBW path (from zip `Move`+`Load`, `paths.global`; x=col, y=row)** | (16,1)→(15,1)→(15,2)→(14,2) → end col 14 row 2 | **Load #2 in envelope (action index 18):** (7,2)→(8,2)→(9,2)→(10,2)→(10,1) → end col 10 row 1 |
| **Engine coords (row, col)** | (1,16) → (2,14) | (2,7) → (1,10) |
| **Manhattan displacement** | 3 | 4 |
| **Fuel at failure (from message)** | 68 | 99 |
| **Power envelope (same envelope, action index 0)** | Koal **COP** “Forced March”, `global.units_movement_points: 1` | Same |

**Terrain along path (local CSV `data/maps/{map_id}.csv`, 0-based row/col):**

**1605367 — edges entered:** (1,15) tid **34** (neutral City), (2,15) tid **1** (Plain), (2,14) tid **1** (Plain). Start tile (1,16) tid **1**.

**1630794 — edges for failing Load (action 18):** (2,8) tid **1**, (2,9) tid **3** (Wood), (2,10) tid **1**, (1,10) tid **37** (neutral Port) per repo CSV. *Register message lists destination terrain id **46** (Blue Moon Port in `engine/terrain.py`); repo CSV has **no** id 46 on this grid — possible map revision drift vs live AWBW or worth re-checking at implementation time.*

**Which Load fails:** For 1630794, coordinates match **action index 18** (long path, `units_id` 192399699), not the shorter first Load (index 17).

---

## Section 2 — Reachability analysis

### Engine rules cited

- `engine/unit.py`: Mech `move_range=2`, Infantry `move_range=3`.
- `engine/action.py::compute_reachable_costs`: builds MP cap from `stats.move_range` plus **explicit** CO bonuses (Adder, Eagle SCOP air, Sami inf, Grimm SCOP ground, Jess vehicles on COP/SCOP, Andy SCOP +1). **Koal is not listed here.**
- `engine/weather.py::effective_move_cost`: Koal (`co_id == 21`) **only** during COP/SCOP applies **−1 / −2 MP per road tile** (`_terrain_category == "road"`), not a blanket `move_range` increase.

### Summed terrain costs (using `get_move_cost` + path above)

**1605367 (Mech):** entering three tiles → costs **1+1+1 = 3** MP (City 34 and Plains 1 all use property/plain-style costs for mech). Requires **MP budget ≥ 3**. Base mech range **2** is insufficient; AWBW snapshot on the same `Move` shows **`units_movement_points": 3`** for the Mech — consistent with **base 2 + global +1 from Koal COP** (`Power` envelope `global.units_movement_points: 1`).

**1630794 (Infantry):** four edges → **1+1+1+1 = 4** MP with current engine tables (Wood tid 3 uses `MOVE_INF: 1` in `_wood_costs()`). Requires **MP budget ≥ 4**. Base infantry **3** is insufficient; snapshot shows **`units_movement_points": 4`** — consistent with **base 3 + global +1 from Koal COP**.

### Conclusion (expected vs engine)

- **AWBW / zip:** Both games activate **Koal COP Forced March** in the **same** half-turn envelope (first action is `Power` with `global.units_movement_points: 1`).
- **Engine reachability:** With `cop_active` set, **road tiles are cheaper**, but **`move_range` is not increased for Koal** in `compute_reachable_costs`. The unit is still capped as if it had **2 MP (Mech)** or **3 MP (Inf)** for the turn when evaluating reachability and `_move_unit` legality.
- Therefore **`(json_path_end) ∉ reach`** for the recorded path costs → oracle previously surfaced **Phase 9** `oracle_gap` (truncation / drift language); **Phase 10Q** surfaces **`Illegal move: … not reachable`** when the stricter move commit path runs first.

---

## Section 3 — Path-shape findings (compound envelope, transport)

Parsed with `tools/oracle_zip_replay.parse_p_envelopes_from_zip` (gzip `a{games_id}` member, `p:` lines).

| Game | Shape | Transport id (PHP `units_id` in `Load`) | Cargo |
|------|--------|-------------------------------------------|--------|
| 1605367 | Standard `action: "Load"` with nested `Move` + `Load` | 190209048 | Loaded unit 190901660 (Mech) |
| 1630794 | Two `Load` actions back-to-back (indices 17 and 18); same transport | 192321147 | 192391607 (short move), 192399699 (long move — failing coordinates) |

Lane L comment in `tools/oracle_zip_replay.py::_apply_move_paths_then_terminator` (lines ~4001–4019) already distinguishes **Load** boarding (snap `selected_move_pos` to JSON tail when friendly load boarding) vs **Join** — **not** the primary failure here: the path tail is unreachable under **understated MP cap** before terminator semantics matter.

---

## Section 4 — Viewer evidence

**Status:** Deferred — no `AWBW Replay Player.exe` found under `third_party/AWBW-Replay-Player` (paths (1)–(4) from `.cursor/skills/desync-triage-viewer/SKILL.md` §4a not present in this workspace).

**Recommended launch (1630794 — class flip, two Loads):** after installing/building the viewer:

```text
"<repo>\third_party\AWBW-Replay-Player\AWBWApp.Desktop\bin\Release\net6.0\AWBW Replay Player.exe"
  "C:\Users\phili\AWBW\replays\amarriner_gl\1630794.zip"
  --goto-envelope=37
```

Optional: `--goto-day=19 --goto-player=3768108` to align with envelope tuple. Step through in-viewer actions to confirm Koal COP banner and the infantry path onto the APC/Lander tile.

**1605367 (same COP pattern):**

```text
... "C:\Users\phili\AWBW\replays\amarriner_gl\1605367.zip" --goto-envelope=32
```

---

## Section 5 — Root cause classification (per game)

| games_id | Classification | Rationale |
|----------|----------------|-----------|
| **1605367** | **OBE — engine MP cap / CO parity** | Mech needs **3** MP for cost-3 path; Koal COP grants **+1** in site `global` and unit snapshot; engine applies Koal **road** discount only, **not** +1 to `move_range`. |
| **1630794** | **OBE — engine MP cap / CO parity** (same mechanism) | Infantry needs **4** MP for cost-4 path under current tile costs; Koal COP +1 missing from `move_range` in `compute_reachable_costs`. Not oracle envelope misread and not replay corruption given consistent `Power` + `Move`+`Load` structure. |

*Optional nuance:* destination `terrain id=46` vs local CSV port id **37** at (1,10) — track under **loader/map revision** only if a fix to Koal MP does not clear the row.

---

## Section 6 — Fix lane proposal

**11J-F2-KOAL-COP-MP (recommended)**  
- **Where:** `engine/action.py::compute_reachable_costs` (and verify no duplicate logic in any move precheck).  
- **What:** When `co.co_id == 21` and `co.cop_active`, add **`move_range += 1`** to mirror AWBW `Power` envelope `global.units_movement_points: 1` for **Forced March**.  
- **SCOP follow-up:** `data/co_data.json` describes Trail of Woe as **+2 on road tiles**; `effective_move_cost` already applies **−2** per road under SCOP. **Do not** blindly add `+2` to global `move_range` without wiki/viewer confirmation — scope COP first (this recon’s smoking gun).

**Not recommended as primary:** Large Phase 9 “path-end snap before Load” patches — both failures are explained by **MP budget**, not missing snap for a reachable path.

**Complexity:** **LOW** for COP +1 only (small, testable change + one replay regression per gid). **MED** if SCOP/global power plumbing must be unified with envelope `global` for multiple COs.

---

## Section 7 — Per-game closure recommendation

| games_id | Recommendation |
|----------|------------------|
| **1605367** | **Engine fix** — Koal COP movement budget parity (`compute_reachable_costs`). **Do not delete** zip. |
| **1630794** | **Engine fix** — same. **Do not delete** zip. |

---

## Section 8 — Verdict letter

**GREEN** — Both rows share a **single, coherent** explanation (Koal **COP** global **+1 MP** missing from reachability cap while road discount exists), a **clear engine fix lane**, and **no** evidence the zips are scuffed.

---

*Phase 11D-F2-RECON complete (read-only).*
