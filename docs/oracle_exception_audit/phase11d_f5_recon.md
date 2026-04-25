# Phase 11D-F5-RECON — `games_id` 1626642 (BLACK_BOAT / “unarmed attacker” row)

**Mode:** read-only recon (no engine/oracle/data edits)  
**Register baseline:** `logs/desync_register_post_phase10q.jsonl`  
**Triage refs:** `phase11d_residual_engine_bug_triage.md` row 4, `phase10d_non_b_copter_triage.md` Class F

---

## Section 1 — Game data table

| Field | Value |
|--------|--------|
| **games_id** | 1626642 |
| **map_id** | 134930 |
| **map_name** | Swiss Banking (`data/amarriner_gl_std_catalog.json`) |
| **map size** | 17 × 24 (`data/maps/134930.csv`) |
| **tier** | T3 |
| **CO P0 / P1 (catalog)** | `co_p0_id` 28, `co_p1_id` 16 (matchup: Bongus vs Kalanin) |
| **zip** | `replays/amarriner_gl/1626642.zip` (present in workspace) |
| **Failing locator (register)** | `approx_envelope_index` **22**, `approx_day` **12**, `approx_action_kind` **Fire** |
| **Register class / exception** | `engine_bug`, `ValueError` |
| **Full register message** | `_apply_attack: target (2, 3) not in attack range for BLACK_BOAT from (1, 3) (unit_pos=(1, 3))` |
| **Stream counts** | `envelopes_total` 27, `envelopes_applied` 22, `actions_applied` 289 |

---

## Section 2 — Envelope payload decode (failing envelope)

**Indexing:** `parse_p_envelopes_from_zip` → 0-based envelope index **22**, AWBW player id **3759202**, **day 12**. The failing blob is **action index 6** inside that envelope (seventh JSON object).

**Shape:** `action: "Fire"` with non-empty nested `Move` (path + unit global) and `Fire.combatInfoVision`.

**Pseudo-structure (essential fields):**

```json
{
  "action": "Fire",
  "Move": {
    "unit": { "global": { "units_id": 192575774, "units_name": "Md.Tank", "units_x": 3, "units_y": 1, ... } },
    "paths": { "global": [ {x:6,y:2}, {x:5,y:2}, {x:5,y:1}, {x:4,y:1}, {x:3,y:1} ] }
  },
  "Fire": {
    "combatInfoVision": {
      "global": {
        "combatInfo": {
          "attacker": { "units_id": 192575774, "units_x": 3, "units_y": 1, ... },
          "defender": { "units_id": 192534114, "units_x": 3, "units_y": 2, ... }
        }
      }
    },
    "copValues": { ... }
  }
}
```

**Coordinate convention:** AWBW `units_y` / path `y` → engine **row**; `units_x` / path `x` → engine **col**.

- Path **start (engine):** (2, 6) — **MED_TANK** present in engine state before this blob.
- Path **end (engine):** (1, 3) — matches global/combatInfo attacker tile.
- **Defender tile (engine):** (2, 3) — **enemy TANK** in engine snapshot before this blob.

**Verdict on envelope kind:** AWBW genuinely records **`Fire`** with a **Md.Tank** mover and attacker id **192575774**, not `Repair` / `Load` mis-tagged as `Fire`.

---

## Section 3 — Attacker_id resolution analysis

### 3.1 AWBW vs engine ids

- `_unit_by_awbw_units_id(state, 192575774)` → **`None`** before this blob: the live engine does **not** store PHP `units_id` on `Unit.unit_id` for this mover (predeploy / id scheme mismatch is normal; oracle uses geometry + `units_name`).

### 3.2 Oracle mover selection (`Fire` **with** path)

Per `tools/oracle_zip_replay.py` (Fire branch with `paths` non-empty):

- Declared type from `units_name` **`Md.Tank`** → `UnitType.MED_TANK`.
- First position in `((path_start), (global), (path_end))` that passes `_fire_move_mover_ok` is **path start (2, 6)** → **`u` = MED_TANK** at `(2, 6)` (engine `unit_id` **41** in the replayed snapshot).

So **`_resolve_fire_or_seam_attacker` is not the primary chooser here**; the mover is pinned by path + type.

### 3.3 Tile occupancy vs ZIP tail (critical)

Before blob 6 of envelope 22:

| Tile | Engine `get_unit_at` |
|------|----------------------|
| (2, 6) | **MED_TANK** (mover) |
| (1, 3) | **BLACK_BOAT** (friendly) |
| (2, 3) | **TANK** (defender) |

AWBW path tail and combatInfo place the **Md.Tank** on **(1, 3)**. The engine **cannot** reach (1, 3) with the MED_TANK in `compute_reachable_costs` (blocked / occupied), but **`get_attack_targets(state, MED_TANK, (1,3))` still lists (2,3)** as a legal direct-fire target (orthogonal adjacency).

### 3.4 Forced move vs attack step

`GameState.get_unit_at` (`engine/game.py`) scans **all** alive units and returns the **first** whose **`u.pos` equals the queried tile** — there is no separate occupancy grid.

In the Fire-with-path pipeline:

1. `_oracle_resolve_fire_move_pos` initially picks a reachable stance (e.g. **(1, 4)**) that **does not** hit (2, 3).
2. The ZIP-tail preference block sets **`fire_pos = (1, 3)`** because the target **is** in range **from (1, 3)** in `get_attack_targets`, even though (1, 3) is **not** in `compute_reachable_costs`.
3. Because **`fire_pos` ∉ `costs_fire`**, the oracle calls **`state._move_unit_forced(u, (1, 3))`**.
4. **`_move_unit_forced`** only assigns **`unit.pos = new_pos`** for the **MED_TANK** — it does **not** move or eliminate the **BLACK_BOAT** that already sits on **(1, 3)**.
5. After the forced step, **two** alive friendly units both have **`pos == (1, 3)`** (illegal stacking in real AWBW, but possible in this replay path).
6. **`start = u.pos`** is **(1, 3)** for the MED_TANK, but **`get_unit_at(1, 3)`** returns whichever unit the scan hits first — here the **BLACK_BOAT**.
7. **`ActionType.ATTACK`** uses **`unit_pos=(1, 3)`** → `_apply_attack` uses **`get_unit_at` only** (it does **not** consult `get_unit_at_oracle_id` / `select_unit_id`), so it loads the **BLACK_BOAT**, which has **no** damage table vs the defender → **`get_attack_targets` empty** → **`ValueError`** message seen in the register.

**Conclusion:** The register string **“BLACK_BOAT …”** is a **symptom of stacked / ambiguous tile occupancy after `_move_unit_forced`**, not evidence that AWBW attributed the strike to a Black Boat.

---

## Section 4 — Viewer evidence

**Not run** in this recon (optional). Expected AWBW behavior: **Md.Tank** executes the path and fires at the adjacent enemy from **(1, 3)**. C# viewer should **not** show a Black Boat as the striker for this envelope if the export matches the site.

---

## Section 5 — Root cause classification

**Primary:** **ENGINE_DRIFT** (friendly **BLACK_BOAT** still occupies **(1, 3)** in engine state while AWBW’s path tail puts the **MED_TANK** on that hex) **combined with** an **oracle implementation gap**: **`_move_unit_forced` teleports the striker without resolving the incumbent unit on the tile**, yielding **duplicate `pos`** and **`get_unit_at` returning the wrong unit** for **`ActionType.ATTACK`**.

**Not supported by evidence:**

- **BUCKET_B** (wrong `_resolve_fire_or_seam_attacker` substitution): combatInfo and mover resolution agree on **Md.Tank**; Lane I pinning is not the failure mode here.
- **REPLAY_CORRUPT**: envelope is internally consistent (`Fire` + Md.Tank path + combatInfo).
- **Pure “oracle misclassified Repair as Fire”**: envelope is explicitly **`Fire`** with standard combat blocks.

**Relation to Phase 10D “Class F”:** That classification assumed the failure was “unarmed attacker / semantic Fire.” This drill shows the **first-order AWBW payload is a legal MED_TANK Fire**; the **BLACK_BOAT** appears from **stale occupancy + forced placement**, so **F5 as “Black Boat attack” is misleading** for this gid.

---

## Section 6 — Fix lane proposal

### Discarded / low value for this gid

- **11J-F5-ORACLE-FIX (skip BLACK_BOAT in `_resolve_fire_or_seam_attacker`)** — **does not address** this failure: resolver already selects **MED_TANK**; the bug is **post-resolution** ATTACK addressing.
- **11J-F5-DELETE** — **not recommended**: replay appears **valid**; deleting would hide a real **sync** bug.

### Recommended lanes

1. **11J-F5-OCCUPANCY (oracle + small `GameState` / replay helper)** — **~15–40 LOC**:
   - When forcing the MED_TANK onto **(1, 3)**, **displace** the incumbent **BLACK_BOAT** to a legal adjacent sea tile (or nudge per AWBW semantics), **or** issue **ATTACK** with **`select_unit_id` / `unit_pos` tied to the resolved mover `u`** so `_apply_attack` cannot latch onto the wrong stacked unit.
   - **Regression:** replay `1626642` through envelope 22+; add a **unit test** covering **forced move onto a friendly-occupied tile** + Fire.

2. **Phase 11 F1 / nested-Fire board sync (upstream)** — **medium–high LOC** (campaign-scale): reduce the likelihood that **(1, 3)** still holds a **BLACK_BOAT** when AWBW has already moved the **Md.Tank** into that hex in the action stream. This is the **same strategic lane** as Bucket A / Lane G / Lane L, not a separate “Black Boat weapon” lane.

---

## Section 7 — Closure recommendation

| Option | Recommendation |
|--------|------------------|
| **Oracle fix** | **Yes** — treat **forced fire stance + stacked friendly occupancy** (or ATTACK unit addressing) as the surgical defect exposed by this replay. |
| **Engine rules fix** | **No** — Black Boat should not gain an attack; `damage_table` / `get_attack_targets` behavior is correct. |
| **Replay delete** | **No** — file is coherent; failure is tooling/state sync. |
| **Reclassify register row** | After fix, expect this gid to **either** `ok` or fail **earlier** with a **different** message if deeper drift remains; the current **“BLACK_BOAT range”** message should be understood as **grid desync**, not AWBW semantics. |

---

## Section 8 — Verdict letter

**To:** Phase 11 campaign / implementers  
**Re:** F5 residual `1626642` — recon complete  

The **`engine_bug` row for 1626642 is not, in fact, “AWBW recorded a Black Boat attack.”** The zip at envelope **22** shows a standard **`Fire`** by **Md.Tank** **192575774** along a **`Move.paths`** ending at **(1, 3)** into an adjacent enemy at **(2, 3)**. The oracle correctly identifies a **MED_TANK** mover at path start **(2, 6)** but then **forces** that tank’s **`Unit.pos`** to **(1, 3)** **without clearing the BLACK_BOAT already on that tile**. **`get_unit_at(1, 3)`** then returns the **wrong** stacked unit, so **`_apply_attack`** validates range for a **BLACK_BOAT**, producing the misleading register message.

**Classification:** **ENGINE_DRIFT** (stale friendly on firing hex) **+** **OTHER / implementation** (**`_move_unit_forced` creates duplicate `pos`**). **Not Bucket B.** **Not replay corrupt.**

**Recommended fix:** **Resolve incumbent occupancy** when forcing the ZIP firing stance (and/or pin ATTACK to the resolved mover), **optionally paired** with broader **F1** work so the Black Boat is not left on the Md.Tank’s terminal hex in the first place.

**Signed:** Phase 11D-F5-RECON (read-only)

---

*End of recon.*
