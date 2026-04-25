# Phase 2.5 — Legality reconnaissance

**Campaign:** `desync_purge_engine_harden`  
**Method:** Minimal synthetic `GameState` instances + direct `step()` / `get_attack_targets()` calls. No engine/oracle/test edits.  
**Evidence:** One-off Python probes from repo root with `PYTHONPATH` set to the workspace (outputs captured in the sections below).

---

## Probe 1 — Mech adjacent to seam

**Hypothesis:** `get_attack_targets` lists the seam; `step(ATTACK)` resolves seam damage (AWBW allows **direct** units to attack intact pipe seams).

**Observed engine behavior (exact):**

- Terrain: plain at `(5,4)`, terrain id `113` (HPipe Seam) at `(5,5)`. Mech (P0) at `(5,4)`, no enemy unit on seam.
- `get_attack_targets(state, mech, (5,4))` **includes** `(5,5)`.
- `state.step(Action(ATTACK, unit_pos=(5,4), move_pos=(5,4), target_pos=(5,5)))` completes without raising.
- `seam_hp[(5,5)]` goes **99 → 44**; game log entry `type: "attack_seam"`, `dmg: 55`; Mech ammo **3 → 2**.

**AWBW canon:** Direct-fire units adjacent to an intact seam can attack it (standard AW / AWBW pipe-seam rules; see [Pipe Seam — AWBW Wiki](https://awbw.fandom.com/wiki/Pipe_Seam)). Mech using MG vs seam matches `engine/combat.py::_SEAM_BASE_DAMAGE` (55% base).

**Defect status:** **OK** — consistent with direct-fire seam attacks.

**Phase 3 thread owner:** **SEAM** (context only; no defect filed on this probe).

---

## Probe 2 — All indirect units vs seam

**Hypothesis:** Every `UnitType` with `UNIT_STATS[u].is_indirect == True` is either (a) able to list the seam in `get_attack_targets` when in range, or (b) excluded; behavior should align with `get_seam_base_damage` / campaign expectation that **many indirects must not** target seams on AWBW.

**Indirect types in `UNIT_STATS`:** `ARTILLERY`, `ROCKET`, `MISSILES`, `BATTLESHIP`, `CARRIER`, `PIPERUNNER`.

**Observed engine behavior (exact):**

| Indirect | `get_seam_base_damage` (engine) | Seam in `get_attack_targets` (synthetic layout, in range) |
|----------|----------------------------------|-----------------------------------------------------------|
| ARTILLERY | 70 | **true** |
| ROCKET | 80 | **true** |
| MISSILES | `null` | **false** |
| BATTLESHIP | 80 | **true** |
| CARRIER | `null` | **false** |
| PIPERUNNER | 80 | **true** |

Layouts: land indirects on plains with Manhattan distance to seam `113` within each unit’s min/max range; Battleship/Carrier on sea with seam six tiles away on same row; Piperunner on horizontal pipe tiles leading to seam at `(1,7)` from `(1,5)`.

**AWBW canon:** Campaign plan and engine audit treat **indirect fire on intact pipe seams** as a known parity gap vs AWBW (see `desync_purge_engine_harden` plan Phase 3 Thread SEAM; `.cursor/plans/desync_purge_engine_harden_d85bd82c.plan.md`). Community/wiki expectation: indirect artillery-class units do not use the seam strike the way direct units do; Piperunner is a special case requiring primary-source replay triage in Phase 3.

**Defect status:** ~~**BUG** (for **`ARTILLERY`, `ROCKET`, `BATTLESHIP`, `PIPERUNNER`**): engine currently exposes seam targets where Phase 3 SEAM work expects exclusion or AWBW-verified exceptions. **`MISSILES` / `CARRIER`**: **OK** vs current `_SEAM_BASE_DAMAGE` (no entry → no seam targeting).~~

**Defect status:** ~~BUG~~ → **OK** (overturned by Phase 3 SEAM canon investigation; see `phase3_seam_canon.md` and `phase3_seam.md`).

**Phase 3 thread owner:** **SEAM**

---

## Probe 3 — Range Chebyshev vs Manhattan (Mech & Infantry vs diagonal adjacency)

~~**Hypothesis:** Both 1-range direct units use **Chebyshev** distance 1 (eight neighbors), so a diagonally adjacent enemy at `(dr,dc)=(1,1)` appears in `get_attack_targets`.~~

**Observed engine behavior (exact):** Mech at `(3,3)`, enemy Tank at `(4,4)` → `(4,4) ∈ get_attack_targets`. Infantry same layout → `(4,4) ∈ get_attack_targets`.

~~**AWBW canon:** Matches AWBW/adjacency convention for direct fire (8-directional adjacent attack).~~

~~**Defect status:** **OK** — engine matches AWBW. Mech and Infantry both use Chebyshev distance 1.~~

**AMENDED IN PHASE 6 (2026-04-20):** **BUG → FIXED in Phase 6.** AWBW canon: direct
range-1 units attack at **Manhattan distance 1** (the four orthogonal neighbours only).
The original probe concluded Chebyshev because the engine implementation matched
itself — circular reasoning. Verified by:
- AWBW Wiki ("directly adjacent") — see Phase 6 regression-log entry for URLs
- Carnaghi 2022 ("on axis not diagonally")
- 936 GL std-tier replays: 62,614 direct-r1 Fire envelopes, **zero diagonals**

Fix: `engine/action.py:301-310` collapsed the Chebyshev special case to Manhattan.
Test `test_mech_can_attack_diagonal_chebyshev_1` deleted; replaced with parametrized
`test_direct_r1_unit_cannot_attack_diagonally` covering 9 direct-r1 unit types.

**Phase 3 thread owner:** **NONE**

---

## Probe 4 — `_apply_attack` friendly fire

**Hypothesis:** Crafted `ATTACK` from one P0 unit onto an adjacent P0 ally either **raises** or is rejected; AWBW does not allow attacking own units.

**Observed engine behavior (exact):** No exception. Defender ally HP **100 → 44** (tank vs tank base damage applied).

**AWBW canon:** You cannot order attacks against your own units.

**Defect status:** **BUG**

**Phase 3 thread owner:** **ATTACK-INV**

---

## Probe 5 — `END_TURN` with unmoved unit

**Hypothesis:** `get_legal_actions` omits `END_TURN` while an unmoved non-carved-out unit exists; `step(END_TURN)` may still succeed.

**Observed engine behavior (exact):** Single P0 Infantry at `(2,2)`, `moved=False`. `step(END_TURN)` **does not raise**; `active_player` **0 → 1**; infantry **still** `moved=False`.

**AWBW canon:** Must pass turn only when rules allow (typically all units have acted or been skipped via explicit WAIT chain).

**Defect status:** **BUG** — `step()` does not enforce the same gate as `_get_select_actions` for END_TURN.

**Phase 3 thread owner:** **STEP-GATE**

---

## Probe 6 — `_activate_power` with `power_bar=0`

**Hypothesis:** With charge zero, `can_activate_cop()` is false and `step(ACTIVATE_COP)` should not activate COP.

**Observed engine behavior (exact):** `co_states[0].power_bar = 0`. `can_activate_cop()` → **false**. After `step(ACTIVATE_COP)`, **`cop_active` is true** (COP activated without sufficient charge).

**AWBW canon:** COP requires meter threshold (see CO power rules on [AWBW Wiki — CO Powers](https://awbw.fandom.com/wiki/Category:COs)).

**Defect status:** **BUG**

**Phase 3 thread owner:** **POWER+TURN+CAPTURE**

---

## Probe 7 — `_apply_capture` with non-capturer (Tank)

**Hypothesis:** Tank on neutral city issuing `CAPTURE` should not apply capture logic / ownership flip.

**Observed engine behavior (exact):** Neutral city (`terrain_id` 34) at `(4,4)`, Tank on tile. `UNIT_STATS[TANK].can_capture` is **false**. After `step(CAPTURE)`, `capture_points` **20 → 10**; `owner` remains **null**.

**AWBW canon:** Only Infantry/Mech capture properties.

**Defect status:** **BUG** — capture **progress** was applied for a non-capturing unit type.

**Phase 3 thread owner:** **POWER+TURN+CAPTURE**

---

## Probe 8 — `_apply_build` on enemy-owned factory

**Hypothesis:** `step(BUILD)` on opponent factory **no-ops** and does not debit funds.

**Observed engine behavior (exact):** Blue Moon factory tile (`terrain_id` 44), `owner=1`, P0 active with sufficient funds. After `step(BUILD, move_pos=(5,5), unit_type=TANK)`, `funds[0]` **unchanged** (20000 → 20000), no new unit on P0.

**AWBW canon:** Cannot build from enemy properties.

**Defect status:** **OK** — matches `engine/game.py::_apply_build` guard (`prop.owner != player` → return).

**Phase 3 thread owner:** **NONE**

---

## Summary table

| # | Probe | Defect | Owner |
|---|-------|--------|-------|
| 1 | Mech vs seam | OK | SEAM |
| 2 | All indirects vs seam | ~~BUG~~ → **OK** (Phase 3 canon: `_SEAM_BASE_DAMAGE` / wiki + replays; see `phase3_seam_canon.md`, `phase3_seam.md`; MISSILES/CARRIER still aligned with no chart entry) | SEAM |
| 3 | Mech & Inf diagonal range | ~~OK~~ → **BUG (FIXED)** (Phase 6; see Probe 3 amendment) | NONE |
| 4 | Friendly-fire ATTACK | BUG | ATTACK-INV |
| 5 | END_TURN unmoved unit | BUG | STEP-GATE |
| 6 | COP @ power_bar=0 | BUG | POWER+TURN+CAPTURE |
| 7 | Tank CAPTURE | BUG | POWER+TURN+CAPTURE |
| 8 | BUILD enemy factory | OK | NONE |

### Bug counts by Phase 3 thread owner

| Owner | Count | Probes |
|-------|------:|--------|
| **STEP-GATE** | 1 | 5 |
| **SEAM** | ~~1~~ **0** | ~~2~~ Probe 2 overturned to **OK** (probes 1–2 both OK post–Phase 3 canon) |
| **POWER+TURN+CAPTURE** | 2 | 6, 7 |
| **ATTACK-INV** | 1 | 4 |
| **NONE** | — | 3, 8 |

**Surprises:** (1) **`COP` activates with zero charge** if `step()` is called directly — stronger than “mask mismatch” alone. (2) **Friendly-fire applies full combat damage** with no `ValueError`. (3) **Tank capture shaved 10 capture points** in one step — non-capturers still move the counter.
