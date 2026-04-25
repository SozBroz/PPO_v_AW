# Phase 11M — SUSPECT Wave 2 recon (`oracle_mode` silent returns, remaining 16)

**Campaign:** `desync_purge_engine_harden`  
**Mode:** read-only drill (no edits to engine, oracle, tests, or data).  
**Scope:** The **16** SUSPECT branches left after Phase 11B (8 of 24 closed: BUILD ×6, JOIN ×1, REPAIR head ×1).  
**Source catalog:** `docs/oracle_exception_audit/phase10o_oracle_mode_silent_return_audit.md`, `logs/phase10o_oracle_mode_audit.json`.  
**Current code:** `engine/game.py` as of this recon; **line numbers below are current** (Phase 11B shifted ranges vs the JSON’s original 10O snapshot).

---

## Section 1 — Per-branch drill table (16 rows)

| ID | File:line (current) | Function | Precondition that fails | Behavior (`oracle_mode=True`, STEP-GATE off) | Dirty / inconsistent state | Risk | Replay evidence (10Q / extras) |
|----|---------------------|----------|-------------------------|---------------------------------------------|----------------------------|------|--------------------------------|
| S10O-01 | `engine/game.py` **917–918** | `_apply_wait` | No unit at `action.unit_pos` | `return` | No `_move_unit`, no `_finish_action`, no `game_log`; `action_stage` / selection unchanged | **HIGH** — full no-op masks unit missing at envelope tile | None direct |
| S10O-02 | **976–977** | `_apply_dive_hide` | No unit at `unit_pos` | `return` | Same class as WAIT | **HIGH** | None direct |
| S10O-03 | **978–979** | `_apply_dive_hide` | Unit type `can_dive` is false | `return` | Full no-op | **MED** — envelope/unit-type disagreement; 10F causal **N** | None direct |
| S10O-05 | **1084–1086** | `_apply_repair` | `target_pos is None` after optional move | `_finish_action(boat)` then `return` | Turn ends; no heal/resupply/log; AWBW may still expect a target | **HIGH** — treasury/HP path skipped vs site | None direct |
| S10O-06 | **1088–1091** | `_apply_repair` | No unit at `target_pos` or wrong `player` | `_finish_action(boat)` | Turn ends; repair skipped | **HIGH** | None direct |
| S10O-07 | **1092–1097** | `_apply_repair` | `target.unit_id == boat.unit_id` (self-target) | `_finish_action(boat)` | Turn ends; no repair | **LOW** — scripted edge; “valid” AWBW path unlikely | None direct |
| S10O-08 | **1102–1104** | `_apply_repair` | Boat and target not Manhattan-1 adjacent | `_finish_action(boat)` | Turn ends; resupply/heal skipped | **HIGH** — adjacency drift after move | None direct |
| S10O-09 | **1232–1233** | `_apply_load` | Mover **or** transport missing at `unit_pos` / `move_pos` | `return` | No load, no stage clear in `_apply_load` (caller may still advance inconsistently vs AWBW stack) | **HIGH** — strong stack/ID drift signal | **Weak / tangential:** `games_id` **1605367** (`approx_action_kind` **Load**) — fails earlier with `Illegal move: ... not reachable` (`engine_bug`), not this silent branch |
| S10O-11 | **1344–1345** | `_apply_unload` | `transport` None or `move_pos` / `target_pos` None | `return` | Full no-op | **HIGH** | None direct |
| S10O-12 | **1347–1348** | `_apply_unload` | `loaded_units` empty | `return` | Full no-op | **MED/HIGH** — AWBW thinks cargo exists | None direct |
| S10O-13 | **1356–1357** | `_apply_unload` | No cargo matching `action.unit_type` | `return` | Full no-op (before transport move) | **MED** — ordering/type drift | None direct |
| S10O-14 | **1369–1370** | `_apply_unload` | Drop tile not orthogonally adjacent to transport **after** `_move_unit` | `return` | **Transport may already have moved** (1361–1362); cargo still aboard; stage not finished | **HIGH** — partial commit / position drift | None direct |
| S10O-15 | **1371–1372** | `_apply_unload` | `target_pos` OOB | `return` | Same partial-commit hazard if transport moved | **MED** — often corrupt envelope | None direct |
| S10O-16 | **1373–1374** | `_apply_unload` | Drop tile occupied | `return` | Same partial-commit hazard | **HIGH** | None direct |
| S10O-17 | **1379–1380** | `_apply_unload` | `effective_move_cost >= INF_PASSABLE` for cargo on drop terrain | `return` | Same partial-commit hazard | **HIGH** — terrain/weather walkability mismatch | None direct |
| S10O-24 | **685–693** | `_apply_attack` | After `_move_unit`, no defender; `_apply_seam_attack` returns **False** | `_finish_action(attacker)` without combat | Turn consumed, no damage; masks empty-tile fire vs AWBW | **HIGH** for HP/combat parity; **MED** for funds-first 10F | Many **Fire** rows are **move drift** or **range** `ValueError`s — **no** register line explicitly attributes to “empty non-seam fire” silent path |

**Evidence sweep:** `grep` on `logs/desync_register_post_phase10q.jsonl` and `logs/desync_register_extras_baseline.jsonl` for unload/repair/wait/dive/cargo semantics found **no** rows whose `message` clearly maps to these silent returns. Only **Load**-adjacent hit is **1605367** (different failure mode).

---

## Section 2 — Tightening proposal per branch

Convention: mirror Phase 11B — under `oracle_strict=True`, raise `IllegalActionError` with a **stable, grep-friendly** message prefix; preserve silent `return` when `oracle_strict=False` unless a separate bugfix lane changes default (not in scope here).

| ID | Proposal | Rationale |
|----|----------|-----------|
| S10O-01 | **TIGHTEN** | Missing unit at WAIT envelope is always audit-worthy in strict oracle. |
| S10O-02 | **TIGHTEN** | Same as WAIT for DIVE_HIDE. |
| S10O-03 | **TIGHTEN** | Wrong-type DIVE_HIDE is a hard oracle/engine disagreement; strict should surface. |
| S10O-05 | **TIGHTEN** | REPAIR without `target_pos` after move should not silently “end turn healed nothing.” |
| S10O-06 | **TIGHTEN** | Missing / enemy repair target — same. |
| S10O-07 | **TIGHTEN** | Rare but cheap to enforce; eliminates “silent self-repair” garbage. |
| S10O-08 | **TIGHTEN** | Adjacency failure is core legality. |
| S10O-09 | **TIGHTEN** | Missing endpoints — strongest stack drift signal. |
| S10O-11 | **TIGHTEN** | Null transport or coordinates — same family as LOAD. |
| S10O-12 | **TIGHTEN** | UNLOAD with empty cargo — AWBW/export disagreement. |
| S10O-13 | **TIGHTEN** | Cargo type not aboard — ordering/export bug. |
| S10O-14 | **TIGHTEN** | **Critical:** strict should fire after bad drop geometry (consider whether to raise *before* move in a later refactor — out of scope for this recon). |
| S10O-15 | **TIGHTEN** | OOB drop — strict oracle should fail loud. |
| S10O-16 | **TIGHTEN** | Occupied drop — core stack parity. |
| S10O-17 | **TIGHTEN** | Impassable drop for cargo — terrain parity. |
| S10O-24 | **TIGHTEN** (implementation **deferred**) | Empty non-seam target after move — strict should catch; **do not land until Phase 11J-FIRE-DRIFT clears** (see Section 4). |

**DELETE:** **None** — every branch is a reachable guard for hand-crafted actions, oracle drift, or export bugs.

**JUSTIFY-only (no strict raise):** **Not recommended** for these 16 — silent success hides the same classes Phase 10O flagged. At most, **S10O-07** could be documented as ultra-rare if strict noise ever appears in golden replays (unlikely).

---

## Section 3 — Test design (mirror `tests/test_oracle_strict_apply_invariants.py`)

For each **TIGHTEN**, add paired tests where useful: `oracle_strict=True` → `IllegalActionError` (match substring); `oracle_strict=False` → no raise and assert state consistent with today’s silent path (especially UNLOAD partial-move cases).

| ID | Suggested test name | Scenario sketch |
|----|---------------------|-----------------|
| S10O-01 | `test_apply_wait_missing_unit_at_unit_pos_oracle_strict_raises` | Empty map tile at `unit_pos`; `ActionType.WAIT` with valid `move_pos`. |
| S10O-02 | `test_apply_dive_hide_missing_unit_oracle_strict_raises` | Same pattern for `ActionType.DIVE_HIDE`. |
| S10O-03 | `test_apply_dive_hide_unit_cannot_dive_oracle_strict_raises` | Infantry (or any `can_dive=False`) at `unit_pos`. |
| S10O-05 | `test_apply_repair_black_boat_missing_target_pos_oracle_strict_raises` | Black Boat at `unit_pos`, `target_pos=None` after move setup. |
| S10O-06 | `test_apply_repair_target_missing_or_enemy_oracle_strict_raises` | Target tile empty or opponent unit. |
| S10O-07 | `test_apply_repair_self_target_oracle_strict_raises` | Craft `target_pos` so `get_unit_at` resolves to boat (non-orthogonal cheat via wrong data if needed — may require contrived `unit_id` equality). |
| S10O-08 | `test_apply_repair_target_not_adjacent_oracle_strict_raises` | Boat and ally separated by >1 Manhattan. |
| S10O-09 | `test_apply_load_missing_mover_or_transport_oracle_strict_raises` | `unit_pos` empty or `move_pos` without transport. |
| S10O-11 | `test_apply_unload_null_transport_or_positions_oracle_strict_raises` | Null `move_pos` or `target_pos` or missing transport. |
| S10O-12 | `test_apply_unload_no_cargo_oracle_strict_raises` | Transport with empty `loaded_units`. |
| S10O-13 | `test_apply_unload_cargo_type_not_aboard_oracle_strict_raises` | Transport holds INF only; action requests MECH drop type. |
| S10O-14 | `test_apply_unload_drop_not_adjacent_after_move_oracle_strict_raises` | Transport must move (`pos != move_pos`); `target_pos` diagonal or distance 2. |
| S10O-15 | `test_apply_unload_drop_out_of_bounds_oracle_strict_raises` | `target_pos` (-1,0) or past map edge. |
| S10O-16 | `test_apply_unload_drop_tile_occupied_oracle_strict_raises` | Friendly/enemy unit on drop tile. |
| S10O-17 | `test_apply_unload_drop_terrain_impassable_for_cargo_oracle_strict_raises` | e.g. sea tile for ground cargo (map-dependent). |
| S10O-24 | `test_apply_attack_empty_tile_non_seam_oracle_strict_raises` | **Deferred** with FIRE-DRIFT — attacker moves onto legal tile; `target_pos` empty plain; seam false. |

---

## Section 4 — Coordination notes

- **`_apply_attack` (S10O-24)** is **actively owned by Phase 11J-FIRE-DRIFT** (Opus). Current `game.py` already contains **11J** attacker-selection and ammo-oracle commentary in `_apply_attack` (lines ~631–683). **Recommendation:** do **not** land `oracle_strict` / `IllegalActionError` tightening for S10O-24 until that lane merges or explicitly yields — merge conflict and semantic overlap risk are high.
- **Safe parallel lane (Wave 2 implementation):** `_apply_wait`, `_apply_dive_hide`, `_apply_repair` (tail guards S10O-05..08 only — head S10O-04 already strict in 11B), `_apply_load`, `_apply_unload`. Thread `oracle_strict` into `step()` → these `_apply_*` signatures consistent with `_apply_build` / `_apply_join` / `_apply_repair` pattern.

---

## Section 5 — Priority ranking

**Method:** Descending **severity of masked drift** (full no-op < partial unload after move < repair skip with turn end). **Replay evidence** from 10Q/extras is **sparse** for this slice; ranking is **risk-first**.

### Top 5 — ship in Wave 2 implementation

1. **S10O-14** — UNLOAD bad adjacency **after** transport move (partial state).  
2. **S10O-16** — UNLOAD occupied drop **after** move.  
3. **S10O-17** — UNLOAD impassable terrain **after** move.  
4. **S10O-15** — UNLOAD OOB **after** move.  
5. **S10O-09** — LOAD missing mover/transport (clean no-op but high diagnostic value for stack drift).

### Middle 6 — ship in Wave 3

6. **S10O-01** — WAIT missing unit.  
7. **S10O-05** — REPAIR missing `target_pos`.  
8. **S10O-06** — REPAIR bad/missing target unit.  
9. **S10O-08** — REPAIR not adjacent.  
10. **S10O-02** — DIVE_HIDE missing unit.  
11. **S10O-11** — UNLOAD null transport/positions.

### Bottom 5 — JUSTIFY or DELETE (this recon: **JUSTIFY doc-only** if strict deferred; **no DELETE**)

12. **S10O-12** — UNLOAD empty cargo (often secondary to earlier drift).  
13. **S10O-13** — UNLOAD cargo type mismatch.  
14. **S10O-03** — DIVE_HIDE wrong unit type (10F **N**).  
15. **S10O-07** — REPAIR self-target (rare).  
16. **S10O-24** — ATTACK empty non-seam — **TIGHTEN deferred** to **11J-FIRE-DRIFT** (not “low value,” **blocked**).

---

## Section 6 — Wave 2 implementation lane proposal

| Field | Proposal |
|-------|----------|
| **Lane name** | `Phase-11M-W2-apply-strict-transport-wait-repair` (or shorter: `11M-STRICT-W2`) |
| **Owner** | Opus (implementation agent after this recon) |
| **Files** | `engine/game.py` (`step` signature/wiring, `_apply_wait`, `_apply_dive_hide`, `_apply_repair`, `_apply_load`, `_apply_unload`), `tests/test_oracle_strict_apply_invariants.py` (or split file if >~400 lines) |
| **LOC estimate (engine)** | **~55–85** — five functions × `oracle_strict` param + 1–3 raises each + `step` plumbing (pattern exists from 11B). |
| **LOC estimate (tests)** | **~130–200** — ~8–15 lines per branch × 16 rows, minus deferred S10O-24; many helpers reusable from existing test module. |
| **Risk note** | UNLOAD branches **1369–1380** may need **extra** `oracle_strict=False` assertions that transport position/cargo match pre-recon expectations when silent-returning after a move — golden behavior must be frozen before tightening. |

---

## Section 7 — Verdict

**YELLOW → GREEN path**

- **YELLOW** components: (1) **S10O-14..17** partial-move semantics need careful regression tests; (2) **no strong replay fingerprints** in 10Q for these exact branches — yield is **preventive / audit**, not “closes known gid.”  
- **GREEN** components: **no** architectural blockers; Phase 11B proved the `oracle_strict` pattern; scope is contained to named `_apply_*` functions with clear `IllegalActionError` messages.

**Overall letter for the campaign brief:** **GREEN** (proceed), with **unload-after-move** flagged as the only flank needing extra test rigor.

---

## Return summary (parent agent)

| Metric | Value |
|--------|------:|
| **TIGHTEN** (primary recommendation) | **16** (including S10O-24 — **implementation deferred**) |
| **JUSTIFY** (exclusive; doc-only silent) | **0** |
| **DELETE** | **0** |
| **Branches with strong replay evidence** | **0** direct; **1 weak/tangential** (**1605367** Load — not silent-return path) |
| **Top 5 priorities (1 line each)** | (1) S10O-14 unload bad adjacency post-move. (2) S10O-16 occupied drop post-move. (3) S10O-17 impassable drop post-move. (4) S10O-15 OOB drop post-move. (5) S10O-09 load missing endpoints. |
| **Estimated LOC (top-5 implementation)** | **~45–70** `game.py` + **~90–130** tests (rough order-of-magnitude; unload scenarios dominate fixture size). |
| **Verdict letter** | **GREEN** (with unload staging **YELLOW** caution) |

---

*“The thing that is really hard, and really amazing, is giving up on being perfect and beginning the work of becoming yourself.”* — Anna Quindlen, *A Short Guide to a Happy Life* (1998)  
*Quindlen: American journalist and novelist; often read as a compact on authenticity over performance.*
