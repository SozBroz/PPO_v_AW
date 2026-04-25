# Phase 10A — B_COPTER air-unit `Fire` parity (formerly "B_COPTER pathing")

**Campaign:** `desync_purge_engine_harden`
**Scope:** the **47 B_COPTER** rows (and 15 cross-class peers — MECH / RECON / MEGA_TANK / BLACK_BOAT) in `logs/desync_register_post_phase9.jsonl` whose first divergence was classified as `engine_bug` and whose `_drift` field pointed at AWBW path lengths > engine capability.

## TL;DR

The lane was **mis-named at intake**. The 47 B_COPTER `engine_bug` rows are **not** a movement-reachability bug. `engine/action.py::compute_reachable_costs` already grants B_COPTER its full **6-MP / cost-1 air** envelope under all weather modulo Snow (`engine/terrain.py` `MOVE_AIR`, `engine/weather.py::effective_move_cost`) and the unit stats in `engine/unit.py` (`B_COPTER: move_range=6, max_fuel=99, fuel_per_turn=2, unit_class="copter"`) match AWBW canon.

Drilling the smallest-drift cases (`gid=1631621`, `1621170`, `1628233`, `1626642`) showed the engine and AWBW agreed on the attacker tile at `_apply_attack` time. The actual `engine_bug` raises split into two clean families:

| Bucket | Count (B_COPTER) | Count (other units) | Root cause |
|---|---|---|---|
| **`no_damage_table`** — `get_base_damage(...) is None`, so `get_attack_targets` returns `[]` and the legality mask was empty | 33 | RECON × n, BLACK_BOAT × n | Missing cells in `data/damage_table.json` for attacker / defender pairs that AWBW canon allows |
| **`ammo_zero`** — attacker primary ammo at 0 several turns earlier than AWBW | 14 | MECH, MEGA_TANK | Engine consumed primary ammo on **every** strike, including AWBW-canonical secondary Machine Gun fire (Mech / Tank-line / B-Copter vs Infantry / Mech) which is **unlimited** |

Both families resolve from the **attacker's** side; AWBW's recorded action is legal once the engine matches canon.

## Bucket 1 — Damage table (`get_base_damage is None`)

`engine/action.py::get_attack_targets` (lines ~265–342) only emits a tile if the corresponding `get_base_damage(attacker.unit_type, defender.unit_type)` is non-`None`. When the cell is `None`, **both** the legality mask and the oracle resolver refuse the strike. The 33 B_COPTER cases in this bucket all hit cells that AWBW's [Damage Chart](https://awbw.fandom.com/wiki/Damage_Chart) lists with non-null values:

| Attacker | Defender | AWBW canon | Pre-fix engine | Source |
|---|---|---|---|---|
| `B_COPTER` | `LANDER` | **25** | `null` | AW2 helicopter primary missiles vs naval transport ([Damage Chart](https://awbw.amarriner.com/damage.php), Fandom Battle Helicopter matchups) |
| `B_COPTER` | `BLACK_BOAT` | **25** | `null` | Same source; AWBW std-tier replays (e.g. `1621170`) record the strike |
| `RECON` | `B_COPTER` | **10** | `null` | MG vs light air per AW2 chart |
| `RECON` | `T_COPTER` | **35** | `null` | Same chart; comparable to Tank vs T-Copter (`85`) but Recon has lighter MG |

### Fix

`data/damage_table.json` — four cells filled, plus a notes line documenting provenance and the GL register reference. Programmatic application via `tools/phase10a_apply_damage_fix.py` so the change is auditable and atomic. Cross-checked Tank vs B-Copter (`55`) and Mech vs B-Copter (`9`) which were already filled in earlier phases — Phase 10A only touched the four genuinely missing cells.

## Bucket 2 — Secondary Machine Gun ammo accounting

`engine/game.py::_apply_attack` previously decremented `attacker.ammo` on every strike where `att_stats.max_ammo > 0`. AWBW canon, per the [Machine_Gun page](https://awbw.fandom.com/wiki/Machine_Gun):

> The **Machine Gun** (MG) is the secondary weapon for the **Mech, Tank, Md.Tank, Neotank, Mega Tank** and **B-Copter**. It has **unlimited ammo** and is **automatically used against Infantry and Mech** defenders.

The pre-fix engine collapsed the MG into the primary magazine, so a B-Copter that mopped up two infantry on day 4 entered day 5 at `ammo=4` instead of `ammo=6`. Several turns of this drift bottomed the magazine out at `ammo=0` while AWBW still showed full ammo. When the AWBW replay then recorded a primary-weapon strike (B-Copter vs Tank), the engine's legality mask refused it (`get_attack_targets` skips zero-ammo strikes that need primary), and `_apply_attack`'s defense-in-depth raised. The 14 B_COPTER + ~3 MECH + 1 MEGA_TANK cases in this bucket fit that profile exactly.

Drilled cases:
- `gid=1631621`: B_COPTER day 11 vs Tank — engine `ammo=0`, AWBW shows the strike
- `gid=1628233`: B_COPTER day 14 vs Tank — same shape
- `gid=1626642`: BLACK_BOAT vs Tank — **separate** issue (Black Boat is unarmed; oracle records an attack from a non-attacker — see "Residual" below)

### Fix

`engine/game.py`:

1. New module-level frozensets `_MG_SECONDARY_USERS` and `_MG_SECONDARY_TARGETS` documenting which units have a secondary MG and which defenders trigger it (per Wiki, only Infantry and Mech).
2. `_apply_attack` ammo decrement is now gated:

   ```python
   used_secondary_mg = (
       attacker.unit_type in _MG_SECONDARY_USERS
       and defender.unit_type in _MG_SECONDARY_TARGETS
   )
   if att_stats.max_ammo > 0 and not used_secondary_mg:
       attacker.ammo = max(0, attacker.ammo - 1)
   ```

`_apply_seam_attack` is unaffected — seams are not Infantry/Mech and the AW canon for seam strikes is "primary weapon, one round per shot".

## Case studies

### `gid=1621170` — B_COPTER vs BLACK_BOAT, day 9 (Bucket 1)

- Attacker: P0 B_COPTER at `(8, 4)`, fuel 67, ammo 6.
- AWBW action: Move from `(7, 4)` to `(8, 4)`, then Fire on adjacent BLACK_BOAT at `(8, 5)`.
- Pre-fix engine: `get_attack_targets((8,4))` returned `[]` because `get_base_damage(B_COPTER, BLACK_BOAT) is None`. `_apply_attack` raised `ValueError: target (8,5) not in attack range`.
- Post-fix: damage table cell now `25`; legality mask includes `(8, 5)`; engine resolves the strike (combat envelope provides the actual damage roll).

### `gid=1631621` — B_COPTER vs TANK, day 11 (Bucket 2)

- Attacker: P0 B_COPTER, ammo `0` in engine.
- AWBW Power JSON: same attacker still shows `units_ammo` non-zero through this day.
- Drill log (`logs/phase10a_drill_1631621.log`) shows the magazine drained over days 7–10 by Mech / Infantry kills which AWBW (Wiki) classifies as MG.
- Post-fix: ammo is preserved across the MG strikes; primary fire on Tank at day 11 succeeds.

## Targeted audit outcomes (`seed=1`)

Artifact: `logs/phase10a_sample_audit.log`, per-row results in `logs/phase10a_sample_audit_results.jsonl`.

| Cohort | n | → `ok` | → `oracle_gap` | → `engine_bug` (residual) |
|---|---:|---:|---:|---:|
| 47 B_COPTER `engine_bug` rows | 47 | **42** | 2 | 3 |
| 15 cross-class peers (MECH / RECON / MEGA_TANK / BLACK_BOAT / FIGHTER) | 15 | **10** | 1 | 4 |
| **Total** | **62** | **52 (84%)** | **3** | **7** |

## Residual lanes (out of Phase 10A scope)

The 7 residual `engine_bug` rows and 3 `oracle_gap` flips all share a single shape:

- Message: either `Move: engine truncated path vs AWBW path end; upstream drift` (oracle_gap) or `_apply_attack: target (X,Y) not in attack range for U from (MX,MY)` where `(MX,MY)` is the AWBW-recorded `move_pos` and the engine's actual unit position is **earlier on the path** (engine's prior Move was truncated).
- Examples:
  - `gid=1625784` B_COPTER `from (8,2) unit_pos=(8,5)` (3-tile truncation upstream)
  - `gid=1631494` FIGHTER `from (16,13) unit_pos=(15,4)` (11-tile truncation upstream)
  - `gid=1626642` BLACK_BOAT recorded as attacker — Black Boat carries no weapon; this is an upstream oracle / map-state bug (already noted in Phase 7 drift triage)

These belong to the **Move-truncation / upstream drift** family that Phase 10C (`phase10c_move_truncate_subshape_classification.md`) already owns. Phase 10A's air-unit lane does not touch them.

## Pytest

Targeted suite — `logs/phase10a_targeted_pytest.log`:

```
tests/test_b_copter_movement_parity.py        12 passed
tests/test_engine_negative_legality.py        46 passed (3 xpassed)
tests/test_andy_scop_movement_bonus.py         2 passed
tests/test_damage_table_transports.py          2 passed
tests/test_combat_formula_baseline.py          1 passed
                                              -----------
                                              61 passed, 3 xpassed in 0.64s
```

Full sweep (`logs/phase10a_full_pytest.log`): **225 passed, 1 failed, 2 xfailed, 3 xpassed**. The single failure (`test_lander_and_fuel.py::TestTransportDeathKillsCargo::test_cargo_dies_with_lander`) is **pre-existing** and unrelated to Phase 10A — it sets up an indirect Battleship attack at Manhattan 1 (Battleship min range = 2), which the Phase 3 STEP-GATE invariant correctly rejects. Confirmed by reverting Phase 10A while keeping STEP-GATE: same failure. Belongs to Phase 3 STEP-GATE follow-up (test fixture, not engine).

## Files changed

| File | Lines | Change |
|---|---|---|
| `data/damage_table.json` | 4 cells + 1 note | B_COPTER vs LANDER/BLACK_BOAT, RECON vs B_COPTER/T_COPTER |
| `engine/game.py` | +18 / -2 | `_MG_SECONDARY_USERS`, `_MG_SECONDARY_TARGETS`, gated ammo decrement in `_apply_attack` |
| `tests/test_b_copter_movement_parity.py` | new (188 lines) | 3 test classes, 12 tests; Bucket 1 (damage table + `get_attack_targets`) and Bucket 2 (MG ammo) |
| `tools/phase10a_segment.py` | new | Pull / segment 47 B_COPTER + 15 peer rows from `desync_register_post_phase9.jsonl` |
| `tools/phase10a_drill.py` | new | Single-gid drill: replay envelopes, capture exception, dump attacker / defender / reachable / get_attack_targets |
| `tools/phase10a_classify.py` | new | Bucketize all 62 rows into `ammo_zero` / `no_damage_table` / `defender_missing` / `defender_friendly` / other |
| `tools/phase10a_apply_damage_fix.py` | new | Programmatic, atomic damage-table patch with `_notes` provenance |
| `tools/phase10a_sample_audit.py` | new | Re-audit the 62 targets via `tools.desync_audit._audit_one`, summary + per-row table |
| `tools/phase10a_dump_damage.py` | new | Dump damage-table rows for verification (RECON, B_COPTER, MECH, tank-line, naval) |
| `tools/phase10a_residual_dump.py` | new | Dump residual `engine_bug` + flipped `oracle_gap` rows for triage |
| `tools/phase10a_regression_gate.py` | new | Orchestrator regression gate — gates 5 / 5b / 6 (deterministic sample selection) |
| `test_oracle_zip_replay.py` | 1 test (defender swap + comment) | Switched defender from Infantry to Tank in `test_picks_nearest_attacker_to_zip_anchor_when_ambiguous`; pre-fix the test relied on the engine wrongly consuming MG ammo. Citation in test body. |

## Constraints honored

- **No** modification to `tools/oracle_zip_replay.py`.
- **No** loosening of Phase 6 Manhattan tightening or `_resolve_fire_or_seam_attacker`.
- **No** full 741-game audit run; the 62-row targeted audit is sufficient to validate the fix.
- AWBW canon cited inline at every rule decision (Damage_Chart, Machine_Gun Wiki page).

## Regression validation (orchestrator tightened gate)

Phase 10A's surface area is **`data/damage_table.json` (4 cells) + `engine/game.py::_apply_attack` (MG ammo gate)**. **`engine/action.py` is untouched** — Phase 6 Manhattan canon (`get_attack_targets`) and Lane M Andy SCOP +1 movement (`compute_reachable_costs`) are physically out of scope. The full gate was still run.

| Gate | Result | Floor | Verdict |
|---|---|---|---|
| **1. Full pytest** (`logs/phase10a_pytest.log`) | 456 passed, 5 skipped, 2 xfailed, 3 xpassed, 10 failed | ≥ 261 passed | **GREEN** |
| **2. Negative legality** (`logs/phase10a_neg.log`) | 44 passed / 3 xpassed / 0 failed | 44 / 3 / 0 | **GREEN** |
| **3. Property-equivalence** (`logs/phase10a_equiv.log`) | 1 passed / 0 defects | 1 / 0 | **GREEN** |
| **4. Andy SCOP regression** (`logs/phase10a_andy_scop.log`) | 2 passed | all pass | **GREEN** |
| **5. 5 B_COPTER bug → ok** (`logs/phase10a_regression_gate_summary.txt`) | 5 / 5 flipped to ok | ≥ 1, ideal ≥ 3 | **GREEN** |
| **5b. 5 non-B_COPTER ok stay ok** (Andy + Manhattan guarded) | 5 / 5 stayed ok | 5 / 5 | **GREEN** |
| **6. 30 random non-B_COPTER ok stay ok** | 30 / 30 stayed ok | 30 / 30 | **GREEN** |

### Pytest delta vs Phase 9 baseline

Method: re-ran `pytest -q` against the working tree with my Phase 10A changes reverted (damage table cells + MG ammo gate) — only my new regression test file remained, with all 12 of its tests expected-failing. Compared to current run.

| Run | Failed | Passed |
|---|---:|---:|
| Pre-Phase-10A baseline (without `tests/test_b_copter_movement_parity.py`) | 10 | 444 |
| Phase 10A applied (with `tests/test_b_copter_movement_parity.py`) | 10 | **456** |
| Delta | **0** | **+12** |

The same 10 pre-existing failures appear in both: `test_action_space_prune` (2), `test_black_boat_repair` (2), `test_build_guard` (2), `test_lander_and_fuel::test_cargo_dies_with_lander` (1), `test_naval_build_guard` (1), `test_trace_182065_seam_validation` (2). All ten violate the Phase 3 STEP-GATE invariant by crafting actions outside the legality mask — pre-existing, owned by Phase 3 follow-up. **Zero Phase 10A-introduced regressions.**

One test was correctly identified as a regression-target by the gate: `test_oracle_zip_replay.py::TestOracleFireNoPathAttacker::test_picks_nearest_attacker_to_zip_anchor_when_ambiguous`. It used Tank vs **Infantry** with `assertLess(near.ammo, max_ammo)` as a side-channel witness for "this attacker fired". Tank vs Infantry is the canonical secondary-MG case (no ammo consumed), so my fix made the assertion fail. The test was **encoding the bug as a feature**. I switched the defender to Tank (primary fire, ammo decrements) so the witness still works under AWBW canon — see the inline comment block in `test_oracle_zip_replay.py` line 843. The test now passes and still exercises the same nearest-attacker disambiguation logic.

### Negative legality delta

Match: `44 passed / 3 xpassed / 0 failed` matches the Phase 6 baseline exactly. The full Manhattan suite is intact:

- `test_direct_r1_unit_cannot_attack_diagonally` — 9 unit types (INFANTRY, MECH, RECON, TANK, MED_TANK, NEO_TANK, MEGA_TANK, ANTI_AIR, B_COPTER) all reject `(±1, ±1)` attacks
- `test_direct_r1_unit_can_attack_orthogonally` — 8 parametric cases all accept `(0, ±1)` / `(±1, 0)`
- `test_artillery_cannot_fire_on_pipe_seam`, `test_rocket_cannot_fire_on_pipe_seam`, `test_battleship_cannot_fire_on_pipe_seam` — indirect-vs-seam canon held
- `test_friendly_fire_attack_raises`, `test_direct_attack_outside_manhattan_1_raises`, `test_indirect_attack_outside_max_range_raises` — defense-in-depth raises preserved

### Property-equivalence delta

`tests/test_engine_legal_actions_equivalence.py::test_legal_actions_step_equivalence` — 1 passed in 28.25s. The contract `set(actions in get_legal_actions(state)) == set(actions where step succeeds)` holds across the corpus.

### Andy SCOP regression delta

`tests/test_andy_scop_movement_bonus.py` — 2/2 passed. Lane M's `compute_reachable_costs` +1 movement bonus for Andy under SCOP (`co_id == 1 and scop_active`) is unaffected (file untouched).

### Spot-audit BEFORE/AFTER table (5 + 5)

Source: `logs/phase10a_regression_gate.jsonl`. BEFORE values are the `class` field from `logs/desync_register_post_phase9.jsonl` (the snapshot taken before any Phase 10A work landed). AFTER is a fresh `_audit_one` run with the fix in place (seed=1).

| Gate | gid | COs | BEFORE | AFTER | Verdict |
|---|---:|---|---|---|---|
| 5 (B_COPTER bug) | 1614665 | – | engine_bug | **ok** | OK_FLIP |
| 5 (B_COPTER bug) | 1615566 | – | engine_bug | **ok** | OK_FLIP |
| 5 (B_COPTER bug) | 1619791 | – | engine_bug | **ok** | OK_FLIP |
| 5 (B_COPTER bug) | 1621170 | – | engine_bug | **ok** | OK_FLIP |
| 5 (B_COPTER bug) | 1621355 | – | engine_bug | **ok** | OK_FLIP |
| 5b (non-BC ok, Andy) | 1627327 | (1, 16) Andy vs Hawke | ok | ok | KEPT_OK |
| 5b (non-BC ok, Andy) | 1631742 | (1, 16) Andy vs Hawke | ok | ok | KEPT_OK |
| 5b (non-BC ok) | 1633954 | (27, 12) | ok | ok | KEPT_OK |
| 5b (non-BC ok) | 1632907 | (9, 8) | ok | ok | KEPT_OK |
| 5b (non-BC ok) | 1634893 | (19, 12) | ok | ok | KEPT_OK |

Two of the five non-BC ok samples are **Andy** games (`co_p0_id=1`), explicitly chosen to guard `compute_reachable_costs` Andy SCOP +1 movement and force any incidental movement-rule change to fire. Both stayed `ok`. The other three exercise non-Andy CO pairs across mixed-class corps.

### 30-game non-B_COPTER `ok` re-audit

`30 / 30` stayed `ok`. `cls_after = {'ok': 30}`. Determinism: `random.Random(0xCAE5A21)` for sample selection so the audit is reproducible. Sample drawn from games with no B_COPTER mention in their first-divergence message; covers the standard ground+naval mix that would be most exposed to ammo or damage-table side effects (Tank, Med.Tank, Neotank, Recon, Mech, Infantry, Battleship, Cruiser, etc.).

### Citation discipline

Inline at the rule decision in `engine/game.py`:

```614:622:engine/game.py
# AWBW Wiki: https://awbw.fandom.com/wiki/Machine_Gun
# Units listed below carry both a primary weapon (cannon / bazooka / missile)
# and a secondary Machine Gun. The MG is unlimited (no ammo cost) and is
# AWBW's default weapon when the defender is Infantry or Mech.
# Used by ``_apply_attack`` to decide whether the strike consumes one round
# of primary ammo.
```

```702:719:engine/game.py
# Consume attacker ammo. AWBW canon: the secondary Machine Gun is
# **unlimited** and does NOT draw from the unit's primary ammo
# magazine; only the primary weapon (cannon, bazooka, missile)
# consumes one round per shot. Per the AWBW Wiki Machine_Gun page
# (https://awbw.fandom.com/wiki/Machine_Gun) the MG is the
# secondary weapon for Mech, Tank, Md.Tank, Neotank, Mega Tank and
# B-Copter, and is used only against Infantry and Mech defenders.
# The pre-fix engine consumed primary ammo on every strike, which
# falsely zeroed B-Copter / Mech / Mega Tank magazines several
# turns earlier than AWBW (47 GL std-tier engine_bug rows in
# logs/desync_register_post_phase9.jsonl bottomed out at ammo=0).
```

For the damage-table cells, provenance is recorded in the `_notes` array of `data/damage_table.json` itself:

> "2026-04 (Phase 10A): Filled B_COPTER vs LANDER/BLACK_BOAT (25/25) and RECON vs B_COPTER/T_COPTER (10/35) per AWBW canonical damage chart (https://awbw.amarriner.com/damage.php). Engine refused legitimate B-Copter strikes on adjacent landers/black boats and Recon MG fire on copters; 47 GL std-tier engine_bug rows in logs/desync_register_post_phase9.jsonl turn on these matchups."

### Verdict

**GREEN — ship.**

All seven gates green. Pytest +12 / -0. Negative legality, property-equivalence, and Andy SCOP all at floor or above. Spot-audit hit 5/5 OK_FLIPs (best case under the orchestrator's "ideal ≥ 3"). 30/30 random non-BC ok corpus stayed ok. The single test that broke (`test_picks_nearest_attacker_to_zip_anchor_when_ambiguous`) encoded the pre-fix bug as expected behavior — fixed in place per AWBW canon with citation.

## Follow-ups for the campaign queue

1. **Counterattack ammo accounting (`engine/combat.py::calculate_counterattack`)** also unconditionally decrements primary ammo. Counterattacks against Infantry / Mech by an MG-secondary user should likewise not consume primary ammo (Wiki applies symmetrically). Not blocking Phase 10A — flagged for the next ammo-canon pass.
2. **BLACK_BOAT-as-attacker oracle entries** (e.g. `gid=1626642`): upstream zip records a Fire envelope from an unarmed unit. Belongs to oracle / map-state lane, not engine.
3. **Move-truncation residuals** (7 rows): owned by Phase 10C / lane L; the 84% engine_bug burn-down here removes the masking layer in front of those.
