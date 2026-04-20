# Engine ⊂ AWBW parity

Single source of truth for known deviations between this engine and the
AWBW reference (https://awbw.amarriner.com / https://awbw.fandom.com).

## Why this matters

Replay validation (oracle zip → engine) covers the **AWBW → engine** direction
extensively (see `docs/desync_audit.md`, `tools/oracle_zip_replay.py`). The
opposite direction — **engine → AWBW** — is what this document tracks. We
want the strict subset relation `engine_legal_actions(s) ⊆ awbw_legal_actions(s)`
for every state `s`. Engine being more *restrictive* than AWBW is acceptable
(it just narrows the RL action space); engine being more *permissive* is a
silent correctness bug because trained policies could learn moves AWBW
would reject, then desync as soon as they touch the live site.

There is no AWBW reference implementation we can query. We approximate
the subset check with two complementary tools:

1. **`tools/engine_awbw_legality_probe.py`** — runtime probe that walks
   `get_legal_actions(state)` and validates each emitted action against
   AWBW canonical predicates (range, ammo, ownership, transport
   compatibility, terrain passability, factory ownership, etc.). Run with
   `random` (random self-play) or `zip` (replay-driven).
2. **`tests/test_engine_awbw_subset.py`** — focused regression tests for
   the specific permissive bugs surfaced by the audit below.

## Audit summary

| Tag | Direction | Severity | Status   |
|-----|-----------|----------|----------|
| A1  | engine > AWBW (permissive) | Severe — free extra action | **Fixed** |
| A2  | engine > AWBW (permissive) | Low — Oozium movement type | Open |
| A3  | engine ≠ AWBW (state drift) | Medium — capture progress | **Fixed** |
| B1  | engine < AWBW (restrictive) | Low — RL shaping       | Documented (kept as RL hint; `step` accepts END_TURN unconditionally) |
| B2  | engine < AWBW (restrictive) | Low — RL shaping       | **Apply path relaxed**, mask kept |
| B3  | engine < AWBW (restrictive) | Low — RL shaping       | **Apply path relaxed**, mask kept |
| B4  | engine < AWBW (restrictive) | Medium — replay parity | **Fixed** |
| B5  | engine < AWBW (restrictive) | Medium — replay parity | **Fixed** |
| C-* | engine ≠ AWBW (state drift) | Variable               | Tracked separately in `desync_register.jsonl` |
| D-* | engine ⊂ AWBW (coverage gap)| Low — not yet exercised| Open (Fog, Flares, predeployed fuel/ammo) |

## A. Permissive (engine > AWBW) — must fix

### A1 — `BUILD` as a Stage-2 ACTION terminator  *(fixed)*

**Symptom.** `_get_action_actions` used to append `Action(BUILD, unit_pos=u, move_pos=mp, unit_type=ut)` whenever the active unit ended its move on its own empty factory. `_apply_build` never invoked `_move_unit` or `_finish_action`, so the acting unit stayed unmoved and remained selectable — a free extra action AWBW does not allow.

**Fix.** Deleted the Stage-2 BUILD branch (`engine/action.py`). Stage-0 factory direct BUILD remains the AWBW-correct path (factory issues build without unit activation).

**Tests.** `tests/test_engine_awbw_subset.py::test_a1_build_never_appears_as_stage2_action`, `::test_a1_stage0_build_still_works`.

### A2 — Oozium movement type  *(open)*

**Symptom.** `UNIT_STATS[OOZIUM]` uses `MOVE_INF`, which lets Oozium walk on mountains and rivers — both forbidden by AWBW's custom Oozium movement type.

**Why deferred.** Adding a new `MOVE_OOZIUM` type plus terrain table entries is mechanical but not exercised by current replays (no Oozium in indexed games). Fix when the first Oozium replay is ingested or when the random probe is extended to maps containing pipes / Oozium predeploys.

**Tracking.** Reproduce with the legality probe by adding an "Oozium reachability against impassable terrain" check in a future PR.

### A3 — `capture_points` not reset on capturer death  *(fixed)*

**Symptom.** When a unit mid-capture (a property's `capture_points < 20`) is killed by a counter-attack on the property tile, or killed as defender on its own mid-capture property, the property kept the partial capture progress. AWBW resets to 20.

**Fix.** Added explicit reset in `_apply_attack` for both attacker (counter-killed on `move_pos`) and defender (killed on `target_pos`). Tile-vacated cases (move-off, LOAD, JOIN) are already handled by `_move_unit`.

**Tests.** `tests/test_engine_awbw_subset.py::test_a3_capture_points_reset_when_defender_dies`.

## B. Restrictive (engine < AWBW) — relaxed where it broke replay parity

### B1 — `END_TURN` requires every unit to have moved

**Where.** `_get_select_actions` only emits `END_TURN` when `has_unmoved is False`.

**Status.** Kept as RL shaping (forces the agent to use every unit). `_apply_end_turn` (`step`) accepts `END_TURN` unconditionally, so oracle replays / hand-built actions are not blocked.

### B2 / B3 — `WAIT` / `DIVE_HIDE` raise on capturable property  *(fixed)*

**Symptom.** `_apply_wait` (and `_apply_dive_hide`) used to `raise ValueError` when the destination tile was a capturable enemy property and the unit could capture. AWBW lets the player decline the capture and just WAIT.

**Fix.** Removed the raise from both apply paths. The pruning in `_get_action_actions` is preserved as RL shaping (drops dominated `WAIT`/`DIVE_HIDE` from the legal mask), but `step` now accepts the action.

**Tests.** `tests/test_engine_awbw_subset.py::test_relax_wait_on_capturable_property_does_not_raise`, plus updates in `test_action_space_prune.py` (`test_step_accepts_*` replacing the old `test_step_rejects_*`).

### B4 — `JOIN` required injured partner  *(fixed)*

**Symptom.** `units_can_join` rejected the join when the *partner* was at full HP, even if the *mover* was damaged. AWBW only forbids the both-full case.

**Fix.** Predicate now requires `mover.hp < 100 OR occupant.hp < 100` (both-full → False).

**Tests.** `tests/test_unit_join.py::test_units_can_join_requires_at_least_one_damaged` (renamed and extended).

### B5 — Carrier could not load anything  *(fixed)*

**Symptom.** `UnitType.CARRIER` was missing from `_LOADABLE_INTO` despite `carry_capacity = 2`. AWBW Carriers hold up to two air units (Fighter / Bomber / Stealth / B-Copter / T-Copter / Black Bomb).

**Fix.** Added the AWBW canonical air-unit list to `_LOADABLE_INTO[CARRIER]`.

**Tests.** `tests/test_engine_awbw_subset.py::test_b5_carrier_loads_air_units`, `::test_b5_carrier_rejects_ground_naval`.

## C. State drift — tracked but not in legality scope

These deviations do not change *which* actions are legal but do change the resulting state (HP, funds, comm-tower bonus, weather counters, power charge, etc.). They surface in the desync audit pipeline (`docs/desync_audit.md`), not in the engine⊂AWBW probe.

Open items as of this writing:

* CO comm-tower bonus only applied for Javier (correct), but tower count refresh after capture has lag.
* Hawke / Drake / Sensei / Sasha power application paths have small numeric drift vs AWBW (rounding, order of operations).
* APC end-of-WAIT resupply timing relative to property repair / income — currently fires before income, AWBW order TBD.
* Black Bomb detonation tile-targeting and Oozium auto-attack/consume not modelled.

## D. Coverage gaps

Features the engine intentionally does not implement:

* Fog of War (vision masking, unit visibility flags). Sub `Hide` / Stealth `Dive` are partially modelled (`is_submerged` toggle) but no fog masking.
* Flares (predeploy + active-weather visibility removal).
* Predeployed unit fuel / ammo overrides — current loader resets to `max_*`. See `engine/predeployed.py`.

These are restrictive (engine offers fewer/simpler actions than AWBW) and therefore safe under engine ⊂ AWBW. They will become parity bugs when training is extended into Fog or Black Hole COs that depend on Flares.

## Running the probe

```powershell
# Random self-play probe (deterministic with --seed)
python tools/engine_awbw_legality_probe.py random --map-id 123858 --turns 60 --seed 0

# Replay-driven probe (validates legal_actions before each oracle step)
python tools/engine_awbw_legality_probe.py zip replays/amarriner_gl/1628539.zip `
    --map-id 171596 --co0 1 --co1 2 --tier T2 --seed 0
```

Exit code is non-zero iff at least one violation is recorded; the summary
groups counts by rule tag and prints the first 20 violations verbatim.
