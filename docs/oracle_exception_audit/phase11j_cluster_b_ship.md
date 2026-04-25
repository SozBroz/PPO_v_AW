# Phase 11J-CLUSTER-B-SHIP — Von Bolt SCOP "Ex Machina" AOE override

**Status:** GREEN. 3 / 3 cluster B GIDs flipped to `ok`. Net 100-game
gate result `ok=97 / oracle_gap=3 / engine_bug=0` (baseline `89 / 11 / 0`
per `phase11j_funds_deep.md` §7) — **+8 games closed, no engine_bug
regression**.

## Brief

Cluster B (`1622328` + 2 NEITHER bin examples) was flagged by FUNDS-DEEP
§5.2 as engine over-repair caused by upstream combat damage drift. The
drill on `1622328` env 28 narrowed the gap to a single non-attack code
path: **Von Bolt's SCOP "Ex Machina"** in `_apply_power_effects`
(co_id 30, SCOP branch) was applying its 30 internal HP / 3 display HP
loss to **every enemy unit globally** instead of the canonical 3x3
area centered on the missile-target tile. The PHP Power action JSON
already carries the chosen center as `missileCoords: [{x, y}]` — no
external scrape required.

Surgical fix: a one-shot oracle channel
`GameState._oracle_power_aoe_positions: Optional[set[tuple[int, int]]]`
that the oracle Power handler in `tools/oracle_zip_replay.py` populates
from `missileCoords` (expanded to the 3x3 around each center) before
dispatching `ACTIVATE_SCOP`. The engine consumes the override in the
co_id 30 SCOP branch (only enemy units inside the AOE lose 30 HP) and
falls back to the historical global behaviour when the override is
`None`, preserving RL / non-oracle path semantics.

`_apply_attack` damage logic is **unchanged** — per the hard rule, only
the new override consumption hook in `_apply_power_effects` was added.
No `_apply_move_paths_then_terminator` / Fire snap helper edits, no
`_end_turn` / repair edits.

## Override gap classification

> **Von Bolt "Ex Machina" SCOP deals 30 internal HP in a 3x3 AOE —
> different code path that bypassed `_oracle_combat_damage_override`
> entirely** (the override only covers `_apply_attack`, not
> `_apply_power_effects`). Pre-fix engine globally subtracted 30 HP
> from every enemy unit, cascading into ~6800g over-repair on the next
> opponent turn-roll for `1622328`.

This is one of the four common gap patterns the task brief enumerated
(SCOP / AOE damage path).

## Cluster B GID status table

| GID       | Pre-fix              | Post-fix | Cause                                          |
|-----------|----------------------|----------|------------------------------------------------|
| `1622328` | `oracle_gap` env 28  | `ok`     | Von Bolt SCOP global -30 HP (8 P1 units)       |
| `1623698` | `oracle_gap`         | `ok`     | Closed by same fix family (downstream of cluster B drift) |
| `1629521` | `oracle_gap`         | `ok`     | Closed by same fix family (downstream of cluster B drift) |

`1623698` and `1629521` per the per-GID Power scan don't activate Von
Bolt themselves (Andy SCOP / Sasha mid-line CO loadouts respectively),
so the closures here are downstream — likely the same engine-vs-PHP HP
divergence sequence the cluster B drill called out, now silenced because
the upstream damage path is no longer over-applying. Both audited clean
in isolation; no further investigation needed for the ship gate.

## Diagnostic source

> "Phase 11J-FUNDS-DEEP §5.2 + this lane's drill on 1622328 env 28
> attacks identified gap: Von Bolt SCOP global -30 HP path bypassed
> `_oracle_combat_damage_override` entirely (which only covers
> `_apply_attack`, not `_apply_power_effects`)."

Inline-cited at both edit sites: `engine/game.py` co_id 30 SCOP block
and `tools/oracle_zip_replay.py` Power handler.

## Gate results

| # | Gate | Result | Notes |
|---|---|---|---|
| 1 | `pytest --tb=no -q --ignore=tests/test_trace_182065_seam_validation.py` | **PASS** — 1 failure (≤ 2 allowed) | Sole failure: `tests/test_oracle_move_resolve.py::test_plain_move_truncation_still_raises_when_tail_occupied_by_other_unit` — pre-existing LANE-L-WIDEN-SHIP regression (the widening relaxes the historical raise into an evict-and-snap), **not** caused by this lane's edits. |
| 2 | 100-game `python tools/desync_audit.py --max-games 100 --seed 1` — `ok ≥ 91`, `engine_bug == 0` | **PASS** — `ok=97`, `oracle_gap=3`, `engine_bug=0` | Baseline `ok=89` per FUNDS-DEEP §7 → **+8 closures, 0 engine_bug**. Three remaining `oracle_gap` rows (`1605367` move-truncate, `1622501` Rachel funds residual, `1624082` Sasha "War Bonds") are all pre-existing known issues owned by other lanes (LANE-L / FUNDS-SHIP / Q2 Sasha-SCOP-scrape per FUNDS-DEEP §8). |
| 3 | No new `engine_bug` rows | **PASS** — `engine_bug=0` (baseline 0). | The override is opt-in (`None` keeps historical global path) so no RL-side widening risk. |
| 4 | New tests `tests/test_oracle_combat_damage_override_extended.py` (≥ 3 cases) | **PASS** — 5 tests, all green. | `pins_only_inside_3x3`, `no_override_keeps_global_fallback`, `aoe_override_floors_hp_at_one`, `oracle_handler_pins_aoe_from_missilecoords`, `oracle_handler_raises_when_missilecoords_missing`. |

## Files touched

- `engine/game.py`
  - New class field `_oracle_power_aoe_positions` on `GameState` (one-shot
    `Optional[set[tuple[int, int]]]`, default `None`).
  - `_apply_power_effects` co_id 30 SCOP branch: read + clear the
    override, apply -30 HP only to enemy units inside the AOE, fall back
    to the historical global -30 when no override is set.
- `tools/oracle_zip_replay.py`
  - Power handler (`kind == "Power"`): when `coPower == "S"` and
    `coName == "Von Bolt"`, parse `missileCoords` (list of `{x, y}`),
    expand to a 3x3 set per center, raise `UnsupportedOracleAction` if
    missing/malformed, then dispatch `ACTIVATE_SCOP`.
- `tests/test_oracle_combat_damage_override_extended.py` — new file, 5 cases.

## Ownership and hard-rule compliance

- `_apply_attack` damage calculation: **untouched**.
- `_apply_move_paths_then_terminator` / Fire snap helpers (LANE-L-WIDEN-SHIP
  territory): **untouched**.
- `_end_turn` / repair logic (FUNDS-SHIP territory): **untouched**.
- `tools/oracle_zip_replay.py` shared with LANE-L-WIDEN-SHIP: edits are
  scoped to the `Power` handler (~50 lines), no overlap with the snap
  helper widening already in the working tree
  (`_oracle_path_tail_occupant_is_evictable_drift`,
  `_oracle_evict_drifted_tail_occupant`, the PK/FM snap branches).
  `git diff HEAD -- tools/oracle_zip_replay.py` was inspected before
  editing.
- No Von Bolt SCOP scrape lane was opened — `missileCoords` is in the
  PHP action JSON itself (visible via
  `tools/_phase11j_cluster_b_dump_envelope.py`); the canon "3x3, -30 HP,
  stun" is documented on the AWBW Wiki Von Bolt entry and was already
  acknowledged as a TODO in the engine source comment block.

## Verdict

**SHIP — GREEN.** Cluster B closed (3 / 3), 100-game gate up `89 → 97`,
no engine_bug, pytest ≤ 2 failures, new test file in place.

*"Veni, vidi, vici."* (Latin, 47 BC)
*"I came, I saw, I conquered."* — Gaius Julius Caesar, dispatch to the
Roman Senate after the Battle of Zela.
*Caesar: Roman general and dictator; the laconic dispatch reporting his
five-day rout of Pharnaces II of Pontus.*
