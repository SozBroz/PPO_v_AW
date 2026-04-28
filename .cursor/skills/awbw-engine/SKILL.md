---
name: awbw-engine
description: Navigate the AWBW (Advance Wars by Web) Python engine, replay export pipeline, and viewer-compatible zip format. Use when working in D:\AWBW, touching files under engine/, tools/, rl/, replays/, or data/, debugging replays, generating AWBW replay zips, modifying the GameState/Action/Unit model, working with maps/predeployed units, or anything that mentions COs, tiers, snapshots, p: stream, full_trace, or unit_id stability.
---

# AWBW Engine — Orientation Map

A Python re-implementation of Advance Wars by Web with an RL training loop and an exporter that produces zips playable in the AWBW Replay Player.

Read this file first. Read `reference.md` only when you need exact field-by-field details (PHP serialization layout, action JSON shapes, full unit table, CO id list).

Informal **player 1 / player 2** (first vs second human) vs engine **P0/P1** (seats 0/1): see `.cursor/skills/awbw-seat-vocabulary/SKILL.md`.

## Repository layout

```
engine/        # pure game logic, no I/O beyond map loading
tools/         # CLIs: replay export, diagnostics, viewer scraping
rl/            # AI vs AI, encoder, network, PPO self-play
replays/       # generated <id>.zip + <id>.trace.json artifacts
data/          # gl_map_pool.json + maps/<id>.csv + maps/<id>_units.json
test_*.py      # unit tests at the repo root (not under tests/)
```

## Module index

| File | Owns |
|------|------|
| `engine/unit.py` | `UnitType` (IntEnum, 27 units), `UNIT_STATS` table, `Unit` dataclass, `unit_id` invariant doc |
| `engine/action.py` | `ActionType`, `ActionStage` (SELECT→MOVE→ACTION), `Action` dataclass, `get_legal_actions`, `get_reachable_tiles`, `get_attack_targets`, `get_producible_units`, `_build_cost` |
| `engine/game.py` | `GameState`, `make_initial_state`, `step`, `_apply_build`, `_apply_power_effects`, `_grant_income`, `_check_win_conditions`, `full_trace` recording, `_allocate_unit_id` |
| `engine/co.py` | `COState`, `make_co_state_safe`, COP/SCOP charge & activation |
| `engine/combat.py` | `calculate_damage`, `calculate_counterattack`, base damage table |
| `engine/map_loader.py` | `MapData`, `PropertyState`, `TierInfo`, `load_map`, country→player assignment |
| `engine/predeployed.py` | `PredeployedUnitSpec`, `load_predeployed_units_file`, `specs_to_initial_units` |
| `engine/terrain.py` | terrain ID constants, `get_terrain`, `get_move_cost`, `MOVE_*` movement classes |
| `tools/export_awbw_replay.py` | Snapshot zip writer (`<game_id>` entry, PHP-serialized `awbwGame`); accepts optional `full_trace` and delegates the action stream |
| `tools/export_awbw_replay_actions.py` | Builds the `p:` action stream (`a<game_id>` entry); rebuilds state from `full_trace` and emits `Build`/`Move`/`End` JSON wrapped in PHP envelopes |
| `rl/ai_vs_ai.py` | Self-play harness; calls the exporter with `state.full_trace` |

## The non-negotiable invariants

These are easy to break and silently corrupt replays. Hold the line.

1. **`UnitType` is `IntEnum` with `INFANTRY = 0`.** Never test it for truthiness.
   - Wrong: `if action.unit_type:` — drops every infantry build.
   - Right: `if action.unit_type is not None:`.
   - Same trap for `ActionStage.SELECT = 0`. Use `is not None` or compare to the enum.

2. **`Unit.unit_id` is allocated once and never reused.** The viewer indexes `DrawableUnit` by id; reuse causes wrong-sprite / wrong-color bugs.
   - Allocator: `GameState._allocate_unit_id()` (monotonic, starts at 1).
   - Must be called for every Unit constructed in `make_initial_state` (predeploy), `_apply_build`, and `_apply_power_effects` (Sensei COP infantry/mech spawn).
   - Validation: `test_stable_unit_ids.py`. Run it after any change to unit creation paths.

3. **Naval units only on `port` tiles.** Black Boat / Lander / Battleship etc. on `base` is a bug.
   - Filter: `engine/action.py::get_producible_units` keys off `terrain_info.is_port`.
   - Engine guard: `_apply_build` must reject if the destination terrain is wrong, even when actions are crafted manually.
   - Test: `test_naval_build_guard.py`.

4. **Income excludes HQ, Lab, and Comm Tower.** Bases / cities / airports / ports pay 1000g each per turn.
   - `GameState.count_income_properties` filters out `is_comm_tower` and `is_lab`. HQs are not flagged as income props in the property list.
   - Lab-objective maps (e.g. map 126428 "Ft. Fantasy") have **no HQ at all**. 3 income props at turn 1 → 3000g is correct, not a bug.

5. **Trace recording uses `is not None`.** `engine/game.py::step` writes to `self.full_trace` for every action including SELECT and END_TURN. The replay action stream rebuilds from this trace, so any silent drop (see invariant 1) breaks per-move animation.

## The three-stage turn

Every player-turn is a sequence of `Action`s walked through stages:

```
SELECT  → choose unit (SELECT_UNIT) | END_TURN | ACTIVATE_COP/SCOP | direct factory BUILD
  ↓
MOVE    → choose destination (SELECT_UNIT with move_pos set)
  ↓
ACTION  → ATTACK | CAPTURE | WAIT | LOAD | UNLOAD | BUILD
```

`get_legal_actions(state)` dispatches on `state.action_stage`. BUILD can come from either SELECT (factory-driven, no unit selected) or ACTION (unit-on-factory). End of turn flips `active_player` and runs income.

## Replay zip format

A viewer-compatible replay is a regular `.zip` with **one or two gzipped entries**:

| Entry name | Contents | Required |
|------------|----------|----------|
| `<game_id>` | PHP-serialized `O:8:"awbwGame":` snapshots, one per turn boundary | Yes |
| `a<game_id>` | PHP-serialized envelopes wrapping per-turn JSON action arrays (the "p: stream") | No — viewer falls back to snapshot-only animation |

The exporter writes the snapshot entry first, then `tools/export_awbw_replay_actions.append_action_stream_to_zip` appends the action entry. Failure to build the action stream is best-effort: it warns but leaves the zip valid as snapshot-only.

Action stream rebuild walks `state.full_trace` (which is per-stage), groups events by `(player, turn)`, re-executes them on a fresh `GameState`, and emits `Build` / `Move` / `End` JSON. Old traces with the IntEnum=0 bug have null `unit_type` on infantry builds — those are skipped gracefully during rebuild, so counts will be lower than the trace's BUILD count. New traces (post-fix) line up exactly.

For exact JSON shapes, PHP envelope structure, and player_id mappings, read `reference.md`.

## Common workflows

### Generate a new AI vs AI replay
```bash
python rl/ai_vs_ai.py --map <map_id> --tier T2 --co0 <id> --co1 <id>
# writes replays/<game_id>.zip and replays/<game_id>.trace.json
```

### Verify an existing replay's action stream
```bash
python tools/_verify_action_stream.py replays/<id>.zip
# reports envelope count, action counts, final action
```

### Diagnose a map's income / property layout
```bash
python tools/diag_map_income.py <map_id>
# prints HQ presence, property breakdown, income calculation
```

### Reverse-engineer a real AWBW replay's action format
```bash
python tools/_inspect_oracle_actions.py <oracle.zip>
# pretty-prints sample Build/Move/End JSON
```

### Run targeted invariant tests
```bash
python -m unittest test_stable_unit_ids test_naval_build_guard -v
```

### Run everything
```bash
python -m unittest discover -s . -p "test_*.py" -v
# Note: test_awbw_parity.py has 2 known-stale failures asserting map 133665
# has 2 predeployed units when it now has 6. Not from the engine.
```

## When debugging a replay viewer issue

Walk this checklist before reading code:

1. **Wrong color / sprite swap on a unit?** → `unit_id` reuse. Re-run `test_stable_unit_ids`. Inspect the snapshot in the zip and grep for the offending id across turns.
2. **Funds look wrong on turn 1?** → `python tools/diag_map_income.py <map_id>`. Confirm HQ presence and income property count before assuming an engine bug.
3. **Per-move animation missing or stutters?** → Replay was generated before the action stream was wired in, or the trace has null `unit_type` entries. Re-export with `_verify_action_stream.py`.
4. **Unit appears on a tile it shouldn't be buildable on?** → Check `get_producible_units` and `_apply_build` guards. Crafted actions can bypass `get_legal_actions`.
5. **CO power didn't spawn the expected units?** → `_apply_power_effects` in `engine/game.py`. Sensei (CO id 8) is the main spawning power; ensure spawned units get `_allocate_unit_id`.

## Style

- Engine code is `from __future__ import annotations`-style with dataclasses and `IntEnum`s. Match it.
- Avoid `if x:` on enums and tuples. Use `is not None` or explicit comparisons.
- Tests live at the repo root as `test_*.py`, run via `python -m unittest`.
- The exporter's appended action stream must remain best-effort — never fail the snapshot zip if the stream errors.

## Replay ingest (download → normalize → audit)

- **Downloading GL replays** and keeping **Orange Star / Blue Moon** map colors + `desync_audit` register: see **`.cursor/skills/awbw-replay-ingest/SKILL.md`** (`tools/amarriner_download_replays.py`, `tools/normalize_map_to_os_bm.py`).

## Additional resources

- For PHP-serialization layout, p: action JSON shapes, player_id mapping rules, the full UnitType→cost/move table, and the CO id table, see `reference.md`.
