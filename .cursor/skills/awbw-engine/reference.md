# AWBW Engine — Reference Details

Read this only when you need exact field-by-field information beyond `SKILL.md`. Do not load by default.

## Unit table (engine/unit.py)

27 units. `INFANTRY = 0` (the IntEnum=0 trap). All cost / move / range values come from `UNIT_STATS`.

| ID | Name | Class | Move | Range | Cost | Notes |
|----|------|-------|------|-------|------|-------|
| 0 | Infantry | infantry | 3 | 1 | 1000 | captures |
| 1 | Mech | mech | 2 | 1 | 3000 | captures |
| 2 | Recon | vehicle | 8 | 1 | 4000 | unlimited ammo (-1) |
| 3 | Tank | vehicle | 6 | 1 | 7000 | |
| 4 | Med Tank | vehicle | 5 | 1 | 16000 | |
| 5 | Neo Tank | vehicle | 6 | 1 | 22000 | |
| 6 | Mega Tank | vehicle | 4 | 1 | 28000 | |
| 7 | APC | vehicle | 6 | – | 5000 | carries 1 (inf/mech) |
| 8 | Artillery | vehicle | 5 | 2–3 | 6000 | indirect |
| 9 | Rocket | vehicle | 5 | 3–5 | 15000 | indirect |
| 10 | Anti-Air | vehicle | 6 | 1 | 8000 | |
| 11 | Missiles | vehicle | 4 | 3–5 | 12000 | indirect |
| 12 | Fighter | air | 9 | 1 | 20000 | fuel/turn 5 |
| 13 | Bomber | air | 7 | 1 | 22000 | fuel/turn 5 |
| 14 | Stealth | air | 6 | 1 | 24000 | can dive |
| 15 | B-Copter | copter | 6 | 1 | 9000 | fuel/turn 2 |
| 16 | T-Copter | copter | 6 | – | 5000 | carries 1 |
| 17 | Battleship | naval | 5 | 2–6 | 28000 | indirect |
| 18 | Carrier | naval | 5 | 3–8 | 30000 | carries 2 (copters) |
| 19 | Submarine | naval | 5 | 1 | 20000 | can dive |
| 20 | Cruiser | naval | 6 | 1 | 18000 | carries 2 (copters) |
| 21 | Lander | naval | 6 | – | 12000 | carries 2 (ground) |
| 22 | Gunboat | naval | 7 | 1 | 6000 | carries 1 (inf/mech) |
| 23 | Black Boat | naval | 7 | – | 7500 | carries 2 (inf/mech) |
| 24 | Black Bomb | air | 9 | 1 | 25000 | suicide |
| 25 | Piperunner | pipe | 9 | 2–5 | 20000 | indirect, pipe-only |
| 26 | Oozium | vehicle | 1 | 1 | 0 | spawned by Flak/Jugger SCOP |

Move types: `MOVE_INF`, `MOVE_MECH`, `MOVE_TREAD`, `MOVE_TIRE_A`, `MOVE_TIRE_B`, `MOVE_AIR`, `MOVE_SEA`, `MOVE_LANDER`, `MOVE_PIPELINE` (engine/terrain.py).

Producible-by-terrain (engine/action.py::`get_producible_units`):
- `is_base` → ground units (including Piperunner; excluding Oozium)
- `is_airport` → air units
- `is_port` → naval units
- Bans (`unit_bans` on MapData) filter Black Bomb, Stealth, Piperunner, Oozium

## CO ids and special behaviors referenced in code

Search `engine/action.py::get_reachable_tiles` and `engine/game.py::_apply_power_effects` for the canonical list. Notable ids:

| ID | CO | Where it shows up |
|----|----|-------------------|
| 3 | Kanbei | `_build_cost`: 120% cost |
| 8 | Sami | move bonus on infantry (COP +1, SCOP +2); SCOP spawns infantry |
| 10 | Eagle | SCOP: air/copter +2 move |
| 11 | Adder | DTD +1 move, COP +1, SCOP +2 |
| 15 | Colin | `_build_cost`: 80% cost |
| 17 | Hachi | `_build_cost`: 50% on bases |
| 20 | Grimm | SCOP: ground +3 move |

The full CO table lives in `engine/co.py::make_co_state_safe`. Extend the lookup tables there when adding new COs.

## Action stage transitions

```
ActionStage.SELECT (0)
  ├── Action(SELECT_UNIT, unit_pos=...) → ActionStage.MOVE
  ├── Action(END_TURN)                  → next player, ActionStage.SELECT
  ├── Action(ACTIVATE_COP/SCOP)         → stays SELECT
  └── Action(BUILD, move_pos=..., unit_type=...)  → resolves immediately, stays SELECT

ActionStage.MOVE (1)
  └── Action(SELECT_UNIT, unit_pos=..., move_pos=...) → ActionStage.ACTION

ActionStage.ACTION (2)
  ├── Action(WAIT, unit_pos, move_pos)                       → SELECT
  ├── Action(ATTACK, unit_pos, move_pos, target_pos)         → SELECT
  ├── Action(CAPTURE, unit_pos, move_pos)                    → SELECT
  ├── Action(LOAD, unit_pos, move_pos)                       → SELECT
  ├── Action(UNLOAD, unit_pos, move_pos, unload_pos, ...)    → SELECT
  └── Action(BUILD, unit_pos, move_pos, unit_type)           → SELECT
```

## full_trace entry shape

Each entry recorded by `GameState.step` (engine/game.py):

```python
{
  "type":       <ActionType.name>,           # e.g. "BUILD", "SELECT_UNIT", "END_TURN"
  "player":     int,
  "turn":       int,                          # 1-indexed day; bumps after P1 ends turn
  "stage":      <ActionStage.name>,           # "SELECT" | "MOVE" | "ACTION"
  "unit_pos":   [r, c] | None,
  "move_pos":   [r, c] | None,
  "target_pos": [r, c] | None,
  "unit_type":  "INFANTRY" | "MECH" | ... | None,  # only set on BUILD
}
```

All `Optional` fields use `is not None` checks. Never re-introduce `if action.unit_type:` style — `INFANTRY = 0` is falsy.

## Replay zip — entry layout

```
<game_id>.zip
├── <game_id>          # gzipped, PHP-serialized: O:8:"awbwGame":N:{...} per turn boundary
└── a<game_id>         # gzipped, PHP-serialized envelopes; one per (player, day)
```

Both entries are gzipped *inside* the zip. The viewer (`AWBWJsonReplayParser.cs`) reads either entry as gzip first.

### Snapshot entry — PHP `awbwGame` shape

A list of state snapshots. Each snapshot is `O:8:"awbwGame":N:{...}` containing:

- `players`: array keyed by `players_id` (PHP int), with funds, CO id, power bar, eliminated flag.
- `units`: array keyed by `units_id` (== `Unit.unit_id`). Stable across snapshots is mandatory.
- `buildings`: array keyed by tile, with terrain id and `buildings_team` owner.
- `weather`, `day`, `turn`, `funds`, `start_date`.

The serializer is in `tools/export_awbw_replay.py::_serialize_unit` and friends. It already uses `unit.unit_id` — do not "regenerate" ids per snapshot.

### Action stream — `a<game_id>` envelope shape

The stream is a list of envelopes, one per (player, turn). Each envelope is PHP-serialized and contains a JSON string with an array of action objects.

Per-envelope JSON schema (the array):

```json
[
  { "action": "Build", "newUnit": { "global": { "units_id": ..., "units_players_id": ..., "units_name": "Infantry", ... } }, "discovered": { "<players_id>": null } },
  { "action": "Move", "unit": { "<players_id>": {...}, "global": {...} }, "paths": { "global": [ {"unit_visible": true, "x": .., "y": ..}, ... ] }, "dist": <int>, "trapped": false, "discovered": {...} },
  { "action": "End", "updatedInfo": { "event": "NextTurn", "nextPId": ..., "nextFunds": {"global": <int>}, "nextTimer": 0, "nextWeather": "C", "supplied": {"global": []}, "repaired": {"global": []}, "day": <int>, "nextTurnStart": "YYYY-MM-DD HH:MM:SS" } }
]
```

MVP scope (`tools/export_awbw_replay_actions.py`):
- `Build` — fully implemented
- `Move` — emitted for `WAIT`, `ATTACK`, `CAPTURE`, `LOAD` (movement portion only; combat resolution / capture point ticking lives in the snapshot diff)
- `End` — emitted on `END_TURN`

Not yet implemented in the action stream:
- `Attack` payload with HP deltas + counter
- `Capt` action with capture point tick
- `Load` / `Unload` as standalone actions

The viewer falls back to snapshot diffs when an action is missing, so missing payloads degrade animation quality but don't break replay validity.

### players_id mapping

AWBW uses opaque integer player ids per replay. The exporter generates them deterministically per game; see `tools/export_awbw_replay.py` for the assignment scheme. The action stream and snapshot entries must agree on these ids. The `"global"` key in action JSON holds the actor's view; per-player keys hold opponent-visible state when fog applies.

## Map data files

```
data/gl_map_pool.json              # pool of playable map ids + metadata (tiers, bans)
data/maps/<map_id>.csv             # comma-separated terrain grid, no header
data/maps/<map_id>_units.json      # optional predeployed unit list (PredeployedUnitSpec[])
```

`engine/map_loader.py::load_map(map_id, pool_path, maps_dir)` reads all three and assigns the first country encountered (row-major scan) to player 0, second to player 1.

`PredeployedUnitSpec` fields: `player`, `unit_type` (UnitType), `pos` (row, col), `hp`, `ammo`, `fuel`, `loaded_units`, `is_submerged`, `capture_progress`. `specs_to_initial_units` converts to `Unit` instances — this is the path that must call `_allocate_unit_id`.

## Test inventory

| File | Covers |
|------|--------|
| `test_stable_unit_ids.py` | unit_id allocation, monotonicity, identity stability across turns |
| `test_naval_build_guard.py` | port-only naval builds (legal action filter + engine guard) |
| `test_build.py`, `test_build_guard.py`, `test_build_action_encoding.py` | BUILD action paths |
| `test_movement_parity.py` | reachable tiles vs AWBW reference |
| `test_awbw_parity.py` | predeploy + army-wipe rules. **Has 2 stale assertions** for map 133665 (claims 2 predeployed units, actually 6) — pre-existing, not engine bugs |

## Tools inventory (high-signal subset)

Most useful when debugging or extending the export pipeline:

| Tool | Purpose |
|------|---------|
| `tools/export_awbw_replay.py` | Snapshot writer; called by `rl/ai_vs_ai.py` |
| `tools/export_awbw_replay_actions.py` | p: stream builder |
| `tools/_verify_action_stream.py` | Round-trip parse a generated p: stream |
| `tools/_inspect_oracle_actions.py` | Pretty-print actions from a real AWBW replay zip |
| `tools/diag_map_income.py` | Per-map property + income breakdown |
| `tools/dump_turn.py` | Decode a single snapshot from a replay zip |
| `tools/diff_replay_zips.py`, `tools/deep_diff_replays.py` | Compare two replay zips |
| `tools/validate_predeployed.py`, `tools/fetch_predeployed_units.py` | Maintain `data/maps/<id>_units.json` |
| `tools/inspect_replay.py`, `tools/inspect_replay2.py` | Ad-hoc replay inspection |

The `tools/fetch_*`, `tools/find_*`, `tools/decode_*` scripts are one-shot reverse-engineering utilities for the AWBW viewer binary; do not modify unless explicitly extending viewer parity work.
