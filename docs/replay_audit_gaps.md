# Replay Desync Audit — Gap Analysis
## Date: 2026-05-05
## Scope: 1410 replays in `replays/amarriner_gl/`

---

## Executive Summary

The desync audit suite has **6 critical gaps** that cause real engine divergences to be silently classified as `ok`. The default audit path (`--enable-state-mismatch` OFF) only detects **hard crashes** (exceptions in oracle or engine). It does **not** compare engine state against PHP snapshots. As a result, ~74.5% of `ok` rows (per Phase 11K audit on n=200) hide silent state drift.

Additionally, the snapshot comparison itself (`compare_snapshot_to_engine`) only checks **funds and units** — it completely ignores property state, CO power state, weather, turn number, and fog.

---

## Gap 1: Default Audit Path Silences All State Divergences

**Severity: CRITICAL**

### What happens
- `desync_audit.py` runs the oracle replay and classifies based on **exceptions only**
- When `--enable-state-mismatch` is OFF (the default), the audit **never calls `_diff_engine_vs_snapshot`**
- Games where the engine silently diverges (wrong funds, wrong unit HP, missing units) are marked `ok`

### Evidence
From `desync_audit.py` lines 339-344:
```python
# These knobs are OFF by default. When --enable-state-mismatch is passed,
# ``_run_replay_instrumented`` calls ``_diff_engine_vs_snapshot`` after every
# successful ``p:`` envelope...
# (Phase 10F: 78% of "ok" rows hide PHP drift; Phase 11K: 74.5% on n=200).
```

### Impact
- 936/936 games can report as `ok` while funds are wrong, units are missing, or HP is wrong
- The only way to catch these is to run with `--enable-state-mismatch` (3x slower)

### Fix
**Either** make `--enable-state-mismatch` the default, **or** add a separate lightweight state comparison that runs by default (funds + unit count + HP bars at minimum).

---

## Gap 2: Snapshot Comparison Ignores Property State

**Severity: HIGH**

### What's missing
`compare_snapshot_to_engine()` in `replay_snapshot_compare.py` only calls:
1. `compare_funds()` — checks `state.funds[0]` and `state.funds[1]`
2. `compare_units()` — checks unit positions, types, and HP bars

**Not compared** (PHP snapshots have this data; engine `GameState` has this data):

| State Axis | PHP Snapshot Field | Engine Field | Current Check |
|------------|-------------------|---------------|---------------|
| Property ownership | `buildings[].capture` (capture points), `buildings[].terrain_id` | `state.properties[].owner` | ❌ **MISSING** |
| Property capture points | `buildings[].capture` (0-99) | `state.properties[].capture_points` (0-20) | ❌ **MISSING** |
| CO meter/star count | `players[].co_power`, `players[].co_max_power` | `state.co_states[].cop_stars`, `scop_stars`, meter values | ❌ **MISSING** |
| CO power active | `players[].co_power_on` | `state.co_states[].cop_active`, `scop_active` | ❌ **MISSING** |
| Weather | `weather_type`, `weather_code` | `state.weather` | ❌ **MISSING** |
| Turn/day number | `day` | `state.turn` | ❌ **MISSING** |
| Active player | `turn` (active player turn) | `state.active_player` | ❌ **MISSING** |
| Fog of war | `fog` | Not in GameState (info only) | N/A |

### PHP Building Structure (from `1358720.zip` frame 0):
```python
{
    'id': 69227696,
    'games_id': 1358720,
    'terrain_id': 113,      # Pipe Seam
    'x': 0, 'y': 24,
    'capture': 99,            # Capture points (99 = neutral/full)
    'last_capture': 20,      # Last capture points (20 = fully owned)
    'last_updated': '2025-02-10 13:33:21'
}
```

### Engine PropertyState:
```python
@dataclass
class PropertyState:
    terrain_id: int
    row: int; col: int
    owner: Optional[int]    # 0, 1, or None
    capture_points: int     # 20 = fully owned; <20 = being captured
```

### Impact
- Property capture bugs are **never detected** by the audit
- A property that should be owned by P0 but is owned by P1 (or neutral) won't be caught
- Capture point drift (e.g., engine says 12 points, PHP says 8 points) is invisible

### Fix
Add `compare_properties()` to `replay_snapshot_compare.py`:
```python
def compare_properties(
    php_frame: dict[str, Any],
    state: GameState,
    awbw_to_engine: dict[int, int],
) -> list[str]:
    """Compare property ownership and capture points."""
    out = []
    php_buildings = php_frame.get("buildings") or {}
    # Build a map: (row, col) -> PHP building
    php_by_pos: dict[tuple[int, int, int], dict] = {}
    for _k, b in php_buildings.items():
        if not isinstance(b, dict):
            continue
        try:
            r, c = int(b["y"]), int(b["x"])
        except (KeyError, ValueError):
            continue
        # Need to determine which engine seat owns this property
        # Match by terrain_id + position
        php_by_pos[(r, c)] = b

    for prop in state.properties:
        key = (prop.row, prop.col)
        pb = php_by_pos.get(key)
        if pb is None:
            out.append(f"property at ({prop.row},{prop.col}) tid={prop.terrain_id} not in PHP snapshot")
            continue
        # Compare ownership
        php_owner_pid = int(pb.get("capture", 0))
        # PHP capture=99 means neutral; capture=20 means fully owned by whoever's id matches
        # This needs proper mapping logic
        # ...
    return out
```

---

## Gap 3: CO State Not Compared

**Severity: HIGH**

### What's missing
The PHP snapshots track CO power state in `players[]`:
- `co_power` — current power meter (stars filled)
- `co_max_power` — max stars for COP
- `co_power_on` — whether a CO power is active this turn

Engine `COState` tracks:
- `cop_stars`, `scop_stars` — star counts
- `cop_active`, `scop_active` — power active flags
- Meter fill values (stored internally)

### Impact
- CO power activation bugs are **never detected**
- If the engine activates COP but PHP didn't (or vice versa), audit reports `ok`
- Star meter drift is invisible

### Fix
Add `compare_co_states()` to `replay_snapshot_compare.py`:
```python
def compare_co_states(
    php_frame: dict[str, Any],
    state: GameState,
    awbw_to_engine: dict[int, int],
) -> list[str]:
    """Compare CO power meter and activation state."""
    out = []
    players = php_frame.get("players") or {}
    for _k, pl in players.items():
        if not isinstance(pl, dict):
            continue
        pid = int(pl["id"])
        eng = awbw_to_engine.get(pid)
        if eng is None:
            continue
        co_state = state.co_states[eng]
        # Compare power activation
        php_power_on = int(pl.get("co_power_on", 0))
        engine_power_active = co_state.cop_active or co_state.scop_active
        if bool(php_power_on) != engine_power_active:
            out.append(f"P{eng} power_active engine={engine_power_active} php={bool(php_power_on)}")
        # Compare meter (approximate, since PHP uses stars and engine uses internal meter)
        # ...
    return out
```

---

## Gap 4: Weather State Not Compared

**Severity: MEDIUM**

### What's missing
- PHP: `weather_type`, `weather_code` in frame
- Engine: `state.weather` ("clear", "rain", "snow")

### Impact
- Weather bugs (rare but possible with CO powers like Olaf's) are not detected

### Fix
Add weather comparison to `compare_snapshot_to_engine()`.

---

## Gap 5: Turn/Day Number Not Compared

**Severity: MEDIUM**

### What's missing
- PHP: `day` field in frame
- Engine: `state.turn`

### Impact
- Turn counter drift is not detected (should rarely happen, but worth checking)

### Fix
Add turn comparison to `compare_snapshot_to_engine()`.

---

## Gap 6: HP Comparison Uses Display Bars, Not Internal HP

**Severity: MEDIUM** (partially mitigated in Phase 11J)

### What's missing
In `compare_units()`, HP comparison uses **display bars** (`_php_unit_bars()` which returns `ceil(hit_points)`), not internal HP.

The `_diff_engine_vs_snapshot()` function in `desync_audit.py` does compare internal HP (via `php_internal_from_snapshot_hit_points()`), but this **only runs when `--enable-state-mismatch` is ON**.

### Impact
- When state-mismatch is OFF (default), HP drift within the same display bar is invisible
- E.g., engine HP = 25 (2.5 bars), PHP HP = 29 (also 2.5 bars when ceiled) → no mismatch detected

### Fix
Either:
1. Always use internal HP comparison (not bars) in `compare_units()`
2. Or make `--enable-state-mismatch` the default

---

## Quantifying the Problem

### Current default audit (state-mismatch OFF):
- Only detects `oracle_gap` (unsupported actions), `engine_bug` (exceptions), `loader_error`, `replay_no_action_stream`
- **Silently passes**: wrong funds, wrong unit HP, missing units, wrong property ownership, wrong CO state, wrong weather

### With `--enable-state-mismatch`:
- Compares funds + unit types + unit HP (internal) after **every envelope**
- Still missing: property state, CO state, weather, turn number

---

## Recommended Fixes (Priority Order)

### P0 (Do Immediately)
1. **Add `compare_properties()`** to `replay_snapshot_compare.py`
2. **Add `compare_co_states()`** to `replay_snapshot_compare.py`
3. **Make `--enable-state-mismatch` the default** in `desync_audit.py`, or at minimum add a `--quick-check` that compares funds + unit count + HP bars after every envelope

### P1 (Soon)
4. **Add weather comparison**
5. **Add turn/day comparison**
6. **Add `--fail-on-any-drift`** flag that exits non-zero on any state mismatch (for CI gates)

### P2 (Nice to Have)
7. **Compare terrain changes** (seam hp, broken pipes)
8. **Compare unit ammo/fuel** (PHP exports have this data)
9. **Compare unit action counts** (for join validation)

---

## Testing the Fixes

After implementing the above, run:
```bash
# Full audit with all comparisons enabled
python tools/desync_audit.py --enable-state-mismatch --state-mismatch-hp-tolerance 0 --register logs/desync_full_audit.jsonl

# Compare before/after counts
python -c "
import json
rows = [json.loads(l) for l in open('logs/desync_full_audit.jsonl')]
from collections import Counter
print(Counter(r['class'] for r in rows))
"
```

Expected: The number of `ok` rows will **drop significantly** (Phase 11K: 74.5% of `ok` rows had hidden drift).

---

## Appendix: PHP Snapshot Structure (from `1358720.zip`)

### Top-level keys:
```python
['id', 'name', 'password', 'creator', 'start_date', 'end_date', 'activity_date',
 'maps_id', 'weather_type', 'weather_start', 'weather_code',
 'win_condition', 'turn', 'day', 'active', 'funds', 'capture_win',
 'fog', 'comment', 'type', 'boot_interval', 'starting_funds',
 'official', 'min_rating', 'max_rating', 'league', 'team',
 'aet_interval', 'aet_date', 'use_powers',
 'players', 'buildings', 'units',
 'timers_initial', 'timers_increment', 'timers_max_turn']
```

### Player keys (per player):
```python
['id', 'users_id', 'games_id', 'countries_id', 'co_id',
 'funds', 'turn', 'co_power', 'co_power_on',
 'co_max_power', 'co_max_spower',
 'eliminated', 'order', ...]
```

### Building keys (per building):
```python
['id', 'games_id', 'terrain_id', 'x', 'y',
 'capture', 'last_capture', 'last_updated']
```

### Unit keys (per unit):
```python
['id', 'games_id', 'players_id', 'units_id', 'name', 'hit_points',
 'x', 'y', 'carried', 'movement_points', 'fuel', 'ammo', ...]
```
