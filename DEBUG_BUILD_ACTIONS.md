# BUILD Action Debug Analysis

## Problem: Models Never Building Units

### Current BUILD Action Logic (engine/action.py lines 362-382)

BUILD actions are generated in `_get_action_actions()` which is **Stage 2 (ACTION stage)**.

**Requirements for BUILD action to be available:**

1. ✅ Must be in ACTION stage (after selecting unit and move destination)
2. ✅ `move_pos` must be on a base/airport/port tile
3. ✅ Property at `move_pos` must be owned by active player
4. ❌ **CRITICAL ISSUE**: `existing = state.get_unit_at(*move_pos)`
   - Line 369: `if existing is None or existing.pos == unit.pos:`
   - This means: "No unit at destination OR the moving unit is already there"

### The Problem:

**BUILD actions require a unit to move onto an empty factory tile.**

But in AWBW, you don't need a unit on the factory to build - **factories build automatically if owned and empty!**

### Current Flow:

```
Stage 0 (SELECT): Player selects a unit
    ↓
Stage 1 (MOVE): Player chooses where to move that unit
    ↓
Stage 2 (ACTION): Player chooses what to do at destination
    - ATTACK
    - CAPTURE
    - WAIT
    - LOAD
    - BUILD ← Only if moved onto owned factory
```

### Why Models Never Build:

1. **No starting units on factories** - Maps start with units in the field
2. **Moving onto factory wastes a turn** - Unit could be doing something useful
3. **BUILD requires unit presence** - But AWBW factories build without units!

### The Real AWBW Mechanic:

In actual AWBW:
- Factories are **independent entities** that can build if:
  - Owned by player
  - Empty (no unit on tile)
  - Player has funds
- **No unit needs to "activate" the factory**

### Solution Options:

#### Option 1: Add Factory-Direct BUILD Actions (Correct AWBW behavior)

In `_get_select_actions()` (Stage 0), add BUILD actions for each owned empty factory:

```python
def _get_select_actions(state: GameState, player: int) -> list[Action]:
    actions: list[Action] = [Action(ActionType.END_TURN)]
    
    # ... existing CO power actions ...
    
    # ... existing unit selection ...
    
    # NEW: Direct factory BUILD actions
    for prop in state.properties:
        if prop.owner == player:
            terrain = get_terrain(state.map_data.terrain[prop.row][prop.col])
            if terrain.is_base or terrain.is_airport or terrain.is_port:
                # Check if factory is empty
                if state.get_unit_at(prop.row, prop.col) is None:
                    # Generate BUILD actions for this factory
                    for ut in get_producible_units(terrain, state.map_data.unit_bans):
                        cost = _build_cost(ut, state, player, (prop.row, prop.col))
                        if state.funds[player] >= cost:
                            actions.append(Action(
                                ActionType.BUILD,
                                unit_pos=None,  # No unit required
                                move_pos=(prop.row, prop.col),
                                unit_type=ut,
                            ))
    
    return actions
```

#### Option 2: Keep Current System, Add Reward for Factory Occupation

Less correct but simpler - reward models for moving units onto factories.

### Recommendation:

**Implement Option 1** - This matches actual AWBW mechanics and will allow models to build units properly.

### Testing:

After fix, verify:
```python
# Check game log for BUILD actions
games = load_game_log()
for game in games:
    if game.get('gold_spent', [0, 0]) != [0, 0]:
        print(f"Game {game['map_id']}: P0 spent {game['gold_spent'][0]}, P1 spent {game['gold_spent'][1]}")