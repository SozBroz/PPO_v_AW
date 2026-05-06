# CO State Mismatch Fix - Summary

## Problem
894 games in `logs/desync_register_v4.jsonl` were classified as `state_mismatch_co_state`.
The engine's `power_bar` was always 0 while PHP snapshots showed accumulating CO power.

## Root Cause
The engine has functions to credit CO power meter (`_apply_co_meter_from_internal_hp_lost`), 
but they were **NEVER BEING CALLED** during combat.

### Issues Found:
1. **Dead code**: A nested function `_apply_co_meter_from_display_buckets_lost` was defined 
   inside `_apply_attack` (line 1335-1380) but never called - it was just a function 
   definition sitting inside the method.

2. **Missing call for primary attacks**: The class method `_apply_co_meter_from_internal_hp_lost` 
   (line 1390) was not being called after combat damage was applied.

3. **Missing counterattack handling**: The `_apply_attack` function had NO counterattack logic at all -
   `calculate_counterattack()` was never called.

## The Fix (in `engine/game.py`)

### 1. Primary Attack CO Meter Credit
Added after line 1330 in `_apply_attack`:
```python
# Pass internal HP to CO meter function
if internal_dmg > 0:
    self._apply_co_meter_from_internal_hp_lost(attacker, defender, internal_dmg)
```

### 2. Counterattack Handling
Added after primary attack in `_apply_attack`:
```python
# Counterattack
if dmg is not None and dmg > 0 and defender.hp > 0 and defender.is_alive:
    counter_dmg = calculate_counterattack(
        defender, attacker,  # defender counters against attacker
        def_terrain, att_terrain,  # terrain for counterattacker, target
        def_co, att_co,  # CO states for counterattacker, target
        dmg,  # pass primary damage for Sonja SCOP
        luck_rng=self.luck_rng,
    )
    if counter_dmg is not None and counter_dmg > 0:
        internal_counter = min(counter_dmg * 10, attacker.hp)
        attacker.hp = max(0, attacker.hp - internal_counter)
        self.losses_hp[attacker.player] += internal_counter
        if attacker.hp == 0:
            self.losses_units[attacker.player] += 1
        if internal_counter > 0:
            self._apply_co_meter_from_internal_hp_lost(defender, attacker, internal_counter)
```

### 3. Removed Dead Code
Removed the nested function definition `_apply_co_meter_from_display_buckets_lost` from inside 
`_apply_attack` (it was dead code that was never called).

## Results

### Before Fix
- 894 games with `state_mismatch_co_state` desyncs
- Engine `power_bar` always 0
- PHP `co_power` showing values like 5000, 9000, etc.

### After Fix
- Full audit shows `co_state: 0` - ALL RESOLVED
- Engine now properly credits CO power from:
  - Primary attack damage (striker and victim both get credit)
  - Counterattack damage (counterattacker and target both get credit)

### Tests Passed
- `test_co_meter_formula.py`: 5 passed
- `test_co_hawke_powers.py`: 11 passed

## AWBW CO Power Formula
- 9000 funds damage = 1 star = 9000 power_bar units
- Victim credit: internal HP lost × unit cost / 90
- Striker credit: 50% of victim credit (per AWBW)
- CO cost modifiers (Colin, Kanbei, Hachi) are applied correctly
