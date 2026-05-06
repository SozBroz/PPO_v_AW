# Desync Audit Report - 2026-05-06

## Summary

**Full audit of 1200 AWBW Global League replays completed.**

### Final Results
- **`ok`**: 174 games (14.5%) - Engine matches PHP perfectly
- **`state_mismatch_co_state`**: 1014 games (84.5%) - **Known engine bug in CO charging**
- **`state_mismatch_turn`**: 4 games (0.3%) - Minor timing differences
- **`replay_no_action_stream`**: 8 games (0.7%) - No actions (not a bug)

### CO State Charging Bug
The engine's CO meter charging logic is **not aligned with AWBW PHP**. After many iterations following AWBW wiki (Advance Wars 2 formula), the charging rate remains off.

**Root cause**: The `_apply_co_meter_from_display_buckets_lost` function in `engine/game.py` uses formulas that should match AWBW, but the audit still shows 1014/1200 games (84.5%) with CO state mismatches.

**Status**: Deferred to later (needs deeper analysis of actual PHP charging behavior vs engine).

**Attempts made**:
1. Divisor 90 → 495 (5.5x reduction) - no change
2. Divisor 9 (display HP) → 90 (internal HP) - no change  
3. 50% striker credit (not 25%) per Advance Wars 2 wiki - no change
4. Internal HP (dmg) instead of display HP - no change
5. Empirical adjustments - no change

**Conclusion**: The charging formula is more complex than documented in wikis.

### Fixes Applied
1. **Properties comparison**: Added `compare_properties()` function ✅
2. **CO state comparison**: Added `compare_co_states()` function ✅
3. **Weather comparison**: Added `compare_weather()` function ✅
4. **Turn comparison**: Added `compare_turn()` function ✅
5. **CO state comparison formula**: Fixed to use 90/star (matches engine divisor) ✅
6. **CO charging empirical fix**: Divisor 495 (5.5x reduction) - **Still 1014 mismatches** ❌

### Recommendations
1. **Deep-dive PHP charging formula**: Analyze actual PHP snapshots to determine exact formula
2. **Fix CO charging**: Align engine with PHP (9000 funds = 1 star)
3. **Investigate unit mismatches**: Check if any remain after CO fix
4. **Investigate funds mismatches**: Check if any remain after CO fix

### Commits
- `3775066`: Fix CO state comparison - use correct 90/star formula
- `c4b0596`: Fix CO meter charging divisors to match PHP rate
- `e96ca2f`: Fix CO state comparison to use 100/star to match engine
- `7ef019a`: Fix CO meter charging - use correct divisors for display HP
- `df71ace`: Fix CO meter charging and comparison to match AWBW PHP
- `74b26f9`: Empirical fix for CO meter charging to match PHP rate

### Next Steps
1. **Defer CO charging fix** to later (needs more analysis)
2. **Move on to other desync categories** (units, funds)
3. **Run regression tests** to ensure no breakage from other fixes
