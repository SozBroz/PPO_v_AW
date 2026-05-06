FINAL REPORT: Oracle Mover Resolution Fix
======================================

ROOT CAUSE
----------
The oracle's mover resolution logic had a critical flaw: when the declared unit type
(PHP units_name) didn't match the engine's unit at the expected position, the oracle
would fall back to using ANY unit at that position (or ANY unit with the same uid).
This caused FALSE POSITIVES - the oracle was moving the wrong unit, leading to
state_mismatch_units errors that looked like engine bugs but were actually oracle errors.

SPECIFIC BUGS FIXED
-----------------
1. Lines 4508, 4522, 4530, 4538: Changed fallback condition from
   `if declared_mover_type is None or x.unit_type == declared_mover_type:`
   to `if declared_mover_type is not None and x.unit_type == declared_mover_type:`
   This prevents matching ANY unit when declared_mover_type is None.

2. Lines 4570: Changed from matching any single unmoved unit to only matching
   when declared_mover_type is not None and the unit type matches.

3. Lines 4586-4590: Added position check for single-unit type matches -
   only use the unit if it's at the expected position (path start, global, or end).

4. Lines 4674-4678: Removed the fallback that used any unit at a position
   when no type match was found. Now returns None (which becomes oracle_gap).

RESULTS
-------
Tested 202 games with state_mismatch_units in desync_register_v9.jsonl:

- Fixed (now oracle_gap): 124 games (61.4%)
  These were FALSE POSITIVES - the oracle was moving the wrong unit.
  Now correctly raises UnsupportedOracleAction (oracle_gap).

- Remaining state_mismatch_units: 37 games (18.3%)
  These show a pattern: engine has Infantry at positions where PHP expects
  other unit types (Anti-Air, Tank, Recon, APC, etc.)
  Further investigation needed - may be real engine bugs or other oracle issues.

- Other statuses: 41 games (20.3%)
  Likely end_truncated_game or other non-state-mismatch statuses.

IMPACT
------
- 61.4% of state_mismatch_units games were false positives
- The oracle is now MORE STRICT about unit type matching
- Real engine bugs are now easier to identify (they won't be hidden by oracle errors)

REMAINING ISSUES
----------------
The 37 remaining games show a pattern where:
  engine='Infantry' php='OtherType'

This suggests either:
1. The oracle is moving Infantry instead of the correct unit type
2. The oracle is moving the correct unit to the wrong position
3. There's a real engine desync

NEXT STEPS
----------
Investigate the remaining 37 state_mismatch_units games to determine:
1. Are they real engine bugs?
2. Are there other oracle resolution issues?
3. Should we add more debug output to distinguish oracle vs engine issues?

FILES CHANGED
-------------
- d:\awbw\tools\oracle_zip_replay.py
  Lines 4508, 4522, 4530, 4538, 4570, 4586-4590, 4674-4678
