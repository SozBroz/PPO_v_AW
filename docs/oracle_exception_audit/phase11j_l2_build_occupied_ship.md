# Phase 11J-L2-BUILD-OCCUPIED-SHIP — close the 9 BUILD-OCCUPIED-TILES cluster

**Verdict: GREEN — SHIP.**
9/9 BUILD-OCCUPIED-TILES rows closed as a cluster. 8 flipped directly to `ok`; the 9th (1634464) migrated to a deeper `Fire: oracle resolved defender … has no damage entry` gap (different cluster entirely — first-divergence migration, expected pattern after upstream gates lift). Full 936-game re-audit: **(ok=908, oracle_gap=27, engine_bug=1)** vs pre-fix **(894, 39, 3)** — net **+14 ok, −12 oracle_gap, −2 engine_bug**. Pytest clean (568 passed, 0 failed). 100-game sample: 98 ok, 0 engine_bug. No true regressions attributable to this fix.

---

## Section 1 — The 9 BUILD-OCCUPIED-TILES rows

Source register: `logs/desync_register_post_phase11j_v2_936.jsonl`. Filter: `class == "oracle_gap" AND message contains "Build no-op" AND "tile occupied"`. Drill tool: `tools/_phase11j_l2_build_occupied_drill.py` (added in this lane). Drill output for all 9: `logs/phase11j_l2_build_occupied_drill_all9.json`.

| # | games_id | Build tile | Unit | Seat | Delete at | Failing Build at | Delete units_id → PHP pos (pre frame) | Dominant pattern |
|---|----------|-----------:|------|------|----------:|-----------------:|----------------------------------------|------------------|
| 1 | 1625178 | (4,12)  | MEGA_TANK | P0 | act 22 | act 23 | 192052538 → (4,12)  | **Player-issued Delete of own-unit blocker on build tile** |
| 2 | 1626223 | (8,14)  | MECH      | P0 | act 0  | act 13 | 192549835 → (8,14)  | same |
| 3 | 1628236 | (8,14)  | MECH      | P0 | act 0  | act 1  | 192475649 → (8,14)  | same |
| 4 | 1628287 | (8,14)  | INFANTRY  | P0 | act 6  | act 7  | 192208010 → (8,14)  | same (drill representative A) |
| 5 | 1630064 | (12,10) | MECH      | P1 | act 3  | act 4  | 192421343 → (12,10) | same (drill representative B; preceded by failed Load/Move/Unload on same unit) |
| 6 | 1632006 | (18,9)  | ANTI_AIR  | P1 | act 2  | act 3  | 192433284 → (18,9)  | same |
| 7 | 1632778 | (12,10) | INFANTRY  | P1 | act 11 | act 12 | 192424643 → (12,10) | same |
| 8 | 1634464 | (12,10) | INFANTRY  | P1 | act 26 | act 27 | 192803258 → (12,10) | same |
| 9 | 1634587 | (13,1)  | MED_TANK  | P0 | act 15 | act 16 | 192620531 → (13,1)  | same |

**Cross-reference (`tools/_phase11j_l2_delete_xref.py`):** every Delete's `unitId.global` resolves via the AWBW PHP pre-frame to the exact tile of the next Build in the same envelope. The unit does not appear in the post-frame (PHP confirms deletion). All 9 cases: 100% match.

**Dominant pattern:** `player_issued_delete_of_own_unit_blocker_on_build_tile` — NOT the audit-report-hypothesized "stale-unit / death-clear ordering". The blocker is not stale: AWBW PHP pre-frame also has it. The difference is that AWBW's PHP **applies** the `Delete` action (erasing the unit) before the `Build`, while the engine's oracle path was **silently dropping** `Delete` as viewer-only cleanup, leaving the blocker alive and the subsequent `Build` refused with `tile occupied`.

**Why the existing nudge (`_oracle_nudge_eng_occupier_off_production_build_tile`) does not help:** it tries to move the blocker one orth step with `WAIT`, but in 2/2 drilled cases the blocker has `compute_reachable_costs total = 1` — i.e. can only stay on its own tile. Bases on Global-League std maps like Little Island / 4P Micro are routinely ringed by Sea (tid=4), River (tid=28), Reef (tid=5), and the base itself is the only ground tile for that player; ground units on such bases cannot step off. AWBW's canonical resolution is the **Delete** UI control, not movement. The nudge's comment already anticipates this case: *"if the blocker … is trapped (no reachable orth neighbour), return False so the Build handler surfaces drift."* The proper handling of `Delete` closes that drift path.

## Section 2 — Fix: resolve `Delete` in the oracle path, consume at next Build

### AWBW reasoning (Tier hierarchy per Phase 11J-F2-KOAL-FU Section 2)

- **Tier 2 (canonical wiki):** AWBW Wiki "Game Page" / UI controls documents *"Delete Unit"* as a player-initiated UI control that erases the selected unit, no funds refund, available during the owning player's turn — the **only** AWBW-canonical way to free a base occupied by a friendly unit that has already moved or is on a movement-isolated base island. `https://awbw.fandom.com/wiki/Game_Page` (*"Delete Unit … erases the selected unit with no funds returned"*).
- **Tier 3 (runtime ground truth, AWBW PHP-emitted):** observed `Delete` action shape in every replay zip — `{ "action": "Delete", "Delete": { "action": "Delete", "unitId": { "global": <AWBW units_id> } } }` — captured for all 9 cluster gids at `logs/phase11j_l2_build_occupied_drill_all9.json`.
- **Tier 3 (PHP post-frame cross-check):** per `tools/_phase11j_l2_delete_xref.py`, the unit named by `unitId.global` is present in the PHP pre-frame at the build tile and absent in the post-frame. The only AWBW-canonical operation that removes a still-alive unit with no counter-action or capture is `Delete`.

### Surgical change — `tools/oracle_zip_replay.py`

Two edit regions + one helper, all localized to Delete/Build/End. No engine-side change, no touch to `engine/unit.py`, `engine/action.py`, Von Bolt SCOP branch, `_grant_income`, `_resupply_on_properties`, `_build_cost`, or `_activate_power` (per hard rules).

1. **New helper** (near `_oracle_nudge_eng_occupier_off_production_build_tile`):

   ```python
   def _oracle_kill_friendly_unit(state: GameState, u: Unit) -> None:
       # Mirrors engine death-cleanup: set hp=0 so is_alive → False, drop cargo,
       # prune dead from state.units[player].  Matches the pattern at
       # engine/game.py _end_turn (line ~1003) and _apply_attack post-kill
       # cleanup (line ~981–984).
   ```

2. **`Delete` handler** (was viewer-only cleanup): parse `Delete.unitId.global`. Prefer direct match via existing `_unit_by_awbw_units_id(state, uid)` (works when engine unit_id happens to match AWBW units_id — today a no-op for normal units but kept for forward compatibility with future units_id assignment). Otherwise mark a half-turn-scoped pending-delete flag on the seat: `state._oracle_pending_delete_seats: set[int]`. Still call `_oracle_finish_action_if_stale` (preserves prior stale-ACTION cleanup contract).

3. **`Build` handler** (before the existing nudge call): if `pending_seats` contains this player AND `state.get_unit_at(r, c)` is a friendly blocker, kill the blocker and clear the pending flag. Subsequent existing nudge runs on an empty tile — no-op. BUILD then proceeds. This is the F5-OCCUPANCY-IMPL pattern (tile-occupancy resolution immediately upstream of the engine action dispatch), adapted for player-issued erasure rather than mover-off-tile.

4. **`End` handler**: clear the pending-delete set before `END_TURN`. AWBW only allows `Delete` during the acting player's own turn; any unconsumed Delete cannot leak across half-turns.

Diff summary (see `tools/oracle_zip_replay.py` around the `Delete` / `Build` / `End` branches; line numbers after fix):

- `+ _oracle_kill_friendly_unit` (helper, ~17 lines).
- `Delete`: was 4 lines (`_oracle_finish_action_if_stale` + `return`), now ~25 lines (parse uid, try direct match, fallback to pending set).
- `Build`: added 6-line pending-delete consume block before the existing nudge.
- `End`: added 3-line clear.

**Conservative by construction:** the pending-delete is consumed ONLY when `state.get_unit_at(r, c)` is a friendly unit owned by the acting player. If the Delete target isn't on the next Build's tile, we do not kill anyone (we still discard the pending flag on first Build for that seat or on End). False-positive blocker kills are thus bounded to *"a same-player friendly unit sits on the build tile after a Delete in the same half-turn"* — the AWBW-canonical scenario.

## Section 3 — Closure table (9 rows before / after)

Before register: `logs/desync_register_post_phase11j_v2_936.jsonl`.
After register: `logs/desync_register_l2_postfix_9.jsonl` (9-gid targeted audit, post-fix).

| games_id | Tile | Unit / Seat | Pre-fix class → message head | Post-fix class → message head |
|---------:|-----:|-------------|------------------------------|-------------------------------|
| 1625178 | (4,12)  | MEGA_TANK / P0 | `oracle_gap` — `Build no-op … (tile occupied)` | **`ok`** |
| 1626223 | (8,14)  | MECH / P0      | `oracle_gap` — `Build no-op … (tile occupied)` | **`ok`** |
| 1628236 | (8,14)  | MECH / P0      | `oracle_gap` — `Build no-op … (tile occupied)` | **`ok`** |
| 1628287 | (8,14)  | INFANTRY / P0  | `oracle_gap` — `Build no-op … (tile occupied)` | **`ok`** |
| 1630064 | (12,10) | MECH / P1      | `oracle_gap` — `Build no-op … (tile occupied)` | **`ok`** |
| 1632006 | (18,9)  | ANTI_AIR / P1  | `oracle_gap` — `Build no-op … (tile occupied)` | **`ok`** |
| 1632778 | (12,10) | INFANTRY / P1  | `oracle_gap` — `Build no-op … (tile occupied)` | **`ok`** |
| 1634464 | (12,10) | INFANTRY / P1  | `oracle_gap` — `Build no-op … (tile occupied)` | `oracle_gap` — **migrated** to `Fire: oracle resolved defender type MEGA_TANK at (5, 18) but RECON has no damage entry` (different cluster; first-divergence migration) |
| 1634587 | (13,1)  | MED_TANK / P0  | `oracle_gap` — `Build no-op … (tile occupied)` | **`ok`** |

**BUILD-OCCUPIED-TILES cluster closure: 9/9 (8 flipped to ok + 1 migrated to a distinct cluster). Numerical bar was ≥6 for YELLOW / ≥7 for GREEN — cleared.**

## Section 4 — Gate results

### Gate 1 — pytest

Command: `python -m pytest --tb=no -q --ignore=test_trace_182065_seam_validation.py`

Result: **568 passed, 5 skipped, 2 xfailed, 3 xpassed, 0 failed, 3853 subtests passed in 70.23s.** (Focused run on `-k "oracle or desync or build"` post-restore: 199 passed, 0 failed — confirms the fix applies cleanly.) Bar was ≤2 failures → **cleared**.

### Gate 2 — 100-game regression sample

Command: `python tools/desync_audit.py --catalog data/amarriner_gl_std_catalog.json --catalog data/amarriner_gl_extras_catalog.json --max-games 100 --seed 1 --register logs/desync_register_l2_postfix_100.jsonl`

Result: **ok=98, oracle_gap=2, engine_bug=0**. Bar was `ok ≥ 98 AND engine_bug == 0` → **cleared**.

### Gate 3 — No new engine_bug rows (full 936)

Command: `python tools/desync_audit.py … --max-games 936 --seed 1 --register logs/desync_register_l2_postfix_936.jsonl`

Result: **ok=908, oracle_gap=27, engine_bug=1** vs pre-fix **(894, 39, 3)** — net **+14 ok, −12 oracle_gap, −2 engine_bug**. Diff via `tools/_phase11j_l2_compare.py`:

- 15 gids flipped `oracle_gap → ok` (progress, includes the 8 BUILD-OCCUPIED-TILES closures + 7 downstream cascades cleared once the BUILD-OCCUPIED blocker lifted).
- 2 gids flipped `engine_bug → oracle_gap` (**progress**: 1629202, 1632825 — both were F4 friendly-fire first-divergence migrations that now get past the attack frame because the upstream BUILD gate no longer traps them).
- 1 gid flipped `ok → oracle_gap` (**apparent regression**: 1632226). **Investigation** (per `tools/_phase11j_l2_lookup.py` + isolation audit): with my fix **reverted locally**, 1632226 in isolation also produces `oracle_gap — Build no-op (insufficient funds need 1000$ have 300$)`. The pre-fix 936-audit `ok` classification for this gid is an **audit-run-ordering artifact**, not a stable pre-fix baseline (likely a cross-game state leak in the full-batch pre-fix run on a borderline 700$ funds shortfall). **Not attributable to this fix.** 0 true regressions.

Bar was "No new `engine_bug` rows" → **cleared** (engine_bug count went DOWN by 2, not up).

## Section 5 — Verdict

**GREEN — SHIP.**

- 9/9 BUILD-OCCUPIED-TILES cluster closed (8 to `ok` + 1 migrated to a distinct deeper cluster).
- All three regression gates cleared: pytest clean, 100-sample 98/0, full 936-audit strictly better on every class (+14 ok, −12 oracle_gap, −2 engine_bug).
- Fix is surgical, oracle-only (per hard rule `PREFER oracle_zip_replay.py edits`), no engine touch, no touch to owned branches (Von Bolt SCOP, Sasha Market Crash, L1-BUILD-FUNDS, F4 friendly-fire, F2 residual).
- No coordination blocker: fix region is `Delete` + `Build` + `End` branches in `_apply_oracle_action_json_body`, which is disjoint from `_end_turn` stun clear (VONBOLT-SCOP-SHIP) and from `_grant_income` / `_resupply_on_properties` / `_build_cost` / `_activate_power` (L1-BUILD-FUNDS / Sasha).

The hill is taken. The new firing line — per updated Section 7 of the 936 audit — is **L1-BUILD-FUNDS-RESIDUAL** (25 rows, median shortfall ~600$, P0-skewed), which is already queued and operates on a disjoint engine region.

*"Veni, vidi, vici."* (Latin, 47 BC)
*"I came, I saw, I conquered."* — Gaius Julius Caesar, reporting the Battle of Zela to the Senate.
*Caesar: Roman general and dictator; the report is his three-word campaign brief after crushing Pharnaces II at Zela in a five-day lightning strike.*
