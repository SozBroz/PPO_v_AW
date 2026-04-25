# Phase 11J-F5-OCCUPANCY-IMPL — Option A (`select_unit_id` on ATTACK)

**Verdict: GREEN**

## Diff summary

### `engine/game.py` — `_apply_attack`

- After the existing **P-COLO-ATTACKER** block (`selected_unit` preferred when pinned on `unit_pos`), the fallback that used `get_unit_at(*action.unit_pos)` now calls `get_unit_at_oracle_id(*action.unit_pos, action.select_unit_id)`.
- Comment added: `# Phase 11J-F5-OCCUPANCY: prefer oracle id when set, defends against duplicate-position oracle states.`
- **Composes with P-COLO:** `selected_unit` still wins first; oracle id only applies when `attacker is None`.

### `tools/oracle_zip_replay.py`

Three `Action(ActionType.ATTACK, ...)` construction sites now pass `select_unit_id`:

1. **Fire (no path)** — `select_unit_id=int(u.unit_id)` (striker `u` already resolved).
2. **Fire (with `Move.paths`)** — `select_unit_id=su_id` (same value already used for `SELECT_UNIT`).
3. **AttackSeam (no path)** — `select_unit_id=int(u.unit_id)`.

No refactor of `_resolve_fire_or_seam_attacker` or move/forced-snap logic.

## Tests — `tests/test_attack_select_unit_id_pin.py`

| Test | Intent |
|------|--------|
| `test_attack_select_unit_id_resolves_striker_on_duplicate_tile` | BLACK_BOAT before MED_TANK on same tile: without pin, `ValueError` (boat “not in range”); with `select_unit_id` of the tank, strike applies. |
| `test_attack_select_unit_id_none_single_occupant_matches_legacy` | Single occupant, `select_unit_id=None` — normal attack unchanged. |
| `test_attack_select_unit_id_dead_oracle_id_falls_back_to_get_unit_at` | Pin to dead `unit_id` on stack: `get_unit_at_oracle_id` falls back; other alive unit on tile attacks. |
| `test_attack_select_unit_id_wrong_tile_falls_back_to_get_unit_at` | Pin id is a unit on another hex: no match at `unit_pos`; tile scan picks unit on `unit_pos`. |

## Gate results

1. **New tests:** `python -m pytest tests/test_attack_select_unit_id_pin.py -v` — **PASS** (4/4).
2. **Engine / oracle tests:** `python -m pytest tests/test_engine_negative_legality.py tests/test_oracle_strict_apply_invariants.py test_oracle_zip_replay.py -v --tb=short` — **PASS** (123 passed, 3 xpassed).
3. **Full pytest:** `python -m pytest --tb=no -q` — **1 failed, 526 passed** (same single failure as baseline on clean tree: `test_trace_182065_seam_validation.py::test_full_trace_replays_without_error`, `Illegal move ... is not reachable` in `_apply_wait` / `_move_unit`). **Not introduced by this change** (reproduced with `engine/game.py` + `tools/oracle_zip_replay.py` stashed to HEAD). Within campaign allowance of ≤2 deferred trace failures.
4. **Targeted gid 1626642:** `python tools/desync_audit.py --games-id 1626642 --register logs/desync_register_f5_targeted.jsonl --seed 1` — **PASS** (`ok`, 1 game).
5. **100-game sample:** `python tools/desync_audit.py --max-games 100 --register logs/desync_register_post_f5_100.jsonl --seed 1` — **PASS**: `ok=89`, `oracle_gap=11`, **`engine_bug=0`** (matches FU-ORACLE baseline `logs/desync_register_post_phase11j_fu_100.jsonl`: `ok=89`, `oracle_gap=11`).

## Commander brief

Option A is shipped: ATTACK resolution now respects `select_unit_id` when STEP-GATE has not already pinned `selected_unit`, and the oracle threads the striker’s engine `unit_id` on every ATTACK it emits from the Fire and AttackSeam no-path branches. Duplicate-position oracle states no longer mis-attribute the attacker via first-hit `get_unit_at`. Registers and the 100-game audit are unchanged versus the FU baseline; `1626642` remains `ok`.

---

*End of Phase 11J-F5-OCCUPANCY-IMPL.*
