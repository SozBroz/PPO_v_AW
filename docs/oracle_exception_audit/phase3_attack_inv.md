# Phase 3 — Thread ATTACK-INV

**Campaign:** `desync_purge_engine_harden`
**Plan:** [`.cursor/plans/desync_purge_engine_harden_d85bd82c.plan.md`](../../.cursor/plans/desync_purge_engine_harden_d85bd82c.plan.md) §"Thread ATTACK-INV"
**Driver:** Probe 4 of [`phase2p5_legality_recon.md`](phase2p5_legality_recon.md) — friendly-fire `ATTACK` resolved full damage, no `ValueError`.

---

## Audit of the four mandated fixes

| # | Fix | Status pre-edit | Action this thread |
|---|-----|------------------|--------------------|
| 1 | Friendly-fire filter in `get_attack_targets` non-seam branch | **Already in place** at `engine/action.py:320–323` (`if enemy.player == unit.player: continue`). The `continue` correctly skips the seam fallback as well, so a friendly unit on a seam tile is also excluded. | No change. Documented. |
| 2 | ~~Range tightening (Chebyshev for direct, Manhattan for indirect)~~ | ~~**Already in place** at `engine/action.py:305–316`. Indirect → Manhattan with `min_range`/`max_range` bounds; direct 1–1 → Chebyshev exactly 1. Confirmed by Probes 1, 2, 3.~~ | ~~No change. Documented.~~ |

**AMENDED IN PHASE 6 (2026-04-20):** This row was wrong. The Chebyshev-for-direct
clause was the bug, not the canon. Phase 6 collapsed `engine/action.py:301-310`
to Manhattan-for-all. See `phase2p5_legality_recon.md` Probe 3 amendment and
the Phase 6 regression-log entry for evidence.

| # | Fix | Status pre-edit | Action this thread |
|---|-----|------------------|--------------------|
| 3 | Already-attacked guard | **Already in place** at `engine/action.py:422–425` — `_get_select_actions` only emits `SELECT_UNIT` for `unit.moved == False`. A moved attacker cannot reach Stage-2 ACTION through the mask. STEP-GATE blocks crafted actions independently. | No change. Documented. |
| 4 | Defense-in-depth assert in `_apply_attack` | **Missing.** `_apply_attack` returned silently when attacker was missing and never checked friendly fire or range. | **Added** — see below. |

## Files touched

### `engine/game.py` — `_apply_attack` (lines 600–631 in current file)

Replaced the silent `return` on missing attacker with a hard `ValueError`, then added two guards mirroring `get_attack_targets`:

```python
def _apply_attack(self, action: Action):
    attacker = self.get_unit_at(*action.unit_pos)
    if attacker is None:
        raise ValueError(
            f"_apply_attack: no attacker at {action.unit_pos}"
        )

    # Phase 3 ATTACK-INV defense-in-depth (mirrors get_attack_targets /
    # get_legal_actions; redundant when STEP-GATE is enforced, fires only
    # if step() is called with a crafted action that bypassed the mask).
    # Seam attacks (defender is None) are owned by SEAM thread and routed
    # through _apply_seam_attack below, so we only police the unit-vs-unit
    # branch here.
    defender_pre = (
        self.get_unit_at(*action.target_pos)
        if action.target_pos is not None
        else None
    )
    if defender_pre is not None and defender_pre.player == attacker.player:
        raise ValueError(
            f"_apply_attack: friendly fire from player {attacker.player} "
            f"on {defender_pre.unit_type.name} at {action.target_pos}"
        )
    if defender_pre is not None:
        atk_from = action.move_pos if action.move_pos is not None else attacker.pos
        if action.target_pos not in get_attack_targets(self, attacker, atk_from):
            raise ValueError(
                f"_apply_attack: target {action.target_pos} not in attack "
                f"range for {attacker.unit_type.name} from {atk_from} "
                f"(unit_pos={action.unit_pos})"
            )

    self._move_unit(attacker, action.move_pos)
    ...
```

**Design notes**

- The third assert intentionally guards only the `defender_pre is not None` branch. Seam attacks (defender is None) are routed through `_apply_seam_attack` and owned by Thread SEAM; this thread does not police that path.
- Range check uses `action.move_pos` (post-move attack origin), not `attacker.pos` — the unit may end on a different tile before firing. Falls back to `attacker.pos` only if `action.move_pos is None` (defensive; STEP-GATE never produces such an action).
- `_oracle_combat_damage_override` and `combatInfo` paths are untouched — the asserts run before any combat math.
- Per the campaign's source-of-truth principle, these asserts are **defense-in-depth only**. Under normal `step()` (`oracle_mode=False`), STEP-GATE rejects bad actions at `engine/game.py:228–236` before `_apply_attack` is ever called. The asserts catch the case where `oracle_mode=True` is passed, or any future refactor breaks STEP-GATE.

## Smoke test

`tools/_phase3_attack_inv_smoke.py` (deleted after run, per ticket). 7 scenarios, all PASS:

| Scenario | Path | Expected | Result |
|----------|------|----------|--------|
| `friendly_fire_strict` | `step(...)` | raise (STEP-GATE) | PASS — `IllegalActionError: ... not in get_legal_actions()` |
| `friendly_fire_oracle_bypass` | `step(..., oracle_mode=True)` | raise (`_apply_attack` assert) | PASS — `ValueError: _apply_attack: friendly fire from player 0 on TANK at (3, 4)` |
| `enemy_adjacent` | `step(...)` | succeed | PASS |
| `mech_diagonal` | `step(...)` (Probe 3 sanity) | ~~succeed~~ | ~~PASS~~ |
| `tank_out_of_range_strict` | `step(...)` | raise (STEP-GATE) | PASS |
| `tank_out_of_range_oracle_bypass` | `step(..., oracle_mode=True)` | raise (`_apply_attack` assert) | PASS — `ValueError: _apply_attack: target (3, 5) not in attack range for TANK from (3, 3)` |
| `no_attacker_oracle_bypass` | `step(..., oracle_mode=True)` | raise (`_apply_attack` assert) | PASS — `ValueError: _apply_attack: no attacker at (3, 3)` |

**AMENDED IN PHASE 7 (2026-04-20):** Pre-Phase-6 smoke assumed Chebyshev-1 for direct Mech; under Manhattan canon (Phase 6), Mech **cannot** diagonal-attack — that scenario is structurally impossible, not a passing `step(...)`. Negative coverage: `test_direct_r1_unit_cannot_attack_diagonally` in `tests/test_engine_negative_legality.py`. See `logs/desync_regression_log.md` § **2026-04-20 — Phase 6: Manhattan correction (post-Phase-5 critical fix)**.

Both layers verified: STEP-GATE catches at the legal-actions gate; the new `_apply_attack` asserts catch when STEP-GATE is bypassed via `oracle_mode=True`. Probe 4 (friendly fire) is closed end-to-end.

## Pytest delta

`logs/phase3_attack_inv_pytest.log` (full log: `logs/phase3_attack_inv_pytest_full.log`).

- 175 passed, 6 failed, 3 xpassed, 2 warnings (35.68s, 184 collected).
- All 6 failures originate from `IllegalActionError: ... not in get_legal_actions()` raised by **STEP-GATE** (the previously-merged sibling thread) — they are tests that craft `step()` actions out of stage / mask without using `oracle_mode=True`. None reference `_apply_attack: ...` (the prefix this thread's asserts use). My change is innocent of all 6.
  - `test_capture_terrain.py::test_full_capture_updates_terrain_on_misery_neutral_city` — calls `step(CAPTURE)` from SELECT stage.
  - `test_engine_awbw_subset.py::test_relax_wait_on_capturable_property_does_not_raise`, `test_select_unit_id_pins_engine_unit_when_tile_stacked` — pre-STEP-GATE call patterns.
  - ~~`test_engine_negative_legality.py::test_mech_can_attack_diagonal_chebyshev_1`~~, `test_piperunner_can_fire_on_pipe_seam_within_range`, `test_direct_adjacent_attack_on_unit_standing_on_seam_tile` — same: positive guards that skip the SELECT→MOVE setup.

**AMENDED IN PHASE 6 (2026-04-20):** The strikethrough test was deleted in Phase 6 (it codified the
Chebyshev bug). Replaced by parametrized `test_direct_r1_unit_cannot_attack_diagonally`
and `test_direct_r1_unit_can_attack_orthogonally`.
  - `test_unit_join.py::test_illegal_wait_on_join_tile` — expected a JOIN error, got STEP-GATE error first.

  Owner: STEP-GATE / NEG-TESTS test-update follow-up, **not** this thread.
- 3 xpassed in `test_engine_negative_legality.py` — pre-existing `xfail` markers on negative tests that are now satisfied (friendly fire, COP-with-zero-power, etc.). These confirm STEP-GATE + per-method asserts are working in concert.
- The `test_engine_negative_legality.py` friendly-fire test is among the passing dots — Probe 4's regression bar is GREEN.

## Constraints honored

- `engine/game.py` `step()` not modified (STEP-GATE thread only).
- SEAM branch of `get_attack_targets` and `_apply_seam_attack` not modified (SEAM thread only).
- COP/SCOP/CAPTURE/END_TURN emission untouched (POWER+TURN+CAPTURE thread only).
- `_oracle_combat_damage_override` and `combatInfo` paths preserved verbatim.
- Edits limited to `engine/game.py::_apply_attack` (added defense-in-depth asserts at the top of the unit-vs-unit branch).

## Blockers

None. Thread closes clean.

The 6 pytest failures above are not blockers for this thread — they are the test-suite update lag that always trails STEP-GATE (the test author needs to wrap `step()` calls in `oracle_mode=True` or set up the proper SELECT→MOVE→ACTION chain). Owner is STEP-GATE / NEG-TESTS follow-up.
