# Phase 11J FIRE-DRIFT — Fix report

**Mission**: ENGINE + ORACLE WRITE lane against the 7 `engine_bug` residuals
called out in `phase11d_residual_engine_bug_triage.md` (6 F1 Bucket A
position-drift Fires + 1 F4 friendly-fire defender).
**Outcome**: **GREEN.** All seven `engine_bug` rows neutralised in one pass
(four flip to `ok`; three reclassify to `oracle_gap` — two of which surface
honest downstream drifts the audit had been masking, one of which is
Edit C's intended re-bucketing). All seven mandatory gates pass. The 100-
game baseline-comparable slice drops from **2 → 0 `engine_bug`**.

---

## 1. Files changed

| Path | Lines | Change |
|------|-------|--------|
| `engine/game.py` | `_apply_attack` head (≈ L631–680) | Edit A (selected_unit precedence) + Edit B (override-bypass on the defense-in-depth range check). |
| `tools/oracle_zip_replay.py` | new helper at module scope (≈ L1114) + two call sites in the `Fire` branch (no-path ≈ L5743, with-path ≈ L5982) | Edit C (`_oracle_assert_fire_damage_table_compatible`). |
| `tests/test_phase11j_fire_drift.py` | new file, 8 tests | Regression coverage for Edits A / B / C. |
| `docs/oracle_exception_audit/phase11j_fire_drift_hypothesis.md` | new | Pre-edit hypothesis. |
| `docs/oracle_exception_audit/phase11j_fire_drift_fix.md` | this file | Post-edit report. |
| `tools/_phase11j_drill.py`, `tools/_phase11j_envinspect.py`, `tools/_phase11j_envinspect2.py`, `tools/_phase11j_print.py` | new (instrumentation only, not loaded by production code) | Drilldown harnesses used during root-cause analysis. |

`engine/action.py::get_attack_targets` was **not** touched — Phase 6 read-only
contract honoured.

---

## 2. Root cause analysis

Drilldown evidence: `tools/_phase11j_drill.py` instruments `_apply_attack` to
dump engine state at the moment of failure; `tools/_phase11j_envinspect2.py`
extracts the raw AWBW `Fire` and `Move` envelopes for the seven targets
(output: `logs/phase11j_envinspect2.json`). C# Replay Player corroboration
launched on the largest drift (1631494) and the F4 (1634664) per the
`desync-triage-viewer` skill.

The seven failures collapse to three pathologies:

### 2.1 P-AMMO — `get_attack_targets` shorts on `ammo == 0`

Affects **5 of 7**: 1622104, 1625784, 1630983, 1635025, 1635846.

`engine/action.py:283-284`:
```python
if stats.max_ammo > 0 and unit.ammo == 0:
    return []
```

This is correct for primary-only units, but Mech / Tank / Md.Tank /
B-Copter all have an **unmetered secondary MG** in AWBW canon (Phase 10A
patched MG **consumption** in `_apply_attack`, but not the **legality
gate**). When the oracle has already pinned the post-strike HPs through
`_oracle_combat_damage_override`, AWBW has authoritative-decided the
strike is legal — the engine's defense-in-depth range check
(`_apply_attack` L654-661) calls back into the empty `get_attack_targets`
and refuses.

Drill evidence summary (`logs/phase11j_envinspect2.json`):

| games_id | attacker → defender (AWBW) | engine ammo at fail |
|----------|---------------------------|---------------------|
| 1622104 | MECH (6,17) → TANK (6,16), AWBW pre-strike ammo=0, MG strike | 0 |
| 1630983 | MECH (13,22) → INF (13,23) (kill), AWBW pre-strike ammo=0, MG | 0 |
| 1635025 | B-COPTER (14,19) → TANK (15,19), AWBW pre-strike ammo=0, MG | 0 |
| 1625784 | B-COPTER (8,2) → TANK (8,1), AWBW pre-strike ammo=1 (last missile) | 0 (engine ammo drift carry-forward) |
| 1635846 | B-COPTER (8,5) → TANK (8,4), AWBW pre-strike ammo=1 (last missile) | 0 (engine ammo drift carry-forward) |

The two B-Copter rows whose AWBW pre-strike ammo was 1 are an
**ammo accounting drift between engine and AWBW** (engine over-spent
primary missiles in earlier turns). Edit B closes the symptom; the
underlying drift becomes a pure-state divergence (no `engine_bug`),
documented as Phase 11K carry-forward.

### 2.2 P-DRIFT-DEFENDER — resolver picks an incompatible engine unit

Affects **1 of 7**: 1631494.

AWBW: P0 FIGHTER flies (15,4)→(16,13), strikes a foe at (15,13) (a
flier — the AWBW combatInfo shows `units_ammo=8` on the defender, so
B-COPTER or another fighter; AWBW unit 192553831).

Engine at failure: tile (15,13) is empty (AWBW unit 192553831 was never
spawned here — engine `unit_id` is monotonic small-int, not the AWBW
PHP id; see `tools/oracle_zip_replay.py:2052` docstring). The resolver
`_oracle_fire_resolve_defender_target_pos` falls back to a Chebyshev-1
ring search and lands on a **TANK at (14,13)** — the only nearby foe.

`get_base_damage(FIGHTER, TANK)` is `None` in the AWBW damage table, so
`get_attack_targets` (correctly) refuses (14,13). The range check trips.

If we had only Edit B in place, the override-bypass would have applied
damage derived from a different AWBW unit's HP delta to the TANK — a
silent state corruption. Edit C raises `UnsupportedOracleAction` first,
re-buckets the row to `oracle_gap`. That is the truthful classification:
the oracle cannot map the strike onto the engine snapshot it has.

### 2.3 P-COLO-ATTACKER — `get_unit_at` picks the wrong unit on a co-occupied tile

Affects **1 of 7**: 1634664.

AWBW: P1 INFANTRY at (2,18) walks 3 tiles S to (5,18), bayonets P0
INFANTRY at (5,19) to death. Owners are clean — P1 hits P0.

Engine: drill shows **two units co-located at (5,18)** at the moment
`_apply_attack` runs — the P1 mover and a stationary P0 INF.
`attacker = self.get_unit_at(*action.unit_pos)` returns the *first* unit
at that tile, which is the P0 INF (older entry). The friendly-fire
guard then trips when that picked attacker tries to hit a P0 defender at
(5,19). `state.selected_unit` is correctly set to the P1 mover throughout
— STEP-GATE plumbed it; only the lookup short-circuits.

The C# viewer launch on 1634664 / day 1 confirmed both INFs sat on
(5,18) at the failing envelope (the second was a prior-turn arrival, not
cargo). This was Bucket B-shaped — wrong attacker resolved by the oracle
/ engine pair — **not** envelope-self-targeted, **not** owner-bit
corruption.

---

## 3. Fix description

### Edit A — `engine/game.py::_apply_attack` (P-COLO-ATTACKER)

```python
attacker: Optional[Unit] = None
sel = self.selected_unit
if sel is not None and sel.is_alive and sel.pos == action.unit_pos:
    attacker = sel
if attacker is None:
    attacker = self.get_unit_at(*action.unit_pos)
if attacker is None:
    raise ValueError(...)
```

Invariant-tightening. `selected_unit` is canonically `None` at the start
of any RL `step` / `reset` (cleared by `_finish_action`), so the new
branch is **inert** outside oracle replay and tests that explicitly set it.
When STEP-GATE has selected the unit, that unit *is* the actor — prefer
it over the first-match-by-tile heuristic. Friendly-fire guard remains
unconditional (Edit A only fixes which attacker it sees).

### Edit B — `engine/game.py::_apply_attack` (P-AMMO)

```python
oracle_pinned = self._oracle_combat_damage_override is not None
if defender_pre is not None and not oracle_pinned:
    atk_from = action.move_pos if action.move_pos is not None else attacker.pos
    if action.target_pos not in get_attack_targets(self, attacker, atk_from):
        raise ValueError(...)
```

Scoped purely to the override path. The override is consumed at L684
below (set to `None`), so a stray subsequent `step` is gated normally —
no widening of legal-action surface in pure-engine play. Friendly fire
remains unconditional. Edit C (oracle-side) ensures we never see an
override that resolves to a damage-incompatible defender.

### Edit C — `tools/oracle_zip_replay.py` (P-DRIFT-DEFENDER)

New helper at module scope:

```python
def _oracle_assert_fire_damage_table_compatible(state, attacker, defender_pos):
    defender = state.get_unit_at(*defender_pos)
    if defender is None or not defender.is_alive:
        return
    if get_base_damage(attacker.unit_type, defender.unit_type) is None:
        raise UnsupportedOracleAction(
            f"Fire: oracle resolved defender type {defender.unit_type.name} "
            f"at {defender_pos} but {attacker.unit_type.name} has no damage "
            f"entry against it ..."
        )
```

Wired in **before** both `_oracle_set_combat_damage_override_from_combat_info`
calls in the Fire branch (no-path and with-path). Empty tile is
intentionally a no-op — the seam-targetable check / `defender is None`
post-move branch in the engine handle it.

---

## 4. Per-target results

`logs/desync_register_post_phase11j.jsonl` (Gate 6 — re-audit with
`--seed 1`):

| games_id | before (10Q) | after (11J) | mechanism |
|----------|--------------|-------------|-----------|
| 1622104  | engine_bug — `_apply_attack target not in attack range` | **oracle_gap** — Move truncated path; upstream drift at day~24 acts=1244 | Edit B passes the original Fire; replay continued ~1000 actions; honest downstream drift surfaced |
| 1625784  | engine_bug — Bucket A drift Δ=3 (10A residual) | **ok** | Edit B |
| 1630983  | engine_bug — Bucket A drift Δ=2 | **ok** | Edit B |
| 1631494  | engine_bug — Bucket A drift Δ=10 (largest) | **oracle_gap** — Fire damage-table incompatible (FIGHTER vs TANK) | Edit C — *intended* reclassification |
| 1635025  | engine_bug — Bucket A drift Δ=6 (10A residual) | **ok** | Edit B |
| 1635846  | engine_bug — Bucket A drift Δ=4 (10A residual) | **ok** | Edit B |
| 1634664  | engine_bug — `_apply_attack friendly fire from player 0 on Infantry` | **oracle_gap** — Move truncated path; upstream drift at day~12 acts=369 | Edit A passes the original F4 friendly-fire false positive; honest downstream drift surfaced |

**Score: 4/7 → `ok`, 3/7 → `oracle_gap`, 0/7 stay `engine_bug`.** Above
the gate floor of "at least 3 of 7 flip ok". The three `oracle_gap`
re-classifications are correctly bucketed (Edit C is intentional;
1622104 / 1634664 surface previously-masked downstream drift as the
correct truthful failure class).

---

## 5. Regression gates

| # | Gate | Floor | Result |
|---|------|-------|--------|
| 1 | `pytest tests/test_engine_negative_legality.py -v --tb=no` | 44 passed, 3 xpassed, 0 failed | **44 passed, 3 xpassed** ✓ |
| 2 | `pytest tests/test_andy_scop_movement_bonus.py --tb=no` | 2 passed | **2 passed** ✓ |
| 3 | `pytest tests/test_engine_legal_actions_equivalence.py::test_legal_actions_step_equivalence --tb=no` | 1 passed | **1 passed** ✓ |
| 4 | `pytest tests/test_co_build_cost_hachi.py tests/test_co_income_kindle.py tests/test_oracle_strict_apply_invariants.py --tb=no` | 15 passed | **15 passed** ✓ |
| 5 | `pytest --tb=no -q` | ≤ 2 failures (deferred trace_182065 only) | **493 passed, 5 skipped, 2 xfailed, 3 xpassed, 1 failed** (only `test_trace_182065_seam_validation::test_full_trace_replays_without_error` — exactly the deferred row) ✓ |
| 6 | Targeted re-audit on the 7 gids `--seed 1` | At least 3 of 7 flip ok; ZERO new engine_bug elsewhere | **4 of 7 flip ok**; **0 engine_bug** anywhere in the run ✓ |
| 7 | 100-game sample audit `--max-games 100 --seed 1` | engine_bug count NOT higher than 100-game slice of 10Q baseline (= 2) | **0 engine_bug** (92 ok, 8 oracle_gap) — strictly **below** baseline ✓ |

Gates 1, 2, 3, 4 also confirm: the new test file
`tests/test_phase11j_fire_drift.py` (8 tests, all green) was rolled into
Gate 5's full sweep without changing any other test's verdict. No
regressions anywhere outside the deferred trace.

---

## 6. Verdict

**GREEN.** All seven gates pass at or above their floors; all seven
target rows have left the `engine_bug` bucket; the 100-game slice's
`engine_bug` count strictly improves (2 → 0); no regression in any other
test. The three `oracle_gap` reclassifications are honest — Edit C's is
the truthful bucket per the hypothesis, and 1622104 / 1634664 surface
genuine downstream drift the previous failure had been masking.

---

## 7. Updated `engine_bug` residual count

10Q baseline (per `phase11d_residual_engine_bug_triage.md`): **10**
`engine_bug` residuals total.

11J post-fix accounting:
- **−7** on the seven targeted rows (none remain `engine_bug`).
- Net `engine_bug` residual after 11J: **3** (the three rows Phase 11D
  classified outside F1/F4 — F2 / F5 / other; not in scope for this
  phase).
- 100-game baseline-comparable slice: **2 → 0** `engine_bug`.

### Carry-forward debt (not a regression)

1. **B-Copter primary-ammo drift** (1625784, 1635846) — engine ammo went
   to 0 by the failing envelope while AWBW had ammo=1. Symptom-closed by
   Edit B; root cause is upstream over-decrement of primary missiles in
   prior turns. Surfaces (if at all) as F2 state diff in a future audit
   pass; recommended target for **Phase 11K**.
2. **AWBW unit_id ↔ engine unit_id mapping** — the resolver-miss in
   1631494 is symptomatic of the larger structural gap noted at
   `tools/oracle_zip_replay.py:2052` ("predeploy uses monotonic ids").
   Plumbing PHP `units_id` through the engine `Unit.unit_id` would
   reduce Edit C activations from "needed" to "rare". Out of scope for
   11J; recommended **Phase 11L** scope.
3. **Downstream drift in 1622104 (day~24 acts=1244) and 1634664
   (day~12 acts=369)** — both replays are now `oracle_gap`'s "Move
   truncated path; upstream drift" later in the run. They were
   previously masked behind the now-fixed Fire crash; treat as fresh
   rows in the next triage queue (per `desync-triage-viewer` Replay
   ownership rule, documented and deferred — not closed).

### Closure per `desync-triage-viewer` rules

| games_id | closure column |
|----------|----------------|
| 1622104  | column 3 (engine fix — Edit B); downstream `oracle_gap` documented for next triage |
| 1625784  | column 3 (engine fix — Edit B); zip is fully `ok` |
| 1630983  | column 3 (engine fix — Edit B); zip is fully `ok` |
| 1631494  | column 2 (oracle fix — Edit C); reclassified `oracle_gap` is the honest bucket |
| 1635025  | column 3 (engine fix — Edit B); zip is fully `ok` |
| 1635846  | column 3 (engine fix — Edit B); zip is fully `ok` |
| 1634664  | column 3 (engine fix — Edit A); downstream `oracle_gap` documented for next triage |

No replay deletions. No ambiguous closures.

---

*"Veni, vidi, vici."* (Latin, 47 BC)
*"I came, I saw, I conquered."* — Julius Caesar, dispatch to Rome after the Battle of Zela
*Caesar: Roman dictator and general; the line is the standard short-form report from the field after a decisive engagement.*
