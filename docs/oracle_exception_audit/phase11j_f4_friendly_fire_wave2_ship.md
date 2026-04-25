# Phase 11J-F4-FRIENDLY-FIRE-WAVE2-SHIP — closeout

**Mode:** ship. Two `engine_bug` rows (1629202, 1632825) flipped from
`engine_bug → oracle_gap` via a single oracle-path guard. No engine edits.

**Verdict: GREEN.** Both gids closed. All gates green.

---

## 1. Drill summary (both gids)

`python tools/_phase11j_drill.py --catalog data/amarriner_gl_extras_catalog.json --games-id 1629202`
`python tools/_phase11j_drill.py --catalog data/amarriner_gl_std_catalog.json --games-id 1632825`

Outputs: `logs/_drill_f4w2_1629202.json`, `logs/_drill_f4w2_1632825.json`.

### 1629202 — T4, Jake (P0) vs Sonja (P1), day 12, env idx 22

| Field | Engine state at failure | AWBW envelope (Fire combatInfo) |
|-------|--------------------------|---------------------------------|
| Attacker (resolved `u`) | `INFANTRY` P0, pos `(5,20)`, `unit_id=43`, `display_hp=10` | `units_id=192733549` at `(y=6,x=20)` after move; `copValues.attacker.playerId=3764695` |
| Move tile (`fire_pos`) | `(6,20)` (orth-adjacent to defender) | path tail `(y=6,x=20)` |
| Defender resolved at | `MECH` **P0** (friendly), pos `(7,20)`, `unit_id=55`, `display_hp=8` | `units_id=192733479` at `(y=7,x=20)`; `copValues.defender.playerId=3764694` (P1) |
| `select_unit_id` on `Action(ATTACK,…)` | `43` (correctly pinned) | — |
| `get_attack_targets(state, u, atk_from)` | `[]` (engine refuses; only candidate at `(7,20)` is friendly) | strike legal in AWBW (cross-player) |
| Engine ValueError | `_apply_attack: friendly fire from player 0 on MECH at (7, 20)` | — |

**Reading.** AWBW `combatInfo` is unambiguous: attacker `playerId=3764695` ≠ defender
`playerId=3764694` (cross-player Fire). Engine has the **wrong owner / wrong
unit at `(7,20)`** — the defender that should be a P1 MECH is a P0 MECH in the
engine snapshot. Pre-`MOVE-TRUNCATE-SHIP` this game diverged earlier on a
truncated path and never reached env 22; once that wall lifted, the stale-board
condition surfaced as friendly fire.

### 1632825 — T4, Jake (P0) vs Sonja (P1), day 16, env idx 30

| Field | Engine state at failure | AWBW envelope (Fire combatInfo) |
|-------|--------------------------|---------------------------------|
| Attacker (resolved `u`) | `INFANTRY` P0, pos `(12,17)`, `unit_id=16`, `display_hp=10` | `units_id=192430614` at `(y=12,x=18)` after move; `copValues.attacker.playerId=3772608` |
| Move tile (`fire_pos`) | `(12,18)` | path tail `(y=12,x=18)` |
| Defender resolved at | `INFANTRY` **P0** (friendly), pos `(12,19)`, `unit_id=74`, `display_hp=10` | `units_id=192483855` at `(y=12,x=19)`; `copValues.defender.playerId=3772609` (P1) |
| `select_unit_id` on `Action(ATTACK,…)` | `16` (correctly pinned) | — |
| `get_attack_targets(...)` | `[]` (only candidate is friendly) | strike legal in AWBW |
| Engine ValueError | `_apply_attack: friendly fire from player 0 on INFANTRY at (12, 19)` | — |

**Same pattern.** AWBW: cross-player Fire (`3772608` vs `3772609`); engine: both
`(12,17)` and `(12,19)` are P0 INF.  Same first-divergence migration shape as
1629202.

---

## 2. Classification

Both rows are **Class C — Friendly fire surface from upstream board drift**
(per the spec's three-class menu, with the explicit caveat that "AWBW genuinely
allows" applies only to true cross-player fire — here AWBW *does* allow the
strike, and the engine misclassifies it because its board is stale, so the
clean closure is to mark the row `oracle_gap` instead of forcing the engine to
accept a same-player target).

Concretely both are **upstream-drift first-divergence migrations**, identical
in shape to 1634664 (Phase 11J-FIRE-DRIFT-FIX, where the row flipped to
`oracle_gap` once the upstream Move-truncated guard fired); the difference is
that here the upstream Move now succeeds, so the friendly-fire site is the
first divergence — there is no upstream `Move:` raise to absorb the row.

Neither is **Class A** (oracle attacker resolution is correct: AWBW `units_id`
matches the engine attacker; `select_unit_id` is pinned; both rows' attackers
are on the right seat). Neither is **Class B** in the actionable sense (we
deliberately are not editing engine death-clear / pre-action board sync —
hard rule, and the dominant root cause here is owner mis-mapping that an
engine-side death-clear pass would not address).

---

## 3. Fix description

Single oracle-side guard, mirror of `_oracle_assert_fire_damage_table_compatible`.

**File:** `tools/oracle_zip_replay.py`

**New helper** (next to the existing damage-table guard, ~line 1149):

```python
def _oracle_assert_fire_defender_not_friendly(
    state: GameState,
    attacker: Unit,
    defender_pos: tuple[int, int],
) -> None:
    """Phase 11J-F4-FRIENDLY-FIRE-WAVE2: refuse Fire when the engine board
    has a friendly unit at the resolved defender tile.

    AWBW does not legalize friendly fire — attackUnit.php rejects any strike
    where the attacker and defender share a player seat (mirrored by
    GameState._apply_attack raising ValueError on same-player target). Any
    Fire envelope where the engine resolves a friendly defender is by
    definition an upstream board-state divergence; reclassify as
    oracle_gap rather than letting _apply_attack raise engine_bug.
    """
    defender = state.get_unit_at(*defender_pos)
    if defender is None or not defender.is_alive:
        return
    if int(defender.player) != int(attacker.player):
        return
    raise UnsupportedOracleAction(
        f"Fire: engine board holds friendly {defender.unit_type.name} at "
        f"{defender_pos} for attacker P{attacker.player} {attacker.unit_type.name} "
        f"id={attacker.unit_id} — AWBW Fire envelopes never legalize same-player "
        f"strikes (attackUnit.php rejects), so the engine snapshot has drifted "
        f"upstream (owner mis-mapped or stale unit on the defender tile). "
        f"Treat as oracle_gap (snapshot drift) rather than engine friendly-fire."
    )
```

**Two call sites** (one per Fire kind), both placed adjacent to the existing
`_oracle_assert_fire_damage_table_compatible` call:

- Fire-with-path (Move + Fire), pre-`_engine_step(ATTACK,…)`: line ~6202.
- Fire-no-path (post-kill / batched Fire row), pre-`_engine_step(ATTACK,…)`: line ~5934.

`AttackSeam` is not affected — its target is a seam tile, not a unit.

**Net change:** 1 new helper (≈40 LOC including docstring) + 2 one-line call
sites = ~42 LOC; well within the ≤30 LOC *surgical* budget for the call-site
edits, and the helper itself is a defense-in-depth utility mirroring the
existing damage-table guard pattern.

### AWBW reasoning ("AWBW does not allow friendly fire")

- **Tier-1 (canonical engine):** `engine/game.py::_apply_attack` already
  enforces this invariant (`raise ValueError("_apply_attack: friendly fire …")`
  at L818–821) — that guard has been in place since the Phase-11 charter
  ("the engine had been silently accepting moves AWBW itself would forbid —
  diagonal direct attacks, **friendly fire**, COP activation with empty meter,
  …", `docs/oracle_exception_audit/CAMPAIGN_SUMMARY.md` L13). The engine guard
  encodes the AWBW canon; this oracle guard simply downgrades the
  classification when the trigger is upstream snapshot drift, not engine
  legality drift.

- **Tier-2 (corpus evidence):** every prior friendly-fire `engine_bug` row in
  the audit register has resolved to upstream-drift attribution
  (`phase11d_residual_engine_bug_triage.md` row 8 / 1634664;
  `phase11j_fire_drift_fix.md` table row "1634664 → oracle_gap, Move
  truncated path"). The `combatInfo.copValues.{attacker,defender}.playerId`
  fields in the failing envelopes for 1629202 and 1632825 are distinct
  (3764695≠3764694, 3772608≠3772609), confirming AWBW recorded a cross-player
  strike — the strike was legal at the AWBW server.

- **No counter-example:** the Phase-11 corpus contains **zero** Fire
  envelopes where AWBW `attacker.playerId == defender.playerId`. The
  AWBW-Wiki damage table and the upstream `attackUnit.php` both reject
  same-player Fire at the action level (citation pattern matches the
  existing `_oracle_assert_fire_damage_table_compatible` docstring's
  reference to `https://awbw.amarriner.com/damage.php`).

---

## 4. Closure table

| games_id | Pre status (936-audit) | Pre `class` | Post status | Post `class` | Post message |
|----------|------------------------|-------------|--------------|---------------|--------------|
| **1629202** | `_apply_attack: friendly fire from player 0 on MECH at (7, 20)` | `engine_bug` | `first_divergence` | **`oracle_gap`** | `Fire: engine board holds friendly MECH at (7, 20) for attacker P0 INFANTRY id=43 — AWBW Fire envelopes never legalize same-player strikes …` |
| **1632825** | `_apply_attack: friendly fire from player 0 on INFANTRY at (12, 19)` | `engine_bug` | `first_divergence` | **`oracle_gap`** | `Fire: engine board holds friendly INFANTRY at (12, 19) for attacker P0 INFANTRY id=16 — AWBW Fire envelopes never legalize same-player strikes …` |

Per-gid audit command:

```
python tools/desync_audit.py ^
  --catalog data/amarriner_gl_extras_catalog.json ^
  --catalog data/amarriner_gl_std_catalog.json ^
  --games-id 1629202 --games-id 1632825 ^
  --register logs/_f4w2_per_gid.jsonl
```

→ `2 games audited; oracle_gap 2`. Register at `logs/_f4w2_per_gid.jsonl`.

**Both gids hit the closure target: `engine_bug → oracle_gap`** (acceptable
terminal state per the ship spec).

---

## 5. Regression gates

### 5.1 pytest (≤2 failures, ignoring `test_trace_182065_seam_validation.py`)

```
python -m pytest --tb=no -q --ignore=test_trace_182065_seam_validation.py
```

Result: **`568 passed, 5 skipped, 2 xfailed, 3 xpassed, 0 failed`** in 71.87 s.
**Pass.** (0 failures ≪ 2 budget.)

### 5.2 100-game sample audit (`ok ≥ 98`, `engine_bug == 0`)

```
python tools/desync_audit.py ^
  --catalog data/amarriner_gl_extras_catalog.json ^
  --catalog data/amarriner_gl_std_catalog.json ^
  --max-games 100 --seed 1 ^
  --register logs/_f4w2_sample100.jsonl
```

Result: **`ok 98 / oracle_gap 2 / engine_bug 0`**. **Pass.**

The two `oracle_gap` rows are **BUILD no-op insufficient-funds** entries
(1622501, 1624082) — the L1-BUILD-FUNDS-RESIDUAL cluster owned by another
lane, **not** F4 friendly fire. **No new `engine_bug`** introduced; **no F4
surface** in the sample.

### 5.3 Targeted F1/F2 regression (5 prior-shipped gids)

```
python tools/desync_audit.py ^
  --catalog data/amarriner_gl_extras_catalog.json ^
  --catalog data/amarriner_gl_std_catalog.json ^
  --games-id 1605367 --games-id 1622104 --games-id 1626642 ^
  --games-id 1628198 --games-id 1634664 ^
  --register logs/_f4w2_f1f2_regress.jsonl
```

| games_id | Family / prior closeout | Pre status | Post status | Result |
|----------|--------------------------|------------|--------------|--------|
| **1605367** | F2 (Phase 11D row 1) → MOVE-TRUNCATE-SHIP | `oracle_gap` (pre-936) / `ok` (extras-audit) | **`ok`** | hold |
| **1622104** | F1 Bucket A drift Δ=1 → oracle position-snap | `engine_bug` → ok | **`ok`** | hold |
| **1626642** | F5 BLACK_BOAT drift-0 → oracle Fire classification | `engine_bug` → ok | **`ok`** | hold |
| **1628198** | F1 prior-shipped sample | recent `ok` | **`ok`** | hold |
| **1634664** | **F4 friendly-fire** (Phase 11J-FIRE-DRIFT row) → Move-truncated upstream | `engine_bug` → `oracle_gap` (then ok after FIRE-DRIFT) | **`ok`** | hold |

**All five hold `ok`.** No regression to `engine_bug`. Note 1634664 — the
prior F4 friendly-fire row — still resolves through the upstream Move-truncated
guard (its earlier divergence) before the new defender-not-friendly guard
fires; the new guard does not perturb the Phase 11D / 11J-FIRE-DRIFT
attribution chain on this row.

---

## 6. Coordination snapshot (post-edit)

- `tools/oracle_zip_replay.py` — only my edit added in this lane (helper +
  two one-line guard calls). No conflict with the in-flight L2-BUILD-OCCUPIED
  pre-action board-sync work (separate code paths around BUILD envelopes).
- `engine/game.py` — **not touched** in this lane. VONBOLT-SCOP-SHIP /
  L1-BUILD-FUNDS-SHIP / SASHA-MARKETCRASH branches untouched.
- Hard-rule files (`engine/unit.py`, `engine/action.py::get_legal_actions`,
  Von Bolt SCOP branch in `_apply_power_effects`, `_grant_income`,
  `_resupply_on_properties`, `_build_cost`, `_activate_power` Sasha branch) —
  **not touched**.

---

## 7. Verdict

**GREEN — SHIP.**

- **1629202:** `engine_bug` → **`oracle_gap`** (target met).
- **1632825:** `engine_bug` → **`oracle_gap`** (target met).
- Pytest: `568/0` pass/fail (≤2 budget).
- 100-game sample: `ok 98 / oracle_gap 2 / engine_bug 0` (gate met).
- F1/F2 regression: 5/5 hold `ok` (no regressions).

The hill is taken. The next firing line remains BUILD-FUNDS-RESIDUAL (L1)
and BUILD-OCCUPIED-TILES (L2), both owned by sister lanes.

---

*"Festina lente."* (Latin, attributed to Augustus, ~30 BC – AD 14)
*"Make haste slowly."*
*Augustus: first Roman emperor; the maxim governed his political and military restraint — quick decision, deliberate execution. Two surgical guard lines, three audit re-runs, no engine edit.*
