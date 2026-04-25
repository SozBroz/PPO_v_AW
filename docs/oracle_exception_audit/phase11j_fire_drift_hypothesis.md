# Phase 11J FIRE-DRIFT — Hypothesis (pre-edit)

**Imperator's order**: ENGINE + ORACLE WRITE lane against the seven `engine_bug`
residuals from `logs/desync_register_post_phase10q.jsonl` — six F1 Bucket A
position-drift Fires plus one F4 friendly-fire defender. No edits to
`engine/action.py::get_attack_targets` (read-only since Phase 6).

This document fixes the hypothesis on paper *before* a single line of engine or
oracle code is touched. Closure of each replay must be one of: delete, oracle
fix, or engine fix.

---

## 1. Targets at a glance

Pulled from `logs/desync_register_post_phase10q.jsonl` and confirmed via the
`tools/_phase11j_drill.py` instrumented runner plus `tools/_phase11j_envinspect2.py`
raw envelope dump (output: `logs/phase11j_envinspect2.json`).

| games_id | day | env | aidx | unit         | engine `unit_pos` | oracle `move_pos` | target_pos | engine err |
|----------|-----|-----|------|--------------|-------------------|-------------------|-----------:|------------|
| 1622104  | 1   |  43 | 19   | MECH (P1)    | (7,17)            | (6,17)            | (6,16)     | range (Δ=1) |
| 1625784  | 4   |  35 | 28   | B_COPTER (P1)| (8,5)             | (8,2)             | (8,1)      | range (Δ=3) |
| 1630983  | 1   |  24 | 1    | MECH (P0)    | (13,20)           | (13,22)           | (13,23)    | range (Δ=2) |
| 1631494  | 4   |  46 | 10   | FIGHTER (P0) | (15,4)            | (16,13)           | (15,13)    | range (Δ=10)|
| 1634664  | 1   |  23 | 3    | INFANTRY (P1)| (5,18)            | (5,18)            | (5,19)     | friendly fire |
| 1635025  | 4   |  36 | 21   | B_COPTER (P0)| (16,15)           | (14,19)           | (15,19)    | range (Δ=6) |
| 1635846  | 4   |  31 | 15   | B_COPTER (P1)| (8,9)             | (8,5)             | (8,4)      | range (Δ=4) |

All seven envelopes carry a populated `combatInfoVision.global.combatInfo` —
i.e. AWBW says the strike happened, with concrete attacker/defender HP and
ammo. The engine refuses to apply, throwing inside `_apply_attack`.

---

## 2. AWBW vs engine reconciliation per replay

Source for every row: `logs/phase11j_envinspect2.json` (Move.unit.global +
combatInfo.{attacker,defender}) cross-referenced against the engine snapshot
from `tools/_phase11j_drill.py`.

### 2.1 — 1622104 MECH @ (6,17)→(6,16) — F1 / ammo=0

```
Move.unit.global: id=192101222 type=Mech (y,x)=(6,17) ammo=0 fuel=49
paths: start=(7,17) end=(6,17)         (single tile move N)
attacker (post): (y,x)=(6,17) ammo=0 hp=0  ← attacker died from counter
defender (post): id=192263550 (y,x)=(6,16) ammo=7 hp=3   (TANK survives)
```

- **AWBW reality**: MECH at (7,17), zero primary ammo, walks 1 tile N to
  (6,17) and strikes the TANK at (6,16) with its **secondary MG**. Mech-MG vs
  Tank does ~5 damage (TANK was 8, ends at 3 display); counter wipes the
  attacker.
- **Engine reality at failure**: attacker correctly resolved (MECH at (7,17),
  `selected_unit` is set), `move_pos=(6,17)`. `get_attack_targets` returns
  empty because `engine/action.py:283-284` shorts out when
  `stats.max_ammo > 0 and unit.ammo == 0` — even though Mech has unlimited MG
  ammo against Inf/Mech. (Phase 10A patched MG accounting *consumption* but
  did not patch the gating.)
- **Why range looks wrong**: it isn't a position drift at all. Engine and
  AWBW agree on the same firing tile (6,17). The `Δ=1` in the residual is
  measured from `unit_pos` (pre-move (7,17)) to `move_pos` (post-move (6,17))
  by the residual classifier — it lumps every empty `get_attack_targets`
  return into Bucket A.
- **Closure plan**: ENGINE — bypass the defense-in-depth range check in
  `_apply_attack` when `_oracle_combat_damage_override` is set. The override
  carries AWBW's actual rolled damage; the strike already occurred upstream
  and the oracle has authority. **Do not touch `get_attack_targets`.**

### 2.2 — 1630983 MECH @ (13,22)→(13,23) — F1 / ammo=0

```
Move.unit.global: id=192308520 type=Mech (y,x)=(13,22) ammo=0 fuel=57
paths: start=(13,20) end=(13,22)
attacker (post): (y,x)=(13,22) ammo=0 hp=6
defender (post): id=192359723 (y,x)=(13,23) ammo=0 hp=0   (INF, killed)
```

- **AWBW reality**: P0 MECH starts at (13,20), walks 2 tiles E to (13,22),
  fires MG into INFANTRY at (13,23) — kill. MECH ammo was 0 at start.
- **Engine reality**: `selected_unit` is the MECH at (13,20),
  `move_pos=(13,22)`, `get_attack_targets` empty for the same reason as 2.1.
- **Closure plan**: same as 2.1 (override-bypass).

### 2.3 — 1635025 B_COPTER @ (14,19)→(15,19) — F1 / ammo=0 (10A residual)

```
Move.unit.global: id=192625206 type=B-Copter (y,x)=(14,19) ammo=0 fuel=36
paths: start=(16,15) end=(14,19)
attacker (post): (y,x)=(14,19) ammo=0 hp=6
defender (post): id=192662624 (y,x)=(15,19) ammo=8 hp=3   (TANK, survives)
```

- **AWBW reality**: B-COPTER at (16,15), no missiles left, hops 4 tiles to
  (14,19) and uses MG on the TANK at (15,19). Phase 10A taught the engine to
  *consume* MG ammo unlimitedly here, but `get_attack_targets` still gates on
  primary ammo, so the strike is still rejected.
- **Closure plan**: same override-bypass.

### 2.4 — 1625784 B_COPTER @ (8,2)→(8,1) — F1 / ammo drift (10A residual)

```
Move.unit.global: id=192059738 type=B-Copter (y,x)=(8,2) ammo=1 fuel=45
attacker (post): (y,x)=(8,2) ammo=0 hp=5     (used last missile, survived)
defender (post): id=192328847 (y,x)=(8,1) ammo=9 hp=2     (TANK, survives)
```

- **AWBW reality**: AWBW pre-strike ammo=1; B-COPTER fires its **last
  missile** (correct primary attack, not MG) into the TANK.
- **Engine reality at fail**: drill snapshot reports engine B-COPTER ammo=0
  (ammo drift between AWBW and engine). With ammo=0 the engine flips to the
  empty-targets branch.
- **Hypothesis on the drift origin**: prior-turn primary missile previously
  consumed by an engine roll the oracle did not pin — i.e. an earlier action
  in this replay is double-decrementing primary ammo (or the engine is
  spending primary on what AWBW serviced as MG). Localising this drift is
  out of scope for this phase.
- **Closure plan**: override-bypass closes the symptomatic crash. The
  underlying ammo drift becomes a pure-state divergence (no `engine_bug`),
  surfacing in F2 (state diff) on the next audit pass — that is the right
  bucket for it. Documented as carry-forward.

### 2.5 — 1635846 B_COPTER @ (8,5)→(8,4) — F1 / ammo drift (10A residual)

```
Move.unit.global: id=192664884 type=B-Copter (y,x)=(8,5) ammo=1 fuel=56
attacker (post): (y,x)=(8,5) ammo=0 hp=1     (last missile, near-dead)
defender (post): id=192646462 (y,x)=(8,4) ammo=4 hp=5    (TANK, survives)
```

- Same shape as 2.4: AWBW had ammo=1, engine drifted to ammo=0.
- **Closure plan**: same as 2.4 — override-bypass; underlying ammo drift
  handled in a follow-up phase.

### 2.6 — 1631494 FIGHTER @ (16,13)→(15,13) — F1 / DEFENDER drift (Δ=10)

```
Move.unit.global: id=192507226 type=Fighter (y,x)=(16,13) ammo=9 fuel=64
paths: start=(15,4) end=(16,13)        (10-tile move SE — fighter has 9MP, OK)
attacker (post): (y,x)=(16,13) ammo=8 hp=8
defender (post): id=192553831 (y,x)=(15,13) ammo=8 hp=3
```

- **AWBW reality**: P0 FIGHTER flies (15,4)→(16,13), strikes a foe at
  (15,13) (a flier — has its own ammo=8 in combatInfo, so likely a B-COPTER
  or another fighter). No defender drift on the AWBW side.
- **Engine reality**: at the moment `_apply_attack` runs, engine's tile
  (15,13) holds **no foe**. The oracle resolver
  `_oracle_fire_resolve_defender_target_pos` falls back to a Chebyshev-1 ring
  search and lands on a TANK at (14,13) (the only nearby foe). FIGHTER vs
  TANK has `base_damage = None` in the damage table, so `get_attack_targets`
  refuses to admit (14,13) as a strike target. Range check fires.
- **Why the engine has no foe at (15,13)**: the AWBW unit (192553831) was
  never spawned in the engine, or was killed/displaced in a prior diverged
  action. Without proper AWBW-id ↔ engine-id plumbing
  (`_unit_by_awbw_units_id` returns None because engine ids are monotonic
  small-ints, not PHP ids — see comment at line 2052 of
  `oracle_zip_replay.py`), the resolver cannot tell that the recorded foe
  simply isn't on the engine map.
- **Closure plan**: ORACLE — in the Fire branch, after the defender
  position is resolved, validate that `get_base_damage(attacker.unit_type,
  resolved_defender.unit_type)` is non-`None`. If it is `None`, raise
  `UnsupportedOracleAction` (re-classifies from `engine_bug` → `oracle_gap`,
  the correct bucket for "oracle cannot reproduce this strike with the
  engine units it has"). The override-bypass would otherwise apply nonsense
  damage to a TANK the AWBW Fighter never engaged.

### 2.7 — 1634664 INFANTRY @ (5,18)→(5,19) — F4 / friendly fire

```
Move.unit.global: id=192727534 type=Infantry (y,x)=(5,18) ammo=0 fuel=90
paths: start=(2,18) end=(5,18)        (3 tiles S)
attacker (post): (y,x)=(5,18) ammo=0 hp=4
defender (post): id=192726928 (y,x)=(5,19) ammo=0 hp=0   (INF, killed)
```

- **AWBW reality**: P1 INFANTRY at (2,18), walks S to (5,18), bayonets the
  P0 INFANTRY at (5,19) to death. Owner bits are clean: P1 hits P0.
- **Engine reality**: drill snapshot shows **two units co-located at
  (5,18)** — the moving P1 INF and a stationary P0 INF (cargo-spawned or
  prior turn). `_apply_attack` opens with
  `attacker = self.get_unit_at(*action.unit_pos)` which returns the *first*
  unit at that tile — in this case the P0 INF — so the friendly-fire guard
  trips when the picked attacker (P0) tries to strike the (5,19) defender
  (also P0).
- **`state.selected_unit` is correctly set to the P1 INF** at the moment
  `_apply_attack` is called (engine's STEP-GATE plumbed it through Move).
- **Hypothesis classification**: this is "wrong attacker resolved by
  oracle/engine pair" (Bucket B-shaped). NOT "AWBW envelope truly
  self-targeted" — combatInfo has clean opposite owners. NOT "owner-bit
  corruption" — both INFs are correctly owned in their respective slots.
- **Closure plan**: ENGINE — when `state.selected_unit is not None` and
  `selected_unit.pos == action.unit_pos` and `selected_unit.is_alive`,
  prefer it over `get_unit_at`. This is invariant-tightening: if STEP-GATE
  selected the unit, that unit is canonically the actor. Falls back to
  `get_unit_at` only when no `selected_unit` is recorded (legacy paths,
  seam attacks, tests).

---

## 3. The four hypothesised pathologies

Collapsing the seven into root causes:

**P-AMMO** (cases 2.1, 2.2, 2.3, 2.4, 2.5 — five of seven):
`get_attack_targets` shorts to `[]` whenever `unit.ammo == 0` even though
AWBW canon allows secondary-MG strikes (Inf/Mech/B-Copter/Tank) and even
though `_oracle_combat_damage_override` already pins the post-strike HPs.
The defense-in-depth range check in `_apply_attack` (L654-661) calls into
that same gate and refuses an attack the oracle has authority to apply.
*Why now and not in Phase 10A*: 10A taught `_apply_attack` to consume MG
ammo unlimitedly during the **damage roll**, but the **pre-attack
legality check** still goes through the empty `get_attack_targets`. Also,
2.4/2.5 are "engine ammo drifted to 0 even though AWBW had 1" — the
override-bypass fixes the symptom; the underlying pre-strike accounting
drift is logged for follow-up.

**P-DRIFT-DEFENDER** (case 2.6 — one of seven): defender resolver picks an
incompatible engine unit on the Chebyshev-1 ring (FIGHTER vs TANK has no
damage entry). Override would otherwise apply garbage damage. Needs an
oracle-side compatibility filter that raises `UnsupportedOracleAction`,
which re-buckets into `oracle_gap` (the truthful classification — the
oracle cannot map this strike onto the engine state it has).

**P-COLO-ATTACKER** (case 2.7 — one of seven):
`get_unit_at(action.unit_pos)` returns a non-actor when the tile is
co-occupied (real and a cargo or stationary unit). `state.selected_unit`
is the canonical actor; `_apply_attack` should prefer it.

**P-AMMO-DRIFT** (sub-shape of cases 2.4, 2.5): pre-strike ammo on
B-COPTERs has drifted from AWBW. Symptom-closed by P-AMMO fix. Root
cause: out of scope; carry-forward to a Phase 11K (or later) state-diff
audit. Not gated by the engine_bug count which is the only metric this
phase commits to not regress.

---

## 4. Fix design (paper sketch — not yet code)

### 4.1 Engine — `engine/game.py::_apply_attack` (write zone)

Two surgical edits at the head of the function. Both are invariant-tightening,
neither widens the engine's behaviour outside the oracle path.

**Edit A (P-COLO-ATTACKER)** — prefer `selected_unit` when it agrees with
`action.unit_pos` and is alive:

```python
attacker = None
sel = self.selected_unit
if sel is not None and sel.is_alive and sel.pos == action.unit_pos:
    attacker = sel
if attacker is None:
    attacker = self.get_unit_at(*action.unit_pos)
if attacker is None:
    raise ValueError(f"_apply_attack: no attacker at {action.unit_pos}")
```

This is safe outside oracle replay because `selected_unit` is always None
at the start of any RL `step`/`reset` (cleared by `_finish_action`), so
the new branch is inert in normal play. STEP-GATE in oracle replay sets
it just before the ATTACK action.

**Edit B (P-AMMO)** — when the oracle pinned the damages, skip the
defense-in-depth range/legality check:

```python
oracle_pinned = self._oracle_combat_damage_override is not None
if defender_pre is not None and not oracle_pinned:
    atk_from = action.move_pos if action.move_pos is not None else attacker.pos
    if action.target_pos not in get_attack_targets(self, attacker, atk_from):
        raise ValueError(...)
```

Friendly-fire guard remains unconditional — it is a correctness invariant,
not a legality check. Edit A ensures it sees the right attacker.

The override is consumed (set to None) at L684, so a stray subsequent
`step` will hit the legality check normally.

### 4.2 Oracle — `tools/oracle_zip_replay.py` (write zone)

**Edit C (P-DRIFT-DEFENDER)** — at the Fire path call site
(`apply_oracle_action_json` Fire branch), after
`_oracle_fire_resolve_defender_target_pos` returns and the engine
defender is fetched, validate damage compatibility:

```python
defender_unit = state.get_unit_at(*defender_pos)
if defender_unit is not None and defender_unit.is_alive:
    from engine.combat import get_base_damage
    if get_base_damage(attacker.unit_type, defender_unit.unit_type) is None:
        raise UnsupportedOracleAction(
            "Fire: oracle resolved defender type "
            f"{defender_unit.unit_type.name} that {attacker.unit_type.name} "
            "cannot damage; AWBW combatInfo refers to a unit not on the "
            "engine map (likely upstream resolver miss)."
        )
```

This re-buckets 1631494 from `engine_bug` to `oracle_gap` — the truthful
classification.

### 4.3 What is *not* changed

- `engine/action.py::get_attack_targets` — read-only since Phase 6.
- `_oracle_set_combat_damage_override_from_combat_info` — already correct
  for cases where the engine defender matches the AWBW intent.
- `_unit_by_awbw_units_id` — the engine-id ↔ AWBW-id plumbing is a
  larger restructuring out of scope for this phase.

---

## 5. Expected closures

| games_id | closure       | mechanism |
|----------|---------------|-----------|
| 1622104  | engine fix    | Edit B (P-AMMO override-bypass) |
| 1625784  | engine fix    | Edit B (P-AMMO override-bypass); ammo drift carry-forward |
| 1630983  | engine fix    | Edit B (P-AMMO override-bypass) |
| 1631494  | oracle fix    | Edit C (P-DRIFT-DEFENDER → oracle_gap) |
| 1634664  | engine fix    | Edit A (P-COLO-ATTACKER prefer selected_unit) |
| 1635025  | engine fix    | Edit B (P-AMMO override-bypass) |
| 1635846  | engine fix    | Edit B (P-AMMO override-bypass); ammo drift carry-forward |

Net effect on the desync register:
- `engine_bug` count: −7 on these targets (six become silent, one
  reclassifies to `oracle_gap`).
- `oracle_gap` count: +1 (1631494).
- No other replay touched by Edit A/B/C in normal play.

---

## 6. Risks / weak flanks

- **Edit B over-permits**: oracle could ship a damage override for a
  defender at the wrong tile (as in 1631494). Mitigated by Edit C: any
  type-incompatible defender raises before Edit B sees the override.
  Same-type incompatible distance (e.g. defender at (5,5) but attacker
  at (1,1)) would still let override damage land — but
  `_oracle_set_combat_damage_override_from_combat_info` derives `dmg`
  from the engine defender's HP delta, so a wrong defender would
  produce a damage reading from the wrong unit's HP — still wrong, but
  the existing oracle logic already trusts that resolver. Edit C only
  hardens the type axis; positional drift on same-type-compatible
  defenders is unchanged from current behaviour.
- **Edit A semantics**: there is exactly one place where the wrong-unit
  pickup matters (co-located stack). If an oracle path ever fires
  without setting `selected_unit` and the tile is co-occupied, Edit A
  is inert. Need a regression test that explicitly seeds two units
  on the attacker tile with `selected_unit` set and asserts the right
  one strikes.
- **Phase 6 read-only contract**: `get_attack_targets` is untouched.
- **Carry-forward debt (1625784, 1635846)**: ammo drift remains. The
  acceptance contract is "engine_bug count not higher than 100-game
  10Q baseline" — this phase reduces engine_bug; ammo drift, if it
  surfaces as F2 state diff next pass, is a Phase 11K target.
- **1634664 cargo question**: the second INF at (5,18) — is it cargo
  or a stationary on-tile unit? Viewer triage will confirm. Either
  way Edit A is the right resolution.

---

## 7. Viewer triage plan

Per `desync-triage-viewer` skill, two targets demand the C# AWBW Replay
Player before code edits:
- **1631494** (Δ=10, largest drift): step to env 46 / aidx 10 to see what
  AWBW shows at (15,13) and confirm the resolver-miss hypothesis (P-DRIFT-
  DEFENDER).
- **1634664** (F4): step to env 23 / aidx 3 to see whether the second INF
  at (5,18) is cargo, just-loaded, or co-positioned (P-COLO-ATTACKER).

Both are launched via:
`tools/launch_awbw_viewer.ps1 -Zip <path> -GotoDay <d> -GotoAction <a>`
per the skill's "section 4a" lookup order.

---

*"Si vis pacem, para bellum."* (Latin, c. 4th c. AD)
*"If you wish for peace, prepare for war."* — Vegetius, *De Re Militari*
*Vegetius: late-Roman writer on military doctrine; the line summarises the Roman view that disciplined preparation is what keeps the legions out of unnecessary fights.*
