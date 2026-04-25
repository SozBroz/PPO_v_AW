# Phase 11K-FIRE-STANCE-FRIENDLY-FIX — gid 1635679 / 1635846 close-out

**Status:** SHIPPED. **Audit floor:** 931 ok / 5 oracle_gap → **935 ok / 1 oracle_gap / 0 engine_bug** (+4 closures, 0 regressions).
**Both targets closed:** 1635679 → ok, 1635846 → ok.

## Verdict

The Sturm SCOP work shipped earlier in Phase 11J was correct. The residual `1635679`
funds drift was **not** a Sturm rounding issue. It was a **fire-stance resolver bug**
in `tools/oracle_zip_replay.py` that picked a friendly-occupied tile as the
attacker's firing position, causing the engine to pass an attacker through a
friendly capping infantry on its way to the real fire position. That transit
silently reset the friendly's `capture_points` from 3 → 20 in
`engine/game.py::_move_unit_forced`, blocking the capture of an enemy base.
The missing base wiped one Sturm-owned property, costing him **−1000 g/turn**
(income loss + repair attribution drift), enough to fail a NEO_TANK build
several days later.

Once the resolver no longer returns friendly-occupied stances, the cap stays
intact, the base is captured on schedule, and Sturm's funds match PHP through
end-of-game.

## 1-line root cause

`_oracle_resolve_fire_move_pos` accepted a friendly-occupied path waypoint as
`fire_pos` because `compute_reachable_costs` legitimately returns friendly
tiles when the mover would JOIN (same-type injured ally) or LOAD (transport),
and the resolver only filtered out **transport-stacking**, not generic
friendly occupancy.

## Empirical anchor — gid 1635679, env 22, ai=6

AWBW Fire action: P0 Sturm INFANTRY (id=21) at (7,10) → path
`[(7,10),(7,9),(6,9),(6,8)]` → strike target enemy INFANTRY at (7,8).
The intermediate (7,9) was occupied by a friendly Sturm INFANTRY (id=7) capping
a neutral city (`capture_points=3`).

| step | engine state at (7,9) | engine state at (6,8) | unit 21 pos |
|---|---|---|---|
| pre ai=6 | INFANTRY id=7 hp=68, cap=3 | empty | (7,10) |
| `_oracle_resolve_fire_move_pos` returned `(7,9)` (BUG) | unchanged | empty | (7,10) |
| `_apply_attack` `_move_unit(unit_21, (7,9))` | id=7 + id=21 (overlap) | empty | (7,9) |
| post-attack snap `_move_unit_forced(u, json_path_end=(6,8))` triggers `old_prop.capture_points = 20` for (7,9) | **cap reset 3 → 20** | id=21 hp=77 | (6,8) |

The cap reset wiped the in-progress capture. The base never flipped to Sturm
in the engine, so the engine was missing 1 Sturm-owned property from env 25
day 13 onward. Income (−1000 g/day base) and the funds-cascade explanation in
`logs/phase11j_repair_trace_1635679.txt` both flow from this single missing
property.

## Why `(7,9)` looked legal to the resolver

`compute_reachable_costs` (engine/action.py) returns a friendly-occupied tile
when **either** of these is true (line 296–305):

- `units_can_join(unit, occupant)` — same `unit_type` and `occupant.hp < 100`,
- `unit.unit_type in get_loadable_into(occupant.unit_type)` and the transport
  has free cargo capacity.

Unit 21 (INFANTRY hp=86) joining unit 7 (INFANTRY hp=68) is a legal JOIN
terminator, so `(7,9)` appears in `costs`. The resolver then asked
"can I attack `(7,8)` from `(7,9)`?" — yes, infantry are direct-fire — and
returned `(7,9)` as the firing stance.

`(6,8)` (the AWBW path tail) was **not** in `costs` because the engine
reachability frontier from `(7,10)` couldn't carry enough remaining MP through
the friendly chain at `(6,9)` and `(7,9)` to land at `(6,8)` and stop. The
existing post-attack snap (`Phase 11J-MOVE-TRUNCATE-SHIP`, line 6916–6930 in
`oracle_zip_replay.py`) would have correctly forced the attacker to `(6,8)`
**after** the strike — but only if the resolver returned `(6,8)` as `fire_pos`
in the first place, or returned an unreachable tile so the unreachable-snap
branch (line 6868–6875) kicked in.

## Citations (Tier 1 — AWBW canonical)

The semantics being protected are not new CO mechanics — they are the engine's
own capture-point reset rule. Engine canon comment (`engine/game.py`
`_move_unit` and `_move_unit_forced`):

> "AWBW capture-progress reset on tile vacated: when a capping unit leaves
> its tile, the partial capture resets to 20."

This rule is correct. The bug was that the **oracle replay tool** caused the
attacker to vacate `(7,9)` even though AWBW's path never put the attacker on
`(7,9)` as a stop — it only passed through. The fix keeps the engine's
capture-reset semantics intact and instead prevents the oracle from generating
the spurious vacate event.

## What shipped

### `tools/oracle_zip_replay.py::_oracle_resolve_fire_move_pos`

Two surgical guards that exclude any tile occupied by a different friendly
unit (the attacker's own start tile is still allowed). One inside
`attacks_from` (covers the path-waypoint search and the ranked-fallback loop),
one mirroring the same check on the `(er, ec)` snapped-end fallback at
line 220.

```python
def attacks_from(pos: tuple[int, int]) -> bool:
    if pos not in costs:
        return False
    # Phase 11K-FIRE-STANCE-FRIENDLY-FIX: ``compute_reachable_costs`` includes
    # friendly tiles when the mover would JOIN (same-type injured ally) or
    # LOAD into a transport — both legal *terminators* but never legal *fire
    # stances*. ...
    occ = state.get_unit_at(*pos)
    if occ is not None and occ is not unit:
        return False
    if _oracle_fire_stance_would_stack_on_transport(state, unit, pos):
        return False
    return (tr, tc) in get_attack_targets(state, unit, pos)
```

When the resolver now finds no friendly-free reachable tile that can hit the
target, the existing fallback chain in the Fire action handler (line 6868–6875)
forces the attacker to `(er, ec)` directly — exactly the post-strike behavior
the post-attack snap was always doing for `json_fire_path_end`.

### `test_co_power_direct_damage.py::TestHawkeWaveVsStorm`

Pre-existing test was asserting the old uniform-heal Hawke behavior (both COP
and SCOP heal +20 internal HP). The Phase 11J-FINAL-HAWKE-CLUSTER fix already
in the working tree corrected this to canonical Black Wave +10 / Black Storm
+20 per the AWBW CO Chart, but the test was not updated. Renamed and split
the assertion to reflect canon. Engine code unchanged.

## Validation

### Targeted gids

```
[1635679] ok day~None acts=862
[1635846] ok day~None acts=809
```

### Full 936 audit

```
register -> logs/desync_register_phase11k_friendly_fix.jsonl
936 games audited
  ok           935
  oracle_gap     1
```

`Δ` from prior 931/5/0 floor: **+4 closures, 0 regressions**.

Surviving `oracle_gap`:

- `1607045` — TANK build no-op at `(14,2)` for engine P1, short **180 g** at
  envelope index 27, day 14. Outside the Sturm/Hawke target lane and outside
  this fix's blast radius. Already documented as a pre-existing residual in
  `phase11j_gid_1607045_regression_close.md`.

### Pytest

```
1 failed, 690 passed, 5 skipped, 2 xfailed, 3 xpassed
FAILED test_trace_182065_seam_validation.py::test_full_trace_replays_without_error
```

The single red is the same pre-existing failure called out in the prior
Phase 11J-FINAL-STURM closeout (`Illegal move: Infantry from (9,8) to (11,7)
terrain id=29` — verified pre-existing via `git stash`). No new red.

### Targeted env 22 trace (`tools/_phase11k_env23_step.py --env 22`)

Before fix:

```
ai=5 Capt: prop_at_(7,9)=(None, 34, 3) unit_at_(7,9)=(0, 'INFANTRY', 68, 7, True)
[_move_unit] unit_id=21 from (7, 10) -> (7, 9)         ← spurious snap
[_move_unit_forced] unit_id=21 from (7, 9) -> (6, 8)
[cap-reset _move_unit_forced] unit_id=21 from (7, 9) cap_was=3->20   ← BUG
ai=6 Fire: prop_at_(7,9)=(None, 34, 20)                ← cap wiped
```

After fix:

```
ai=5 Capt: prop_at_(7,9)=(None, 34, 3) unit_at_(7,9)=(0, 'INFANTRY', 68, 7, True)
ai=6 Fire: prop_at_(7,9)=(None, 34, 3)                 ← cap preserved
```

Capture completes on schedule in env 23, base flips to Sturm, income matches
PHP through end-of-game.

## Cluster impact

This fix is a **generic** safety guard on a generic resolver path. Any other
gid where an attacker's AWBW path passes through a friendly that could have
been a JOIN or LOAD terminator was exposed to the same silent cap-reset.
The +4 closures from a single targeted change suggests at least 3 other gids
in the corpus were sitting on the same root cause.

## Counsel

The remaining 1 oracle_gap (`1607045`) is a **180 g** TANK build no-op on
day 14 — small and isolated. It has its own existing closeout doc
(`phase11j_gid_1607045_regression_close.md`) classifying it as a known
residual. If you want the floor flat at 936/0/0, that gid is the next lane;
otherwise the campaign is concluded with a clean 935/1/0.

A second-order hardening worth considering: the same JOIN/LOAD vs Fire-stance
confusion exists in `tools/oracle_zip_replay.py::_oracle_path_tail_*` helpers
and a few other resolvers that consult `compute_reachable_costs` without
filtering by occupant. None are firing in the current 936 set, but the same
pattern would surface again if the catalog grows. A small refactor to a
shared `_costs_excluding_friendly_terminators(state, unit)` helper would
eliminate that class of bug entirely.

---

*"Strategy without tactics is the slowest route to victory. Tactics without strategy is the noise before defeat."* — Sun Tzu, *The Art of War* (5th c. BC)
*Sun Tzu: ancient Chinese general and military strategist, traditional author of The Art of War.*
