# HP belief — engine truth vs bot-visible information

This document specifies how the AI must observe enemy HP the way an AWBW
player does — by **bucket plus combat-derived intervals** — without leaking
the engine's exact integer HP through the observation tensor.

## Why this matters

AWBW shows enemy HP as a single bar (10% bucket). Strong humans then narrow
that bucket using the **damage formula**: each combat has a known min/max
damage range driven by luck, terrain, CO bonuses, and CO power state. They
use the *range of plausible outcomes* to:

- Find lucky-flip lines (e.g. "if I roll high, I one-shot this Tank and my
  Md.Tank lives — even a low roll keeps me alive").
- Force kills (stack damage so the **upper** plausible HP after the attack is
  ≤ 0 — the kill is guaranteed regardless of luck).
- Open follow-up lines (kill the screen unit so the indirect behind it is
  exposed next turn).

Today the encoder writes the **exact integer HP** of every unit on the board:

```190:191:rl/encoder.py
            # HP: latest unit written wins — acceptable for non-stacked units
            spatial[r, c, hp_ch] = unit.hp / 100.0
```

That is full-information for the policy and is wrong for any "AWBW-fair"
play, evaluation, or BC inference against humans.

## Two layers, one engine

| Layer | What it sees | Owner |
|-------|--------------|-------|
| **Engine ground truth** (`engine.unit.Unit.hp`, `engine.combat`, `engine.game.GameState`) | Exact integer HP, exact damage roll, exact funds | Simulation. Untouched by this work. |
| **Bot belief** (per-perspective overlay) | Per-unit `(hp_min, hp_max)` integers, plus a known `display_bucket` | New module. Consumed by encoder / policy / search. |

The engine continues to roll real damage. The belief layer is a **per-player
overlay** that mirrors what each side actually knows.

## Belief representation

For each unit visible to the observer, maintain:

```text
display_bucket  : int in 1..10           # ceil(hp / 10)
hp_min, hp_max  : int in 1..100, hp_min <= hp_max
```

Invariants:

1. `display_bucket * 10 - 9 <= hp_min`  and  `hp_max <= display_bucket * 10`
   (the interval is always inside the visible bar).
2. For the observer's **own** units, `hp_min == hp_max == unit.hp`.
3. For unobserved units (fog), no belief entry exists (not modelled in the
   current engine, but the API should leave that hole open).

## Update rules

| Event | Effect on belief |
|-------|------------------|
| Build / deploy | `hp_min = hp_max = 100` (revealed exact). |
| Visible attack on this unit | New `hp_min = max(0, prev_hp_min - dmg_max)`, `hp_max = max(0, prev_hp_max - dmg_min)`. Then re-clamp to the new `display_bucket`. |
| Counter-attack on this unit | Same as above (counter damage range). |
| Day-end repair on owned terrain (visible) | Add `+20` to both bounds, then clamp to `[1, 100]` and to the new bucket. |
| Black Boat repair on adjacent ally (visible) | Add `+10` to both bounds, then clamp. |
| Capture / load / unload / wait | No HP change. |
| Re-sight after losing sight (FoW future work) | Reset to bucket-only: `hp_min = bucket*10 - 9`, `hp_max = bucket*10`. |

The damage range `(dmg_min, dmg_max)` comes from the combat formula already
in [`engine/combat.py`](../engine/combat.py): swap the single rolled value
for the `(luck=0, luck=9)` pair under the same terrain / CO / power state.
That function should be exposed as `damage_range(...)` and called by the
belief updater alongside the existing roll.

## API sketch (new file)

`engine/belief.py` — engine-adjacent because update rules depend on engine
combat math, but observer-scoped state lives outside `GameState`:

```python
@dataclass
class UnitBelief:
    unit_id: int
    display_bucket: int
    hp_min: int
    hp_max: int

class BeliefState:
    def __init__(self, observer: int): ...
    def on_unit_built(self, unit: Unit) -> None: ...
    def on_attack(self, attacker: Unit, defender: Unit, dmg_min: int, dmg_max: int) -> None: ...
    def on_repair(self, unit: Unit, amount: int) -> None: ...
    def reveal_bucket(self, unit: Unit) -> None: ...
    def get(self, unit_id: int) -> UnitBelief | None: ...
```

Two `BeliefState` instances (one per seat) are wired into the env wrapper
(see below). The engine emits the events; the belief layer maintains the
overlay.

## Encoder migration

Replace the single HP channel in [`rl/encoder.py`](../rl/encoder.py) with
**two** channels per perspective:

- `hp_lo_ch`: `belief.hp_min / 100.0`
- `hp_hi_ch`: `belief.hp_max / 100.0`

Own units land at `hp_lo == hp_hi == unit.hp / 100`. Enemy units land at
their belief interval. The policy can read both ends and learn to evaluate
"forced kills" (where `hp_hi - dmg_lo <= 0` after a planned strike) versus
"flip-shots" (where `hp_lo - dmg_hi <= 0` only on lucky rolls).

This is a **breaking change** to encoder dimensionality — same caveat as the
existing comment in [`rl/encoder.py`](../rl/encoder.py): old `latest.zip` is
incompatible. Plan a coordinated retrain.

## One rule, same logic everywhere

**Decision (Imperator directive, 2026-04-19):** no `hp_observation` toggle.
The bot and the on-screen viewer see the **same** bucketed information a
human AWBW player does. The engine alone owns the exact 0–100 integer.

Consequences:

- `encode_state` always emits two HP channels. Observer's own units land at
  `hp_lo == hp_hi == unit.hp / 100`; enemy units land at the belief
  interval inside their visible bucket.
- `server/static/board.js` draws the HP strip at **bucket granularity**
  (multiples of 10%) by design. `server/write_watch_state.py` serialises
  `display_hp * 10` (10, 20, …, 100) so exact HP never leaves the engine
  across the JSON boundary into any viewer or replay frame.
- BC (`scripts/train_bc.py`, `tools/oracle_zip_replay.py`) also owes a
  belief-aware ingest — follow-up task below. Until regenerated, legacy BC
  zips encode exact-HP observations and will drift against a belief-trained
  policy.

## Where this hooks in

- [`engine/combat.py`](../engine/combat.py) — `damage_range(attacker, defender, ...)` returning `(min, max)` via `luck_roll=0` / `luck_roll=9` (plus the negative-luck corner cases for Flak/Jugger). Reuses the existing rolled path, no state mutation.
- New [`engine/belief.py`](../engine/belief.py) — `BeliefState` + `UnitBelief`. Engine-adjacent; no `GameState` mutation.
- [`rl/env.py`](../rl/env.py) — owns `self._beliefs = {0: BeliefState(0), 1: BeliefState(1)}`. Seeds from initial state, updates around every `step()`. The opponent policy is now fed an observation rendered from `observer=1` + `self._beliefs[1]`, not P0's perspective (closes a long-standing info-leak on the blue seat).
- [`rl/encoder.py`](../rl/encoder.py) — two HP channels (`hp_lo_ch`, `hp_hi_ch`). `encode_state(state, *, observer=0, belief=None)`. When `belief=None`, both channels fall back to `unit.hp / 100` (debug / tool use). Spatial channel count 62 → 63.
- [`server/write_watch_state.py`](../server/write_watch_state.py) — `units_list` emits `hp = unit.display_hp * 10` (bucket-aligned 10..100). The raw 0–100 integer never crosses the JSON boundary.
- [`server/static/board.js`](../server/static/board.js) — `drawUnitHpBar` receives bucket-aligned values; rendering is implicitly bucketed. Optional: snap to exact 10% steps defensively.

## Follow-up — BC belief regeneration

Legacy BC zips (`human_bc.zip`, `amarriner_bc.zip`) were recorded against the
old encoder (single exact-HP channel). A belief-trained policy warm-started
from them will drift. To fully honour "same logic everywhere":

1. Extend `tools/oracle_zip_replay.py` to maintain a `BeliefState(observer=seat)` alongside the engine replay, emitting the same events the env does.
2. Encoder now takes `belief` — pass the belief from the replay into `encode_state`.
3. Regenerate BC zips through the belief-aware ingest.

Until this is done, BC pipelines call `encode_state` with `belief=None`
(fallback = exact HP both channels). Noted here so the drift is explicit
and auditable, not silent.

## Checkpoint retirement

This change retires every encoder-incompatible checkpoint:

- `Z:\checkpoints\latest.zip` (last fleet-resume, ~251k steps past the capture-fix)
- `Z:\checkpoints\checkpoint_0000.zip`
- `Z:\checkpoints\promoted\best.zip` (the capture-fix winner)
- All `Z:\checkpoints\pool\*\latest.zip`

Archive (do not delete — provenance) to `Z:\checkpoints\_attic_pre_hp_belief_<ts>\`.
Start the next training run from scratch or from a belief-regenerated BC.

## Test plan (sketch)

- `damage_range` returns `min <= roll <= max` for many sampled cases.
- Belief shrinks to a singleton when a unit is freshly built; grows back to
  bucket-only after a heal that crosses bucket boundaries.
- Forced-kill detection: a stack of attacks whose summed `dmg_min` exceeds
  current `hp_hi` is reported as guaranteed lethal.
- Flip-shot detection: `dmg_max >= hp_lo` while `dmg_min < hp_lo`.
- Encoder regression: with `hp_observation=exact`, channel layout matches
  today's tests; with `belief`, lo == hi for own units.

## Risks and counsel

- **Checkpoint breakage.** Any encoder layout change retires existing
  `latest.zip`. Coordinate with the next planned retrain rather than
  shipping mid-run.
- **BC data leakage.** BC rows generated today encode exact HP; training
  a "belief" policy on `exact` rows will drift. Either keep BC on `exact`
  permanently, or regenerate rows through the belief layer (more work).
- **Combat power states.** CO powers (e.g. Max, Sturm, Hawke) shift the
  damage range; `damage_range` must take the same `co_state` snapshot the
  rolled call uses to avoid silent inconsistency.
- **Counter-attacks.** Both attacker and defender HP shift in one engine
  call; emit two belief updates from the same combat event, in order.
- **FoW.** Not currently modelled; design the API so adding fog later only
  requires `reveal_bucket` calls and an "unknown" sentinel — no rewrite.
