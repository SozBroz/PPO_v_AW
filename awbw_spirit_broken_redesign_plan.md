# AWBW Spirit Broken Redesign Plan

## Goal

Rearchitect the current `spirit_broken` surrender mechanic so it is fair between Player 1 and Player 2.

The current mechanic checks only at the end of each full day, after Player 2 has completed their turn. This makes the heuristic asymmetric:

```text
P1 advantage is measured after P2 had a chance to respond.
P2 advantage is measured immediately after P2 created or restored it.
```

Because AW-style games naturally oscillate after each player's turn, this means P2 needs a smaller true advantage to meet the same thresholds. P1 needs a much larger advantage because P1's lead must survive P2's entire response before it is measured.

The new mechanic should evaluate advantage at equivalent timing for both seats.

---

## Current Rule

Current trigger:

```text
At end of full day, after P2's turn:
  captured_property_lead >= 2
  unit_count_lead >= 2
  unit_value_lead >= 10%
for 3 days in a row
```

Problems:

```text
1. P2 gets measured immediately after P2 acts.
2. P1 gets measured only after P2 responds.
3. P1 has to hold a lead across the opponent's full turn.
4. P2 only has to create a lead at the end of its own turn.
5. The same numeric thresholds have different effective difficulty by seat.
```

---

## Recommended Architecture

Replace the global end-of-day check with **seat-relative own-end-turn checks**.

New concept:

```text
After P0 ends turn:
  evaluate whether P0 is crushing P1.
  update P0's spirit-pressure streak.

After P1 ends turn:
  evaluate whether P1 is crushing P0.
  update P1's spirit-pressure streak.
```

Trigger spirit broken when either player maintains the required advantage for N of their own end-turn checkpoints.

This compares equivalent tactical timing:

```text
P0 judged after P0 acts.
P1 judged after P1 acts.
```

Instead of:

```text
P0 judged after P1 acts.
P1 judged after P1 acts.
```

---

## New Trigger Rule

Replace:

```text
3 days in a row
```

with:

```text
3 own-turn checkpoints in a row
```

Example:

```text
P0 end turn 5:
  P0 meets threshold -> P0 pressure streak = 1

P1 end turn 5:
  P1 does not meet threshold -> P1 pressure streak = 0

P0 end turn 6:
  P0 meets threshold -> P0 pressure streak = 2

P1 end turn 6:
  P1 does not meet threshold -> P1 pressure streak = 0

P0 end turn 7:
  P0 meets threshold -> P0 pressure streak = 3
  P1 spirit broken
```

---

## Core Design

Use:

```text
1. own-end-turn checks
2. separate pressure streaks per player
3. minimum day gate
4. hard threshold
5. soft hold threshold
6. pressure EMA / hysteresis
7. detailed logging
```

This avoids both problems:

```text
- End-of-day asymmetry.
- One-turn spike surrender.
```

---

## Data Model

Add a persistent spirit state to `GameState` or equivalent game metadata.

```python
from dataclasses import dataclass, field

@dataclass
class SpiritState:
    pressure_streak: list[int] = field(default_factory=lambda: [0, 0])
    pressure_ema: list[float] = field(default_factory=lambda: [0.0, 0.0])
    last_pressure_score: list[float] = field(default_factory=lambda: [0.0, 0.0])

    broken_player: int | None = None
    broken_by: int | None = None
    broken_reason: str | None = None
    broken_day: int | None = None
    broken_turn_actor: int | None = None
```

Add configurable thresholds:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class SpiritConfig:
    enabled: bool = True

    # Prevent early false positives from opening-book/capture-order asymmetry.
    min_day: int = 8

    # Number of own-end-turn checkpoints required.
    required_own_turn_streak: int = 3

    # Hard pass thresholds.
    income_property_lead: int = 2
    unit_count_lead: int = 2
    unit_value_ratio: float = 1.10

    # Optional captured-property threshold.
    captured_property_lead: int | None = None

    # Soft hold thresholds.
    soft_income_property_lead: int = 1
    soft_unit_count_lead: int = 1
    soft_unit_value_ratio: float = 1.06

    # EMA smoothing.
    ema_alpha: float = 0.35
    ema_threshold: float = 1.0

    # Optional score weights.
    property_weight: float = 0.35
    income_property_weight: float = 0.65
    unit_count_weight: float = 0.35
    unit_value_ratio_weight: float = 2.0
    funds_weight: float = 0.00002
```

---

## Advantage Features

At each own-end-turn checkpoint, compute features from the perspective of the player who just ended their turn.

```python
@dataclass(frozen=True)
class SpiritPressureDetails:
    actor: int
    opponent: int
    day: int

    captured_property_lead: int
    income_property_lead: int
    unit_count_lead: int
    unit_value_actor: int
    unit_value_opponent: int
    unit_value_ratio: float
    funds_lead: int

    pressure_score: float
    pressure_ema: float

    hard_pass: bool
    soft_hold: bool
```

Recommended feature semantics:

```text
captured_property_lead:
  actor captured properties - opponent captured properties

income_property_lead:
  actor income-producing properties - opponent income-producing properties

unit_count_lead:
  actor units alive - opponent units alive

unit_value_actor:
  total current value of actor's units

unit_value_opponent:
  total current value of opponent's units

unit_value_ratio:
  unit_value_actor / max(1, unit_value_opponent)

funds_lead:
  actor funds - opponent funds
```

Prefer `income_property_lead` over raw property lead for surrender logic, because income-producing properties better represent strategic collapse.

---

## Pressure Score

Compute a scalar score for smoothing and debugging.

Example:

```python
def compute_pressure_score(details: SpiritPressureDetails, cfg: SpiritConfig) -> float:
    unit_ratio_excess = details.unit_value_ratio - 1.0

    score = 0.0
    score += cfg.property_weight * details.captured_property_lead
    score += cfg.income_property_weight * details.income_property_lead
    score += cfg.unit_count_weight * details.unit_count_lead
    score += cfg.unit_value_ratio_weight * unit_ratio_excess
    score += cfg.funds_weight * details.funds_lead

    return score
```

This score is not the main trigger by itself. It is used for hysteresis and logging.

The hard thresholds remain interpretable.

---

## Evaluation Timing

Call spirit evaluation only when a player actually ends their turn.

Pseudo:

```python
def on_turn_end(state: GameState, actor: int, cfg: SpiritConfig) -> None:
    if not cfg.enabled:
        return

    if state.day < cfg.min_day:
        return

    details = compute_spirit_pressure(state, actor, cfg)

    old_ema = state.spirit.pressure_ema[actor]
    new_ema = (1.0 - cfg.ema_alpha) * old_ema + cfg.ema_alpha * details.pressure_score
    state.spirit.pressure_ema[actor] = new_ema

    details = replace(details, pressure_ema=new_ema)

    if details.hard_pass and new_ema >= cfg.ema_threshold:
        state.spirit.pressure_streak[actor] += 1
    elif details.soft_hold:
        # Hold existing streak; do not increment, do not reset.
        pass
    else:
        state.spirit.pressure_streak[actor] = 0

    if state.spirit.pressure_streak[actor] >= cfg.required_own_turn_streak:
        opponent = 1 - actor
        state.spirit.broken_player = opponent
        state.spirit.broken_by = actor
        state.spirit.broken_reason = "own_turn_pressure_streak"
        state.spirit.broken_day = state.day
        state.spirit.broken_turn_actor = actor
        state.done = True
        state.winner = actor
        state.win_condition = "spirit_broken"
```

---

## Hard Threshold

A hard pass means the current actor is clearly ahead.

Recommended initial hard pass:

```python
hard_pass = (
    details.income_property_lead >= cfg.income_property_lead
    and details.unit_count_lead >= cfg.unit_count_lead
    and details.unit_value_ratio >= cfg.unit_value_ratio
)
```

Optional captured-property threshold:

```python
if cfg.captured_property_lead is not None:
    hard_pass = hard_pass and (
        details.captured_property_lead >= cfg.captured_property_lead
    )
```

Recommended default:

```text
income_property_lead >= 2
unit_count_lead >= 2
unit_value_ratio >= 1.10
```

Do not require raw captured-property lead by default unless logs show false positives.

---

## Soft Hold Threshold

A soft hold prevents one minor oscillation from deleting a two-turn dominance streak.

Recommended soft hold:

```python
soft_hold = (
    details.income_property_lead >= cfg.soft_income_property_lead
    and details.unit_count_lead >= cfg.soft_unit_count_lead
    and details.unit_value_ratio >= cfg.soft_unit_value_ratio
)
```

Recommended default:

```text
income_property_lead >= 1
unit_count_lead >= 1
unit_value_ratio >= 1.06
```

Update behavior:

```text
hard pass:
  streak += 1

soft hold:
  streak unchanged

miss:
  streak = 0
```

This is important because AW games naturally oscillate. A single infantry loss or recapture should not erase overwhelming strategic dominance.

---

## Minimum Day

Add a minimum day gate.

Recommended:

```text
min_day = 8
```

Conservative alternative:

```text
min_day = 10
```

Why:

```text
1. Opening-book routes can create temporary property asymmetries.
2. P1/P2 move order can create natural early tempo swings.
3. Early captures are not always game-ending advantage.
4. Surrender should represent strategic collapse, not capture-order variance.
```

---

## Optional Recovery Logic

Optional but useful: if the acting player strongly recovers, decay the opponent's streak.

Example:

```python
if details.hard_pass:
    opponent = 1 - actor
    state.spirit.pressure_streak[opponent] = max(
        0,
        state.spirit.pressure_streak[opponent] - 1,
    )
```

Do not include this in the first version unless the logs show stale streaks causing weird outcomes.

---

## Why Not Beginning-of-Turn Checks?

Beginning-of-turn checks are also symmetric:

```text
P0 is judged after P1 acts.
P1 is judged after P0 acts.
```

This measures whether advantage survived the opponent response.

Pros:

```text
stricter
less likely to trigger on temporary spikes
```

Cons:

```text
slower
may delay obvious surrender
can under-reward decisive attacks
```

Recommended choice:

```text
Use own-end-turn checks with EMA/hysteresis.
```

This is faster and fairer than full-day checks while still controlling one-turn spikes.

---

## Why Not End-of-Day Checks With Adjusted Thresholds?

You could try different thresholds for P1 and P2:

```text
P1 needs smaller threshold
P2 needs larger threshold
```

Do not do this.

It is brittle because the correct correction depends on:

```text
map
COs
phase of game
opening player
capture route
combat density
turn order
```

Better architecture:

```text
Evaluate both seats at equivalent timing.
```

Do not patch asymmetric timing with asymmetric constants.

---

## Logging

Add per-check telemetry.

Example:

```json
{
  "event": "spirit_check",
  "spirit_check_actor": 0,
  "spirit_day": 12,
  "spirit_captured_property_lead": 3,
  "spirit_income_property_lead": 2,
  "spirit_unit_count_lead": 4,
  "spirit_unit_value_actor": 94000,
  "spirit_unit_value_opponent": 79000,
  "spirit_unit_value_ratio": 1.1899,
  "spirit_funds_lead": 5000,
  "spirit_pressure_score": 2.41,
  "spirit_pressure_ema": 1.72,
  "spirit_hard_pass": true,
  "spirit_soft_hold": true,
  "spirit_streak_p0": 2,
  "spirit_streak_p1": 0,
  "spirit_broken_player": null
}
```

When surrender triggers:

```json
{
  "event": "spirit_broken",
  "win_condition": "spirit_broken",
  "winner": 0,
  "spirit_broken_player": 1,
  "spirit_broken_by": 0,
  "spirit_streak": 3,
  "spirit_day": 13,
  "spirit_reason": "own_turn_pressure_streak"
}
```

Also add summary fields to game logs:

```json
{
  "win_condition": "spirit_broken",
  "spirit_broken_player": 1,
  "spirit_broken_by": 0,
  "spirit_broken_day": 13,
  "spirit_broken_turn_actor": 0,
  "spirit_streak_final_p0": 3,
  "spirit_streak_final_p1": 0
}
```

---

## Tests

Add tests covering fairness, timing, hysteresis, and terminal behavior.

### Test 1 — P1 and P2 same-phase fairness

Construct mirrored states where each player has the same advantage immediately after their own turn.

Expected:

```text
P0 end-turn advantage increments P0 streak.
P1 end-turn advantage increments P1 streak.
Same numeric advantage produces same streak behavior for both seats.
```

### Test 2 — Old end-of-day asymmetry regression

Simulate:

```text
P0 creates threshold lead.
P1 partially erases it.
End-of-day check would fail P0.
```

Expected new behavior:

```text
P0 gets credit at P0 own-end-turn checkpoint.
```

### Test 3 — Soft hold

Sequence:

```text
Turn A: hard pass
Turn B: hard pass
Turn C: soft hold
Turn D: hard pass
```

Expected:

```text
streak progression = 1, 2, 2, 3
spirit broken triggers on Turn D
```

### Test 4 — Miss resets streak

Sequence:

```text
hard pass
hard pass
miss
hard pass
```

Expected:

```text
streak progression = 1, 2, 0, 1
no spirit broken
```

### Test 5 — Minimum day prevents early trigger

Create threshold advantage on day 4.

Expected:

```text
no streak increment if min_day = 8
```

### Test 6 — EMA prevents one-turn spike

Create one massive spike, then miss.

Expected:

```text
no spirit broken
```

### Test 7 — Terminal state

When streak reaches required threshold:

Expected:

```text
state.done == True
state.winner == actor
state.win_condition == "spirit_broken"
state.spirit.broken_player == opponent
```

### Test 8 — Save/load or copy safety

If GameState is copied during MCTS or rollout, spirit state should copy correctly.

Expected:

```text
copy has independent SpiritState
mutation of copy does not mutate original
```

---

## Implementation Steps

### Step 1 — Add config

Add `SpiritConfig` in the engine config area or game config module.

Expose CLI/env config if needed:

```text
--spirit-enabled
--spirit-min-day
--spirit-required-own-turn-streak
--spirit-income-property-lead
--spirit-unit-count-lead
--spirit-unit-value-ratio
--spirit-ema-alpha
--spirit-ema-threshold
```

### Step 2 — Add state

Add `SpiritState` to `GameState`.

Ensure:

```text
copy/deepcopy works
serialization works if needed
reset initializes cleanly
```

### Step 3 — Add feature computation

Implement:

```python
def compute_spirit_pressure(
    state: GameState,
    actor: int,
    cfg: SpiritConfig,
) -> SpiritPressureDetails:
    ...
```

Use existing property/unit-value helpers if they exist. Otherwise add small helpers:

```python
count_income_properties(state, player)
count_captured_properties(state, player)
count_units(state, player)
compute_unit_value(state, player)
```

### Step 4 — Add evaluation hook

Call:

```python
evaluate_spirit_on_turn_end(state, actor, cfg)
```

from the exact place where a player's turn is finalized.

Important: `actor` must be the player who just ended their turn, not the new active player.

### Step 5 — Add terminal resolution

When spirit broken triggers:

```text
winner = actor
loser = 1 - actor
win_condition = "spirit_broken"
done = true
```

Ensure this is treated like other terminal win conditions by:

```text
env reward
episode logging
eval scripts
training logs
MCTS terminal handling
```

### Step 6 — Add telemetry

Log per-check events if verbose logging is enabled.

Always add final summary fields to episode logs.

### Step 7 — Add tests

Implement tests listed above.

### Step 8 — Run A/B diagnostic

Compare old vs new on a fixed seed batch:

```text
old end-of-day spirit broken
new own-end-turn spirit broken
no spirit broken
```

Metrics:

```text
terminal_rate
spirit_broken_rate
winner seat distribution
average day of spirit broken
false positive examples
P0/P1 trigger symmetry
promotion eval winrate
truncation_rate
```

The critical metric is seat bias:

```text
P0 spirit wins vs P1 spirit wins
```

On symmetric maps/CO mirrors, this should not show an unexplained P2 advantage.

---

## Recommended Default Config

Start with:

```python
SpiritConfig(
    enabled=True,
    min_day=8,
    required_own_turn_streak=3,

    income_property_lead=2,
    unit_count_lead=2,
    unit_value_ratio=1.10,

    captured_property_lead=None,

    soft_income_property_lead=1,
    soft_unit_count_lead=1,
    soft_unit_value_ratio=1.06,

    ema_alpha=0.35,
    ema_threshold=1.0,

    property_weight=0.35,
    income_property_weight=0.65,
    unit_count_weight=0.35,
    unit_value_ratio_weight=2.0,
    funds_weight=0.00002,
)
```

If it triggers too early:

```text
raise min_day to 10
raise required_own_turn_streak to 4
raise unit_value_ratio to 1.15
raise ema_threshold
```

If it triggers too rarely:

```text
lower min_day to 7
lower unit_value_ratio to 1.08
remove EMA threshold
use required_own_turn_streak = 2 only for training, not eval
```

---

## Final Recommendation

Replace:

```text
global end-of-day check after P2 turn
```

with:

```text
seat-relative own-end-turn checks
```

Use:

```text
3 own-turn streaks
minimum day
hard threshold + soft hold
EMA/hysteresis
income-producing property lead
unit count lead
unit value ratio
```

This preserves the spirit of the mechanic — early surrender when the game is strategically over — while removing the structural P2 advantage caused by measuring only after P2's turn.
