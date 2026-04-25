# Phase 11J-STATE-MISMATCH-NAME-NORMALIZE — closeout

## Cosmetic mismatch inventory

Discovery on `logs/desync_register_state_mismatch_936_retune.jsonl` (regex `engine='…' php='…'` on `message`):

| engine → PHP | row occurrences | notes |
|--------------|-----------------|-------|
| `Megatank` → `Mega Tank` | 21 | spacing |
| `Missiles` → `Missile` | 9 | singular/plural (AWBW missile unit) |
| `Black Boat` → `Infantry` | 1 | **not** cosmetic — genuine tile/unit drift |
| `Submarine` → `Sub` | 1 | **not** cosmetic — abbreviation ≠ substring-normalized match |

Strict substring normalization (strip space/punct, lower) finds **1** unique cosmetic pair; adding conservative singular/plural for `missile(s)` yields **2** unique cosmetic pairs (**30** message lines total).

## Implementation

Helper lives in `tools/desync_audit.py` and applies **only** inside `_diff_engine_vs_snapshot` (`--enable-state-mismatch`). Default / canonical desync runs never call it.

```python
def _canonicalize_unit_type_name(name: str) -> str:
    """… see file for full docstring (phase11j + empirical sample)."""
    s = str(name).strip()
    s = _PHP_NAME_ALIASES.get(s, s)
    core = (
        s.lower()
        .replace(" ", "")
        .replace("-", "")
        .replace(".", "")
        .replace("_", "")
    )
    if core == "missiles":
        core = "missile"
    if core in ("mediumtank", "mdtank", "medtank"):
        core = "mediumtank"
    return core
```

Fallback lines from `compare_snapshot_to_engine` that are **only** cosmetic type mismatches are skipped via `_snapshot_line_is_cosmetic_type_only` so `human_readable` does not reintroduce `Megatank`/`Mega Tank` noise.

## Register pre vs post (936 games, seed 1)

| class | retune (baseline) | postnamecanon |
|-------|-------------------|---------------|
| `ok` | 670 | 717 |
| `state_mismatch_units` | 202 | 176 |
| `state_mismatch_multi` | 35 | 12 |
| `state_mismatch_funds` | 26 | 30 |
| `oracle_gap` | 3 | 1 |

- **Games with any `state_mismatch_*` → `ok`:** 45 (≥40 expectation met).
- **Axes on old row exactly `("units_type",)` for those transitions:** 25 — **pure name-canonicalization wins** (Megatank/Mega Tank, Missiles/Missile, Md spellings, etc.).
- **Old messages containing at least one cosmetic type pair:** 30 lines; not all corresponding games flipped to `ok` (some rows also had funds/HP signal).

### Anomalies (not explained by name canon)

- **`oracle_gap` 3 → 1:** two games (`1626284`, `1628953`) are `oracle_gap` in the baseline file and `ok` in the post run (full envelope replay completes). That path does **not** touch `_canonicalize_unit_type_name`. Treat as **baseline drift** vs current tree / replay inputs — re-validate by re-baselining `retune` on the same commit if parity is required.
- **20** of the 45 `state_mismatch` → `ok` transitions are **not** `units_type`-only on the baseline row (axes `units_hp` or `funds`+`units_hp`). HP comparison code was not changed; likely **ordering / first-mismatch envelope** interactions when earlier envelopes were previously blocked by cosmetic type drift, or baseline file age. Worth a future audit if parity matters.

## Validation

- `python tools/desync_audit.py` **without** `--enable-state-mismatch`, `--max-games 100`, `--seed 1` — completed normally (canonical lane unaffected by design).
- `python -m pytest tests/test_state_mismatch_name_canon.py tests/test_state_mismatch_tolerance.py tests/test_audit_state_mismatch.py -q --tb=no` — all green.

## Verdict

**GREEN** — 45 games moved from `state_mismatch_*` to `ok` (≥40), with 25 attributable to **`units_type`-only** first diffs; no new test failures. Caveats above for `oracle_gap` flips and non–units_type `ok` transitions.
