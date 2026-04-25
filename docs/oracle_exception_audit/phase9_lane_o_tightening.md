# Phase 9 Lane O — Lane K DELETE candidates (diagnostic tightening)

**Campaign:** `desync_purge_engine_harden`  
**Scope:** `tools/oracle_zip_replay.py` only (no `engine/`; Phase 6 Manhattan direct-fire block untouched).

## Summary

| # | Location | Change |
|---|----------|--------|
| 1 | `_oracle_ensure_envelope_seat` | Replaced silent `return` on bad / unmapped `p:` id with `UnsupportedOracleAction` (chain `from e` on int failure). |
| 2 | `Supply` branch (`eng_hint`) | Replaced `except (TypeError, ValueError): pass` with `raise UnsupportedOracleAction(...)` chained from the parse error. |
| 3 | `_guess_unmoved_mover_from_site_unit_name` | On `UnsupportedOracleAction` from `_name_to_unit_type`, re-raise when `raw` is non-empty after strip; keep `return None` only for empty-name paths. |
| 4 | `_resolve_fire_or_seam_attacker` terminal path | Replaced terminal `return None` with **`OracleFireSeamNoAttackerCandidate`** (subclass of `UnsupportedOracleAction`) carrying the Lane K diagnostic string. Call sites that retry the opposite `engine_player` catch this type only so Lane I **pin** errors still propagate as plain `UnsupportedOracleAction`. |

**Rollbacks:** none.

## Edit 1 — `_oracle_ensure_envelope_seat`

**Before:** `int(envelope_awbw_player_id)` failure and `pid not in awbw_to_engine` both `return` without action.

**After:**

```python
    except (TypeError, ValueError) as e:
        raise UnsupportedOracleAction(
            f"envelope seat: bad p: player id not int-convertible: {envelope_awbw_player_id!r}"
        ) from e
    if pid not in awbw_to_engine:
        raise UnsupportedOracleAction(
            f"envelope seat: unmapped p: player id {pid} (awbw_to_engine keys={sorted(awbw_to_engine)!r})"
        )
```

**Rationale:** Silent exit skipped `_oracle_finish_action_if_stale` / `_oracle_advance_turn_until_player` and hid corrupt or unmapped envelope player ids.

## Edit 2 — Supply `eng_hint`

**Before:** `except (TypeError, ValueError): pass` after parsing `envelope_awbw_player_id` for `eng_hint`.

**After:** `raise UnsupportedOracleAction(f"Supply: envelope_awbw_player_id not int-convertible: {envelope_awbw_player_id!r}") from e`

**Rationale:** Invalid ids no longer drop the hint without surfacing why.

## Edit 3 — `_guess_unmoved_mover_from_site_unit_name`

**Before:** `except UnsupportedOracleAction: return None`

**After:**

```python
    except UnsupportedOracleAction:
        if raw is not None and str(raw).strip():
            raise
        return None
```

**Rationale:** Non-empty but unmapped unit names (zip corruption / unknown labels) now fail loud; empty-name path remains soft.

## Edit 4 — `_resolve_fire_or_seam_attacker` exhaustive miss

**Lane I note:** Function was re-read after Phase 8 Lane I; the terminal **`return None` still existed** at the end of the exhaustive-search path (after the `defender_u` / `adj_one` block). **Not obsolete.**

**Before:** `return None` when no attacker could be resolved.

**After:** `raise OracleFireSeamNoAttackerCandidate(...)` with:

`Fire/seam: no attacker candidate for awbw id {awbw_units_id} anchor=(…) target=(…) hp_hint=…`

### Why `OracleFireSeamNoAttackerCandidate` (not only `UnsupportedOracleAction` at the raise site)

Cross-seat fallback (`engine_player` then `1 - eng`) **relied on `None`**. Raising a single exception type would either:

- break the fallback (always fatal on first seat), or
- force a broad `except UnsupportedOracleAction` that would incorrectly swallow **Lane I** pin / upstream-drift raises.

The new subclass is `UnsupportedOracleAction` for all `isinstance` / outer-oracle boundaries but is catchable **narrowly** at the three call sites (`_resolve_attackseam_no_path_attacker` loop, Fire no-path block, Fire nested-Move block).

## Pytest

Command:

`python -m pytest tests/ test_oracle_zip_replay.py --tb=short -q` (tee: `logs/phase9_lane_o_full_pytest.log`)

**Result:** `261 passed`, `2 xfailed`, `3 xpassed` (matches Phase 8 Lane I full sweep count). **No test file changes** were required.

## Sample audit (20 games)

The one-liner in the mission targeted an older `_audit_one(games_id, zip_path, seed)` signature; the harness is keyword-only and needs `meta`, `map_pool`, `maps_dir`.

A corrected sample run against the first 20 catalog `games_id`s (sorted ascending, `seed=1`) produced **no uncaught exceptions** — rows were `ok`, `oracle_gap`, or `loader_error` (missing local zip), not `CRASH` from oracle tight-enings propagating past `apply_oracle_action_json`.

Artifact: `logs/phase9_lane_o_sample_audit.log` (paths abbreviated in messages).

## Artifacts

- `logs/phase9_lane_o_full_pytest.log`
- `logs/phase9_lane_o_sample_audit.log`
- `docs/oracle_exception_audit/phase9_lane_o_tightening.md` (this file)
