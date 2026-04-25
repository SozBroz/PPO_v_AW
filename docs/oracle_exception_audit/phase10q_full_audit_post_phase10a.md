# Phase 10Q — Full 741-game `desync_audit` (post Phase 10A / 10B / 10E)

## Purpose

Establish the canonical post–Phase-10A baseline on the full **741** Global League std-tier games (same scope as Phase 9), replacing the 41-game Phase 10L sample. Comparison anchor: **Phase 9 floor** — 627 `ok` / 51 `oracle_gap` / 63 `engine_bug` (`logs/desync_register_post_phase9.jsonl`).

## Methodology

- **Driver:** `tools/desync_audit` (module entrypoint matches `logs/desync_regression_log.md` Phase 5+ convention).
- **Catalog:** `data/amarriner_gl_std_catalog.json` (800 rows in file; **741** games matched to zips after GL std map-pool + CO filters).
- **Map pool:** `data/gl_map_pool.json` (std rotation).
- **Maps directory:** `data/maps`
- **Zips:** `replays/amarriner_gl`
- **Seed:** `1` (`CANONICAL_SEED` — required for deterministic regression gate; see regression log).
- **Output register:** `logs/desync_register_post_phase10q.jsonl`
- **Run log:** `logs/phase10q_audit_run.log`

### Invocation (exact)

```text
python -m tools.desync_audit ^
  --catalog data/amarriner_gl_std_catalog.json ^
  --map-pool data/gl_map_pool.json ^
  --maps-dir data/maps ^
  --zips-dir replays/amarriner_gl ^
  --register logs/desync_register_post_phase10q.jsonl ^
  --seed 1
```

(PowerShell: `*` stream redirection was used with `Tee-Object` to capture `logs/phase10q_audit_run.log`.)

**CLI note:** The tool uses **`--register`** for the output JSONL path, not `--output`. Keyword-only changes elsewhere in the codebase do not apply to this script’s `main()` argparse (see `tools/desync_audit.py`).

### Run metadata (stderr)

From `logs/phase10q_audit_run.log`:

- `total_games=800` (catalog file)
- `zips_matched=741` `filtered_out_by_map_pool=0` `filtered_out_by_co=0`

Wall time (this environment): ~3 minutes.

## Top-line counts (Phase 10Q)

| Class        | Count |
|-------------|------:|
| ok          | **680** |
| oracle_gap| **51** |
| engine_bug| **10** |
| **Total**   | **741** |

### Delta vs Phase 9 floor (627 / 51 / 63)

| Metric       | Phase 9 | Phase 10Q | Δ     |
|-------------|--------:|----------:|------:|
| ok          | 627     | 680       | **+53** |
| oracle_gap  | 51      | 51        | **0**   |
| engine_bug  | 63      | 10        | **−53** |

## Flip table — Phase 9 class → Phase 10Q class

| Phase 9 class | Phase 10Q class | Count |
|---------------|-----------------|------:|
| ok            | ok              | 627 |
| engine_bug    | ok              | 52 |
| oracle_gap    | oracle_gap      | 48 |
| engine_bug    | engine_bug      | 8 |
| engine_bug    | oracle_gap      | 3 |
| oracle_gap    | engine_bug      | 2 |
| oracle_gap    | ok              | 1 |

**Cross-check:** 627+52+48+8+3+2+1 = 741.

### REGRESSIONS (`ok` → `engine_bug` or `ok` → `oracle_gap`)

**Count: 0.**

No game that was `ok` in Phase 9 worsened in Phase 10Q. Escalation threshold (mission: halt if `ok` → worse **> 5**) is **not** triggered.

### Notable non-ok flips (not regressions)

- **`engine_bug` → `ok` (52):** bulk of the Phase 9 `engine_bug` surface cleared by Phase 10 work (10A B-Copter pathing and related lanes per campaign docs).
- **`oracle_gap` → `ok` (1):** `games_id` **1634072** — previously truncated-path / gap; now completes under current oracle + engine.
- **`oracle_gap` → `engine_bug` (2):** **1605367**, **1630794** — first divergence is now `ValueError` on **Load** (`Illegal move … is not reachable`) instead of the earlier `oracle_gap` truncation message. Same underlying drift family; failure classified earlier in the envelope stream.
- **`engine_bug` → `oracle_gap` (3):** **1617442**, **1624764**, **1634965** — first failure is now the Move-truncation `UnsupportedOracleAction` path rather than an `_apply_attack` invariant. Triage label shift, not an `ok` regression.

## Per-class breakdown — remaining `engine_bug` (10 rows)

Approximate buckets from `message` / `approx_action_kind`:

| Bucket | games_id | Notes |
|--------|----------|--------|
| **B_COPTER** — `_apply_attack` range / `unit_pos` mismatch | 1625784, 1635025, 1635846 | Residual air position drift vs AWBW Fire anchor (10A reduced the cohort; these remain). |
| **MECH** — range / `unit_pos` mismatch | 1622104, 1630983 | Non-rotor Bucket-A Fire drift (Phase 10D / 11 family). |
| **Load** — `Illegal move … not reachable` | 1605367, 1630794 | Advanced envelope shape; same two gids as `oracle_gap` → `engine_bug` flips. |
| **BLACK_BOAT** — range message | 1626642 | Naval direct-fire / position drift. |
| **FIGHTER** — large `unit_pos` vs `from` gap | 1631494 | Strong upstream position drift. |
| **Friendly fire** | 1634664 | `_apply_attack: friendly fire from player 0 on INFANTRY …` — distinct from range-drift pattern; needs separate triage. |

## Verdict

**GREEN**

- **`ok` count increased** (627 → 680).
- **`engine_bug` count decreased sharply** (63 → 10).
- **`oracle_gap` unchanged** (51).
- **Zero `ok` → worse flips.**

Phase 10A (and aligned 10B/10E work) **delivers at scale** on the canonical 741-game metric: the post–Phase-9 `engine_bug` backlog is largely cleared without introducing regression-gate violations against the Phase 9 `ok` set.

## Artifacts

| Artifact | Path |
|----------|------|
| Register | `logs/desync_register_post_phase10q.jsonl` |
| Console log | `logs/phase10q_audit_run.log` |
| Phase 9 baseline | `logs/desync_register_post_phase9.jsonl` |
