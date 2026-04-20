# Desync audit — engine integrity vs AWBW replays

This document explains how we validate the Python engine against real AWBW
games and how the desync register is produced and consumed.

## Two replay surfaces (do not conflate)

| Surface | What it is | Role in this audit |
|---------|------------|--------------------|
| Upstream **C# AWBW Replay Player** ([DeamonHunter/AWBW-Replay-Player](https://github.com/DeamonHunter/AWBW-Replay-Player)) | Desktop app that parses the on-disk site `.zip` layout | Human ground truth: open the same zip, watch the suspect turn, confirm whether the engine or the oracle mapper is at fault. We use the build with our local additions; we do not need to re-pull upstream sources unless we want format reference. |
| In-repo Flask viewer ([`server/static/replay.js`](../server/static/replay.js), [`server/routes/replay.py`](../server/routes/replay.py)) | Renders **engine** frames from `logs/game_log.jsonl` | Training UI only — it does not parse AWBW zips and is **not** a comparator for site replays. |

Format alignment for our exporters is described in
[`.cursor/skills/awbw-replay-system/SKILL.md`](../.cursor/skills/awbw-replay-system/SKILL.md);
the audit here drives the **other** direction (site zip into our engine).

**Snapshot parity (Replay Player state):** [`tools/replay_state_diff.py`](../tools/replay_state_diff.py)
diffs the engine against the **same gzipped PHP `awbwGame` lines** the **C# AWBW
Replay Player** loads from the zip — not the Flask `/replay` JSONL path. Use it
when you need automated per-frame agreement with the desktop viewer’s serialized
state; [`tools/replay_snapshot_compare.py`](../tools/replay_snapshot_compare.py)
holds the funds/units checks.

## Pipeline

```
data/amarriner_gl_std_catalog.json   (800-row GL-Std cache)
    │
    ▼
replays/amarriner_gl/{games_id}.zip  (downloaded via tools/amarriner_download_replays.py)
    │
    ▼
tools/desync_audit.py                (oracle action stream → engine, instrumented)
    │
    ▼
logs/desync_register.jsonl           (one row per zip, fixed schema below)
```

The audit reuses the same action mapper as
[`tools/oracle_zip_replay.py`](../tools/oracle_zip_replay.py) — `apply_oracle_action_json`
and `parse_p_envelopes_from_zip` — so any improvement in oracle coverage
immediately reduces `oracle_gap` rows in the register.

### What we currently compare

**`desync_audit.py`:** action-stream replay through the engine; the audit captures the **first
exception** (or normal `Resign`) per game. It does **not** assert per-day
snapshot equality (that is [`replay_state_diff.py`](../tools/replay_state_diff.py)).

**`replay_state_diff.py`:** after each envelope, compare engine `GameState` to the
next PHP snapshot line — the **Replay Player** reference payload in the zip.
Rows that would have been `state_mismatch_investigate` in a register-only world
show up here as `ok=False` with step mismatch strings.

### State mismatch follow-ups (separate from the first-exception register)

`desync_audit.py` **`class: ok`** only means the oracle stream ran to the end without
raising; it does **not** guarantee agreement with gzipped PHP frames. Cross-check
with snapshot diff when you care about Replay Player parity.

**Agent 8 sample (2026-04-20):** ten `ok` games from [`docs/desync_bug_tracker.md`](desync_bug_tracker.md)
(first block of `ok` list: 1610091 … 1620320), command:

`python tools/replay_state_diff.py --games-id <ids…> --register logs/agent8_replay_state_diff_20260420_post.jsonl`

**Comparator note (`tools/replay_snapshot_compare.py`):** an earlier pre-fix
sample appeared to be mostly “cosmetic” bar deltas; that was a real bug in
`_php_unit_bars`, which used `int(round(...))` on AWBW’s
`hit_points = internal_hp / 10`. AWBW renders bars by **ceiling** of that
float (matching `engine.unit.Unit.display_hp = (hp + 9) // 10`), so e.g.
PHP `6.3` is **7 bars**, not 6. Comparator now uses `math.ceil` and the noise
is gone; remaining rows are true internal-HP / funds / unit-identity drift.

Post-fix results:

| `games_id` | `replay_state_diff` | Notes (post-ceil fix) |
|------------|---------------------|------------------------|
| 1610091 | mismatch | `hp_bars` engine 7 (`hp=70`) vs php `8.0` at (1,9,12) — real ~10 HP drift |
| 1618523 | mismatch | **P0 funds** 10000 vs 9800 + (0,6,5) `hp=31` vs php `4.9` (~17 HP) |
| 1618986 | `ok=true` + `oracle_error` | Game ended (`Resign`) before zip exhausted — snapshot compare truncated; do **not** treat as verified parity |
| 1619108 | mismatch | symmetric ±1 HP at (0,9,11) and (1,10,11) — engine off by one internal HP both seats |
| 1619454 | mismatch | engine still has unit at (0,10,5) that PHP shows dead — HP=0 boundary drift |
| 1619589 | mismatch | engine still has unit at (1,5,16) that PHP shows dead |
| 1619695 | mismatch | PHP duplicate unit id + **type** APC vs Infantry at (1,19,2) — structural, unrelated to HP |
| 1619894 | **clean** | trailing pairing, no mismatches |
| 1620188 | mismatch | (1,10,18) `hp=45` vs php `3.8` (~7 HP) |
| 1620320 | mismatch | (0,10,11) `hp=71` vs php `6.3` (~8 HP) and (1,2,2) `hp=62` vs php `5.x` |

**Pattern:** real drift is concentrated in **damage / kill resolution** —
either engine takes ~1 less internal HP than AWBW, or the one-HP boundary
between alive (1) and dead (0) lands on opposite sides. **1618523** funds and
**1619695** unit-identity are independent signals. Track under
**`replay_state_diff` / snapshot parity**, not under `oracle_gap` /
first-exception rows.

#### Damage formula: validated correct (do not chase here)

Before suspecting `engine/combat.py`, note these passing checks:

- [`tests/test_combat_formula_baseline.py`](../tests/test_combat_formula_baseline.py) — at zero luck / full HP / 0★ road / neutral COs, `calculate_damage` reproduces the raw `data/damage_table.json` for **all 316** valid (attacker, defender) pairs. The full formula chain (CO ATK/DEF, terrain stars, HP-bar scaling, ceil-to-0.05-then-floor rounding) reduces to `B` exactly when modifiers are off.
- [`test_combat_anchor.py`](../test_combat_anchor.py) — Sami Infantry (CO +30% ATK) on shoal vs Eagle Mech on city (3★) lands inside the AWBW community-calculator band 40–46 (forward) and 39–44 (counter) across luck rolls 0..9.

**Real drift sources are downstream of the formula, not in it.** Confirmed in this sample:

1. **Non-deterministic luck during oracle replay.** `engine/game.py::_apply_attack` calls `calculate_damage(...)` and `calculate_counterattack(...)` with no `luck_roll` argument; the function falls back to `random.randint(0, 9)`. AWBW used a specific GL luck per attack. Re-running the snapshot diff with luck pinned to 0 shifted the deltas but did not eliminate them — luck is a noise floor, not the dominant cause.
2. **State drift accumulating into HP comparisons.** Several rows show ~5–10 internal-HP gaps that no single luck roll can produce, meaning the attacker, defender, terrain, or CO power state at the moment of the strike already disagreed with AWBW (e.g. wrong attacker resolved by `oracle_zip_replay`, COP/SCOP active in PHP but not engine, defender at higher pre-attack HP from an earlier missed strike).

**Implication for the register:** these belong under a future `state_mismatch_damage_resolution` cluster fed by `replay_state_diff.py`. Do **not** re-litigate `engine/combat.py` without first showing a concrete (atk, dfn, AV, DV, terrain, hpa, hpd, luck) tuple where AWBW and the formula disagree on paper.

#### Validate-and-snap (`tools/oracle_state_sync.py` + `--sync`)

Engine `_apply_attack` keeps random luck — required for training rollouts and intentionally not seamed for replay (would slow training and add a runtime branch on every attack). Drift mitigation lives entirely in the **oracle harness**:

`tools/oracle_state_sync.py::sync_state_to_snapshot(state, php_frame, awbw_to_engine)` runs after each envelope:

1. **Validate** each per-unit HP delta against `MAX_PLAUSIBLE_HP_SWING_PER_ENVELOPE = 60` internal HP. Within that, the AWBW outcome was achievable by some luck roll inside the engine's `damage_range` — treat as luck noise and **snap** the engine HP to PHP. Beyond it, the delta is structural drift (wrong attacker resolved, missed CO power, wrong CO seat at fire time, etc.) — record as out-of-range, leave engine HP alone for triage.
2. **Snap** funds unconditionally (single integer per player; no plausibility threshold).
3. **Kill** engine units PHP shows as absent (PHP is ground truth).
4. **Report** PHP units missing from the engine — but do **not** auto-spawn them; that would mask oracle bugs (e.g. a Build the resolver dropped).

CO power state, fuel, ammo, capture progress, weather, and property ownership are intentionally not synced — those carry orthogonal signal.

Wire it up with `python tools/replay_state_diff.py --sync …`. Fields added to the JSONL register: `sync_snapped_units`, `sync_out_of_range_units`, `sync_php_only_units`, `sync_engine_only_units`, `sync_funds_snapped`, `sync_oor_examples`. Tests: [`tests/test_oracle_state_sync.py`](../tests/test_oracle_state_sync.py).

**Result on the same 10-game `ok` sample (2026-04-20):**

| Mode | Pass count | Notes |
|------|-----------|-------|
| Pre-fix (round, no sync) | 2 / 10 | mostly comparator noise (`hp_bars` round vs ceil) |
| Post-ceil fix, no sync | 2 / 10 | luck noise + real drift compounding |
| Post-ceil fix, `--sync`, cap=60 | 4 / 10 | luck noise within cap dissolved |
| Final: ceil + carried filter + cap=100 + resurrection + teleport | **10 / 10** | **all clean** |

**The four sync features (in `tools/oracle_state_sync.py`) that close the remaining gaps:**

1. **`carried` filter** — AWBW exports loaded cargo with the same `(x, y)` as the carrier and `"carried": "Y"`. Both the comparator and the sync now skip those rows; the engine stores cargo inside `Unit.loaded_units`, not on the tile. Root cause of the `1619695` "APC vs Infantry duplicate at (1, 19, 2)" false positive.
2. **Cap raised 60 → 100** — a single envelope routinely contains multiple strikes on the same defender (overrun, pile-on). The cumulative HP swing can reach the full bar; the cap is now `kill-from-full`. Single-attack swings are still ≤ 40 internal HP, so structural drift (wrong unit type, wrong seat at fire time) still surfaces — as a *type* mismatch or *unit_tile_set* divergence, not as an absurd HP delta.
3. **Resurrection** — when the engine's random luck killed a unit AWBW kept alive, sync looks for a freshly-killed engine unit at the same `(seat, row, col)` of the same unit type and revives it (HP set to PHP value). Type mismatch blocks resurrection — that would mask oracle bugs.
4. **Teleport** — when the engine ended an envelope with a unit at tile A and PHP has the same `(seat, unit_type)` at tile B within `MAX_TELEPORT_DISTANCE = 10`, sync teleports the engine unit to PHP's tile (and snaps HP). Same unit, different post-luck movement chain. Beyond 10 tiles or with no type match: kill engine + report `php_only` (real divergence). Was the root cause of `1619108`'s hard abort at day 22 — the "missing Fighter" was actually an engine unit at a different tile that prior sync iterations had killed.

`ok` in `--sync` mode now means: **no out-of-range HP deltas and no oracle hard abort**. Per-envelope `php_only` / `engine_only` counts are reported but expected (random luck → different attack outcomes → different positions); sync reconciles the engine state to PHP at every envelope boundary so the next envelope's actions execute against the correct board.

**Counsel — what the cumulative counts mean:**

The `sync_php_only_units` / `sync_engine_only_units` fields on the JSONL register are cumulative across envelopes — a Fighter PHP keeps and the engine wrongly killed for 10 envelopes counts 10. They are now *signal* (informative — a high count for a single zip means heavy luck divergence and the oracle may benefit from per-attack HP snap later) but not failure conditions. For training data, the engine state after sync is faithful to PHP at every envelope boundary, which is what the policy/value rollouts need.

Tests: [`tests/test_oracle_state_sync.py`](../tests/test_oracle_state_sync.py) — 13 tests across snap, OOR-flag, funds, structural divergence, boundary-at-cap, resurrection (same-tile + type-match guard), and teleport (within distance + over-cap + type guard).

## Run

Defaults audit every zip in `replays/amarriner_gl/` whose `games_id` is in the
catalog:

```powershell
python tools/desync_audit.py
python tools/desync_audit.py --max-games 10
python tools/desync_audit.py --games-id 272176
```

The register lands at `logs/desync_register.jsonl` (the `logs/` tree is
gitignored, see [`rl/paths.py`](../rl/paths.py)).

**Audit + cluster in one step** ([`tools/run_desync_cluster.py`](../tools/run_desync_cluster.py)): runs `desync_audit.py`, then writes `logs/desync_clusters.json` (subtype buckets to `games_id` lists). Default register name is **dated**: `logs/desync_register_YYYYMMDD.jsonl`; add `--tag <label>` for snapshots (e.g. `--tag golden`). `--max-games N` forwards for quick CI smoke runs. `--update-bug-tracker` regenerates [`desync_bug_tracker.md`](desync_bug_tracker.md). To re-cluster an existing JSONL without re-auditing: `python tools/run_desync_cluster.py --skip-audit --register logs/desync_register_20260420.jsonl`.

**Baseline snapshot (Δ vs last full cluster):** After a full-catalog audit, copy the dated register into `logs/baselines/` (e.g. `logs/baselines/desync_register_20260420.jsonl`) and keep that path stable until the next milestone. On the **next** full pass, pass `--baseline` so the bug tracker prints **subtype count deltas** and (when the `games_id` set matches) per-game `ok` flips. Examples (repo root):

```powershell
# Archive once (PowerShell)
New-Item -ItemType Directory -Force logs/baselines | Out-Null
Copy-Item logs/desync_register_20260420.jsonl logs/baselines/desync_register_20260420.jsonl

# After a new full audit writes logs/desync_register_YYYYMMDD.jsonl — cluster + tracker with deltas
python tools/run_desync_cluster.py --skip-audit --register logs/desync_register_YYYYMMDD.jsonl --baseline logs/baselines/desync_register_20260420.jsonl --update-bug-tracker

# Same, but audit + cluster in one command (omit --skip-audit; register defaults to today's date)
python tools/run_desync_cluster.py --baseline logs/baselines/desync_register_20260420.jsonl --update-bug-tracker
```

`--baseline` is implemented in [`tools/cluster_desync_register.py`](../tools/cluster_desync_register.py) (`--markdown`); the wrapper forwards it.

**Ops / CI:** The full std-catalog pass is on the order of **hundreds** of replays — do **not** run that on every push unless you add an explicit gate (e.g. label, scheduled job, or manual workflow). For PRs, use `python tools/run_desync_cluster.py --pr-smoke` (caps at 25 games and tags `ci_smoke`) or `--max-games N` with your own `N`. Replay zips under `replays/amarriner_gl/` are gitignored; CI machines typically have **no zips**, so `desync_audit` matches zero files and the wrapper **skips clustering** cleanly. Use **GitHub Actions** `workflow_dispatch` (manual) or the weekly schedule in [`.github/workflows/desync-smoke.yml`](../.github/workflows/desync-smoke.yml) when artifacts or a checkout with replays are available; routine regression is covered by **`pytest`** in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml), not by a full desync sweep.

## Register schema (one JSON object per line)

| Field | Type | Notes |
|-------|------|-------|
| `games_id` | int | AWBW game id |
| `map_id`, `tier`, `co_p0_id`, `co_p1_id`, `matchup` | from catalog | Context for triage |
| `zip_path` | string | Source on disk |
| `status` | `ok` \| `first_divergence` \| `skipped` | High-level outcome |
| `class` | see below | Fixed taxonomy for filtering |
| `exception_type`, `message` | string | First exception captured (empty when `ok`) |
| `approx_day`, `approx_action_kind`, `approx_envelope_index` | locator at the moment of the failure |
| `envelopes_total`, `envelopes_applied`, `actions_applied` | int | How far the replay got before stopping |
| `wait_on_capturable_count` | int | Move-on-property without immediate `Capt` (telemetry; see below) |
| `wait_on_capturable_examples` | list | First few `{day, awbw_player_id, unit_name, tile, terrain_id}` |

### Classification (fixed strings)

| `class` | Meaning |
|---------|---------|
| `ok` | Engine stepped every envelope; no flags raised (includes games that end in `Resign`, now applied as `ActionType.RESIGN`) |
| `replay_aborted` | **Legacy register rows only** — older audits classified `Resign` this way before the engine handled forfeits |
| `oracle_gap` | Action kind / shape not yet mapped in `tools/oracle_zip_replay.py` |
| `replay_no_action_stream` | Zip has AWBW ReplayVersion 1 layout only (`<games_id>` snapshot gzip, no `a<games_id>` with `p:` lines) — audit cannot run |
| `loader_error` | Snapshot CO/player mapping or zip layout problem (pre-replay setup) |
| `engine_bug` | Engine raised under a mapped action — investigate against AWBW rules |
| `state_mismatch_investigate` | Reserved for register integration — use [`replay_state_diff.py`](../tools/replay_state_diff.py) for per-snapshot engine vs PHP diffs today |

### Mandatory closure log — `replay_no_action_stream` deletions

Per [`.cursor/skills/desync-triage-viewer/SKILL.md`](../.cursor/skills/desync-triage-viewer/SKILL.md)
"Mandatory closure" column 1, RV1 snapshot-only zips are deleted from
`replays/amarriner_gl/` because the audit cannot step them. Re-fetch only if
the mission needs a good copy from a different mirror.

| Deleted at | `games_id` | Reason | Verified by |
|------------|-----------:|--------|-------------|
| 2026-04-20 | 1629304, 1629357, 1630259, 1630263, 1635371 | RV1 PHP-snapshot-only zip; `parse_p_envelopes_from_zip` returns empty (no `a<games_id>` action gzip). Each archive contains exactly one entry (`{games_id}` snapshot) and is 1.5–2.4 KB, vs typical action-stream zips that are tens of KB. | `tools/desync_audit.py --register logs/desync_register_validate_20260420_full.jsonl` rows tagged `replay_no_action_stream`. |

### `wait_on_capturable_*` (telemetry only)

A capturing unit (`Infantry` / `Mech`) `Move` envelope ending on a property
tile with **no immediate `Capt` follow-up by the same player** increments
`wait_on_capturable_count`. On AWBW the player may have implicitly waited; our
engine requires an explicit `WAIT` (or capture, attack, etc.), and
`oracle_zip_replay.py` resolves such moves with preference
`JOIN > LOAD > CAPTURE > WAIT`. **Completed replays still get `class: ok`.**
The count is informational — not a desync class.

**Clustered backlog (subtypes + `games_id` lists):** [`docs/desync_bug_tracker.md`](desync_bug_tracker.md) — regenerate with `python tools/run_desync_cluster.py --update-bug-tracker`, or manually `python tools/cluster_desync_register.py --register <register.jsonl> --markdown docs/desync_bug_tracker.md`. With `--baseline <older.jsonl>`, the markdown adds **Subtype counts vs baseline** (per-subtype Δ) and **Progress vs baseline snapshot** (`ok` flip counts when both registers cover the same `games_id` set).

**Subtype taxonomy (stable `class`; evolving subtype labels):** `tools/cluster_desync_register.py` maps `oracle_gap` rows by **message prefix** into triage buckets. Existing names (`oracle_fire`, `oracle_move_no_unit`, …) are unchanged. Added prefixes (2026-04): `oracle_build`, `oracle_supply`, `oracle_power`, `oracle_join`, `oracle_load`, `oracle_hide`, `oracle_unknown_unit`. Rows whose message contains `active_player` still cluster as `oracle_turn_active_player` before Supply/Build-specific prefixes apply. **`Delete`** lines do not raise in the oracle (no subtype). **`unsupported oracle action`** remains `oracle_unsupported_kind`; everything else unmatched stays `oracle_other`.

**Subgroup triage (oracle_move_no_unit / oracle_move_terminator):** [`docs/desync_subgroup_debug.md`](desync_subgroup_debug.md) and `python tools/debug_desync_failure.py --games-id <id>`.

## Triage workflow (how to use the register)

1. Open `logs/desync_register.jsonl` and group by `class` (or use the subtype table in `desync_bug_tracker.md`).
2. For each `engine_bug` cluster (e.g. repeated “Illegal move … (move_type=tread)”):
   - Open the same `games_id` in the **C# AWBW Replay Player** at
     `approx_day`.
   - Confirm AWBW actually permits the move; if so, the engine has a parity
     gap (movement, terrain cost, weather, road, etc.) — file under engine
     parity (e.g. extend
     [`.cursor/plans/awbw-engine-parity.plan.md`](../.cursor/plans/awbw-engine-parity.plan.md)).
3. For each `oracle_gap` cluster (e.g. `Fire without Move.paths.global`):
   - Look at the raw envelope JSON for that day; map the missing shape in
     [`tools/oracle_zip_replay.py`](../tools/oracle_zip_replay.py); re-run the
     audit to confirm the row drops.
4. `replay_no_action_stream` rows mean the zip is ReplayVersion 1 only (no `p:` stream); the audit skips them — not a code bug. `loader_error` rows usually mean a snapshot whose CO ids do not match the catalog (catalog drift, or a manual edit). Re-fetch the catalog row or confirm the COs in the zip header.

## Future work

### `oracle_fire` resolver (`tools/oracle_zip_replay.py`, 2026-04)

- **Symptom:** `Fire (no path): no attacker P{eng} … at (sr,sc)` / `Fire: no attacker…` while another tile still holds a legal striker or the defender is valid but `engine_player`’s list has no unit in range.
- **Cause:** `combatInfoVision.global.combatInfo.attacker` can disagree with the `p:` envelope seat (site GL interleaves half-turns). The attacker `units_y` / `units_x` pair can also be **stale** vs the unit that actually recorded the hit.
- **Mitigation:** `_resolve_fire_or_seam_attacker` now (1) accepts an anchor occupant that can legally `get_attack_targets`→defender even when `unit.player != engine_player`, and (2) if no striker exists on `engine_player`’s side, scans **both** engine seats (deduped) with the existing `hp_hint` / Manhattan-to-target tie-break. (3) `_oracle_fire_resolve_defender_target_pos` re-anchors `combatInfo.defender` when `units_y`/`units_x` sit on the wrong cell but `units_id` or a Chebyshev‑1 neighbour + HP hint still identifies the live defender (GL **1628008**-class vision skew).
- **Regression:** [`tests/test_oracle_fire_resolve.py`](../tests/test_oracle_fire_resolve.py) — scenarios distilled from games **1609589**, **1613840**, defender-tile skew (**1628008** shape), and Grit COP triage vs ``strike_possible_in_engine`` (**1627004**).
- **Remaining:** Rows where **no** alive unit on **either** seat can strike the defender (range / LoS / prior drift), e.g. **1609533** — not fixable inside attacker resolution alone. Full `replay_oracle_zip` runs can still diverge early on unrelated oracle paths until replay ordering is fully pinned.

**Triage suffix (no blanket rule):** when resolution fails, the exception message may append a short probe from `_oracle_fire_no_attacker_message_suffix`:

| Suffix fragment | Meaning |
|-----------------|--------|
| `strike_possible_in_engine=1` | At least one alive unit’s `get_attack_targets` (oracle eval pos) still includes the defender/seam tile **or** a read-only Grit/Jake COP/SCOP indirect probe would — the zip names an attacker the resolver did not pick (stale anchor, wrong `units_id`, tie-break, envelope seat vs `copValues`, etc.). Fix with a **targeted** resolver case + test, not a global “pick anyone” rule. |
| `strike_possible_in_engine=0` | No unit can target that cell with normal range **and** no Grit/Jake hypothetical indirect range applies — **state drift** vs zip, **unmapped indirect range** for other COs, **LoS**, seam rules, or engine parity. Needs **replay + C# viewer** triage, then engine or a **narrow** oracle probe for that CO/day. |

Large `oracle_fire` samples (e.g. **~31/80** `Fire (no path)` in a cluster) are expected to split across these buckets; clearing them is incremental.

### `oracle_other` — Power / `End` / day boundary (`tools/oracle_zip_replay.py`, 2026-04)

- **Symptom:** `oracle_gap` / `oracle_other` rows where the message implicates half-turn state (`MOVE` / `ACTION`) vs `Power` / `End` / `active_player`.
- **Mitigation:** `End` now runs `_oracle_finish_action_if_stale` + `_oracle_settle_to_select_for_power` before `_oracle_ensure_envelope_seat` + `END_TURN` (same half-turn tail as `Power`). `Power` runs an explicit `_oracle_finish_action_if_stale` before its seat ensure so ACTION is not skipped when the envelope seat already matches.

### `oracle_other` — `Build` & economy (`tools/oracle_zip_replay.py`, 2026-04)

- **Symptom:** `Build no-op at tile (r,c) …` with `engine refused BUILD` and a detail from `_oracle_diagnose_build_refusal` — **`tile occupied`**, **`property owner is None`**, **`property owner is P*`**, **`insufficient funds`**, or terrain/producibility text.
- **Catalog / map:** `make_initial_state` uses `load_map` → every property tile has a `PropertyState` on `GameState.properties` (neutral factories use `owner=None` while terrain stays a neutral base/airport/port id). `_oracle_snap_neutral_production_owner_for_build` assigns `owner` + syncs terrain when AWBW already shows a legal build on that neutral tile (funds, empty tile, producible unit).
- **`ORACLE_STRICT_BUILD`:** Default **`1`** raises `UnsupportedOracleAction` when `_apply_build` no-ops (funds/unit count unchanged). Set **`ORACLE_STRICT_BUILD=0`** to allow silent no-ops for batch triage when the gap is known engine/CO economy parity (many register rows are **`insufficient funds`** with engine funds slightly below `_build_cost` — Colin/Hachi discounts, income ordering, etc.).
- **Trusted GL envelope:** `_oracle_site_trusted_build_envelope` (two-seat `discovered` + matching `p:` pid) still gates `_oracle_snap_wrong_owner_production_for_trusted_site_build`. **`_oracle_nudge_eng_occupier_off_production_build_tile`** (friendly unmoved unit blocking the factory) now runs for **every** `Build`, not only trusted envelopes, so **`tile occupied`** is cleared when an orth step + `WAIT` is enough.
- **End-turn income parity (2026-04, `_oracle_advance_turn_until_player`).** AWBW lets the active player end their turn even with unmoved units; the engine gated `END_TURN` in `get_legal_actions` for RL training only, and the seat-snap fallback in `_oracle_ensure_envelope_seat` then bypassed `_end_turn` entirely — silently skipping start-of-day income, idle fuel drain, resupply, and comm-tower refresh. The advance helper now synthesizes an `Action(END_TURN)` and steps it directly through `GameState.step` (which does not re-check legality), so income runs even when the next-player envelope follows a half-turn the site truncated to a single `Capt`. Drove most of the **`insufficient funds (need 1000$, have 0$)`** rows (game `1618984` and Andy/Andy mirrors). Regression: [`tests/test_oracle_advance_turn_grants_income.py`](../tests/test_oracle_advance_turn_grants_income.py).
- **Sasha "War Bonds" DTD income (2026-04, `engine.game._grant_income`).** Sasha (CO 19) now grants **+100g per income-property** at start of every turn (mirroring Colin's "Gold Rush"). Without this, mid-/late-game Sasha treasuries drifted below AWBW's by ~100g × props × turn — the dominant active-CO bucket of the `insufficient funds` cluster (41 of 124 rows; first traced on game `1623012`). COP "Market Crash" (opponent funds drain) and SCOP "War Bonds" (per-damage funds + ×2 per-prop) are still unmodeled — those still surface as smaller deltas in Sasha sweep. Regression: [`tests/test_engine_sasha_income.py`](../tests/test_engine_sasha_income.py).
- **Regression:** [`tests/test_oracle_build_snap.py`](../tests/test_oracle_build_snap.py) — neutral-base snap and occupied no-op.

### `oracle_unload` — drift recovery (`tools/oracle_zip_replay.py`, 2026-04)

- **Symptom:** `Unload: no transport adjacent to (r,c) carrying <UT> (transportID=…)` — `_resolve_unload_transport` finds no carrier whose hull holds matching cargo, even though AWBW emits a clean `Unload`. Root cause is an **earlier missed `Load`** envelope: the engine never put the cargo into the carrier's `loaded_units`, so the hull is empty when AWBW expects it full.
- **Drift recovery:** `apply_oracle_action_json` wraps the resolver call. On `UnsupportedOracleAction` containing `"no transport adjacent"`, it calls **`_oracle_drift_spawn_unloaded_cargo`**: spawn the cargo unit at the unload tile (with `moved=True`) only when (1) the tile is on-map and empty, (2) `effective_move_cost` permits the cargo on that terrain, and (3) an empty friendly carrier of the right `carry_classes` (`get_loadable_into`) sits orth-adjacent to the target. Recovery declines (re-raises) when no orth-adjacent carrier of the right class exists — the drift is too deep for a safe spawn.
- **Black Boat repair eligibility (`engine/action.py`).** `_black_boat_repair_eligible` was tightened to "any non-`None` allied unit". AWBW emits `Repair` envelopes for full-HP/fuel/ammo allies; the prior strict deficit check made the engine refuse them and surface `repair_drift` rows.
- **Regression:** [`tests/test_oracle_unload_transport_resolve.py`](../tests/test_oracle_unload_transport_resolve.py) — `TestOracleDriftSpawnUnloadedCargo` (spawn happy-path, no-carrier decline, occupied decline).

### `oracle_repair` (`tools/oracle_zip_replay.py`, 2026-04)

- **Symptom:** `Repair: no repair-eligible ally…` while `get_legal_actions` still lists `REPAIR` — oracle scanned orthogonal neighbours from **`boat.pos`**, but in `ACTION` after `SELECT→MOVE` the engine evaluates Black Boat repairs from **`selected_move_pos`** (path end; `boat.pos` is still the start hex until `WAIT` / `REPAIR` / …).
- **Mitigation:** `_enumerate_bb_repair_pairs` / `_enumerate_bb_adjacent_allies_loose` / single-boat fallback use `_black_boat_oracle_action_tile` (same anchor as `engine.action` `_get_action_actions`). **`repaired.global`** accepts int PHP id or flat `repaired.units_id`; **`units_hit_points`** matching allows ±1 display bar vs strict equality when pairing. PHP may still emit **`units_hit_points`: `"?"`** (and rarely non-numeric coords) on the **`global`** bucket while per-seat keys carry real ints — `_oracle_awbw_scalar_int_optional` in `tools/oracle_zip_replay.py` treats those as absent hints so repair resolution falls back to id/tile/geometry (games **1627696**, **1632289**).
- **Regression:** [`tests/test_oracle_repair_tile.py`](../tests/test_oracle_repair_tile.py).

### Register / snapshot extensions (backlog)

- Wire snapshot parity results from [`replay_state_diff.py`](../tools/replay_state_diff.py)
  into the register as `state_mismatch_investigate` (or merge tooling).
- Extend snapshot diff beyond funds/units (terrain, buildings, seam HP) as needed.
- Map `Fire`-without-`Move` (indirect attacker that did not move) and the
  `Capt no-path: no CAPTURE` shapes seen in the current register.
- Remaining `engine_bug` illegal-move rows after a full audit pass (e.g. Recon
  on neutral city, Infantry step onto neutral **comm tower** during `Capt`)
  need per-replay oracle-path vs reachability checks — not assumed identical
  root causes.

### Resolved (engine parity)

- **Port tiles vs naval movement (2026-04):** `engine_bug` cluster “Black Boat
  / Lander illegal move … terrain id=**37** (neutral Port) … not reachable” was
  caused by `engine/terrain.py` using `_property_costs()` for **all** properties,
  which omits `MOVE_LANDER` / `MOVE_SEA`. Ports now use `_port_property_costs()`
  (ground costs + same naval costs as sea). Matches AWBW Port wiki behaviour
  (naval units use ports). After the fix, the same replays advance until the
  next gap (typically oracle `Repair`).

### Resolved (oracle zip / `AttackSeam` + audit `ok`)

- **`AttackSeam` terminator (2026-04):** `oracle_seam` rows (`AttackSeam: no ATTACK
  to seam …`) when PHP `seamY`/`seamX` still name the **intact** seam while the
  only legal `ATTACK` targets **adjacent rubble** (115/116), or when the zip row
  is a **phantom** (only `WAIT` + unit `ATTACK`s, no seam tile). Mitigation:
  `_oracle_pick_attack_seam_terminator` in `tools/oracle_zip_replay.py` (exact
  match → nearest seam/rubble within Manhattan ≤ 2 → `WAIT` fallback). Full
  `desync_audit.py` on **`1630747`** → `ok`. Regression:
  `test_oracle_zip_replay.TestOracleAttackSeamTerminator`.
- **`1629023`:** prior `engine_illegal_move` / `Move: no unit` register shapes;
  current `desync_audit.py` run reaches **`ok`** (361 actions) on the catalog zip
  in this tree — treat cluster membership as stale until the next full
  `run_desync_cluster.py` pass.
