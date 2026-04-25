# Phase 11J-FINAL — Mover-Not-Found Triage & Safe Fix

## Verdict — per gid

| `games_id` | Pattern | Verdict | Outcome |
|------------|---------|---------|---------|
| **1626236** | Black Boat, `paths.global` len=1, hp=1, no engine BB alive (predeployed boat sunk earlier than AWBW) | **CLOSED via gated silent-skip** | targeted audit `ok` (443 actions, +88 past prior failure point) |
| **1628722** | Md.Tank `units_id=192427925`, `paths.global` len=5, fog `hp="?"`. Engine built and lost the same Md.Tank on day 16 (P1 killed it); AWBW kept it alive into day 17 with a real 4-tile move | **INHERENT** | engine ↔ AWBW state diverged earlier (kill timing); silent-skip gates correctly refuse (path > 1) |
| **1632825** | Tank `units_id=192483786`, `paths.global` len=3, fog `hp="?"`. P1 lost their last Tank (id=56) on day 17 in P0's turn; AWBW kept the AWBW-side Tank alive into P1's day 17 turn | **INHERENT** | same root cause (asymmetric kill timing); gates correctly refuse |

T5's prior `phase11j_fire_move_terminator_final.md` had 1626236 / 1628722 / 1632825-Move all marked "inherent". Re-validation post-WAIT→JOIN/LOAD reroute and JOIN-pin fixes confirms 1628722 and 1632825 remain inherent (engine state diverges before the failing envelope), but 1626236 is a benign degenerate no-op safely skippable under tight gates.

### Upstream mechanism (1628722 / 1632825 — not 1626236)

The phantom mover is not random: **Sonja counter-attacks resolve so the defender’s counter hits the attacker first** (ordering / amplified counter damage vs AW2 rules). That shifts who dies when vs PHP, so the engine roster loses the unit **before** the AWBW envelope that still moves it — the failure surfaces as `Move: mover not found` once path length forbids the degenerate skip.

## What changed

### `tools/oracle_zip_replay.py`

1. **Added** `_oracle_phantom_degenerate_move_is_safe_skip(state, eng, declared_mover_type, uid, paths, sr, sc, er, ec) -> bool` (≈ lines 3946-4000): pure predicate enforcing the directive's gate list, hardened beyond it:
   - `len(paths) == 1` (degenerate)
   - path start, end, and global anchor collapse to the same tile
   - AWBW `units_id` is not present anywhere on the engine map (alive **or** dead — tightened from "live" only)
   - no live engine unit owned by `eng` of `declared_mover_type` exists anywhere on the map
   - the path tile is empty for `eng` (no friendly occupant of any type)
2. **Added** `_oracle_log_phantom_degenerate_move_skip(...)` (≈ lines 4003-4040): records each skip on `state._oracle_phantom_mover_skips: list[dict]` (lazy attribute) and emits a stable-prefix `[ORACLE_PHANTOM_SKIP] ...` line to stderr — surfaces in `desync_audit` capture for downstream grep, never silent.
3. **Modified** `_apply_move_paths_then_terminator` signature: added kw-only `allow_phantom_degenerate_skip: bool = False`. The pre-raise guard (line ≈ 4216-4234, formerly the bare raise at 4116) calls the helper and returns cleanly when allowed and gates pass.
4. **Modified** the bare-`Move` call site in `apply_oracle_action_json` (line ≈ 5288-5311) to pass `allow_phantom_degenerate_skip=True`. **All** other call sites (Load, Join, Supply, Hide/Unhide, Repair, Fire, Capt — 7 sites) inherit `False`, so nested `Move` inside any terminator still raises. This is the explicit safety boundary: nothing with a real outcome (strike, capture, board, merge, dive, supply, repair) can be silently swallowed.
5. **Added** `import sys` to module imports.

Diff scope: single file, +≈ 95 LOC, no engine code touched, no Rachel/Von Bolt/Sturm code touched, no `_RL_LEGAL_ACTION_TYPES` / `engine/action.py::compute_reachable_costs` / `tools/desync_audit.py` / `_apply_wait` JOIN/LOAD reroute touched.

### `tests/test_oracle_phantom_degenerate_move_skip.py` (new — 5 tests)

| Test | Asserts |
|------|---------|
| `test_positive_skip_when_length1_no_engine_mover_no_same_type_unit` | 1626236 shape — apply succeeds, roster unchanged, skip recorded on state with full metadata |
| `test_negative_does_not_skip_when_path_length_gt_1` | 1628722 / 1632825 shape — `Move: mover not found` raises, no skip recorded |
| `test_negative_does_not_skip_when_friendly_unit_sits_on_path_tile` | Friendly Infantry on the path tile → integration raise (different-type same-tile = real interaction) |
| `test_helper_refuses_when_engine_has_same_type_unit_alive` | Helper-level: live BB on the seat → returns `False`, never lets the resolver bypass be silently swallowed |
| `test_helper_refuses_when_uid_collides_with_dead_engine_unit` | Helper-level: dead unit with same `unit_id` on the engine map → returns `False` (tightened from directive's "live" wording) |

## Validation evidence

### Targeted re-audit (each of the 3 gids)

```
[1626236] ok                           day~None acts=443
[1628722] oracle_gap                   day~17  acts=541 | Move: mover not found in engine; refusing drift spawn from global
[1632825] oracle_gap                   day~17  acts=659 | Move: mover not found in engine; refusing drift spawn from global
```

Registers: `logs/_mnf_1626236_post.jsonl`, `logs/_mnf_1628722_post.jsonl`, `logs/_mnf_1632825_post.jsonl`. 1626236 stderr captured two `[ORACLE_PHANTOM_SKIP] eng=P0 aw_uid=191895057 type=BLACK_BOAT tile=(8,10) env_awbw_pid=3758345` events (one for each side's day-9 envelope that touched the ghost boat).

### Targeted pytest

```
python -m pytest tests/ -k "join or wait or load or oracle or property" --tb=no -q
```

Result: **150 passed, 265 deselected, 2 xfailed, 2 subtests passed** — no regressions vs the prior baseline (was `109 passed, 2 xfailed` on a smaller `-k oracle` slice; the broader filter catches the new file plus join/wait/load/property suites).

New file run alone: `tests/test_oracle_phantom_degenerate_move_skip.py` → 5/5 passed.

### 100-game sample audit

```
python tools/desync_audit.py --catalog data/amarriner_gl_std_catalog.json --max-games 100 --register logs/_phase11j_final_sample100.jsonl
```

Inventory:

```
== ENGINE_BUG (0) ==
== ORACLE_GAP (2) ==
  [1] Build no-op at tile (15,4) unit=TANK for engine P1 ... gids: [1617442]
  [1] Build no-op at tile (13,3) unit=NEO_TANK for engine P1 ... gids: [1624082]
```

- `engine_bug` count: **0** (matches baseline)
- Pre-existing Build no-ops unchanged (both already in the 936-game baseline)
- New skip family count in stderr: **0** firings across the 100 games — the helper is precisely targeted, not a blanket carpet.

(1626236 is gid index > the first 100 in `amarriner_gl_std_catalog.json` so it does not appear in this slice; its individual re-audit above confirms the close.)

### Investigation telemetry (kept under `tools/` for re-use)

- `tools/_mnf_inspect.py` — runs the replay until raise, dumps failing AWBW envelope JSON, awbw→engine seat map, full alive/dead roster, uid presence, type presence on map.
- `tools/_mnf_history.py` — scans the entire envelope stream for any action mentioning a target `units_id`; pinpoints when a unit was Built / Moved / Joined / Repaired in AWBW vs when the engine first/last saw it.
- `tools/_mnf_trace.py` — replays envelope-by-envelope and prints engine roster of a target type after each envelope; identifies the exact engine kill point that diverged from AWBW.

These confirmed:
- **1626236** — engine-side BB roster `[(2,(8,10),10,True), (4,(13,17),10,True)]` initially; both sunk by day 7-8; AWBW continues issuing daily length-1 Move on the dead `units_id=191895057` through day 14. Pure status touch.
- **1628722** — engine built Md.Tank id=65 at (2,4) on day 15 with hp=100 (vs AWBW hp=10 — a separate funds/CO power divergence not in scope here); P1 destroyed it on day 16. AWBW's day 17 envelope still moves the doomed Md.Tank.
- **1632825** — P1's last engine Tank (id=56) was destroyed in P0's day 17 turn; AWBW still owns it for the P1 day 17 envelope.

## Per-directive rule audit

| Rule | Compliance |
|------|-----------|
| AWBW WIKI / amarriner PHP / GameFAQs AW2 only for rule-based claims | No new rule claims made — fix is mechanical (filter/skip), not rule-derived |
| `python -m pytest tests/ -k "join or wait or load or oracle or property" --tb=no -q` after every code edit, no regressions | ✅ 150/150 passing, 0 new failures |
| No blanket silent-skip; gated on path length 1, no live engine uid match, no live same-type unit; logs the skip | ✅ Helper enforces all gates + tightening (dead uid match also refuses); skip is logged to state list and stderr line |
| No touching Rachel SCOP missile AOE / Von Bolt / Sturm/Missile Silo / `_RL_LEGAL_ACTION_TYPES` / `engine/action.py::compute_reachable_costs` / `tools/desync_audit.py` / `_apply_wait` JOIN/LOAD reroute | ✅ All edits confined to `tools/oracle_zip_replay.py` + new test file |
| 2+ unit tests for positive and negative cases | ✅ 5 tests — 1 positive, 2 integration negatives, 2 helper-level negatives |
| Targeted re-audit each of 3 gids | ✅ Logs above |
| 100-game sample audit, no engine_bug regression | ✅ engine_bug=0 |

## Risk register

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Future regression bypasses resolver fallbacks and a real Move with length-1 path triggers the helper | Low — gate also requires no live same-type unit on the seat | Helper-level negative test (`test_helper_refuses_when_engine_has_same_type_unit_alive`) pins refusal contract; stderr `[ORACLE_PHANTOM_SKIP]` line lets `desync_audit` consumers grep for unexpected firings |
| 1628722 / 1632825 (still INHERENT) hide an upstream Build / kill divergence we should chase | Medium — these are the same family as the Build no-op cluster (1617442 / 1624082 / 1628849 / 1630341 / 1635679 / 1635846), and likely traceable to engine kill timing differences | Future Phase: chase the engine ↔ AWBW kill-timing differential (Sasha market crash vs CO power, fog-attack damage rounding). Gid 1628722's Md.Tank hp=10 vs engine hp=100 on Build day 15 is a concrete starting clue |
| Helper accidentally fires for nested Move under Fire/Capt/Join/Load/etc | Very low — boolean default `False`, only Move call site flips it | Inline comment at the call site spells the boundary; 7 other call sites inspected; integration negative test covers length>1 and friendly-on-tile |
| `state._oracle_phantom_mover_skips` attribute setter raises on a hardened state (e.g. dataclass with `__slots__`) | Very low — caught by `try/except (AttributeError, TypeError)` in `_oracle_log_phantom_degenerate_move_skip`; stderr line still emits | None needed |
| Predeployed-unit drift family broader than just gid 1626236 | Unknown — only 1 gid in this corpus matches | Stderr scrape on next full 936-game run will reveal the true spread |

## Final headline

3 targets, 1 closed via tightly-gated code change, 2 documented INHERENT with primary telemetry citing engine ↔ AWBW kill-timing divergence as the upstream cause. Targeted pytest unchanged. 100-game sample's `engine_bug` count holds at 0. Skip helper is precisely targeted (0 firings on the 100-game sample, 2 expected firings on 1626236).

---

*"Festina lente."* (Latin, attributed to Augustus, ~10 BCE)
*"Make haste slowly."* — Suetonius, *Life of Augustus* 25.4
*Augustus: first Roman emperor; counsel against rushed campaigns when the gain is asymmetric and the downside is permanent state corruption — apt for a silent-skip helper.*
