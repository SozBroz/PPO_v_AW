# Phase 11J-FIRE-DAMAGE-FIGHTER-TANK — AWBW chart vs oracle (gid 1631494)

## Verdict: **B** (no damage cell; oracle raises `UnsupportedOracleAction`)

**Report verdict letter: YELLOW** — chart lane and code changes are sound; two validation caveats (targeted gid does not reach the historical Fire envelope under current stepping; 100-game `ok` count vs on-disk baseline artifact differs by one stable row). **ESCALATE: N/A.**

### AWBW wiki / chart (mandatory source)

- **Primary URL used:** [https://awbw.amarriner.com/damage.php](https://awbw.amarriner.com/damage.php) (live **Damage Chart** table; same pattern as Phase 10A notes in `data/damage_table.json`).
- **Secondary URL (requested):** [https://awbw.amarriner.com/text_damage.php](https://awbw.amarriner.com/text_damage.php) — **HTTP 404** at time of audit (2026-04-21). No second source to conflict with `damage.php`.

### Verbatim Fighter row (attacker = Fighter)

Parsed from the HTML table `id="tablehighlight"` on `damage.php` (attacker row identified by first-column unit image `gefighter.gif`; defender columns follow the site header row). **Data cells only, in site column order** (defender: Anti-Air, APC, Artillery, B-Copter, Battleship, Black Boat, Black Bomb, Bomber, Carrier, Cruiser, Fighter, Infantry, Lander, Md Tank, Mech, Mega Tank, Missiles, Neo Tank, Piperunner, Recon, Rocket, Submarine, Stealth, T-Copter, Tank):

`| - | - | - | 100 | - | - | 120 | 100 | - | - | 55 | - | - | - | - | - | - | - | - | - | - | 85 | - | 100 | - |`

**Fighter vs Tank (defender column `Tank`, last column): `-`** (dash — no base damage / cannot strike per chart).

### Cross-check: `data/damage_table.json` Fighter row

- **FIGHTER → TANK:** `null` — **matches** the chart (`-`).
- **Spot-check vs same row (air targets):** B-Copter **100**, T-Copter **100**, Fighter **55**, Bomber **100** — **match** the wiki row above.
- **Wider gap (flag only, out of lane):** For the two columns parsed as **`85`** and **`-`** (site filenames for columns 22–23 were truncated in the HTML as `yc` fragments), local `damage_table.json` does not align with **85** on the same semantic column map as our 27×27 `unit_order` (naval / stealth tail differs). **Not corrected here** per single-cell mission scope; treat as a possible follow-up chart audit.

## Recon — gid 1631494

- **Zip:** `replays/amarriner_gl/1631494.zip` (present locally).
- **Register snapshot (pre-change):** `logs/desync_register_post_phase11j_combined.jsonl` — `oracle_gap` / `UnsupportedOracleAction` with resolver-miss / no damage entry (FIGHTER vs engine **TANK** at **(14, 13)**).
- **Envelope 46 (0-based), `Fire` at action index 10:** AWBW `combatInfo` shows **Fighter** attacker at **(16, 13)** (`units_y`/`units_x`), defender **(15, 13)** with post-strike **3** display HP; attacker **8** display HP — i.e. AWBW logged a real engagement and HP deltas (defender lost roughly **7** display bars if pre-strike was **10**).
- **Current stepping (engine + oracle, `--seed 1`):** Replay ends early at action **820** (envelope **45**) with `winner=1`, `win_reason=cap_limit`. The historical envelope **46** Fire is **not reached**, so a fresh `desync_audit` run reports **`ok`** for this gid even though the chart still has no FIGHTER→TANK cell. This is an **audit / game-end ordering** caveat, not a reversal of verdict B.

## Implementation (verdict B)

- **No change** to `data/damage_table.json` (cell correctly `null`).
- **`tools/oracle_zip_replay.py`:** Sharpened `_oracle_assert_fire_damage_table_compatible` message to cite [https://awbw.amarriner.com/damage.php](https://awbw.amarriner.com/damage.php) and the **`'-'`** sentinel for this matchup; still raises `UnsupportedOracleAction` (audit class **`oracle_gap`** when that path is hit).

## Tests

- **New:** `tests/test_damage_table_fighter_ground.py` — `get_base_damage(FIGHTER, TANK)` is `None`; `apply_oracle_action_json` smoke test with **`unittest.mock.patch`** on `_resolve_fire_or_seam_attacker` (vanilla engine never resolves a FIGHTER striker against a TANK tile) asserts `UnsupportedOracleAction` with `no damage entry`.

## Validation gates

1. **`python -m pytest tests/test_damage_table_fighter_ground.py -v`** — **PASS**
2. **`python tools/desync_audit.py --games-id 1631494 --register logs/desync_register_fighter_tank_targeted.jsonl --seed 1`** — **PASS** with caveat: row is **`ok`** / class **`ok`** because replay stops at **cap_limit** before envelope 46; does **not** contradict verdict B on the chart.
3. **`python tools/desync_audit.py --max-games 100 --register logs/desync_register_post_fighter_tank_100.jsonl --seed 1`** — **`engine_bug` = 0**; **`ok` = 88**, **`oracle_gap` = 12**. On-disk baseline `logs/desync_register_post_phase11j_fu_100.jsonl` has **`ok` = 89** / **`oracle_gap` = 11**; stable delta is gid **1622501** (`Build` no-op oracle_gap). **Not introduced by this lane** (message-only + tests). **Strict gate “ok ≥ 89” vs that file: FAIL**; interpret as **stale baseline artifact** unless repro shows otherwise.
4. **`python -m pytest --tb=no -q`** — **2 failures** (same count as deferred suite expectation: `test_trace_182065_seam_validation`, `test_property_day_repair_respects_insufficient_funds`). **No new third failure.**

## Diff summary

- `tools/oracle_zip_replay.py`: error string for `_oracle_assert_fire_damage_table_compatible` (chart URL + `'-'` note).
- `tests/test_damage_table_fighter_ground.py`: new.
- `docs/oracle_exception_audit/phase11j_fire_damage_fighter_tank.md`: new.

---

*"In the long history of the world, only a few generations have been granted the role of defending freedom in its hour of maximum danger. I do not shrink from this responsibility — I welcome it."* — John F. Kennedy, inaugural address, 1961  
*Kennedy: 35th President of the United States; passage on accepting hard duty when the stakes are real.*
