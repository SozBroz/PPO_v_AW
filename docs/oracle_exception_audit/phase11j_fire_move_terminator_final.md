# Phase 11J-FIRE-MOVE-TERMINATOR-FINAL — Lane L last pass (5-row cohort)

## Verdict: **YELLOW** — 1/5 games fully `ok`; 1/5 Fire terminator cleared with downstream Move exhaustion; 3/5 inherent; 100-game sample clean (`engine_bug` absent)

Lane L’s path-tail snap (`json_path_was_unreachable` + `_oracle_path_tail_occupant_is_evictable_drift`) never ran on these rows for the reasons below. The only closable shape in-budget was **post-kill duplicate Fire** where JSON `defender.units_hit_points <= 0` but the defender hex still held a **live same-seat** unit (drift ghost on the strike tile).

### Per-row drill

| `games_id` | Original message (prefix) | Root cause (why Lane L / existing Fire snap didn’t apply) | Outcome |
|------------|---------------------------|------------------------------------------------------------|---------|
| **1626236** | `Move: mover not found…` | **Pre-path**: envelope is a degenerate Black Boat `Move` with `paths.global` length 1; resolver finds `gu` fine but **no alive `BLACK_BOAT` exists** in engine (`black boats []`). Lane L runs only after `u` is found. | **Inherent** — AWBW action for a unit the engine roster no longer has (upstream drift / silent-skip). |
| **1628722** | `Move: mover not found…` | **Pre-path**: AWBW `units_id` **192427925** (Md.Tank) matches **no** `engine.Unit.unit_id** on any seat; only other Md.Tanks are on the opposite seat with different ids. | **Inherent** — PHP id not mapped to any live engine unit (phantom mover). |
| **1629202** | `Fire: engine board holds friendly MECH…` | **Not a path-tail blocker**: `_oracle_fire_defender_row_is_postkill_noop` required empty/dead tile; engine had **live friendly** on AWBW’s defender hex `(7,20)` while JSON defender **hp = 0** (duplicate Fire). Full Fire+Move path then hit `_oracle_assert_fire_defender_not_friendly`. | **Closed** — widened postkill predicate (see code). Audit: full replay **`ok`**. |
| **1632825** | `Fire: engine board holds friendly INFANTRY…` | Same **post-kill / friendly-on-defender-tile** shape as 1629202 (`defender` hp 0 at `(12,19)`). | **Fire terminator closed** — replay **advances** to a **later** first divergence: `Move: mover not found…` (day 17, Tank `units_id` **192483786**, JSON `units_hit_points: "?"`). | **Inherent downstream** — original Fire row no longer the stopper; exhaustion is phantom mover / malformed HP token, outside this ship’s Lane L scope. |
| **1634464** | `Fire: oracle resolved defender type MEGA_TANK…` | **`_oracle_assert_fire_damage_table_compatible`**: RECON vs MEGA_TANK has no AWBW chart entry (`get_base_damage` `None`). Not occupancy / reach / postkill. | **Inherent** — resolver/AWBW combatInfo vs engine damage table disagreement on a **live** defender (hp 7). |

### Code edits (only `tools/oracle_zip_replay.py`)

**`_oracle_fire_defender_row_is_postkill_noop`** (`1286:1324:tools/oracle_zip_replay.py`): optional kw-only `attacker_engine_player: Optional[int] = None`. When JSON defender hp ≤ 0 and the tile’s live occupant’s `player` equals that seat, return `True` (duplicate Fire / drift on defender hex). Docstring cites **phase11j_move_truncate_ship.md** and **phase11j_lane_l_widen_ship.md**. Callers without the kwarg behave as before (tests / legacy).

**Call sites** (pass acting engine seat):

- Fire **no-path** branch: `6133:6135:tools/oracle_zip_replay.py`
- Fire **Move** nested branch (postkill snap path): `6276:6278:tools/oracle_zip_replay.py`

**Not changed**: `_apply_move_paths_then_terminator`, `_oracle_path_tail_occupant_is_evictable_drift`, Von Bolt / Rachel block (`5041+`), `_RL_LEGAL_ACTION_TYPES`, any `engine/*`.

### Closure count

| Bucket | Count |
|--------|------:|
| Full replay `ok` (this cohort) | **1** (`1629202`) |
| Fire-only closure with later different `oracle_gap` | **1** (`1632825`) |
| Inherent / documented | **3** (`1626236`, `1628722`, `1634464`) + downstream note on `1632825` Move |

### Validation

- **Targeted re-audit** (same catalogs as directive): `1629202` → `ok`; others remain `oracle_gap` except `1632825` now fails later on Move (see above).
- **`python -m pytest tests/ -k oracle --tb=no -q`**: `109 passed`, `2 xfailed` (unchanged), `0` failed.
- **100-game `desync_audit` sample** (`--max-games 100`): **`engine_bug` rows: 0** in `logs/_t5_sample100_post.jsonl`; `oracle_gap` only on pre-existing Build no-op shapes (no new failure family observed from this change).

### Verdict letter rationale

- Not **GREEN** (need ≥3/5 cleanly `ok` on the five gids).
- **YELLOW**: ≤2 closable outcomes in this cohort, clean documentation of the rest, regression sample and tests held.
- Not **RED**: 100-game sample did not gain `engine_bug`; oracle test slice green.

---

*“You will not find it difficult to prove that battles, campaigns, and even wars have been won or lost primarily because of logistics.”* — Dwight D. Eisenhower, speech to a logistics class (ca. 1940s)  
*Eisenhower: U.S. Army general and later U.S. President; here on supply lines as the hidden decisive front.*
