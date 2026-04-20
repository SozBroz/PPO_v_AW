# Debug notes — `oracle_move_no_unit` & `oracle_move_terminator`

Handoff for agents working the **second and third** largest subtype buckets (see `logs/desync_clusters.json` keys). Regenerate clusters after changing `tools/oracle_zip_replay.py` or rerunning `tools/desync_audit.py`.

## Shared tooling

| Tool | Role |
|------|------|
| `logs/desync_clusters.json` | `games_id` lists per subtype key |
| `docs/desync_bug_tracker.md` | Example messages + truncated ID lists |
| `python tools/debug_desync_failure.py --games-id <id>` | Stops at first exception; prints `action_stage`, selection, `len(get_legal_actions())`, alive unit count |
| `python tools/desync_audit.py --games-id <id>` | Authoritative first-divergence `class` / `message` (may differ from older JSONL rows) |

**Important:** First failure **type** can change as oracle code evolves (e.g. a game that was `oracle_move_terminator` may now fail earlier on `Fire` or `engine_bug`). Always confirm with a fresh `--games-id` audit.

---

## Subgroup A — `"oracle_move_no_unit"`

**Oracle error shape:** `Move: no unit for engine P… (awbw id …) at path (…) or global (…)`  
**Code:** `tools/oracle_zip_replay.py` → `_apply_move_paths_then_terminator` (unit resolution before SELECT/move commit).

**Envelope-aware `paths` / `unit` (2026-04):** GL exports sometimes omit `paths.global` / `unit.global` and nest only under the AWBW player id. `_apply_move_paths_then_terminator` now uses `_oracle_resolve_move_paths` and `_oracle_resolve_move_global_unit` with the current `p:` line’s `envelope_awbw_player_id` (same idea as `_oracle_move_paths_for_envelope` / `_oracle_move_unit_global_for_envelope` on `Fire`). Regression: `tests/test_oracle_move_resolve.py`.

### What we verified (games_id **1619191**)

1. Failure triggers on **`Capt`** (nested `Move`), not only plain `Move`.
2. At the exception, `debug_desync_failure` reports:
   - `action_stage=SELECT`, `alive_units_total=1`
   - Active player is P0, but the zip expects a **P0 infantry** on the `paths.global` march while **the engine only has one unit on the entire map** (the opponent’s piece after day-1 play).
3. PHP **first snapshot** in the zip also lists **one** unit — so the divergence is not “oracle forgot a tile”; it is **replay state not accumulating units** the site had at the same logical day (e.g. builds not materializing, or ordering so P0 never received their field army).

### Triage split for agents

| Situation | Meaning | Next step |
|-----------|---------|-----------|
| Engine has units for that player, but none on path/global anchors | Geometry / `units_id` / naming resolver | Extend `_guess_unmoved_mover_from_site_unit_name`, denser path cells, etc. |
| Engine unit count vs PHP snapshot already wrong **before** the move | Initial/deploy/build pipeline | `make_initial_state`, `Build` handling, `tools/replay_state_diff.py` |
| Failure only on `Capt` | Same resolver as `Move`; nested `Move` uses `_oracle_resolve_move_global_unit` + envelope id | Trace `apply_oracle_action_json` `Capt` branch |

---

## Subgroup B — `"oracle_move_terminator"`

**Oracle error shape:** `Move resolved to ACTION but no legal terminator at (r, c) …; legal=[]`  
**Code:** `tools/oracle_zip_replay.py` → `_finish_move_join_load_capture_wait` after the move commit.

### Why `legal=[]` is possible

`get_legal_actions` in `engine/action.py` returns **no** ACTION-stage actions when:

1. **`selected_move_pos is None`** (or `selected_unit` is None) while `action_stage == ACTION` — should not happen after a clean commit; indicates a bad prior step.
2. **Boarding branch:** mover ended on a **friendly** tile that is neither a legal **LOAD** nor **JOIN** target — `_get_action_actions` intentionally returns `[]` (see comment: non-loadable friendly occupant).

So many “terminator” rows are **move end tile vs engine occupancy/rules**, not a missing `WAIT` enum.

### What we verified

- Several games in older registers that were labeled terminator now fail **earlier** (e.g. `Fire`, `Illegal move`) after other fixes — **re-run audit** before prioritizing.
- When the message is still `no legal terminator` with `legal=[]`, use `debug_desync_failure` **at that** failure; if the run stops earlier, fix the earlier defect first.

---

## Subgroup C — `loader_rv1_no_action_stream` / `loader_snapshot_or_zip`

**Register `class`:** `loader_error`  
**Typical cause:** The zip from `replay_download.php` is **ReplayVersion 1**: one gzipped PHP snapshot member named `<games_id>` only. There is **no** `a<games_id>` gzip containing `p:` envelopes, so `parse_p_envelopes_from_zip` returns `[]` and `desync_audit` stops with `ReplaySnapshotOnly` / “snapshots only”.

**Verified (e.g. games_id 1629304):** `zip.namelist() == ['1629304']` — no `a*` member. The snapshot gzip has **no** embedded `p:` lines. This is **not** fixable in the oracle without a new download or a trace rebuild.

### Code / ops

| Piece | Role |
|-------|------|
| `tools/oracle_zip_replay.replay_zip_has_action_stream(path)` | `True` iff some `a*` member decompresses to text containing `p:` |
| `tools/oracle_zip_replay._pick_action_gzip_member` | Prefers `a{games_id}` when multiple `a*` members exist |
| `tools/amarriner_download_replays.py` | After each save, warns if RV1; use `--require-action-stream` to reject and delete snapshot-only zips |

Re-fetch affected games with `python tools/amarriner_download_replays.py --games-id <id> --force` — the mirror may later serve a full zip.

---

## `oracle_capture_path` — `Capt (no path): no unit on tile (r,c)`

**Code:** `tools/oracle_zip_replay.py` — `_oracle_capt_no_path_*`, envelope `Capt` with `Move: []`.

**Message tags (failure split):**

| Tag | Meaning |
|-----|---------|
| `[resolver]` | `buildingInfo` does not point at a **property** terrain id — wrong `buildings_y` / `buildings_x` in the JSON, or export shape issue. Fix mappers / coords. |
| `[drift]` | Tile is a property, but the **engine state has no capturer** orth / diagonal / outer-ring to that tile. The zip expects a unit the replay never placed (funds, build order, prior desync). **Not fixable by oracle-only mapping** — fix engine replay parity or initial state. |
| `[resolver gap]` | Capturers exist in the engine neighborhood but selection failed — extend `_oracle_capt_no_path_*` (rare; indicates a new AWBW envelope shape). |

**Resolver tools:** optional `Capt.unit.global` (`units_y`/`units_x`) pins the walker; `buildings_players_id` on `buildingInfo` (including nested `"0"`) disambiguates multiple orth capturers when it maps through `awbw_to_engine`.

---

## Quick repro commands

```powershell
python tools/debug_desync_failure.py --games-id 1619191
python tools/desync_audit.py --games-id 1619191 --register logs/tmp.jsonl
python -c "from pathlib import Path; from tools.oracle_zip_replay import replay_zip_has_action_stream; print(replay_zip_has_action_stream(Path('replays/amarriner_gl/1629304.zip')))"
```

---

## Files to give each agent (reminder)

| Subgroup | Primary code | Engine reference |
|----------|----------------|------------------|
| `oracle_move_no_unit` | `tools/oracle_zip_replay.py` (`_apply_move_paths_then_terminator`, `_guess_unmoved_mover_from_site_name`) | `engine/game.py` unit positions |
| `oracle_move_terminator` | `tools/oracle_zip_replay.py` (`_finish_move_join_load_capture_wait`, `_oracle_finish_action_if_stale`) | `engine/action.py` `get_legal_actions` / `_get_action_actions` |
| `oracle_capture_path` | `tools/oracle_zip_replay.py` (`Capt` no-path, `_oracle_capt_no_path_*`) | Unit positions vs zip; see message `[drift]` vs `[resolver]` |
| `loader_rv1_no_action_stream` / snapshot zip | `tools/amarriner_download_replays.py`, `replay_zip_has_action_stream` | — |

Attach **`logs/desync_clusters.json`** (filter JSON to the single key) + this doc + `docs/desync_bug_tracker.md` for ID lists.
