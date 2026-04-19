# Human vs bot — Play UI and learning

In-repo index for `/play/` (human vs MaskablePPO bot), `human_demos.jsonl`, and behaviour cloning. Paths are relative to the repository root.

## Run the server

```powershell
python -m server.app
```

Use `use_reloader=False` in `server/app.py` (or `flask run --no-reload`) so the in-memory session dict is not wiped when Werkzeug reloads on file edits.

**Session TTL:** MVP does not expire sessions; `server/play_human.py` defines `_SESSION_TTL_S = None` as a stub for a future sweep.

## Repository index

| Need | Location |
|------|----------|
| Game rules, `step`, win, turn flip | `engine/game.py` |
| Legal actions, SELECT/MOVE/ACTION, COP/SCOP/END/BUILD/REPAIR | `engine/action.py` — `get_legal_actions`, `_get_select_actions`, `_get_action_actions` |
| CO charge, thresholds | `engine/co.py` |
| Flat action index ↔ `Action` | `rl/env.py` — `_action_to_flat`, `_flat_to_action`, `_get_action_mask`, `ACTION_SPACE_SIZE` |
| **Flat RL indices: END=0, COP=1, SCOP=2** (not `ActionType` enum ordinals) | `rl/env.py` |
| MOVE destination collapse (BC limitation) | `rl/env.py` `_action_to_flat` `SELECT_UNIT` branch |
| Observations | `rl/encoder.py` — `encode_state`, `N_SPATIAL_CHANNELS`=59, `N_SCALARS`=16 |
| Board JSON | `server/write_watch_state.py` — `board_dict` |
| Canvas | `server/static/board.js` — `renderBoard`: **16×16** logical tiles, **×3** scale; Replay Player PNGs via `manifest.json` (`tools/sync_awbw_textures.py`, includes `Map/AW2/…` buildings); procedural fallback |
| Play client | `server/templates/play.html`, `server/static/play.js` |
| Red/blue seats (P0 first, human = P0) | `docs/player_seats.md` |
| Slice `game_log.jsonl` by who opens (seat / tempo) | `docs/seat_measurement.md` |
| Play API + sessions + bot + demo log | `server/play_human.py` |
| Routes | `server/routes/game.py` |
| App factory | `server/app.py` |
| Checkpoint opponent / training | `rl/self_play.py` |
| Network | `rl/network.py` |
| END_TURN tests | `test_action_space_prune.py` |
| BC trainer | `scripts/train_bc.py` |
| Post-BC win-rate smoke | `scripts/eval_imitation.py` |
| Trace / replay zip → `human_demos.jsonl` | `scripts/replay_to_human_demos.py`, `tools/human_demo_rows.py` |
| GL Std catalog (Amarriner HTML) | `tools/amarriner_gl_catalog.py` |
| GL Std replay download (auth) | `tools/amarriner_download_replays.py` |
| Batch oracle zip → `human_demos` JSONL | `tools/amarriner_zips_to_jsonl.py` |
| Symmetric zip vs zip eval (both seats) | `scripts/symmetric_checkpoint_eval.py` |
| Oracle zip → engine (Move/Build/Fire/End) | `tools/oracle_zip_replay.py` |
| Replay export (not MVP UI) | `.cursor/skills/awbw-replay-system/SKILL.md`, `tools/export_awbw_replay.py` |

## HTTP API (summary)

- `POST /play/api/new` — optional JSON: `map_id`, `tier`, `human_co_id` / `co_id`, `bot_co_id`. Returns full state envelope + `session_id`.
- `POST /play/api/step` — `{ "session_id", "kind", ... }` per plan; blocks until bot finishes its turn.
- `GET /play/api/state/<session_id>` — same envelope.
- `POST /play/api/cancel_selection` — `{ "session_id" }`; mutates selection only; **never** logged to `human_demos.jsonl`.

Payload includes `legal_global`, `co_p0` / `co_p1` with `cop_pct` / `scop_pct`, `action_stage`, hint arrays (`selectable_unit_tiles`, `factory_build_tiles`, `factory_build_menu`, `reachable_tiles`, `attack_targets`, `repair_targets`, `unload_options`, `action_options`), and `board`.

### Factory BUILD (SELECT)

- **`factory_build_tiles`**: `[row, col]` of owned, empty bases / airports / ports where at least one legal `BUILD` exists (same rules as `engine/action.py` `_get_select_actions`).
- **`factory_build_menu`**: `[{ "pos": [r, c], "options": [{ "unit_type": "INFANTRY", "type_id": 0, "cost": 1000 }, ...] }, ...]` — one entry per factory; options are exactly the legal builds the server would accept.
- **Client**: click a highlighted production tile (gold tint in SELECT) → choose a unit in the **dropdown**, click **Build**, then `POST /play/api/step` with body:

```json
{
  "session_id": "<uuid>",
  "kind": "build",
  "factory_pos": [row, col],
  "unit_type": "INFANTRY"
}
```

(`unit_type` may be the `UnitType` enum **name** string or integer id.)

**END_TURN** is only legal when every friendly unit has finished its action for the turn (see engine). If you have no units to move but can still BUILD, use a factory first.

### ACTION — join (same-type allies)

Move a unit onto an **injured** (ally **HP** below max) **same-type** friendly unit to **merge**: combined HP caps at full; overflow **display** HP bars convert to **funds** at `(unit price / 10)` per excess bar (AW2-style). Fuel and ammo take the **max** of the two. **Transports with cargo** cannot join. Use **`JOIN`** (destination tile click or **Join** button), not **Wait**. Engine: `ActionType.JOIN`, `GameState._apply_join`.

### ACTION — capture (cities, bases, ports, HQ, labs)

After **MOVE**, the unit is still drawn at its start tile until you finish ACTION; the yellow highlight shows **`selected_move_pos`** (the tile you moved onto). If **`action_options`** includes **`CAPTURE`** (infantry/mech on neutral or enemy **property**), either:

- Click that **same destination tile**, or  
- Press the **Capture** button.

Each successful `POST /play/api/step` with `kind: "capture"` applies one capture tick. **`board.properties[].capture_points`** (1–20, default 20) is drawn on the map when below 20 so partial progress is visible. Full capture flips `owner` and resets points to 20. On **full** capture only, the engine also updates **`board.terrain`** (and the property’s `terrain_id`) to the capturer’s **faction-coloured building tile** so the canvas matches AWBW; partial capture leaves the neutral/enemy building art until the tile is owned.

At the **start** of each player’s turn, units on **owned** repair tiles may gain up to **+2 displayed HP** per day (and fuel/ammo top-up where applicable). That heal costs **20% of the unit’s deployment price** for a full +2 bars (internal +20 on the 0–100 HP scale); if only part of that heal applies (e.g. hitting max HP sooner) or the treasury cannot afford the full tick, the engine heals **only the affordable / cap-limited** HP and charges **the same linear rate** (integer gold, min 1G when the listed cost is positive). Ground units use owned **HQ / base / city**; air uses **airport**; naval uses **port**. **Labs and comm towers** do not grant this heal. **CO power heals** are separate and do not use this path. Implemented in `engine/game.py` (`GameState._resupply_on_properties`); same path for **`train.py` / RL** and **`/play/`**.

## `human_demos.jsonl`

**Training path:** every legal human `POST /play/api/step` appends one JSON line to **`data/human_demos.jsonl`** (repo root). That path is the **default** for `scripts/train_bc.py --demos`, so games you play in `/play/` are already in the right place for behaviour cloning—no export step. Rows are flushed after each write so a crash between turns does not lose completed actions.

Each line logs (before `step`): `encoder_version`, `spatial`, `scalars`, `action_mask`, `action_idx`, `action_stage`, `action_label` (includes `move_pos` for MOVE), `active_player`, `map_id`, `tier`, `session_id`. Bot moves are not logged (imitation is human-only). MOVE-stage `action_idx` does **not** encode destination — BC skips MOVE rows by default.

### Offline ingest (trace JSON or oracle zip)

**Preferred input:** `replays/<id>.trace.json` from self-play / export — it is the engine’s own `full_trace`, so rows align exactly with `_trace_to_action` in `tools/export_awbw_replay_actions.py`.

```powershell
python scripts/replay_to_human_demos.py --trace-json replays/272176.trace.json --out data/imported_demos.jsonl
python scripts/train_bc.py --demos data/imported_demos.jsonl --load checkpoints/latest.zip --save checkpoints/post_import.zip --epochs 2
```

**Oracle AWBW Replay Player `.zip`:** `scripts/replay_to_human_demos.py` can emit rows from the `p:` stream when you pass `--oracle-zip` plus `--map-id`, `--co0`, `--co1`, `--tier` (must match the game). The mapper in `tools/oracle_zip_replay.py` supports **Move**, **Build**, **Fire**, and **End** as produced by this repo’s exporter; it round-trips zips built via `write_awbw_replay_from_trace` in `tools/export_awbw_replay.py` (see `test_oracle_zip_replay.py`). Live-site zips may include extra action kinds until they are mapped.

**MOVE rows and BC (explicit policy):** we **do not** extend the flat `action_idx` for MOVE destinations. Offline ingest **skips MOVE-stage rows by default** (same as `scripts/train_bc.py` without `--include-move`). Use `--include-move` only if you accept degenerate supervision (many destinations collapse to the same index). For stronger move supervision, a future change would need a **separate move head** or a **different flat encoding** — not implemented here.

**Amarriner Global League catalog (no per-game HTTP):** `tools/amarriner_gl_catalog.py build` walks `gamescompleted.php?league=Y&type=std&start=` with **start = 1, 51, 101, …** (50 per page), parses each row’s `games_id`, GL tier, matchup text, `map_id`, map name, and both CO portrait ids from the listing HTML, and **merges** into `data/amarriner_gl_std_catalog.json` so repeat runs do not rescrape the whole history unless you add pages. Filter the cache with `python tools/amarriner_gl_catalog.py count --map-id 123858 --co-id 1`.

### Amarriner replay download → ingest → BC fork → symmetric eval

**Credentials:** repo-root `secrets.txt` (line 1 username, line 2 password), same as `tools/fetch_predeployed_units.py`. The file is gitignored; never commit credentials.

**How many zips (catalog slice currently in repo):** up to **800** games if you download the full merged catalog; **52** if you filter **Misery** only (`map_id` 123858). Use flags to match the cohort you want before burning disk or site load.

**1. Download replay ZIPs** to `replays/amarriner_gl/{games_id}.zip` (directory is gitignored):

```powershell
# Dry-run: count Misery games in catalog
python tools/amarriner_download_replays.py --map-id 123858 --dry-run

# Download all Misery replays (52), polite pacing
python tools/amarriner_download_replays.py --map-id 123858 --sleep 1.0 --manifest data/amarriner_dl_failures.jsonl

# Full GL Std slice (800) — only when you intend the full mirror
python tools/amarriner_download_replays.py --sleep 0.75 --manifest data/amarriner_dl_failures.jsonl
```

Optional filters: `--tier T3`, `--mirror-andy` (both CO id 1), `--co-p0-id` / `--co-p1-id`, `--max-games N`, repeated `--games-id ID`. Existing non-empty zips are skipped unless `--force`.

**2. Oracle ingest to one JSONL** (metadata per game comes from the catalog; only games with a zip on disk are ingested). Failures append to `--manifest` if set.

```powershell
python tools/amarriner_zips_to_jsonl.py --map-id 123858 --out data/amarriner_bc_rows.jsonl --manifest data/amarriner_ingest_failures.jsonl
```

**Human Andy mirrors first:** convert your traces with `scripts/replay_to_human_demos.py --trace-json … --out data/my_andy.jsonl`, then:

```powershell
python tools/amarriner_zips_to_jsonl.py --map-id 123858 --prepend-jsonl data/my_andy.jsonl --out data/merged_bc_misery.jsonl --manifest data/amarriner_ingest_failures.jsonl
```

**3. Fork `latest` and behaviour-clone** so self-play does not overwrite the zip you load. `checkpoints/` is gitignored; adjust paths if yours differ.

```powershell
copy checkpoints\latest.zip checkpoints\latest_fork_pre_bc.zip
python scripts/train_bc.py --demos data/merged_bc_misery.jsonl --load checkpoints\latest_fork_pre_bc.zip --save checkpoints\amarriner_bc.zip --epochs 2
```

**4. Symmetric head-to-head on Misery Andy mirror** (~7 games: default 4 with candidate as P0, 3 as P1). Writes optional JSON summary for a findings log.

```powershell
python scripts/symmetric_checkpoint_eval.py --candidate checkpoints\amarriner_bc.zip --baseline checkpoints\latest.zip --map-id 123858 --tier T3 --co-p0 1 --co-p1 1 --json-out data/symmetric_eval_last.json
```

**Episode length:** `AWBWEnv` supports `max_env_steps` (count of P0 `env.step` calls) and an internal cap on P1 “microsteps” per P0 step so opponent auto-play cannot spin forever. `symmetric_checkpoint_eval.py` and `bo3_checkpoint_playoff.py` expose `--max-env-steps` (default **100**; use `0` for unlimited) and optional `--max-p1-microsteps`. This is separate from engine calendar **`MAX_TURNS`** in `engine/game.py` (still the day-based game-end rule for training).

Interpretation: the script prints **candidate** wins as P0 and as P1 and a **promotion_heuristic_ok** flag (candidate ahead overall and no all-loss collapse in either seat). This is guidance only; you decide whether to overwrite `latest.zip`.

**5. If you promote and restart PPO:** before stopping `train.py`, copy the **full** command line from your terminal or Task Manager (see `train.py --help` for all flags: `--iters`, `--n-envs`, `--n-steps`, `--device`, `--map-id`, `--watch-only`, `--co-p0` / `--co-p1`, `--tier`, `--curriculum-broad-prob`, `--curriculum-tag`, `--save-every`, `--checkpoint-pool`, `--rank`, `--features`). Prefer a single Ctrl+C for a clean stop. Then back up and replace:

```powershell
copy checkpoints\latest.zip checkpoints\latest_pre_promote_<timestamp>.zip
copy /Y checkpoints\amarriner_bc.zip checkpoints\latest.zip
python train.py …   # same argv as before
```

For a Bo3 **series** that replaces `latest` when the challenger wins first-to-2, keep using `scripts/bo3_checkpoint_playoff.py` (see below); it always keeps challenger as P0, so use `symmetric_checkpoint_eval.py` when you need both seatings.

## Behaviour cloning

**Seat:** behaviour cloning ([`scripts/train_bc.py`](../scripts/train_bc.py)) and RL ([`rl/env.py`](../rl/env.py)) both train only on **engine player 0** with fixed P0-vs-P1 channel semantics in [`rl/encoder.py`](../rl/encoder.py); they do not learn a separate P1 policy head. On maps where **asymmetric predeploy** makes the engine open on P1 first, the agent can still be **second on the clock** on its first P0 turn after the opponent’s opening (see `make_initial_state` in `engine/game.py`) — that is tempo, not a seat swap in the tensor.

```powershell
python scripts/train_bc.py --demos data/human_demos.jsonl --load checkpoints/latest.zip --save checkpoints/post_bc.zip --epochs 2
```

### Bo3: human BC zip vs `latest.zip` (replace on win)

The repo’s env always steps **your** policy as **P0** and the opponent as **P1** (no seat swap). A first-to-2 playoff script lives at `scripts/bo3_checkpoint_playoff.py` (run from repo root). By default it runs up to three games in **parallel processes** (each worker loads both zips on CPU); use `--no-parallel` for the old sequential loop. Example after saving a BC zip as `checkpoints/human_bc.zip`:

```powershell
# Preview without overwriting
python scripts/bo3_checkpoint_playoff.py --challenger checkpoints/human_bc.zip --dry-run --map-id 123858 --tier T3 --co-p0 1 --co-p1 1

# Series + replace latest if challenger wins 2
python scripts/bo3_checkpoint_playoff.py --challenger checkpoints/human_bc.zip --defender checkpoints/latest.zip --map-id 123858 --tier T3 --co-p0 1 --co-p1 1
```

Add `--max-env-steps 100` (default) or `--max-env-steps 0` to disable the P0 step cap; same flags as symmetric eval.

Defender is backed up to `checkpoints/latest_pre_bo3_<UTC>.zip` before overwrite.

**Seat note:** both zips use the same observation layout; the challenger is always **P0** and the defender always **P1** for every game in the series (no automatic seat swap).

Use `--save` to a new zip, then point **`train.py`** (or `/play/`) at that checkpoint if you want PPO or the bot to pick up the BC-updated policy.

Options: `--head-only`, `--include-move` (usually a bad idea). **Copy** `checkpoints/latest.zip` to e.g. `checkpoints/latest_pre_human.zip` before destructive runs.

## Imitation eval (post heavy BC)

1. Run `scripts/eval_imitation.py` for a quick win/loss count vs **random** or **checkpoint** pool.
2. For full metrics, resume short self-play with `python train.py ... --curriculum-tag post_imitation` and compare `data/game_log.jsonl` + TensorBoard to `MASTERPLAN.md` gates.
3. Qualitative: export a replay (`tools/export_awbw_replay*.py`) if you need AWBW-compatible zip semantics; day-to-day inspection uses the in-repo `/replay/` viewer (`python -m server.app`). See `.cursor/skills/awbw-replay-system/SKILL.md`.

## Assets

Terrain and units use paths from `tools/sync_awbw_textures.py` (Classic terrain from `Tiles.json` plus **AW2** building sprites keyed by engine terrain IDs). Re-run the script after changing mappings.
