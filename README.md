# AWBW (PPO_v_AW)

Advance Wars engine, AI training pipeline, in-browser replay, and zip export tooling (AWBW-compatible replays for external tools).

**GitHub repository name:** `PPO_v_AW`.

**AWBW site login (optional):** copy `secrets.txt.example` to `secrets.txt` (line 1 username, line 2 password) for `tools/fetch_predeployed_units.py` and related scripts. `secrets.txt` is gitignored—do not commit it.

## Reference

**In-repo replay:** run `python -m server.app` and open `/replay/` — games append to `data/game_log.jsonl` from training (`rl/env`). See `server/routes/replay.py`, `server/static/replay.js`.

**Zip export format** (`tools/export_awbw_replay*.py`) targets the same on-disk layout as the open-source [AWBW Replay Player](https://github.com/DeamonHunter/AWBW-Replay-Player) (MIT). We do not ship that C# app; read parsers and JSON on GitHub when debugging. Optional local clone: `git clone --depth 1 … third_party/AWBW-Replay-Player` (still gitignored if present).

**Textures:** `python tools/sync_awbw_textures.py` pulls PNGs + JSON metadata from raw GitHub (no clone required).

## Human vs bot (Play UI)

Train or copy a MaskablePPO zip into `checkpoints/latest.zip` (or `checkpoint_*.zip`), then from the repo root:

```powershell
python -m server.app
```

Open `/play/`. The dev server must run **without** the Werkzeug reloader wiping in-memory sessions (`use_reloader=False` is set in `server/app.py`; or use `flask run --no-reload`). API, BC pipeline, post-BC eval, and BUILD/END_TURN caveats are documented in **`docs/play_ui.md`**.
