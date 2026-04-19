# AWBW (PPO_v_AW)

Advance Wars engine, AI training pipeline, and AWBW Replay Player export tooling.

**GitHub repository name:** `PPO_v_AW`.

**AWBW site login (optional):** copy `secrets.txt.example` to `secrets.txt` (line 1 username, line 2 password) for `tools/fetch_predeployed_units.py` and related scripts. `secrets.txt` is gitignored—do not commit it.

## Reference

The official AWBW Replay Player source is cloned (shallow) into
`third_party/AWBW-Replay-Player/` for local inspection when debugging export
compatibility. It is gitignored. Re-clone with:

```powershell
git clone --depth 1 https://github.com/DeamonHunter/AWBW-Replay-Player third_party/AWBW-Replay-Player
```

Useful paths inside the clone:

- `AWBWApp.Game/API/Replay/AWBWJsonReplayParser.cs` — PHP replay deserialization.
- `AWBWApp.Game/Game/Logic/GameMap.cs` — canonical map + building validation.
- `AWBWApp.Game/Game/Country/CountryStorage.cs` — country ID -> CountryData.
- `AWBWApp.Resources/Json/Countries.json` — AWBW country IDs, codes, colors.

## Human vs bot (Play UI)

Train or copy a MaskablePPO zip into `checkpoints/latest.zip` (or `checkpoint_*.zip`), then from the repo root:

```powershell
python -m server.app
```

Open `/play/`. The dev server must run **without** the Werkzeug reloader wiping in-memory sessions (`use_reloader=False` is set in `server/app.py`; or use `flask run --no-reload`). API, BC pipeline, post-BC eval, and BUILD/END_TURN caveats are documented in **`docs/play_ui.md`**.
