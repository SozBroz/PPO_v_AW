# AWBW (PPO_v_AW)

Advance Wars engine, AI training pipeline, in-browser replay, and zip export tooling (AWBW-compatible replays for external tools).

**GitHub repository name:** `PPO_v_AW`.

**AWBW site login (optional):** copy `secrets.txt.example` to `secrets.txt` (line 1 username, line 2 password) for `tools/fetch_predeployed_units.py` and related scripts. `secrets.txt` is gitignored—do not commit it.

## Reference

**In-repo replay:** run `python -m server` or `python -m server.app` and open `/replay/` — games append to `logs/game_log.jsonl` from training (`rl/env`). See `server/routes/replay.py`, `server/static/replay.js`.

**Zip export format** (`tools/export_awbw_replay*.py`) targets the same on-disk layout as the open-source [AWBW Replay Player](https://github.com/DeamonHunter/AWBW-Replay-Player) (MIT). We do not ship that C# app; read parsers and JSON on GitHub when debugging. Optional local clone: `git clone --depth 1 … third_party/AWBW-Replay-Player` (still gitignored if present).

**Textures:** `python tools/sync_awbw_textures.py` pulls PNGs + JSON metadata from raw GitHub (no clone required).

## Native extensions (Cython)

`engine` and `rl` ship `.pyx` modules (`setup.py` at repo root). From the repo root, after installing build deps (`pip install numpy cython setuptools wheel`), run either:

- `python setup.py build_ext --inplace` — places compiled modules next to packages, or  
- `pip install -e .` — editable install (also runs the build; needs a C compiler: **build-essential** on Linux, **Build Tools for Visual Studio** on Windows).

CI (`.github/workflows/ci.yml`) installs `build-essential` + `python3-dev` on Ubuntu, then `pip install -e .` before `pytest`, so `import engine.action` and encoder Cython paths resolve in automation.

**Windows:** If `build_ext --inplace` fails with *Access is denied* on `*.pyd`, another process (e.g. `train.py`, tests, REPL) still has the DLL loaded. Stop it, then run `build_ext --inplace` again—or use `python scripts/rebuild_cython_extensions.py`, which builds into `build/` and copies (same lock rules apply to the copy step).

## Optional: `torch.compile` on Windows (`AWBW_TORCH_COMPILE`)

GPU training can wrap the policy with `torch.compile` for a potential inference-side speedup (see `rl/self_play.py`). On **Windows** this is **off by default**: Inductor/Triton’s first run may need the **MSVC** C++ compiler (`cl.exe`), which is *not* bundled with the usual PyTorch+CUDA wheel. Machines that have CUDA for training but no **Visual Studio Build Tools** (C++ workload) should keep the default so training runs without a compiler install.

- **Enable on Windows:** install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (workload: **Desktop development with C++**) or a full Visual Studio with that workload, ensure `cl.exe` is available to the process, then set `AWBW_TORCH_COMPILE=1` (or `true` / `yes` / `on`) for the training session.
- **Linux and typical CI:** no variable needed; the code treats non-Windows as opted-in when CUDA and Triton/Inductor are usable.

## Fleet training (main + optional auxiliary PCs)

Main training is **sovereign**: `python train.py` uses only the local repo’s `checkpoints/` (or `--checkpoint-dir`) and does not require a network share, eval fleet, or promoted `best.zip`.

Optional **auxiliary** machines mount the main repo as `Z:\` (same tree: `checkpoints/`, `data/`, `scripts/`). Identity is set with `AWBW_MACHINE_ROLE=auxiliary`, `AWBW_MACHINE_ID` (e.g. `eval1`), and `AWBW_SHARED_ROOT` (default `Z:\`). See `rl/fleet_env.py` for validation rules.

| Path under repo | Role |
|-----------------|------|
| `checkpoints/promoted/candidate_*.zip`, `best.zip` | Eval aux + operator `scripts/promote.py` |
| `checkpoints/bc/bc_warmstart_*.zip` | BC aux → main `--bc-init` on fresh runs |
| `checkpoints/pool/<MACHINE_ID>/checkpoint_*.zip` | Pool aux → main `--pool-from-fleet` |
| `fleet/<MACHINE_ID>/status.json`, `eval/*.json` | Heartbeat + eval verdicts |

**Solo aux / walk-away (Tier 1):** `python scripts/start_solo_training.py --machine-id pc-b --auto-apply` probes the box, writes proposed args, launches `train.py` + `fleet_orchestrator.py`, and coordinates defensive restarts when probe-driven args change. See **`docs/SOLO_TRAINING.md`**.

**Scripts:** `scripts/fleet_eval_daemon.py` (symmetric eval loop), `scripts/promote.py` (manual or `--auto-promote` best swap). **Deferred:** `--shared-training` / MASTERPLAN §10 async weight sync (mount already removes file-copy friction).

## Human vs bot (Play UI)

Train or copy a MaskablePPO zip into `checkpoints/latest.zip` (or `checkpoint_*.zip`), then from the repo root:

```powershell
python -m server.app
```

Equivalent: ``python -m server`` from the repo root, or ``scripts\run_play_server.cmd``. Alternatively, with Flask’s CLI (after ``pip install -r requirements.txt`` loads ``python-dotenv`` and `.flaskenv`):

```powershell
flask run --no-reload --port 5000
```

Open `/play/`. The dev server must run **without** the Werkzeug reloader wiping in-memory sessions (`use_reloader=False` is set in `server/app.py`; or use `flask run --no-reload`). API, BC pipeline, post-BC eval, and BUILD/END_TURN caveats are documented in **`docs/play_ui.md`**.
