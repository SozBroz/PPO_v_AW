---
name: awbw-replay-system
description: Describes how AWBW replay is split between the in-repo web viewer (game_log.jsonl), the AWBW Replay Player zip pipeline (Python exporters vs third-party C# viewer source), and using WORKING_REPLAY.zip as a ground-truth fixture. Stresses that the desktop replay viewer is not modified — generated zips must match the existing viewer. Use when discussing replays, replay viewers, viewer compatibility, export_awbw_replay, export_awbw_replay_actions, game_log.jsonl, replay.js, WORKING_REPLAY, AWBW-Replay-Player, diffing or validating zip output.
---

# AWBW Replay System — Architecture

Replay work spans **two independent surfaces** in this repo plus **vendor viewer implementation**. Do not conflate them.

## 1. In-repo web replay (training / debugging)

**Purpose:** Step through games produced by the RL stack without leaving the browser.

- **Data:** `data/game_log.jsonl` — one JSON object per completed game (append model in `rl/env.py::_append_game_log_line`). Lines may be separated by blank lines; stable indexing skips empties (`server/routes/replay.py::_load_game_records`).
- **Optional frames:** Set env `AWBW_LOG_REPLAY_FRAMES=1` so each log record includes a `frames` array (per-engine-step snapshots). Default is off due to log size.
- **Server:** Flask blueprint `replay` — `GET /replay`, `/replay/<game_idx>`, `/replay/api/<game_idx>` (`server/routes/replay.py`). `DATA_DIR` is `ROOT / "data"` (`server/app.py`).
- **Client:** `server/static/replay.js` + `server/templates/replay.html`. Expects API payload with `board` plus optional `frames` (see header comment in `replay.js`).

This path is **JSON**, not the AWBW Replay Player `.zip` format.

## 2. AWBW Replay Player (.zip) — external desktop viewer

**Purpose:** Play replays in the same ecosystem as live AWBW (PHP-serialized `awbwGame` snapshots, optional per-action JSON stream).

**Constraint (non-negotiable):** We **do not** fork or change the AWBW Replay Player application itself. The vendor tree under `third_party/AWBW-Replay-Player/` is **read-only reference** for understanding parsers and the on-disk format. All fixes land in **our** pipeline — engine, trace, and `tools/export_awbw_replay*.py` — so output zips **conform to the existing shipped viewer**. Treat parser behavior as law; align exports with it and with **`WORKING_REPLAY.zip`**.

- **Vendor source (we have it):** `third_party/AWBW-Replay-Player/` — C# codebase. Authoritative parse/render types include `AWBWApp.Game/API/Replay/ReplayData.cs`, `AWBWJsonReplayParser.cs`, `AWBWXmlReplayParser.cs`, `ReplayController.cs`. Read these when inferring required fields or parser behavior.
- **Our generators (contrast with “having the viewer”):** Python tools under `tools/` build zips from `GameState` / `full_trace`:
  - `tools/export_awbw_replay.py` — writes the gzip-compressed snapshot stream (`<game_id>` entry: lines starting with `O:8:"awbwGame":`). Older/minimal path omitted the action stream; viewer still loads turn snapshots.
  - `tools/export_awbw_replay_actions.py` — builds the `a<game_id>` entry with the `p:` action envelopes for animation; documents limitations (e.g. which action types are fully emitted vs snapshot-sync only).
- **Diagnostics:** `tools/diff_replay_zips.py`, `tools/compare_awbw_replays.py`, `tools/deep_diff_replays.py`, `tools/inspect_replay.py`, `tools/validate_new_replay.py` — use when reconciling bytes, positions, or PHP layout.

Ground-truth zip semantics, serialization details, and engine invariants live in the sibling skill **[awbw-engine](../awbw-engine/SKILL.md)** (and `reference.md` there for line-by-line layout).

## 3. Golden reference: `WORKING_REPLAY.zip`

**Role:** A **known-good** AWBW Replay Player zip (filename **`WORKING_REPLAY.zip`**) to **contrast** against output from **our** replay generator (`export_awbw_replay*.py`). Use it when:

- Diffing structure, gzip members, or first-line PHP shapes.
- Proving whether a bug is in exporter logic vs viewer expectations.
- Regression-testing after exporter or engine changes.

If the file is not in the working tree, obtain or place it where your tooling expects (commonly repo root or `replays/`). Prefer the **same viewer build** you use for manual playback when judging correctness.

## Agent checklist

1. Identify which surface you are changing: **JSONL web replay** vs **zip for AWBW Replay Player**.
2. For the desktop viewer: assume the **viewer is fixed** — adjust exporters/engine output until the zip matches what the player accepts; do not plan C# patches unless the user explicitly overrides this.
3. For zip/export issues, read vendor parsers under `third_party/AWBW-Replay-Player/` and cross-check with **`WORKING_REPLAY.zip`** before blaming the engine alone.
4. For engine/action/trace invariants, follow **[awbw-engine](../awbw-engine/SKILL.md)** — this skill does not duplicate those rules.
