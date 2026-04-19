---
name: awbw-replay-system
description: Describes how AWBW replay is split between the in-repo web viewer (game_log.jsonl + /replay/), the zip export pipeline (tools/export_awbw_replay*) compatible with upstream AWBW Replay Player format, and WORKING_REPLAY.zip as a ground-truth fixture. Upstream C# viewer is referenced on GitHub only — not vendored in this repo. Use when discussing replays, replay viewers, viewer compatibility, export_awbw_replay, export_awbw_replay_actions, game_log.jsonl, replay.js, WORKING_REPLAY, diffing or validating zip output.
---

# AWBW Replay System — Architecture

Replay work spans **two independent surfaces** in this repo, plus the **upstream** AWBW Replay Player format on GitHub. Do not conflate them.

## 1. In-repo web replay (training / debugging) — primary UI

**Purpose:** Step through games produced by the RL stack without leaving the browser.

- **Data:** `data/game_log.jsonl` — one JSON object per completed game (append model in `rl/env.py::_append_game_log_line`). Lines may be separated by blank lines; stable indexing skips empties (`server/routes/replay.py::_load_game_records`).
- **Optional frames:** Set env `AWBW_LOG_REPLAY_FRAMES=1` so each log record includes a `frames` array (per-engine-step snapshots). Default is off due to log size.
- **Server:** Flask blueprint `replay` — `GET /replay`, `/replay/<game_idx>`, `/replay/api/<game_idx>` (`server/routes/replay.py`). `DATA_DIR` is `ROOT / "data"` (`server/app.py`).
- **Client:** `server/static/replay.js` + `server/templates/replay.html`. Expects API payload with `board` plus optional `frames` (see header comment in `replay.js`).

This path is **JSON**, not the AWBW Replay Player `.zip` format.

## 2. AWBW-compatible `.zip` — exporters + upstream format (no vendored viewer)

**Purpose:** Emit replays that external tooling (same ecosystem as live AWBW: PHP-serialized `awbwGame` snapshots, optional per-action `p:` stream) can consume.

**We do not ship or maintain the C# AWBW Replay Player.** Parser and type reference code live on GitHub ([DeamonHunter/AWBW-Replay-Player](https://github.com/DeamonHunter/AWBW-Replay-Player)). Align our zip output with that format and with **`WORKING_REPLAY.zip`**. All fixes land in **our** pipeline — engine, trace, and `tools/export_awbw_replay*.py`.

- **Our generators:** Python tools under `tools/` build zips from `GameState` / `full_trace`:
  - `tools/export_awbw_replay.py` — gzip-compressed snapshot stream (`<game_id>` entry: lines starting with `O:8:"awbwGame":`).
  - `tools/export_awbw_replay_actions.py` — `a<game_id>` entry with `p:` action envelopes; documents limitations (e.g. which action types are fully emitted vs snapshot-sync only).
- **Diagnostics:** `tools/diff_replay_zips.py`, `tools/compare_awbw_replays.py`, `tools/deep_diff_replays.py`, `tools/inspect_replay.py`, `tools/validate_new_replay.py` — use when reconciling bytes, positions, or PHP layout.

Ground-truth zip semantics, serialization details, and engine invariants live in the sibling skill **[awbw-engine](../awbw-engine/SKILL.md)** (and `reference.md` there for line-by-line layout).

## 3. Golden reference: `WORKING_REPLAY.zip`

**Role:** A **known-good** AWBW Replay Player zip (filename **`WORKING_REPLAY.zip`**) to **contrast** against output from **our** replay generator (`export_awbw_replay*.py`). Use it when:

- Diffing structure, gzip members, or first-line PHP shapes.
- Proving whether a bug is in exporter logic vs upstream expectations.
- Regression-testing after exporter or engine changes.

If the file is not in the working tree, obtain or place it where your tooling expects (commonly repo root or `replays/`).

## Agent checklist

1. Identify which surface you are changing: **JSONL web replay** vs **zip export**.
2. For zip compatibility: treat the **upstream GitHub** C# sources as the format reference; adjust our exporters/engine output — do not add a forked viewer to this repo unless the user explicitly demands it.
3. For zip/export issues, read the relevant parsers on GitHub and cross-check with **`WORKING_REPLAY.zip`** before blaming the engine alone.
4. For engine/action/trace invariants, follow **[awbw-engine](../awbw-engine/SKILL.md)** — this skill does not duplicate those rules.
