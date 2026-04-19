"""
AI vs AI self-play runner.

Usage
-----
  python -m rl.ai_vs_ai                         # default map, latest checkpoint
  python -m rl.ai_vs_ai --map-id 98             # specific map
  python -m rl.ai_vs_ai --ckpt checkpoints/latest.zip
  python -m rl.ai_vs_ai --co0 1 --co1 7        # CO IDs
  python -m rl.ai_vs_ai --random                # force random vs random
  python -m rl.ai_vs_ai --no-open               # don't launch replay viewer

Decision rule
-------------
1. Load checkpoints/latest.zip (or --ckpt path) as a MaskablePPO for BOTH players.
2. If loading fails: fall back to uniform-random legal actions for both sides.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so we can import engine/rl modules
# ---------------------------------------------------------------------------
_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from engine.action import Action, ActionType, get_legal_actions
from engine.game import GameState, make_initial_state
from engine.map_loader import load_map
from rl.env import _action_to_flat, _flat_to_action, _get_action_mask
from rl.encoder import encode_state

_MAP_POOL_PATH = _REPO / "data" / "gl_map_pool.json"
_MAPS_DIR      = _REPO / "data" / "maps"
_CKPT_DEFAULT  = _REPO / "checkpoints" / "latest.zip"
_REPLAY_OUT    = _REPO / "replays"

# ---------------------------------------------------------------------------
# AWBW Replay Player installation helpers (Windows)
# ---------------------------------------------------------------------------

# Local dev build (CLI opens replay zip automatically) — prefer over packaged installs
_DEV_DESKTOP = _REPO / "AWBW-Replay-Player" / "AWBWApp.Desktop" / "bin"
_DEV_BUILD_EXE = [
    _DEV_DESKTOP / "Release" / "net6.0" / "AWBW Replay Player.exe",
    _DEV_DESKTOP / "Debug" / "net6.0" / "AWBW Replay Player.exe",
]
# Clowd.Squirrel default install path on Windows
_SQUIRREL_INSTALL = Path(os.environ.get("LOCALAPPDATA", "")) / "AWBWReplayPlayer"
# Portable extraction from nupkg (local to the repo)
_PORTABLE_EXE = _REPO / "tools" / "awbw-player" / "lib" / "native" / "AWBW Replay Player.exe"
_APP_EXE_CANDIDATES = [
    *_DEV_BUILD_EXE,
    _PORTABLE_EXE,
    _SQUIRREL_INSTALL / "current" / "AWBWApp.Desktop.exe",
    _SQUIRREL_INSTALL / "AWBWApp.Desktop.exe",
    Path(os.environ.get("APPDATA", "")) / "AWBWApp" / "AWBWApp.Desktop.exe",
]

_INSTALLER_URL = (
    "https://github.com/DeamonHunter/AWBW-Replay-Player/releases/download"
    "/v0.13.1/AWBWReplayPlayerInstaller.exe"
)

# osu-framework storage root on Windows: %APPDATA%\AWBWApp or %APPDATA%\AWBW Replay Player
_APPDATA_ROOT  = Path(os.environ.get("APPDATA", "")) / "AWBWApp"
_APPDATA_ROOT2 = Path(os.environ.get("APPDATA", "")) / "AWBW Replay Player"
# Portable: data sibling to the nupkg exe
_PORTABLE_DATA = _REPO / "tools" / "awbw-player" / "lib" / "native"

_REPLAY_DIR_CANDIDATES = [
    _APPDATA_ROOT  / "ReplayData" / "Replays",
    _APPDATA_ROOT2 / "ReplayData" / "Replays",
    _PORTABLE_DATA / "ReplayData" / "Replays",
]
# Primary = first AppData candidate (create if needed)
_REPLAY_DIR = _REPLAY_DIR_CANDIDATES[0]

# Progress: log every N actions during play (verbose but bounded).
_ACTION_PROGRESS_INTERVAL = 500

# Sentinel raised by _choose_action when the engine reports an active player
# with no legal actions while state.done is still False. Keeping it as a
# module-level constant so the error-handling path in run_game can recognise
# the exact message without drifting out of sync.
_NO_LEGAL_ACTIONS_MSG = "No legal actions -- game should be over."

# Fuses against runaway loops. The engine is designed to terminate each
# player-turn in O(units * stages) steps; anything well above that strongly
# suggests a desync between ``get_legal_actions`` and ``step`` (see
# ``_dump_partial_replay_on_failure``'s ``fuse`` branch for the diagnostic
# dump we produce when these trip).
_DEFAULT_MAX_TOTAL_ACTIONS            = 200_000
_DEFAULT_MAX_ACTIONS_PER_ACTIVE_TURN  = 20_000
_FUSE_TOTAL_MSG      = "Action fuse tripped -- max-total-actions exceeded."
_FUSE_PER_TURN_MSG   = "Action fuse tripped -- max-actions-per-active-turn exceeded."
_FUSE_MESSAGES       = (_FUSE_TOTAL_MSG, _FUSE_PER_TURN_MSG)


def _log(msg: str) -> None:
    """UTC wall-clock timestamp on every line (ISO-8601 with ms)."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    print(f"[ai_vs_ai] {ts} | {msg}", flush=True)


def _find_exe() -> Optional[Path]:
    for p in _APP_EXE_CANDIDATES:
        if p.exists():
            return p
    return None


def _install_replay_player() -> Optional[Path]:
    """Download and run the AWBW Replay Player installer. Returns exe path if successful."""
    installer = _REPO / "tools" / "AWBWReplayPlayerInstaller.exe"
    if not installer.exists():
        _log(f"installer: downloading AWBW Replay Player -> {installer}")
        try:
            urllib.request.urlretrieve(_INSTALLER_URL, str(installer))
            _log(f"installer: downloaded to {installer}")
        except Exception as e:
            _log(f"installer: download failed: {e}")
            return None

    _log("installer: running (a window may open)")
    try:
        subprocess.run([str(installer)], check=False)
    except Exception as e:
        _log(f"installer: error: {e}")
        return None

    # Give Squirrel a moment to finish
    time.sleep(5)
    return _find_exe()


def _ensure_player_installed() -> Optional[Path]:
    exe = _find_exe()
    if exe:
        return exe
    _log("viewer: AWBW Replay Player not found — attempting install")
    return _install_replay_player()


def open_in_replay_player(replay_zip: Path, game_id: int) -> None:
    """Copy replay into the app data dir and launch the viewer."""
    import shutil

    exe = _ensure_player_installed()

    # Copy to every known candidate replay dir (we don't know which one the app uses
    # until it runs at least once and creates its own dir structure)
    dest_path: Optional[Path] = None
    for candidate in _REPLAY_DIR_CANDIDATES:
        candidate.mkdir(parents=True, exist_ok=True)
        dest = candidate / f"{game_id}.zip"
        try:
            shutil.copy2(str(replay_zip), str(dest))
            dest_path = dest
            _log(f"viewer: replay placed -> {dest}")
        except Exception as e:
            _log(f"viewer: could not copy to {dest}: {e}")

    if exe is None:
        _log(
            "viewer: executable not found — install from "
            f"{_INSTALLER_URL}  then drag-drop the zip onto the app"
        )
        return

    _log(f"viewer: launching {exe}")
    try:
        # Pass the replay path as a CLI arg; osu!framework picks it up via PresentFile
        subprocess.Popen([str(exe), str(replay_zip)])
        _log(
            "viewer: process started with replay path "
            f"(if it does not open, drag-drop {replay_zip} onto the window)"
        )
    except Exception as e:
        _log(f"viewer: launch failed: {e} — open manually and drag-drop {replay_zip}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(ckpt_path: Path):
    """Load a MaskablePPO checkpoint. Returns None on failure."""
    try:
        from sb3_contrib import MaskablePPO  # type: ignore[import]
        model = MaskablePPO.load(str(ckpt_path), device="cpu")
        _log(f"checkpoint: loaded MaskablePPO from {ckpt_path}")
        return model
    except Exception as exc:
        _log(f"checkpoint: could not load {ckpt_path}: {exc} — using random policy")
        return None


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def _obs_from_state(state: GameState) -> dict:
    spatial, scalars = encode_state(state)
    return {"spatial": spatial, "scalars": scalars}


def _choose_action(state: GameState, model, rng: random.Random) -> Action:
    """
    Pick an action for the current player.
    Uses *model* (MaskablePPO) if provided, else uniform random over legal actions.
    """
    legal = get_legal_actions(state)
    if not legal:
        raise RuntimeError(_NO_LEGAL_ACTIONS_MSG)

    if model is None:
        return rng.choice(legal)

    obs  = _obs_from_state(state)
    mask = _get_action_mask(state)
    try:
        idx, _ = model.predict(obs, action_masks=mask, deterministic=False)
        action = _flat_to_action(int(idx), state)
        if action is None:
            return rng.choice(legal)
        return action
    except Exception:
        return rng.choice(legal)


# ---------------------------------------------------------------------------
# Map sampling
# ---------------------------------------------------------------------------

def _sample_map_id(rng: random.Random) -> int:
    with open(_MAP_POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    std_maps = [m["map_id"] for m in pool if m.get("type") == "std"]
    candidates = std_maps if std_maps else [m["map_id"] for m in pool]
    return rng.choice(candidates)


def _sample_co(rng: random.Random, map_id: int, tier: str, player: int) -> int:
    """Sample a random legal CO for the given player and tier."""
    try:
        with open(_MAP_POOL_PATH, encoding="utf-8") as f:
            pool = json.load(f)
        meta = next((m for m in pool if m["map_id"] == map_id), None)
        if meta:
            for t in meta.get("tiers", []):
                if t["tier_name"] == tier and t["enabled"] and t["co_ids"]:
                    return rng.choice(t["co_ids"])
    except Exception:
        pass
    # Fallback: Andy=1 vs Grit=7
    return [1, 7][player]


# ---------------------------------------------------------------------------
# Core game loop
# ---------------------------------------------------------------------------

def run_game(
    map_id: Optional[int] = None,
    ckpt_path: Optional[Path] = None,
    co0: Optional[int] = None,
    co1: Optional[int] = None,
    tier: str = "T2",
    seed: Optional[int] = None,
    max_turns: int = 100,
    force_random: bool = False,
    open_viewer: bool = True,
    output_dir: Optional[Path] = None,
    game_id: Optional[int] = None,
    max_total_actions: int = _DEFAULT_MAX_TOTAL_ACTIONS,
    max_actions_per_active_turn: int = _DEFAULT_MAX_ACTIONS_PER_ACTIVE_TURN,
) -> Path:
    """
    Run one AI vs AI game and export an AWBW Replay Player–compatible zip.
    Returns the path to the created .zip file.

    Turn / day semantics
    --------------------
    ``state.turn`` is the AWBW "day" counter. It advances **only** when
    ``P1`` calls ``END_TURN`` (see ``engine/game.py:_end_turn`` — the
    ``if opponent == 0`` branch). Consequently, heartbeat lines that show
    ``day=N`` with ``active=P0`` mean "P0 is still playing their slice of
    day N" — P0 has not yet issued ``END_TURN``. The day only ticks to
    ``N+1`` after P1 finishes their own slice of day N. A stalled heartbeat
    where **both** ``day`` and ``active_player`` stay constant across many
    hundreds of actions is a signal that the per-turn fuse should trip.
    """
    rng = random.Random(seed)

    _log(
        f"session: start  seed={seed!r}  max_turns={max_turns}  "
        f"force_random={force_random}  map_id={map_id!r}"
    )

    # ---- Map ----
    if map_id is None:
        map_id = _sample_map_id(rng)
        _log(f"map: sampled map_id={map_id}")
    _log(f"map: loading id={map_id}")
    map_data = load_map(map_id, _MAP_POOL_PATH, _MAPS_DIR)
    _log(
        f"map: {map_data.name}  id={map_id}  size={map_data.height}x{map_data.width}  "
        f"tiles={map_data.height * map_data.width}"
    )

    # ---- COs ----
    if co0 is None:
        co0 = _sample_co(rng, map_id, tier, 0)
    if co1 is None:
        co1 = _sample_co(rng, map_id, tier, 1)
    _log(f"COs: P0={co0}  P1={co1}  tier={tier}")

    # ---- Checkpoint ----
    model = None
    if not force_random:
        path = ckpt_path or _CKPT_DEFAULT
        if path.exists():
            model = _load_model(path)
        else:
            _log(f"checkpoint: not found at {path} — using random policy")

    if model is None:
        _log("policy: uniform random over legal actions")
    else:
        _log("policy: MaskablePPO (same checkpoint for both players, CPU)")

    # ---- Initial state ----
    _log("engine: building initial GameState")
    state = make_initial_state(map_data, co0, co1, starting_funds=0, tier_name=tier)
    _log(
        f"engine: ready  active_player=P{state.active_player}  day={state.turn}  "
        f"done={state.done}"
    )

    # ---- Output paths (resolved up front so error handler can also use them) ----
    gid = game_id or (int(time.time()) % 999000 + 1000)
    out_dir = output_dir or _REPLAY_OUT
    out_path = Path(out_dir) / f"{gid}.zip"
    start_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # ---- Snapshot collection ----
    # Take one snapshot at the start of EACH player-turn (after END_TURN processes)
    snapshots: list[GameState] = []

    def _snap(s: GameState) -> GameState:
        """Deep-copy the state for archival."""
        return copy.deepcopy(s)

    # First snapshot: P0 start of turn 1
    snapshots.append(_snap(state))
    _log(f"snapshots: initial turn-start snapshot (total {len(snapshots)})")

    # ---- Main loop ----
    action_count = 0
    # Per-active-turn action counter. Resets whenever ``active_player`` flips,
    # so a stuck single player-turn can trip the per-turn fuse without being
    # masked by a long game that nonetheless makes forward progress day to day.
    turn_action_count = 0
    t_start = time.monotonic()
    prev_day = state.turn

    _log(
        f"play: loop start  max_turns={max_turns}  "
        f"max_total_actions={max_total_actions}  "
        f"max_actions_per_active_turn={max_actions_per_active_turn}  "
        f"progress every {_ACTION_PROGRESS_INTERVAL} actions + on day change"
    )

    try:
        while not state.done:
            if state.turn > max_turns:
                _log(f"play: hit max_turns={max_turns} (day={state.turn}) — stopping")
                break

            # Fuse checks are issued *before* we pick the next action so the
            # partial-dump path sees the state at the point we gave up.
            if action_count >= max_total_actions:
                raise RuntimeError(_FUSE_TOTAL_MSG)
            if turn_action_count >= max_actions_per_active_turn:
                raise RuntimeError(_FUSE_PER_TURN_MSG)

            action = _choose_action(state, model, rng)
            prev_player = state.active_player
            state, _reward, _done = state.step(action)
            action_count += 1
            turn_action_count += 1

            if action_count % _ACTION_PROGRESS_INTERVAL == 0:
                lap  = time.monotonic() - t_start
                rate = action_count / lap if lap > 0 else 0.0
                diag = _turn_diagnostics(state)
                _log(
                    f"play: heartbeat  actions={action_count}  "
                    f"turn_actions={turn_action_count}  "
                    f"day={state.turn}  active=P{state.active_player}  "
                    f"stage={diag['stage']}  unmoved={diag['unmoved']}  "
                    f"legal={diag['legal']}  "
                    f"elapsed={lap:.1f}s  ~{rate:.1f} actions/s"
                )

            if state.turn != prev_day:
                lap = time.monotonic() - t_start
                _log(
                    f"play: day -> {state.turn}  actions_so_far={action_count}  "
                    f"elapsed={lap:.1f}s  active=P{state.active_player}"
                )
                prev_day = state.turn

            # Detect turn boundary: active player changed -> start of a new player-turn.
            # Reset the per-turn counter here so the fuse measures contiguous
            # activity under a single active player.
            if state.active_player != prev_player:
                turn_action_count = 0
                if not state.done:
                    snapshots.append(_snap(state))
                    _log(f"snapshots: +1 turn-start (total {len(snapshots)})")
    except RuntimeError as exc:
        msg = str(exc)
        if msg != _NO_LEGAL_ACTIONS_MSG and msg not in _FUSE_MESSAGES:
            raise
        reason = (
            "no_legal_actions" if msg == _NO_LEGAL_ACTIONS_MSG
            else ("fuse_total" if msg == _FUSE_TOTAL_MSG else "fuse_per_turn")
        )
        _log(
            f"play: FAILED with '{msg}' [{reason}] at "
            f"day={state.turn} active=P{state.active_player} "
            f"stage={state.action_stage.name} actions={action_count} "
            f"turn_actions={turn_action_count} — dumping partial replay"
        )
        if reason != "no_legal_actions":
            # Fuse trip: print the exact legal-action distribution at the
            # stall point so a follow-up investigation can diff
            # ``get_legal_actions`` against the engine's ``step`` early-returns.
            _log_fuse_diagnostics(state)
        _dump_partial_replay_on_failure(
            exc=exc,
            reason=reason,
            state=state,
            snapshots=snapshots,
            action_count=action_count,
            turn_action_count=turn_action_count,
            gid=gid,
            out_dir=Path(out_dir),
            start_date_str=start_date_str,
            map_id=map_id,
            co0=co0,
            co1=co1,
            tier=tier,
            map_name=map_data.name,
        )
        raise

    elapsed = time.monotonic() - t_start

    # Final snapshot (game-over state)
    snapshots.append(_snap(state))
    _log(f"snapshots: final game-over snapshot (total {len(snapshots)})")

    # ---- Summary ----
    winner_str = (
        f"P{state.winner} wins ({state.win_reason})"
        if state.winner is not None and state.winner >= 0
        else "Draw"
    )
    _log(
        f"play: finished — {winner_str} | day={state.turn} | actions={action_count} | "
        f"snapshots={len(snapshots)}"
    )
    _log(
        f"play: wall time {elapsed:.1f}s  (replay export next; may be much slower than play)"
    )

    # ---- Export replay (heavy: PHP snapshots + gzip + full_trace replay for p: stream) ----
    # gid / out_dir / out_path / start_date_str were resolved before the play loop
    # so the error handler can reuse them.
    game_name = f"AI-vs-AI  {map_data.name}  [{winner_str}]"

    # Copy trace data before the worker runs so the main thread can return a clear
    # play-time line immediately; the worker does not read mutating `state`.
    full_trace_copy = list(state.full_trace)
    trace_record = {
        "map_id": map_id,
        "co0": co0,
        "co1": co1,
        "tier": tier,
        "turns": state.turn,
        "winner": winner_str,
        "win_reason": state.win_reason,
        "n_actions_full_trace": len(state.full_trace),
        "n_actions_game_log": len(state.game_log),
        "full_trace": full_trace_copy,
        "game_log": list(state.game_log),
    }

    export_error: list[BaseException | None] = [None]

    def _export_replay_worker() -> None:
        t_export = time.monotonic()
        try:
            _log(
                f"export: begin  game_id={gid}  out={out_path}  "
                f"snapshots={len(snapshots)}  full_trace_actions={len(full_trace_copy)}"
            )
            from tools.export_awbw_replay import write_awbw_replay

            t_zip = time.monotonic()
            write_awbw_replay(
                snapshots=snapshots,
                output_path=out_path,
                game_id=gid,
                game_name=game_name,
                start_date=start_date_str,
                full_trace=full_trace_copy,
            )
            _log(f"export: zip + p: stream done in {time.monotonic() - t_zip:.1f}s -> {out_path}")
            trace_path = out_path.with_suffix(".trace.json")
            t_tr = time.monotonic()
            _write_trace_record(trace_record, trace_path)
            _log(f"export: trace JSON written in {time.monotonic() - t_tr:.1f}s -> {trace_path}")
            dt = time.monotonic() - t_export
            _log(f"export: complete  total {dt:.1f}s")
        except Exception as exc:
            export_error[0] = exc
            _log(f"export: FAILED — {exc!r}")

    _log("export: worker thread starting (main thread will join when done)")
    worker = threading.Thread(
        target=_export_replay_worker,
        name="ai_vs_ai-replay-export",
        daemon=False,
    )
    worker.start()
    worker.join()
    if export_error[0] is not None:
        raise export_error[0]

    _log(f"session: replay + trace ready  path={out_path}")

    # ---- Open viewer ----
    if open_viewer:
        _log(f"viewer: opening (game_id={gid})")
        open_in_replay_player(out_path, gid)
    else:
        _log("viewer: skipped (--no-open)")

    _log(f"session: exit ok  {out_path}")
    return out_path


def _write_trace_record(record: dict, path: Path) -> None:
    """Write a pre-built trace dict to JSON (used by replay export worker)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def _turn_diagnostics(state: GameState) -> dict:
    """Cheap per-heartbeat snapshot of the quantities most useful for
    diagnosing a stuck player-turn.

    ``unmoved`` counts the active player's units that have not yet acted this
    turn. Because ``END_TURN`` only becomes legal once ``unmoved == 0`` (see
    ``engine/action.py:_get_select_actions``), a heartbeat that shows a large
    constant ``unmoved`` while actions keep climbing is a strong hint that
    either (a) an action masked as legal is no-opping in ``step``, or
    (b) the policy is thrashing between SELECT and MOVE without committing a
    terminator. Either way the per-turn fuse will eventually trip.
    """
    try:
        legal_n = len(get_legal_actions(state))
    except Exception:
        legal_n = -1
    unmoved = sum(1 for u in state.units[state.active_player] if not u.moved)
    return {
        "stage":   state.action_stage.name,
        "unmoved": unmoved,
        "legal":   legal_n,
        "units":   len(state.units[state.active_player]),
    }


def _log_fuse_diagnostics(state: GameState) -> None:
    """Dump the legal-action distribution at a fuse trip.

    Prints a small histogram of ``ActionType`` counts from
    ``get_legal_actions(state)`` plus the per-turn ``moved`` tally. This is
    the investigation hook for the plan's ``investigate-noop`` todo: when
    the fuse fires we want enough context to diff against the engine's
    ``step`` handlers without having to rerun the whole game.
    """
    try:
        legal = get_legal_actions(state)
    except Exception as exc:
        _log(f"fuse-diag: get_legal_actions raised {exc!r}")
        return

    counts: dict[str, int] = {}
    for a in legal:
        counts[a.action_type.name] = counts.get(a.action_type.name, 0) + 1
    hist = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "<empty>"
    unmoved = [u for u in state.units[state.active_player] if not u.moved]
    _log(
        f"fuse-diag: stage={state.action_stage.name}  "
        f"active=P{state.active_player}  "
        f"legal={len(legal)}  by_type=[{hist}]  "
        f"unmoved={len(unmoved)}/{len(state.units[state.active_player])}"
    )
    if state.selected_unit is not None:
        _log(
            f"fuse-diag: selected_unit={state.selected_unit.unit_type.name} "
            f"at {state.selected_unit.pos}  "
            f"selected_move_pos={state.selected_move_pos}"
        )
    if unmoved:
        preview = ", ".join(
            f"{u.unit_type.name}@{u.pos}" for u in unmoved[:8]
        )
        suffix = "" if len(unmoved) <= 8 else f" (+{len(unmoved) - 8} more)"
        _log(f"fuse-diag: unmoved_units=[{preview}]{suffix}")


def _dump_partial_replay_on_failure(
    *,
    exc: BaseException,
    reason: str,
    state: GameState,
    snapshots: list[GameState],
    action_count: int,
    turn_action_count: int,
    gid: int,
    out_dir: Path,
    start_date_str: str,
    map_id: int,
    co0: int,
    co1: int,
    tier: str,
    map_name: str,
) -> None:
    """Best-effort dump of an in-flight game that crashed mid-loop.

    Writes a ``{gid}.partial.trace.json`` alongside a ``{gid}.partial.zip``
    built from the snapshots gathered so far plus ``state.full_trace``.
    If the zip export itself throws, we retry without the per-action stream
    so the user still gets a turn-snapshot-only replay for manual inspection.
    Never raises — the caller is expected to re-raise the original exception.

    ``reason`` distinguishes the failure mode (``no_legal_actions``,
    ``fuse_total``, ``fuse_per_turn``) so downstream tooling can filter.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"{gid}.partial.trace.json"
    zip_path   = out_dir / f"{gid}.partial.zip"

    full_trace_copy = list(state.full_trace)
    game_log_copy   = list(state.game_log)

    diagnostics = {
        "active_player":     state.active_player,
        "turn":              state.turn,
        "done":              state.done,
        "action_stage":      state.action_stage.name,
        "winner":            state.winner,
        "win_reason":        state.win_reason,
        "n_snapshots":       len(snapshots),
        "action_count":      action_count,
        "turn_action_count": turn_action_count,
        "reason":            reason,
    }

    winner_label = {
        "no_legal_actions": "PARTIAL (no legal actions)",
        "fuse_total":       "PARTIAL (action fuse: total)",
        "fuse_per_turn":    "PARTIAL (action fuse: per-turn)",
    }.get(reason, "PARTIAL")

    record = {
        "map_id":               map_id,
        "co0":                  co0,
        "co1":                  co1,
        "tier":                 tier,
        "turns":                state.turn,
        "winner":               winner_label,
        "win_reason":           state.win_reason,
        "n_actions_full_trace": len(full_trace_copy),
        "n_actions_game_log":   len(game_log_copy),
        "full_trace":           full_trace_copy,
        "game_log":             game_log_copy,
        "partial":              True,
        "error":                str(exc),
        "diagnostics":          diagnostics,
    }

    try:
        _write_trace_record(record, trace_path)
        _log(f"partial: trace JSON written -> {trace_path.resolve()}")
    except Exception as trace_exc:
        _log(f"partial: trace JSON write FAILED — {trace_exc!r}")

    reason_tag = {
        "no_legal_actions": "no legal actions",
        "fuse_total":       "action fuse: total",
        "fuse_per_turn":    "action fuse: per-turn",
    }.get(reason, reason)
    partial_game_name = f"AI-vs-AI  {map_name}  [PARTIAL — {reason_tag}]"

    try:
        from tools.export_awbw_replay import write_awbw_replay

        write_awbw_replay(
            snapshots=snapshots,
            output_path=zip_path,
            game_id=gid,
            game_name=partial_game_name,
            start_date=start_date_str,
            full_trace=full_trace_copy,
        )
        _log(f"partial: replay zip (with p: stream) -> {zip_path.resolve()}")
    except Exception as zip_exc:
        _log(
            f"partial: zip export with full_trace FAILED — {zip_exc!r} — "
            f"retrying snapshot-only"
        )
        try:
            from tools.export_awbw_replay import write_awbw_replay

            write_awbw_replay(
                snapshots=snapshots,
                output_path=zip_path,
                game_id=gid,
                game_name=partial_game_name,
                start_date=start_date_str,
                full_trace=None,
            )
            _log(f"partial: replay zip (snapshot-only fallback) -> {zip_path.resolve()}")
        except Exception as zip_exc2:
            _log(f"partial: snapshot-only zip export also FAILED — {zip_exc2!r}")


def _save_trace(
    state: GameState,
    path: Path,
    map_id: int,
    co0: int,
    co1: int,
    tier: str,
    winner_str: str,
) -> None:
    """Write a JSON debug trace including the full action log."""
    record = {
        "map_id": map_id,
        "co0": co0,
        "co1": co1,
        "tier": tier,
        "turns": state.turn,
        "winner": winner_str,
        "win_reason": state.win_reason,
        "n_actions_full_trace": len(state.full_trace),
        "n_actions_game_log": len(state.game_log),
        "full_trace": state.full_trace,
        "game_log": state.game_log,
    }
    _write_trace_record(record, path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one AI vs AI game and export an AWBW Replay Player–compatible replay."
    )
    parser.add_argument("--map-id",   type=int,  default=None,
                        help="AWBW map ID to play on (default: random from pool)")
    parser.add_argument("--ckpt",     type=Path, default=None,
                        help=f"Checkpoint path (default: {_CKPT_DEFAULT})")
    parser.add_argument("--co0",      type=int,  default=None, help="P0 CO id")
    parser.add_argument("--co1",      type=int,  default=None, help="P1 CO id")
    parser.add_argument("--tier",     type=str,  default="T2", help="Tier name (default: T2)")
    parser.add_argument("--seed",     type=int,  default=None, help="RNG seed")
    parser.add_argument("--max-turns",type=int,  default=100,  help="Turn limit (default: 100)")
    parser.add_argument("--random",   action="store_true",
                        help="Force random-vs-random (ignore checkpoint)")
    parser.add_argument("--no-open",  action="store_true",
                        help="Do not launch the replay viewer after the game")
    parser.add_argument("--out-dir",  type=Path, default=None,
                        help=f"Directory for output files (default: {_REPLAY_OUT})")
    parser.add_argument("--game-id",  type=int,  default=None,
                        help="Replay file ID (numeric filename stem)")
    parser.add_argument(
        "--max-total-actions", type=int,
        default=_DEFAULT_MAX_TOTAL_ACTIONS,
        help=(
            "Abort the game and dump a partial replay if this many actions "
            f"are taken across the whole match (default: {_DEFAULT_MAX_TOTAL_ACTIONS})."
        ),
    )
    parser.add_argument(
        "--max-actions-per-active-turn", type=int,
        default=_DEFAULT_MAX_ACTIONS_PER_ACTIVE_TURN,
        help=(
            "Abort and dump a partial replay if a single active player "
            "consumes this many consecutive actions without ending their turn "
            f"(default: {_DEFAULT_MAX_ACTIONS_PER_ACTIVE_TURN})."
        ),
    )
    args = parser.parse_args()

    run_game(
        map_id=args.map_id,
        ckpt_path=args.ckpt,
        co0=args.co0,
        co1=args.co1,
        tier=args.tier,
        seed=args.seed,
        max_turns=args.max_turns,
        force_random=args.random,
        open_viewer=not args.no_open,
        output_dir=args.out_dir,
        game_id=args.game_id,
        max_total_actions=args.max_total_actions,
        max_actions_per_active_turn=args.max_actions_per_active_turn,
    )


if __name__ == "__main__":
    main()
