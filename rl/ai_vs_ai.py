"""
AI vs AI self-play runner.

Usage
-----
  python -m rl.ai_vs_ai                         # mirror a running train.py (map/tier/COs + ckpt)
  python -m rl.ai_vs_ai --map-id 98             # specific map
  python -m rl.ai_vs_ai --ckpt checkpoints/latest.zip
  python -m rl.ai_vs_ai --co0 1 --co1 7        # CO IDs
  python -m rl.ai_vs_ai --random                # force random vs random
  python -m rl.ai_vs_ai --no-open               # don't open replay output folder
  python -m rl.ai_vs_ai --no-follow-train       # ignore train.py; use defaults below

With **no arguments** after the module name, this process scans for another local
process whose command line contains ``train.py`` in training mode (not
``--watch-only`` / ``--rank`` / ``--features``), parses its CLI with the same
rules as ``train.py``, and samples one episode from the same distribution as
``AWBWEnv`` (including ``--curriculum-broad-prob``). The checkpoint is resolved
like ``SelfPlayTrainer`` startup: ``--checkpoint-dir``, ``--load-promoted`` vs
``latest.zip``. If the trainer uses ``--capture-move-gate``, the same
``AWBW_CAPTURE_MOVE_GATE`` behavior is applied.

After export, the **AWBW Replay Player** desktop app is started with the new
``.zip`` (see ``AWBW_REPLAY_PLAYER_EXE`` and ``third_party/AWBW-Replay-Player``).
If the exe is missing, the replay folder is opened in the file manager.

If no suitable ``train.py`` process is found, falls back to the legacy defaults
(random Std map, tier T2, ``checkpoints/latest.zip``).

Decision rule
-------------
1. Load the resolved checkpoint as a MaskablePPO for BOTH players.
2. If loading fails: fall back to uniform-random legal actions for both sides.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so we can import engine/rl modules
# ---------------------------------------------------------------------------
_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from engine.action import Action, ActionType, get_legal_actions
from engine.game import GameState, make_initial_state
from engine.map_loader import load_map
from rl.env import _action_to_flat, _flat_to_action, _get_action_mask, sample_training_matchup
from rl.encoder import encode_state

_MAP_POOL_PATH = _REPO / "data" / "gl_map_pool.json"
_MAPS_DIR      = _REPO / "data" / "maps"
_CKPT_DEFAULT  = _REPO / "checkpoints" / "latest.zip"
_REPLAY_OUT    = _REPO / "replays"

_TRAIN_PY_TAIL = re.compile(r"train\.py[\"']?$", re.IGNORECASE)


def _log(msg: str) -> None:
    """UTC wall-clock timestamp on every line (ISO-8601 with ms)."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    print(f"[ai_vs_ai] {ts} | {msg}", flush=True)


def _argv_for_this_module() -> list[str]:
    """Arguments after ``python -m rl.ai_vs_ai`` or ``python .../ai_vs_ai.py``."""
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == "-m":
        return a[2:]
    if a:
        return a[1:]
    return []


def _split_argv_after_train_py(argv: list[str]) -> list[str]:
    """Return CLI tokens that belong to ``train.py`` (after the script path)."""
    for i, p in enumerate(argv):
        norm = p.strip().strip('"').strip("'").replace("\\", "/")
        if norm.endswith("/train.py") or norm.endswith("train.py") or _TRAIN_PY_TAIL.search(p):
            return argv[i + 1 :]
    return []


def _argv_contains_train_py(parts: list[str]) -> bool:
    for p in parts:
        n = p.strip().strip('"').strip("'").replace("\\", "/")
        if n.endswith("train.py"):
            return True
    return False


def _list_train_py_argv_processes() -> list[tuple[int, list[str]]]:
    """``(pid, argv)`` for local processes whose command line runs ``train.py``."""
    me = os.getpid()
    out: list[tuple[int, list[str]]] = []
    if sys.platform == "win32":
        ps_cmd = (
            "Get-CimInstance Win32_Process | Where-Object { "
            "$_.CommandLine -and ($_.CommandLine -match 'train\\.py') "
            "} | ForEach-Object { "
            "$_.ProcessId.ToString() + [char]9 + $_.CommandLine "
            "}"
        )
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=45,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            _log(f"follow-train: PowerShell process scan failed: {exc}")
            return out
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line or "\t" not in line:
                continue
            pid_s, cmd = line.split("\t", 1)
            try:
                pid = int(pid_s)
            except ValueError:
                continue
            if pid == me:
                continue
            parts = _parse_cmdline_to_argv_windows(cmd)
            out.append((pid, parts))
        return out

    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return out
    for name in os.listdir(proc_root):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == me:
            continue
        cmdline_path = proc_root / name / "cmdline"
        try:
            raw = cmdline_path.read_bytes()
        except OSError:
            continue
        parts = [x.decode("utf-8", errors="replace") for x in raw.split(b"\0") if x]
        if not _argv_contains_train_py(parts):
            continue
        out.append((pid, parts))
    return out


def _parse_cmdline_to_argv_windows(cmd: str) -> list[str]:
    """Best-effort argv split for a WMI ``CommandLine`` string."""
    s = cmd.strip()
    if not s:
        return []
    try:
        return shlex.split(s, posix=False)
    except ValueError:
        return s.split()


def _pick_training_train_argv() -> tuple[list[str], int] | None:
    """
    Find a ``train.py`` process in **training** mode and return ``(argv_tail, pid)``.
    If several match, prefer the highest PID (typically most recently started).
    """
    from train import build_train_argument_parser

    parser = build_train_argument_parser()
    candidates: list[tuple[int, list[str], int]] = []
    for pid, parts in _list_train_py_argv_processes():
        if not _argv_contains_train_py(parts):
            continue
        tail = _split_argv_after_train_py(parts)
        candidates.append((pid, tail, pid))

    if not candidates:
        return None

    training: list[tuple[int, list[str], int]] = []
    for pid, tail, sort_key in candidates:
        try:
            ns, unknown = parser.parse_known_args(tail)
        except SystemExit:
            continue
        if unknown:
            _log(f"follow-train: pid={pid} ignored unknown args: {unknown}")
        if ns.watch_only or ns.rank or ns.features:
            continue
        training.append((pid, tail, sort_key))

    if not training:
        return None

    training.sort(key=lambda x: x[2])
    pid, tail, _ = training[-1]
    return tail, pid


def _resolve_ckpt_from_train_ns(train_ns: Any) -> Path:
    """
    Match ``SelfPlayTrainer`` resume checkpoint selection (``latest.zip`` vs
    ``promoted/best.zip`` when ``--load-promoted``).

    Uses only ``train_ns`` + ``resolve_checkpoint_dir`` — no fleet env validation,
    because ``train.py`` may run in another process with different
    ``AWBW_MACHINE_*`` than this shell.
    """
    from rl.fleet_env import REPO_ROOT, resolve_checkpoint_dir

    checkpoint_dir = resolve_checkpoint_dir(REPO_ROOT, train_ns.checkpoint_dir, None)
    latest_path = checkpoint_dir / "latest.zip"
    promoted_path = checkpoint_dir / "promoted" / "best.zip"
    resume_path = latest_path
    if train_ns.load_promoted and promoted_path.is_file():
        if not latest_path.is_file():
            resume_path = promoted_path
        elif promoted_path.stat().st_mtime > latest_path.stat().st_mtime:
            resume_path = promoted_path
    return resume_path


def _sample_from_train_ns(train_ns: Any, rng: random.Random) -> tuple[int, str, int, int]:
    """One ``(map_id, tier, co0, co1)`` matching the running trainer's env distribution."""
    with open(_MAP_POOL_PATH, encoding="utf-8") as f:
        pool: list[dict] = json.load(f)
    map_pool = pool
    if train_ns.map_id is not None:
        map_pool = [m for m in pool if m["map_id"] == train_ns.map_id]
        if not map_pool:
            raise ValueError(f"follow-train: no map with map_id={train_ns.map_id}")
    _std = [m for m in map_pool if m.get("type") == "std"]
    sample_map_pool = _std if _std else map_pool
    mid, tier, c0, c1, _name = sample_training_matchup(
        sample_map_pool,
        co_p0=train_ns.co_p0,
        co_p1=train_ns.co_p1,
        tier_name=train_ns.tier,
        curriculum_broad_prob=train_ns.curriculum_broad_prob,
        rng=rng,
    )
    return mid, tier, c0, c1


# ---------------------------------------------------------------------------
# Replay export UX: AWBW Replay Player (desktop) + folder fallback
# ---------------------------------------------------------------------------

_CAPTURE_GATE_ENV = "AWBW_CAPTURE_MOVE_GATE"
_REPLAY_PLAYER_EXE_ENV = "AWBW_REPLAY_PLAYER_EXE"


def _log_capture_move_gate_status() -> None:
    """Mirror train.py: log when legal-action mask uses capture move gate."""
    raw = os.environ.get(_CAPTURE_GATE_ENV, "").strip().lower()
    if raw not in ("", "0", "false", "no"):
        _log(f"env: {_CAPTURE_GATE_ENV}={raw!r} (infantry/mech MOVE mask active)")


def _resolve_awbw_replay_player_exe(repo: Path) -> Path | None:
    """
    Locate the desktop AWBW Replay Player (see ``desync-triage-viewer`` §4a).

    Order: ``AWBW_REPLAY_PLAYER_EXE``, then Release/Debug ``net*`` folders under
    ``third_party/AWBW-Replay-Player/AWBWApp.Desktop/bin/``.
    """
    env = os.environ.get(_REPLAY_PLAYER_EXE_ENV, "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p.resolve()
    base = repo / "third_party" / "AWBW-Replay-Player" / "AWBWApp.Desktop" / "bin"
    for cfg in ("Release", "Debug"):
        for tfm in ("net8.0", "net7.0", "net6.0"):
            cand = base / cfg / tfm / "AWBW Replay Player.exe"
            if cand.is_file():
                return cand.resolve()
    for cfg in ("Release", "Debug"):
        d = base / cfg
        if not d.is_dir():
            continue
        for sub in sorted(d.iterdir(), key=lambda x: x.name, reverse=True):
            if sub.is_dir() and sub.name.startswith("net"):
                exe = sub / "AWBW Replay Player.exe"
                if exe.is_file():
                    return exe.resolve()
    return None


def _open_replay_in_desktop_viewer(replay_zip: Path) -> None:
    """
    Start the AWBW Replay Player with the given ``.zip`` (argv: exe + absolute zip).
    Falls back to opening the containing folder if the exe is missing or spawn fails.
    """
    zp = replay_zip.resolve()
    exe = _resolve_awbw_replay_player_exe(_REPO)
    if exe is None:
        _log(
            f"viewer: AWBW Replay Player.exe not found — set {_REPLAY_PLAYER_EXE_ENV} "
            "or build third_party/AWBW-Replay-Player (see README / desync-triage-viewer §4a)"
        )
        _open_replay_output_folder(replay_zip)
        return
    try:
        subprocess.Popen(
            [str(exe), str(zp)],
            cwd=str(exe.parent),
            close_fds=sys.platform != "win32",
        )
        _log(f"viewer: AWBW Replay Player — {exe.name} loaded {zp}")
    except OSError as exc:
        _log(f"viewer: could not start {exe}: {exc}")
        _open_replay_output_folder(replay_zip)


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


def _open_replay_output_folder(replay_zip: Path) -> None:
    """Open the folder holding the exported zip/trace; hint at in-repo web replay."""
    folder = replay_zip.resolve().parent
    _log(f"viewer: (fallback) folder {folder}")
    _log(
        "viewer: training games logged to logs/game_log.jsonl — run `python -m server.app` "
        "and open http://127.0.0.1:5000/replay/"
    )
    try:
        if sys.platform == "win32":
            os.startfile(str(folder))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(folder)], check=False)
        else:
            subprocess.run(["xdg-open", str(folder)], check=False)
    except Exception as exc:
        _log(f"viewer: could not open folder in file manager: {exc}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(ckpt_path: Path):
    """Load a MaskablePPO checkpoint. Returns None on failure."""
    try:
        from rl.ckpt_compat import load_maskable_ppo_compat

        model = load_maskable_ppo_compat(ckpt_path, device="cpu")
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
    capture_move_gate: bool = False,
) -> Path:
    """
    Run one AI vs AI game and export an AWBW Replay Player–compatible zip.
    Returns the path to the created .zip file.

    Legal actions honor ``AWBW_CAPTURE_MOVE_GATE`` (see ``engine.action``): set the
    environment variable before running, or pass ``capture_move_gate=True`` (same
    effect as ``train.py --capture-move-gate``). Follow-train copies a running
    trainer's ``--capture-move-gate`` into the environment.

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

    if capture_move_gate:
        os.environ[_CAPTURE_GATE_ENV] = "1"
    _log_capture_move_gate_status()

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
        if open_viewer:
            pz = Path(out_dir) / f"{gid}.partial.zip"
            if pz.is_file():
                _open_replay_in_desktop_viewer(pz)
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

    # ---- Open viewer (desktop AWBW Replay Player loads the zip) ----
    if open_viewer:
        _log(f"viewer: opening replay (game_id={gid})")
        _open_replay_in_desktop_viewer(out_path)
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
    user = _argv_for_this_module()
    parser = argparse.ArgumentParser(
        description=(
            "Run one AI vs AI game and export a replay zip + trace "
            "(AWBW-compatible zip format for external tooling)."
        )
    )
    parser.add_argument(
        "--no-follow-train",
        action="store_true",
        help=(
            "With no other ai_vs_ai flags, do not scan for a running train.py — "
            "use random Std map, tier T2, and checkpoints/latest.zip."
        ),
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
                        help="Do not launch AWBW Replay Player (or folder fallback) after export")
    parser.add_argument(
        "--capture-move-gate",
        action="store_true",
        help=(
            "Set AWBW_CAPTURE_MOVE_GATE=1 for this run (same legal-action mask as "
            "train.py --capture-move-gate; infantry/mech MOVE restricted when capture tiles reachable)."
        ),
    )
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
    args = parser.parse_args(user)

    want_follow = len(user) == 0 and not args.no_follow_train
    if want_follow:
        picked = _pick_training_train_argv()
        if picked is not None:
            tail, pid = picked
            from train import build_train_argument_parser

            tparser = build_train_argument_parser()
            train_ns, unk = tparser.parse_known_args(tail)
            if unk:
                _log(f"follow-train: pid={pid} ignored unknown train.py args: {unk}")
            _log(
                f"follow-train: matched train.py pid={pid} "
                f"(map_id={train_ns.map_id} tier={train_ns.tier!r} "
                f"co_p0={train_ns.co_p0} co_p1={train_ns.co_p1} "
                f"broad_prob={train_ns.curriculum_broad_prob} "
                f"capture_move_gate={getattr(train_ns, 'capture_move_gate', False)})"
            )
            rng = random.Random(args.seed) if args.seed is not None else random.Random()
            try:
                map_id, tier, co0, co1 = _sample_from_train_ns(train_ns, rng)
            except ValueError as exc:
                _log(f"follow-train: matchup sampling failed ({exc}); using legacy defaults")
            else:
                ckpt_path = _resolve_ckpt_from_train_ns(train_ns)
                _log(f"follow-train: sampled map_id={map_id} tier={tier} co0={co0} co1={co1}")
                _log(f"follow-train: checkpoint -> {ckpt_path}")
                run_game(
                    map_id=map_id,
                    ckpt_path=ckpt_path,
                    co0=co0,
                    co1=co1,
                    tier=tier,
                    seed=args.seed,
                    max_turns=args.max_turns,
                    force_random=args.random,
                    open_viewer=not args.no_open,
                    output_dir=args.out_dir,
                    game_id=args.game_id,
                    max_total_actions=args.max_total_actions,
                    max_actions_per_active_turn=args.max_actions_per_active_turn,
                    capture_move_gate=(
                        args.capture_move_gate
                        or getattr(train_ns, "capture_move_gate", False)
                    ),
                )
                return
        _log(
            "follow-train: no training train.py process found "
            "(or none parseable); using legacy defaults (Std map, T2, latest.zip)"
        )

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
        capture_move_gate=args.capture_move_gate,
    )


if __name__ == "__main__":
    main()
