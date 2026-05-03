"""
AI vs AI self-play runner.

Usage
-----
  python -m rl.ai_vs_ai                         # mirror a running train.py (map/tier/COs + ckpt)
  python -m rl.ai_vs_ai --map-id 98             # train defaults, but force this map
  python -m rl.ai_vs_ai --ckpt checkpoints/latest.zip
  python -m rl.ai_vs_ai --ckpt-p0 checkpoints/latest.zip \\
        --ckpt-p1 Z:/checkpoints/pool/workhorse1/latest.zip   # asymmetric checkpoints
  python -m rl.ai_vs_ai --co0 1 --co1 7        # CO IDs
  python -m rl.ai_vs_ai --random                # force random vs random
  python -m rl.ai_vs_ai --no-open               # don't open replay output folder
  python -m rl.ai_vs_ai --no-follow-train       # never read train.py from memory
  python -m rl.ai_vs_ai --from-latest-export    # newest ``engine_snapshot.pkl`` under
                                                # replays/amarinner_my_games
  python -m rl.ai_vs_ai --no-follow-train --map-id 123858 --tier T4 --co0 14 --co1 14 \\
        --opening-book data/opening_books/std_pool_precombat.jsonl \\
        --capture-move-gate 0.10 --learner-greedy-mix 0.15
``train.py`` process, parses its CLI, and uses that as the **base** run (same
idea as ``SelfPlayTrainer`` / ``AWBWEnv``). **Any** ``--map-id``, ``--tier``,
``--co0`` / ``--co1``, ``--ckpt``, ``--ckpt-p0`` / ``--ckpt-p1``,
``--capture-move-gate``, ``--opening-book`` /
``--opening-book-strict-co``, or ``--learner-greedy-mix`` you add on the
``ai_vs_ai`` command line **overrides** only those fields; everything else still
comes from the live trainer (checkpoint dir, promoted load, curriculum
broad prob, capture gate when not overridden, etc.).

After export, the **AWBW Replay Player** desktop app is started with the new
``.zip`` (see ``rl.paths.REPLAY_PLAYER_EXE_ENV`` and
``rl.paths.REPLAY_PLAYER_THIRD_PARTY_DIR``). If the exe is missing, the replay
folder is opened in the file manager.

If no suitable ``train.py`` process is found, falls back to the legacy defaults
(random Std map, tier T2, checkpoint: ``Z:\\checkpoints\\latest.zip`` when that
file exists, else repo ``checkpoints/latest.zip``).

Decision rule
-------------
1. Load MaskablePPO per seat from ``--ckpt-p0`` / ``--ckpt-p1`` when set;
   otherwise use ``--ckpt`` for both seats (or follow-train / default ``latest.zip``).
   Same resolved path loads one model shared by both seats.
2. If loading fails for a seat: uniform-random legal actions for that seat only.
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
from rl.log_timestamp import log_now_iso, log_now_wall
from pathlib import Path
from typing import Any, Optional

# Windows: without CREATE_NO_WINDOW + hidden STARTUPINFO + detached stdio, GUI or
# .NET children can briefly allocate/inherit a console (black ``cmd`` flash).
_WIN32_CREATE_NO_WINDOW = 0x08000000


def _win32_hidden_subprocess_kwargs() -> dict[str, Any]:
    """Keyword args for ``subprocess.{run,Popen}`` so children do not open a console."""
    if sys.platform != "win32":
        return {}
    flags = int(getattr(subprocess, "CREATE_NO_WINDOW", _WIN32_CREATE_NO_WINDOW))
    out: dict[str, Any] = {"creationflags": flags}
    if hasattr(subprocess, "STARTUPINFO"):
        si = subprocess.STARTUPINFO()
        si.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0x1)
        si.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
        out["startupinfo"] = si
    return out

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so we can import engine/rl modules
# ---------------------------------------------------------------------------
_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from train import _parse_capture_move_gate_probability_cli

from engine.action import Action, ActionType, get_legal_actions
from engine.game import GameState, make_initial_state
from engine.map_loader import MapData, load_map
from rl.env import _action_to_flat, _flat_to_action, _get_action_mask, sample_training_matchup
from rl.encoder import encode_state
from rl.paths import REPLAY_PLAYER_EXE_ENV, REPLAY_PLAYER_THIRD_PARTY_DIR, resolve_awbw_replay_player_exe

_MAP_POOL_PATH = _REPO / "data" / "gl_map_pool.json"
_MAPS_DIR      = _REPO / "data" / "maps"
_CKPT_DEFAULT  = _REPO / "checkpoints" / "latest.zip"
_Z_PREFERRED_CKPT = Path("Z:/checkpoints/latest.zip")
_REPLAY_OUT    = _REPO / "replays"

# From ``engine_snapshot.pkl`` / live export: extend the calendar day cap by this
# many in-game days from the snapshot (simple short validation run).
LIVE_SNAPSHOT_CALENDAR_CAP_DAYS = 10


def _default_ckpt_path() -> Path:
    """Repo ``checkpoints/latest.zip``, unless a file exists at ``Z:/checkpoints/latest.zip``."""
    if _Z_PREFERRED_CKPT.is_file():
        return _Z_PREFERRED_CKPT
    return _CKPT_DEFAULT

_TRAIN_PY_TAIL = re.compile(r"train\.py[\"']?$", re.IGNORECASE)


def _resolve_max_calendar_days(
    max_days: Optional[int],
    max_turns: Optional[int],
    *,
    default: int = 100,
) -> int:
    """Single end-inclusive calendar day cap from optional ``max_days`` / deprecated ``max_turns``."""
    if max_days is not None and max_turns is not None and int(max_days) != int(max_turns):
        raise ValueError("pass only one of max_days and max_turns")
    if max_days is not None:
        return int(max_days)
    if max_turns is not None:
        return int(max_turns)
    return int(default)


class _MaxCalendarDaysAction(argparse.Action):
    """Warn when ``--max-turns`` is used; ``--max-days`` is canonical."""

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string == "--max-turns":
            sys.stderr.write(
                "[ai_vs_ai] --max-turns is deprecated; use --max-days (same end-inclusive calendar cap).\n"
            )
        setattr(namespace, self.dest, int(values))


def _log(msg: str) -> None:
    """US Eastern wall clock on every line (ISO-8601 with ms, America/New_York)."""
    ts = log_now_iso()
    print(f"[ai_vs_ai] {ts} | {msg}", flush=True)


def _is_ai_vs_ai_script_argv_token(arg: str) -> bool:
    """True if *arg* is a path to this file (direct ``python path/to/ai_vs_ai.py``)."""
    base = arg.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower()
    return base == "ai_vs_ai.py"


def _argv_for_this_module() -> list[str]:
    """Arguments meant for :mod:`rl.ai_vs_ai` after the interpreter / launcher tokens.

    Must not pass ``ai_vs_ai.py`` through to :mod:`argparse` (unrecognized argument),
    which previously aborted the run before export — especially with
    ``python -u …/ai_vs_ai.py …`` where the old logic left the script path in *user*.

    ``python -m rl.ai_vs_ai`` does **not** put ``-m`` in ``sys.argv``; CPython sets
    ``sys.argv[0]`` to this module's path and places all flags in ``argv[1:]``.
    Without the ``Path(sys.argv[0]) == __file__`` fast-path, the scan below could
    treat ``--max-days 2``'s ``2`` as a bogus "script" token and return ``[]``,
    wiping out every CLI flag (``--max-days``, ``--max-turns``, ``--random``, …).
    """
    a = sys.argv[1:]
    if not a:
        return []
    if a[0] == "-c":
        return []

    try:
        if Path(sys.argv[0]).resolve() == Path(__file__).resolve():
            return a
    except OSError:
        pass

    for i, arg in enumerate(a):
        if arg == "-m" and i + 1 < len(a):
            return a[i + 2 :]

    for i, arg in enumerate(a):
        if _is_ai_vs_ai_script_argv_token(arg):
            return a[i + 1 :]

    if not a[0].startswith("-"):
        return a[1:]

    i = 0
    while i < len(a) and a[i].startswith("-") and a[i] not in ("-", "--"):
        i += 1
    if i < len(a) and _is_ai_vs_ai_script_argv_token(a[i]):
        return a[i + 1 :]

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
        # Prefer psutil: no PowerShell subprocess, so no extra console window and
        # all logging stays in the shell that launched ai_vs_ai.
        try:
            import psutil
        except ImportError:
            psutil = None  # type: ignore[assignment]
        if psutil is not None:
            for proc in psutil.process_iter(["pid", "cmdline"]):
                try:
                    info = proc.info
                    pid = info.get("pid")
                    if pid is None or pid == me:
                        continue
                    parts = info.get("cmdline")
                    if not parts:
                        continue
                    if not _argv_contains_train_py(parts):
                        continue
                    out.append((pid, list(parts)))
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            return out

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
                stdin=subprocess.DEVNULL,
                **_win32_hidden_subprocess_kwargs(),
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
        mids = train_ns.map_id
        if isinstance(mids, int):
            allowed = {int(mids)}
        else:
            allowed = {int(x) for x in mids}
        map_pool = [m for m in pool if m["map_id"] in allowed]
        if not map_pool:
            raise ValueError(f"follow-train: no map with map_id in {sorted(allowed)}")
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


def _merge_train_ns_with_explicit_ai_args(
    train_ns: Any,
    args: argparse.Namespace,
) -> Any:
    """Shallow copy of ``train_ns`` with only CLI fields present on ``args`` applied."""
    merged = copy.copy(train_ns)
    if hasattr(args, "map_id"):
        merged.map_id = args.map_id
    if hasattr(args, "co0"):
        merged.co_p0 = args.co0
    if hasattr(args, "co1"):
        merged.co_p1 = args.co1
    if hasattr(args, "tier"):
        merged.tier = args.tier
    return merged


def _user_mentions_capture_move_gate(argv: list[str]) -> bool:
    """True if ``argv`` overrides capture-move-gate on the ai_vs_ai CLI."""
    for tok in argv:
        if tok == "--capture-move-gate" or tok.startswith("--capture-move-gate="):
            return True
    return False


def _capture_move_gate_effective(train_ns: Any, user: list[str], ai_args: Any) -> float:
    """Probability from ai_vs_ai CLI if present, else from the live trainer's argparse namespace."""
    if _user_mentions_capture_move_gate(user):
        return float(getattr(ai_args, "capture_move_gate", 0.0) or 0.0)
    if train_ns is not None:
        return float(getattr(train_ns, "capture_move_gate", 0.0) or 0.0)
    return 0.0


def _user_mentions_learner_greedy_mix(argv: list[str]) -> bool:
    for tok in argv:
        if tok == "--learner-greedy-mix" or tok.startswith("--learner-greedy-mix="):
            return True
    return False


def _learner_greedy_mix_effective(train_ns: Any, user: list[str], ai_args: Any) -> float:
    if _user_mentions_learner_greedy_mix(user):
        return max(
            0.0,
            min(1.0, float(getattr(ai_args, "learner_greedy_mix", 0.0) or 0.0)),
        )
    if train_ns is not None:
        return max(
            0.0,
            min(1.0, float(getattr(train_ns, "learner_greedy_mix", 0.0) or 0.0)),
        )
    return 0.0


def _opening_book_path_effective(train_ns: Any, ai_args: Any) -> Optional[Path]:
    p = getattr(ai_args, "opening_book", None)
    if p is not None:
        return Path(p)
    if train_ns is not None:
        t = getattr(train_ns, "opening_book", None)
        if t is not None:
            return Path(t)
    return None


def _opening_book_strict_co_effective(train_ns: Any, ai_args: Any) -> bool:
    if bool(getattr(ai_args, "opening_book_strict_co", False)):
        return True
    if train_ns is not None:
        return bool(getattr(train_ns, "opening_book_strict_co", False))
    return False


# ---------------------------------------------------------------------------
# Replay export UX: AWBW Replay Player (desktop) + folder fallback
# ---------------------------------------------------------------------------

_CAPTURE_GATE_ENV = "AWBW_CAPTURE_MOVE_GATE"


def _log_capture_move_gate_status() -> None:
    """Mirror train.py: log when legal-action mask uses capture move gate."""
    from engine.action import parse_capture_move_gate_env_value

    raw = os.environ.get(_CAPTURE_GATE_ENV, "")
    prob = parse_capture_move_gate_env_value(raw)
    if prob > 0.0:
        _log(
            f"env: {_CAPTURE_GATE_ENV}={raw!r} (stochastic capture gate P={prob}; "
            "infantry/mech MOVE may be restricted when capturable tiles are reachable)"
        )


def _open_replay_in_desktop_viewer(replay_zip: Path) -> None:
    """
    Start the AWBW Replay Player with the given ``.zip`` (argv: exe + absolute zip).

    On Windows we use ``cmd /c start "" exe zip`` so that:

    * A **new** top-level process is spawned (single-instance / COM apps that
      ignore a second direct ``CreateProcess`` still get a fresh window).
    * We do **not** pass ``CREATE_NO_WINDOW`` to the viewer itself — spawning the
      desktop player with that flag can prevent the WPF window from appearing.

    The ``cmd.exe`` process may use hidden-console flags only to avoid a flash;
    the Replay Player child is started by ``start`` and is not hidden.

    Falls back to opening the containing folder if the exe is missing or spawn fails.
    """
    zp = replay_zip.resolve()
    exe = resolve_awbw_replay_player_exe(_REPO)
    if exe is None:
        _log(
            f"viewer: AWBW Replay Player.exe not found - set {REPLAY_PLAYER_EXE_ENV} "
            f"or build under {REPLAY_PLAYER_THIRD_PARTY_DIR} (see README / desync-triage-viewer §4a)"
        )
        _open_replay_output_folder(replay_zip)
        return
    try:
        if sys.platform == "win32":
            popen_kw: dict[str, Any] = dict(
                cwd=str(exe.parent),
                close_fds=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            popen_kw.update(_win32_hidden_subprocess_kwargs())
            subprocess.Popen(
                ["cmd.exe", "/c", "start", "", str(exe), str(zp)],
                **popen_kw,
            )
        else:
            subprocess.Popen(
                [str(exe), str(zp)],
                cwd=str(exe.parent),
                close_fds=sys.platform != "win32",
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        _log(f"viewer: AWBW Replay Player - {exe.name} loaded {zp} (new process)")
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
            # ``os.startfile`` can route through a visible console on some setups;
            # ``explorer.exe`` with hidden creation flags keeps I/O in this terminal only.
            subprocess.Popen(
                ["explorer.exe", str(folder)],
                close_fds=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **_win32_hidden_subprocess_kwargs(),
            )
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
        _log(f"checkpoint: could not load {ckpt_path}: {exc} - using random policy")
        return None


def _resolve_seat_checkpoint_paths(
    *,
    ckpt_path: Optional[Path],
    ckpt_path_p0: Optional[Path],
    ckpt_path_p1: Optional[Path],
) -> tuple[Path, Path]:
    """Return concrete zip paths for P0 / P1 (defaults + shared ``ckpt_path`` fill-in)."""
    fb = _default_ckpt_path()
    p0 = ckpt_path_p0 if ckpt_path_p0 is not None else ckpt_path
    p1 = ckpt_path_p1 if ckpt_path_p1 is not None else ckpt_path
    return (fb if p0 is None else Path(p0), fb if p1 is None else Path(p1))


def _load_models_for_two_seats(
    path_p0: Path,
    path_p1: Path,
    *,
    force_random: bool,
) -> tuple[Any | None, Any | None]:
    """(model_p0, model_p1). Same path ⇒ one load, shared reference. Random if *force_random*."""
    if force_random:
        return None, None
    rp0 = path_p0.expanduser().resolve(strict=False)
    rp1 = path_p1.expanduser().resolve(strict=False)
    if not rp0.is_file():
        _log(f"checkpoint: P0 — not found at {rp0} - using random policy")
    if not rp1.is_file():
        _log(f"checkpoint: P1 — not found at {rp1} - using random policy")
    if rp0.is_file() and rp1.is_file() and rp0 == rp1:
        m = _load_model(rp0)
        return (m, m)
    m0 = _load_model(rp0) if rp0.is_file() else None
    m1 = _load_model(rp1) if rp1.is_file() else None
    return m0, m1


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def _obs_from_state(state: GameState) -> dict:
    """Observation from **active** player's seat (ego-centric, matches training rollout)."""

    seat = int(state.active_player)
    spatial, scalars = encode_state(state, observer=seat)
    return {"spatial": spatial, "scalars": scalars}


def _choose_action(
    state: GameState,
    seat_models: tuple[Any | None, Any | None],
    rng: random.Random,
    *,
    opening_book_mgr: Any | None = None,
    learner_greedy_mix: float,
) -> Action:
    """
    Pick an action for the current player.
    Optionally consumes ``TwoSidedOpeningBookManager`` indices (same as env opponent/book path),
    then applies capture-greedy teacher mix on **P0** only (matching ``AWBWEnv``).
    Uses *MaskablePPO* with ego-centric observations when ``seat_models[seat]`` is set.
    """
    legal = get_legal_actions(state)
    if not legal:
        raise RuntimeError(_NO_LEGAL_ACTIONS_MSG)

    seat = int(state.active_player)
    model = seat_models[seat] if 0 <= seat < 2 else None
    mask = _get_action_mask(state)
    cal = int(getattr(state, "turn", 0) or 0)

    if opening_book_mgr is not None:
        bk = opening_book_mgr.suggest_flat(seat=seat, calendar_turn=cal, action_mask=mask)
        if bk is not None:
            act = _flat_to_action(int(bk), state)
            if act is not None:
                return act

    skip_greedy = False
    if opening_book_mgr is not None and seat == 0 and float(learner_greedy_mix) > 0.0:
        if (
            opening_book_mgr.peek_book_candidate_flat_safe(
                seat=0, calendar_turn=cal, action_mask=mask
            )
            is not None
        ):
            skip_greedy = True

    if (
        not skip_greedy
        and seat == 0
        and float(learner_greedy_mix) > 0.0
        and rng.random() < float(learner_greedy_mix)
    ):
        from rl.self_play import pick_capture_greedy_flat

        greedy_idx = pick_capture_greedy_flat(state, mask)
        ga = _flat_to_action(greedy_idx, state)
        if ga is not None:
            return ga

    if model is None:
        return rng.choice(legal)

    obs = _obs_from_state(state)
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
# Live snapshot (exported games) — discover pkls for batch runs
# ---------------------------------------------------------------------------


def _list_snapshot_pkls_in_dir(root: Path) -> list[tuple[int, Path]]:
    """(games_id, pkl) for ``<root>/<id>/engine_snapshot.pkl`` and ``<root>/<id>.pkl``."""
    out: list[tuple[int, Path]] = []
    root = Path(root)
    if not root.is_dir():
        return out
    seen: set[int] = set()
    for sub in sorted(root.iterdir(), key=lambda p: p.name):
        if not sub.is_dir():
            continue
        try:
            gid = int(sub.name)
        except ValueError:
            continue
        nested = sub / "engine_snapshot.pkl"
        if nested.is_file():
            out.append((gid, nested))
            seen.add(gid)
    for p in sorted(root.iterdir(), key=lambda x: x.name):
        if p.is_file() and p.suffix == ".pkl" and p.stem.isdigit():
            gid = int(p.stem)
            if gid not in seen and p.is_file():
                out.append((gid, p))
    return sorted(out, key=lambda t: t[0])


def _find_latest_engine_snapshot_pkl(root: Path) -> Path | None:
    """Newest mtime among ``<root>/**/engine_snapshot.pkl`` and ``<root>/*.pkl`` (numeric stem)."""
    root = Path(root)
    if not root.is_dir():
        return None
    best: tuple[float, Path] | None = None
    for p in root.rglob("engine_snapshot.pkl"):
        if p.is_file():
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            if best is None or m > best[0]:
                best = (m, p)
    for p in root.glob("*.pkl"):
        if p.is_file() and p.stem.isdigit():
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            if best is None or m > best[0]:
                best = (m, p)
    return best[1] if best else None


# ---------------------------------------------------------------------------
# Core game loop
# ---------------------------------------------------------------------------

def _apply_max_turn_tiebreak(state: GameState) -> None:
    """Mirror :meth:`GameState._end_turn` property tiebreak when the calendar cap is hit.

    Called when ``state.turn > state.max_turns`` but the engine has not yet
    marked ``done`` (belt-and-suspenders for ``ai_vs_ai`` day limits).
    """
    if state.done:
        return
    state.done = True
    p0_props = state.count_properties(0)
    p1_props = state.count_properties(1)
    # Keep in lockstep with ``GameState._end_turn`` calendar cap (AWBW parity).
    if p0_props > p1_props:
        state.winner = 0
        state.win_reason = "max_days_tiebreak"
    elif p1_props > p0_props:
        state.winner = 1
        state.win_reason = "max_days_tiebreak"
    else:
        state.winner = -1
        state.win_reason = "max_days_draw"


def run_game(
    map_id: Optional[int] = None,
    ckpt_path: Optional[Path] = None,
    ckpt_path_p0: Optional[Path] = None,
    ckpt_path_p1: Optional[Path] = None,
    co0: Optional[int] = None,
    co1: Optional[int] = None,
    tier: str = "T2",
    seed: Optional[int] = None,
    max_days: Optional[int] = None,
    max_turns: Optional[int] = None,
    force_random: bool = False,
    open_viewer: bool = True,
    output_dir: Optional[Path] = None,
    game_id: Optional[int] = None,
    max_total_actions: int = _DEFAULT_MAX_TOTAL_ACTIONS,
    max_actions_per_active_turn: int = _DEFAULT_MAX_ACTIONS_PER_ACTIVE_TURN,
    capture_move_gate: float = 0.0,
    live_snapshot_path: Optional[Path] = None,
    opening_book_path: Optional[Path] = None,
    opening_book_strict_co: bool = False,
    learner_greedy_mix: float = 0.0,
) -> Path:
    """
    Run one AI vs AI game and export an AWBW Replay Player–compatible zip.
    Returns the path to the created .zip file.

    With ``live_snapshot_path`` set, loads that pickle (``write_live_snapshot`` / export
    layout) and sets ``state.max_turns = current day + 10`` (see ``LIVE_SNAPSHOT_CALENDAR_CAP_DAYS``).

    Legal actions honor ``AWBW_CAPTURE_MOVE_GATE`` (see ``engine.action``): set the
    environment variable before running, or pass ``capture_move_gate`` in ``[0,1]``
    (same effect as ``train.py --capture-move-gate P``). Follow-train inherits a
    live trainer's probability when the ai_vs_ai CLI does not override the flag.

    Opening book: pass ``opening_book_path`` to a flat-action JSONL
    (e.g. ``data/opening_books/std_pool_precombat.jsonl``); both seats consume lines
    like ``TwoSidedOpeningBookManager`` in ``AWBWEnv``. Ignored for live snapshots.

    ``learner_greedy_mix``: P0-only capture-greedy teacher probability (parity with
    ``train.py --learner-greedy-mix`` / ``AWBW_LEARNER_GREEDY_MIX``); skipped when
    seat-0 joint book expects the next legal flat.

    Checkpoints: ``ckpt_path`` applies to both seats unless overridden per seat by
    ``ckpt_path_p0`` / ``ckpt_path_p1``. Paths are resolved independently; identical
    resolved paths load a single MaskablePPO shared by both players.

    Observations passed to MaskablePPO use **observer = active_player** (ego-centric).

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

    calendar_cap = _resolve_max_calendar_days(max_days, max_turns, default=100)
    if live_snapshot_path is None and calendar_cap < 1:
        raise ValueError(
            "max_days / max_turns must be >= 1 (``make_initial_state`` / engine contract). "
            "For a very short run use e.g. --max-days 2."
        )

    cmg = float(capture_move_gate or 0.0)
    if cmg > 0.0:
        os.environ[_CAPTURE_GATE_ENV] = str(cmg)
    _log_capture_move_gate_status()

    state: GameState
    map_data: MapData
    snapshot_games_id: int | None = None

    if live_snapshot_path is not None:
        from rl.live_snapshot import load_live_snapshot_dict

        p = Path(live_snapshot_path)
        if not p.is_file():
            raise FileNotFoundError(f"live snapshot not found: {p.resolve()}")
        raw = load_live_snapshot_dict(p)
        sg = raw.get("games_id")
        if sg is not None:
            try:
                snapshot_games_id = int(sg)
            except (TypeError, ValueError):
                snapshot_games_id = None
        state = copy.deepcopy(raw["state"])
        map_data = state.map_data
        map_id = int(map_data.map_id)
        co0 = int(state.co_states[0].co_id)
        co1 = int(state.co_states[1].co_id)
        tier = str(state.tier_name)
        day0 = int(state.turn)
        state.max_turns = day0 + int(LIVE_SNAPSHOT_CALENDAR_CAP_DAYS)
        _log(
            f"session: from live snapshot  path={p}  games_id={raw.get('games_id')!r}  "
            f"seed={seed!r}  day={day0}  cap -> {state.max_turns} "
            f"(+{LIVE_SNAPSHOT_CALENDAR_CAP_DAYS} in-game days)"
        )
        _log(
            f"map: {map_data.name}  id={map_id}  (from snapshot)  "
            f"COs: P0={co0}  P1={co1}  tier={tier}"
        )
    else:
        _log(
            f"session: start  seed={seed!r}  max_days={calendar_cap}  "
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

    # ---- Checkpoint(s) ----
    path_p0, path_p1 = _resolve_seat_checkpoint_paths(
        ckpt_path=ckpt_path,
        ckpt_path_p0=ckpt_path_p0,
        ckpt_path_p1=ckpt_path_p1,
    )
    seat_models = _load_models_for_two_seats(path_p0, path_p1, force_random=force_random)
    m0, m1 = seat_models

    _policy_log_line = ""
    if force_random:
        _policy_log_line = "uniform random over legal actions (both seats)"
    elif m0 is None and m1 is None:
        _policy_log_line = (
            "uniform random over legal actions (both seats; checkpoints missing or unloadable)"
        )
    elif m0 is m1 and m0 is not None:
        _policy_log_line = f"MaskablePPO shared (CPU): {path_p0}"
    elif m0 is not None and m1 is not None:
        _policy_log_line = f"MaskablePPO per seat (CPU): P0={path_p0}  P1={path_p1}"
    elif m0 is not None:
        _policy_log_line = (
            f"MaskablePPO P0 (CPU)={path_p0}; P1 random (missing ckpt={path_p1})"
        )
    else:
        _policy_log_line = (
            f"P0 random (missing ckpt={path_p0}); MaskablePPO P1 (CPU)={path_p1}"
        )
    _log(f"policy: {_policy_log_line}")

    diag_primary = m0 if m0 is not None else m1
    diag_observer1 = None
    if diag_primary is not None and m1 is not None and m1 is not diag_primary:
        diag_observer1 = m1

    if live_snapshot_path is None:
        # ---- Initial state (fresh) ----
        _log("engine: building initial GameState")
        _mka: dict = {"starting_funds": 0, "tier_name": tier, "luck_seed": seed}
        rfm = getattr(map_data, "replay_first_mover", None)
        if rfm is not None:
            _mka["replay_first_mover"] = int(rfm)
        state = make_initial_state(map_data, co0, co1, max_days=calendar_cap, **_mka)
        # Single source of truth for the play loop (must match CLI ``max_days``).
        state.max_turns = int(calendar_cap)
        _log(
            f"engine: ready  active_player=P{state.active_player}  day={state.turn}  "
            f"max_days={state.max_turns}  done={state.done}"
        )

    # ---- Output paths (resolved up front so error handler can also use them) ----
    gid = game_id or snapshot_games_id or (int(time.time()) % 999000 + 1000)
    out_dir = output_dir or _REPLAY_OUT
    out_path = Path(out_dir) / f"{gid}.zip"
    start_date_str = log_now_wall()

    lgm = max(0.0, min(1.0, float(learner_greedy_mix or 0.0)))
    from rl.env import LEARNER_GREEDY_MIX_ENV

    if lgm > 0.0:
        os.environ[LEARNER_GREEDY_MIX_ENV] = str(lgm)
        _log(f"env: {LEARNER_GREEDY_MIX_ENV}={lgm!r}")
    else:
        os.environ.pop(LEARNER_GREEDY_MIX_ENV, None)

    opening_book_mgr: Any | None = None
    if live_snapshot_path is None and opening_book_path is not None:
        p_ob = Path(opening_book_path)
        if not p_ob.is_file():
            _log(f"opening_book: not found ({p_ob.resolve()}) — skipping")
        else:
            try:
                from rl.opening_book import TwoSidedOpeningBookManager

                ob_seed = int((seed if seed is not None else gid) ^ 0xABC5_DEE5) & (
                    (1 << 31) - 1
                )
                opening_book_mgr = TwoSidedOpeningBookManager(
                    p_ob,
                    seats="both",
                    prob=1.0,
                    strict_co=bool(opening_book_strict_co),
                    max_day=None,
                    seed=int(ob_seed),
                )
                epi = int(gid) % (2**31 - 2)
                opening_book_mgr.on_episode_start(
                    episode_id=max(1, epi),
                    map_id=int(map_id),
                    co_ids=[int(co0), int(co1)],
                )
                fld = opening_book_mgr.log_fields()
                _log(
                    f"opening_book: {p_ob.name}  strict_co={opening_book_strict_co}  "
                    f"id_p0={fld.get('opening_book_id_p0')!r}  id_p1={fld.get('opening_book_id_p1')!r}"
                )
            except Exception as exc:
                _log(f"opening_book: init failed ({exc!r}) — skipping")
                opening_book_mgr = None

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
        f"play: loop start  day_cap={state.max_turns}  "
        f"max_total_actions={max_total_actions}  "
        f"max_actions_per_active_turn={max_actions_per_active_turn}  "
        f"progress every {_ACTION_PROGRESS_INTERVAL} actions + on day change"
    )

    with open(_MAP_POOL_PATH, encoding="utf-8") as _pf:
        _pool = json.load(_pf)
    _mp_meta = next((m for m in _pool if m.get("map_id") == int(map_id)), {})
    _map_is_std = str(_mp_meta.get("type", "")).lower() == "std"
    state.spirit_map_is_std = bool(_map_is_std)
    from rl.heuristic_termination import (  # noqa: I001
        DEFAULT_DISAGREEMENT_LOG,
        config_from_env,
        diag_enabled_from_env,
        run_calendar_day,
    )

    _diag_n = 0

    try:
        while not state.done:
            # Calendar cap: same rule as ``GameState._end_turn`` (``turn`` increments
            # when P1 ends — see engine). Always compare to ``state.max_turns``.
            if state.turn > state.max_turns:
                _log(
                    f"play: day limit — day {state.turn} > cap {state.max_turns} "
                    "(end-inclusive calendar cap; property tiebreak / stop)"
                )
                _apply_max_turn_tiebreak(state)
                break

            # Fuse checks are issued *before* we pick the next action so the
            # partial-dump path sees the state at the point we gave up.
            if action_count >= max_total_actions:
                raise RuntimeError(_FUSE_TOTAL_MSG)
            if turn_action_count >= max_actions_per_active_turn:
                raise RuntimeError(_FUSE_PER_TURN_MSG)

            turn_before = int(state.turn)
            action = _choose_action(
                state,
                seat_models,
                rng,
                opening_book_mgr=opening_book_mgr,
                learner_greedy_mix=lgm,
            )
            prev_player = state.active_player
            state, _reward, _done = state.step(action)
            action_count += 1
            turn_action_count += 1

            if (
                not state.done
                and diag_enabled_from_env()
                and int(state.turn) > turn_before
                and int(state.active_player) == 0
            ):
                _cfg = config_from_env()
                _tier_ok = not _cfg.allowed_tiers or str(tier) in _cfg.allowed_tiers
                p_log = str(os.environ.get("AWBW_HEURISTIC_DIAG_LOG", "") or str(DEFAULT_DISAGREEMENT_LOG))

                def _enc(s, o: int):
                    sp, sc = encode_state(s, observer=o, belief=None)
                    return {"spatial": sp, "scalars": sc}

                _, _d = run_calendar_day(
                    state,
                    diag_primary,
                    _cfg,
                    _enc,
                    is_std_map=bool(_map_is_std),
                    map_tier_ok=_tier_ok,
                    episode_id=int(gid),
                    map_id=int(map_id),
                    learner_seat=0,
                    log_path=Path(p_log),
                    diag_line_budget=_diag_n,
                    observer1_model=diag_observer1,
                )
                _diag_n += int(_d)

            if state.turn > state.max_turns and not state.done:
                _log(
                    f"play: day limit after step — day {state.turn} > {state.max_turns}"
                )
                _apply_max_turn_tiebreak(state)
                break

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
            f"turn_actions={turn_action_count} - dumping partial replay"
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
            luck_seed=seed,
        )
        if open_viewer:
            pz = Path(out_dir) / f"{gid}.partial.zip"
            if pz.is_file():
                _log(f"viewer: opening partial replay -> {pz.name}")
                _open_replay_in_desktop_viewer(pz)
            else:
                _log("viewer: partial zip missing; opening output folder")
                _open_replay_output_folder(pz)
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
        f"play: finished - {winner_str} | day={state.turn} | actions={action_count} | "
        f"snapshots={len(snapshots)}"
    )
    _log(
        f"play: wall time {elapsed:.1f}s  (replay export next; may be much slower than play)"
    )

    # ---- Export replay (heavy: PHP snapshots + gzip + full_trace replay for p: stream) ----
    # gid / out_dir / out_path / start_date_str were resolved before the play loop
    # so the error handler can reuse them.
    if live_snapshot_path is not None:
        game_name = f"AI-vs-AI (live +{LIVE_SNAPSHOT_CALENDAR_CAP_DAYS}d)  {map_data.name}  [{winner_str}]"
    else:
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
        "days": state.turn,
        "winner": winner_str,
        "win_reason": state.win_reason,
        "n_actions_full_trace": len(state.full_trace),
        "n_actions_game_log": len(state.game_log),
        "full_trace": full_trace_copy,
        "game_log": list(state.game_log),
    }
    if seed is not None:
        trace_record["luck_seed"] = int(seed)

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
                luck_seed=seed,
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
            _log(f"export: FAILED - {exc!r}")

    _log("export: worker thread starting (main thread will join when done)")
    worker = threading.Thread(
        target=_export_replay_worker,
        name="ai_vs_ai-replay-export",
        daemon=False,
    )
    worker.start()
    worker.join()
    export_exc = export_error[0]
    if export_exc is not None:
        _log(f"export: thread returned error - {export_exc!r}")

    _log(f"session: replay + trace ready  path={out_path}")

    # ---- Open viewer (always when requested, even if export failed) ----
    zip_for_viewer: Path | None = None
    if export_exc is None and out_path.is_file():
        zip_for_viewer = out_path
    else:
        partial_try = Path(out_dir) / f"{gid}.partial.zip"
        if partial_try.is_file():
            zip_for_viewer = partial_try

    if open_viewer:
        if zip_for_viewer is not None:
            _log(f"viewer: opening replay (game_id={gid}) -> {zip_for_viewer.name}")
            _open_replay_in_desktop_viewer(zip_for_viewer)
        else:
            _log(
                "viewer: no .zip to open after export — opening output folder "
                f"(game_id={gid})"
            )
            # ``_open_replay_output_folder`` uses ``.parent`` — ``out_path`` may
            # not exist on disk yet, but its parent is still the export directory.
            _open_replay_output_folder(out_path)
    else:
        _log("viewer: skipped (--no-open)")

    if export_exc is not None:
        raise export_exc

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
    luck_seed: Optional[int] = None,
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
        "days":                 state.turn,
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
    if luck_seed is not None:
        record["luck_seed"] = int(luck_seed)

    try:
        _write_trace_record(record, trace_path)
        _log(f"partial: trace JSON written -> {trace_path.resolve()}")
    except Exception as trace_exc:
        _log(f"partial: trace JSON write FAILED - {trace_exc!r}")

    reason_tag = {
        "no_legal_actions": "no legal actions",
        "fuse_total":       "action fuse: total",
        "fuse_per_turn":    "action fuse: per-turn",
    }.get(reason, reason)
    partial_game_name = f"AI-vs-AI  {map_name}  [PARTIAL - {reason_tag}]"

    try:
        from tools.export_awbw_replay import write_awbw_replay

        write_awbw_replay(
            snapshots=snapshots,
            output_path=zip_path,
            game_id=gid,
            game_name=partial_game_name,
            start_date=start_date_str,
            full_trace=full_trace_copy,
            luck_seed=luck_seed,
        )
        _log(f"partial: replay zip (with p: stream) -> {zip_path.resolve()}")
    except Exception as zip_exc:
        _log(
            f"partial: zip export with full_trace FAILED - {zip_exc!r} - "
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
            _log(f"partial: snapshot-only zip export also FAILED - {zip_exc2!r}")


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
        "days": state.turn,
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
            "Do not read a running train.py process — use only ai_vs_ai flags "
            "and defaults (e.g. tier T2, random Std map, latest.zip: "
            f"Z: if present, else {_CKPT_DEFAULT.name})."
        ),
    )
    parser.add_argument(
        "--map-id",
        type=int,
        default=argparse.SUPPRESS,
        help="AWBW map ID (when following train: overrides trainer's --map-id for sampling)",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=argparse.SUPPRESS,
        help=(
            "Checkpoint zip (when following train: overrides trainer checkpoint "
            f"resolution; when alone: {_Z_PREFERRED_CKPT} if that file exists, else {_CKPT_DEFAULT}) "
            "(shared default for both seats unless overridden with --ckpt-p0/--ckpt-p1)."
        ),
    )
    parser.add_argument(
        "--ckpt-p0",
        type=Path,
        default=argparse.SUPPRESS,
        metavar="ZIP",
        help="Checkpoint zip for seat 0 only (falls back to shared --ckpt / train / default zip).",
    )
    parser.add_argument(
        "--ckpt-p1",
        type=Path,
        default=argparse.SUPPRESS,
        metavar="ZIP",
        help="Checkpoint zip for seat 1 only (falls back to shared --ckpt / train / default zip).",
    )
    parser.add_argument(
        "--co0",
        type=int,
        default=argparse.SUPPRESS,
        help="P0 CO id (when following train: overrides trainer --co-p0)",
    )
    parser.add_argument(
        "--co1",
        type=int,
        default=argparse.SUPPRESS,
        help="P1 CO id (when following train: overrides trainer --co-p1)",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default=argparse.SUPPRESS,
        help="Tier name (when following train: overrides trainer --tier)",
    )
    parser.add_argument("--seed",     type=int,  default=None, help="RNG seed")
    parser.add_argument(
        "--max-days",
        "--max-turns",
        dest="max_days",
        type=int,
        default=100,
        metavar="N",
        action=_MaxCalendarDaysAction,
        help=(
            "End-inclusive calendar day cap (play days 1..N; tiebreak when day N+1 would start). "
            "Same as engine ``GameState.max_turns``. Default 100. "
            "Alias ``--max-turns`` (deprecated). Short smoke: --max-days 2."
        ),
    )
    parser.add_argument("--random",   action="store_true",
                        help="Force random-vs-random (ignore checkpoint)")
    parser.add_argument("--no-open",  action="store_true",
                        help="Do not launch AWBW Replay Player (or folder fallback) after export")
    parser.add_argument(
        "--capture-move-gate",
        nargs="?",
        const=1.0,
        default=0.0,
        metavar="P",
        type=_parse_capture_move_gate_probability_cli,
        help=(
            "Sets AWBW_CAPTURE_MOVE_GATE=stochastic probability P in [0,1] (same as "
            "train.py). Omit the value for P=1. See engine.action capture gate."
        ),
    )
    parser.add_argument(
        "--opening-book",
        type=Path,
        default=None,
        metavar="JSONL",
        help=(
            "Flat-action opening book for both seats "
            "(e.g. data/opening_books/std_pool_precombat.jsonl). "
            "Ignored for --from-live-* snapshot loads. "
            "When following train: used if set here, else trainer --opening-book."
        ),
    )
    parser.add_argument(
        "--opening-book-strict-co",
        action="store_true",
        help=(
            "Restrict book lines to per-seat CO metadata. "
            "When following train: true if this flag or trainer --opening-book-strict-co."
        ),
    )
    parser.add_argument(
        "--learner-greedy-mix",
        type=float,
        default=0.0,
        metavar="P",
        help=(
            "P0-only: with probability P in [0,1], act as capture-greedy teacher "
            "(train.py / AWBW_LEARNER_GREEDY_MIX); skipped when a seat-0 book line "
            "expects the next flat. When following train: overridden by this flag if "
            "you pass it; else copied from train --learner-greedy-mix."
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
            "Fuse: cap actions within one seat's player-turn through END_TURN "
            "(not a calendar-day limit). Abort and dump a partial replay if that seat "
            "takes this many consecutive actions without ending their turn "
            f"(default: {_DEFAULT_MAX_ACTIONS_PER_ACTIVE_TURN})."
        ),
    )
    parser.add_argument(
        "--from-live-snapshot",
        type=Path,
        default=None,
        metavar="PKL",
        help=(
            "Load ``engine_snapshot.pkl`` (or ``write_live_snapshot`` output); "
            f"day cap = snapshot day + {LIVE_SNAPSHOT_CALENDAR_CAP_DAYS}. "
            "Implies not following train for map/CO sampling."
        ),
    )
    parser.add_argument(
        "--from-live-games-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Run once per ``<games_id>/engine_snapshot.pkl`` (or ``<games_id>.pkl``) "
            f"under DIR; same {LIVE_SNAPSHOT_CALENDAR_CAP_DAYS}-day cap. "
            "Opens each exported zip in the desktop viewer unless --no-open."
        ),
    )
    _default_export_dir = _REPO / "replays" / "amarinner_my_games"
    parser.add_argument(
        "--from-latest-export",
        type=Path,
        nargs="?",
        const=_default_export_dir,
        default=None,
        metavar="DIR",
        help=(
            "Load the most recently written ``engine_snapshot.pkl`` (or top-level "
            f"``<id>.pkl``) under DIR. With no path, use {_default_export_dir} "
            "(re-export from site to refresh). Implies not following train."
        ),
    )
    args = parser.parse_args(user)
    if args.max_days < 1:
        parser.error("--max-days must be >= 1 (try 2 for a very short run)")
    # If CPython/launcher preflags made ``user`` empty or incomplete, the flag
    # can still appear in raw ``sys.argv``; honor it so ``--no-follow-train`` is reliable.
    if not args.no_follow_train and any(x == "--no-follow-train" for x in sys.argv[1:]):
        args.no_follow_train = True

    n_live = sum(
        1
        for v in (args.from_live_snapshot, args.from_live_games_dir, args.from_latest_export)
        if v is not None
    )
    if n_live > 1:
        parser.error(
            "use at most one of --from-live-snapshot, --from-live-games-dir, and --from-latest-export"
        )

    def _run_live_pkl(
        pkl: Path, *, out_gid: int | None, open_viewer: bool
    ) -> Path:
        ck = getattr(args, "ckpt", None)
        cp0 = getattr(args, "ckpt_p0", None)
        cp1 = getattr(args, "ckpt_p1", None)
        return run_game(
            ckpt_path=Path(ck) if ck is not None else None,
            ckpt_path_p0=Path(cp0) if cp0 is not None else None,
            ckpt_path_p1=Path(cp1) if cp1 is not None else None,
            map_id=0,  # unused when loading snapshot
            co0=0,  # unused
            co1=0,  # unused
            tier="T2",  # unused
            seed=args.seed,
            max_days=args.max_days,  # ignored for live; kept for API
            force_random=args.random,
            open_viewer=open_viewer,
            output_dir=args.out_dir,
            game_id=out_gid,
            max_total_actions=args.max_total_actions,
            max_actions_per_active_turn=args.max_actions_per_active_turn,
            capture_move_gate=args.capture_move_gate,
            live_snapshot_path=pkl,
        )

    if args.from_latest_export is not None:
        d = args.from_latest_export
        pkl = _find_latest_engine_snapshot_pkl(d)
        if pkl is None:
            _log(
                f"from-latest-export: no engine_snapshot.pkl (or <id>.pkl) under {d.resolve()!s}"
            )
            raise SystemExit(1)
        try:
            age_s = time.time() - pkl.stat().st_mtime
        except OSError:
            age_s = -1.0
        _log(
            f"from-latest-export: using {pkl} "
            f"(mtime age ~{age_s/3600.0:.2f} h — re-run export to refresh from AWBW)"
        )
        _run_live_pkl(
            pkl,
            out_gid=args.game_id,
            open_viewer=not args.no_open,
        )
        return

    if args.from_live_games_dir is not None:
        pairs = _list_snapshot_pkls_in_dir(args.from_live_games_dir)
        if not pairs:
            _log(
                f"from-live-games-dir: no engine_snapshot.pkl (or <id>.pkl) under {args.from_live_games_dir}"
            )
            raise SystemExit(1)
        n = len(pairs)
        for i, (gid, pth) in enumerate(pairs):
            _run_live_pkl(
                pth,
                out_gid=gid,
                open_viewer=not args.no_open,
            )
        _log(
            f"from-live-games-dir: wrote {n} replay(s) under "
            f"{(args.out_dir or _REPLAY_OUT).resolve()!s}"
        )
        return

    if args.from_live_snapshot is not None:
        _run_live_pkl(
            args.from_live_snapshot,
            out_gid=args.game_id,
            open_viewer=not args.no_open,
        )
        return

    want_follow = not args.no_follow_train
    if want_follow:
        picked = _pick_training_train_argv()
        if picked is not None:
            tail, pid = picked
            from train import build_train_argument_parser

            tparser = build_train_argument_parser()
            train_ns, unk = tparser.parse_known_args(tail)
            if unk:
                _log(f"follow-train: pid={pid} ignored unknown train.py args: {unk}")
            merged = _merge_train_ns_with_explicit_ai_args(train_ns, args)
            override_bits = [
                n
                for n in (
                    "map_id",
                    "ckpt",
                    "ckpt_p0",
                    "ckpt_p1",
                    "co0",
                    "co1",
                    "tier",
                )
                if hasattr(args, n)
            ]
            if _user_mentions_capture_move_gate(user):
                override_bits.append("capture-move-gate")
            if getattr(args, "opening_book", None) is not None:
                override_bits.append("opening-book")
            if getattr(args, "opening_book_strict_co", False):
                override_bits.append("opening-book-strict-co")
            if _user_mentions_learner_greedy_mix(user):
                override_bits.append("learner-greedy-mix")
            if override_bits:
                _log(
                    "follow-train: CLI overrides on top of train.py: "
                    + ", ".join(override_bits)
                )
            _log(
                f"follow-train: matched train.py pid={pid} "
                f"(map_id={merged.map_id} tier={merged.tier!r} "
                f"co_p0={merged.co_p0} co_p1={merged.co_p1} "
                f"broad_prob={merged.curriculum_broad_prob} "
                f"capture_move_gate={_capture_move_gate_effective(train_ns, user, args)})"
            )
            rng = random.Random(args.seed) if args.seed is not None else random.Random()
            try:
                map_id, tier, co0, co1 = _sample_from_train_ns(merged, rng)
            except ValueError as exc:
                _log(f"follow-train: matchup sampling failed ({exc}); using legacy defaults")
            else:
                if hasattr(args, "ckpt"):
                    ckpt_path = Path(args.ckpt)
                else:
                    ckpt_path = _resolve_ckpt_from_train_ns(merged)
                _log(f"follow-train: sampled map_id={map_id} tier={tier} co0={co0} co1={co1}")
                _log(f"follow-train: checkpoint -> {ckpt_path}")
                run_game(
                    map_id=map_id,
                    ckpt_path=ckpt_path,
                    ckpt_path_p0=(
                        Path(args.ckpt_p0)
                        if hasattr(args, "ckpt_p0")
                        else None
                    ),
                    ckpt_path_p1=(
                        Path(args.ckpt_p1)
                        if hasattr(args, "ckpt_p1")
                        else None
                    ),
                    co0=co0,
                    co1=co1,
                    tier=tier,
                    seed=args.seed,
                    max_days=args.max_days,
                    force_random=args.random,
                    open_viewer=not args.no_open,
                    output_dir=args.out_dir,
                    game_id=args.game_id,
                    max_total_actions=args.max_total_actions,
                    max_actions_per_active_turn=args.max_actions_per_active_turn,
                    capture_move_gate=_capture_move_gate_effective(train_ns, user, args),
                    opening_book_path=_opening_book_path_effective(train_ns, args),
                    opening_book_strict_co=_opening_book_strict_co_effective(
                        train_ns, args
                    ),
                    learner_greedy_mix=_learner_greedy_mix_effective(
                        train_ns, user, args
                    ),
                )
                return
        else:
            _log(
                "follow-train: no training train.py process found "
                "(or none parseable); using legacy defaults (Std map, T2, ckpt: Z: if present else repo latest.zip)"
            )

    run_game(
        map_id=getattr(args, "map_id", None),
        ckpt_path=getattr(args, "ckpt", None),
        ckpt_path_p0=getattr(args, "ckpt_p0", None),
        ckpt_path_p1=getattr(args, "ckpt_p1", None),
        co0=getattr(args, "co0", None),
        co1=getattr(args, "co1", None),
        tier=getattr(args, "tier", "T2"),
        seed=args.seed,
        max_days=args.max_days,
        force_random=args.random,
        open_viewer=not args.no_open,
        output_dir=args.out_dir,
        game_id=args.game_id,
        max_total_actions=args.max_total_actions,
        max_actions_per_active_turn=args.max_actions_per_active_turn,
        capture_move_gate=_capture_move_gate_effective(None, user, args),
        opening_book_path=_opening_book_path_effective(None, args),
        opening_book_strict_co=_opening_book_strict_co_effective(None, args),
        learner_greedy_mix=_learner_greedy_mix_effective(None, user, args),
    )


if __name__ == "__main__":
    main()
