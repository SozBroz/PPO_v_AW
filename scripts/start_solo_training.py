#!/usr/bin/env python3
"""
Tier 1 walk-away bootstrap for a single fleet machine (e.g. pc-b).

Async IMPALA uses one OS process per env worker; size ``--n-envs`` to host RAM.
``tools/propose_train_args`` guesses ``--n-envs`` / steps / batch from ``probe.json``.
Pass ``--train-extra-args`` with explicit ``--n-envs`` / ``--max-env-steps`` when you
mean a fixed rollout footprint.

**Curriculum stages:** each stage merges map / CO / tier / scaffolding / opening-book
knobs into ``proposed_args.json``.  Those are **not** the small PPO-geometry set handled
by ``train_reconfig_request.json`` in ``rl.self_play``.

To advance the live ``train.py`` onto a new stage **without stopping it yourself**, the
orchestrator needs ``--auto-apply`` (controlled SIGTERM and respawn of ``train.py`` with
updated argv from merged ``proposed``).

**Freeze semantics:** the orchestrator compares **only** the curriculum-relevant keys
(``CURRICULUM_RESTART_KEYS`` in ``fleet_orchestrator.py``) when deciding whether to
restart.  When it **does** restart, it builds the new process from ``applied_args.json``
(the last running set) and overlays **only** the curriculum keys from ``proposed_args.json``.
All non-curriculum keys — ``--n-envs``, ``--max-env-steps``, ``--n-steps``,
``--batch-size``, ``--training-backend``, live-game knobs, etc. — are **frozen**
permanently from whatever this bootstrap first wrote.  Probe churn (e.g. ``--n-envs``
flipping from 18 to 11 because the RAM heuristic says 11) **never** triggers a restart
and **never** changes the running process arguments.

This means one explicit invocation (e.g. ``--train-extra-args "--n-envs 18 --max-env-steps 8000"``)
is the **last word** on rollout geometry; it survives all future curriculum ticks.

Soft in-process reload without any process recycle would require extending the learner to
rebuild env/policy from arbitrary curriculum deltas; that pipeline does **not** exist
beyond ``--n-envs/--n-steps/--batch-size``.

**Default bootstrap:** ``fleet_orchestrator`` is launched **with** ``--auto-apply`` unless
you pass ``--no-orchestrator-auto-apply`` (e.g. only want disk-side ``proposed_args.json``
churn or debugging).

Async example::

    python scripts/start_solo_training.py --machine-id <id> --train-extra-args "--n-envs 18 --max-env-steps 8000" --training-backend async

One command probes the box, proposes PPO args (Phase 10f), writes launch metadata,
starts ``train.py`` and ``fleet_orchestrator.py``, and tears both down cleanly on Ctrl-C.
  Child ``train.py`` stdout/stderr are appended to ``logs/solo_train_train_py_<machine_id>.log``;
  ``fleet_orchestrator.py`` to ``logs/solo_train_fleet_orchestrator_<machine_id>.log`` (also when
  a console exists) so tracebacks and terminations are not lost.
  Optional ``--orchestrator-curriculum-window-games N`` forwards ``--curriculum-window-games``
  to the orchestrator (rolling game-log window for curriculum metrics; default if omitted: 200).

Early-game defaults (when ``proposed_args.json`` only supplies n_envs / n_steps / batch_size):
  Misery Andy mirror ``--map-id 123858 --tier T3 --co-p0 1 --co-p1 1`` (matches fleet
  map/CO defaults). ``--cold-opponent`` / ``--learner-greedy-mix`` match bare ``train.py``
  (``random`` / ``0.0``) unless ``proposed_args`` or curriculum sets them. Stage A/B
  capture/bootstrap knobs come from the orchestrator curriculum merge into ``proposed_args.json``,
  not from static ``propose_train_args`` output. After curriculum, the **first** launch applies
  ``fleet/<id>/operator_train_args_override.json`` per-flag (same order as
  :mod:`scripts.fleet_orchestrator` on a tick). **Opening book:** if
  ``--opening-book*`` flags are **absent** from ``proposed['args']``, they are filled from
  :data:`tools.curriculum_advisor.DEFAULT_OPENING_BOOK_TRAIN_ARGS` (``data/opening_books/std_pool_precombat.jsonl``,
  both seats). Pass ``--no-default-opening-book`` to skip that injection.
  Remaining gaps: ``--train-extra-args``.
  ``n_envs<=4`` on pc-b is operator-validated
  (FPS plan 2026-04-22); propose_train_args enforces the cap from probe.

Does not enable MCTS (implicit ``--mcts-mode off``). Does not auto-respawn crashed children.

**Why PowerShell "closes" ~10 minutes in:** ``fleet_orchestrator`` only restarts
``train.py`` when **curriculum keys** differ between ``proposed_args.json`` and
``applied_args.json``, and only after ``apply_cooldown_s`` (default **600s**) has
elapsed since the last apply.  Exit code **15** is the normal SIGTERM-style stop;
the bootstrap adopts the replacement PID from ``fleet/<id>/train.pid`` (a short race
wait is implemented).  Durable state: ``logs/solo_bootstrap_watch_<id>.log``
(30s heartbeat) and ``logs/start_solo_training.log``.

**Monitor without losing the window:** use ``Tee-Object`` or
``Start-Process python -ArgumentList '...' -NoNewWindow -Wait -RedirectStandardOutput ...``;
or run under ``cmd /k`` so the host stays open after exit.

The ``train.py`` child environment is the host env **minus**
``AWBW_LEARNER_GREEDY_MIX`` / ``AWBW_CAPTURE_MOVE_GATE`` (see ``rl.train_launch_env``)
so PowerShell or orchestrator merges cannot override curriculum argv; ``train.py`` republishes
those from flags after parsing.

Before ``train.py`` starts, Cython extensions are rebuilt (``build_ext``): on Windows
``scripts/rebuild_cython_extensions.py`` (avoids in-place ``.pyd`` lock issues), else
``setup.py build_ext --inplace``. Use ``--skip-cython-rebuild`` to skip when extensions
are already current. If copy fails with WinError 32, check for **multiprocessing.spawn**
workers (VecEnv children), not only ``train.py`` — run ``python scripts/diagnose_cython_lock.py``.

When ``--auto-apply`` is enabled, ``fleet_orchestrator`` may **terminate and respawn**
``train.py`` (rewriting ``fleet/<id>/train.pid``) **only when curriculum keys change**
(see ``CURRICULUM_RESTART_KEYS``).  The replacement process is built from the last
``applied_args.json`` with **only** the curriculum keys overlaid from the merged
``proposed_args.json`` — all other arguments (``--n-envs``, ``--max-env-steps``,
``--batch-size``, ``--training-backend``, live-game knobs, etc.) are **frozen** from
the first bootstrap invocation.  The main loop **detects a replacement** PID in
``train.pid`` that is still a live ``train.py`` for this machine and **adopts** it
instead of treating the event as fatal or tearing the orchestrator down.

Pass ``--no-orchestrator-auto-apply`` so ``fleet_orchestrator`` omits ``--auto-apply`` — no
terminate/respawn for curriculum or proposed-args drift (**live** ``train`` stays on prior argv until
you restart it yourself).

``--torch-compile`` sets ``AWBW_TORCH_COMPILE=1`` in the child ``train.py`` process environment
(see ``rl/self_play.py`` policy Inductor path). On Windows you still need MSVC C++ and a working
Triton install; default remains compile-off there for reliability.

**Spirit-broken heuristics** (calendar-day early termination + value gates; see ``rl/heuristic_termination.py``):
by default this script sets ``AWBW_SPIRIT_BROKEN=1`` for the ``train.py`` child **only when** the
variable is unset in the host environment. Use ``--no-spirit-broken`` to force off, or set
``AWBW_SPIRIT_BROKEN=0`` before launching.

**Throughput + NN metrics:** ``--fps-diag`` defaults **on** (pass ``--no-fps-diag`` to disable).
That forwards ``train.py --fps-diag`` and sets ``AWBW_FPS_DIAG=1`` for the child so
``logs/fps_diag.jsonl`` and ``[fps_diag]`` stdout stay live; learner loss is also appended to
``logs/nn_train.jsonl`` every step (see ``rl/async_impala.py`` / diagnostics callback in
``rl/self_play.py``).

``--throughput-tune`` (after live-game inject / snapshot refresh / ``n-envs`` bump) runs short
``train.py`` probes with ``--fps-diag`` and picks ``--n-envs`` by median ``env_steps_per_s_*``
from ``logs/fps_diag.jsonl``. The default probe ceiling is probe-derived

**Hybrid GPU/CPU opponents** (default on): when effective ``--n-envs`` is at least
``--hybrid-opponent-min-envs`` (default 8), the bootstrap adds env vars so the first
``--hybrid-gpu-opponent-workers`` sets the **semaphore size** (cap 4) for concurrent CUDA
checkpoint opponents; any worker acquires a permit non-blocking or falls back to CPU
when ``probe.json`` reports a GPU; remaining workers use **CPU** opponent inference with
``AWBW_CPU_WORKER_THREADS`` sized from physical cores. Use ``--no-hybrid-gpu-cpu-opponents`` to
skip. This pairs with ``rl/self_play.py`` (``AWBW_ALLOW_CUDA_OPPONENT``, ``AWBW_N_LEAN_WORKERS``).
(``max(1, physical_cores-2)`` with a RAM heuristic, capped at
``tools.propose_train_args.THROUGHPUT_TUNE_MAX_ENVS_CAP``) for **all** machines including
pc-b — not ``PC_B_MAX_ENVS``, so higher core counts can be measured. Probes are skipped when
host RAM or CPU (after ``--throughput-tune-host-wait-s``) already exceeds
``--throughput-tune-max-host-ram-pct`` / ``--throughput-tune-max-host-cpu-pct`` (default 90%),
leaving the baseline ``n-envs``. On success it also writes
``fleet/<id>/operator_train_args_override.json`` with ``--n-envs`` and a PPO-valid
``--batch-size`` so orchestrator refresh keeps that geometry.

By default, when ``replays/amarinner_my_games`` (or ``--live-games-dir``) contains export
subfolders (``tools/amarriner_export_my_games_replays.py``), this script:

- Refreshes each ``engine_snapshot.pkl`` from that game's ``live_replay.json`` when the JSON is
  newer (or the pkl is missing), so the first Subproc envs start from the **latest** state
  in the on-disk live replay. Use ``--no-refresh-live-snapshots`` to skip (e.g. slow disks).
- Appends ``--live-snapshot-dir`` and one ``--live-games-id`` per game (``train.py`` dedicates
  the first N workers to those snapshots; remaining envs use normal curriculum).
- Appends ``--live-learner-seats`` so the training POV matches line 1 of ``secrets.txt`` in each
  game (``meta.json`` + ``live_replay.json``), unless you pass ``--live-learner-seats`` yourself.
- Raises effective ``--n-envs`` to at least N when needed. Use ``--no-auto-live-games`` to
  disable all of the above auto behavior.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Windows: first try to signal the train process group (CREATE_NEW_PROCESS_GROUP)
# so Python can run KeyboardInterrupt / SB3 checkpoint path; TerminateProcess is a last resort.
if sys.platform == "win32":
    _SIG_BREAK = getattr(signal, "CTRL_BREAK_EVENT", None)
else:
    _SIG_BREAK = None

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_secrets_username() -> str | None:
    """First non-empty line of ``secrets.txt`` (Amarriner login); used for live POV."""
    p = REPO_ROOT / "secrets.txt"
    if not p.is_file():
        return None
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return None
    return lines[0]


def _infer_live_learner_seats(
    live_dir: Path, gids: list[int], username: str, log: logging.Logger
) -> list[int] | None:
    """
    For each *games_id*, map ``secrets`` user to engine seat 0/1 using the same rule as
    :func:`tools.oracle_zip_replay.map_snapshot_player_ids_to_engine` (see export ``meta.json``
    and ``live_replay.json``).
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tools.oracle_zip_replay import map_snapshot_player_ids_to_engine  # noqa: PLC0415

    ul = username.strip().lower()
    if not ul:
        return None
    out: list[int] = []
    for gid in gids:
        sub = live_dir / str(gid)
        meta_path = sub / "meta.json"
        lr_path = sub / "live_replay.json"
        if not meta_path.is_file() or not lr_path.is_file():
            log.warning(
                "live PPO games_id=%s: missing meta.json or live_replay.json; using learner_seat=0",
                gid,
            )
            out.append(0)
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            lr = json.loads(lr_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(
                "live PPO games_id=%s: could not read export (%s); learner_seat=0",
                gid,
                exc,
            )
            out.append(0)
            continue
        snap0 = lr.get("first_snap") or {}
        gs0 = lr.get("game_state_turn0") or {}
        players = gs0.get("players") or {}
        my_pid: int | None = None
        for _k, p in players.items():
            if not isinstance(p, dict):
                continue
            if str(p.get("users_username", "")).strip().lower() == ul:
                my_pid = int(p["players_id"])
                break
        if my_pid is None:
            log.warning(
                "live PPO games_id=%s: no users_username matching secrets user %r; learner_seat=0",
                gid,
                username,
            )
            out.append(0)
            continue
        try:
            co0 = int(meta["co_p0_id"])
            co1 = int(meta["co_p1_id"])
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("live PPO games_id=%s: bad meta CO ids (%s); learner_seat=0", gid, exc)
            out.append(0)
            continue
        try:
            m = map_snapshot_player_ids_to_engine(snap0, co0, co1)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "live PPO games_id=%s: map_snapshot_player_ids_to_engine failed (%s); learner_seat=0",
                gid,
                exc,
            )
            out.append(0)
            continue
        seat = m.get(int(my_pid))
        if seat is None:
            log.warning(
                "live PPO games_id=%s: players_id %s not in awbw->engine map; learner_seat=0",
                gid,
                my_pid,
            )
            out.append(0)
            continue
        out.append(int(seat) & 1)
    return out


def _discover_live_games_subdirs(live_dir: Path) -> list[int]:
    """
    Subdirectories named with decimal ``games_id`` that contain an engine snapshot
    and/or ``live_replay.json`` (``replays/amarinner_my_games`` export layout).
    """
    out: list[int] = []
    if not live_dir.is_dir():
        return out
    for sub in sorted(live_dir.iterdir(), key=lambda p: p.name):
        if not sub.is_dir() or not sub.name.isdigit():
            continue
        if (sub / "engine_snapshot.pkl").is_file() or (sub / "live_replay.json").is_file():
            out.append(int(sub.name))
    return out


def _last_int_for_flag(argv: list[str], flag: str) -> int | None:
    last: int | None = None
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            try:
                last = int(argv[i + 1], 0)
            except ValueError:
                pass
    return last


def _envelopes_jsonable_to_tuples(
    raw: list[Any],
) -> list[tuple[int, int, list[dict[str, Any]]]]:
    """Invert ``_envelopes_to_jsonable`` from ``tools/amarriner_export_my_games_replays``."""
    out: list[tuple[int, int, list[dict[str, Any]]]] = []
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        acts = row[2]
        if not isinstance(acts, list):
            continue
        out.append((int(row[0]), int(row[1]), [a for a in acts if isinstance(a, dict)]))
    return out


def _max_safe_n_envs_from_probe(
    probe: dict[str, Any], *, absolute_cap: int = 12
) -> int:
    """Delegate to :func:`tools.propose_train_args.max_safe_n_envs_from_probe`."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tools.propose_train_args import max_safe_n_envs_from_probe  # noqa: PLC0415

    return max_safe_n_envs_from_probe(probe, absolute_cap=absolute_cap)


def _default_throughput_tune_max_envs(machine_id: str, probe: dict[str, Any]) -> int:
    """
    Probe-derived ceiling for ``--throughput-tune`` (includes pc-b).

    Unlike initial ``proposed_args`` (pc-b still capped at ``PC_B_MAX_ENVS`` for
    stability), the sweep may try higher ``n_envs`` up to
    :data:`tools.propose_train_args.THROUGHPUT_TUNE_MAX_ENVS_CAP` when hardware
    and RAM heuristic allow.
    """
    del machine_id  # same formula for all fleet ids; explicit cap is intentional
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tools.propose_train_args import (  # noqa: PLC0415
        THROUGHPUT_TUNE_MAX_ENVS_CAP,
    )

    return _max_safe_n_envs_from_probe(probe, absolute_cap=int(THROUGHPUT_TUNE_MAX_ENVS_CAP))


def _prepare_proposed_live_games(
    proposed: dict[str, Any],
    train_extra: list[str],
    live_dir: Path,
    *,
    no_auto_live: bool,
    no_refresh: bool,
    map_pool: Path,
    maps_dir: Path,
    log: logging.Logger,
) -> tuple[list[str], list[int]]:
    """
    Run live PPO inject, optional snapshot refresh, and ``n-envs`` floor **before**
    throughput tuning or final ``train.py`` argv build.
    """
    extra, gids = _inject_live_games_train_extra(
        train_extra, live_dir, no_auto=no_auto_live, log=log
    )
    if gids:
        _refresh_live_engine_snapshots_if_stale(
            live_dir,
            gids,
            log=log,
            no_refresh=no_refresh,
            map_pool=map_pool,
            maps_dir=maps_dir,
        )
        _bump_n_envs_in_proposed_for_live(proposed, len(gids), log)
    return extra, gids


def _bump_n_envs_in_proposed_for_live(
    proposed: dict[str, Any], n_live: int, log: logging.Logger
) -> None:
    """Set ``proposed['args']['--n-envs']`` to at least *n_live* (fleet default is 4 if unset)."""
    if n_live <= 0:
        return
    am: dict[str, Any] = dict(proposed.get("args") or {})
    raw = am.get("--n-envs", None)
    if raw is None:
        base = 4
    else:
        try:
            base = int(raw)
        except (TypeError, ValueError):
            base = 4
    need = max(base, n_live)
    if need > base:
        log.info(
            "auto live PPO: raising --n-envs in proposed args from %s to %s "
            "(%d live game worker slot(s) required)",
            base,
            need,
            n_live,
        )
    am["--n-envs"] = need
    proposed["args"] = am


def _ensure_train_argv_n_envs_for_live(
    train_argv: list[str], n_live: int, log: logging.Logger
) -> None:
    """
    Append ``--n-envs`` if the effective value (last wins) is still below *n_live*.

    Catches a too-low value from ``--train-extra-args`` after ``proposed`` is merged
    in :func:`_build_train_argv`.
    """
    if n_live <= 0:
        return
    n_last = _last_int_for_flag(train_argv, "--n-envs")
    if n_last is None:
        n_last = 4
    need = max(n_last, n_live)
    if need > n_last:
        train_argv.extend(["--n-envs", str(need)])
        log.info(
            "auto live PPO: appended --n-envs %s (argv had %s; need at least %d live slot(s))",
            need,
            n_last,
            n_live,
        )


def _refresh_live_engine_snapshots_if_stale(
    live_dir: Path,
    gids: list[int],
    *,
    log: logging.Logger,
    no_refresh: bool,
    map_pool: Path,
    maps_dir: Path,
) -> None:
    """
    Rebuild ``engine_snapshot.pkl`` from ``live_replay.json`` + ``meta.json`` when the JSON is
    newer, so each live env reset loads the latest state captured in the export stream.
    """
    if no_refresh or not gids:
        return
    if not map_pool.is_file():
        log.warning("refresh live snapshots: missing map pool %s; skip", map_pool)
        return
    if not maps_dir.is_dir():
        log.warning("refresh live snapshots: missing maps dir %s; skip", maps_dir)
        return
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from rl.live_snapshot import write_live_snapshot  # noqa: PLC0415
    from tools.desync_audit_amarriner_live import (  # noqa: PLC0415
        build_live_engine_state_from_fetched,
    )

    for gid in gids:
        sub = live_dir / str(int(gid))
        lr = sub / "live_replay.json"
        mpath = sub / "meta.json"
        pkl = sub / "engine_snapshot.pkl"
        if not lr.is_file() or not mpath.is_file():
            log.debug(
                "refresh live snapshot: games_id=%s skip (need live_replay.json + meta.json)", gid
            )
            continue
        try:
            younger = (not pkl.is_file()) or (lr.stat().st_mtime > pkl.stat().st_mtime)
        except OSError as exc:
            log.warning("refresh live snapshot: games_id=%s stat failed: %s", gid, exc)
            continue
        if not younger:
            log.debug("refresh live snapshot: games_id=%s pkl up to date with live_replay.json", gid)
            continue
        try:
            meta = json.loads(mpath.read_text(encoding="utf-8"))
            payload = json.loads(lr.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("refresh live snapshot: games_id=%s read failed: %s", gid, exc)
            continue
        try:
            envs_raw = payload.get("envelopes") or []
            envs = _envelopes_jsonable_to_tuples(envs_raw)
            snap0 = payload.get("first_snap") or {}
            gs0 = payload.get("game_state_turn0") or {}
            per_turn = payload.get("per_turn_units") or []
            state, awbw = build_live_engine_state_from_fetched(
                meta, envs, snap0, gs0, per_turn, map_pool=map_pool, maps_dir=maps_dir
            )
            write_live_snapshot(
                pkl, state, games_id=int(gid), learner_seat=0, awbw_to_engine=awbw
            )
            log.info(
                "refreshed engine_snapshot.pkl from live_replay.json (games_id=%s, turn/phase at end of stream)",
                gid,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "refresh live snapshot: games_id=%s engine rebuild failed (keeping existing pkl if any): %s",
                gid,
                exc,
            )


def _inject_live_games_train_extra(
    train_extra: list[str],
    live_dir: Path | None,
    *,
    no_auto: bool,
    log: logging.Logger,
) -> tuple[list[str], list[int]]:
    """
    Append ``--live-snapshot-dir`` and ``--live-games-id`` for each export under
    *live_dir*, unless ``--no-auto-live-games`` or the operator already passed
    ``--live-games-id`` in *train_extra*.
    """
    if no_auto or live_dir is None:
        return train_extra, []
    if not live_dir.is_dir():
        log.info(
            "live games dir not found (%s); skip auto --live-games-id",
            live_dir,
        )
        return train_extra, []
    gids = _discover_live_games_subdirs(live_dir)
    if not gids:
        log.info("no live game subdirs under %s; skip auto --live-games-id", live_dir)
        return train_extra, []
    if any(t == "--live-games-id" for t in train_extra):
        log.info(
            "train_extra_args already sets --live-games-id; skip auto from %s",
            live_dir,
        )
        return train_extra, []
    out = list(train_extra)
    out.extend(["--live-snapshot-dir", str(live_dir.resolve())])
    for gid in gids:
        out.extend(["--live-games-id", str(gid)])
    log.info(
        "auto live PPO: %d env slot(s) for games_id=%s (snapshot dir %s)",
        len(gids),
        gids,
        live_dir.resolve(),
    )
    if any(t == "--live-learner-seats" for t in train_extra):
        log.info("train_extra_args already has --live-learner-seats; skip auto POV from secrets")
        return out, gids
    uname = _read_secrets_username()
    if not uname:
        log.warning(
            "auto live PPO: missing or empty %s — cannot set --live-learner-seats; "
            "train defaults to learner_seat=0 for every live game",
            REPO_ROOT / "secrets.txt",
        )
        return out, gids
    seats = _infer_live_learner_seats(live_dir, gids, uname, log)
    if seats is not None and len(seats) == len(gids):
        out.extend(["--live-learner-seats", ",".join(str(s) for s in seats)])
        log.info("auto --live-learner-seats for user %r: %s", uname, seats)
    return out, gids


def _rebuild_cython_extensions(log: logging.Logger) -> int:
    """
    Rebuild native extensions so ``train.py`` loads fresh ``.pyd`` / ``.so`` artifacts.

    Windows: ``scripts/rebuild_cython_extensions.py`` (``build/`` + copy; avoids
    in-place overwrites of loaded DLLs). Other platforms: ``setup.py build_ext --inplace``.
    """
    if sys.platform == "win32":
        cmd: list[str] = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "rebuild_cython_extensions.py"),
        ]
    else:
        cmd = [sys.executable, str(REPO_ROOT / "setup.py"), "build_ext", "--inplace"]
    log.info("recompiling Cython: %s", " ".join(shlex.quote(c) for c in cmd))
    try:
        pr = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        log.error("Cython rebuild could not start: %s", exc)
        return 1
    if pr.returncode != 0:
        log.error(
            "Cython rebuild failed (exit %s). stdout: %s stderr: %s",
            pr.returncode,
            (pr.stdout or "").strip() or "(empty)",
            (pr.stderr or "").strip() or "(empty)",
        )
        return int(pr.returncode) if pr.returncode is not None else 1
    if pr.stdout and pr.stdout.strip():
        log.info("Cython build output:\n%s", pr.stdout.rstrip())
    return 0


def _proposed_args_content_sha256(proposed_doc: dict[str, Any]) -> Optional[str]:
    """Delegate to ``fleet_orchestrator.proposed_args_content_sha256`` (single source of truth)."""
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import fleet_orchestrator as fo  # noqa: PLC0415

    return fo.proposed_args_content_sha256(proposed_doc)


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _game_log_path_for_initial_curriculum(machine_id: str) -> Path:
    """Match fleet_orchestrator's per-machine game-log fallback."""
    per_machine = REPO_ROOT / "logs" / str(machine_id).strip() / "game_log.jsonl"
    if per_machine.is_file():
        return per_machine
    return REPO_ROOT / "logs" / "game_log.jsonl"


def _merge_curriculum_for_initial_launch(
    proposed: dict[str, Any],
    *,
    machine_id: str,
    state_path: Path,
    log: logging.Logger,
    window_games: int = 100,
    write_state: bool = True,
) -> dict[str, Any]:
    """
    Apply curriculum overrides before the first ``train.py`` spawn.

    ``tools.propose_train_args`` only writes probe-owned geometry. Without this merge,
    ``build_train_argv_from_proposed_args`` fills omitted map/CO defaults (Misery Andy),
    and the trainer starts pinned even when ``curriculum_state.json`` is already stage D+
    where ``--map-id: null`` should omit the flag.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tools import curriculum_advisor as ca  # noqa: PLC0415

    gpath = _game_log_path_for_initial_curriculum(machine_id)
    prev = ca.read_state(state_path)
    prop, st_new = ca.compute_proposal_stable(
        gpath,
        prev,
        window_games=int(window_games),
        machine_id=machine_id,
    )

    merged_args = dict(proposed.get("args") or {})
    merged_args.update(prop.args_overrides)
    merged = {
        **proposed,
        "args": merged_args,
        "curriculum": {"stage": prop.stage_name},
    }
    rs = str(proposed.get("reasoning") or "").strip()
    cr = f"curriculum: {prop.reason}" if prop.reason else ""
    if rs and cr:
        merged["reasoning"] = f"{rs}; {cr}"
    elif cr:
        merged["reasoning"] = cr

    if write_state:
        ca.write_state(state_path, st_new)

    log.info(
        "initial curriculum merge: stage=%s game_log=%s overrides=%s",
        prop.stage_name,
        gpath,
        sorted(prop.args_overrides),
    )
    return merged


_OPERATOR_TRAIN_ARGS_OVERRIDE_BASENAME = "operator_train_args_override.json"


def _merge_default_opening_book_train_args(
    proposed: dict[str, Any],
    *,
    log: logging.Logger,
    enabled: bool = True,
) -> None:
    """
    If ``proposed['args']`` does not set opening-book train flags, apply
    :data:`tools.curriculum_advisor.DEFAULT_OPENING_BOOK_TRAIN_ARGS` (precombat std-pool book,
    both seats). Keys already present are left unchanged.
    """
    if not enabled:
        return
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tools.curriculum_advisor import DEFAULT_OPENING_BOOK_TRAIN_ARGS  # noqa: PLC0415

    am: dict[str, Any] = proposed.setdefault("args", {})
    added: list[str] = []
    for k, v in DEFAULT_OPENING_BOOK_TRAIN_ARGS.items():
        if k not in am:
            am[k] = v
            added.append(k)
    if added:
        log.info(
            "default opening book: filled missing arg(s) %s (same as curriculum "
            "DEFAULT_OPENING_BOOK_TRAIN_ARGS)",
            ", ".join(added),
        )


def _merge_operator_train_args_override_into_proposed(
    proposed: dict[str, Any],
    *,
    fleet_dir: Path,
    log: logging.Logger,
) -> dict[str, Any]:
    """Apply ``fleet/<id>/operator_train_args_override.json`` ``args`` on top of *proposed*.

    Matches :mod:`scripts.fleet_orchestrator` tick path (curriculum merge, then per-flag override)
    so the **first** ``train.py`` matches what the next orchestrator refresh would have applied.
    """
    path = fleet_dir / _OPERATOR_TRAIN_ARGS_OVERRIDE_BASENAME
    if not path.is_file():
        return proposed
    try:
        raw = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("operator override unreadable %s: %s", path, exc)
        return proposed
    override_args = raw.get("args") if isinstance(raw, dict) else None
    if not isinstance(override_args, dict) or not override_args:
        return proposed
    out = {**proposed}
    am = dict(out.get("args") or {})
    applied: list[str] = []
    for k, ov in override_args.items():
        if not isinstance(k, str) or not k.startswith("--"):
            log.warning("ignoring non-flag key in %s: %r", path, k)
            continue
        am[k] = ov
        applied.append(k)
    out["args"] = am
    if applied:
        log.info(
            "operator_train_args_override applied (%s): %s",
            path,
            ", ".join(sorted(applied)),
        )
    return out


def _tail_text_file(
    path: Path, *, max_bytes: int = 64_000, max_lines: int = 40
) -> str:
    """Last ``max_lines`` lines of a text file, reading only from the end (best-effort)."""
    try:
        if not path.is_file():
            return f"(not found: {path})"
    except OSError as exc:
        return f"(stat failed: {path}: {exc})"
    try:
        with path.open("rb") as f:
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
            except OSError as exc:
                return f"(seek failed: {path}: {exc})"
            f.seek(max(0, size - max_bytes), os.SEEK_SET)
            data = f.read().decode("utf-8", errors="replace")
    except OSError as exc:
        return f"(read failed: {path}: {exc})"
    lines = data.splitlines()
    return "\n".join(lines[-max_lines:]) if lines else "(empty)"


def _configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "start_solo_training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )


def _coerce_cli_value(val_str: str) -> Any:
    try:
        return int(val_str, 0)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        return val_str


def _train_extra_to_args_map(train_extra: list[str]) -> dict[str, Any]:
    """
    Shlex-style ``--flag`` / ``--flag value`` tokens into the same key shape as
    :func:`_argv_to_args_dict` (reuses that parser; train.py path is synthetic only).
    """
    if not train_extra:
        return {}
    return _argv_to_args_dict(
        [sys.executable, str((REPO_ROOT / "train.py").resolve()), *train_extra]
    )


def _proposed_args_synced_from_train_argv(
    proposed: dict[str, Any], train_argv: list[str]
) -> dict[str, Any]:
    """
    Rebuild the ``proposed`` document after a launch so on-disk state matches argv.

    :func:`_argv_to_args_dict` cannot see flags that were **omitted** because JSON had
    ``null`` (e.g. ``--map-id: null`` for random GL std pool). Replacing ``args``
    with only the argv-derived map would drop those ``null`` keys, and the next
    ``build_train_argv_from_proposed_args`` would re-inject fleet defaults
    (Misery ``123858``). Merge: argv wins for any flag present; prior ``args`` entries
    remain for keys absent from argv (including explicit JSON ``null``).
    """
    orig = dict(proposed.get("args") or {})
    from_argv = _argv_to_args_dict(train_argv)
    return {**proposed, "args": {**orig, **from_argv}}


def _argv_to_args_dict(train_argv: list[str]) -> dict[str, Any]:
    """
    Parse ``train.py`` argv (skip ``sys.executable`` and ``train.py`` path) into the
    ``proposed_args.json`` ``args`` map shape expected by ``build_train_argv_from_proposed_args``.
    """
    toks = train_argv[2:]
    out: dict[str, Any] = {}
    i = 0
    while i < len(toks):
        tok = toks[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok
        if i + 1 >= len(toks) or toks[i + 1].startswith("--"):
            if key == "--capture-move-gate":
                out[key] = "_FLAG_PRESENT"
            else:
                out[key] = True
            i += 1
        else:
            val = _coerce_cli_value(toks[i + 1])
            if key == "--live-games-id":
                prev = out.get(key)
                if prev is None:
                    out[key] = val
                elif isinstance(prev, list):
                    prev.append(val)
                else:
                    out[key] = [prev, val]
            else:
                out[key] = val
            i += 2
    return out


def _read_train_pid(path: Path) -> Optional[int]:
    try:
        line = path.read_text(encoding="utf-8").strip().splitlines()[0]
        return int(line.strip())
    except (OSError, ValueError, IndexError):
        return None


def _wait_for_orchestrator_train_replacement(
    pid_path: Path,
    old_pid: int,
    machine_id: str,
    *,
    log: logging.Logger,
    max_wait_s: float = 4.0,
) -> Optional[int]:
    """
    After ``fleet_orchestrator`` --auto-apply terminates the bootstrap-spawned train (often
    exit code 15 on Windows = SIGTERM-style), it respawns and rewrites ``train.pid`` slightly
    later. Without a short wait, the monitor loop can read the stale pid and treat restart as
    fatal, tearing down orchestrator + the replacement train.
    """
    t0 = time.monotonic()
    step_s = 0.12
    while time.monotonic() - t0 < max_wait_s:
        n = _read_train_pid(pid_path)
        if (
            n is not None
            and n != int(old_pid)
            and _pid_is_fleet_train_for_machine(n, machine_id)
            and _pid_is_live_python(n)
        ):
            log.info(
                "saw replacement train.py pid=%s in %s (orchestrator respawn; waited %.2fs)",
                n,
                pid_path,
                time.monotonic() - t0,
            )
            return int(n)
        time.sleep(step_s)
    return None


def _pid_is_live_python(pid: int) -> bool:
    import psutil

    try:
        if not psutil.pid_exists(pid):
            return False
        proc = psutil.Process(pid)
        if not proc.is_running():
            return False
        name = (proc.name() or "").lower()
        return name.startswith("python")
    except (psutil.Error, ValueError):
        return False


def _cmdline_mentions_train_py(cmdline: list[str]) -> bool:
    return any(part and "train.py" in part for part in cmdline)


def _cmdline_machine_id_match(cmdline: list[str], machine_id: str) -> bool:
    flat = " ".join(cmdline)
    if f"--machine-id {machine_id}" in flat or f"--machine-id={machine_id}" in flat:
        return True
    i = 0
    while i < len(cmdline):
        if cmdline[i] == "--machine-id" and i + 1 < len(cmdline):
            return cmdline[i + 1] == machine_id
        if cmdline[i].startswith("--machine-id="):
            return cmdline[i].split("=", 1)[-1] == machine_id
        i += 1
    return False


def _process_matches_train_for_machine(
    proc: Any, machine_id: str, cmdline: list[str]
) -> bool:
    import psutil

    if not _cmdline_mentions_train_py(cmdline):
        return False
    if _cmdline_machine_id_match(cmdline, machine_id):
        return True
    try:
        env = proc.environ()
    except (psutil.Error, AttributeError):
        env = {}
    return env.get("AWBW_MACHINE_ID") == machine_id


def _cmdline_orchestrator_for_pool(cmdline: list[str], machine_id: str) -> bool:
    if not any(part and "fleet_orchestrator.py" in part for part in cmdline):
        return False
    flat = " ".join(cmdline)
    if f"--pools {machine_id}" in flat or f"--pools={machine_id}" in flat:
        return True
    for i, tok in enumerate(cmdline):
        if tok == "--pools" and i + 1 < len(cmdline) and cmdline[i + 1] == machine_id:
            return True
    return False


def _find_cohort_conflicts(machine_id: str) -> list[tuple[int, list[str]]]:
    try:
        import psutil
    except ImportError:
        return []
    out: list[tuple[int, list[str]]] = []
    self_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        pid = proc.info.get("pid")
        if pid is None or pid == self_pid:
            continue
        raw = proc.info.get("cmdline")
        if not raw:
            continue
        cmdline = list(raw)
        name = (proc.info.get("name") or "").lower()
        if not name.startswith("python"):
            continue
        try:
            if _process_matches_train_for_machine(proc, machine_id, cmdline):
                out.append((int(pid), cmdline))
                continue
            if _cmdline_orchestrator_for_pool(cmdline, machine_id):
                out.append((int(pid), cmdline))
        except (psutil.Error, TypeError, ValueError):
            continue
    return out


def _cwd_under_repo(cwd: str, repo_root: Path) -> bool:
    """True if *cwd* resolves to *repo_root* or a subdirectory."""
    if not cwd:
        return False
    try:
        cwd_r = Path(cwd).resolve()
        repo_r = repo_root.resolve()
    except OSError:
        return False
    if cwd_r == repo_r:
        return True
    try:
        cwd_r.relative_to(repo_r)
        return True
    except ValueError:
        return False


def _find_cython_rebuild_blockers(repo_root: Path) -> list[tuple[int, list[str]]]:
    """
    Python processes that typically keep ``engine/*.pyd`` / ``rl/*.pyd`` mapped on Windows.

    Includes ``train.py`` **and** ``multiprocessing.spawn`` workers (e.g. SubprocVecEnv):
    those children run ``python -c from multiprocessing.spawn import spawn_main ...`` with
    no ``train.py`` in argv, so a train-only grep misses them — a common WinError 32 cause.
    """
    try:
        import psutil
    except ImportError:
        return []
    out: list[tuple[int, list[str]]] = []
    self_pid = os.getpid()
    repo_r = repo_root.resolve()
    rps = str(repo_r).lower()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        pid = proc.info.get("pid")
        if pid is None or int(pid) == self_pid:
            continue
        raw = proc.info.get("cmdline")
        if not raw:
            continue
        cmdline = list(raw)
        name = (proc.info.get("name") or "").lower()
        if not name.startswith("python"):
            continue
        try:
            pobj = psutil.Process(int(pid))
            try:
                cwd = str(pobj.cwd())
            except (psutil.Error, OSError):
                cwd = ""
            joined = " ".join(cmdline)
            low = joined.lower()
            blocked = False
            if _cmdline_mentions_train_py(cmdline):
                blocked = True
            elif "multiprocessing.spawn" in low and (
                _cwd_under_repo(cwd, repo_r) or rps in low
            ):
                blocked = True
            elif "pytest" in low and _cwd_under_repo(cwd, repo_r):
                blocked = True
            elif "ipykernel" in low and _cwd_under_repo(cwd, repo_r):
                blocked = True
            if blocked:
                out.append((int(pid), cmdline))
        except (psutil.Error, TypeError, ValueError):
            continue
    return out


def _poll_adopted_train_pid(pid: int) -> Optional[int]:
    """
    If ``train`` is still running, return ``None``.
    If it has exited, return a best-effort exit code (``1`` if unknown / reaped).
    """
    import psutil

    if not psutil.pid_exists(pid):
        return 1
    try:
        p = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 1
    if p.is_running():
        return None
    try:
        return int(p.wait(timeout=0.1))
    except psutil.TimeoutExpired:
        return None
    except psutil.NoSuchProcess:
        return 1


def _pid_is_fleet_train_for_machine(pid: int, machine_id: str) -> bool:
    import psutil

    try:
        proc = psutil.Process(pid)
        raw = proc.cmdline()
    except (psutil.Error, OSError, TypeError, ValueError):
        return False
    if not raw:
        return False
    cmdline = list(raw)
    return _process_matches_train_for_machine(proc, machine_id, cmdline)


def _terminate_process_tree_by_pid(pid: int, timeout_s: float = 30.0) -> None:
    """
    End ``train`` and any worker children (aligns with ``fleet_orchestrator`` restart helper).
    """
    import psutil

    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = list(proc.children(recursive=True)) + [proc]
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout_s)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def _ensure_no_zombie_cohort(*, machine_id: str, pid_path: Path) -> Optional[str]:
    """
    Return an error message if a prior cohort appears to be running; else ``None``.
    """
    try:
        import psutil  # noqa: F401
    except ImportError:
        return (
            "start_solo_training requires psutil for zombie protection (pip install psutil)."
        )

    if pid_path.is_file():
        tid = _read_train_pid(pid_path)
        if tid is not None and _pid_is_live_python(tid):
            return (
                f"Refusing to start: {pid_path} points to live Python PID {tid}. "
                "Stop the prior train cohort (or if this PID is stale, remove the pid file) "
                "before re-running start_solo_training."
            )
    conflicts = _find_cohort_conflicts(machine_id)
    if not conflicts:
        return None
    lines = [
        "Refusing to start: found Python process(es) already running train.py or "
        f"fleet_orchestrator.py for machine_id={machine_id!r}:"
    ]
    for pid, cmdline in conflicts:
        lines.append(f"  pid={pid} cmdline={' '.join(cmdline)!r}")
    lines.append("Stop these processes explicitly, then retry.")
    return "\n".join(lines)


def _ensure_training_backend_argv(train_argv: list[str], backend: str) -> None:
    """Append ``--training-backend`` unless ``train_extra`` already set it."""
    for tok in train_argv:
        if tok == "--training-backend":
            return
    train_argv.extend(["--training-backend", str(backend)])


def _build_train_argv(
    *,
    proposed: dict[str, Any],
    machine_id: str,
    train_extra: list[str],
    log_replay_frames: bool = False,
    fps_diag: bool = False,
) -> list[str]:
    """Delegate to :func:`fleet_orchestrator.build_train_argv_from_proposed_args` (single source of truth)."""
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import fleet_orchestrator as fo  # noqa: PLC0415

    am = dict(proposed.get("args") or {})
    if train_extra:
        # Merge before a single :func:`build_train_argv` emission so we do not duplicate
        # flags (e.g. two ``--n-envs`` from proposed + ``--train-extra-args``).
        am.update(_train_extra_to_args_map(train_extra))
    am["--machine-id"] = machine_id
    if log_replay_frames:
        am["--log-replay-frames"] = True
    if fps_diag:
        am["--fps-diag"] = True
    doc = {**proposed, "args": am}
    return fo.build_train_argv_from_proposed_args(doc, repo_root=REPO_ROOT)


def _clamp_ppo_batch_size_for_envs(
    args: dict[str, Any], *, n_envs: int, n_steps: int
) -> int:
    """``batch_size <= n_steps * n_envs`` for MaskablePPO; at least 1."""
    cap = int(n_steps) * int(n_envs)
    try:
        bs = int(args.get("--batch-size", 256))
    except (TypeError, ValueError):
        bs = 256
    return max(1, min(bs, cap))


def _write_operator_train_args_override_for_throughput_tune(
    fleet_dir: Path,
    *,
    n_envs: int,
    batch_size: int,
    info: dict[str, Any],
    log: logging.Logger,
) -> None:
    """Merge-winning train geometry for orchestrator ticks (sparse ``args`` map)."""
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import fleet_orchestrator as fo  # noqa: PLC0415

    path = fleet_dir / fo.OPERATOR_TRAIN_ARGS_OVERRIDE_NAME
    doc: dict[str, Any] = {
        "args": {
            "--n-envs": int(n_envs),
            "--batch-size": int(batch_size),
        },
        "source": "throughput_tune",
        "throughput_tune": {
            "applied_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "winner_median": info.get("winner_median"),
            "winner_n_envs": int(n_envs),
        },
    }
    _atomic_write_json(path, doc)
    log.info("throughput_tune: wrote %s (--n-envs=%s --batch-size=%s)", path, n_envs, batch_size)


def _compose_train_argv_with_live_ppo(
    proposed: dict[str, Any],
    *,
    machine_id: str,
    train_extra: list[str],
    live_dir: Path,
    no_auto_live: bool,
    no_refresh: bool,
    map_pool: Path,
    maps_dir: Path,
    log: logging.Logger,
    log_replay_frames: bool = False,
    fps_diag: bool = False,
) -> tuple[list[str], list[int]]:
    """
    Inject live PPO args, optional snapshot refresh, n-envs floor, then build the train argv.
    """
    extra, gids = _prepare_proposed_live_games(
        proposed,
        train_extra,
        live_dir,
        no_auto_live=no_auto_live,
        no_refresh=no_refresh,
        map_pool=map_pool,
        maps_dir=maps_dir,
        log=log,
    )
    train_argv = _build_train_argv(
        proposed=proposed,
        machine_id=machine_id,
        train_extra=extra,
        log_replay_frames=log_replay_frames,
        fps_diag=fps_diag,
    )
    _ensure_train_argv_n_envs_for_live(train_argv, len(gids), log)
    return train_argv, gids


def _tune_n_envs_throughput_inplace(
    proposed: dict[str, Any],
    *,
    machine_id: str,
    train_extra: list[str],
    live_gids: list[int],
    max_envs: int,
    per_candidate_s: float,
    min_iters: int,
    max_host_ram_pct: float,
    max_host_cpu_pct: float,
    host_wait_s: float,
    log_replay_frames: bool,
    log: logging.Logger,
) -> dict[str, Any]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tools.throughput_tune import choose_n_envs_throughput  # noqa: PLC0415

    try:
        n_steps0 = int((proposed.get("args") or {}).get("--n-steps", 512))
    except (TypeError, ValueError):
        n_steps0 = 512

    def make_probe_argv(n: int) -> list[str]:
        doc = copy.deepcopy(proposed)
        pam = dict(doc.get("args") or {})
        pam["--n-envs"] = int(n)
        pam["--batch-size"] = _clamp_ppo_batch_size_for_envs(
            pam, n_envs=int(n), n_steps=int(n_steps0)
        )
        doc["args"] = pam
        full = _build_train_argv(
            proposed=doc,
            machine_id=machine_id,
            train_extra=list(train_extra),
            log_replay_frames=log_replay_frames,
            fps_diag=True,
        )
        return full[2:]

    fleet_dir = REPO_ROOT / "fleet" / str(machine_id).strip()
    winner, info = choose_n_envs_throughput(
        machine_id=str(machine_id).strip(),
        proposed=proposed,
        gids=list(live_gids),
        max_envs=int(max_envs),
        per_candidate_s=float(per_candidate_s),
        min_iters=int(min_iters),
        max_host_ram_pct=float(max_host_ram_pct),
        max_host_cpu_pct=float(max_host_cpu_pct),
        host_wait_s=float(host_wait_s),
        repo_root=REPO_ROOT,
        fleet_dir=fleet_dir,
        log=log,
        make_probe_argv=make_probe_argv,
    )
    if winner < 1:
        log.warning("throughput_tune: no winner applied (%s)", info.get("abort_reason"))
        return info
    pam = proposed.setdefault("args", {})
    pam["--n-envs"] = int(winner)
    pam["--batch-size"] = _clamp_ppo_batch_size_for_envs(
        pam, n_envs=int(winner), n_steps=int(n_steps0)
    )
    _write_operator_train_args_override_for_throughput_tune(
        fleet_dir,
        n_envs=int(winner),
        batch_size=int(pam["--batch-size"]),
        info=info,
        log=log,
    )
    rs = proposed.get("reasoning")
    if isinstance(rs, str) and rs.strip():
        proposed["reasoning"] = (
            rs.rstrip() + f" | throughput_tune: n_envs={int(winner)}"
        )
    log.info("throughput_tune: chose --n-envs=%s (%s)", winner, info)
    return info


def _train_popen_environ(env_overlay: dict[str, str]) -> dict[str, str]:
    """
    Environment for ``train.py``: full host env minus CLI-owned curriculum keys
    (see ``rl.train_launch_env``), then *env_overlay* (machine id, perf defaults, …).
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from rl.train_launch_env import environ_for_train_subprocess  # noqa: PLC0415

    return {**environ_for_train_subprocess(), **env_overlay}


def _launch_env(
    *,
    machine_id: str,
    log_replay_frames: bool = False,
    torch_compile: bool = False,
    fps_diag: bool = False,
    no_spirit_broken: bool = False,
) -> dict[str, str]:
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import fleet_orchestrator as fo  # noqa: PLC0415

    env: dict[str, str] = {
        **fo.DEFAULT_TRAIN_PERF_ENV,
        "AWBW_MACHINE_ID": machine_id,
        # Capture-move gate: do not force here — curriculum / proposed_args adds
        # ``--capture-move-gate`` for stage_a/b only; stage_c+ drops it.
        "AWBW_TRACK_PER_WORKER_TIMES": "1",
        # NOTE: AWBW_ASYNC_VEC=1 disabled - incompatible with SB3's MaskablePPO
        # The env loading/setting fails with AsyncVectorEnv. Kept for reference.
    }
    if log_replay_frames:
        env["AWBW_LOG_REPLAY_FRAMES"] = "1"
    if torch_compile:
        env["AWBW_TORCH_COMPILE"] = "1"
    if fps_diag:
        env["AWBW_FPS_DIAG"] = "1"
    # Spirit-broken (rl/heuristic_termination.py): default on for solo bootstrap if host left unset.
    if no_spirit_broken:
        env["AWBW_SPIRIT_BROKEN"] = "0"
    elif os.environ.get("AWBW_SPIRIT_BROKEN") is None:
        env["AWBW_SPIRIT_BROKEN"] = "1"
    return env


def _read_probe_json(fleet_dir: Path) -> dict[str, Any]:
    p = fleet_dir / "probe.json"
    if not p.is_file():
        return {}
    try:
        return _read_json(p)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def _hybrid_gpu_cpu_opponent_env(
    *,
    n_envs: int | None,
    probe: dict[str, Any],
    enabled: bool,
    min_n_envs: int,
    cuda_opponent_workers: int,
    log: logging.Logger,
) -> dict[str, str]:
    """
    For large ``n_envs`` with a CUDA probe, enable ``AWBW_GPU_OPPONENT_POOL`` so workers
    share a process-wide semaphore (``AWBW_GPU_OPPONENT_POOL_SIZE``): up to *N* concurrent
    GPU checkpoint forwards; others use CPU for that predict. Also set uniform
    ``AWBW_WORKER_OMP_THREADS`` from core count.

    See ``rl/self_play.gpu_opponent_pool_enabled`` and ``BoundedSemaphore`` in ``_build_vec_env``.
    """
    out: dict[str, str] = {}
    if not enabled or n_envs is None:
        return out
    ne = int(n_envs)
    if ne < int(min_n_envs):
        return out

    cpu_block = probe.get("cpu") or {}
    phys = int(cpu_block.get("physical_cores") or 0)
    if phys <= 0:
        phys = 8

    gpu_block = probe.get("gpu") or {}
    cuda_ok = bool(gpu_block.get("available"))

    from rl.self_play import OPPONENT_CUDA_WORKERS_MAX  # noqa: PLC0415

    cap = max(0, min(int(cuda_opponent_workers), int(OPPONENT_CUDA_WORKERS_MAX), ne))

    if cuda_ok:
        permits = max(1, min(int(cuda_opponent_workers), int(OPPONENT_CUDA_WORKERS_MAX), ne))
        out["AWBW_ALLOW_CUDA_OPPONENT"] = "1"
        out["AWBW_GPU_OPPONENT_POOL"] = "1"
        out["AWBW_GPU_OPPONENT_POOL_SIZE"] = str(permits)
        spare = max(1, phys - 2)
        wt = max(2, min(8, spare // max(1, ne)))
        out["AWBW_WORKER_OMP_THREADS"] = str(wt)
        log.info(
            "hybrid opponents: GPU semaphore pool size=%s (non-blocking acquire or CPU infer); "
            "AWBW_WORKER_OMP_THREADS=%s (physical_cores=%s, n_envs=%s)",
            permits,
            wt,
            phys,
            ne,
        )
    else:
        spare = max(1, phys - 2)
        wt = max(2, min(8, spare // max(1, ne)))
        out["AWBW_WORKER_OMP_THREADS"] = str(wt)
        log.info(
            "hybrid opponents: all CPU (cuda_ok=%s, cuda_cap=%s, n_envs=%s); "
            "AWBW_WORKER_OMP_THREADS=%s",
            cuda_ok,
            cap,
            ne,
            wt,
        )
    return out


def _terminate_process(
    proc: Optional[subprocess.Popen], *, timeout_s: float, windows_ctrl_break: bool = False
) -> None:
    if proc is None or proc.poll() is not None:
        return
    if windows_ctrl_break and sys.platform == "win32" and _SIG_BREAK is not None:
        try:
            os.kill(proc.pid, _SIG_BREAK)
        except OSError:
            pass
        else:
            try:
                proc.wait(timeout=min(30.0, max(5.0, timeout_s * 0.5)))
                if proc.poll() is not None:
                    return
            except subprocess.TimeoutExpired:
                pass
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--machine-id", type=str, required=True)
    ap.add_argument(
        "--auto-apply",
        action="store_true",
        default=False,
        help=(
            "Ignored (compatibility shim). Fleet orchestrator is started with curriculum respawn enabled "
            "by default unless you pass --no-orchestrator-auto-apply."
        ),
    )
    ap.add_argument(
        "--no-orchestrator-auto-apply",
        action="store_true",
        default=False,
        help=(
            "Do not pass --auto-apply to fleet_orchestrator: no train.py restarts for "
            "proposed_args drift / zombie heal (diagnostics; overrides --auto-apply for the "
            "orchestrator only)."
        ),
    )
    ap.add_argument(
        "--orchestrator-tick-s",
        type=float,
        default=300.0,
        help="Orchestrator sleep between ticks (seconds); default 300 = 5 min",
    )
    ap.add_argument(
        "--train-extra-args",
        type=str,
        default="",
        help="Extra train.py CLI (quoted), e.g. '--ent-coef 0.02'",
    )
    ap.add_argument(
        "--dry-run-bootstrap",
        action="store_true",
        help="Print planned commands and exit without spawning processes",
    )
    ap.add_argument(
        "--log-replay-frames",
        action="store_true",
        help=(
            "Enable AWBW_LOG_REPLAY_FRAMES / train.py --log-replay-frames so game_log "
            "includes frames for the /replay/ viewer (large disk use)."
        ),
    )
    ap.add_argument(
        "--fps-diag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Forward to train.py --fps-diag (AWBW_FPS_DIAG=1): logs/fps_diag.jsonl + [fps_diag] stdout "
            "for sync and async; async rows/lines include NN loss fields. "
            "NN scalars are also written to logs/nn_train.jsonl during learning. "
            "Default: on. Use --no-fps-diag to disable."
        ),
    )
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=REPO_ROOT / "logs",
        help="Directory for start_solo_training.log",
    )
    ap.add_argument(
        "--skip-cython-rebuild",
        action="store_true",
        help="Do not run build_ext before train.py (use when Cython sources are unchanged).",
    )
    ap.add_argument(
        "--torch-compile",
        action="store_true",
        default=False,
        help=(
            "Set AWBW_TORCH_COMPILE=1 for the train.py process so rl/self_play may apply "
            "torch.compile to the policy (GPU; on Windows also needs MSVC C++ + Triton)."
        ),
    )
    ap.add_argument(
        "--no-spirit-broken",
        action="store_true",
        default=False,
        help=(
            "Disable spirit_broken calendar heuristics for this run: sets AWBW_SPIRIT_BROKEN=0 "
            "in the train.py child env (overrides the default on from start_solo_training when "
            "the host did not set AWBW_SPIRIT_BROKEN)."
        ),
    )
    ap.add_argument(
        "--train-bootstrap-grace-s",
        type=float,
        default=180.0,
        help=(
            "Forwarded to fleet_orchestrator --train-bootstrap-grace-s when >0: suppress "
            "args-drift restarts and zombie-heal briefly after applied_args.json is written "
            "(avoids killing the first train on tick 1 when curriculum/MCTS refresh proposed). "
            "Set 0 to disable."
        ),
    )
    ap.add_argument(
        "--orchestrator-curriculum-window-games",
        type=int,
        default=None,
        metavar="N",
        help=(
            "When set, forwarded to fleet_orchestrator --curriculum-window-games (rolling "
            "game-log window for curriculum metrics). Omit to use orchestrator default (200)."
        ),
    )
    ap.add_argument(
        "--live-games-dir",
        type=Path,
        default=REPO_ROOT / "replays" / "amarinner_my_games",
        help=(
            "If this folder has per-games_id subdirs (from tools/amarriner_export_my_games_replays.py), "
            "by default: refresh pkl from live_replay when newer, then append --live-snapshot-dir, one "
            "--live-games-id per game, optional --live-learner-seats from secrets + meta, and raise --n-envs. "
            "Use --no-auto-live-games or --no-refresh-live-snapshots to opt out of parts of this."
        ),
    )
    ap.add_argument(
        "--no-auto-live-games",
        action="store_true",
        help="Do not append --live-games-id from --live-games-dir.",
    )
    ap.add_argument(
        "--no-refresh-live-snapshots",
        action="store_true",
        help=(
            "Do not rebuild each engine_snapshot.pkl from live_replay.json when the JSON is "
            "newer (default: refresh so the engine starts from the end of the on-disk live stream)."
        ),
    )
    ap.add_argument(
        "--live-map-pool",
        type=Path,
        default=REPO_ROOT / "data" / "gl_map_pool.json",
        help="Map pool for offline live_replay -> pkl refresh (default: data/gl_map_pool.json).",
    )
    ap.add_argument(
        "--live-maps-dir",
        type=Path,
        default=REPO_ROOT / "data" / "maps",
        help="Maps directory for live_replay -> pkl refresh (default: data/maps).",
    )
    ap.add_argument(
        "--throughput-tune",
        action="store_true",
        help=(
            "After live-game prep, run short train.py probes (--fps-diag) to pick --n-envs. "
            "Skipped on --dry-run-bootstrap (prints probe range only)."
        ),
    )
    ap.add_argument(
        "--throughput-tune-max-envs",
        type=int,
        default=None,
        help=(
            "Max n_envs to probe (default: probe-derived max(1, phys_cores-2) "
            "with RAM heuristic, capped at propose_train_args.THROUGHPUT_TUNE_MAX_ENVS_CAP; "
            "same for all machine_id including pc-b — unlike initial proposed_args, "
            "tune is not limited to PC_B_MAX_ENVS)."
        ),
    )
    ap.add_argument(
        "--throughput-tune-per-candidate-s",
        type=float,
        default=105.0,
        help="Wall time budget per n_envs probe (subprocess timeout).",
    )
    ap.add_argument(
        "--throughput-tune-min-iters",
        type=int,
        default=0,
        help="Probe --iters (0 = auto max(32768, 2*n_steps*n_envs) per candidate).",
    )
    ap.add_argument(
        "--throughput-tune-max-host-ram-pct",
        type=float,
        default=90.0,
        help="Abort tuning (keep baseline n_envs) if host RAM %% exceeds this after wait.",
    )
    ap.add_argument(
        "--throughput-tune-max-host-cpu-pct",
        type=float,
        default=90.0,
        help="Abort tuning if host CPU %% exceeds this after wait.",
    )
    ap.add_argument(
        "--throughput-tune-host-wait-s",
        type=float,
        default=45.0,
        help="Sleep before host headroom check (stabilizes psutil sample).",
    )
    ap.add_argument(
        "--hybrid-opponent-min-envs",
        type=int,
        default=8,
        help=(
            "If effective --n-envs >= this value, set hybrid opponent env vars: first "
            "N workers may use CUDA for checkpoint opponents, rest use CPU (default 8). "
            "Set high (e.g. 999) to effectively disable unless you pass --no-hybrid-gpu-cpu-opponents."
        ),
    )
    ap.add_argument(
        "--hybrid-gpu-opponent-workers",
        type=int,
        default=4,
        help=(
            "When hybrid applies: ``AWBW_GPU_OPPONENT_POOL_SIZE`` (semaphore permits for concurrent "
            "CUDA checkpoint opponents; cap rl.self_play.OPPONENT_CUDA_WORKERS_MAX, default 4). "
            "Extra workers use CPU for that predict when the pool is saturated."
        ),
    )
    ap.add_argument(
        "--no-hybrid-gpu-cpu-opponents",
        action="store_true",
        help="Do not inject AWBW_ALLOW_CUDA_OPPONENT / lean+fat thread env (use rl/self_play defaults only).",
    )
    ap.add_argument(
        "--no-default-opening-book",
        action="store_true",
        help=(
            "Do not fill missing --opening-book* keys from curriculum "
            "DEFAULT_OPENING_BOOK_TRAIN_ARGS (std_pool_precombat.jsonl, both seats)."
        ),
    )
    ap.add_argument(
        "--training-backend",
        type=str,
        default="sync",
        choices=("sync", "async"),
        help=(
            "Forwarded to train.py: sync = SubprocVecEnv + MaskablePPO (default); "
            "async = IMPALA-style parallel actors + V-trace learner."
        ),
    )
    args = ap.parse_args()
    # Curriculum stage progress needs proposed vs applied reconciliation; orchestrator defaults to
    # --auto-apply so train.py respawns when merged proposed drifts (--no-orchestrator-auto-apply off).
    orchestrator_auto_apply = not bool(args.no_orchestrator_auto_apply)
    _configure_logging(args.log_dir)
    log = logging.getLogger("start_solo_training")
    machine_id = str(args.machine_id).strip()
    if not machine_id:
        print("--machine-id must be non-empty", file=sys.stderr)
        return 2

    fleet_dir = REPO_ROOT / "fleet" / machine_id
    proposed_path = fleet_dir / "proposed_args.json"
    launch_path = fleet_dir / "train_launch_cmd.json"
    pid_path = fleet_dir / "train.pid"
    applied_path = fleet_dir / "applied_args.json"

    log.info(
        "start_solo_training pid=%d machine_id=%s orchestrator_auto_apply=%s "
        "(no_orch_auto_apply=%s) torch_compile=%s "
        "training_backend=%s hybrid_gpu_cpu_opponents=%s (min_envs=%s hybrid_gpu_opp_workers_arg=%s)",
        os.getpid(),
        machine_id,
        orchestrator_auto_apply,
        args.no_orchestrator_auto_apply,
        args.torch_compile,
        args.training_backend,
        not args.no_hybrid_gpu_cpu_opponents,
        args.hybrid_opponent_min_envs,
        args.hybrid_gpu_opponent_workers,
    )
    if args.no_spirit_broken:
        log.info("spirit_broken: disabled (--no-spirit-broken); train env will set AWBW_SPIRIT_BROKEN=0")
    elif os.environ.get("AWBW_SPIRIT_BROKEN") is None:
        log.info(
            "spirit_broken: default AWBW_SPIRIT_BROKEN=1 for train.py (host unset; override with env or --no-spirit-broken)"
        )

    if args.dry_run_bootstrap:
        print(f"[dry-run] would mkdir {fleet_dir}")
        print(
            f"[dry-run] would run: {sys.executable} "
            f"{REPO_ROOT / 'tools' / 'probe_machine_caps.py'} --machine-id {machine_id}"
        )
        print(
            f"[dry-run] would run: {sys.executable} "
            f"{REPO_ROOT / 'tools' / 'propose_train_args.py'} --machine-id {machine_id}"
        )
        if not args.skip_cython_rebuild:
            if sys.platform == "win32":
                dry_cy = f"{sys.executable} {REPO_ROOT / 'scripts' / 'rebuild_cython_extensions.py'}"
            else:
                dry_cy = f"{sys.executable} {REPO_ROOT / 'setup.py'} build_ext --inplace"
            print(f"[dry-run] would recompile Cython: {dry_cy}")
        else:
            print("[dry-run] would skip Cython rebuild (--skip-cython-rebuild)")
        if not proposed_path.is_file():
            print(
                f"[dry-run] (no file yet) proposed read from {proposed_path}; "
                "train argv uses defaults + proposed when present",
                file=sys.stderr,
            )
        extra = shlex.split(args.train_extra_args, posix=os.name != "nt")
        proposed_fake: dict[str, Any] = {}
        if proposed_path.is_file():
            proposed_fake = _read_json(proposed_path)
            proposed_fake = _merge_curriculum_for_initial_launch(
                proposed_fake,
                machine_id=machine_id,
                state_path=fleet_dir / "curriculum_state.json",
                log=log,
                write_state=False,
            )
            proposed_fake = _merge_operator_train_args_override_into_proposed(
                proposed_fake, fleet_dir=fleet_dir, log=log
            )
        _merge_default_opening_book_train_args(
            proposed_fake,
            log=log,
            enabled=not args.no_default_opening_book,
        )
        probe_tune: dict[str, Any] = {}
        pp = fleet_dir / "probe.json"
        if pp.is_file():
            probe_tune = _read_json(pp)
        tune_max = (
            int(args.throughput_tune_max_envs)
            if args.throughput_tune_max_envs is not None
            else _default_throughput_tune_max_envs(machine_id, probe_tune)
        )
        live_extra, live_gids = _prepare_proposed_live_games(
            proposed_fake,
            extra,
            args.live_games_dir,
            no_auto_live=args.no_auto_live_games,
            no_refresh=args.no_refresh_live_snapshots,
            map_pool=args.live_map_pool,
            maps_dir=args.live_maps_dir,
            log=log,
        )
        if args.throughput_tune:
            am0 = dict(proposed_fake.get("args") or {})
            try:
                lo = int(am0.get("--n-envs", 4))
            except (TypeError, ValueError):
                lo = 4
            n_cand = (tune_max - lo + 1) if tune_max >= lo else 1
            print(
                f"[dry-run] throughput-tune: would probe n_envs in [{lo}, {tune_max}] "
                f"({n_cand} candidate(s), ~{args.throughput_tune_per_candidate_s:g}s wall each, "
                "--fps-diag; no subprocess in dry-run)"
            )
        train_argv = _build_train_argv(
            proposed=proposed_fake,
            machine_id=machine_id,
            train_extra=live_extra,
            log_replay_frames=args.log_replay_frames,
            fps_diag=args.fps_diag,
        )
        _ensure_training_backend_argv(train_argv, args.training_backend)
        _ensure_train_argv_n_envs_for_live(train_argv, len(live_gids), log)
        if live_gids:
            ne = _last_int_for_flag(train_argv, "--n-envs")
            if ne is not None and ne < len(live_gids):
                print(
                    f"[dry-run] ERROR: effective --n-envs ({ne}) is less than the number of "
                    f"auto live games ({len(live_gids)}). Raise --n-envs in --train-extra-args "
                    f"(e.g. --n-envs {len(live_gids)}) or use --no-auto-live-games.",
                    file=sys.stderr,
                )
                return 1
        env_overlay = _launch_env(
            machine_id=machine_id,
            log_replay_frames=args.log_replay_frames,
            torch_compile=args.torch_compile,
            fps_diag=args.fps_diag,
            no_spirit_broken=args.no_spirit_broken,
        )
        neff = _last_int_for_flag(train_argv, "--n-envs")
        env_overlay.update(
            _hybrid_gpu_cpu_opponent_env(
                n_envs=neff,
                probe=_read_probe_json(fleet_dir),
                enabled=not args.no_hybrid_gpu_cpu_opponents,
                min_n_envs=int(args.hybrid_opponent_min_envs),
                cuda_opponent_workers=int(args.hybrid_gpu_opponent_workers),
                log=log,
            )
        )
        env = _train_popen_environ(env_overlay)
        print(f"[dry-run] train cmd: {train_argv}")
        print(f"[dry-run] train env (merged, cli-owned keys stripped from host): {env}")
        print(f"[dry-run] would write {launch_path} + {pid_path} + {applied_path}")
        tick_min = max(args.orchestrator_tick_s, 1.0) / 60.0
        orch_bits = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "fleet_orchestrator.py"),
            "--shared-root",
            str(REPO_ROOT),
            "--pools",
            machine_id,
            "--apply",
            "--tick-minutes",
            str(tick_min),
            "--audit-log",
            str(REPO_ROOT / "logs" / "fleet_orchestrator.jsonl"),
        ]
        if orchestrator_auto_apply:
            orch_bits.append("--auto-apply")
        if args.train_bootstrap_grace_s > 0:
            orch_bits.extend(
                [
                    "--train-bootstrap-grace-s",
                    str(args.train_bootstrap_grace_s),
                ]
            )
        if args.orchestrator_curriculum_window_games is not None:
            orch_bits.extend(
                [
                    "--curriculum-window-games",
                    str(args.orchestrator_curriculum_window_games),
                ]
            )
        print(f"[dry-run] orchestrator cmd: {orch_bits}")
        return 0

    zerr = _ensure_no_zombie_cohort(machine_id=machine_id, pid_path=pid_path)
    if zerr is not None:
        print(zerr, file=sys.stderr)
        return 1

    fleet_dir.mkdir(parents=True, exist_ok=True)

    try:
        pr = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "probe_machine_caps.py"),
                "--machine-id",
                machine_id,
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if pr.returncode != 0:
            log.error("probe_machine_caps failed: %s", pr.stderr or pr.stdout)
            return 1
    except OSError as exc:
        log.error("probe_machine_caps could not run: %s", exc)
        return 1

    try:
        pr2 = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "propose_train_args.py"),
                "--machine-id",
                machine_id,
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if pr2.returncode != 0:
            log.error("propose_train_args failed: %s", pr2.stderr or pr2.stdout)
            return 1
    except OSError as exc:
        log.error("propose_train_args could not run: %s", exc)
        return 1

    if not proposed_path.is_file():
        log.error("missing proposed_args after propose: %s", proposed_path)
        return 1

    if not args.skip_cython_rebuild:
        cy_blockers = _find_cython_rebuild_blockers(REPO_ROOT)
        if cy_blockers:
            lines = [
                "Refusing Cython rebuild: a Python process still has *.pyd loaded (Windows WinError 32). "
                "Often a stray train.py **or** a SubprocVecEnv multiprocessing.spawn worker "
                "(no 'train.py' in argv). Stop those PIDs, or run "
                "`python scripts/diagnose_cython_lock.py`, or use --skip-cython-rebuild if .pyx unchanged:"
            ]
            for pid, cmdline in cy_blockers:
                lines.append(f"  pid={pid} cmdline={' '.join(cmdline)!r}")
            msg = "\n".join(lines)
            log.error("%s", msg)
            print(msg, file=sys.stderr)
            return 1
        rc = _rebuild_cython_extensions(log)
        if rc != 0:
            log.error(
                "Cython rebuild failed; if sources are unchanged retry with --skip-cython-rebuild"
            )
            return rc

    proposed = _read_json(proposed_path)
    proposed = _merge_curriculum_for_initial_launch(
        proposed,
        machine_id=machine_id,
        state_path=fleet_dir / "curriculum_state.json",
        log=log,
        write_state=True,
    )
    proposed = _merge_operator_train_args_override_into_proposed(
        proposed, fleet_dir=fleet_dir, log=log
    )
    _merge_default_opening_book_train_args(
        proposed,
        log=log,
        enabled=not args.no_default_opening_book,
    )
    extra = shlex.split(args.train_extra_args, posix=os.name != "nt")
    # Merge train_extra into proposed args so build_train_argv preserves them
    extra_map = _train_extra_to_args_map(extra)
    for k, v in extra_map.items():
        proposed.setdefault("args", {})[k] = v
    probe_tune: dict[str, Any] = {}
    pp = fleet_dir / "probe.json"
    if pp.is_file():
        probe_tune = _read_json(pp)
    tune_max = (
        int(args.throughput_tune_max_envs)
        if args.throughput_tune_max_envs is not None
        else _default_throughput_tune_max_envs(machine_id, probe_tune)
    )
    live_extra, live_gids = _prepare_proposed_live_games(
        proposed,
        extra,
        args.live_games_dir,
        no_auto_live=args.no_auto_live_games,
        no_refresh=args.no_refresh_live_snapshots,
        map_pool=args.live_map_pool,
        maps_dir=args.live_maps_dir,
        log=log,
    )
    if args.throughput_tune:
        _tune_n_envs_throughput_inplace(
            proposed,
            machine_id=machine_id,
            train_extra=live_extra,
            live_gids=live_gids,
            max_envs=tune_max,
            per_candidate_s=args.throughput_tune_per_candidate_s,
            min_iters=args.throughput_tune_min_iters,
            max_host_ram_pct=args.throughput_tune_max_host_ram_pct,
            max_host_cpu_pct=args.throughput_tune_max_host_cpu_pct,
            host_wait_s=args.throughput_tune_host_wait_s,
            log_replay_frames=args.log_replay_frames,
            log=log,
        )
    train_argv = _build_train_argv(
        proposed=proposed,
        machine_id=machine_id,
        train_extra=live_extra,
        log_replay_frames=args.log_replay_frames,
        fps_diag=args.fps_diag,
    )
    _ensure_training_backend_argv(train_argv, args.training_backend)
    _ensure_train_argv_n_envs_for_live(train_argv, len(live_gids), log)
    if live_gids:
        ne = _last_int_for_flag(train_argv, "--n-envs")
        if ne is not None and ne < len(live_gids):
            log.error(
                "effective --n-envs (%s) must be >= number of auto live games (%s) "
                "under %s. Increase --n-envs in --train-extra-args or proposed_args, "
                "or use --no-auto-live-games.",
                ne,
                len(live_gids),
                args.live_games_dir,
            )
            return 1
    proposed_synced = _proposed_args_synced_from_train_argv(proposed, train_argv)
    _atomic_write_json(proposed_path, proposed_synced)

    env_overlay = _launch_env(
        machine_id=machine_id,
        log_replay_frames=args.log_replay_frames,
        torch_compile=args.torch_compile,
        fps_diag=args.fps_diag,
        no_spirit_broken=args.no_spirit_broken,
    )
    env_overlay.update(
        _hybrid_gpu_cpu_opponent_env(
            n_envs=_last_int_for_flag(train_argv, "--n-envs"),
            probe=_read_probe_json(fleet_dir),
            enabled=not args.no_hybrid_gpu_cpu_opponents,
            min_n_envs=int(args.hybrid_opponent_min_envs),
            cuda_opponent_workers=int(args.hybrid_gpu_opponent_workers),
            log=log,
        )
    )
    launch_doc = {
        "cmd": train_argv,
        "env": env_overlay,
        "cwd": str(REPO_ROOT.resolve()),
    }
    _atomic_write_json(launch_path, launch_doc)

    train_subproc_log_fh: Any = None
    train_subproc_log_path = (REPO_ROOT / "logs" / f"solo_train_train_py_{machine_id}.log").resolve()
    try:
        train_subproc_log_path.parent.mkdir(parents=True, exist_ok=True)
        train_subproc_log_fh = open(
            train_subproc_log_path, "a", encoding="utf-8", buffering=1
        )
        train_subproc_log_fh.write(
            f"\n--- train.py child log session {datetime.now(timezone.utc).isoformat()} "
            f"parent_pid={os.getpid()} machine_id={machine_id!r} ---\n"
        )
    except OSError as exc:
        log.warning(
            "could not open train stdout/stderr log %s (%s); train inherits this process stdio",
            train_subproc_log_path,
            exc,
        )
        train_subproc_log_fh = None
    else:
        log.info("train.py child stdout+stderr are also written to %s", train_subproc_log_path)

    popen_kw: dict[str, Any] = {
        "cwd": str(REPO_ROOT),
        "env": _train_popen_environ(env_overlay),
    }
    if sys.platform == "win32":
        popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    if train_subproc_log_fh is not None:
        popen_kw["stdout"] = train_subproc_log_fh
        popen_kw["stderr"] = subprocess.STDOUT

    try:
        train_proc = subprocess.Popen(train_argv, **popen_kw)  # noqa: S603
    except OSError:
        if train_subproc_log_fh is not None:
            try:
                train_subproc_log_fh.close()
            except OSError:
                pass
        raise
    # After orchestrator --auto-apply restarts, train pid may be replaced: track Popen or adopted pid.
    train_handle: subprocess.Popen | int = train_proc
    _atomic_write_text(pid_path, str(train_proc.pid) + "\n")

    prop_h = _proposed_args_content_sha256(proposed_synced)
    if prop_h is not None:
        applied_doc = {
            **proposed_synced,
            "applied_at": time.time(),
            "args_content_sha256": prop_h,
        }
        _atomic_write_json(applied_path, applied_doc)

    tick_min = max(args.orchestrator_tick_s, 1.0) / 60.0
    orch_argv: list[str] = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "fleet_orchestrator.py"),
        "--shared-root",
        str(REPO_ROOT.resolve()),
        "--pools",
        machine_id,
        "--apply",
        "--tick-minutes",
        str(tick_min),
        "--audit-log",
        str((REPO_ROOT / "logs" / "fleet_orchestrator.jsonl").resolve()),
    ]
    if orchestrator_auto_apply:
        orch_argv.append("--auto-apply")
    if args.train_bootstrap_grace_s > 0:
        orch_argv.extend(
            ["--train-bootstrap-grace-s", str(args.train_bootstrap_grace_s)]
        )
    if args.orchestrator_curriculum_window_games is not None:
        orch_argv.extend(
            [
                "--curriculum-window-games",
                str(args.orchestrator_curriculum_window_games),
            ]
        )

    orch_subproc_log_fh: Any = None
    orch_subproc_log_path = (
        REPO_ROOT / "logs" / f"solo_train_fleet_orchestrator_{machine_id}.log"
    ).resolve()
    try:
        orch_subproc_log_path.parent.mkdir(parents=True, exist_ok=True)
        orch_subproc_log_fh = open(
            orch_subproc_log_path, "a", encoding="utf-8", buffering=1
        )
        orch_subproc_log_fh.write(
            f"\n--- fleet_orchestrator child log session {datetime.now(timezone.utc).isoformat()} "
            f"parent_pid={os.getpid()} machine_id={machine_id!r} ---\n"
        )
    except OSError as exc:
        log.warning(
            "could not open fleet_orchestrator stdout/stderr log %s (%s); "
            "orchestrator inherits this process stdio",
            orch_subproc_log_path,
            exc,
        )
        orch_subproc_log_fh = None
    else:
        log.info(
            "fleet_orchestrator child stdout+stderr are also written to %s",
            orch_subproc_log_path,
        )

    orch_popen_kw: dict[str, Any] = {
        "cwd": str(REPO_ROOT),
        "env": os.environ.copy(),
    }
    if orch_subproc_log_fh is not None:
        orch_popen_kw["stdout"] = orch_subproc_log_fh
        orch_popen_kw["stderr"] = subprocess.STDOUT

    try:
        orch_proc = subprocess.Popen(orch_argv, **orch_popen_kw)  # noqa: S603
    except OSError:
        if train_subproc_log_fh is not None:
            try:
                train_subproc_log_fh.close()
            except OSError:
                pass
        if orch_subproc_log_fh is not None:
            try:
                orch_subproc_log_fh.close()
            except OSError:
                pass
        raise

    orch_audit_path = (REPO_ROOT / "logs" / "fleet_orchestrator.jsonl").resolve()

    shutdown = False

    def _on_signal(_signum: int, _frame: Any) -> None:
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_signal)

    def _append_solo_bootstrap_watch_line(message: str) -> None:
        """Durable 30s heartbeat so a closed PowerShell window does not lose the last state."""
        wpath = args.log_dir / f"solo_bootstrap_watch_{machine_id}.log"
        try:
            wpath.parent.mkdir(parents=True, exist_ok=True)
            with wpath.open("a", encoding="utf-8", buffering=1) as wfh:
                wfh.write(
                    f"{datetime.now(timezone.utc).isoformat()}Z pid={os.getpid()} {message}\n"
                )
        except OSError:
            pass

    _next_bootstrap_heartbeat = time.monotonic() + 30.0

    exit_code = 0
    try:
        while not shutdown:
            # Distinguish adopted int pid from subprocess.Popen: tests may mock Popen.
            if isinstance(train_handle, int):
                tr = _poll_adopted_train_pid(train_handle)
            else:
                tr = train_handle.poll()
            oc = orch_proc.poll()
            nowm = time.monotonic()
            if nowm >= _next_bootstrap_heartbeat:
                _next_bootstrap_heartbeat = nowm + 30.0
                ta = "up" if tr is None else f"exited {tr}"
                oa = "up" if oc is None else f"exited {oc}"
                th = str(train_handle) if isinstance(train_handle, int) else f"popen_pid={train_handle.pid}"
                _append_solo_bootstrap_watch_line(f"train {th} {ta} | orchestrator {oa}")
            if tr is not None:
                new_pid = _read_train_pid(pid_path)
                old: int
                if isinstance(train_handle, int):
                    old = int(train_handle)
                else:
                    old = int(train_handle.pid)
                adopted: int | None = None
                if (
                    new_pid is not None
                    and new_pid != old
                    and _pid_is_fleet_train_for_machine(new_pid, machine_id)
                ):
                    adopted = int(new_pid)
                if adopted is None and orchestrator_auto_apply and tr in (15, -15):
                    if new_pid == old or new_pid is None:
                        log.info(
                            "train exited with code %s; waiting briefly for "
                            "fleet_orchestrator --auto-apply to rewrite %s (restart race)",
                            tr,
                            pid_path,
                        )
                    adopted_int = _wait_for_orchestrator_train_replacement(
                        pid_path, old, machine_id, log=log
                    )
                    if adopted_int is not None:
                        adopted = int(adopted_int)
                if adopted is not None:
                    log.info(
                        "orchestrator replaced train (expected when --auto-apply restarts): "
                        "exited_pid=%s exit_code=%s adopted_file_pid=%s; continuing bootstrap",
                        old,
                        tr,
                        adopted,
                    )
                    if tr == 15 or tr == -15:
                        log.info(
                            "exit_code 15 is often a SIGTERM-style stop on Windows (orchestrator "
                            "terminated this train for args drift (apply_cooldown_s=600) or "
                            "zombie heal — not necessarily a Python crash). See "
                            "logs/fleet_orchestrator.jsonl for restart_train / arg_diff_keys."
                        )
                    train_handle = adopted
                else:
                    log.error(
                        "train.py exited before shutdown (code %s) monitored_pid=%s file_pid=%s",
                        tr,
                        old,
                        new_pid,
                    )
                    log.error(
                        "for orchestrator context see %s (tail below)",
                        orch_audit_path,
                    )
                    log.error("fleet_orchestrator.jsonl tail:\n%s", _tail_text_file(orch_audit_path))
                    log.error(
                        "captured train.py stdio (bootstrap-spawned process only) %s (tail below)",
                        train_subproc_log_path,
                    )
                    log.error("solo_train train_py log tail:\n%s", _tail_text_file(train_subproc_log_path))
                    exit_code = 1
                    break
            if oc is not None:
                log.error("fleet_orchestrator exited before shutdown (code %s)", oc)
                log.error("fleet_orchestrator.jsonl tail:\n%s", _tail_text_file(orch_audit_path))
                log.error(
                    "solo_train fleet_orchestrator captured log tail (%s):\n%s",
                    orch_subproc_log_path,
                    _tail_text_file(orch_subproc_log_path),
                )
                exit_code = 1
                break
            time.sleep(1.0)
    finally:
        log.info("shutting down children (orchestrator then train)")
        _terminate_process(orch_proc, timeout_s=60.0, windows_ctrl_break=False)
        if isinstance(train_handle, int):
            _terminate_process_tree_by_pid(int(train_handle), timeout_s=60.0)
        else:
            _terminate_process(train_handle, timeout_s=60.0, windows_ctrl_break=True)
        if train_subproc_log_fh is not None:
            try:
                train_subproc_log_fh.close()
            except OSError:
                pass
        if orch_subproc_log_fh is not None:
            try:
                orch_subproc_log_fh.close()
            except OSError:
                pass
        if pid_path.is_file():
            try:
                pid_path.unlink()
            except OSError:
                pass

    return exit_code


if __name__ == "__main__":
    import traceback as _traceback

    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\n[start_solo_training] interrupted", file=sys.stderr, flush=True)
        raise SystemExit(130) from None
    except BaseException:  # noqa: BLE001 — bootstrap last-resort
        _tb = _traceback.format_exc()
        _crash = REPO_ROOT / "logs" / "start_solo_training_bootstrap_crash.log"
        try:
            _crash.parent.mkdir(parents=True, exist_ok=True)
            _crash.write_text(_tb, encoding="utf-8")
        except OSError:
            pass
        print(_tb, file=sys.stderr, flush=True)
        raise SystemExit(1) from None
