"""Fleet log layout helper.

Convention (formalized 2026-04-22 as part of the Phase 10/11 logging prereqs):

- ``logs/<file>.jsonl`` is *this machine's* log (the writer always lands here;
  see the writer-root assert in :mod:`rl.paths`). On the auxiliary box this is
  ``pc-b``.
- ``logs/logs/<file>.jsonl`` is the **Main mirror**. The operator manually
  copies these files out of ``D:/awbw/logs/`` on Main during/after Main's
  offline window. The doubled ``logs/logs/`` is intentional: it preserves the
  upstream relative path so a recursive copy of Main's ``logs/`` tree drops in
  cleanly without rewriting filenames.
- ``logs/<machine_id>/<file>.jsonl`` is the **reserved namespace** for any
  future auxiliary machine (e.g. ``keras-aux``, ``fake-aux-1``). The directory
  does not need to exist yet; :func:`fleet_log_path` is purely a path mapper.

The orchestrator (Phase 10) reads from each machine's path; the writer on each
machine only ever writes to its own local ``logs/<file>.jsonl``.

This module has no external dependencies beyond :mod:`pathlib` / :mod:`re`.
"""
from __future__ import annotations

import re
from pathlib import Path

from rl.paths import LOGS_DIR, REPO_ROOT

THIS_MACHINE_LOGS_DIR: Path = LOGS_DIR
"""Local writer root. Re-exported from :mod:`rl.paths` for callers that want
the fleet-aware view without importing both modules."""

MAIN_MIRROR_DIR_NAME: str = "logs"
"""Sub-directory name under ``logs/`` that holds the operator-managed Main
mirror (so the full path is ``logs/logs/``)."""

FLEET_LOG_FILES: tuple[str, ...] = ("game_log.jsonl", "slow_games.jsonl")
"""Canonical per-machine log filenames the orchestrator reads. Extend as new
fleet-wide log streams are introduced."""

_MACHINE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_machine_id(machine_id: str) -> None:
    """Raise ``ValueError`` if ``machine_id`` is not a safe single path segment.

    Guards against path traversal (``../etc``), nested paths (``a/b``), and
    shell-significant characters (``a;b``). Empty / ``None`` is handled by the
    caller (it means "this machine") and never reaches this function.
    """
    if not _MACHINE_ID_RE.fullmatch(machine_id):
        raise ValueError(
            f"[fleet_logs] invalid machine_id {machine_id!r}: must match "
            f"{_MACHINE_ID_RE.pattern} (single path segment, no separators)"
        )


def fleet_log_path(machine_id: str | None, filename: str) -> Path:
    """Return the absolute path to ``filename`` for the named machine.

    - ``machine_id`` is ``None`` or empty -> local writer path
      (``REPO_ROOT/logs/<filename>``).
    - ``machine_id`` is ``"main"`` (case-insensitive) -> operator-managed Main
      mirror (``REPO_ROOT/logs/logs/<filename>``).
    - Any other id (validated against ``^[A-Za-z0-9_-]+$``) -> reserved aux
      slot (``REPO_ROOT/logs/<machine_id>/<filename>``). The directory is not
      required to exist; this function is a pure mapper.

    Raises ``ValueError`` for ids that would escape the ``logs/`` tree.
    """
    if machine_id is None or machine_id == "":
        return LOGS_DIR / filename
    if machine_id.lower() == "main":
        return LOGS_DIR / MAIN_MIRROR_DIR_NAME / filename
    _validate_machine_id(machine_id)
    return LOGS_DIR / machine_id / filename


def iter_fleet_log_paths(filename: str) -> dict[str, Path]:
    """Discover every machine that currently has ``filename`` on disk.

    Returns a mapping ``{machine_id: path}`` where:

    - ``"_local"`` is always present and points at this machine's
      ``logs/<filename>``, regardless of whether the file itself exists yet
      (so callers can tail a not-yet-created log).
    - ``"main"`` is included iff ``logs/logs/<filename>`` exists.
    - Each subdirectory ``logs/<id>/`` (other than ``logs``) is included iff
      ``logs/<id>/<filename>`` exists.

    Subdirectories without the requested file are skipped. Non-directory
    entries under ``logs/`` are ignored.
    """
    out: dict[str, Path] = {"_local": LOGS_DIR / filename}

    main_path = LOGS_DIR / MAIN_MIRROR_DIR_NAME / filename
    if main_path.is_file():
        out["main"] = main_path

    if LOGS_DIR.is_dir():
        for child in LOGS_DIR.iterdir():
            if not child.is_dir():
                continue
            if child.name == MAIN_MIRROR_DIR_NAME:
                continue
            if not _MACHINE_ID_RE.fullmatch(child.name):
                continue
            candidate = child / filename
            if candidate.is_file():
                out[child.name] = candidate

    return out


def infer_machine_id_from_path(path: Path) -> str | None:
    """Inverse of :func:`fleet_log_path` for orchestrator readers.

    - Returns ``None`` if ``path`` lives directly under ``logs/`` (local
      machine, matching the ``machine_id=None`` writer case).
    - Returns ``"main"`` if ``path`` lives under ``logs/logs/``.
    - Returns the directory name (e.g. ``"keras-aux"``) for any other
      ``logs/<machine_id>/<file>`` layout.
    - Returns ``None`` if ``path`` is not under ``LOGS_DIR`` at all (caller
      passed something unrelated).
    """
    try:
        rel = path.resolve().relative_to(LOGS_DIR.resolve())
    except ValueError:
        return None

    parts = rel.parts
    if len(parts) <= 1:
        return None
    if parts[0] == MAIN_MIRROR_DIR_NAME:
        return "main"
    return parts[0]


__all__ = [
    "FLEET_LOG_FILES",
    "MAIN_MIRROR_DIR_NAME",
    "THIS_MACHINE_LOGS_DIR",
    "fleet_log_path",
    "iter_fleet_log_paths",
    "infer_machine_id_from_path",
]
