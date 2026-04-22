"""
Fleet / multi-machine training layout: main vs auxiliary roles, shared mount paths.

Environment (document in README):
  AWBW_MACHINE_ROLE   — ``main`` (default) or ``auxiliary``
  AWBW_MACHINE_ID     — short stable name for aux boxes (e.g. ``eval1``, ``pool-east``)
  AWBW_SHARED_ROOT    — on auxiliary: path to main repo mount (default ``Z:\\``); on main: unset or same as repo

Optional:
  AWBW_CHECKPOINT_DIR — override checkpoint directory (pool trainers); must stay under repo or shared tree
"""
from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Any top-level `checkpoint_*.zip` in a checkpoint_dir is a managed snapshot
# (historical PPO zips, opponent pool, prune targets).  Not used for
# `latest.zip`, `promoted/*.zip`, or `bc/*.zip`.
def _is_managed_checkpoint_zip_name(name: str) -> bool:
    return name.startswith("checkpoint_") and name.lower().endswith(".zip")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Verdict JSON written by fleet eval daemon / consumed by promote.py
VERDICT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FleetConfig:
    """Resolved fleet identity and paths for this process."""

    role: str  # "main" | "auxiliary"
    machine_id: str | None
    shared_root: Path | None  # auxiliary: mount root; main: optional mirror of repo
    repo_root: Path

    @property
    def is_main(self) -> bool:
        return self.role == "main"

    @property
    def is_auxiliary(self) -> bool:
        return self.role == "auxiliary"


def _norm(p: Path) -> Path:
    try:
        return p.resolve()
    except OSError:
        return p


def load_machine_role(cli_override: str | None) -> str:
    raw = (cli_override or os.environ.get("AWBW_MACHINE_ROLE") or "main").strip().lower()
    if raw not in ("main", "auxiliary"):
        raise SystemExit(
            f"Invalid AWBW_MACHINE_ROLE / --machine-role={raw!r}; expected 'main' or 'auxiliary'."
        )
    return raw


def load_machine_id() -> str | None:
    mid = os.environ.get("AWBW_MACHINE_ID")
    if mid is None or str(mid).strip() == "":
        return None
    return str(mid).strip()


def load_shared_root_for_role(role: str, cli_override: str | None) -> Path | None:
    if role == "auxiliary":
        s = cli_override or os.environ.get("AWBW_SHARED_ROOT") or ("Z:" + os.sep)
        return Path(s)
    # main
    if cli_override:
        return Path(cli_override)
    env = os.environ.get("AWBW_SHARED_ROOT")
    if env is None or str(env).strip() == "":
        return None
    return Path(env)


def validate_fleet_at_startup(cfg: FleetConfig) -> None:
    """Hard checks before training or aux daemons touch shared disks."""
    repo = _norm(cfg.repo_root)
    if cfg.role == "main":
        if cfg.shared_root is not None:
            sr = _norm(cfg.shared_root)
            if sr != repo:
                raise SystemExit(
                    f"[fleet] main: AWBW_SHARED_ROOT / --shared-root points at {sr} but repo is {repo}. "
                    "Unset shared root, or set it equal to the repo path (same machine sanity check)."
                )
        return

    # auxiliary
    if not cfg.shared_root:
        raise SystemExit("[fleet] auxiliary: AWBW_SHARED_ROOT (or default Z:\\) is required.")
    sr = cfg.shared_root
    if not sr.exists():
        raise SystemExit(f"[fleet] auxiliary: shared root does not exist: {sr}")
    ck = sr / "checkpoints"
    data = sr / "data"
    if not ck.is_dir():
        raise SystemExit(f"[fleet] auxiliary: expected checkpoints/ under shared root: {ck}")
    if not data.is_dir():
        raise SystemExit(f"[fleet] auxiliary: expected data/ under shared root: {data}")


def default_checkpoint_dir(repo_root: Path) -> Path:
    return repo_root / "checkpoints"


def resolve_checkpoint_dir(
    repo_root: Path,
    cli_dir: Path | None,
    env_override: str | None,
) -> Path:
    if cli_dir is not None:
        return _norm(cli_dir)
    e = env_override or os.environ.get("AWBW_CHECKPOINT_DIR")
    if e:
        return _norm(Path(e))
    return _norm(default_checkpoint_dir(repo_root))


def checkpoint_dir_is_aux_pool_tree(checkpoint_dir: Path, cfg: FleetConfig) -> bool:
    """True when auxiliary writes under shared ``checkpoints/pool/`` (divergent pool export)."""
    if cfg.role != "auxiliary" or not cfg.shared_root:
        return False
    sr = _norm(cfg.shared_root)
    ck = _norm(checkpoint_dir)
    pool_root = sr / "checkpoints" / "pool"
    try:
        ck.relative_to(pool_root)
        return True
    except ValueError:
        return False


def validate_aux_pool_checkpoint_dir(
    cfg: FleetConfig,
    checkpoint_dir: Path,
) -> None:
    """Pool aux must write only under shared checkpoints/pool/<MACHINE_ID>/."""
    if not checkpoint_dir_is_aux_pool_tree(checkpoint_dir, cfg):
        return
    mid = cfg.machine_id
    if not mid:
        raise SystemExit(
            "[fleet] auxiliary pool training requires AWBW_MACHINE_ID (e.g. pool-east) "
            "so snapshots stay under checkpoints/pool/<ID>/."
        )
    sr = _norm(cfg.shared_root)  # type: ignore[union-attr]
    ck = _norm(checkpoint_dir)
    allowed = sr / "checkpoints" / "pool" / mid
    try:
        ck.relative_to(allowed)
    except ValueError:
        raise SystemExit(
            f"[fleet] auxiliary: --checkpoint-dir must resolve under {allowed}, got {ck}"
        )


def fleet_subdirs(checkpoint_dir: Path) -> dict[str, Path]:
    """Well-known layout under checkpoints/ (main's tree; aux sees it via Z:\\)."""
    ck = checkpoint_dir
    return {
        "promoted": ck / "promoted",
        "bc": ck / "bc",
        "pool": ck / "pool",
    }


def bootstrap_fleet_layout(
    repo_or_shared_root: Path,
    *,
    machine_id: str | None,
    role: str,
) -> None:
    """
    Idempotent mkdir for fleet contract dirs. Safe on main (creates empty dirs ignored by training globs).
    """
    ck = repo_or_shared_root / "checkpoints"
    subs = fleet_subdirs(ck)
    subs["promoted"].mkdir(parents=True, exist_ok=True)
    subs["bc"].mkdir(parents=True, exist_ok=True)
    subs["pool"].mkdir(parents=True, exist_ok=True)
    fleet_root = repo_or_shared_root / "fleet"
    fleet_root.mkdir(parents=True, exist_ok=True)
    if role == "auxiliary" and machine_id:
        (fleet_root / machine_id / "eval").mkdir(parents=True, exist_ok=True)


def write_status_json(
    path: Path,
    *,
    role: str,
    machine_id: str | None,
    task: str,
    current_target: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "role": role,
        "machine_id": machine_id,
        "task": task,
        "last_poll": time.time(),
        "current_target": current_target,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def allowed_aux_write_prefixes(shared_root: Path, machine_id: str) -> tuple[Path, ...]:
    """Prefixes aux processes may write under (cheap guard)."""
    ck = shared_root / "checkpoints"
    return (
        ck / "promoted",
        ck / "bc",
        ck / "pool" / machine_id,
        shared_root / "fleet" / machine_id,
        shared_root / "replays",
    )


def assert_aux_write_path(path: Path, shared_root: Path, machine_id: str) -> Path:
    """Refuse aux writes outside designated subtrees."""
    p = _norm(path)
    sr = _norm(shared_root)
    for pref in allowed_aux_write_prefixes(sr, machine_id):
        try:
            p.relative_to(_norm(pref))
            return p
        except ValueError:
            continue
    raise SystemExit(
        f"[fleet] auxiliary write refused (outside allowed prefixes for {machine_id!r}): {p}"
    )


def verdict_summary_from_symmetric_json(data: dict[str, Any]) -> dict[str, Any]:
    """Map symmetric_checkpoint_eval --json-out payload to fleet verdict fields."""
    cw = int(data.get("candidate_wins", 0))
    bw = int(data.get("baseline_wins", 0))
    total = cw + bw
    wr = float(cw / total) if total else 0.0
    # Wilson-ish simple bound: skip heavy stats; store counts for promote thresholds
    return {
        "schema_version": VERDICT_SCHEMA_VERSION,
        "candidate_wins": cw,
        "baseline_wins": bw,
        "games_decided": total,
        "winrate": wr,
        "map_id": data.get("map_id"),
        "tier": data.get("tier"),
        "co_p0": data.get("co_p0"),
        "co_p1": data.get("co_p1"),
        "symmetric_summary": {
            "per_seat": data.get("per_seat"),
            "promotion_heuristic_ok": data.get("promotion_heuristic_ok"),
        },
    }


def iter_pool_checkpoint_zips(checkpoint_dir: Path) -> list[str]:
    """Globs checkpoints/pool/*/checkpoint_*.zip for opponent mixing."""
    pattern = str(checkpoint_dir / "pool" / "*" / "checkpoint_*.zip")
    return sorted(glob.glob(pattern))


def iter_fleet_opponent_checkpoint_zips(fleet_checkpoint_root: Path) -> list[str]:
    """
    All ``checkpoint_*.zip`` paths under a fleet **root** ``checkpoints/`` dir:
    top-level snapshots plus ``pool/*/checkpoint_*.zip`` from auxiliary exports.
    """
    r = str(_norm(fleet_checkpoint_root))
    top = sorted(glob.glob(os.path.join(r, "checkpoint_*.zip")))
    pool = iter_pool_checkpoint_zips(Path(r))
    return sorted(set(top + pool))


def resolve_fleet_opponent_pool_root(checkpoint_dir: Path, cfg: FleetConfig) -> Path:
    """
    Root directory used to merge fleet-wide opponent checkpoints when
    ``pool_from_fleet`` is enabled.

    Auxiliary pool trainers write under ``<shared>/checkpoints/pool/<ID>/`` but
    should still draw opponents from the shared ``<shared>/checkpoints/`` tree
    (main line + every pool machine).
    """
    ck = _norm(checkpoint_dir)
    if cfg.is_auxiliary and checkpoint_dir_is_aux_pool_tree(checkpoint_dir, cfg):
        if cfg.shared_root is None:
            return ck
        return _norm(cfg.shared_root) / "checkpoints"
    return ck


def sorted_checkpoint_zip_paths(checkpoint_dir: Path) -> list[Path]:
    """
    ``checkpoint_*.zip`` under ``checkpoint_dir`` only, oldest first.

    Order is (modification time, path name) so merged directories from
    different hosts stay chronological; legacy ``checkpoint_0000.zip``-style
    names remain supported.  Timestamp-style names
    (``checkpoint_YYYYMMDDTHHMMSS_*Z``) sort lexicographically by time for
    equal mtimes.
    """
    ck = _norm(checkpoint_dir)
    if not ck.is_dir():
        return []
    paths: list[Path] = []
    for p in ck.iterdir():
        if not p.is_file():
            continue
        if not _is_managed_checkpoint_zip_name(p.name):
            continue
        paths.append(p)
    return sorted(paths, key=lambda p: (p.stat().st_mtime, p.name))


def new_checkpoint_stem_utc() -> str:
    """
    Return a new snapshot stem (no ``.zip``) that is unique across processes
    and valid on both Windows and POSIX (no ``:`` or other reserved chars).

    Format: ``checkpoint_<UTC YYYYMMDDTHHMMSS>_<N>Z`` where *N* is a 20-digit,
    zero-padded count of nanoseconds since the Unix epoch (from
    :func:`time.time_ns`, bumped if needed so successive saves in this process
    never reuse a stem on hosts where the clock has sub-microsecond
    resolution).
    """
    n = time.time_ns()
    last: int = getattr(new_checkpoint_stem_utc, "_last_n", 0)
    if n <= last:
        n = last + 1
    setattr(new_checkpoint_stem_utc, "_last_n", n)
    sec = n // 1_000_000_000
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(sec))
    return f"checkpoint_{stamp}_{n:020d}Z"


def prune_checkpoint_zip_snapshots(checkpoint_dir: Path, max_keep: int) -> int:
    """
    Keep at most ``max_keep`` ``checkpoint_*.zip`` files under the directory
    (oldest by modification time removed first; tie-break by name). ``max_keep
    <= 0`` disables pruning.

    Returns the number of files removed.
    """
    if max_keep <= 0:
        return 0
    paths = sorted_checkpoint_zip_paths(checkpoint_dir)
    if len(paths) <= max_keep:
        return 0
    to_remove = paths[: len(paths) - max_keep]
    n = 0
    for p in to_remove:
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n
