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
import re
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Optional

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


def _discover_fleet_eval_verdict_paths(verdicts_root: Path) -> list[Path]:
    """All ``<verdicts_root>/*/eval/*.json`` (normalized)."""
    root = _norm(verdicts_root)
    if not root.is_dir():
        return []
    return sorted(root.glob("*/eval/*.json"))


def _verdict_winrate_by_stem(verdict_paths: list[Path]) -> dict[str, float]:
    """
    Map checkpoint **stem** -> winrate, preferring the newest on-disk verdict
    for each stem (tie-break: larger ``timestamp`` in JSON if present).

    The eval daemon names verdict files ``<candidate_zip_stem>.json`` and also
    sets ``"ckpt": "<file>.zip"`` in the payload; both identify the same stem.
    """
    best: dict[str, tuple[tuple[float, float], float]] = {}
    for vp in verdict_paths:
        try:
            raw = json.loads(vp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"[curator] skipping malformed verdict {vp}: {exc}",
                file=sys.stderr,
            )
            continue
        if not isinstance(raw, dict):
            continue
        stem = vp.stem
        ck = raw.get("ckpt")
        if isinstance(ck, str) and ck:
            try:
                stem = Path(ck).stem
            except (TypeError, ValueError):
                pass
        if "winrate" in raw:
            try:
                wr = float(raw["winrate"])
            except (TypeError, ValueError):
                wr = float(verdict_summary_from_symmetric_json(raw)["winrate"])
        else:
            wr = float(verdict_summary_from_symmetric_json(raw)["winrate"])
        try:
            vm = float(vp.stat().st_mtime)
        except OSError:
            vm = 0.0
        try:
            ts = float(raw.get("timestamp", 0.0))
        except (TypeError, ValueError):
            ts = 0.0
        ksort = (vm, ts)
        prev = best.get(stem)
        if prev is None or ksort > prev[0]:
            best[stem] = (ksort, wr)
    return {s: t[1] for s, t in best.items()}


def _training_step_from_stem(stem: str) -> int | None:
    """
    If the stem is legacy ``checkpoint_<small_int>``-style, return the int.
    The default UTC+nanos stems (20-digit count before ``Z``) are **not** a
    training step — return None so callers fall back to mtime deciles.
    """
    m = re.search(r"_([0-9]+)Z$", stem)
    if m and len(m.group(1)) <= 6:
        try:
            return int(m.group(1), 10)
        except ValueError:
            return None
    m2 = re.search(r"checkpoint_0*([0-9]+)$", stem)
    if m2:
        try:
            return int(m2.group(1), 10)
        except ValueError:
            return None
    return None


def prune_checkpoint_zip_curated(
    checkpoint_dir: Path,
    *,
    k_newest: int = 8,
    m_top_winrate: int = 12,
    d_diversity: int = 4,
    verdicts_root: Optional[Path] = None,
    min_age_minutes: float = 5.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Phase 10b: quality-curated pool pruning. Replaces FIFO-by-mtime.

    Keeps the UNION of three sets:
      * K newest checkpoint_*.zip by mtime (recency).
      * M top by verdict winrate (quality, sourced from
        ``<verdicts_root>/<MACHINE_ID>/eval/*.json`` written by
        scripts/fleet_eval_daemon.py).
      * D diversity slots: bucket the remaining survivors by training-step
        decile (or by mtime decile if step is not encoded in the stem),
        keep one per bucket so distinct old policies don't all evict at
        once.

    Files younger than ``min_age_minutes`` are NEVER candidates for
    deletion (protects freshly-published zips before any verdict has had
    a chance to land; also makes the 10a async-publish lag safe).

    When ``verdicts_root`` is None or no verdicts exist, behavior falls
    back to ``prune_checkpoint_zip_snapshots(checkpoint_dir,
    max_keep=k_newest + m_top_winrate + d_diversity)`` — i.e. mtime-FIFO
    at the same total cap. Cold-start safe.

    Returns a dict with diagnostic shape:
      {
        "kept_total":       int,
        "kept_by_recency":  list[str],   # zip stems
        "kept_by_winrate":  list[str],
        "kept_by_diversity":list[str],
        "removed":          list[str],
        "fallback_used":    bool,        # True if FIFO path taken
        "reason":           str,         # human-readable summary
      }

    With ``dry_run=True`` no files are deleted; the dict still reports
    what would have been removed. Used by the orchestrator (Phase 10e).
    """
    total_cap = k_newest + m_top_winrate + d_diversity
    paths = sorted_checkpoint_zip_paths(checkpoint_dir)
    if not paths:
        reason = "empty checkpoint_dir; nothing to prune"
        return {
            "kept_total": 0,
            "kept_by_recency": [],
            "kept_by_winrate": [],
            "kept_by_diversity": [],
            "removed": [],
            "fallback_used": True,
            "reason": reason,
        }
    vroot = _norm(verdicts_root) if verdicts_root is not None else None
    verdict_files: list[Path] = []
    if vroot is not None:
        verdict_files = _discover_fleet_eval_verdict_paths(vroot)
    if verdicts_root is None or not verdict_files:
        n_before = len(paths)
        if total_cap > 0 and n_before > total_cap:
            fifo_remove = paths[: n_before - total_cap]
        else:
            fifo_remove = []
        rem_stems_fb = [p.stem for p in fifo_remove]
        if not dry_run:
            n_removed = prune_checkpoint_zip_snapshots(checkpoint_dir, total_cap)
        else:
            n_removed = len(fifo_remove)
        paths_after = sorted_checkpoint_zip_paths(checkpoint_dir)
        kept_n = n_before - len(rem_stems_fb) if dry_run else len(paths_after)
        reason = (
            f"cold start: no verdicts_root or no eval/*.json; FIFO cap={total_cap} "
            f"({n_removed} removed)"
        )
        if dry_run:
            print(
                f"[curator dry-run] kept K={0} M={0} D={0} total={kept_n} "
                f"removed={len(rem_stems_fb)} (fallback=True)"
            )
        else:
            print(
                f"[curator] kept K={0} M={0} D={0} total={kept_n} "
                f"removed={n_removed} (fallback=True)"
            )
        return {
            "kept_total": kept_n,
            "kept_by_recency": [],
            "kept_by_winrate": [],
            "kept_by_diversity": [],
            "removed": rem_stems_fb,
            "fallback_used": True,
            "reason": reason,
        }
    # --- curated path (at least one verdict file on disk) ---
    wr_by_stem = _verdict_winrate_by_stem(verdict_files)
    now = time.time()
    min_age_s = max(0.0, float(min_age_minutes)) * 60.0

    def is_protected(p: Path) -> bool:
        try:
            return (now - p.stat().st_mtime) < min_age_s
        except OSError:
            return True

    protected = {p.stem for p in paths if is_protected(p)}

    k_take = min(max(0, k_newest), len(paths))
    k_stems = [p.stem for p in paths[-k_take:]] if k_take else []

    present = {p.stem for p in paths}
    ranked: list[tuple[float, str]] = []
    for st in present:
        if st in wr_by_stem:
            ranked.append((wr_by_stem[st], st))
    ranked.sort(key=lambda t: (-t[0], t[1]))
    m_take = min(max(0, m_top_winrate), len(ranked))
    m_stems = [st for _wr, st in ranked[:m_take]] if m_take else []

    km = set(k_stems) | set(m_stems)
    remaining = [p for p in paths if p.stem not in km]

    d_stems: list[str] = []
    d_slots = max(0, d_diversity)
    if d_slots and remaining:
        step_vals = [(_p, _training_step_from_stem(_p.stem)) for _p in remaining]
        use_step = all(t is not None for _p, t in step_vals) and len(remaining) >= 1
        if use_step:
            st_list = [t for _p, t in step_vals if t is not None]
            mn_s, mx_s = min(st_list), max(st_list)
        else:
            mn_s, mx_s = 0, 0
        if use_step and mn_s != mx_s:
            span_s = max(mx_s - mn_s, 1)
            buckets: dict[int, list[Path]] = {}
            for p, t in step_vals:
                if t is None:
                    continue
                b = min(9, int(9 * (t - mn_s) / span_s + 0.0))
                buckets.setdefault(b, []).append(p)
        else:
            mtimes: list[tuple[Path, float]] = []
            for p in remaining:
                try:
                    mtimes.append((p, float(p.stat().st_mtime)))
                except OSError:
                    continue
            if not mtimes:
                buckets = {}
            else:
                ts = [m for _p, m in mtimes]
                mn_t, mx_t = min(ts), max(ts)
                buckets = {}
                if mn_t == mx_t:
                    buckets[0] = [p for p, _m in mtimes]
                else:
                    span_t = max(mx_t - mn_t, 1e-9)
                    for p, m in mtimes:
                        frac = (m - mn_t) / span_t
                        b = min(9, int(10.0 * frac))
                        if b == 10:
                            b = 9
                        buckets.setdefault(b, []).append(p)
        left = d_slots
        for b in sorted(buckets.keys()):
            if left <= 0:
                break
            group = buckets[b]
            if not group:
                continue
            best_p = max(group, key=lambda p: p.stat().st_mtime)
            d_stems.append(best_p.stem)
            left -= 1

    keep: set[str] = set(k_stems) | set(m_stems) | set(d_stems) | protected
    to_delete: list[Path] = [p for p in paths if p.stem not in keep]
    rem_stems = [p.stem for p in to_delete]
    n_removed = 0
    if not dry_run:
        for p in to_delete:
            try:
                p.unlink()
                n_removed += 1
            except OSError:
                pass
    else:
        n_removed = len(to_delete)
    n_before = len(paths)
    kept_n = n_before - len(to_delete) if dry_run else len(sorted_checkpoint_zip_paths(checkpoint_dir))
    a, b, c = len(k_stems), len(m_stems), len(d_stems)
    reason = (
        f"curated keep union K={a} M={b} D={c} + {len(protected)} min-age; "
        f"removed {n_removed} zip(s)"
    )
    if dry_run:
        print(
            f"[curator dry-run] kept K={a} M={b} D={c} total={kept_n} "
            f"removed={n_removed} (fallback=False)"
        )
    else:
        print(
            f"[curator] kept K={a} M={b} D={c} total={kept_n} "
            f"removed={n_removed} (fallback=False)"
        )
    return {
        "kept_total": kept_n,
        "kept_by_recency": list(k_stems),
        "kept_by_winrate": list(m_stems),
        "kept_by_diversity": list(d_stems),
        "removed": rem_stems,
        "fallback_used": False,
        "reason": reason,
    }
