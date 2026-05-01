# -*- coding: utf-8 -*-
"""Phase 11 Slice E — capture per-machine ``--mcts-mode off`` baseline.

CLI wrapper around :mod:`scripts.symmetric_checkpoint_eval`. For one machine,
runs a fixed-seat head-to-head with ``--mcts-mode off``, aggregates
``games_decided`` and ``winrate``, hashes the candidate zip, and writes the
result into ``<shared>/fleet/<machine_id>/mcts_off_baseline.json`` (atomic).

The escalator (Slice D, future composer) consumes this file as the
``mcts_off_baseline`` term in :class:`tools.mcts_escalator.EscalatorCycleResult`.

Defaults intentionally mirror :mod:`scripts.fleet_eval_daemon`:

* candidate     → ``<shared>/checkpoints/pool/<machine_id>/latest.zip``
                  (newest snapshot if ``latest.zip`` is absent)
* baseline opp. → ``<shared>/checkpoints/promoted/best.zip`` then
                  ``<shared>/checkpoints/latest.zip``
* map / tier / COs → same baseline (123858 / T3 / 1v1).

Exit codes::

  0   wrote baseline file
  1   missing checkpoint or eval entrypoint failed (any non-success)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.fleet_env import (  # noqa: E402
    sorted_checkpoint_zip_paths,
    verdict_summary_from_symmetric_json,
)
from tools.mcts_baseline import (  # noqa: E402
    MCTS_OFF_BASELINE_SCHEMA_VERSION,
    MctsOffBaseline,
    baseline_path,
    utc_now_iso_z,
    write_baseline,
)

SOURCE_TAG = "tools/capture_mcts_baseline.py"
DEFAULT_MAP_ID = 123858
DEFAULT_TIER = "T3"
DEFAULT_GAMES = 200


def resolve_pool_latest(shared_root: Path, machine_id: str) -> Path | None:
    """Return ``<shared>/checkpoints/pool/<machine_id>/latest.zip``, or newest snapshot."""
    pool_dir = Path(shared_root) / "checkpoints" / "pool" / str(machine_id)
    latest = pool_dir / "latest.zip"
    if latest.is_file():
        return latest
    paths = sorted_checkpoint_zip_paths(pool_dir)
    return paths[-1] if paths else None


def resolve_default_opponent(shared_root: Path) -> Path | None:
    """Same fallback chain as ``scripts/fleet_eval_daemon.py``."""
    ck = Path(shared_root) / "checkpoints"
    for cand in (ck / "promoted" / "best.zip", ck / "latest.zip"):
        if cand.is_file():
            return cand
    return None


def sha256_of_file(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for blob in iter(lambda: f.read(chunk), b""):
            h.update(blob)
    return h.hexdigest()


def split_games_first_second(total: int) -> tuple[int, int]:
    """Split *total* into (first_seat, second_seat) — first seat gets the odd one."""
    total = max(0, int(total))
    half = total // 2
    return (half + (total % 2), half)


def _run_symmetric_eval(
    *,
    candidate: Path,
    baseline: Path,
    map_id: int,
    tier: str,
    co_p0: int,
    co_p1: int,
    games_first: int,
    games_second: int,
    seed: int,
    json_out: Path,
    repo_root: Path,
    max_env_steps: int | None,
    max_days: int | None,
) -> dict:
    """Invoke ``scripts/symmetric_checkpoint_eval.py --mcts-mode off`` and return parsed JSON.

    Tests monkeypatch this entrypoint to inject canned verdicts without
    touching the real PPO loader.
    """
    sym = Path(repo_root) / "scripts" / "symmetric_checkpoint_eval.py"
    cmd = [
        sys.executable,
        str(sym),
        "--candidate",
        str(candidate),
        "--baseline",
        str(baseline),
        "--map-id",
        str(int(map_id)),
        "--tier",
        str(tier),
        "--co-p0",
        str(int(co_p0)),
        "--co-p1",
        str(int(co_p1)),
        "--games-first-seat",
        str(int(games_first)),
        "--games-second-seat",
        str(int(games_second)),
        "--seed",
        str(int(seed)),
        "--mcts-mode",
        "off",
        "--json-out",
        str(json_out),
    ]
    if max_env_steps is not None:
        cmd += ["--max-env-steps", str(int(max_env_steps))]
    if max_days is not None:
        cmd += ["--max-days", str(int(max_days))]
    subprocess.run(cmd, check=True, cwd=str(repo_root))
    return json.loads(Path(json_out).read_text(encoding="utf-8"))


def build_arg_parser() -> argparse.ArgumentParser:
    """Public so tests / docs can introspect CLI surface without invoking ``main``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--machine-id", required=True, help="Fleet machine id (e.g. pc-b)")
    ap.add_argument(
        "--checkpoint-zip",
        type=Path,
        default=None,
        help="Candidate zip; default: <shared>/checkpoints/pool/<machine-id>/latest.zip",
    )
    ap.add_argument(
        "--baseline-zip",
        type=Path,
        default=None,
        help="Opponent zip; default: promoted/best.zip then latest.zip under shared root",
    )
    ap.add_argument(
        "--shared-root",
        type=Path,
        default=REPO_ROOT,
        help="Shared root (Z:\\ on aux, repo root on main); default repo root.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Override output path; default: "
            "<shared>/fleet/<machine-id>/mcts_off_baseline.json"
        ),
    )
    ap.add_argument(
        "--games",
        type=int,
        default=DEFAULT_GAMES,
        help=f"Total decided-or-not games (split across both seats); default {DEFAULT_GAMES}.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map-id", type=int, default=DEFAULT_MAP_ID)
    ap.add_argument("--tier", type=str, default=DEFAULT_TIER)
    ap.add_argument("--co-p0", type=int, default=1)
    ap.add_argument("--co-p1", type=int, default=1)
    ap.add_argument(
        "--max-env-steps",
        type=int,
        default=0,
        help="Forwarded to symmetric eval; 0 = unlimited (Slice E default — runs to natural end).",
    )
    ap.add_argument(
        "--max-days",
        "--max-turns",
        dest="max_days",
        type=int,
        default=None,
        help="Forwarded to symmetric eval (end-inclusive calendar tiebreak).",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    shared_root = Path(args.shared_root).resolve()
    machine_id = str(args.machine_id).strip()
    if not machine_id:
        print("[capture_mcts_baseline] --machine-id must be non-empty", file=sys.stderr)
        return 1

    candidate = (
        Path(args.checkpoint_zip).resolve()
        if args.checkpoint_zip is not None
        else resolve_pool_latest(shared_root, machine_id)
    )
    if candidate is None or not Path(candidate).is_file():
        print(
            f"[capture_mcts_baseline] candidate checkpoint missing: {candidate!s} "
            f"(machine={machine_id}, shared_root={shared_root})",
            file=sys.stderr,
        )
        return 1

    baseline = (
        Path(args.baseline_zip).resolve()
        if args.baseline_zip is not None
        else resolve_default_opponent(shared_root)
    )
    if baseline is None or not Path(baseline).is_file():
        print(
            "[capture_mcts_baseline] baseline opponent missing "
            f"(looked under {shared_root}/checkpoints/promoted/best.zip and latest.zip)",
            file=sys.stderr,
        )
        return 1

    games_first, games_second = split_games_first_second(int(args.games))
    if games_first + games_second <= 0:
        print("[capture_mcts_baseline] --games must be > 0", file=sys.stderr)
        return 1

    out_path = (
        Path(args.out).resolve()
        if args.out is not None
        else baseline_path(machine_id, shared_root)
    )

    max_env_steps = None if int(args.max_env_steps) <= 0 else int(args.max_env_steps)

    print(
        f"[capture_mcts_baseline] machine={machine_id} candidate={candidate} "
        f"baseline={baseline} games={games_first}+{games_second} mode=off seed={int(args.seed)}"
    )

    with tempfile.TemporaryDirectory(prefix="mcts_baseline_") as td:
        sym_json = Path(td) / "sym.json"
        try:
            sym_data = _run_symmetric_eval(
                candidate=Path(candidate),
                baseline=Path(baseline),
                map_id=int(args.map_id),
                tier=str(args.tier),
                co_p0=int(args.co_p0),
                co_p1=int(args.co_p1),
                games_first=int(games_first),
                games_second=int(games_second),
                seed=int(args.seed),
                json_out=sym_json,
                repo_root=REPO_ROOT,
                max_env_steps=max_env_steps,
                max_days=args.max_days,
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"[capture_mcts_baseline] symmetric eval failed (rc={exc.returncode})",
                file=sys.stderr,
            )
            return 1
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[capture_mcts_baseline] eval output unreadable: {exc}", file=sys.stderr)
            return 1

    summary = verdict_summary_from_symmetric_json(sym_data)
    games_decided = int(summary.get("games_decided", 0))
    winrate = float(summary.get("winrate", 0.0))

    baseline_obj = MctsOffBaseline(
        schema_version=MCTS_OFF_BASELINE_SCHEMA_VERSION,
        machine_id=machine_id,
        captured_at=utc_now_iso_z(),
        checkpoint_zip=str(candidate),
        checkpoint_zip_sha256=sha256_of_file(Path(candidate)),
        games_decided=games_decided,
        winrate_vs_pool=winrate,
        mcts_mode="off",
        source=SOURCE_TAG,
    )

    written = write_baseline(baseline_obj, out_path.parent)
    if written != out_path:
        # Honor explicit --out filename, not just its parent dir.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Path(written).replace(out_path)
        written = out_path
    print(
        f"[capture_mcts_baseline] wrote {written} games_decided={games_decided} "
        f"winrate={winrate:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
