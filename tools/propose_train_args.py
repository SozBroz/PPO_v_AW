#!/usr/bin/env python3
"""
Propose PPO training args (n_envs, n_steps, batch_size) from a probe.json.

Writes fleet/<id>/proposed_args.json for operator review; does not auto-apply.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

PC_B_MAX_ENVS = 4
# Looser cap for throughput-tune sweeps (``start_solo_training --throughput-tune``);
# initial ``proposed_args`` still uses ``absolute_cap=12`` via ``max_safe_n_envs_from_probe``.
THROUGHPUT_TUNE_MAX_ENVS_CAP = 16
PC_B_REASON = (
    "operator-validated stable point; n_envs=6 dies at iter 5 (FPS plan 2026-04-22)"
)
HEURISTIC_REASON = "heuristic from probe; not yet validated by operator"

# Rollout geometry (PPO): raise --n-steps only when RAM suggests headroom and
# n_envs is moderate so n_envs * n_steps (buffer footprint) stays bounded.
_N_STEPS_DEFAULT = 512
_N_STEPS_LONG = 1024
_N_STEPS_EXTENDED = 2048
_RAM_GB_FOR_LONG_ROLLOUT = 24.0
_RAM_GB_FOR_EXTENDED_ROLLOUT = 64.0
_N_ENVS_MODERATE_FOR_LONG_ROLLOUT = 8


def _choose_n_steps(ram_gb: float, n_envs: int) -> tuple[int, str | None]:
    """Pick n_steps and an optional reasoning fragment (no fragment if default)."""
    if n_envs > _N_ENVS_MODERATE_FOR_LONG_ROLLOUT:
        return _N_STEPS_DEFAULT, None
    if ram_gb >= _RAM_GB_FOR_EXTENDED_ROLLOUT:
        return _N_STEPS_EXTENDED, (
            f"n_steps={_N_STEPS_EXTENDED} (total_ram>={_RAM_GB_FOR_EXTENDED_ROLLOUT}GiB, "
            f"n_envs<={_N_ENVS_MODERATE_FOR_LONG_ROLLOUT}, rollout headroom)"
        )
    if ram_gb >= _RAM_GB_FOR_LONG_ROLLOUT:
        return _N_STEPS_LONG, (
            f"n_steps={_N_STEPS_LONG} (total_ram>={_RAM_GB_FOR_LONG_ROLLOUT}GiB, "
            f"n_envs<={_N_ENVS_MODERATE_FOR_LONG_ROLLOUT}, rollout headroom)"
        )
    return _N_STEPS_DEFAULT, None


def _default_paths(machine_id: str) -> tuple[Path, Path]:
    return (
        REPO_ROOT / "fleet" / machine_id / "probe.json",
        REPO_ROOT / "fleet" / machine_id / "proposed_args.json",
    )


def _parse_probe_path(probe_path: Path) -> dict[str, Any]:
    return json.loads(probe_path.read_text(encoding="utf-8"))


def max_safe_n_envs_from_probe(
    probe: dict[str, Any], *, absolute_cap: int = 12
) -> int:
    """
    Upper bound for parallel env workers from probe hardware.

    Uses ``max(1, physical_cores - 2)`` (reserve two cores for OS / main / jitter)
    instead of ``phys // 2``, since each Subproc worker pins BLAS/torch to one thread
    by default — parallelism is mostly process count, not thread pools per env.

    Still bounded by ``total_ram_gb // 4`` (heuristic GiB per env) and *absolute_cap*.
    """
    cpu = probe.get("cpu") or {}
    ram = probe.get("ram") or {}
    try:
        phys = int(cpu.get("physical_cores", 0))
    except (TypeError, ValueError):
        phys = 0
    try:
        ram_gb = float(ram.get("total_gb", 0.0))
    except (TypeError, ValueError):
        ram_gb = 0.0
    core_budget = max(1, phys - 2)
    return min(core_budget, max(1, int(ram_gb // 4)), int(absolute_cap))


def propose_from_probe(probe: dict[str, Any]) -> dict[str, Any]:
    """Return full proposed_args document (not written)."""
    mid = str(probe.get("machine_id", "unknown"))
    based_on = str(probe.get("probed_at", ""))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    ram = probe.get("ram") or {}
    try:
        ram_gb = float(ram.get("total_gb", 0.0))
    except (TypeError, ValueError):
        ram_gb = 0.0

    max_safe = max_safe_n_envs_from_probe(probe, absolute_cap=12)

    if mid == "pc-b":
        n_envs = min(PC_B_MAX_ENVS, max_safe)
    else:
        n_envs = max_safe

    n_steps, geom_note = _choose_n_steps(ram_gb, n_envs)

    if mid == "pc-b":
        if n_steps > _N_STEPS_DEFAULT:
            batch_size = min(n_envs * n_steps // 4, 1024)
            if batch_size < 1:
                batch_size = 1
            reason = f"{PC_B_REASON}; {geom_note}" if geom_note else PC_B_REASON
        else:
            batch_size = 256
            reason = PC_B_REASON
    else:
        batch_size = min(n_envs * n_steps // 4, 1024)
        if batch_size < 1:
            batch_size = 1
        reason = (
            f"{HEURISTIC_REASON}; {geom_note}" if geom_note else HEURISTIC_REASON
        )

    return {
        "machine_id": mid,
        "proposed_at": now,
        "based_on_probe_at": based_on,
        "args": {
            "--n-envs": n_envs,
            "--n-steps": n_steps,
            "--batch-size": batch_size,
        },
        "reasoning": reason,
        "auto_apply": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--machine-id",
        type=str,
        default=None,
        help=(
            "Fleet machine id (e.g. pc-b). Default probe: fleet/<id>/probe.json; "
            "default out: fleet/<id>/proposed_args.json (unless --probe/--out)."
        ),
    )
    ap.add_argument(
        "--probe",
        type=Path,
        default=None,
        help="Path to probe.json (default: fleet/<--machine-id or AWBW_MACHINE_ID>/probe.json)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: fleet/<id>/proposed_args.json from probe JSON)",
    )
    ap.add_argument("--print-only", action="store_true", help="Stdout only; no file write")
    args = ap.parse_args()

    cli_mid = str(args.machine_id).strip() if args.machine_id else None
    env_id = os.environ.get("AWBW_MACHINE_ID")
    env_id = str(env_id).strip() if env_id else "unknown"
    default_id = cli_mid if cli_mid else env_id
    if args.probe is not None:
        probe_path = args.probe
    else:
        probe_path = _default_paths(default_id)[0]

    if not probe_path.is_file():
        print(f"probe not found: {probe_path}", file=sys.stderr)
        return 1

    probe = _parse_probe_path(probe_path)
    doc = propose_from_probe(probe)
    if args.out is not None:
        out: Path = args.out
    else:
        mid = str(doc.get("machine_id", default_id))
        out = REPO_ROOT / "fleet" / mid / "proposed_args.json"

    text = json.dumps(doc, indent=2) + "\n"
    sys.stdout.write(text)
    if not args.print_only:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
