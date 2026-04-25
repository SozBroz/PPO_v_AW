#!/usr/bin/env python3
"""
Fleet checkpoint zip sync — delegates to tools/sync_checkpoint_zips_fleet.ps1.

Only checkpoint*.zip files are transferred. latest.zip is never synced.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync checkpoint*.zip with workhorse1 (never latest.zip).",
    )
    parser.add_argument(
        "--direction",
        choices=("Pull", "Push", "Both"),
        default="Pull",
        help="Pull = remote root zips -> local pool; Push = local root zips -> remote pool; Both = both.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="AWBW repo root (default: parent of tools/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Forwarded to PowerShell (remote inventory still runs).",
    )
    args = parser.parse_args()

    tools_dir = Path(__file__).resolve().parent
    script = tools_dir / "sync_checkpoint_zips_fleet.ps1"
    if not script.is_file():
        print(f"Missing {script}", file=sys.stderr)
        return 2

    repo = args.repo_root
    if repo is None:
        repo = tools_dir.parent

    ps_args = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-Direction",
        args.direction,
        "-RepoRoot",
        str(repo.resolve()),
    ]
    if args.dry_run:
        ps_args.append("-DryRun")

    return subprocess.call(ps_args)


if __name__ == "__main__":
    raise SystemExit(main())
