#!/usr/bin/env python3
"""
Recompile Cython extensions and place ``*.pyd`` next to ``engine/`` and ``rl/`` packages.

Use this on Windows when ``python setup.py build_ext --inplace`` fails with
"could not delete ... .pyd: Access is denied" — a running ``train.py``, REPL, or
test process has the old DLL loaded and Windows will not let setuptools overwrite it.

This script:
1. Runs ``setup.py build_ext`` (output under ``build/``, no in-place delete).
2. Copies fresh ``.pyd`` files into ``engine/`` and ``rl/``.

If step 2 still fails, stop Python processes that import ``engine`` or ``rl`` (or
reboot) and re-run, or copy manually from ``build/lib.*/ engine/`` and ``rl/``.
"""
from __future__ import annotations

import errno
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Windows: ERROR_SHARING_VIOLATION when another process has the .pyd loaded.
_LOCK_WINERROR = 32


def _is_file_in_use_error(exc: BaseException) -> bool:
    if isinstance(exc, PermissionError):
        return True
    if isinstance(exc, OSError):
        if getattr(exc, "winerror", None) == _LOCK_WINERROR:
            return True
        if exc.errno in (errno.EACCES, errno.EPERM, errno.ETXTBSY):
            return True
    return False


def _copy_pyd_with_retries(src: Path, dest: Path, *, attempts: int = 6) -> None:
    delays_s = (0.4, 0.8, 1.2, 2.0, 2.5, 3.0)
    for attempt in range(attempts):
        try:
            shutil.copy2(src, dest)
            return
        except OSError as exc:
            if not _is_file_in_use_error(exc):
                raise
            if attempt + 1 >= attempts:
                raise
            delay = delays_s[attempt] if attempt < len(delays_s) else delays_s[-1]
            time.sleep(delay)


def main() -> int:
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "setup.py"), "build_ext"],
        cwd=str(REPO_ROOT),
    )
    if r.returncode != 0:
        return r.returncode

    # build/lib.*-*/{engine,rl}/*.pyd
    build_lib = None
    for c in (REPO_ROOT / "build").iterdir():
        if c.is_dir() and c.name.startswith("lib."):
            build_lib = c
            break
    if build_lib is None:
        print("No build/lib.* directory found; build_ext may have been a no-op.", file=sys.stderr)
        return 1

    copied = 0
    for sub in ("engine", "rl"):
        src_dir = build_lib / sub
        if not src_dir.is_dir():
            continue
        dest_dir = REPO_ROOT / sub
        for pyd in src_dir.glob("*.pyd"):
            dest = dest_dir / pyd.name
            try:
                _copy_pyd_with_retries(pyd, dest)
                print(f"copied {pyd.name} -> {dest.relative_to(REPO_ROOT)}")
                copied += 1
            except OSError as exc:
                print(
                    f"ERROR: could not copy {pyd} -> {dest}:\n  {exc}\n"
                    "Stop any process that has imported the old extension (train.py, "
                    "pytest, an IDE-embedded REPL) and re-run, or copy the file by hand while idle.\n"
                    "If Cython sources are unchanged, re-run the bootstrap with "
                    "`--skip-cython-rebuild`.\n"
                    "To list suspects (including SubprocVecEnv multiprocessing.spawn workers): "
                    f"`{sys.executable} scripts/diagnose_cython_lock.py`.",
                    file=sys.stderr,
                )
                return 1
    if copied == 0:
        print("No .pyd files found under build/; nothing to copy.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
