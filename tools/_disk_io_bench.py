"""One-off sequential R/W MB/s benchmark. Delete after use."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

CHUNK = 1 << 20  # 1 MiB


def run(path: Path, size_mib: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fn = path / "_seq_bench_temp.bin"
    block = os.urandom(CHUNK)
    if fn.exists():
        fn.unlink()
    t0 = time.perf_counter()
    with open(fn, "wb", buffering=CHUNK) as f:
        for _ in range(size_mib):
            f.write(block)
        f.flush()
        os.fsync(f.fileno())
    t1 = time.perf_counter()
    with open(fn, "rb", buffering=CHUNK) as f:
        n = 0
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            n += len(b)
    t2 = time.perf_counter()
    try:
        fn.unlink()
    except OSError:
        pass
    w_s = t1 - t0
    r_s = t2 - t1
    mib = size_mib
    print(f"path={path}")
    print(f"size_MiB={mib} seq_write_s={w_s:.3f} write_MiB_s={mib / w_s:.1f} seq_read_s={r_s:.3f} read_MiB_s={mib / r_s:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path, help="Directory to create temp file in")
    ap.add_argument("--mib", type=int, default=256, help="File size in MiB (default 256)")
    args = ap.parse_args()
    run(args.dir.resolve(), args.mib)


if __name__ == "__main__":
    main()
