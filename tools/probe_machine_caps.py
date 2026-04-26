#!/usr/bin/env python3
"""
Emit a JSON descriptor of this machine's hardware (CPU, RAM, GPU, disk) for fleet auto-tune.

Default: fleet/<AWBW_MACHINE_ID>/probe.json (or fleet/unknown/ if unset). Requires psutil.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Repo root: tools/ -> parent.parent
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from rl import _win_triton_warnings

_win_triton_warnings.apply()

try:
    import psutil
except ImportError:  # pragma: no cover
    print("psutil is required: pip install psutil", file=sys.stderr)
    raise SystemExit(2)


def _default_out_path(machine_id_cli: str | None = None) -> Path:
    if machine_id_cli is not None and str(machine_id_cli).strip():
        mid = str(machine_id_cli).strip()
    else:
        mid = os.environ.get("AWBW_MACHINE_ID")
        if mid is None or not str(mid).strip():
            mid = "unknown"
        else:
            mid = str(mid).strip()
    return REPO_ROOT / "fleet" / mid / "probe.json"


def _try_cuda_props() -> tuple[bool, str | None, float | None]:
    try:
        import torch
    except ImportError:
        return False, None, None
    if not torch.cuda.is_available():
        return False, None, None
    try:
        p = torch.cuda.get_device_properties(0)
        name = p.name
        vram_bytes = p.total_memory
        vram_gb = round(vram_bytes / (1024.0**3), 2)
        return True, str(name), vram_gb
    except Exception:  # noqa: BLE001
        return True, None, None


def _cpu_model() -> str:
    import platform

    p = (platform.processor() or "").strip()
    if p:
        return p
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if "model name" in line.lower():
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return (os.environ.get("PROCESSOR_IDENTIFIER", "") or "").strip()


def build_probe_payload(*, machine_id_override: str | None = None) -> dict:
    probed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if machine_id_override is not None and str(machine_id_override).strip():
        machine_id = str(machine_id_override).strip()
    else:
        mid = os.environ.get("AWBW_MACHINE_ID")
        if mid is None or not str(mid).strip():
            machine_id = "unknown"
        else:
            machine_id = str(mid).strip()

    phys = psutil.cpu_count(logical=False) or 0
    logical = psutil.cpu_count(logical=True) or 0
    model = _cpu_model()

    vm = psutil.virtual_memory()
    total_gb = round(vm.total / (1024.0**3), 2)
    free_gb = round(vm.available / (1024.0**3), 2)

    gpu_ok, dev_name, vram_gb = _try_cuda_props()

    ck = os.environ.get("AWBW_CHECKPOINT_DIR")
    if ck and str(ck).strip():
        checkpoint_root = Path(ck).expanduser().resolve()
    else:
        checkpoint_root = (REPO_ROOT / "checkpoints").resolve()
    w_ok = False
    try:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        t = checkpoint_root / ".awbw_probe_write_test"
        t.write_text("ok", encoding="utf-8")
        t.unlink(missing_ok=True)  # type: ignore[arg-type]
        w_ok = True
    except OSError:
        w_ok = False

    return {
        "machine_id": machine_id,
        "probed_at": probed_at,
        "cpu": {
            "physical_cores": int(phys),
            "logical_processors": int(logical),
            "model_name": model,
        },
        "ram": {
            "total_gb": float(total_gb),
            "free_gb_at_probe": float(free_gb),
        },
        "gpu": {
            "available": gpu_ok,
            "device_name": dev_name,
            "vram_total_gb": vram_gb,
        },
        "disk": {
            "checkpoint_root_writable": w_ok,
            "checkpoint_root_path": str(checkpoint_root),
        },
        "platform": sys.platform,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--machine-id",
        type=str,
        default=None,
        help=(
            "Fleet machine id (e.g. pc-b). Sets payload machine_id and default "
            "--out to fleet/<id>/probe.json."
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output JSON (default: fleet/<machine-id or AWBW_MACHINE_ID>/probe.json)",
    )
    ap.add_argument(
        "--print-only",
        action="store_true",
        help="Write JSON to stdout only; do not write a file",
    )
    args = ap.parse_args()
    cli_mid = str(args.machine_id).strip() if args.machine_id else None
    out = args.out if args.out is not None else _default_out_path(cli_mid)
    payload = build_probe_payload(machine_id_override=cli_mid)
    text = json.dumps(payload, indent=2) + "\n"
    sys.stdout.write(text)
    if not args.print_only:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
