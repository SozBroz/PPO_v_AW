"""
Triton (Windows) emits UserWarning: Failed to find CUDA when the CUDA *toolkit*
is not on PATH. That is only for Triton's own JIT tooling — PyTorch still uses
the GPU via the NVIDIA driver and ``torch.cuda``. Silence that noisy warning
after training/tests install the filter (see ``apply()``).

**Performance:** `warnings.filterwarnings` only affects *printing*; it does not
change device placement, kernels, or steps/sec. Training speed is unchanged.

**Windows + Py3.12+:** if ``PROCESSOR_IDENTIFIER`` is missing, the stdlib
:func:`platform.processor` / ``uname`` path calls WMI (``_wmi_query``), which
can KeyError or otherwise fail on some hosts. Setting a placeholder is what
``platform`` would do after WMI success anyway; it avoids the WMI call.
"""
from __future__ import annotations

import os
import sys
import warnings

_TRITON_FILTER_APPLIED = False


def ensure_win32_processor_identifier() -> None:
    """
    If unset, set ``PROCESSOR_IDENTIFIER`` so ``platform`` never queries WMI
    (see CPython :mod:`platform` ``_Processor.get`` on win32). Idempotent; safe
    to call from every :func:`apply` and before ``platform.processor()``.
    """
    if sys.platform == "win32" and not (os.environ.get("PROCESSOR_IDENTIFIER") or "").strip():
        os.environ["PROCESSOR_IDENTIFIER"] = "Unknown"


def apply() -> None:
    if sys.platform != "win32":
        return
    ensure_win32_processor_identifier()
    global _TRITON_FILTER_APPLIED
    if _TRITON_FILTER_APPLIED:
        return
    _TRITON_FILTER_APPLIED = True
    warnings.filterwarnings(
        "ignore",
        message="Failed to find CUDA",
        category=UserWarning,
        module=r"triton\.windows_utils",
    )
