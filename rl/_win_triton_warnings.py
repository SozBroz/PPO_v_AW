"""
Triton (Windows) emits UserWarning: Failed to find CUDA when the CUDA *toolkit*
is not on PATH. That is only for Triton's own JIT tooling — PyTorch still uses
the GPU via the NVIDIA driver and ``torch.cuda``. Silence that noisy warning
after training/tests install the filter (see ``apply()``).

**Performance:** `warnings.filterwarnings` only affects *printing*; it does not
change device placement, kernels, or steps/sec. Training speed is unchanged.
"""
from __future__ import annotations

import sys
import warnings

_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED or sys.platform != "win32":
        return
    _APPLIED = True
    warnings.filterwarnings(
        "ignore",
        message="Failed to find CUDA",
        category=UserWarning,
        module=r"triton\.windows_utils",
    )
