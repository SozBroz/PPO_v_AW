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
import platform

_TRITON_FILTER_APPLIED = False


def ensure_win32_processor_identifier() -> None:
    """
    If unset, set ``PROCESSOR_IDENTIFIER`` so ``platform`` never queries WMI
    (see CPython :mod:`platform` ``_Processor.get`` on win32). Idempotent; safe
    to call from every :func:`apply` and before ``platform.processor()``.
    
    Also handles the case where PROCESSOR_IDENTIFIER might be set to an empty
    string or whitespace-only string, which would still trigger WMI queries.
    
    Additional protection: also monkey-patch platform._get_machine_win32 to
    handle KeyError from WMI queries directly.
    """
    if sys.platform == "win32":
        proc_id = os.environ.get("PROCESSOR_IDENTIFIER")
        if not proc_id or not proc_id.strip():
            # Set to a non-empty value to avoid WMI queries
            # Using "Unknown" as a safe fallback that won't cause issues
            os.environ["PROCESSOR_IDENTIFIER"] = "Unknown"
        
        # Also apply direct monkey patch to handle WMI KeyError
        _apply_platform_monkeypatch()


def _apply_platform_monkeypatch() -> None:
    """
    Monkey-patch platform._get_machine_win32 to handle KeyError from WMI.
    
    This addresses the specific issue where _wmi_query('CPU', 'Architecture')
    raises KeyError because the 'Architecture' key doesn't exist in WMI response.
    """
    if sys.platform != "win32":
        return
    
    # Check if platform has _get_machine_win32 function
    if not hasattr(platform, '_get_machine_win32'):
        return
    
    # Store reference to original
    _original_get_machine_win32 = platform._get_machine_win32
    
    def patched_get_machine_win32():
        """Patched version that handles KeyError and other exceptions from WMI."""
        try:
            # Try the original function first
            return _original_get_machine_win32()
        except (KeyError, TypeError, AttributeError):
            # Handle various WMI-related errors
            # Return empty string as fallback
            return ""
        except Exception:
            # Catch any other unexpected errors
            return ""
    
    # Apply the patch
    platform._get_machine_win32 = patched_get_machine_win32


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
