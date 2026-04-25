"""One-shot: verify train.py stack imports (for fleet setup). Run: python tools/_train_env_check.py"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch  # noqa: F401
        import gymnasium  # noqa: F401
        from sb3_contrib import MaskablePPO  # noqa: F401
        from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: F401
    except ImportError as e:
        print("IMPORT_FAIL:", e, file=sys.stderr)
        return 1
    print("OK torch=", __import__("torch").__version__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
