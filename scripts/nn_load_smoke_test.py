#!/usr/bin/env python3
"""
Load the MaskablePPO policy (or build a fresh one) and run one forward pass
to surface **warnings and errors** from PyTorch, SB3, and our code.

Does **not** import ``rl.env`` (to avoid the Windows Triton warning filter in
``env.py``) so you see the same library noise a bare ``torch`` + load path gets.

Run for maximum detail:
  python -W default scripts/nn_load_smoke_test.py
  python -W default scripts/nn_load_smoke_test.py --checkpoint path/to/ckpt.zip
  python -W default scripts/nn_load_smoke_test.py --fresh
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
import zipfile
from pathlib import Path

# Must run before heavy imports: surface everything we can in-process.
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logging.captureWarnings(True)
warnings.simplefilter("always")
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _random_obs() -> dict:
    import numpy as np
    from rl.encoder import GRID_SIZE, N_SCALARS, N_SPATIAL_CHANNELS
    from rl.network import ACTION_SPACE_SIZE

    rng = np.random.default_rng(0)
    return {
        "spatial": rng.random(
            (1, GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS), dtype=np.float32
        ),
        "scalars": rng.standard_normal((1, N_SCALARS), dtype=np.float32) * 0.1,
    }, np.ones((1, ACTION_SPACE_SIZE), dtype=bool)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load AWBW PPO policy and one forward; print all warnings / errors."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO / "checkpoints" / "latest.zip",
        help="Path to MaskablePPO zip (default: checkpoints/latest.zip).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cuda, cpu, or auto (default).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Do not load a zip; build a new MultiInputPolicy + AWBWFeaturesExtractor.",
    )
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = _default_device()

    print("=== nn_load_smoke_test ===", file=sys.stderr)
    print(
        f"device={device!r}  fresh={args.fresh}  checkpoint={args.checkpoint!s}",
        file=sys.stderr,
    )
    print(
        "tip: re-run with  python -W default  to surface dependency deprecations",
        file=sys.stderr,
    )

    import numpy as np
    import torch
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.utils import obs_as_tensor

    from rl.encoder import GRID_SIZE, N_SPATIAL_CHANNELS, N_SCALARS
    from rl.network import ACTION_SPACE_SIZE, AWBWFeaturesExtractor

    class _MinimalGym(gym.Env):
        """Minimal Gymnasium env for SB3 shape checks; only spaces + reset/step."""

        metadata: dict = {"render_modes": []}

        def __init__(self) -> None:
            self.observation_space = spaces.Dict(
                {
                    "spatial": spaces.Box(
                        0.0, 1.0, (GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS), np.float32
                    ),
                    "scalars": spaces.Box(
                        -1.0, 10.0, (N_SCALARS,), np.float32
                    ),
                }
            )
            self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        def reset(self, seed=None, options=None):
            obs, _ = _random_obs()
            return {k: v[0] for k, v in obs.items()}, {}

        def step(self, action):
            o, m = _random_obs()
            return {k: v[0] for k, v in o.items()}, 0.0, True, False, {}

    venv = DummyVecEnv([lambda: _MinimalGym()])
    policy_kwargs = {
        "features_extractor_class": AWBWFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
    }

    if args.fresh:
        from sb3_contrib import MaskablePPO  # type: ignore[import]

        print("--- building fresh MaskablePPO ---", file=sys.stderr)
        model = MaskablePPO(
            "MultiInputPolicy",
            venv,
            device=device,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=8,
            batch_size=8,
        )
    else:
        p = args.checkpoint
        if not p.is_file():
            print(
                f"error: checkpoint not found: {p}  (use --fresh or --checkpoint)",
                file=sys.stderr,
            )
            return 1
        with zipfile.ZipFile(p, "r") as zf:
            inside = "policy.pth" in zf.namelist() or "data" in zf.namelist()
        if not inside:
            print(f"error: not a valid SB3-style zip: {p}", file=sys.stderr)
            return 1
        from rl.ckpt_compat import load_maskable_ppo_compat  # type: ignore[import]

        print(f"--- loading {p} ---", file=sys.stderr)
        model = load_maskable_ppo_compat(
            p,
            env=None,
            device=device,
            custom_objects={"n_steps": 64, "n_envs": 1},
        )
        model.set_env(venv)

    obs, masks = _random_obs()
    print("--- model.predict (numpy obs + masks) ---", file=sys.stderr)
    a, out = model.predict(
        obs,
        deterministic=True,
        action_masks=masks,
    )
    print(f"action={a!r}  out_keys={out if out is not None else None}", file=sys.stderr)

    print("--- policy forward (tensor, batched) ---", file=sys.stderr)
    o_t = obs_as_tensor(obs, model.device)  # type: ignore[arg-type]
    m_t = torch.as_tensor(masks, device=model.device, dtype=torch.bool)
    with torch.no_grad():
        if hasattr(model.policy, "get_distribution"):
            d = model.policy.get_distribution(o_t)  # type: ignore[union-attr]
            print(
                f"distribution: {type(d)}  n_actions? ok",
                file=sys.stderr,
            )
    print("--- done (no unhandled exception) ---", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
