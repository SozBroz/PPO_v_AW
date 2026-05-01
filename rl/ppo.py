"""
PPO configuration utilities for AWBW self-play training.

Primary training is delegated to sb3_contrib.MaskablePPO.
This module centralises hyperparameter sets and provides a factory
function so the rest of the codebase has a single place to change them.
"""
from __future__ import annotations

from typing import Any, Optional

# ── Hyperparameter presets ────────────────────────────────────────────────────

DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_steps": 2048,          # steps collected per update cycle (per env)
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99925,          # long-horizon discount (AWBW episodes are many learner-steps)
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.02,         # entropy bonus encourages exploration
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalize_advantage": True,
}

# Faster / noisier preset for early training or rapid iteration
FAST_HYPERPARAMS: dict[str, Any] = {
    **DEFAULT_HYPERPARAMS,
    "n_steps": 512,
    "batch_size": 32,
    "n_epochs": 4,
    "ent_coef": 0.05,         # more exploration at the start
}

# Conservative preset for fine-tuning a pre-trained model
FINETUNE_HYPERPARAMS: dict[str, Any] = {
    **DEFAULT_HYPERPARAMS,
    "learning_rate": 1e-4,
    "n_epochs": 5,
    "ent_coef": 0.005,
    "clip_range": 0.1,
}

HYPERPARAMS_REGISTRY: dict[str, dict[str, Any]] = {
    "default": DEFAULT_HYPERPARAMS,
    "fast": FAST_HYPERPARAMS,
    "finetune": FINETUNE_HYPERPARAMS,
}


# ── Factory ───────────────────────────────────────────────────────────────────

def make_maskable_ppo(
    env,
    hyperparams: dict[str, Any] | str | None = None,
    tensorboard_log: Optional[str] = None,
    verbose: int = 1,
):
    """
    Create a MaskablePPO model ready for AWBW self-play training.

    Parameters
    ----------
    env:
        A Gymnasium-compatible environment (should already be wrapped with
        ActionMasker so MaskablePPO can retrieve action masks).
    hyperparams:
        Either a dict of PPO kwargs, a preset name from HYPERPARAMS_REGISTRY
        ("default" | "fast" | "finetune"), or None to use DEFAULT_HYPERPARAMS.
    tensorboard_log:
        Directory path for TensorBoard logs. None disables logging.
    verbose:
        Verbosity level passed to MaskablePPO (0 = silent, 1 = info).

    Returns
    -------
    MaskablePPO
        Configured model, not yet trained.

    Raises
    ------
    ImportError
        If sb3-contrib is not installed.
    ValueError
        If `hyperparams` is a string not found in HYPERPARAMS_REGISTRY.
    """
    try:
        from sb3_contrib import MaskablePPO  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "sb3-contrib is required. Install it with: pip install sb3-contrib"
        ) from exc

    if hyperparams is None:
        hp = dict(DEFAULT_HYPERPARAMS)
    elif isinstance(hyperparams, str):
        if hyperparams not in HYPERPARAMS_REGISTRY:
            raise ValueError(
                f"Unknown hyperparams preset '{hyperparams}'. "
                f"Choose from: {list(HYPERPARAMS_REGISTRY)}"
            )
        hp = dict(HYPERPARAMS_REGISTRY[hyperparams])
    else:
        hp = {**DEFAULT_HYPERPARAMS, **hyperparams}

    return MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        **hp,
    )


def load_or_create(
    env,
    checkpoint_path: Optional[str] = None,
    hyperparams: dict[str, Any] | str | None = None,
    tensorboard_log: Optional[str] = None,
    verbose: int = 1,
):
    """
    Load an existing MaskablePPO checkpoint or create a fresh model.

    Parameters
    ----------
    env:
        ActionMasker-wrapped AWBW environment.
    checkpoint_path:
        Path to a .zip checkpoint saved by MaskablePPO.save(). If None or
        the file does not exist, a new model is initialised.
    hyperparams:
        Hyperparameter dict or preset name. Only used when creating a new model;
        ignored when loading an existing checkpoint.
    tensorboard_log:
        TensorBoard log directory.
    verbose:
        Verbosity passed to MaskablePPO.

    Returns
    -------
    MaskablePPO
    """
    try:
        from sb3_contrib import MaskablePPO  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "sb3-contrib is required. Install it with: pip install sb3-contrib"
        ) from exc

    from pathlib import Path

    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        # sb3 saves without extension internally but accepts both forms
        if not p.suffix:
            p = p.with_suffix(".zip")
        if p.exists():
            print(f"[ppo] Loading checkpoint: {p}")
            return MaskablePPO.load(str(p), env=env)

    print("[ppo] No checkpoint found — initialising new model.")
    return make_maskable_ppo(
        env,
        hyperparams=hyperparams,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )
