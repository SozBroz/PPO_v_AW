"""Load PPO checkpoint, ignoring shape mismatches."""
from sb3_contrib import MaskablePPO
import torch

def load_ppo_simple(ckpt_path: str, device: str = "cpu"):
    """Load PPO checkpoint with strict=False to ignore shape mismatches."""
    # Load the checkpoint file directly
    import io
    from rl.ckpt_compat import materialize_sb3_zip_with_spatial_compat
    
    # Patch if needed
    patched_path, was_patched = materialize_sb3_zip_with_spatial_compat(ckpt_path)
    if was_patched:
        print(f"[eval] Using patched checkpoint: {patched_path}")
        ckpt_to_load = patched_path
    else:
        ckpt_to_load = ckpt_path
    
    # Load with strict=False to ignore mismatches
    model = MaskablePPO.load(
        ckpt_to_load,
        device=device,
        custom_objects={"n_steps": 2048, "n_envs": 1},
    )
    return model
