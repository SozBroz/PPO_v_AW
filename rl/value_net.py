from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rl.encoder import GRID_SIZE, N_SCALARS, N_SPATIAL_CHANNELS
from rl.network import TRUNK_CHANNELS, SCALAR_PLANES, FUSED_CHANNELS, _ResBlock128


class AWBWValueNet(nn.Module):
    """
    Value-only copy of the PPO board evaluator.

    This intentionally keeps the same spatial/scalar representation as the PPO
    trunk, while removing candidate-action tensors, policy logits, masks,
    log-probs, entropy, and rollout-buffer policy machinery.
    """

    def __init__(self, hidden_size: int = 256, trunk_blocks: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(N_SPATIAL_CHANNELS, TRUNK_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.trunk_blocks = nn.ModuleList([_ResBlock128() for _ in range(trunk_blocks)])
        self.scalar_to_plane = nn.Linear(N_SCALARS, SCALAR_PLANES)
        self.value_head = nn.Sequential(
            nn.Linear(FUSED_CHANNELS, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, spatial: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        # spatial: [B, 30, 30, C]
        with torch.amp.autocast(
            "cuda",
            dtype=torch.float16,
            enabled=(spatial.device.type == "cuda"),
        ):
            x = spatial.permute(0, 3, 1, 2).contiguous()
            x = self.stem(x)
            for blk in self.trunk_blocks:
                x = blk(x)

            b = scalars.shape[0]
            scalar_planes = self.scalar_to_plane(scalars).view(b, SCALAR_PLANES, 1, 1)
            scalar_planes = scalar_planes.expand(-1, -1, GRID_SIZE, GRID_SIZE)
            fused = torch.cat([x, scalar_planes], dim=1)
            pooled = torch.nn.functional.adaptive_avg_pool2d(fused, (1, 1)).flatten(1)

        return self.value_head(pooled.float()).squeeze(-1)


@torch.no_grad()
def evaluate_value_np(
    model: AWBWValueNet,
    spatial: np.ndarray,
    scalars: np.ndarray,
    *,
    device: str | torch.device = "cuda",
    return_logits: bool = False,
) -> float:
    model.eval()
    dev = torch.device(device)
    # Reuse pinned memory tensors when possible to avoid repeated host→device copies.
    # We cache them on the model object (one per device) so they survive across calls.
    cache = getattr(model, "_value_np_cache", {})
    if dev not in cache:
        cache[dev] = {
            "spatial": torch.zeros((1, GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS), dtype=torch.float32, device=dev),
            "scalars": torch.zeros((1, N_SCALARS), dtype=torch.float32, device=dev),
        }
        model._value_np_cache = cache
    buf = cache[dev]
    buf["spatial"].copy_(torch.as_tensor(spatial, dtype=torch.float32), non_blocking=True)
    buf["scalars"].copy_(torch.as_tensor(scalars, dtype=torch.float32), non_blocking=True)
    logit = float(model(buf["spatial"], buf["scalars"]).item())
    if return_logits:
        return logit
    win_prob = 1.0 / (1.0 + np.exp(-logit))
    return float(win_prob)


@torch.no_grad()
def evaluate_value_batch(
    model: AWBWValueNet,
    spatial_list: list[np.ndarray],
    scalars_list: list[np.ndarray],
    *,
    device: str | torch.device = "cuda",
    return_logits: bool = False,
) -> np.ndarray:
    """Batch evaluate value for multiple states at once.

    Args:
        model: AWBWValueNet model
        spatial_list: List of spatial arrays, each shape (30, 30, C)
        scalars_list: List of scalar arrays, each shape (N_SCALARS,)
        device: Device to run on
        return_logits: If True, return raw logits; else sigmoid probabilities

    Returns:
        numpy array of shape (batch_size,) with win probabilities or logits
    """
    if not spatial_list:
        return np.array([], dtype=np.float32)

    model.eval()
    dev = torch.device(device)
    batch_size = len(spatial_list)

    # Stack all inputs into batches
    spatial_batch = np.stack(spatial_list, axis=0)  # (B, 30, 30, C)
    scalars_batch = np.stack(scalars_list, axis=0)  # (B, N_SCALARS)

    # Cache batch tensors on model (reuse if batch size matches)
    cache = getattr(model, "_value_batch_cache", {})
    if dev not in cache or cache[dev]["batch_size"] < batch_size:
        cache[dev] = {
            "spatial": torch.zeros((batch_size, GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS), dtype=torch.float32, device=dev),
            "scalars": torch.zeros((batch_size, N_SCALARS), dtype=torch.float32, device=dev),
            "batch_size": batch_size,
        }
        model._value_batch_cache = cache
    buf = cache[dev]

    # Copy data to GPU (non-blocking)
    buf["spatial"][:batch_size].copy_(torch.as_tensor(spatial_batch, dtype=torch.float32), non_blocking=True)
    buf["scalars"][:batch_size].copy_(torch.as_tensor(scalars_batch, dtype=torch.float32), non_blocking=True)

    # Single forward pass for entire batch
    logits = model(buf["spatial"][:batch_size], buf["scalars"][:batch_size])  # (B,)

    # Sync once at the end
    logits_np = logits.float().cpu().numpy()

    if return_logits:
        return logits_np

    # Apply sigmoid to get win probabilities
    win_probs = 1.0 / (1.0 + np.exp(-logits_np))
    return win_probs.astype(np.float32)


def load_value_from_maskable_ppo_zip(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> AWBWValueNet:
    """
    Best-effort transplant from the current MaskablePPO checkpoint.

    Loads same-name, same-shape board-trunk/value tensors.
    Handles scalar_to_plane weight shape mismatch (16 vs 20 scalars).
    """
    import torch
    from zipfile import ZipFile
    from io import BytesIO

    dev = torch.device(device)
    checkpoint_path = Path(checkpoint_path)

    # Load policy.pth from the zip manually
    with ZipFile(checkpoint_path, "r") as zf:
        with zf.open("policy.pth") as f:
            policy_sd = torch.load(BytesIO(f.read()), map_location=dev, weights_only=False)

    # Also load optimizer.pth to avoid shape mismatch warnings (not needed for inference)
    # Build a fresh AWBWValueNet
    value = AWBWValueNet().to(dev)

    # Transplant compatible tensors
    new_sd = value.state_dict()
    transplanted: dict[str, torch.Tensor] = {}

    # Handle scalar_to_plane weight specially (16 vs 20 scalars)
    for dst_key in new_sd.keys():
        # Try different prefixes that might be in the PPO checkpoint
        found = False
        for prefix in ["", "features_extractor.", "policy.features_extractor."]:
            src_key = prefix + dst_key
            if src_key in policy_sd:
                src_tensor = policy_sd[src_key]
                dst_tensor = new_sd[dst_key]

                # Special handling for scalar_to_plane.weight
                if "scalar_to_plane.weight" in dst_key:
                    if src_tensor.shape == dst_tensor.shape:
                        transplanted[dst_key] = src_tensor.detach().clone()
                    else:
                        # Shape mismatch: checkpoint has 16 scalars, model wants 20
                        # Copy what we can, zero-fill the rest
                        print(f"[value_net] Resizing {src_key} from {src_tensor.shape} to {dst_tensor.shape}")
                        new_weight = torch.zeros_like(dst_tensor)
                        min_shape = min(src_tensor.shape[0], dst_tensor.shape[0])
                        new_weight[:min_shape, :min_shape] = src_tensor[:min_shape, :min_shape]
                        transplanted[dst_key] = new_weight
                    found = True
                    break
                elif src_tensor.shape == dst_tensor.shape:
                    transplanted[dst_key] = src_tensor.detach().clone()
                    found = True
                    break

    missing, unexpected = value.load_state_dict(transplanted, strict=False)

    print("[value_net] loaded", len(transplanted), "tensors from PPO checkpoint")
    print("[value_net] missing:", list(missing)[:20], "..." if len(missing) > 20 else "")
    print("[value_net] unexpected:", unexpected)

    return value


def load_value_from_pth(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> AWBWValueNet:
    """
    Load AWBWValueNet from a .pth file (e.g., from scalpel_checkpoint_zip_to_awbw_net_state).
    
    The .pth file should contain a dict with either 'state_dict' or 'model_state_dict' key.
    Supports checkpoints from train_rhea_value_parallel.py (model_state_dict) and other sources (state_dict).
    """
    dev = torch.device(device)
    value = AWBWValueNet().to(dev)
    
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    # Support both 'state_dict' and 'model_state_dict' keys
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain 'state_dict' or 'model_state_dict' key")
    
    # Handle scalar_to_plane weight shape mismatch
    # Old checkpoints may have 16 scalars, new model has 20
    if "scalar_to_plane.weight" in state_dict:
        ckpt_weight = state_dict["scalar_to_plane.weight"]
        model_weight = value.scalar_to_plane.weight
        if ckpt_weight.shape != model_weight.shape:
            print(f"[value_net] Resizing scalar_to_plane.weight from {ckpt_weight.shape} to {model_weight.shape}")
            # Create new weight tensor with correct shape
            new_weight = torch.zeros_like(model_weight)
            # Copy old values where they align
            min_rows = min(ckpt_weight.shape[0], model_weight.shape[0])
            min_cols = min(ckpt_weight.shape[1], model_weight.shape[1])
            new_weight[:min_rows, :min_cols] = ckpt_weight[:min_rows, :min_cols]
            state_dict["scalar_to_plane.weight"] = new_weight
    
    if "scalar_to_plane.bias" in state_dict:
        ckpt_bias = state_dict["scalar_to_plane.bias"]
        model_bias = value.scalar_to_plane.bias
        if ckpt_bias.shape != model_bias.shape:
            print(f"[value_net] Resizing scalar_to_plane.bias from {ckpt_bias.shape} to {model_bias.shape}")
            new_bias = torch.zeros_like(model_bias)
            min_size = min(ckpt_bias.shape[0], model_bias.shape[0])
            new_bias[:min_size] = ckpt_bias[:min_size]
            state_dict["scalar_to_plane.bias"] = new_bias
    
    missing, unexpected = value.load_state_dict(state_dict, strict=False)
    
    print(f"[value_net] loaded from {checkpoint_path}")
    print(f"[value_net] missing:", list(missing)[:20], "..." if len(missing) > 20 else "")
    print(f"[value_net] unexpected:", list(unexpected)[:20], "..." if len(unexpected) > 20 else "")
    
    return value


def load_value_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> AWBWValueNet:
    """
    Unified loader for AWBWValueNet checkpoints.
    
    Automatically detects format:
    - .zip files: uses load_value_from_maskable_ppo_zip
    - .pth files: uses load_value_from_pth
    """
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix.lower() == ".zip":
        return load_value_from_maskable_ppo_zip(checkpoint_path, device=device)
    elif checkpoint_path.suffix.lower() in [".pth", ".pt"]:
        return load_value_from_pth(checkpoint_path, device=device)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path.suffix}. Use .zip or .pth/.pt")