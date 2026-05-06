"""Patches for backward-compatible checkpoint loading (encoder/network migration)."""

from __future__ import annotations

import io
import zipfile
import tempfile
import torch
from pathlib import Path

from rl.encoder import N_SCALARS, N_SPATIAL_CHANNELS

_LEGACY_TIER_SCALAR_INDEX = 12  # where the legacy tier column sat


def transplant_spatial_stem_to_current(t: torch.Tensor) -> torch.Tensor:
    """Handle legacy 70-channel stem input: zero-fill new planes."""
    if t.dim() != 4:
        return t
    if t.shape[1] == N_SPATIAL_CHANNELS:
        return t
    if t.shape[1] == 70:
        out = t.new_zeros(t.shape[0], N_SPATIAL_CHANNELS, t.shape[2], t.shape[3])
        out[:, :70] = t
        return out
    return t


def transplant_scalar_to_plane_weight_to_current(t: torch.Tensor) -> torch.Tensor:
    """Handle legacy scalar projections: 17→20 (legacy tier) or 16→20 (power reform)."""
    if t.dim() != 2:
        return t
    if t.shape[1] == N_SCALARS:
        return t

    # Both legacy formats need to expand to N_SCALARS (20)
    if t.shape[1] == 17:
        # Remove legacy tier column at index 12
        t = torch.cat(
            [t[:, :_LEGACY_TIER_SCALAR_INDEX], t[:, _LEGACY_TIER_SCALAR_INDEX + 1:]],
            dim=1,
        ).contiguous()
        # t is now (batch, 16)

    # Now handle 16 → 20
    if t.shape[1] == 16:
        # Map old 16-column layout to new 20-column layout
        out = torch.zeros(t.shape[0], N_SCALARS, dtype=t.dtype, device=t.device)
        # [0:2] funds_me, funds_enemy → [0:2]
        out[:, 0:2] = t[:, 0:2]
        # [2:4] old power_norm → leave 0 (power bars are new)
        # [4:8] cop_active/scop me/en → [5:7] cop_me, [10:12] cop_en
        out[:, 5:7] = t[:, 4:6]      # cop_active_me, scop_active_me
        out[:, 10:12] = t[:, 6:8]    # cop_active_en, scop_active_en
        # [8] turn_norm → [12]
        out[:, 12] = t[:, 8]             # turn_norm
        # [9] my_turn → [13]
        out[:, 13] = t[:, 9]             # my_turn
        # [10:12] co_id_me, co_id_en → [14:16]
        out[:, 14:16] = t[:, 10:12]   # co_id_me, co_id_en
        # [12] weather_rain → [16]
        out[:, 16] = t[:, 12]            # weather_rain
        # [13] weather_snow → [17]
        out[:, 17] = t[:, 13]            # weather_snow
        # [14] weather_segments → [18]
        out[:, 18] = t[:, 14]            # weather_segments
        # [15] income_share → [19]
        out[:, 19] = t[:, 15]            # income_share
        return out

    # If we get here, shape is unrecognized — return as-is
    return t

    # If we get here, shape is unrecognized — return as-is
    return t


def _patch_policy_state_dict(sd: dict) -> None:
    """Patch ALL stem weights and scalar_to_plane weights in the state dict."""
    for k, v in list(sd.items()):
        if not isinstance(v, torch.Tensor):
            continue
        # Patch ANY stem.0.weight (features_extractor, pi_features_extractor, vf_features_extractor, etc.)
        if v.dim() == 4 and k.endswith("stem.0.weight"):
            sd[k] = transplant_spatial_stem_to_current(v)
            continue
        # Patch ANY scalar_to_plane.weight
        if v.dim() == 2 and k.endswith("scalar_to_plane.weight"):
            sd[k] = transplant_scalar_to_plane_weight_to_current(v)


def materialize_sb3_zip_with_spatial_compat(ckpt: str) -> tuple[str, bool]:
    """If checkpoint_needs_spatial_stem_patch(ckpt) is True, materialize a patched copy."""
    if not checkpoint_needs_spatial_stem_patch(ckpt):
        return ckpt, False

    # Create temp file with proper suffix
    import os
    tmp_dir = os.path.dirname(ckpt)
    with tempfile.NamedTemporaryFile(suffix=".zip", dir=tmp_dir, delete=False) as tmp:
        patched = tmp.name

    with zipfile.ZipFile(ckpt, "r") as zin, zipfile.ZipFile(patched, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            buf = zin.read(item.filename)
            if item.filename == "policy.pth":
                sd = torch.load(io.BytesIO(buf), map_location="cpu", weights_only=False)
                _patch_policy_state_dict(sd)
                out_buf = io.BytesIO()
                torch.save(sd, out_buf)
                zout.writestr(item.filename, out_buf.getvalue())
            elif item.filename == "policy.optimizer.pth":
                # Clear optimizer state (shape mismatch on MLP layers)
                od = {"state": {}}
                out_buf = io.BytesIO()
                torch.save(od, out_buf)
                zout.writestr(item.filename, out_buf.getvalue())
            else:
                zout.writestr(item.filename, buf)
    return patched, True


def scalpel_policy_state_dict_to_awbw_net(
    policy_sd: dict,
    *,
    hidden_size: int = 256,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """
    Build an :class:`rl.network.AWBWNet` state dict from a MaskablePPO ``policy.pth`` map.
    (``MultiInputPolicy`` + :class:`~rl.network.AWBWFeaturesExtractor` or
    :class:`~rl.network.AWBWCandidateFeaturesExtractor`).

    **Copied (when shapes match):**
    ``features_extractor.{stem,trunk_blocks,scalar_to_plane}`` → same paths on ``AWBWNet``;
    value input linear → ``value_head.0``: for the legacy extractor (14→``features_dim``),
    full ``fc.0`` when ``features_dim`` matches ``hidden_size``; for the candidate extractor
    (20→``features_dim``), rows ``[:hidden_size]`` and columns ``[:FUSED_CHANNELS]`` (board
    pooled features only; candidate slice omitted).

    **Left at fresh init:** all spatial policy conv heads (select/move/attack/…),
    ``linear_scalar_policy``, ``candidate_mlp``, and ``value_head.2`` (no compatible
    tensor in the SB3 critic head without retraining).

    ``policy_sd`` should already be encoder-patched (e.g. scalars 17→16, spatial 70→
    current) — use :func:`materialize_sb3_zip_with_spatial_compat` before loading bytes
    from disk, or :func:`_patch_policy_state_dict` on raw dicts.
    """
    # 1. Start from a fresh AWBWNet init (no old weights leak)
    from rl.network import AWBWNet
    net = AWBWNet(hidden_size=hidden_size)
    new_sd = {k: v for k, v in net.state_dict().items()}
    unused = []

    # 2. Port compatible tensors
    for k, v in list(policy_sd.items()):
        if not isinstance(v, torch.Tensor):
            continue
        if k not in new_sd:
            unused.append(k)
            continue
        if v.shape != new_sd[k].shape:
            if v.dim() == 4 and "features_extractor.stem" in k:
                policy_sd[k] = transplant_spatial_stem_to_current(v)
            elif v.dim() == 2 and "features_extractor.scalar_to_plane" in k:
                policy_sd[k] = transplant_scalar_to_plane_weight_to_current(v)
        if policy_sd[k].shape == new_sd[k].shape:
            new_sd[k] = policy_sd[k].float()
        else:
            unused.append(k)
            print(f"[scalpel] skip {k}: policy shape {v.shape} vs net shape {new_sd[k].shape}")

    # 3. Strip optimizer state (shape mismatch on MLP layers)
    return new_sd, unused


def checkpoint_needs_spatial_stem_patch(ckpt: str) -> bool:
    with zipfile.ZipFile(ckpt, "r") as zf:
        if "policy.pth" not in zf.namelist():
            return False
        buf = io.BytesIO(zf.read("policy.pth"))
        try:
            sd = torch.load(buf, map_location="cpu", weights_only=False)
        except Exception:
            return False
        for k, v in sd.items():
            if v.dim() == 4 and "features_extractor.stem" in k:
                if v.shape[1] != N_SPATIAL_CHANNELS:
                    return True
                break__
    return False
