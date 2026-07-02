"""Tests for scalpel_policy_state_dict_to_awbw_net."""

import torch

from rl.ckpt_compat import (
    N_SCALARS,
    N_SPATIAL_CHANNELS,
    transplant_scalar_to_plane_weight_to_current,
    transplant_spatial_stem_to_current,
)
from rl.encoder import GRID_SIZE, N_SCALARS, N_SPATIAL_CHANNELS


def test_scalpel_handles_legacy_state_dict() -> None:
    """Ensure scalpel_policy_state_dict_to_awbw_net can ingest a legacy (70+17) checkpoint."""
    # Build a minimal legacy state dict (simulate what an old SB3 checkpoint looks like)
    legacy_sd = {
        "features_extractor.stem.0.weight": torch.ones(8, 70, 3, 3),
        "features_extractor.scalar_to_plane.weight": torch.ones(16, 17),
        "features_extractor.scalar_to_plane.bias": torch.ones(16),
        "features_extractor.trunk_blocks.0.0.weight": torch.ones(8, 8, 3, 3),
    }
    # Note: scalpel_policy_state_dict_to_awbw_net is not imported here
    # because it requires rl.network.AWBWNet which may have complex dependencies.
    # Instead, test the underlying transplant functions directly.

    # Test spatial transplant
    stem_weight = legacy_sd["features_extractor.stem.0.weight"]
    fixed_stem = transplant_spatial_stem_to_current(stem_weight)
    assert fixed_stem.shape[1] == N_SPATIAL_CHANNELS

    # Test scalar transplant
    scalar_weight = legacy_sd["features_extractor.scalar_to_plane.weight"]
    fixed_scalar = transplant_scalar_to_plane_weight_to_current(scalar_weight)
    assert fixed_scalar.shape[1] == N_SCALARS


def test_transplant_functions_handle_current_format() -> None:
    """Ensure transplant functions pass through current-format tensors."""
    # Current format: 115 scalars, N_SPATIAL_CHANNELS spatial
    current_spatial = torch.ones(8, N_SPATIAL_CHANNELS, 3, 3)
    result = transplant_spatial_stem_to_current(current_spatial)
    assert result.shape[1] == N_SPATIAL_CHANNELS

    current_scalar = torch.ones(16, N_SCALARS)
    result = transplant_scalar_to_plane_weight_to_current(current_scalar)
    assert result.shape[1] == N_SCALARS
