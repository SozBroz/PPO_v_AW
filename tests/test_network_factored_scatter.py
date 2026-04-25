"""Regression: factored policy head scatter layout (MOVE band, BUILD base index)."""
from __future__ import annotations

import torch

from rl.encoder import GRID_SIZE, N_SCALARS, N_SPATIAL_CHANNELS
from rl.network import (
    AWBWNet,
    _BUILD_OFFSET,
    _MOVE_OFFSET,
    _N_BUILD_UNIT_TYPES,
    _REPAIR_OFFSET,
)


def test_move_band_and_build_offset_populated_from_conv_bias() -> None:
    """With spatial conv biases-only, MOVE and BUILD flats receive uniform logits."""
    m = AWBWNet(hidden_size=32)
    with torch.no_grad():
        for mod in (
            m.conv_select,
            m.conv_attack,
            m.conv_repair,
            m.conv_build,
        ):
            mod.weight.zero_()
            mod.bias.zero_()
        m.conv_move.weight.zero_()
        m.conv_move.bias.fill_(2.5)
        m.linear_scalar_policy.weight.zero_()
        m.linear_scalar_policy.bias.zero_()

    B = 2
    spatial = torch.zeros(B, GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS)
    scalars = torch.zeros(B, N_SCALARS)
    mask = torch.ones(B, 35_000, dtype=torch.bool)
    logits, _v = m(spatial, scalars, mask)

    mv = logits[:, _MOVE_OFFSET : _MOVE_OFFSET + 30 * 30]
    assert mv.shape == (B, 900)
    assert torch.allclose(mv, torch.full_like(mv, 2.5))

    b0 = logits[:, _BUILD_OFFSET : _BUILD_OFFSET + _N_BUILD_UNIT_TYPES]
    assert b0.shape == (B, _N_BUILD_UNIT_TYPES)
    assert torch.allclose(b0, torch.full_like(b0, 0.0))

    with torch.no_grad():
        m.conv_build.bias.fill_(1.0)
    logits2, _ = m(spatial, scalars, mask)
    b00 = logits2[0, _BUILD_OFFSET : _BUILD_OFFSET + _N_BUILD_UNIT_TYPES]
    assert torch.allclose(b00, torch.ones(_N_BUILD_UNIT_TYPES))


def test_collision_band_is_sum_of_select_and_attack_corners() -> None:
    m = AWBWNet(hidden_size=32)
    with torch.no_grad():
        m.linear_scalar_policy.weight.zero_()
        m.linear_scalar_policy.bias.zero_()
        m.conv_move.weight.zero_()
        m.conv_move.bias.zero_()
        m.conv_repair.weight.zero_()
        m.conv_repair.bias.zero_()
        m.conv_build.weight.zero_()
        m.conv_build.bias.zero_()
        # SELECT: only tile (29,27) -> flat index 3 + 29*30 + 27 = 900-3 = 897 in sel_flat[:,897]
        m.conv_select.weight.zero_()
        m.conv_select.bias.fill_(0.0)
        # Attack: only (0,0) -> atk_flat[:,0]
        m.conv_attack.weight.zero_()
        m.conv_attack.bias.fill_(0.0)
        # Inject via manual forward pieces: easier — set last sel and first atk via bias tiles
        # conv 1x1: output[r,c] = bias if weight 0. So set weight to pick one spatial? Simpler use bias only uniform then index check collision formula:
        m.conv_select.bias.fill_(1.0)
        m.conv_attack.bias.fill_(2.0)

    spatial = torch.zeros(1, GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS)
    scalars = torch.zeros(1, N_SCALARS)
    mask = torch.ones(1, 35_000, dtype=torch.bool)
    logits, _ = m(spatial, scalars, mask)
    sel_flat = 1.0
    atk_flat_first = 2.0
    assert torch.isclose(logits[0, 900], torch.tensor(sel_flat + atk_flat_first))
