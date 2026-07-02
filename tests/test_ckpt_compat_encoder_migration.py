import io
import zipfile
from pathlib import Path

import torch

from rl.ckpt_compat import (
    checkpoint_needs_spatial_stem_patch,
    materialize_sb3_zip_with_spatial_compat,
    transplant_scalar_to_plane_weight_to_current,
    transplant_spatial_stem_to_current,
)
from rl.encoder import N_SCALARS, N_SPATIAL_CHANNELS


def test_spatial_transplant_70_to_current_zero_fills_new_planes() -> None:
    src = torch.arange(4 * 70 * 3 * 3, dtype=torch.float32).reshape(4, 70, 3, 3)
    got = transplant_spatial_stem_to_current(src)
    assert got.shape == (4, N_SPATIAL_CHANNELS, 3, 3)
    assert torch.equal(got[:, :70], src)
    assert torch.count_nonzero(got[:, 70:]) == 0


def test_scalar_transplant_deletes_tier_column() -> None:
    src = torch.arange(16 * 17, dtype=torch.float32).reshape(16, 17)
    got = transplant_scalar_to_plane_weight_to_current(src)
    # After removing tier column (index 12) from 17-col src, we get 16 cols.
    # Then expanding to N_SCALARS (20) with specific mapping:
    # [0:2] funds_me, funds_enemy ← src[0:2]
    # [2:4] power_bar (new) ← leave 0
    # [5:7] cop_me ← src[4:6] (old cop_active_me, scop_active_me)
    # [10:12] cop_en ← src[6:8] (old cop_active_en, scop_active_en)
    # [12] turn_norm ← src[8]
    # [13] my_turn ← src[9]
    # [14:16] co_id ← src[10:12]
    # [16] weather_rain ← src[13]
    # [17] weather_snow ← src[14]
    # [18] weather_segments ← src[15]
    # [19] income_share ← src[16]
    assert got.shape == (16, N_SCALARS)
    # Check copied columns
    assert torch.equal(got[:, 0:2], src[:, 0:2])           # funds
    assert torch.count_nonzero(got[:, 2:4]) == 0          # power_bar (new, left 0)
    assert torch.equal(got[:, 5:7], src[:, 4:6])           # cop_me
    assert torch.equal(got[:, 10:12], src[:, 6:8])        # cop_en
    assert torch.equal(got[:, 12], src[:, 8])                # turn_norm
    assert torch.equal(got[:, 13], src[:, 9])                # my_turn
    assert torch.equal(got[:, 14:16], src[:, 10:12])      # co_id
    assert torch.equal(got[:, 16], src[:, 13])               # weather_rain
    assert torch.equal(got[:, 17], src[:, 14])               # weather_snow
    assert torch.equal(got[:, 18], src[:, 15])               # weather_segments
    assert torch.equal(got[:, 19], src[:, 16])               # income_share
    # Columns 7:10 (cop_stars, scop_stars, power_bar_enemy) should be 0
    assert torch.count_nonzero(got[:, 7:10]) == 0


def test_materialize_zip_patches_policy_and_clears_optimizer_moments(tmp_path) -> None:
    ckpt = tmp_path / "legacy.zip"
    # Create the legacy checkpoint zip
    policy = {
        "features_extractor.stem.0.weight": torch.ones(8, 70, 3, 3),
        "features_extractor.scalar_to_plane.weight": torch.ones(16, 17),
        "features_extractor.scalar_to_plane.bias": torch.ones(16),
    }
    opt = {"state": {"unsafe": torch.ones(1)}}
    with zipfile.ZipFile(ckpt, "w", zipfile.ZIP_DEFLATED) as zf:
        buf = io.BytesIO()
        torch.save(policy, buf)
        zf.writestr("policy.pth", buf.getvalue())
        buf = io.BytesIO()
        torch.save(opt, buf)
        zf.writestr("policy.optimizer.pth", buf.getvalue())
        zf.writestr("data", "{}")

    # Now verify and patch
    assert checkpoint_needs_spatial_stem_patch(ckpt)
    patched, is_temp = materialize_sb3_zip_with_spatial_compat(ckpt)
    assert is_temp
    try:
        with zipfile.ZipFile(patched, "r") as zf:
            assert "policy.optimizer.pth" in zf.namelist()
            sd = torch.load(io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=False)
            od = torch.load(
                io.BytesIO(zf.read("policy.optimizer.pth")),
                map_location="cpu",
                weights_only=False,
            )
        assert sd["features_extractor.stem.0.weight"].shape[1] == N_SPATIAL_CHANNELS
        assert sd["features_extractor.scalar_to_plane.weight"].shape[1] == N_SCALARS
        assert torch.count_nonzero(sd["features_extractor.stem.0.weight"][:, 70:]) == 0
        assert od["state"] == {}
    finally:
        Path(patched).unlink(missing_ok=True)
