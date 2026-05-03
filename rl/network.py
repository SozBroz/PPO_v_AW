"""
CNN policy/value network for AWBW.

Input:
  spatial: (batch, GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS)
  scalars: (batch, N_SCALARS)
Output:
  policy_logits: (batch, ACTION_SPACE_SIZE)
  value:         (batch,)

Flat action layout matches ``rl/env.py`` (scatter indices); constants duplicated here
to avoid importing ``rl.env`` (circular import risk).
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rl.candidate_actions import CANDIDATE_FEATURE_DIM, MAX_CANDIDATES
from rl.encoder import GRID_SIZE, N_SCALARS, N_SPATIAL_CHANNELS

# --- Flat index layout (must stay aligned with rl/env.py) ---------------------
ACTION_SPACE_SIZE = 35_000
_ENC_W = 30
_ATTACK_OFFSET = _ENC_W * _ENC_W
_CAPTURE_IDX = _ATTACK_OFFSET * 2
_WAIT_IDX = _CAPTURE_IDX + 1
_LOAD_IDX = _CAPTURE_IDX + 2
_JOIN_IDX = _CAPTURE_IDX + 3
_DIVE_HIDE_IDX = _CAPTURE_IDX + 4
_UNLOAD_OFFSET = _CAPTURE_IDX + 10
_MOVE_OFFSET = _UNLOAD_OFFSET + 8
_BUILD_OFFSET = 10_000
_REPAIR_OFFSET = 3500
_N_BUILD_UNIT_TYPES = 27

TRUNK_CHANNELS = 128
SCALAR_PLANES = 16
FUSED_CHANNELS = TRUNK_CHANNELS + SCALAR_PLANES  # 144


class AWBWFeaturesExtractor(BaseFeaturesExtractor):
    """
    SB3-compatible features extractor: same trunk + scalar fusion as ``AWBWNet``,
    then pools to a flat vector for SB3 policy/value MLPs.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim=features_dim)

        self.stem = nn.Sequential(
            nn.Conv2d(N_SPATIAL_CHANNELS, TRUNK_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.trunk_blocks = nn.ModuleList(
            [_ResBlock128() for _ in range(10)]
        )
        self.scalar_to_plane = nn.Linear(N_SCALARS, SCALAR_PLANES)

        self.fc = nn.Sequential(
            nn.Linear(FUSED_CHANNELS, features_dim),
            nn.ReLU(inplace=True),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def forward(self, observations: dict) -> torch.Tensor:
        spatial = observations["spatial"]
        scalars = observations["scalars"]
        # Tier 1b: mixed precision for 30-50% speedup on conv layers
        # Only wrap the CNN trunk - keep final FC in float32 to avoid dtype mismatch
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(spatial.device.type == "cuda")):
            x = spatial.permute(0, 3, 1, 2).contiguous()
            x = self.stem(x)
            for blk in self.trunk_blocks:
                x = blk(x)
            b = scalars.shape[0]
            sp = self.scalar_to_plane(scalars).view(b, SCALAR_PLANES, 1, 1)
            sp = sp.expand(-1, -1, GRID_SIZE, GRID_SIZE)
            xf = torch.cat([x, sp], dim=1)
            g = torch.nn.functional.adaptive_avg_pool2d(xf, (1, 1)).flatten(1)
        # FC layer expects float32 input (matching policy/value head expectations)
        g = g.float()
        return self.fc(g)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)


class AWBWCandidateFeaturesExtractor(BaseFeaturesExtractor):
    """SB3-compatible extractor for the padded candidate-action policy."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512) -> None:
        super().__init__(observation_space, features_dim=features_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(N_SPATIAL_CHANNELS, TRUNK_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.trunk_blocks = nn.ModuleList([_ResBlock128() for _ in range(10)])
        self.scalar_to_plane = nn.Linear(N_SCALARS, SCALAR_PLANES)
        self.candidate_mlp = nn.Sequential(
            nn.Linear(CANDIDATE_FEATURE_DIM, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(FUSED_CHANNELS + 64, features_dim),
            nn.ReLU(inplace=True),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def forward(self, observations: dict) -> torch.Tensor:
        spatial = observations["spatial"]
        scalars = observations["scalars"]
        cand = observations.get("candidate_features")
        cand_mask = observations.get("candidate_mask")
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(spatial.device.type == "cuda")):
            x = spatial.permute(0, 3, 1, 2).contiguous()
            x = self.stem(x)
            for blk in self.trunk_blocks:
                x = blk(x)
            b = scalars.shape[0]
            sp = self.scalar_to_plane(scalars).view(b, SCALAR_PLANES, 1, 1)
            sp = sp.expand(-1, -1, GRID_SIZE, GRID_SIZE)
            xf = torch.cat([x, sp], dim=1)
            board_g = torch.nn.functional.adaptive_avg_pool2d(xf, (1, 1)).flatten(1)

            if cand is None:
                cand_g = torch.zeros((b, 64), device=spatial.device, dtype=board_g.dtype)
            else:
                cand = cand.to(dtype=board_g.dtype)
                ce = self.candidate_mlp(cand)
                if cand_mask is not None:
                    m = cand_mask.to(device=ce.device, dtype=ce.dtype).unsqueeze(-1)
                    denom = torch.clamp(m.sum(dim=1), min=1.0)
                    cand_g = (ce * m).sum(dim=1) / denom
                else:
                    cand_g = ce.mean(dim=1)
            g = torch.cat([board_g, cand_g], dim=1)
        return self.fc(g.float())

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)


class AWBWNet(nn.Module):
    """
    Residual CNN trunk (10×128, full 30×30), scalar broadcast fusion → 144 ch,
    factored spatial policy head + pooled value head (MASTER_SPEC §4).
    """

    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.stem = nn.Sequential(
            nn.Conv2d(N_SPATIAL_CHANNELS, TRUNK_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.trunk_blocks = nn.ModuleList(
            [_ResBlock128() for _ in range(10)]
        )
        self.scalar_to_plane = nn.Linear(N_SCALARS, SCALAR_PLANES)

        self.conv_select = nn.Conv2d(FUSED_CHANNELS, 1, kernel_size=1)
        self.conv_move = nn.Conv2d(FUSED_CHANNELS, 1, kernel_size=1)
        self.conv_attack = nn.Conv2d(FUSED_CHANNELS, 1, kernel_size=1)
        self.conv_repair = nn.Conv2d(FUSED_CHANNELS, 1, kernel_size=1)
        self.conv_build = nn.Conv2d(FUSED_CHANNELS, _N_BUILD_UNIT_TYPES, kernel_size=1)

        self.linear_scalar_policy = nn.Linear(FUSED_CHANNELS, 16)

        self.candidate_mlp = nn.Sequential(
            nn.Linear(FUSED_CHANNELS + CANDIDATE_FEATURE_DIM, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(FUSED_CHANNELS, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
        )

        self._init_weights()

    def forward(
        self,
        spatial: torch.Tensor,
        scalars: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        candidate_features: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = spatial.permute(0, 3, 1, 2).contiguous()
        x = self.stem(x)
        for blk in self.trunk_blocks:
            x = blk(x)

        b = scalars.shape[0]
        device = spatial.device
        dtype = spatial.dtype
        sp = self.scalar_to_plane(scalars).view(b, SCALAR_PLANES, 1, 1)
        sp = sp.expand(-1, -1, GRID_SIZE, GRID_SIZE)
        xf = torch.cat([x, sp], dim=1)

        g = torch.nn.functional.adaptive_avg_pool2d(xf, (1, 1)).flatten(1)

        if candidate_features is not None:
            cf = candidate_features.float()
            if (
                cf.dim() != 3
                or cf.shape[1] != MAX_CANDIDATES
                or cf.shape[2] != CANDIDATE_FEATURE_DIM
            ):
                raise ValueError(
                    f"candidate_features must be [B,{MAX_CANDIDATES},{CANDIDATE_FEATURE_DIM}], "
                    f"got {tuple(cf.shape)}"
                )
            gg = g.float().unsqueeze(1).expand(-1, MAX_CANDIDATES, -1)
            logits = self.candidate_mlp(torch.cat([gg, cf], dim=-1)).squeeze(-1)
            if candidate_mask is not None:
                logits = logits.masked_fill(~candidate_mask.bool(), float("-inf"))
            value = self.value_head(g.float()).squeeze(-1)
            return logits, value

        l_sel = self.conv_select(xf).squeeze(1)
        l_move = self.conv_move(xf).squeeze(1)
        l_atk = self.conv_attack(xf).squeeze(1)
        l_rep = self.conv_repair(xf).squeeze(1)
        l_bld = self.conv_build(xf)

        s_all = self.linear_scalar_policy(g)
        s_pow = s_all[:, :3]
        s_misc = s_all[:, 3:8]
        s_unl = s_all[:, 8:16]

        logits = torch.full(
            (b, ACTION_SPACE_SIZE), float("-inf"), device=device, dtype=dtype
        )

        sel_flat = l_sel.view(b, -1)
        atk_flat = l_atk.view(b, -1)
        logits[:, 3:900] = sel_flat[:, 0:897]
        logits[:, 903:1800] = atk_flat[:, 3:900]
        logits[:, 900] = sel_flat[:, 897] + atk_flat[:, 0]
        logits[:, 901] = sel_flat[:, 898] + atk_flat[:, 1]
        logits[:, 902] = sel_flat[:, 899] + atk_flat[:, 2]

        logits[:, 0:3] = s_pow
        logits[:, _CAPTURE_IDX] = s_misc[:, 0]
        logits[:, _WAIT_IDX] = s_misc[:, 1]
        logits[:, _LOAD_IDX] = s_misc[:, 2]
        logits[:, _JOIN_IDX] = s_misc[:, 3]
        logits[:, _DIVE_HIDE_IDX] = s_misc[:, 4]
        logits[:, _UNLOAD_OFFSET : _UNLOAD_OFFSET + 8] = s_unl

        logits[:, _MOVE_OFFSET : _MOVE_OFFSET + _ENC_W * _ENC_W] = l_move.reshape(
            b, -1
        )

        logits[:, _REPAIR_OFFSET : _REPAIR_OFFSET + _ENC_W * _ENC_W] = l_rep.reshape(
            b, -1
        )

        build_flat = l_bld.permute(0, 2, 3, 1).reshape(
            b, _ENC_W * _ENC_W * _N_BUILD_UNIT_TYPES
        )
        logits[:, _BUILD_OFFSET : _BUILD_OFFSET + build_flat.shape[1]] = build_flat

        value = self.value_head(g).squeeze(-1)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))

        return logits, value

    def get_action_and_value(
        self,
        spatial: torch.Tensor,
        scalars: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(spatial, scalars, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.value_head[-1].bias)
        nn.init.orthogonal_(self.linear_scalar_policy.weight, gain=0.01)
        nn.init.zeros_(self.linear_scalar_policy.bias)
        nn.init.orthogonal_(self.candidate_mlp[-1].weight, gain=0.01)
        nn.init.zeros_(self.candidate_mlp[-1].bias)
        for head in (
            self.conv_select,
            self.conv_move,
            self.conv_attack,
            self.conv_repair,
            self.conv_build,
        ):
            nn.init.orthogonal_(head.weight, gain=0.01)
            if head.bias is not None:
                nn.init.zeros_(head.bias)


class _ResBlock128(nn.Module):
    """Residual block 128→128 with depthwise separable convolutions and GroupNorm."""

    def __init__(self) -> None:
        super().__init__()
        # Depthwise convolution
        self.depthwise1 = nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False)
        # Pointwise convolution
        self.pointwise1 = nn.Conv2d(128, 128, 1, bias=False)
        self.gn1 = nn.GroupNorm(8, 128)
        
        self.depthwise2 = nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False)
        self.pointwise2 = nn.Conv2d(128, 128, 1, bias=False)
        self.gn2 = nn.GroupNorm(8, 128)
        
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.gn1(self.pointwise1(self.depthwise1(x))))
        out = self.gn2(self.pointwise2(self.depthwise2(out)))
        return self.act(out + x)
