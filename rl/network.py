"""
CNN policy/value network for AWBW.

Input:
  spatial: (batch, GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS)
  scalars: (batch, N_SCALARS)
Output:
  policy_logits: (batch, ACTION_SPACE_SIZE)
  value:         (batch,)
"""
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from rl.encoder import N_SPATIAL_CHANNELS, N_SCALARS, GRID_SIZE

# Flat action space: non-BUILD actions use indices < 10_000; BUILD uses
# _BUILD_OFFSET + (r * 30 + c) * N_UNIT_TYPES + unit_type (see rl/env.py).
# Max BUILD index ≈ 10_000 + 899 * 27 + 26 = 34_299 → round up.
ACTION_SPACE_SIZE = 35_000


class AWBWFeaturesExtractor(BaseFeaturesExtractor):
    """
    SB3-compatible features extractor wrapping the AWBWNet CNN trunk.

    Replaces SB3's default CombinedExtractor on Dict observations.
    Accepts {'spatial': Box(30,30,62), 'scalars': Box(N_SCALARS,)} and outputs
    a flat (batch, features_dim) tensor consumed by the policy/value heads.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim=features_dim)

        self.stem = nn.Sequential(
            nn.Conv2d(N_SPATIAL_CHANNELS, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.res1 = _ResBlock(64, 64)
        self.res2 = _ResBlock(64, 128, downsample=True)
        self.res3 = _ResBlock(128, 128)

        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        cnn_out_dim = 128 * 8 * 8  # 8192

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + N_SCALARS, features_dim),
            nn.ReLU(inplace=True),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def forward(self, observations: dict) -> torch.Tensor:
        spatial = observations["spatial"]  # (batch, H, W, C)
        scalars = observations["scalars"]  # (batch, N_SCALARS)

        x = spatial.permute(0, 3, 1, 2).contiguous()
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)

        cnn_flat = x.reshape(x.shape[0], -1)
        combined = torch.cat([cnn_flat, scalars], dim=1)
        return self.fc(combined)

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
    Residual CNN trunk → scalar fusion → dual heads (policy + value).

    Architecture:
      - Conv stem: N_SPATIAL_CHANNELS → 64 channels
      - 3× residual blocks (64 → 128 → 128)
      - AdaptiveAvgPool to 8×8 → flatten → 8192-dim vector
      - Concat with scalar features → Linear → ReLU × 2 (hidden_size)
      - Policy head: Linear → ACTION_SPACE_SIZE logits
      - Value head:  Linear → 1 scalar
    """

    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()

        # ── CNN trunk ─────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(N_SPATIAL_CHANNELS, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.res1 = _ResBlock(64, 64)
        self.res2 = _ResBlock(64, 128, downsample=True)
        self.res3 = _ResBlock(128, 128)

        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        cnn_out_dim = 128 * 8 * 8  # 8192

        # ── Fusion MLP ────────────────────────────────────────────────────────
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + N_SCALARS, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
        )

        # ── Heads ─────────────────────────────────────────────────────────────
        self.policy_head = nn.Linear(hidden_size, ACTION_SPACE_SIZE)
        self.value_head = nn.Linear(hidden_size, 1)

        self._init_weights()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        spatial: torch.Tensor,
        scalars: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spatial:     (batch, H, W, C) — channels-last from encoder
            scalars:     (batch, N_SCALARS)
            action_mask: (batch, ACTION_SPACE_SIZE) bool; True = valid action.
                         Invalid positions are set to -inf before softmax.
        Returns:
            logits: (batch, ACTION_SPACE_SIZE)
            value:  (batch,)
        """
        # Permute to (batch, C, H, W) for Conv2d
        x = spatial.permute(0, 3, 1, 2).contiguous()

        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)

        cnn_flat = x.reshape(x.shape[0], -1)
        combined = torch.cat([cnn_flat, scalars], dim=1)
        features = self.fc(combined)

        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)

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
        """
        Sample (or evaluate) an action.

        Returns:
            action:   (batch,) int64
            log_prob: (batch,)
            entropy:  (batch,)
            value:    (batch,)
        """
        logits, value = self.forward(spatial, scalars, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Value head: smaller init for stable early training
        nn.init.orthogonal_(self.value_head.weight, gain=0.01)
        # Policy head: small init to start near-uniform
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)


class _ResBlock(nn.Module):
    """Basic residual block with optional channel-doubling downsampling."""

    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False) -> None:
        super().__init__()
        stride = 1  # spatial size preserved; we use AdaptiveAvgPool later
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        self.shortcut: nn.Module
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)
