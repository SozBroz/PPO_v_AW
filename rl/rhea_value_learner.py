"""Turn-level TD learner for value-guided RHEA.

This is not PPO. There are no action logprobs, advantages, entropy bonuses, or
clipped policy losses. RHEA chooses full turns. The neural net learns to evaluate
turn-boundary board states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from rl.rhea_replay import RheaReplayBuffer
from rl.value_net import AWBWValueNet


@dataclass(slots=True)
class RheaValueLearnerConfig:
    value_lr: float = 1.0e-4
    value_batch_size: int = 128
    replay_buffer_size: int = 50_000
    min_replay_before_train: int = 1_000
    updates_per_real_turn: int = 1
    gamma_turn: float = 0.99
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.0
    target_update_interval: int = 1_000
    target_tau: float | None = None
    target_clip: float | None = 5.0
    freeze_encoder: bool = True
    unfreeze_last_resblocks: int = 0


def configure_trainable_params(model: AWBWValueNet, cfg: RheaValueLearnerConfig) -> None:
    """Apply the staged encoder-freezing schedule.

    Stage 1 should usually train only the value head. Later, unfreeze a few final
    residual blocks, then eventually the full trunk. This avoids blasting a PPO
    donor trunk with noisy early RHEA TD targets.
    """

    for p in model.parameters():
        p.requires_grad_(True)

    if not cfg.freeze_encoder:
        return

    # Freeze everything except the value head by default.
    for name, p in model.named_parameters():
        if not name.startswith("value_head"):
            p.requires_grad_(False)

    # Optionally unfreeze the last N trunk blocks as a middle stage.
    n = int(cfg.unfreeze_last_resblocks)
    if n > 0 and hasattr(model, "trunk_blocks"):
        blocks = list(model.trunk_blocks)
        for blk in blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad_(True)


class RheaValueLearner:
    def __init__(
        self,
        online: AWBWValueNet,
        target: AWBWValueNet,
        replay: RheaReplayBuffer,
        cfg: RheaValueLearnerConfig,
        *,
        device: str = "cuda",
    ) -> None:
        self.online = online.to(device)
        self.target = target.to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.replay = replay
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_updates = 0

        configure_trainable_params(self.online, cfg)
        params = [p for p in self.online.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("no trainable value parameters after freezing config")
        self.opt = torch.optim.AdamW(
            params,
            lr=cfg.value_lr,
            weight_decay=cfg.weight_decay,
        )

    def maybe_train_after_turn(self) -> list[dict[str, float]]:
        if len(self.replay) < self.cfg.min_replay_before_train:
            return []

        logs: list[dict[str, float]] = []
        for _ in range(max(0, int(self.cfg.updates_per_real_turn))):
            logs.append(self.train_one_batch())
        return logs

    def train_one_batch(self) -> dict[str, float]:
        batch = self.replay.sample(self.cfg.value_batch_size)

        spatial_before = torch.as_tensor(batch["spatial_before"], dtype=torch.float32, device=self.device)
        scalars_before = torch.as_tensor(batch["scalars_before"], dtype=torch.float32, device=self.device)
        spatial_after = torch.as_tensor(batch["spatial_after"], dtype=torch.float32, device=self.device)
        scalars_after = torch.as_tensor(batch["scalars_after"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)
        winner = torch.as_tensor(batch["winner"], dtype=torch.int64, device=self.device)
        acting_seat = torch.as_tensor(batch["acting_seat"], dtype=torch.int64, device=self.device)

        # Win prediction: predict P(win | state, acting_seat)
        # The value network outputs logits; use BCE with logits for numerical stability.
        pred_logits = self.online(spatial_before, scalars_before)

        with torch.no_grad():
            next_logits = self.target(spatial_after, scalars_after)

            # Game outcome: winner == acting_seat → 1.0 (win), else 0.0 (loss)
            # winner == -1 means draw → 0.5
            win_target = torch.where(
                winner == -1,
                torch.tensor(0.5, device=self.device).expand_as(winner),
                (winner == acting_seat).float(),
            )

            # For non-terminal states, blend with TD target
            # td_target_win = immediate_win + gamma * next_win_prob
            next_win_prob = torch.sigmoid(next_logits)
            immediate_win = win_target * done
            td_target_win = immediate_win + self.cfg.gamma_turn * next_win_prob * (1.0 - done)

            if self.cfg.target_clip is not None:
                c = float(self.cfg.target_clip)
                td_target_win = torch.clamp(td_target_win, 0.0, 1.0)

        # Binary cross-entropy with logits for numerical stability
        loss = F.binary_cross_entropy_with_logits(pred_logits, td_target_win)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.online.parameters() if p.requires_grad],
            self.cfg.gradient_clip_norm,
        )
        self.opt.step()

        self.num_updates += 1
        self._maybe_update_target()

        with torch.no_grad():
            pred_prob = torch.sigmoid(pred_logits)

        return {
            "value_loss": float(loss.detach().cpu().item()),
            "v_pred_mean": float(pred_prob.detach().mean().cpu().item()),
            "target_mean": float(td_target_win.detach().mean().cpu().item()),
            "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
            "num_updates": float(self.num_updates),
        }

    def _maybe_update_target(self) -> None:
        if self.cfg.target_tau is not None:
            tau = float(self.cfg.target_tau)
            with torch.no_grad():
                for tp, op in zip(self.target.parameters(), self.online.parameters()):
                    tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)
            return

        if self.cfg.target_update_interval > 0 and self.num_updates % self.cfg.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())