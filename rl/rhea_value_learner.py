"""Turn-level TD learner for value-guided RHEA.

This is not PPO. There are no action logprobs, advantages, entropy bonuses, or
clipped policy losses. RHEA chooses full turns. The neural net learns to evaluate
turn-boundary board states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from rl.rhea_replay import RheaReplayBuffer, RheaTransition
from rl.value_net import AWBWValueNet


@dataclass(slots=True)
class RheaValueLearnerConfig:
    value_lr: float = 1.0e-4
    value_batch_size: int = 128
    replay_buffer_size: int = 50_000
    min_replay_before_train: int = 1_000
    updates_per_real_turn: int = 1
    gamma_turn: float = 0.99
    gamma_step: float = 0.99
    use_phi_step_targets: bool = False
    phi_scale: float = 0.05
    phi_loss_weight: float = 0.05
    use_dual_value_head: bool = False
    blend_win_on_turn_done: float = 0.0
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.0
    target_update_interval: int = 1_000
    target_tau: float | None = None
    target_clip: float | None = 5.0
    freeze_encoder: bool = True
    unfreeze_last_resblocks: int = 0


def bootstrap_gamma(cfg: RheaValueLearnerConfig) -> float:
    """Discount for TD bootstrap; step gamma only when phi-step targets are enabled."""
    if cfg.use_phi_step_targets:
        return float(cfg.gamma_step)
    return float(cfg.gamma_turn)


def compute_win_td_targets(
    *,
    next_logits: torch.Tensor,
    done: torch.Tensor,
    winner: torch.Tensor,
    acting_seat: torch.Tensor,
    gamma: float,
    target_clip: float | None,
    device: torch.device,
) -> torch.Tensor:
    """Win-probability TD targets (same formula as legacy train_one_batch)."""
    win_target = torch.where(
        winner == -1,
        torch.tensor(0.5, device=device).expand_as(winner),
        (winner == acting_seat).float(),
    )
    next_win_prob = torch.sigmoid(next_logits)
    immediate_win = win_target * done
    td_target_win = immediate_win + float(gamma) * next_win_prob * (1.0 - done)
    if target_clip is not None:
        td_target_win = torch.clamp(td_target_win, 0.0, 1.0)
    return td_target_win


def compute_rhea_value_loss(
    pred_logits: torch.Tensor,
    td_target_win: torch.Tensor,
    cfg: RheaValueLearnerConfig,
    *,
    phi_delta: torch.Tensor | None = None,
    turn_done: torch.Tensor | None = None,
) -> torch.Tensor:
    """Primary win-TD BCE; optional bounded phi auxiliary when enabled in cfg."""
    loss = F.binary_cross_entropy_with_logits(pred_logits, td_target_win)
    if not cfg.use_phi_step_targets or cfg.use_dual_value_head:
        return loss
    if phi_delta is None:
        return loss

    scale = max(float(cfg.phi_scale), 1e-8)
    aux_target = torch.clamp(0.5 + phi_delta / (2.0 * scale), 0.05, 0.95)
    blend = float(cfg.blend_win_on_turn_done)
    if blend > 0.0 and turn_done is not None:
        aux_target = torch.where(
            turn_done > 0,
            (1.0 - blend) * aux_target + blend * td_target_win,
            aux_target,
        )
    phi_loss = F.binary_cross_entropy_with_logits(pred_logits, aux_target)
    return loss + float(cfg.phi_loss_weight) * phi_loss


def transitions_to_batch_tensors(
    transitions: list[RheaTransition],
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Stack turn transitions for value loss / gradient push (mirrors replay sample keys)."""
    if not transitions:
        raise ValueError("transitions must be non-empty")
    dev = torch.device(device)
    return {
        "spatial_before": torch.as_tensor(
            np.stack([t.spatial_before for t in transitions]),
            dtype=torch.float32,
            device=dev,
        ),
        "scalars_before": torch.as_tensor(
            np.stack([t.scalars_before for t in transitions]),
            dtype=torch.float32,
            device=dev,
        ),
        "spatial_after": torch.as_tensor(
            np.stack([t.spatial_after for t in transitions]),
            dtype=torch.float32,
            device=dev,
        ),
        "scalars_after": torch.as_tensor(
            np.stack([t.scalars_after for t in transitions]),
            dtype=torch.float32,
            device=dev,
        ),
        "done": torch.as_tensor(
            [bool(t.done) for t in transitions],
            dtype=torch.float32,
            device=dev,
        ),
        "winner": torch.as_tensor(
            [int(t.winner) if t.winner is not None else -1 for t in transitions],
            dtype=torch.int64,
            device=dev,
        ),
        "acting_seat": torch.as_tensor(
            [int(t.acting_seat) for t in transitions],
            dtype=torch.int64,
            device=dev,
        ),
        "phi_delta": torch.as_tensor(
            [float(t.phi_delta) for t in transitions],
            dtype=torch.float32,
            device=dev,
        ),
    }


def compute_value_loss_from_batch(
    model: AWBWValueNet,
    batch: dict[str, torch.Tensor],
    cfg: RheaValueLearnerConfig,
    *,
    target_model: AWBWValueNet | None = None,
    detach_next: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared win-TD (+ optional phi aux) loss for learner and push-gradient actors."""
    device = batch["spatial_before"].device
    pred_logits = model(batch["spatial_before"], batch["scalars_before"])
    next_model = target_model if target_model is not None else model
    with torch.no_grad():
        next_logits = next_model(batch["spatial_after"], batch["scalars_after"])
        if detach_next:
            next_logits = next_logits.detach()
        td_target_win = compute_win_td_targets(
            next_logits=next_logits,
            done=batch["done"],
            winner=batch["winner"],
            acting_seat=batch["acting_seat"],
            gamma=bootstrap_gamma(cfg),
            target_clip=cfg.target_clip,
            device=device,
        )
    turn_done = torch.ones_like(batch["done"]) if not cfg.use_phi_step_targets else batch.get("turn_done")
    loss = compute_rhea_value_loss(
        pred_logits,
        td_target_win,
        cfg,
        phi_delta=batch.get("phi_delta"),
        turn_done=turn_done,
    )
    return loss, pred_logits, td_target_win


def aggregate_weighted_gradients(
    contributions: list[tuple[dict[str, torch.Tensor], float]],
) -> dict[str, torch.Tensor] | None:
    """Sum gradient dicts weighted by num_transitions; return None if empty."""
    if not contributions:
        return None
    total_weight = sum(float(w) for _, w in contributions)
    if total_weight <= 0.0:
        return None
    aggregated: dict[str, torch.Tensor] = {}
    for grads_dict, weight in contributions:
        w = float(weight)
        for name, grad_tensor in grads_dict.items():
            scaled = grad_tensor * w
            if name not in aggregated:
                aggregated[name] = scaled.clone()
            else:
                aggregated[name] += scaled
    for name in aggregated:
        aggregated[name] /= total_weight
    return aggregated


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

        tensor_batch = {
            "spatial_before": torch.as_tensor(batch["spatial_before"], dtype=torch.float32, device=self.device),
            "scalars_before": torch.as_tensor(batch["scalars_before"], dtype=torch.float32, device=self.device),
            "spatial_after": torch.as_tensor(batch["spatial_after"], dtype=torch.float32, device=self.device),
            "scalars_after": torch.as_tensor(batch["scalars_after"], dtype=torch.float32, device=self.device),
            "done": torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device),
            "winner": torch.as_tensor(batch["winner"], dtype=torch.int64, device=self.device),
            "acting_seat": torch.as_tensor(batch["acting_seat"], dtype=torch.int64, device=self.device),
            "phi_delta": torch.as_tensor(batch["phi_delta"], dtype=torch.float32, device=self.device),
        }

        loss, pred_logits, td_target_win = compute_value_loss_from_batch(
            self.online,
            tensor_batch,
            self.cfg,
            target_model=self.target,
            detach_next=True,
        )

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