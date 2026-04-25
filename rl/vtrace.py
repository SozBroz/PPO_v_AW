"""
V-trace targets for off-policy actor–critic (IMPALA).

Ported from RLlib's ``vtrace_torch.from_importance_weights`` (Apache-2.0),
trimmed to the scalar log-ρ API used by AWBW async training.
"""
from __future__ import annotations

from typing import NamedTuple

import torch


class VTraceReturns(NamedTuple):
    vs: torch.Tensor
    pg_advantages: torch.Tensor


def from_importance_weights(
    *,
    log_rhos: torch.Tensor,
    discounts: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
) -> VTraceReturns:
    """
    V-trace from log importance weights log(π/μ).

    Parameters
    ----------
    log_rhos:
        Shape ``(T, B)``. ``target_logp - behaviour_logp`` for the taken actions.
    discounts:
        Shape ``(T, B)``. Typically ``γ * (1 - done)``.
    rewards:
        Shape ``(T, B)``.
    values:
        Shape ``(T, B)``. ``V_θ(s_t)`` from the learner policy (may require grad).
    bootstrap_value:
        Shape ``(B,)``. ``V_θ(s_T)`` after the last transition; use zeros if the
        last state was terminal.
    clip_rho_threshold:
        ``ρ̄`` clip for critic trace (``None`` disables).
    clip_pg_rho_threshold:
        Clip for policy-gradient ρ term (``None`` disables).
    """
    if log_rhos.shape != values.shape:
        raise ValueError("log_rhos and values must share shape")
    if rewards.shape != values.shape or discounts.shape != values.shape:
        raise ValueError("rewards/discounts/values shape mismatch")
    if bootstrap_value.dim() != 1 or bootstrap_value.shape[0] != values.shape[1]:
        raise ValueError("bootstrap_value must be (B,) matching batch")

    rhos = torch.exp(log_rhos)
    if clip_rho_threshold is not None:
        clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
    else:
        clipped_rhos = rhos

    cs = torch.clamp(rhos, max=1.0)
    values_t_plus_1 = torch.cat([values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

    vs_minus_v_xs: list[torch.Tensor] = [torch.zeros_like(bootstrap_value)]
    for i in reversed(range(len(discounts))):
        discount_t, c_t, delta_t = discounts[i], cs[i], deltas[i]
        vs_minus_v_xs.append(delta_t + discount_t * c_t * vs_minus_v_xs[-1])
    vs_minus_v_xs_stacked = torch.stack(vs_minus_v_xs[1:])
    vs_minus_v_xs_stacked = torch.flip(vs_minus_v_xs_stacked, dims=[0])
    vs = vs_minus_v_xs_stacked + values

    vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
    if clip_pg_rho_threshold is not None:
        clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
    else:
        clipped_pg_rhos = rhos
    pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

    return VTraceReturns(vs=vs.detach(), pg_advantages=pg_advantages.detach())
