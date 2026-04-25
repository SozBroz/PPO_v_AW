"""Sanity checks for rl.vtrace (IMPALA off-policy correction)."""
from __future__ import annotations

import torch

from rl.vtrace import from_importance_weights


def test_vtrace_shapes_and_finite():
    t, b = 5, 8
    log_rhos = torch.zeros(t, b)
    discounts = torch.full((t, b), 0.99)
    rewards = torch.randn(t, b) * 0.01
    values = torch.randn(t, b) * 0.01
    bootstrap = torch.zeros(b)
    out = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap,
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0,
    )
    assert out.vs.shape == (t, b)
    assert out.pg_advantages.shape == (t, b)
    assert torch.isfinite(out.vs).all()
    assert torch.isfinite(out.pg_advantages).all()


def test_vtrace_importance_ratio_affects_pg_advantage():
    t, b = 2, 4
    discounts = torch.ones(t, b) * 0.99
    rewards = torch.randn(t, b) * 0.1
    values = torch.randn(t, b) * 0.1
    bootstrap = torch.zeros(b)
    out0 = from_importance_weights(
        log_rhos=torch.zeros(t, b),
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap,
    )
    out1 = from_importance_weights(
        log_rhos=torch.full((t, b), 5.0),
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap,
        clip_rho_threshold=100.0,
        clip_pg_rho_threshold=100.0,
    )
    assert not torch.allclose(out0.pg_advantages, out1.pg_advantages)
