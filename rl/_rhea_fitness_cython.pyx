"""
Cython-accelerated RHEA fitness scoring.

Hot paths:
  - phi_cython: bridge to env._compute_phi (state evaluation)
  - evaluate_value_fast: encode_state + value net forward pass
  - score_cython: combined phi_delta + value_delta computation
"""

# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython

from rl.encoder import GRID_SIZE, N_SCALARS, N_SPATIAL_CHANNELS, encode_state
from rl.value_net import evaluate_value_np, AWBWValueNet


# ---- Value evaluation (encode + forward) ----
@cython.boundscheck(False)
@cython.wraparound(False)
def evaluate_value_fast(
    object model,          # AWBWValueNet
    object state,          # GameState
    int observer_seat,
    str device="cuda",
) -> float:
    """
    Cython-accelerated value evaluation.
    Returns win probability [0.0, 1.0].
    """
    cdef np.float32_t[:, :, :] spatial = np.zeros(
        (GRID_SIZE, GRID_SIZE, N_SPATIAL_CHANNELS), dtype=np.float32
    )
    cdef np.float32_t[:] scalars = np.zeros((N_SCALARS,), dtype=np.float32)

    # Zero-fill without using .fill() on memoryview
    spatial[...] = 0.0
    scalars[...] = 0.0

    encode_state(
        state,
        observer=int(observer_seat),
        belief=None,
        out_spatial=spatial,
        out_scalars=scalars,
    )

    win_prob = evaluate_value_np(model, np.asarray(spatial), np.asarray(scalars), device=device)
    return float(win_prob)


# ---- Phi computation bridge ----
@cython.boundscheck(False)
@cython.wraparound(False)
def phi_cython(object env_template, object state, int observer_seat) -> float:
    """
    Cython-accelerated phi computation.
    Temporarily swaps env_template.state and computes phi.
    """
    cdef object old_state = env_template.state
    cdef int old_seat = env_template._learner_seat
    try:
        env_template.state = state
        env_template._learner_seat = int(observer_seat)
        phi_value = float(env_template._compute_phi(state))
        return phi_value
    finally:
        env_template.state = old_state
        env_template._learner_seat = old_seat


# ---- Combined score computation ----
@cython.boundscheck(False)
@cython.wraparound(False)
def score_cython(
    object fitness_obj,     # RheaFitness instance
    object before,           # GameState
    object after,            # GameState
    int observer_seat,
    int illegal_genes,
    list actions,            # list of actions (optional)
) -> tuple:
    """
    Cython-accelerated fitness scoring.
    Returns (RheaFitnessBreakdown, phi_delta, value_advantage).

    RheaFitnessBreakdown fields:
      phi_delta, value, illegal_penalty, total
    """
    cdef float phi_before, phi_after, phi_delta
    cdef float v_before, v_after, win_advantage
    cdef float illegal_penalty, build_punishment, total
    cdef float reward_weight = fitness_obj.reward_weight
    cdef float value_weight = fitness_obj.value_weight

    # Phi delta
    phi_before = fitness_obj.phi(before, observer_seat)
    phi_after = fitness_obj.phi(after, observer_seat)
    phi_delta = phi_after - phi_before

    # Value advantage (scaled to [-1, 1] to match phi_delta)
    v_before = fitness_obj.value(before, observer_seat)   # [0, 1]
    v_after = fitness_obj.value(after, observer_seat)       # [0, 1]
    win_advantage = (v_after - v_before) * 2.0

    # Penalties
    illegal_penalty = -fitness_obj.illegal_gene_penalty * <float>illegal_genes

    # Build punishment (simplified — just return 0.0 for now, full logic stays in Python)
    build_punishment = 0.0

    # Total
    total = reward_weight * phi_delta + value_weight * win_advantage + illegal_penalty + build_punishment

    # Return as a simple tuple (faster than constructing dataclass)
    return (
        phi_delta,
        win_advantage,
        illegal_penalty,
        total,
        phi_before,
        phi_after,
        v_before,
        v_after,
    )
