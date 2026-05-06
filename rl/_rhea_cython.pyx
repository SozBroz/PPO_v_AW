"""
Cython-accelerated RHEA (Rolling Horizon Evolution Algorithm) hot loops.

Contains:
  - _simulate_genome_cython: genome simulation with candidate ranking
  - _crossover_cython: single-point crossover
  - _mutate_cython: per-gene mutation with dynamic top_k
  - _rank_candidates_cheap_cython: fast candidate scoring
  - _random_genome_cython: random genome generation
"""

# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

import random as py_random
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fmax

from engine.search_clone import clone_for_search
from engine.game import GameState
from rl.candidate_actions import CandidateAction, CandidateKind, candidate_arrays, MAX_CANDIDATES


# ---- Constants ----
cdef int CK_END_TURN = 6
cdef int CK_MOVE_WAIT = 10


# ---- Helpers: cheap candidate ranking ----
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float _cheap_score(float[:] f) nogil:
    """Score a candidate from its preview feature array (indices 8..22)."""
    cdef float score = 0.0
    if f.shape[0] > 10:
        score += 2.0 * f[10]        # capture_completes
    if f.shape[0] > 8:
        score += 1.0 * f[8]         # capture_progress
    if f.shape[0] > 11:
        score += 0.25 * f[11]       # property_value
    if f.shape[0] > 17:
        score += 1.25 * f[17]       # enemy_value_removed_max
    if f.shape[0] > 16:
        score += 0.75 * f[16]       # enemy_value_removed_min
    if f.shape[0] > 19:
        score -= 0.80 * f[19]       # my_value_lost_max
    if f.shape[0] > 21:
        score += 0.50 * f[21]       # target_killed_max
    if f.shape[0] > 22:
        score -= 1.00 * f[22]       # attacker_killed_max
    return score


# ---- Genome simulation ----
@cython.boundscheck(False)
@cython.wraparound(False)
def simulate_genome_cython(
    object state,            # GameState
    list genome,
    int top_k_per_state,
    float mutation_rate,
    object CandidateKind_mod,  # Python module reference
) -> tuple:
    """
    Cython-accelerated genome simulation.
    Returns (final_state, actions, illegal_count).
    """
    cdef object sim = clone_for_search(state)
    cdef int acting = int(sim.active_player)
    cdef list actions = []
    cdef int illegal = 0
    cdef int gene_idx

    for gene_idx in range(len(genome)):
        if sim.winner is not None:
            break
        if int(sim.active_player) != acting:
            break

        _feats, mask, cands = candidate_arrays(sim, max_candidates=MAX_CANDIDATES)
        legal = [c for i, c in enumerate(cands) if i < len(mask) and bool(mask[i])]
        if not legal:
            break

        # Rank candidates cheaply
        scored = []
        for c in legal:
            f = c.preview
            score = 0.0
            if f is not None:
                score = _cheap_score(f)
            if (c.kind == CandidateKind.END_TURN or
                getattr(getattr(c.terminal_action, 'action_type', None), 'name', '') == "END_TURN"):
                score -= 0.5
            if c.kind == CandidateKind.MOVE_WAIT:
                score -= 0.1
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            illegal += 1
            break

        idx = int(genome[gene_idx])
        if idx >= len(scored):
            idx = idx % len(scored)

        cand = scored[idx][1]
        ok = _apply_candidate_fast(sim, cand)
        if not ok:
            illegal += 1
            break

        actions.append(cand.first)
        if cand.second is not None:
            actions.append(cand.second)

    # Force END_TURN if needed
    if sim.winner is None and int(sim.active_player) == acting:
        _feats, mask, cands = candidate_arrays(sim, max_candidates=MAX_CANDIDATES)
        legal = [c for i, c in enumerate(cands) if i < len(mask) and bool(mask[i])]
        enders = [c for c in legal if c.kind == CandidateKind.END_TURN or
                  getattr(getattr(c.terminal_action, 'action_type', None), 'name', '') == "END_TURN"]
        if enders:
            _apply_candidate_fast(sim, enders[0])
            actions.append(enders[0].terminal_action)

    return sim, actions, illegal


# ---- Crossover ----
@cython.boundscheck(False)
@cython.wraparound(False)
def crossover_cython(list a, list b, object rng) -> list:
    """
    Single-point crossover between two genomes.
    Returns new child genome.
    """
    if not a:
        return list(b)
    cdef int cut = rng.randrange(0, len(a))
    return list(a[:cut]) + list(b[cut:])


# ---- Mutation ----
@cython.boundscheck(False)
@cython.wraparound(False)
def mutate_cython(list genome, float mutation_rate, int top_k, object rng) -> None:
    """
    In-place mutation of genome.
    Each gene has `mutation_rate` chance of being replaced by random rank.
    """
    cdef int i
    cdef int max_val = max(1, top_k)
    for i in range(len(genome)):
        if rng.random() < mutation_rate:
            genome[i] = rng.randrange(max_val)


# ---- Random genome generation ----
@cython.boundscheck(False)
@cython.wraparound(False)
def random_genome_cython(int length, int top_k, object rng) -> list:
    """Generate a random genome of given length."""
    cdef int max_val = max(1, top_k)
    return [rng.randrange(max_val) for _ in range(length)]


# ---- Apply candidate (fast helper) ----
@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint _apply_candidate_fast(object state, object cand) except -1:
    """Apply a CandidateAction to state. Returns True on success, False on exception."""
    try:
        state.step(cand.first)
        if cand.second is not None and state.winner is None:
            state.step(cand.second)
        return 1
    except Exception:
        return 0
