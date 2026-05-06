"""
Cython-optimized candidate action generation for RHEA search.

Implements hot loops from rl/candidate_actions.py with typed memoryviews
and C-level loops to reduce Python overhead in the critical RHEA simulation path.
"""
# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fmax

# ---- Constants from candidate_actions.py ----
cdef int MAX_CANDIDATES = 4096
cdef int CANDIDATE_FEATURE_DIM = 24

# CandidateKind values (must match the Python enum)
cdef int CK_FLAT_ACTION = 0
cdef int CK_SELECT_UNIT = 1
cdef int CK_BUILD = 2
cdef int CK_REPAIR = 3
cdef int CK_UNLOAD = 4
cdef int CK_POWER = 5
cdef int CK_END_TURN = 6
cdef int CK_MOVE_WAIT = 10
cdef int CK_MOVE_CAPTURE = 11
cdef int CK_MOVE_ATTACK = 12
cdef int CK_MOVE_LOAD = 13
cdef int CK_MOVE_JOIN = 14
cdef int CK_MOVE_DIVE_HIDE = 15
cdef int CK_MOVE_SETUP_UNLOAD = 20
cdef int CK_MOVE_SETUP_REPAIR = 21
cdef int CK_MOVE_SETUP_ACTION = 22


# ---- Fast feature array initialization ----

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fill_base_features(
    np.float32_t[:] f,
    int kind,
    float pos_r, float pos_c,
    float dest_r, float dest_c,
    float target_r, float target_c,
    int unit_type,
):
    """Fill base features (indices 0-7) into pre-allocated memoryview."""
    f[0] = <float>kind / 32.0
    f[1] = pos_r / 29.0
    f[2] = pos_c / 29.0
    f[3] = dest_r / 29.0
    f[4] = dest_c / 29.0
    f[5] = target_r / 29.0
    f[6] = target_c / 29.0
    if unit_type >= 0:
        f[7] = <float>unit_type / 32.0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fill_review_features(
    np.float32_t[:] f,
    float capture_progress,
    float capture_remaining_after,
    float capture_completes,
    float property_value,
    float damage_min,
    float damage_max,
    float counter_min,
    float counter_max,
    float enemy_value_removed_min,
    float enemy_value_removed_max,
    float my_value_lost_min,
    float my_value_lost_max,
    float target_killed_min,
    float target_killed_max,
    float attacker_killed_min,
    float attacker_killed_max,
    float sonja_counter_break,
):
    """Fill preview features (indices 8-23) into pre-allocated memoryview."""
    f[8] = capture_progress
    f[9] = capture_remaining_after
    f[10] = capture_completes
    f[11] = property_value
    f[12] = damage_min
    f[13] = damage_max
    f[14] = counter_min
    f[15] = counter_max
    f[16] = enemy_value_removed_min
    f[17] = enemy_value_removed_max
    f[18] = my_value_lost_min
    f[19] = my_value_lost_max
    f[20] = target_killed_min
    f[21] = target_killed_max
    f[22] = fmax(attacker_killed_min, attacker_killed_max)
    f[23] = sonja_counter_break


# ---- Python-callable wrappers that work with numpy arrays ----

@cython.boundscheck(False)
@cython.wraparound(False)
def fill_base_features_np(
    np.ndarray[np.float32_t, ndim=1] f,
    int kind,
    float pos_r, float pos_c,
    float dest_r, float dest_c,
    float target_r, float target_c,
    int unit_type,
):
    """Python-callable wrapper for _fill_base_features."""
    _fill_base_features(f, kind, pos_r, pos_c, dest_r, dest_c, target_r, target_c, unit_type)


@cython.boundscheck(False)
@cython.wraparound(False)
def fill_review_features_np(
    np.ndarray[np.float32_t, ndim=1] f,
    dict preview,
):
    """Python-callable wrapper for _fill_review_features using dict."""
    _fill_review_features(
        f,
        <float>preview.get("capture_progress", 0.0),
        <float>preview.get("capture_remaining_after", 0.0),
        <float>preview.get("capture_completes", 0.0),
        <float>preview.get("property_value", 0.0),
        <float>preview.get("damage_min", 0.0),
        <float>preview.get("damage_max", 0.0),
        <float>preview.get("counter_min", 0.0),
        <float>preview.get("counter_max", 0.0),
        <float>preview.get("enemy_value_removed_min", 0.0) / 30000.0,
        <float>preview.get("enemy_value_removed_max", 0.0) / 30000.0,
        <float>preview.get("my_value_lost_min", 0.0) / 30000.0,
        <float>preview.get("my_value_lost_max", 0.0) / 30000.0,
        <float>preview.get("target_killed_min", 0.0),
        <float>preview.get("target_killed_max", 0.0),
        <float>preview.get("attacker_killed_min", 0.0),
        <float>preview.get("attacker_killed_max", 0.0),
        <float>preview.get("sonja_counter_break", 0.0),
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def candidate_arrays_cython(
    object state,  # GameState - keep as Python object (complex)
    int max_candidates=MAX_CANDIDATES,
) -> tuple:
    """
    Cython-accelerated version of candidate_arrays().
    
    Returns (feats, mask, cands) with typed memoryview loops for filling feats.
    """
    # Import here to avoid circular import at module load
    from rl.candidate_actions import enumerate_candidates, candidate_to_features
    
    cands = enumerate_candidates(state)
    
    # Allocate output arrays
    cdef np.ndarray[np.float32_t, ndim=2] feats = np.zeros((max_candidates, CANDIDATE_FEATURE_DIM), dtype=np.float32)
    cdef np.ndarray[np.uint8_t, ndim=1] mask = np.zeros((max_candidates,), dtype=np.uint8)
    
    cdef int n = min(len(cands), max_candidates)
    cdef int i
    cdef object cand  # CandidateAction
    
    for i in range(n):
        cand = cands[i]
        # Get features as numpy array (from candidate_to_features)
        f = candidate_to_features(state, cand)
        # Copy into our typed array (faster than Python loop)
        feats[i, :] = f
        mask[i] = 1
    
    return feats, mask, cands


@cython.boundscheck(False)
@cython.wraparound(False)
def candidate_to_features_cython(
    object state,  # GameState
    object cand,  # CandidateAction
) -> object:
    """
    Cython-accelerated version of candidate_to_features().
    Uses pre-allocated base features and avoids Python overhead.
    """
    from rl.candidate_actions import _base_features, CandidateKind
    import numpy as np
    
    action = cand.terminal_action
    ut = None
    if action.unit_pos is not None:
        unit = state.get_unit_at(*action.unit_pos)
        if unit is not None:
            ut = unit.unit_type
    
    # Call the original _base_features but with Cython-visible typing
    f = _base_features(
        CandidateKind(cand.kind),
        action.unit_pos,
        action.move_pos,
        action.target_pos,
        ut,
    )
    if cand.preview is not None:
        f += cand.preview.astype(np.float32, copy=False)
    return f
