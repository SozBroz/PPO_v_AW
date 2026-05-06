"""
Cython-accelerated tactical beam search hot loops.

Contains:
  - _beam_expand_cython: inner loop of search() — clone, apply, score
  - _juicy_score_cython: typed scoring with direct feature-array access
  - _bucket_for_candidate_cython: fast bucket assignment
  - _dedupe_cython: set-based deduplication
  - _dynamic_budget_cython: budget computation
"""

# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, fmax

# ---- Constants ----
cdef int MAX_CANDIDATES = 4096
cdef int CANDIDATE_FEATURE_DIM = 24

# CandidateKind enum values (must match Python enum)
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

# ---- Bucket constants ----
cdef bytes BUCKET_FINISH_CAPTURE = b"finish_capture"
cdef bytes BUCKET_START_CAPTURE = b"start_capture"
cdef bytes BUCKET_KILLSHOT = b"killshot"
cdef bytes BUCKET_STRIKE = b"strike"
cdef bytes BUCKET_BUILD = b"build"
cdef bytes BUCKET_POWER = b"power"
cdef bytes BUCKET_POSITION = b"position"
cdef bytes BUCKET_OTHER = b"other"


# ---- Utility: fast feature access ----
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float _feat_at(np.float32_t[:] f, int i) nogil:
    return f[i] if i < f.shape[0] else 0.0


# ---- Bucket assignment (hot path) ----
@cython.boundscheck(False)
@cython.wraparound(False)
def bucket_for_candidate_cython(object cand, object CandidateKind) -> str:
    """
    Cython-accelerated bucket assignment for a CandidateAction.
    Returns one of: finish_capture, start_capture, killshot, strike, build, power, position, other.
    """
    cdef object kind = cand.kind
    cdef object f = cand.preview
    cdef object terminal_action = cand.terminal_action
    cdef str terminal_name = ""
    cdef float capture_progress, capture_completes, target_killed_max, enemy_removed_max

    # POWER check (highest priority)
    if kind.value == CK_POWER:
        return "power"

    # Build check
    if terminal_action is not None:
        action_type = getattr(terminal_action, 'action_type', None)
        if action_type is not None:
            terminal_name = getattr(action_type, 'name', '')
        if not terminal_name:
            terminal_name = getattr(terminal_action, 'name', '')

    if terminal_name == "BUILD" or kind.value == CK_BUILD:
        return "build"

    # Use preview features for capture/attack decisions
    if f is not None:
        capture_progress = _feat_at(f, 8)
        capture_completes = _feat_at(f, 10)
        target_killed_max = _feat_at(f, 21)
        enemy_removed_max = _feat_at(f, 17)

        if capture_completes > 0.0:
            return "finish_capture"
        if capture_progress > 0.0:
            return "start_capture"
        if target_killed_max > 0.0:
            return "killshot"
        if enemy_removed_max > 0.0:
            return "strike"

    if terminal_name == "CAPTURE":
        return "start_capture"
    if terminal_name == "ATTACK":
        return "strike"

    if kind.value == CK_MOVE_WAIT:
        return "position"

    return "other"


# ---- Juicy score (hot path) ----
@cython.boundscheck(False)
@cython.wraparound(False)
def juicy_score_cython(object cand, str bucket) -> float:
    """
    Cython-accelerated scoring for bucket-sorted candidate selection.
    Returns a float score (higher = more desirable).
    """
    if cand is None or cand.preview is None:
        if bucket in {"build", "position", "power"}:
            return 1.0
        return 0.0

    cdef object f = cand.preview
    cdef float capture_completes, capture_progress, enemy_removed_max
    cdef float enemy_removed_min, target_killed_max, target_killed_min
    cdef float attacker_killed_max, attacker_killed_min, my_value_lost_max, my_value_lost_min
    cdef float damage_min, damage_max, counter_min, counter_max

    capture_completes = _feat_at(f, 10)
    capture_progress = _feat_at(f, 8)
    enemy_removed_max = _feat_at(f, 17)
    enemy_removed_min = _feat_at(f, 16)
    target_killed_max = _feat_at(f, 21)
    target_killed_min = _feat_at(f, 20)
    attacker_killed_max = fmax(_feat_at(f, 19), _feat_at(f, 22))
    attacker_killed_min = fmax(_feat_at(f, 18), _feat_at(f, 21))
    my_value_lost_max = _feat_at(f, 19)
    my_value_lost_min = _feat_at(f, 18)
    damage_min = _feat_at(f, 12)
    damage_max = _feat_at(f, 13)
    counter_min = _feat_at(f, 14)
    counter_max = _feat_at(f, 15)

    if bucket == "finish_capture":
        return 10.0 * capture_completes + 2.0 * capture_progress + _feat_at(f, 8)
    if bucket == "start_capture":
        return 4.0 * capture_progress + 1.5 * capture_completes
    if bucket == "killshot":
        return 6.0 * target_killed_max + enemy_removed_max - 0.75 * attacker_killed_max - 2.0 * attacker_killed_min
    if bucket == "strike":
        return enemy_removed_max + 0.5 * enemy_removed_min - 0.75 * attacker_killed_max - attacker_killed_min
    if bucket == "build":
        return 2.0
    if bucket == "power":
        return 8.0
    if bucket == "position":
        return 0.25
    return 0.0


# ---- Dynamic budget computation ----
@cython.boundscheck(False)
@cython.wraparound(False)
def dynamic_budget_cython(
    int owned_units,
    int juicy,  # Total number of juicy candidates (len(initial))
    dict bucket_counts,  # dict[str, int]
    bint cop_ready,
    bint scop_ready,
    int min_width,
    int max_width,
    int min_depth,
    int max_depth,
    int min_expand,
    int max_expand,
) -> dict:
    """
    Cython-accelerated dynamic budget computation.
    Returns dict with keys: width, depth, expand.
    """
    cdef float complexity = (
        0.40 * owned_units
        + 1.20 * bucket_counts.get("finish_capture", 0)
        + 1.00 * bucket_counts.get("start_capture", 0)
        + 1.25 * bucket_counts.get("killshot", 0)
        + 0.90 * bucket_counts.get("strike", 0)
        + 0.60 * bucket_counts.get("build", 0)
        + 0.80 * bucket_counts.get("power", 0)
        + 0.50 * sum(1 for v in bucket_counts.values() if v > 0)
    )

    if cop_ready or scop_ready:
        complexity += 3.0

    cdef int width = <int>round(min_width + 1.25 * complexity)
    cdef int depth = <int>round(min_depth + sqrt(fmax(0.0, complexity)) / 1.35)
    # Match Python logic: base 8 + 0.60 * juicy + 0.25 * owned_units
    cdef int expand = <int>round(8 + 0.60 * juicy + 0.25 * owned_units)

    return {
        "width": min(max_width, max(min_width, width)),
        "depth": min(max_depth, max(min_depth, depth)),
        "expand": min(max_expand, max(min_expand, expand)),
    }


# ---- Bucket counts ----
@cython.boundscheck(False)
@cython.wraparound(False)
def bucket_counts_cython(object cands, object planner) -> dict:
    """
    Count candidates per bucket.
    cands: list of CandidateAction
    planner: TacticalBeamPlanner instance (for _bucket_for_candidate)
    """
    cdef dict out = {}
    cdef str b
    for c in cands:
        b = planner._bucket_for_candidate(c)
        out[b] = out.get(b, 0) + 1
    return out


# ---- Beam deduplication ----
@cython.boundscheck(False)
@cython.wraparound(False)
def dedupe_cython(list lines, bint dedupe_plans, bint dedupe_states, object planner=None) -> list:
    """
    Deduplicate beam lines.
    lines: list of BeamLine
    Returns deduplicated list.
    Note: planner parameter is accepted but dedup key computation
    remains at Python level since it calls Python methods.
    """
    out = []
    seen = set()
    cdef object line, key_parts, key

    for line in lines:
        key_parts = []
        if dedupe_plans and planner is not None:
            # Python-level call (not Cython-optimizable)
            key_parts.append(tuple(planner._action_signature(a) for a in line.actions))
        if dedupe_states and planner is not None:
            key_parts.append(planner._state_signature(line.state))
        key = tuple(key_parts)
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


# ---- Beam sort key ----
@cython.boundscheck(False)
@cython.wraparound(False)
def beam_sort_key_cython(object line) -> tuple:
    """Return (score, tactical_count, len(buckets_seen)) for sorting."""
    return (line.score, line.tactical_count, len(line.buckets_seen))
