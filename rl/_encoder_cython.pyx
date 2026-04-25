"""
Cython-optimized tile processing kernels for observation encoding.

Implements hot loops from rl/encoder.py for performance-critical operations.
"""

# distutils: language = c++
# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False
# cython: cdivision = True

import numpy as np
cimport numpy as np
import cython
from libc.math cimport fmax

# Terrain category mapping
cdef dict TERRAIN_CATEGORIES = {
    "plain": 0,
    "mountain": 1,
    "wood": 2,
    "road": 3,
    "river": 4,
    "bridge": 5,
    "sea": 6,
    "shoal": 7,
    "reef": 8,
    "city": 9,
    "base": 10,
    "airport": 11,
    "port": 12,
    "hq": 13,
    "lab": 14
}

# Property type indices
cdef int PROP_TYPE_HQ_LAB = 4
cdef int PROP_TYPE_BASE = 1
cdef int PROP_TYPE_AIRPORT = 2
cdef int PROP_TYPE_PORT = 3
cdef int PROP_TYPE_CITY = 0


def fill_terrain_channels(
    np.ndarray[np.float32_t, ndim=3] terrain_block,
    np.ndarray[np.int32_t, ndim=2] tids,
    dict terrain_category_table,
    dict defense_norm_table
):
    """
    Fill terrain channels using precomputed lookup tables.
    
    Args:
        terrain_block: (H, W, N_TERRAIN_CHANNELS) output array
        tids: (H, W) array of terrain IDs
        terrain_category_table: Precomputed terrain_id -> category mapping
        defense_norm_table: Precomputed terrain_id -> defense value
    """
    cdef int H = tids.shape[0]
    cdef int W = tids.shape[1]
    cdef int r, c, tid, cat
    
    for r in range(H):
        for c in range(W):
            tid = tids[r, c]
            cat = terrain_category_table.get(tid, 0)
            terrain_block[r, c, cat] = 1.0


def encode_properties(
    np.ndarray[np.float32_t, ndim=3] spatial,
    object state,
    list properties,
    int observer,
    int prop_ch_offset,
    int cap_ch0,
    int cap_ch1,
    int neutral_inc_ch
):
    """
    Encode property ownership and capture progress.

    ``state.get_unit_at`` must match the Python path in ``rl.encoder`` (capture progress).
    """
    cdef int r, c, ptype, ownership
    cdef float prog
    cdef object occ

    for prop in properties:
        r, c = prop.row, prop.col
        
        # Skip out-of-bounds properties
        if r < 0 or r >= spatial.shape[0] or c < 0 or c >= spatial.shape[1]:
            continue
            
        # Determine property type
        if prop.is_hq or prop.is_lab:
            ptype = PROP_TYPE_HQ_LAB
        elif prop.is_base:
            ptype = PROP_TYPE_BASE
        elif prop.is_airport:
            ptype = PROP_TYPE_AIRPORT
        elif prop.is_port:
            ptype = PROP_TYPE_PORT
        else:
            ptype = PROP_TYPE_CITY
            
        # Determine ownership
        if prop.owner is None:
            ownership = 0
        elif prop.owner == observer:
            ownership = 1
        else:
            ownership = 2
            
        # Set property channel
        spatial[r, c, prop_ch_offset + ptype * 3 + ownership] = 1.0
        
        # Mark neutral income properties
        if prop.owner is None and not prop.is_comm_tower and not prop.is_lab:
            spatial[r, c, neutral_inc_ch] = 1.0
        
        # Handle capture progress (must mirror Python ``get_unit_at`` path).
        if prop.capture_points < 20:
            prog = (20 - prop.capture_points) / 20.0
            occ = state.get_unit_at(r, c)
            if occ is not None:
                if occ.player == observer:
                    spatial[r, c, cap_ch0] = fmax(spatial[r, c, cap_ch0], prog)
                elif occ.player != observer:
                    spatial[r, c, cap_ch1] = fmax(spatial[r, c, cap_ch1], prog)


def encode_units(
    np.ndarray[np.float32_t, ndim=3] spatial,
    list units_list,
    int observer,
    dict belief,
    int hp_lo_ch,
    int hp_hi_ch,
    int n_unit_types,
    int player_ch_offset,
):
    """
    Encode unit presence and HP belief channels.

    Args:
        spatial: (H, W, N_SPATIAL_CHANNELS) observation array
        units_list: List of Unit objects for a player (may be empty)
        observer: Seat index of the observer (0 or 1)
        belief: ``unit_id -> UnitBelief`` dict (from ``BeliefState`` in Python), or empty dict
        hp_lo_ch: Channel index for HP low bound
        hp_hi_ch: Channel index for HP high bound
        n_unit_types: Number of unit types (14)
        player_ch_offset: 0 for observer's corps, ``n_unit_types`` for opponent's block
    """
    cdef int r, c, ch
    cdef float hp_lo, hp_hi, hp_norm

    for unit in units_list:
        r, c = unit.pos
        
        # Skip out-of-bounds units
        if r < 0 or r >= spatial.shape[0] or c < 0 or c >= spatial.shape[1]:
            continue
            
        # Set unit channel
        ch = min(unit.unit_type.value, n_unit_types - 1) + player_ch_offset
        spatial[r, c, ch] = 1.0
        
        # Set HP channels
        if unit.player == observer or belief is None:
            hp_norm = unit.hp / 100.0
            hp_lo = hp_hi = hp_norm
        else:
            b = belief.get(unit.unit_id)
            if b is None:
                bucket = (unit.hp + 9) // 10
                hp_lo = max(0, bucket * 10 - 9) / 100.0
                hp_hi = max(0, bucket * 10) / 100.0
            else:
                hp_lo = b.hp_min / 100.0
                hp_hi = b.hp_max / 100.0
                
        spatial[r, c, hp_lo_ch] = hp_lo
        spatial[r, c, hp_hi_ch] = hp_hi