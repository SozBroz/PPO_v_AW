# engine/_occupancy_cython.pyx
"""Cython implementation of occupancy grid construction."""

cimport numpy as np
import numpy as np
cimport cython
from libc.stdint cimport int32_t, int8_t

@cython.boundscheck(False)
@cython.wraparound(False)
def build_occupancy(
    np.ndarray[np.int32_t, ndim=2] unit_positions,
    int num_players
) -> dict:
    """
    Build occupancy dictionary {(row, col): player} from unit positions array.
    
    Args:
        unit_positions: Array of shape (N, 3) with columns [player_id, row, col]
        num_players: Number of players in the game
    
    Returns:
        dict: Mapping of (row, col) to player ID
    """
    cdef:
        Py_ssize_t i
        int32_t player_id, row, col
        dict occupancy = {}
    
    for i in range(unit_positions.shape[0]):
        player_id = unit_positions[i, 0]
        row = unit_positions[i, 1]
        col = unit_positions[i, 2]
        occupancy[(row, col)] = player_id
    
    return occupancy