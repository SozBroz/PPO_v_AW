cimport numpy as np
import numpy as np
cimport cython
from libc.stdlib cimport malloc, free

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_reachable_costs_bfs(
    np.ndarray[np.int32_t, ndim=2] terrain,      # map terrain IDs
    np.ndarray[np.int8_t, ndim=2] occupancy,     # 0=empty, 1=P0, 2=P1
    np.ndarray[np.int32_t, ndim=2] move_costs,   # terrain_id → base cost
    int move_range,
    int fuel,
    int unit_player,
    int start_r,
    int start_c,
) -> dict:
    """Pure BFS with typed variables - no Python overhead in inner loop"""
    cdef:
        int H = terrain.shape[0]
        int W = terrain.shape[1]
        int dr, dc, nr, nc, new_fuel, cost
        int qcap
        int* queue_r
        int* queue_c
        int* queue_fuel
        int queue_start = 0
        int queue_end = 1
        int [:, :] min_fuel = np.full((H, W), 1000000, dtype=np.int32)
        dict results = {}
        int[4][2] directions = [[0,1], [1,0], [0,-1], [-1,0]]

    # Upper bound is small: relaxations re-queue a cell only on strict improvement.
    qcap = H * W * 8 + 16
    if qcap < 4096:
        qcap = 4096
    queue_r = <int*> malloc(qcap * sizeof(int))
    queue_c = <int*> malloc(qcap * sizeof(int))
    queue_fuel = <int*> malloc(qcap * sizeof(int))
    
    # Initialize queue with starting position
    queue_r[0] = start_r
    queue_c[0] = start_c
    queue_fuel[0] = 0
    min_fuel[start_r, start_c] = 0
    results[(start_r, start_c)] = 0
    
    # Declare variables outside loop
    cdef int r, c, current_fuel
    
    # Process queue
    while queue_start < queue_end:
        r = queue_r[queue_start]
        c = queue_c[queue_start]
        current_fuel = queue_fuel[queue_start]
        queue_start += 1
        
        for i in range(4):
            dr, dc = directions[i]
            nr = r + dr
            nc = c + dc
            
            # Skip out-of-bounds
            if nr < 0 or nc < 0 or nr >= H or nc >= W:
                continue
                
            # Skip impassable terrain or enemy units
            if move_costs[nr, nc] < 0 or \
               (occupancy[nr, nc] != 0 and occupancy[nr, nc] != unit_player + 1):
                continue
                
            # Calculate fuel cost for this move
            new_fuel = current_fuel + move_costs[nr, nc]
            
            # Skip if exceeds movement range or fuel capacity. Strict < matches
            # Python BFS: only re-visit a tile when we improve cost (avoids queue blow-up).
            if new_fuel < min_fuel[nr, nc] and \
               new_fuel <= move_range and \
               new_fuel <= fuel:
                
                min_fuel[nr, nc] = new_fuel
                results[(nr, nc)] = new_fuel
                
                if queue_end >= qcap:
                    free(queue_r)
                    free(queue_c)
                    free(queue_fuel)
                    raise MemoryError("compute_reachable_costs_bfs queue overflow")
                queue_r[queue_end] = nr
                queue_c[queue_end] = nc
                queue_fuel[queue_end] = new_fuel
                queue_end += 1
    
    # Free memory and return
    free(queue_r)
    free(queue_c)
    free(queue_fuel)
    return results