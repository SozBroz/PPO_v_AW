"""
Cython-accelerated GameState clone for RHEA search.

Implements fast selective clone with C-level loops for unit/property copying.
"""
# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

import copy
import random

cimport cython
from libc.stdlib cimport malloc, free

# ---- Unit copy helpers ---

@cython.boundscheck(False)
@cython.wraparound(False)
def copy_unit(object u):
    """Fast copy of a Unit dataclass."""
    new_u = copy.copy(u)
    # Deep-copy loaded_units if present
    if u.loaded_units:
        new_loaded = []
        for lu in u.loaded_units:
            new_lu = copy.copy(lu)
            if lu.loaded_units:
                # Nested transports (very rare)
                new_lu.loaded_units = [copy.copy(x) for x in lu.loaded_units]
            new_loaded.append(new_lu)
        new_u.loaded_units = new_loaded
    return new_u


@cython.boundscheck(False)
@cython.wraparound(False)
cdef dict copy_units_dict(dict units_dict):
    """Copy the units dict: {seat: [Unit, ...]}."""
    cdef dict new_dict = {}
    cdef object seat, units_list
    for seat, units_list in units_dict.items():
        new_dict[seat] = [copy_unit(u) for u in units_list]
    return new_dict


@cython.boundscheck(False)
@cython.wraparound(False)
cdef list copy_properties(list props):
    """Copy PropertyState list."""
    return [copy.copy(p) for p in props]


# ---- Main fast clone ----

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object fast_clone_for_search(object state):
    """
    Cython-accelerated selective clone for RHEA search.
    
    This is the Cython version of clone_for_search() in search_clone.py.
    Uses C-level loops and avoids Python overhead where possible.
    """
    from engine.game import GameState
    from engine.map_loader import MapData, PropertyState
    from engine.co import COState
    from engine.spirit_pressure import SpiritState
    
    cdef object sim = copy.copy(state)
    cdef object seat, units_list, u, cs, su
    cdef object terrain, new_terrain, row
    cdef int r, H
    
    # --- Units: deep copy each Unit ---
    sim.units = copy_units_dict(state.units)
    
    # Fix up selected_unit to point into the cloned units list
    if state.selected_unit is not None:
        su = state.selected_unit
        for u in sim.units.get(su.player, []):
            if u.unit_id == su.unit_id:
                sim.selected_unit = u
                break
    
    # --- Properties: copy each PropertyState ---
    sim.properties = copy_properties(state.properties)
    
    # --- CO states: copy each COState, share immutable _data ---
    sim.co_states = []
    for cs in state.co_states:
        new_cs = copy.copy(cs)
        new_cs._data = cs._data  # noqa: SLF001 — intentional sharing
        sim.co_states.append(new_cs)
    
    # --- MapData: shallow copy, then deep-copy terrain grid ---
    sim.map_data = copy.copy(state.map_data)
    # terrain is list[list[int]] — copy each row
    terrain = state.map_data.terrain
    H = len(terrain)
    new_terrain = []
    for r in range(H):
        new_terrain.append(terrain[r][:])
    sim.map_data.terrain = new_terrain
    # properties reference — point to our cloned properties
    sim.map_data.properties = sim.properties
    # Share immutable fields
    sim.map_data.tiers = state.map_data.tiers
    sim.map_data.hq_positions = state.map_data.hq_positions
    sim.map_data.lab_positions = state.map_data.lab_positions
    sim.map_data.country_to_player = state.map_data.country_to_player
    sim.map_data.predeployed_specs = state.map_data.predeployed_specs
    sim.map_data.unit_bans = state.map_data.unit_bans
    
    # --- Small mutable lists ---
    sim.funds = list(state.funds)
    sim.gold_spent = list(state.gold_spent)
    sim.losses_hp = list(state.losses_hp)
    sim.losses_units = list(state.losses_units)
    
    # --- seam_hp dict ---
    sim.seam_hp = dict(state.seam_hp)
    
    # --- capture_attempted_unit_ids set ---
    sim.capture_attempted_unit_ids = set(state.capture_attempted_unit_ids)
    
    # --- RNG state ---
    sim.luck_rng = copy.copy(state.luck_rng)
    
    # --- SpiritState ---
    if state.spirit is not None:
        sim.spirit = copy.copy(state.spirit)
    else:
        sim.spirit = None
    
    # --- Not needed for RHEA search ---
    sim.game_log = []
    sim.full_trace = []
    
    # --- Oracle fields ---
    sim._oracle_combat_damage_override = state._oracle_combat_damage_override
    if state._oracle_power_aoe_positions is not None:
        if isinstance(state._oracle_power_aoe_positions, set):
            sim._oracle_power_aoe_positions = set(state._oracle_power_aoe_positions)
        else:
            from collections import Counter
            sim._oracle_power_aoe_positions = Counter(state._oracle_power_aoe_positions)
    else:
        sim._oracle_power_aoe_positions = None
    
    return sim
