from __future__ import annotations

import copy
import os
from typing import Optional

from engine.game import GameState
from engine.unit import Unit
from engine.map_loader import PropertyState, MapData
from engine.co import COState
from engine.spirit_pressure import SpiritState


# Try to import Cython-optimized version
try:
    from . import _search_clone_cython
    SEARCH_CLONE_CYTHON_AVAILABLE = True
except ImportError:
    _search_clone_cython = None  # noqa: F401
    SEARCH_CLONE_CYTHON_AVAILABLE = False

USE_SEARCH_CLONE_CYTHON = (
    SEARCH_CLONE_CYTHON_AVAILABLE
    and os.environ.get("AWBW_USE_SEARCH_CLONE_CYTHON", "1") == "1"
)


def clone_for_search(state: GameState) -> GameState:
    """
    Selective clone for RHEA search — copies only what step() mutates.

    Skipped (not mutated during gameplay, or not needed for RHEA):
      - game_log, full_trace: set to empty lists (RHEA doesn't need history)
      - co_states._data: the CO JSON dict is never mutated after init
      - map_data.tiers, hq_positions, lab_positions, country_to_player: immutable
      - map_data.predeployed_specs, unit_bans: immutable
      - weather strings, tier_name: immutable
      - selected_unit: reference into the cloned units list (fixed up below)

    Deep-copied (mutated by step()):
      - units: Unit objects (hp, pos, moved, loaded_units, etc.)
      - properties: PropertyState (capture_points, owner)
      - map_data.terrain: seam breaks mutate the grid
      - co_states: power_bar, cop_active, scop_active, etc. (but share _data)
      - funds, gold_spent, losses_hp, losses_units: small lists
      - seam_hp: dict of remaining seam HP
      - capture_attempted_unit_ids: set
      - luck_rng: Random state
      - spirit: SpiritState
    """
    if USE_SEARCH_CLONE_CYTHON:
        return _search_clone_cython.fast_clone_for_search(state)

    # Pure Python fallback (original selective clone implementation)
    # Shallow-copy the dataclass (new object, same field references)
    sim = copy.copy(state)

    # --- Units: deep copy each Unit, including loaded units ---
    sim.units = {}
    for seat, units_list in state.units.items():
        new_list = []
        for u in units_list:
            new_u = copy.copy(u)  # Unit is small, copy.copy is sufficient
            # Deep-copy loaded_units (list of Unit)
            if u.loaded_units:
                new_u.loaded_units = []
                for lu in u.loaded_units:
                    new_lu = copy.copy(lu)
                    if lu.loaded_units:
                        # Nested transports (very rare) — recurse shallow
                        new_lu.loaded_units = [copy.copy(x) for x in lu.loaded_units]
                    new_u.loaded_units.append(new_lu)
            new_list.append(new_u)
        sim.units[seat] = new_list

    # Fix up selected_unit to point into the cloned units list
    if state.selected_unit is not None:
        su = state.selected_unit
        for u in sim.units.get(su.player, []):
            if u.unit_id == su.unit_id:
                sim.selected_unit = u
                break

    # --- Properties: copy each PropertyState ---
    sim.properties = [copy.copy(p) for p in state.properties]

    # --- CO states: copy each COState, but share the immutable _data dict ---
    sim.co_states = []
    for cs in state.co_states:
        new_cs = copy.copy(cs)
        new_cs._data = cs._data  # noqa: SLF001 — intentional sharing of immutable data
        sim.co_states.append(new_cs)

    # --- MapData: shallow copy, then deep-copy only the mutable terrain grid ---
    sim.map_data = copy.copy(state.map_data)
    sim.map_data.terrain = [row[:] for row in state.map_data.terrain]
    sim.map_data.properties = sim.properties
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

    # --- seam_hp dict (pos -> hp) ---
    sim.seam_hp = dict(state.seam_hp)

    # --- capture_attempted_unit_ids set ---
    sim.capture_attempted_unit_ids = set(state.capture_attempted_unit_ids)

    # --- RNG state ---
    sim.luck_rng = copy.copy(state.luck_rng)

    # --- SpiritState (if present) ---
    if state.spirit is not None:
        sim.spirit = copy.copy(state.spirit)
    else:
        sim.spirit = None

    # --- Not needed for RHEA search — skip to save work ---
    sim.game_log = []
    sim.full_trace = []

    # --- Optional oracle fields (typically None during RHEA) ---
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