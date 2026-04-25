import sys
import os
import numpy as np
import pytest

# Ensure repo root on path when running this file directly (pytest usually adds cwd)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from engine.action import _build_occupancy
from engine.unit import Unit, UnitType

# Minimal MapData class for testing
class MapData:
    def __init__(self, width, height, terrain, properties, unit_bans, unit_limit):
        self.width = width
        self.height = height
        self.terrain = terrain
        self.properties = properties
        self.unit_bans = unit_bans
        self.unit_limit = unit_limit

# Minimal GameState class for testing
class GameState:
    def __init__(self, map_data):
        self.map_data = map_data
        self.units = {0: [], 1: []}  # Initialize with empty units for players 0 and 1

# Test data
def create_test_state():
    map_data = MapData(
        width=5,
        height=5,
        terrain=np.array([[0]*5]*5),
        properties=[],
        unit_bans=[],
        unit_limit=50
    )
    state = GameState(map_data)

    # Add units for P0
    state.units[0] = [
        Unit(unit_id=1, unit_type=UnitType.INFANTRY, player=0, pos=(0,0), hp=100, ammo=0, fuel=99, moved=False, loaded_units=[], is_submerged=False, capture_progress=0),
        Unit(unit_id=2, unit_type=UnitType.TANK, player=0, pos=(0,1), hp=100, ammo=9, fuel=70, moved=False, loaded_units=[], is_submerged=False, capture_progress=0),
    ]
    
    # Add units for P1
    state.units[1] = [
        Unit(unit_id=3, unit_type=UnitType.INFANTRY, player=1, pos=(1,0), hp=100, ammo=0, fuel=99, moved=False, loaded_units=[], is_submerged=False, capture_progress=0),
        Unit(unit_id=4, unit_type=UnitType.TANK, player=1, pos=(1,1), hp=100, ammo=9, fuel=70, moved=False, loaded_units=[], is_submerged=False, capture_progress=0),
    ]
    
    return state

def test_occupancy_equivalence():
    """_build_occupancy is deterministic; two calls match (Cython-backed path)."""
    state = create_test_state()
    a = _build_occupancy(state)
    b = _build_occupancy(state)
    assert set(a.keys()) == set(b.keys())
    for pos in a:
        assert a[pos].unit_id == b[pos].unit_id, f"Unit mismatch at position {pos}"

# Run the test if executed directly
if __name__ == "__main__":
    pytest.main([__file__])