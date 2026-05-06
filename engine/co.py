"""
CO (Commanding Officer) state and data loading.

data/co_data.json schema expected:
{
  "cos": {
    "0": {
      "name": "Andy",
      "cop_stars": 3,
      "scop_stars": 6,
      "atk_modifiers": {"all": 0},
      "def_modifiers": {"all": 0},
      "cop": {
        "atk_modifiers": {},        // unique bonus BEYOND the universal SCOPB (+10/+10)
        "def_modifiers": {}
      },
      "scop": {
        "atk_modifiers": {"all": 10},  // unique bonus BEYOND SCOPB
        "def_modifiers": {"all": 10}
      }
    },
    ...
  }
}

Unit classes used as modifier keys:
  "infantry", "mech", "vehicle", "copter", "air", "naval", "pipe", "all"
  "all" is the catch-all default if a specific class key is absent.
"""
from __future__ import annotations
import sys

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from engine.unit import UnitType, UNIT_STATS

DATA_PATH = Path(__file__).parent.parent / "data" / "co_data.json"

_co_data_cache: Optional[dict] = None


def load_co_data() -> dict:
    global _co_data_cache
    if _co_data_cache is not None:
        return _co_data_cache
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"CO data file not found at {DATA_PATH}. "
            "Run the data-generation script to create data/co_data.json before using CO features."
        )
    with open(DATA_PATH, encoding="utf-8") as f:
        _co_data_cache = json.load(f)
    return _co_data_cache


@dataclass
class COState:
    co_id: int
    name: str
    cop_stars: Optional[int]  # None = no COP (e.g. Von Bolt in AWBW)
    scop_stars: int
    power_bar: int          # current charge accumulation
    cop_active: bool        # COP firing this turn
    scop_active: bool       # SCOP firing this turn
    _data: dict = field(repr=False, compare=False)
    power_uses: int = 0    # incremented each time COP or SCOP is activated

    def unit_cost_modifier_for_unit(self, ut: Any) -> int:
        """Return cost modifier (positive = cheaper) for a given unit type."""
        # This is a simplified version - in reality, COs have complex cost modifiers
        # For now, return 0 (no modifier)
        return 0


def make_co_state(co_id: int) -> COState:
    """
    Build a fresh COState from co_data.json.

    Raises FileNotFoundError if data/co_data.json is missing.
    Raises KeyError if co_id is not present in the JSON.
    """
    data = load_co_data()
    co_data = data["cos"].get(str(co_id))
    if co_data is None:
        raise KeyError(
            f"CO ID {co_id} not found in co_data.json. "
            f"Available IDs: {list(data['cos'].keys())}"
        )
    print(f"[DEBUG] make_co_state({co_id}): scop_stars={co_data.get('scop_stars')!r}, cop_stars={co_data.get('cop_stars')!r}", file=sys.stderr)
    return COState(
        co_id=co_id,
        name=co_data["name"],
        cop_stars=co_data.get("cop_stars"),  # may be null (no COP)
        scop_stars=int(co_data["scop_stars"]),
        power_bar=0,
        cop_active=False,
        scop_active=False,
        _data=co_data,
    )


def make_dummy_co_state(co_id: int = 0, name: str = "Generic CO") -> COState:
    """
    Fallback for when co_data.json doesn't exist yet.
    Creates a zero-modifier CO for testing the engine without data files.
    """
    # Unique modifiers are 0; SCOPB (+10 ATK/DEF) is applied universally in code
    dummy_data: dict = {
        "name": name,
        "cop_stars": 3,
        "scop_stars": 6,
        "atk_modifiers": {"all": 0},
        "def_modifiers": {"all": 0},
        "cop": {"atk_modifiers": {}, "def_modifiers": {}},
        "scop": {"atk_modifiers": {}, "def_modifiers": {}},
    }
    return COState(
        co_id=co_id,
        name=name,
        cop_stars=3,
        scop_stars=6,
        power_bar=0,
        cop_active=False,
        scop_active=False,
        _data=dummy_data,
    )


def make_co_state_safe(co_id: int) -> COState:
    """
    Try to load from co_data.json; fall back to dummy if file is absent.
    Useful during early development.
    """
    try:
        return make_co_state(co_id)
    except (FileNotFoundError, KeyError):
        return make_dummy_co_state(co_id)
