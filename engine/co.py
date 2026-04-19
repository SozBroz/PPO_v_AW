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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
    comm_towers: int = 0   # owned comm towers; updated by GameState each turn

    # ------------------------------------------------------------------
    # Modifier helpers
    # ------------------------------------------------------------------
    def _lookup(self, section: dict, unit_class: str) -> int:
        """Return modifier for unit class, falling back to 'all', then 0."""
        return section.get(unit_class, section.get("all", 0))

    def atk_modifier(self, unit_class: str) -> int:
        """Day-to-day ATK percent bonus (e.g. 20 → +20%)."""
        return self._lookup(self._data.get("atk_modifiers", {}), unit_class)

    def def_modifier(self, unit_class: str) -> int:
        """Day-to-day DEF percent bonus."""
        return self._lookup(self._data.get("def_modifiers", {}), unit_class)

    def cop_atk_modifier(self, unit_class: str) -> int:
        """ATK bonus from active COP/SCOP (0 if no power active).

        Always includes the universal SCOPB (+10% ATK) when any power is active,
        plus whatever CO-specific ATK bonus the power provides.
        """
        if not self.cop_active and not self.scop_active:
            return 0
        source_key = "scop" if self.scop_active else "cop"
        section = self._data.get(source_key, {}).get("atk_modifiers", {})
        return 10 + self._lookup(section, unit_class)  # 10 = SCOPB

    def cop_def_modifier(self, unit_class: str) -> int:
        """DEF bonus from active COP/SCOP.

        Always includes the universal SCOPB (+10% DEF) when any power is active,
        plus whatever CO-specific DEF bonus the power provides.
        """
        if not self.cop_active and not self.scop_active:
            return 0
        source_key = "scop" if self.scop_active else "cop"
        section = self._data.get(source_key, {}).get("def_modifiers", {})
        return 10 + self._lookup(section, unit_class)  # 10 = SCOPB

    # ------------------------------------------------------------------
    # Power activation guards
    # ------------------------------------------------------------------
    @property
    def _cop_threshold(self) -> int:
        if self.cop_stars is None:
            return 10**12  # no COP — threshold unreachable
        # +1800 per star per prior use (AWBW changelog rev 139, 2018-06-30)
        return self.cop_stars * (9000 + self.power_uses * 1800)

    @property
    def _scop_threshold(self) -> int:
        return self.scop_stars * (9000 + self.power_uses * 1800)

    def can_activate_cop(self) -> bool:
        if self.cop_stars is None or self._data.get("cop") is None:
            return False
        return (
            self.power_bar >= self._cop_threshold
            and not self.cop_active
            and not self.scop_active
        )

    def can_activate_scop(self) -> bool:
        return (
            self.power_bar >= self._scop_threshold
            and not self.cop_active
            and not self.scop_active
        )

    # ------------------------------------------------------------------
    # Special CO flags queried elsewhere
    # ------------------------------------------------------------------
    @staticmethod
    def sonja_counter_break(co_id: int) -> bool:
        """Sonja's SCOP: counters use full HP, ignoring damage taken."""
        return co_id == 18

    def has_luck_modifier(self) -> bool:
        return self.co_id in (24, 28, 25, 26)  # Nell, Rachel, Flak, Jugger

    # ------------------------------------------------------------------
    # Javier: comm-tower dynamic bonuses
    # ------------------------------------------------------------------
    def tower_atk_bonus(self) -> int:
        """Javier (co_id=27): +40% ATK per owned comm tower during SCOP only."""
        if self.co_id != 27:
            return 0
        if self.scop_active:
            return self.comm_towers * 40
        return 0

    def tower_def_bonus(self) -> int:
        """Javier (co_id=27): +10% DEF per owned comm tower (D2D); +20 additional per tower
        during COP; +40 additional per tower during SCOP."""
        if self.co_id != 27:
            return 0
        if self.scop_active:
            return self.comm_towers * (10 + 40)  # D2D 10 + SCOP unique 40
        elif self.cop_active:
            return self.comm_towers * (10 + 20)  # D2D 10 + COP unique 20
        return self.comm_towers * 10             # D2D only

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def total_atk(self, unit_class: str) -> int:
        """Base 100 + DTD + active power ATK modifier + Javier tower bonus."""
        return (100 + self.atk_modifier(unit_class)
                + self.cop_atk_modifier(unit_class)
                + self.tower_atk_bonus())

    def total_def(self, unit_class: str) -> int:
        """Base 100 + DTD + active power DEF modifier + Javier tower bonus."""
        return (100 + self.def_modifier(unit_class)
                + self.cop_def_modifier(unit_class)
                + self.tower_def_bonus())


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

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
        "cop":  {"atk_modifiers": {}, "def_modifiers": {}},
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
