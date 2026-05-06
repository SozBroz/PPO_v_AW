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
    # Sasha's War Bonds power: tracks pending funds to be added at turn end
    war_bonds_active: bool = False
    pending_war_bonds_funds: int = 0

    def unit_cost_modifier_for_unit(self, ut: Any) -> int:
        """
        Return cost modifier in % (positive = cheaper) for a given unit type.
        e.g. Kanbei: {"all": 20} → units cost 120% of base (modifier = 20).

        In co_data.json, unit_cost_modifiers lives inside "day_to_day",
        not at the top level. We check there first, then fall back to
        top-level (legacy) and active-power scopes.
        """
        stats = UNIT_STATS.get(ut)
        if stats is None:
            return 0
        cls = stats.unit_class

        def _lookup(mods: dict) -> Optional[int]:
            """Lookup cls or 'all' in mods dict; return int or None."""
            if not mods:
                return None
            if cls in mods:
                return int(mods[cls])
            if "all" in mods:
                return int(mods["all"])
            return None

        # Active power modifiers take priority
        if self.cop_active and "cop" in self._data:
            mods = self._data["cop"].get("unit_cost_modifiers", {})
            r = _lookup(mods)
            if r is not None:
                return r
        if self.scop_active and "scop" in self._data:
            mods = self._data["scop"].get("unit_cost_modifiers", {})
            r = _lookup(mods)
            if r is not None:
                return r

        # Day-to-day modifiers (where unit_cost_modifiers actually lives)
        d2d = self._data.get("day_to_day", {})
        mods = d2d.get("unit_cost_modifiers", {})
        r = _lookup(mods)
        if r is not None:
            return r

        # Legacy: top-level (should not be hit with current co_data.json)
        mods = self._data.get("unit_cost_modifiers", {})
        r = _lookup(mods)
        if r is not None:
            return r
        return 0

    def movement_modifier_for_unit(self, ut: UnitType) -> int:
        """
        Return movement-point bonus for a given unit type.
        Checks active-power modifiers first (COP/SCOP), then day-to-day.
        Keys in co_data.json: unit class strings ("infantry", "vehicle", etc.)
        or "all" (applies to every class).
        """
        stats = UNIT_STATS.get(ut)
        if stats is None:
            return 0
        cls = stats.unit_class  # e.g. "infantry", "mech", "vehicle", "copter", "air", "naval", "pipe"

        # Active power modifiers take priority
        if self.cop_active and "cop" in self._data:
            mods = self._data["cop"].get("movement_modifiers", {})
            if cls in mods:
                return int(mods[cls])
            if "all" in mods:
                return int(mods["all"])
        if self.scop_active and "scop" in self._data:
            mods = self._data["scop"].get("movement_modifiers", {})
            if cls in mods:
                return int(mods[cls])
            if "all" in mods:
                return int(mods["all"])

        # Day-to-day modifiers
        mods = self._data.get("movement_modifiers", {})
        if cls in mods:
            return int(mods[cls])
        if "all" in mods:
            return int(mods["all"])
        return 0

    def range_modifier_for_unit(self, ut: UnitType) -> int:
        """
        Return attack-range bonus for a given unit type (e.g. Grit's +1/+2).
        Checks active-power modifiers first (COP/SCOP), then day-to-day.
        Keys: unit class strings or "all", and "indirect"/"direct"/"land_indirect".
        """
        stats = UNIT_STATS.get(ut)
        if stats is None:
            return 0
        cls = stats.unit_class

        # Build key priority list
        keys = [cls]
        if stats.is_indirect:
            keys.append("indirect")
        else:
            keys.append("direct")
        if stats.land_indirect:
            keys.append("land_indirect")
        keys.append("all")

        # Active power modifiers take priority
        if self.cop_active and "cop" in self._data:
            mods = self._data["cop"].get("range_modifiers", {})
            for key in keys:
                if key in mods:
                    return int(mods[key])
        if self.scop_active and "scop" in self._data:
            mods = self._data["scop"].get("range_modifiers", {})
            for key in keys:
                if key in mods:
                    return int(mods[key])

        # Day-to-day modifiers
        mods = self._data.get("range_modifiers", {})
        for key in keys:
            if key in mods:
                return int(mods[key])
        return 0

    def luck_bounds(self) -> Optional[tuple[int, int]]:
        """
        Return (low, high) luck percentages for this CO's current state.
        Sources: AWBW CO Chart (https://awbw.amarriner.com/co.php).

        Returns None for COs with no luck modification (Andy, Hachi, etc.).
        Dual-luck COs (Sonja, Flak, Jugger) return negative low values.
        """
        # Luck by CO ID: (d2d_low, d2d_high, cop_low, cop_high, scop_low, scop_high)
        # Values from AWBW CO Chart.
        LUCK = {
            5: (0, 19, 0, 59, 0, 99),     # Nell
            18: (-9, 9, -9, 9, -9, 9),       # Sonja (dual: -9..+9)
            22: (-9, 24, -19, 49, -39, 89),   # Flak (dual)
            23: (-14, 29, -24, 54, -44, 94),  # Jugger (dual)
        }
        if self.co_id not in LUCK:
            return None
        d2d_low, d2d_high, cop_low, cop_high, scop_low, scop_high = LUCK[self.co_id]
        if self.scop_active:
            return (scop_low, scop_high)
        if self.cop_active:
            return (cop_low, cop_high)
        return (d2d_low, d2d_high)

    @property
    def total_atk(self) -> int:
        """Total attack modifier (base 100 = 0% mod). Includes SCOPB +10 when SCOP active."""
        base = 100
        mods = self._data.get("atk_modifiers", {})
        base += mods.get("all", 0)
        if self.cop_active:
            cop_mods = self._data.get("cop", {}).get("atk_modifiers", {})
            base += cop_mods.get("all", 0)
        if self.scop_active:
            scop_mods = self._data.get("scop", {}).get("atk_modifiers", {})
            base += scop_mods.get("all", 0)
            base += 10  # SCOPB universal +10 ATK
        return base

    @property
    def total_def(self) -> int:
        """Total defense modifier (base 100 = 0% mod). Includes SCOPB +10 when SCOP active."""
        base = 100
        mods = self._data.get("def_modifiers", {})
        base += mods.get("all", 0)
        if self.cop_active:
            cop_mods = self._data.get("cop", {}).get("def_modifiers", {})
            base += cop_mods.get("all", 0)
        if self.scop_active:
            scop_mods = self._data.get("scop", {}).get("def_modifiers", {})
            base += scop_mods.get("all", 0)
            base += 10  # SCOPB universal +10 DEF
        return base

    def total_atk_for_unit(self, ut: UnitType) -> int:
        """Total attack modifier for a specific unit type (base 100)."""
        base = self.total_atk
        stats = UNIT_STATS.get(ut)
        if stats is None:
            return base
        cls = stats.unit_class
        # Day-to-day unit-specific modifiers
        mods = self._data.get("atk_modifiers", {})
        if cls in mods:
            base += int(mods[cls])
        # COP modifiers
        if self.cop_active:
            cop_mods = self._data.get("cop", {}).get("atk_modifiers", {})
            if cls in cop_mods:
                base += int(cop_mods[cls])
        # SCOP modifiers
        if self.scop_active:
            scop_mods = self._data.get("scop", {}).get("atk_modifiers", {})
            if cls in scop_mods:
                base += int(scop_mods[cls])
        return base

    def total_def_for_unit(self, ut: UnitType) -> int:
        """Total defense modifier for a specific unit type (base 100)."""
        base = self.total_def
        stats = UNIT_STATS.get(ut)
        if stats is None:
            return base
        cls = stats.unit_class
        # Day-to-day unit-specific modifiers
        mods = self._data.get("def_modifiers", {})
        if cls in mods:
            base += int(mods[cls])
        # COP modifiers
        if self.cop_active:
            cop_mods = self._data.get("cop", {}).get("def_modifiers", {})
            if cls in cop_mods:
                base += int(cop_mods[cls])
        # SCOP modifiers
        if self.scop_active:
            scop_mods = self._data.get("scop", {}).get("def_modifiers", {})
            if cls in scop_mods:
                base += int(scop_mods[cls])
        return base

    def total_def_for_unit_against(self, ut: UnitType, attacker_ut: UnitType) -> int:
        """Total defense modifier for a specific unit type against an attacker (base 100)."""
        base = self.total_def
        stats = UNIT_STATS.get(ut)
        if stats is None:
            return base
        cls = stats.unit_class
        # Day-to-day unit-specific modifiers
        mods = self._data.get("def_modifiers", {})
        if cls in mods:
            base += int(mods[cls])
        # COP modifiers
        if self.cop_active:
            cop_mods = self._data.get("cop", {}).get("def_modifiers", {})
            if cls in cop_mods:
                base += int(cop_mods[cls])
        # SCOP modifiers
        if self.scop_active:
            scop_mods = self._data.get("scop", {}).get("def_modifiers", {})
            if cls in scop_mods:
                base += int(scop_mods[cls])
        return base

    def can_activate_cop(self) -> bool:
        """True if COP can be activated (bar >= threshold and COP exists)."""
        if self.cop_stars is None:
            return False
        return self.power_bar >= self._cop_threshold

    def can_activate_scop(self) -> bool:
        """True if SCOP can be activated (bar >= threshold)."""
        return self.power_bar >= self._scop_threshold

    @property
    def _cop_threshold(self) -> int:
        """COP activation threshold (stars * 9000 + uses * 1800), capped at 10 uses."""
        if self.cop_stars is None:
            return 0
        uses = min(self.power_uses, 10)
        return self.cop_stars * 9000 + uses * 1800

    @property
    def _scop_threshold(self) -> int:
        """SCOP activation threshold (stars * 9000 + uses * 1800), capped at 10 uses."""
        uses = min(self.power_uses, 10)
        return self.scop_stars * 9000 + uses * 1800


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
