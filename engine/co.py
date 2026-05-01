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
    # When True (RL curriculum only), COP cannot be activated this episode; SCOP is
    # unchanged. Default False for replays/oracle.
    cop_activation_disabled: bool = False
    comm_towers: int = 0   # owned comm towers; updated by GameState each turn
    # Sasha (co_id 19) SCOP "War Bonds" — credits 50% of damage cost (capped at
    # 9 display HP per attack) to her treasury for damage dealt by her units.
    #
    # Lifecycle:
    #   * Set ``war_bonds_active = True`` when Sasha activates SCOP.
    #   * Each qualifying attack (her own attacks during her turn AND counter-
    #     attacks dealt by her defending units during the opponent's
    #     intervening turn) ACCUMULATES the payout into
    #     ``pending_war_bonds_funds`` rather than crediting funds immediately.
    #   * At the end of the opponent's intervening turn (i.e., immediately
    #     before Sasha's next turn begins), ``_end_turn`` credits
    #     ``pending_war_bonds_funds`` to her treasury, resets the counter to
    #     0, and clears ``war_bonds_active``.
    #
    # Why deferred: PHP credits the War Bonds bonus as part of the start-of-
    # next-turn settlement, not in real-time during each strike. Empirically
    # confirmed on game ``1624082``: the −200g engine→PHP delta first appears
    # at the snapshot AFTER P0's intervening turn (env 22), not during
    # Sasha's own turn (env 21) where the snapshot still matched. Crediting
    # in real-time also creates upstream state drift mid-turn (extra spending
    # power → different unit-build/repair decisions) that regressed 23 of
    # 100 games in the GL std corpus during initial roll-out.
    war_bonds_active: bool = False
    pending_war_bonds_funds: int = 0
    # Kindle (co_id=23): count of urban properties (HQs, bases, airports,
    # ports, cities, labs, comm towers) she currently owns. Updated alongside
    # ``comm_towers`` via ``GameState._refresh_comm_towers``. Consumed by the
    # SCOP "High Society" +3%/prop global ATK rider in ``engine/combat.py``.
    urban_props: int = 0
    # Colin (co_id=15) SCOP "Power of Money" — funds snapshot at SCOP
    # activation. Consumed by ``_colin_atk_rider`` in ``engine/combat.py`` to
    # compute the +(3 * funds / 1000)% attack rider for the SCOP duration.
    # Snapshotted at activation (not read live during each attack) so the
    # bonus stays stable across mid-turn builds/spending. Phase
    # 11J-COLIN-IMPL-SHIP. Source: docs/oracle_exception_audit/phase11y_colin_scrape.md §0.3.
    colin_pom_funds_snapshot: int = 0

    # ------------------------------------------------------------------
    # Modifier helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _unit_can_attack(unit_type: UnitType) -> bool:
        """True for units with a weapon, including MG-only direct attackers."""
        stats = UNIT_STATS[unit_type]
        return stats.max_ammo > 0 or unit_type == UnitType.RECON

    @classmethod
    def _is_direct_awbw_tag(cls, unit_type: UnitType) -> bool:
        """AWBW CO-chart direct tag: attacking non-footsoldier direct units."""
        stats = UNIT_STATS[unit_type]
        if stats.is_indirect:
            return False
        if unit_type in (UnitType.INFANTRY, UnitType.MECH):
            return False
        return cls._unit_can_attack(unit_type)

    @classmethod
    def _modifier_keys_for_unit(cls, unit_type: UnitType) -> tuple[str, ...]:
        """Lookup priority for JSON modifier maps.

        The first matching key wins. This lets specific unit classes override
        broader tags like ``direct`` / ``indirect`` / ``all``.
        """
        stats = UNIT_STATS[unit_type]
        keys: list[str] = [stats.unit_class]
        if stats.land_indirect:
            keys.append("land_indirect")
        if stats.is_transport:
            keys.append("transport")
        if stats.is_indirect and cls._unit_can_attack(unit_type):
            keys.append("indirect")
        if cls._is_direct_awbw_tag(unit_type):
            keys.append("direct")
        keys.append("all")
        return tuple(dict.fromkeys(keys))

    def _section(self, field: str, source_key: Optional[str] = None) -> dict:
        """Return a modifier section from top-level/day-to-day or a power."""
        if source_key is not None:
            source = self._data.get(source_key) or {}
            return source.get(field, {}) or {}
        merged: dict = {}
        top = self._data.get(field, {}) or {}
        if isinstance(top, dict):
            merged.update(top)
        day_to_day = self._data.get("day_to_day", {}) or {}
        dtd = day_to_day.get(field, {}) or {}
        if isinstance(dtd, dict):
            merged.update(dtd)
        return merged

    def _lookup(self, section: dict, unit_class: str) -> int:
        """Return modifier for legacy unit-class callers."""
        return int(section.get(unit_class, section.get("all", 0)) or 0)

    def _lookup_for_unit(self, section: dict, unit_type: UnitType) -> int:
        """Return the first matching modifier for a concrete unit type."""
        for key in self._modifier_keys_for_unit(unit_type):
            if key in section:
                return int(section[key] or 0)
        return 0

    def atk_modifier(self, unit_class: str) -> int:
        """Day-to-day ATK percent bonus (e.g. 20 → +20%)."""
        return self._lookup(self._section("atk_modifiers"), unit_class)

    def atk_modifier_for_unit(self, unit_type: UnitType) -> int:
        """Day-to-day ATK percent bonus for a concrete unit type."""
        return self._lookup_for_unit(self._section("atk_modifiers"), unit_type)

    def def_modifier(self, unit_class: str) -> int:
        """Day-to-day DEF percent bonus."""
        return self._lookup(self._section("def_modifiers"), unit_class)
    
    def def_modifier_for_unit(self, unit_type: UnitType) -> int:
        """Day-to-day DEF percent bonus for a concrete unit type."""
        return self._lookup_for_unit(self._section("def_modifiers"), unit_type)

    def _against_indirect_bonus(self, section: dict, attacker_unit_type: Optional[UnitType]) -> int:
        if attacker_unit_type is None:
            return 0
        if not UNIT_STATS[attacker_unit_type].is_indirect:
            return 0
        return int(section.get("against_indirect", section.get("against_indirects", 0)) or 0)

    def def_modifier_against(self, defender_unit_class: str, attacker_unit_type: Optional[UnitType] = None) -> int:
        """Day-to-day DEF percent bonus against a specific attacker.
        
        Handles contextual keys such as Javier's ``against_indirect``.
        """
        section = self._section("def_modifiers")
        return self._lookup(section, defender_unit_class) + self._against_indirect_bonus(section, attacker_unit_type)

    def def_modifier_for_unit_against(
        self,
        defender_unit_type: UnitType,
        attacker_unit_type: Optional[UnitType] = None,
    ) -> int:
        section = self._section("def_modifiers")
        return self._lookup_for_unit(section, defender_unit_type) + self._against_indirect_bonus(section, attacker_unit_type)

    def cop_atk_modifier(self, unit_class: str) -> int:
        """ATK bonus from active COP/SCOP (0 if no power active).

        Always includes the universal SCOPB (+10% ATK) when any power is active,
        plus whatever CO-specific ATK bonus the power provides.
        """
        if not self.cop_active and not self.scop_active:
            return 0
        source_key = "scop" if self.scop_active else "cop"
        section = self._section("atk_modifiers", source_key)
        return 10 + self._lookup(section, unit_class)  # 10 = SCOPB

    def cop_atk_modifier_for_unit(self, unit_type: UnitType) -> int:
        """ATK bonus from active COP/SCOP for a concrete unit."""
        if not self.cop_active and not self.scop_active:
            return 0
        source_key = "scop" if self.scop_active else "cop"
        return 10 + self._lookup_for_unit(self._section("atk_modifiers", source_key), unit_type)

    def cop_def_modifier(self, unit_class: str) -> int:
        """DEF bonus from active COP/SCOP.

        Always includes the universal SCOPB (+10% DEF) when any power is active,
        plus whatever CO-specific DEF bonus the power provides.
        """
        if not self.cop_active and not self.scop_active:
            return 0
        source_key = "scop" if self.scop_active else "cop"
        section = self._section("def_modifiers", source_key)
        return 10 + self._lookup(section, unit_class)  # 10 = SCOPB

    def cop_def_modifier_for_unit_against(
        self,
        defender_unit_type: UnitType,
        attacker_unit_type: Optional[UnitType] = None,
    ) -> int:
        """DEF bonus from active COP/SCOP, including contextual JSON keys."""
        if not self.cop_active and not self.scop_active:
            return 0
        source_key = "scop" if self.scop_active else "cop"
        section = self._section("def_modifiers", source_key)
        return 10 + self._lookup_for_unit(section, defender_unit_type) + self._against_indirect_bonus(section, attacker_unit_type)

    def movement_modifier_for_unit(self, unit_type: UnitType) -> int:
        """Movement-point bonus from D2D plus active power JSON modifiers."""
        bonus = self._lookup_for_unit(self._section("movement_modifiers"), unit_type)
        bonus += self._lookup_for_unit(self._section("transport_modifiers"), unit_type)
        if self.cop_active:
            bonus += self._lookup_for_unit(self._section("movement_modifiers", "cop"), unit_type)
            bonus += self._lookup_for_unit(self._section("transport_modifiers", "cop"), unit_type)
        if self.scop_active:
            bonus += self._lookup_for_unit(self._section("movement_modifiers", "scop"), unit_type)
            bonus += self._lookup_for_unit(self._section("transport_modifiers", "scop"), unit_type)
        return bonus

    def range_modifier_for_unit(self, unit_type: UnitType) -> int:
        """Attack max-range bonus from D2D plus active power JSON modifiers."""
        bonus = self._lookup_for_unit(self._section("range_modifiers"), unit_type)
        if self.cop_active:
            bonus += self._lookup_for_unit(self._section("range_modifiers", "cop"), unit_type)
        if self.scop_active:
            bonus += self._lookup_for_unit(self._section("range_modifiers", "scop"), unit_type)
        return bonus

    def unit_cost_modifier_for_unit(self, unit_type: UnitType) -> int:
        """Build-cost percent modifier from JSON, e.g. -20 for Colin."""
        return self._lookup_for_unit(self._section("unit_cost_modifiers"), unit_type)

    def luck_bounds(self) -> Optional[tuple[int, int]]:
        """Return active luck bounds from JSON as ``(low, high_exclusive)``."""
        section = self._section("luck_modifiers")
        if self.cop_active:
            section = self._section("luck_modifiers", "cop") or section
        if self.scop_active:
            section = self._section("luck_modifiers", "scop") or section
        raw = section.get("all")
        if raw is None:
            return None
        if isinstance(raw, str):
            low_s, high_s = raw.split(",", 1)
            return int(low_s.strip()), int(high_s.strip())
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            return int(raw[0]), int(raw[1])
        return None
    
    def total_atk_for_unit(self, unit_type: UnitType) -> int:
        """Base 100 + DTD + active power ATK modifier + Javier tower bonus for a unit."""
        return 100 + self.atk_modifier_for_unit(unit_type) + self.cop_atk_modifier_for_unit(unit_type) + self.tower_atk_bonus()
    
    def total_def_for_unit(self, unit_type: UnitType) -> int:
        """Base 100 + DTD + active power DEF modifier + Javier tower bonus for a unit."""
        return 100 + self.def_modifier_for_unit(unit_type) + self.cop_def_modifier_for_unit_against(unit_type) + self.tower_def_bonus()

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
        if self.cop_activation_disabled:
            return False
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
        return self.luck_bounds() is not None

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
    
    def total_def_against(self, defender_unit_class: str, attacker_unit_type: Optional[UnitType] = None) -> int:
        """Base 100 + DTD (with against_indirect) + active power DEF modifier + Javier tower bonus."""
        return (100 + self.def_modifier_against(defender_unit_class, attacker_unit_type)
                + self.cop_def_modifier(defender_unit_class)
                + self.tower_def_bonus())
    
    def total_def_for_unit_against(self, defender_unit_type: UnitType, attacker_unit_type: Optional[UnitType] = None) -> int:
        """Base 100 + DTD (with against_indirect) + active power DEF modifier + Javier tower bonus for a unit."""
        return (
            100
            + self.def_modifier_for_unit_against(defender_unit_type, attacker_unit_type)
            + self.cop_def_modifier_for_unit_against(defender_unit_type, attacker_unit_type)
            + self.tower_def_bonus()
        )


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
