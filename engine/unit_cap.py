"""Unit-limit helpers: AWBW counts owned alive units including cargo aboard transports.

While cargo is loaded it is removed from ``GameState.units[player]``; factory BUILD
and Sensei COP spawns must still count those units toward ``map_data.unit_limit``.
"""
from __future__ import annotations

from engine.unit import Unit


def alive_owned_unit_count(units: list[Unit]) -> int:
    """Return the number of alive units owned by this roster row, including nested cargo."""

    def subtree(u: Unit) -> int:
        if not u.is_alive:
            return 0
        n = 1
        for c in u.loaded_units:
            n += subtree(c)
        return n

    return sum(subtree(u) for u in units)
