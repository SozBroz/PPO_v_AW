"""
Per-observer HP belief overlay.

The engine owns the exact 0–100 integer HP of every unit. Players (humans on
AWBW; bots at runtime) never see that integer directly — they see a **bar**
of 1–10 (``display_hp = ceil(hp/10)``) and, by knowing the damage formula,
narrow a plausible interval inside that bar each time they watch a combat.

``BeliefState`` is the machine analog of that inference. One instance lives
per seat (observer = 0 or 1). It stores a ``UnitBelief`` per unit and is
updated by ``rl/env.py`` after every engine step:

    pre  = self.snapshot_units(state)            # dict: unit_id -> pre-step fields
    state.step(action)                           # engine mutates ground truth
    self._apply_events(pre, state, action)       # update both beliefs

No engine state is mutated by this module. Combat math comes from
``engine.combat.damage_range``.

Invariants (per ``docs/hp_belief.md``):

1.  For the observer's **own** units, ``hp_min == hp_max == unit.hp``.
2.  For other units, ``hp_min >= bucket_low`` and ``hp_max <= bucket_high``
    where the bucket comes from the unit's current ``display_hp``.
3.  After a visible combat event the interval is narrowed by the damage
    formula range and then re-clamped to the new bucket.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from engine.unit import Unit


def _bucket(hp: int) -> int:
    """Display bucket 1..10 for a 1..100 HP; 0 for dead units."""
    if hp <= 0:
        return 0
    return (hp + 9) // 10


def _bucket_bounds(bucket: int) -> tuple[int, int]:
    """Inclusive ``(lo, hi)`` interval on the 1..100 scale for a bucket 1..10.
    Bucket 0 (dead) returns ``(0, 0)``.
    """
    if bucket <= 0:
        return 0, 0
    b = max(1, min(10, bucket))
    return b * 10 - 9, b * 10


@dataclass
class UnitBelief:
    unit_id: int
    player: int
    display_bucket: int  # 0 for dead, 1..10 otherwise
    hp_min: int          # 0..100, inclusive
    hp_max: int          # 0..100, inclusive; >= hp_min

    def is_dead(self) -> bool:
        return self.display_bucket == 0


class BeliefState:
    """Per-observer HP overlay.

    Parameters
    ----------
    observer : int
        Seat (0 or 1) whose perspective this belief represents. Units
        owned by ``observer`` always have ``hp_min == hp_max == unit.hp``
        via ``sync_own_units``; enemy units carry the narrowed interval
        inside their visible display bucket.
    """

    def __init__(self, observer: int) -> None:
        self.observer = int(observer)
        self._beliefs: dict[int, UnitBelief] = {}

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------
    def get(self, unit_id: int) -> Optional[UnitBelief]:
        return self._beliefs.get(unit_id)

    def __contains__(self, unit_id: int) -> bool:
        return unit_id in self._beliefs

    def all(self) -> list[UnitBelief]:
        return list(self._beliefs.values())

    # ------------------------------------------------------------------
    # Lifecycle: seed + per-step reconciliation
    # ------------------------------------------------------------------
    def seed_from_state(self, state) -> None:
        """Initial snapshot at episode start.

        Predeployed units start at whatever HP the map declares; the
        observer cannot see the exact integer. Own units are revealed
        (lo = hi = exact); enemy units clamp to the full bucket interval.
        """
        self._beliefs.clear()
        for player in (0, 1):
            for u in state.units[player]:
                self._add_or_reveal(u)

    def _add_or_reveal(self, unit: Unit) -> None:
        """Insert / reset a unit into belief.

        Own units: exact.
        Enemy units: bucket interval.
        """
        bucket = _bucket(unit.hp)
        if bucket == 0:
            self._beliefs.pop(unit.unit_id, None)
            return
        if unit.player == self.observer:
            hp_min = hp_max = unit.hp
        else:
            hp_min, hp_max = _bucket_bounds(bucket)
        self._beliefs[unit.unit_id] = UnitBelief(
            unit_id=unit.unit_id,
            player=unit.player,
            display_bucket=bucket,
            hp_min=hp_min,
            hp_max=hp_max,
        )

    def on_unit_built(self, unit: Unit) -> None:
        """Newly produced unit — enters at 100 HP (full bar), both seats see it.
        Observer's own build is exact (100/100); enemy build is still bucket 10
        but the interval collapses to the singleton since built = 100 HP.
        """
        bucket = _bucket(unit.hp)
        if bucket == 0:
            return
        # A fresh build is always 100 HP regardless of observer, so both seats
        # collapse to a singleton. Predeploy / CO-power spawns may differ and
        # should call ``_add_or_reveal`` (bucket interval) instead.
        if unit.hp == 100:
            self._beliefs[unit.unit_id] = UnitBelief(
                unit_id=unit.unit_id,
                player=unit.player,
                display_bucket=bucket,
                hp_min=unit.hp,
                hp_max=unit.hp,
            )
        else:
            self._add_or_reveal(unit)

    def on_unit_killed(self, unit_id: int) -> None:
        self._beliefs.pop(unit_id, None)

    def sync_own_units(self, state) -> None:
        """Force own-unit beliefs back to exact HP (lo = hi = unit.hp).

        Call after every engine step. Own-unit intervals drift during
        combat (counter-attacks, CO power HP manipulation) if we only
        narrow through the damage formula; this is the authoritative
        reconciliation.
        """
        seen: set[int] = set()
        for u in state.units[self.observer]:
            seen.add(u.unit_id)
            bucket = _bucket(u.hp)
            if bucket == 0:
                self._beliefs.pop(u.unit_id, None)
                continue
            self._beliefs[u.unit_id] = UnitBelief(
                unit_id=u.unit_id,
                player=self.observer,
                display_bucket=bucket,
                hp_min=u.hp,
                hp_max=u.hp,
            )
        # Drop own beliefs whose unit vanished mid-step (shouldn't happen
        # outside edge cases like CO power removals, but defensive).
        for uid in [uid for uid, b in self._beliefs.items()
                    if b.player == self.observer and uid not in seen]:
            self._beliefs.pop(uid, None)

    # ------------------------------------------------------------------
    # Combat / healing events (enemy units — own units handled via sync)
    # ------------------------------------------------------------------
    def on_damage(
        self,
        defender: Unit,
        dmg_min: int,
        dmg_max: int,
    ) -> None:
        """Apply a visible damage event of range ``[dmg_min, dmg_max]`` to
        the defender's belief, then re-clamp to the defender's new bucket.

        Own units skip this path — ``sync_own_units`` is authoritative.
        """
        if defender.player == self.observer:
            return
        prev = self._beliefs.get(defender.unit_id)
        if prev is None:
            # We never saw this unit pre-combat; fall back to a fresh bucket
            # reveal (still correct, just no formula tightening).
            self._add_or_reveal(defender)
            return
        new_bucket = _bucket(defender.hp)
        if new_bucket == 0:
            self._beliefs.pop(defender.unit_id, None)
            return
        lo_from_formula = max(0, prev.hp_min - dmg_max)
        hi_from_formula = max(0, prev.hp_max - dmg_min)
        b_lo, b_hi = _bucket_bounds(new_bucket)
        # Intersection of [formula range] ∩ [visible bucket]. The bucket is
        # the hard constraint — if the formula disagrees, trust the bar.
        hp_min = max(lo_from_formula, b_lo)
        hp_max = min(hi_from_formula, b_hi)
        if hp_min > hp_max:
            # Disagreement (e.g. formula gave [12,18], bucket is [21,30]
            # because a repair fired the same step) — fall back to the
            # bucket alone.
            hp_min, hp_max = b_lo, b_hi
        self._beliefs[defender.unit_id] = UnitBelief(
            unit_id=defender.unit_id,
            player=defender.player,
            display_bucket=new_bucket,
            hp_min=hp_min,
            hp_max=hp_max,
        )

    def on_heal(self, unit: Unit, delta_min: int, delta_max: int) -> None:
        """Apply a visible heal of amount ``[delta_min, delta_max]`` (both
        positive) to the unit, then re-clamp to the new bucket. Own units
        are handled by ``sync_own_units``.
        """
        if unit.player == self.observer:
            return
        new_bucket = _bucket(unit.hp)
        if new_bucket == 0:
            self._beliefs.pop(unit.unit_id, None)
            return
        prev = self._beliefs.get(unit.unit_id)
        if prev is None:
            self._add_or_reveal(unit)
            return
        lo_from_formula = min(100, prev.hp_min + delta_min)
        hi_from_formula = min(100, prev.hp_max + delta_max)
        b_lo, b_hi = _bucket_bounds(new_bucket)
        hp_min = max(lo_from_formula, b_lo)
        hp_max = min(hi_from_formula, b_hi)
        if hp_min > hp_max:
            hp_min, hp_max = b_lo, b_hi
        self._beliefs[unit.unit_id] = UnitBelief(
            unit_id=unit.unit_id,
            player=unit.player,
            display_bucket=new_bucket,
            hp_min=hp_min,
            hp_max=hp_max,
        )

    def reveal_bucket(self, unit: Unit) -> None:
        """Reset an enemy unit's belief to the full bucket interval (used
        when sight is re-established under fog — not yet wired in).
        """
        self._add_or_reveal(unit)
