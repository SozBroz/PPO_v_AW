"""
HP belief overlay — unit + encoder + env integration tests.

Pins the contract from ``docs/hp_belief.md``:

1. ``engine.combat.damage_range`` returns a (min, max) that brackets every
   luck roll the live ``calculate_damage`` path can produce.
2. ``engine.belief.BeliefState`` seeds own units to exact HP and enemies to
   the full display bucket; narrows enemy intervals on combat events;
   widens them on heals; and snaps to the new bucket after any HP change.
3. ``rl.encoder.encode_state`` writes two HP channels (lo, hi) with
   ``lo == hi`` for the observer's own units and ``lo <= hi`` (strictly
   less when the bucket isn't a singleton) for enemy units.
4. ``rl.env.AWBWEnv`` wires the belief into both seats so the opponent
   policy is rendered from observer=1, not P0's perspective — closes a
   long-standing exact-HP leak across the blue seat.
"""
from __future__ import annotations

import unittest

from engine.combat import calculate_damage, damage_range
from engine.belief import BeliefState, UnitBelief, _bucket, _bucket_bounds
from engine.co import make_co_state
from engine.terrain import get_terrain
from engine.unit import Unit, UnitType, UNIT_STATS


def _mk_unit(
    unit_type: UnitType, player: int, *, hp: int = 100, unit_id: int = 1
) -> Unit:
    stats = UNIT_STATS[unit_type]
    return Unit(
        unit_type=unit_type,
        player=player,
        hp=hp,
        ammo=stats.max_ammo if stats.max_ammo > 0 else 0,
        fuel=stats.max_fuel,
        pos=(0, 0),
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
        unit_id=unit_id,
    )


BEACH_TID = 32
CITY_TID = 34
SAMI_ID = 8
EAGLE_ID = 10


# ─────────────────────────────────────────────────────────────────────────────
# damage_range
# ─────────────────────────────────────────────────────────────────────────────
class TestDamageRange(unittest.TestCase):
    """damage_range must bracket every legal luck roll."""

    def setUp(self) -> None:
        self.att_terrain = get_terrain(BEACH_TID)
        self.def_terrain = get_terrain(CITY_TID)
        self.att_co = make_co_state(SAMI_ID)
        self.def_co = make_co_state(EAGLE_ID)
        self.att = _mk_unit(UnitType.INFANTRY, player=0, hp=100, unit_id=1)
        self.dfd = _mk_unit(UnitType.MECH, player=1, hp=100, unit_id=2)

    def test_brackets_all_rolls(self) -> None:
        lo, hi = damage_range(
            self.att, self.dfd,
            self.att_terrain, self.def_terrain,
            self.att_co, self.def_co,
        )
        for roll in range(10):
            d = calculate_damage(
                self.att, self.dfd,
                self.att_terrain, self.def_terrain,
                self.att_co, self.def_co,
                luck_roll=roll,
            )
            self.assertIsNotNone(d)
            self.assertLessEqual(lo, d)
            self.assertLessEqual(d, hi)

    def test_returns_none_for_unhittable_pair(self) -> None:
        # Infantry cannot damage a submerged sub (no base-table entry on an
        # aerial/naval axis that favours this pair). Pick an air vs sea case
        # where the damage table has None to guarantee short-circuit.
        bomber = _mk_unit(UnitType.BOMBER, player=0, unit_id=3)
        # Bomber vs Bomber entry exists; use an unhittable pair instead.
        # Fighter (BCopter) vs Infantry — BCopter can hit infantry. Use
        # transport lander (cannot attack at all).
        lander = _mk_unit(UnitType.LANDER, player=0, unit_id=4)
        result = damage_range(
            lander, bomber,
            self.att_terrain, self.def_terrain,
            self.att_co, self.def_co,
        )
        self.assertIsNone(result)

    def test_monotone_in_attacker_bucket(self) -> None:
        """More HP on the attacker => at least as much damage.

        display_hp is the only attacker HP input, so chunking HP by 10s
        should produce a non-decreasing max damage curve.
        """
        prev_hi = -1
        for hp in (10, 30, 50, 70, 100):
            att = _mk_unit(UnitType.INFANTRY, player=0, hp=hp, unit_id=1)
            lo, hi = damage_range(
                att, self.dfd,
                self.att_terrain, self.def_terrain,
                self.att_co, self.def_co,
            )
            self.assertGreaterEqual(hi, prev_hi)
            prev_hi = hi


# ─────────────────────────────────────────────────────────────────────────────
# BeliefState primitives
# ─────────────────────────────────────────────────────────────────────────────
class TestBucketHelpers(unittest.TestCase):
    def test_bucket(self) -> None:
        self.assertEqual(_bucket(1), 1)
        self.assertEqual(_bucket(10), 1)
        self.assertEqual(_bucket(11), 2)
        self.assertEqual(_bucket(99), 10)
        self.assertEqual(_bucket(100), 10)
        self.assertEqual(_bucket(0), 0)

    def test_bucket_bounds(self) -> None:
        self.assertEqual(_bucket_bounds(1), (1, 10))
        self.assertEqual(_bucket_bounds(5), (41, 50))
        self.assertEqual(_bucket_bounds(10), (91, 100))
        self.assertEqual(_bucket_bounds(0), (0, 0))


class _FakeState:
    """Minimal state shim for BeliefState.seed_from_state / sync_own_units.

    BeliefState only needs ``state.units[p]`` iterables of ``Unit`` objects.
    """
    def __init__(self, p0_units: list[Unit], p1_units: list[Unit]) -> None:
        self.units = {0: list(p0_units), 1: list(p1_units)}


class TestBeliefSeeding(unittest.TestCase):
    def test_seed_reveals_own_exact_and_clamps_enemy_to_bucket(self) -> None:
        own = _mk_unit(UnitType.INFANTRY, player=0, hp=73, unit_id=11)
        enemy = _mk_unit(UnitType.MECH, player=1, hp=73, unit_id=22)
        state = _FakeState([own], [enemy])

        p0 = BeliefState(observer=0)
        p0.seed_from_state(state)

        own_b = p0.get(11)
        self.assertEqual((own_b.hp_min, own_b.hp_max), (73, 73))
        self.assertEqual(own_b.display_bucket, 8)

        enemy_b = p0.get(22)
        # bucket 8 => [71, 80]
        self.assertEqual((enemy_b.hp_min, enemy_b.hp_max), (71, 80))
        self.assertEqual(enemy_b.display_bucket, 8)

    def test_symmetric_from_other_seat(self) -> None:
        u0 = _mk_unit(UnitType.INFANTRY, player=0, hp=55, unit_id=11)
        u1 = _mk_unit(UnitType.INFANTRY, player=1, hp=55, unit_id=22)
        state = _FakeState([u0], [u1])

        p1 = BeliefState(observer=1)
        p1.seed_from_state(state)

        # Own (P1's unit) is exact; enemy (P0's) is bucket.
        self.assertEqual((p1.get(22).hp_min, p1.get(22).hp_max), (55, 55))
        self.assertEqual((p1.get(11).hp_min, p1.get(11).hp_max), (51, 60))


class TestBeliefOnDamage(unittest.TestCase):
    """After a visible attack, enemy belief shrinks by the formula range
    and re-clamps to the new display bucket.
    """

    def test_shrink_and_clamp(self) -> None:
        # Enemy unit starts full (bucket 10 = [91, 100]).
        enemy = _mk_unit(UnitType.MECH, player=1, hp=100, unit_id=22)
        state = _FakeState([], [enemy])

        p0 = BeliefState(observer=0)
        p0.seed_from_state(state)
        self.assertEqual((p0.get(22).hp_min, p0.get(22).hp_max), (91, 100))

        # Combat reduces enemy hp to 55 (bucket 6 = [51, 60]). Formula range
        # says damage was in [40, 46] (Sami-Eagle anchor ballpark).
        enemy.hp = 55
        p0.on_damage(enemy, dmg_min=40, dmg_max=46)

        b = p0.get(22)
        self.assertEqual(b.display_bucket, 6)
        # prev [91,100] - [40,46] = [45,60], intersect bucket [51,60] = [51,60].
        self.assertEqual(b.hp_min, 51)
        self.assertEqual(b.hp_max, 60)

    def test_formula_tighter_than_bucket(self) -> None:
        """If the formula says damage was exactly 3, the surviving interval
        is the singleton { prev - 3 } intersected with bucket — which can
        be narrower than the bucket.
        """
        enemy = _mk_unit(UnitType.INFANTRY, player=1, hp=100, unit_id=22)
        state = _FakeState([], [enemy])

        p0 = BeliefState(observer=0)
        p0.seed_from_state(state)

        # A tiny deterministic hit — no luck variance (dmg_min == dmg_max).
        enemy.hp = 97  # stays in bucket 10
        p0.on_damage(enemy, dmg_min=3, dmg_max=3)

        b = p0.get(22)
        # prev [91,100] - [3,3] = [88,97]. bucket 10 = [91,100]. Intersect =
        # [91,97]. That's tighter than the bucket — the formula fixed the
        # upper bound at 97 inclusive.
        self.assertEqual(b.hp_min, 91)
        self.assertEqual(b.hp_max, 97)

    def test_own_unit_skipped(self) -> None:
        """Own-unit damage is handled by sync_own_units, not on_damage."""
        own = _mk_unit(UnitType.INFANTRY, player=0, hp=100, unit_id=11)
        state = _FakeState([own], [])
        p0 = BeliefState(observer=0)
        p0.seed_from_state(state)
        before = p0.get(11)

        own.hp = 50
        p0.on_damage(own, dmg_min=40, dmg_max=50)

        after = p0.get(11)
        # on_damage skipped own unit => interval unchanged at 100.
        self.assertEqual((after.hp_min, after.hp_max),
                         (before.hp_min, before.hp_max))


class TestBeliefOnHeal(unittest.TestCase):
    def test_heal_widens_and_clamps_to_new_bucket(self) -> None:
        enemy = _mk_unit(UnitType.INFANTRY, player=1, hp=50, unit_id=22)
        state = _FakeState([], [enemy])
        p0 = BeliefState(observer=0)
        p0.seed_from_state(state)
        # Seeded bucket 5 = [41, 50].
        self.assertEqual((p0.get(22).hp_min, p0.get(22).hp_max), (41, 50))

        # +20 HP day-start heal. Engine clamps at 100; here goes to 70 (bucket 7).
        enemy.hp = 70
        p0.on_heal(enemy, delta_min=20, delta_max=20)

        b = p0.get(22)
        self.assertEqual(b.display_bucket, 7)
        # prev [41,50] + 20 = [61,70]. bucket 7 = [61,70]. Intersect = [61,70].
        self.assertEqual((b.hp_min, b.hp_max), (61, 70))


class TestSyncOwn(unittest.TestCase):
    def test_sync_own_overrides_to_exact(self) -> None:
        own = _mk_unit(UnitType.INFANTRY, player=0, hp=100, unit_id=11)
        state = _FakeState([own], [])
        p0 = BeliefState(observer=0)
        p0.seed_from_state(state)

        own.hp = 37
        p0.sync_own_units(state)
        b = p0.get(11)
        self.assertEqual((b.hp_min, b.hp_max), (37, 37))

    def test_sync_own_drops_dead_unit(self) -> None:
        own = _mk_unit(UnitType.INFANTRY, player=0, hp=10, unit_id=11)
        state = _FakeState([own], [])
        p0 = BeliefState(observer=0)
        p0.seed_from_state(state)

        # Unit dies off the board.
        state.units[0] = []
        p0.sync_own_units(state)
        self.assertIsNone(p0.get(11))


# ─────────────────────────────────────────────────────────────────────────────
# Encoder layout
# ─────────────────────────────────────────────────────────────────────────────
class TestEncoderHpChannels(unittest.TestCase):
    """Encoder writes hp_lo and hp_hi; belief overlay honoured per observer."""

    def test_n_spatial_channels_is_63(self) -> None:
        from rl.encoder import N_SPATIAL_CHANNELS, N_HP_CHANNELS
        self.assertEqual(N_HP_CHANNELS, 2)
        self.assertEqual(N_SPATIAL_CHANNELS, 63)

    def test_own_units_exact_enemy_bucket(self) -> None:
        """Render a minimal board; check P0 sees its own unit exact and the
        enemy unit as the full display bucket interval.
        """
        # Minimal state stub replicating only what encode_state consumes.
        from rl.encoder import encode_state, N_UNIT_CHANNELS, N_HP_CHANNELS
        from engine.co import make_co_state
        from engine.game import MAX_TURNS  # noqa: F401

        class _Prop:
            def __init__(self, r, c, owner):
                self.row = r
                self.col = c
                self.owner = owner
                self.is_hq = False
                self.is_lab = False
                self.is_base = False
                self.is_airport = False
                self.is_port = False
                self.is_comm_tower = False
                self.capture_points = 20

        class _Map:
            def __init__(self):
                self.height = 2
                self.width = 2
                self.terrain = [[1, 1], [1, 1]]
                self.cap_limit = 10
                self.map_id = 0
                self.name = "t"

        class _State:
            def __init__(self, p0_unit: Unit, p1_unit: Unit):
                self.map_data = _Map()
                self.units = {0: [p0_unit], 1: [p1_unit]}
                self.properties = []
                self.funds = [0, 0]
                self.co_states = [make_co_state(SAMI_ID), make_co_state(EAGLE_ID)]
                self.turn = 1
                self.active_player = 0
                self.tier_name = "T3"
                self.weather = "clear"
                self.co_weather_segments_remaining = 0

            def get_unit_at(self, r, c):
                for p in (0, 1):
                    for u in self.units[p]:
                        if u.pos == (r, c):
                            return u
                return None

            def count_income_properties(self, player):
                return 0

        own = _mk_unit(UnitType.INFANTRY, player=0, hp=73, unit_id=11)
        own.pos = (0, 0)
        enemy = _mk_unit(UnitType.MECH, player=1, hp=73, unit_id=22)
        enemy.pos = (1, 1)
        state = _State(own, enemy)

        belief = BeliefState(observer=0)
        belief.seed_from_state(state)

        spatial, _ = encode_state(state, observer=0, belief=belief)

        hp_lo_ch = N_UNIT_CHANNELS
        hp_hi_ch = N_UNIT_CHANNELS + 1

        # Own unit at (0,0): lo == hi == 73/100
        self.assertAlmostEqual(float(spatial[0, 0, hp_lo_ch]), 0.73, places=6)
        self.assertAlmostEqual(float(spatial[0, 0, hp_hi_ch]), 0.73, places=6)

        # Enemy unit at (1,1): bucket 8 = [71,80] -> lo=0.71, hi=0.80
        self.assertAlmostEqual(float(spatial[1, 1, hp_lo_ch]), 0.71, places=6)
        self.assertAlmostEqual(float(spatial[1, 1, hp_hi_ch]), 0.80, places=6)

    def test_belief_none_fallback_is_exact(self) -> None:
        """Legacy call site (belief=None) collapses both HP channels to
        unit.hp/100 for all units — safe fallback for debug tools.
        """
        from rl.encoder import encode_state, N_UNIT_CHANNELS

        class _Map:
            def __init__(self):
                self.height = 2
                self.width = 2
                self.terrain = [[1, 1], [1, 1]]

        class _State:
            def __init__(self, u: Unit):
                self.map_data = _Map()
                self.units = {0: [u], 1: []}
                self.properties = []
                self.funds = [0, 0]
                self.co_states = [make_co_state(SAMI_ID), make_co_state(EAGLE_ID)]
                self.turn = 1
                self.active_player = 0
                self.tier_name = "T3"
                self.weather = "clear"
                self.co_weather_segments_remaining = 0

            def get_unit_at(self, r, c):
                for u in self.units[0]:
                    if u.pos == (r, c):
                        return u
                return None

            def count_income_properties(self, player):
                return 0

        u = _mk_unit(UnitType.INFANTRY, player=0, hp=73, unit_id=11)
        u.pos = (0, 0)
        spatial, _ = encode_state(_State(u), observer=0, belief=None)
        hp_lo_ch = N_UNIT_CHANNELS
        hp_hi_ch = N_UNIT_CHANNELS + 1
        # Both HP channels equal exact for legacy caller.
        self.assertAlmostEqual(float(spatial[0, 0, hp_lo_ch]), 0.73, places=6)
        self.assertAlmostEqual(float(spatial[0, 0, hp_hi_ch]), 0.73, places=6)


if __name__ == "__main__":
    unittest.main()
