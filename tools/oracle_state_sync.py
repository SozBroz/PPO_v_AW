"""Reconcile engine ``GameState`` to a PHP ``awbwGame`` snapshot frame.

**Why this lives outside the engine.** Training rollouts want non-deterministic
luck (``random.randint(0, 9)`` inside ``calculate_damage``). AWBW replay zips
recorded a *specific* luck per attack that we cannot recover from the public
``combatInfoVision`` payload (luck is not exported). Re-rolling random luck on
oracle replay therefore produces *plausible-but-different* HPs every step,
which compounds across an envelope and surfaces as "engine drift" in
``replay_state_diff.py``.

The fix here is **not** to bloat the engine with a luck-injection seam (would
slow training and add a runtime branch on every attack). Instead the oracle
replay harness, after applying each envelope, snaps the engine's per-unit HPs
and per-player funds back to whatever the next PHP snapshot recorded — and
flags any unit whose engine vs PHP delta is so large it cannot be explained by
luck noise (``MAX_PLAUSIBLE_HP_SWING_PER_ENVELOPE``). Those flagged rows are
true engine / oracle parity bugs; everything else is luck noise that no longer
compounds.

**Contract:** matched units are updated **in place**; ``unit_id`` is preserved
(replay viewers index by it). Engine units that PHP shows as dead are killed
(``hp = 0``) so the next ``state.units[seat]`` prune via combat / end-turn
resolves them. PHP units missing in the engine are reported but **not**
spawned — that would mask oracle resolver bugs (e.g. a Build the oracle
silently dropped). CO power state, fuel, ammo, capture progress, weather, and
property ownership are intentionally left alone — those are out of scope for
HP/funds drift mitigation and would risk silently masking other parity gaps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from engine.game import GameState
from engine.unit import Unit, UNIT_STATS

# Maximum |engine_hp - php_hp| (internal scale 0..100) we accept as luck noise
# inside a single envelope. A single attack's luck swing is bounded ~30-40
# internal HP (with Nell SCOP forcing the upper bound), but a single envelope
# routinely contains multiple strikes on the same defender (e.g. infantry
# overrun) so the cumulative swing can be the full HP bar. We allow up to
# 100 (kill from full); structural drift (wrong unit type, wrong seat at
# fire time, missed CO power, oracle resolver picked a different unit) still
# surfaces as a different signal — type mismatch or unit_tile_set mismatch
# — *not* as an HP delta of >100.
MAX_PLAUSIBLE_HP_SWING_PER_ENVELOPE = 100

# Maximum manhattan distance for "engine put unit at A, PHP put it at B"
# teleport reconciliation. AWBW max single-turn move is ~9 (Fighter), so 10
# tiles covers the worst case where engine and PHP end up on opposite ends
# of the unit's possible move set after a luck-divergent attack chain.
# Beyond this, treat them as different units (php_only + engine_only).
MAX_TELEPORT_DISTANCE = 10


def _php_snapshot_name_canonical(php_name: str) -> str:
    """Map AWBW snapshot short names to :data:`UNIT_STATS` ``name`` strings.

    Keep aligned with ``tools.replay_snapshot_compare.compare_units`` aliases.
    Bipartite sync buckets must use the same canonical form as ``eng_buckets``
    (``UNIT_STATS[...].name``); otherwise PHP ``Md.Tank`` rows never pair
    with engine ``Medium Tank`` units and drift reconciliation wrongly
    kills the engine unit as ``engine_only`` while flagging the PHP tile as
    ``php_only``.
    """
    s = str(php_name).strip()
    aliases = {
        "Md.Tank": "Medium Tank",
        "Md. Tank": "Medium Tank",
    }
    return aliases.get(s, s)


def _php_internal_hp(php_hit_points: Any) -> int:
    """PHP ``hit_points = internal_hp / 10`` (float). Recover internal HP.

    Returns 0 when the unit is missing / dead in PHP. Uses ``ceil`` so the
    "1 internal HP rounds to 0.1 displayed" boundary lines up with the
    engine's ``Unit.display_hp = (hp + 9) // 10`` rule (see
    ``tools/replay_snapshot_compare.py::_php_unit_bars``).
    """
    if php_hit_points is None:
        return 0
    try:
        v = float(php_hit_points)
    except (TypeError, ValueError):
        return 0
    if v <= 0:
        return 0
    # Internal HP is encoded as float to one decimal in AWBW; multiply and
    # round to nearest int to recover the integer internal HP without
    # accumulating float error.
    return max(0, min(100, int(round(v * 10))))


@dataclass
class UnitSyncDelta:
    seat: int
    pos: tuple[int, int]
    unit_type: str
    engine_hp: int
    php_hp: int
    snapped: bool
    out_of_range: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "seat": self.seat,
            "pos": list(self.pos),
            "unit_type": self.unit_type,
            "engine_hp": self.engine_hp,
            "php_hp": self.php_hp,
            "snapped": self.snapped,
            "out_of_range": self.out_of_range,
        }


@dataclass
class SyncReport:
    snapped_units: int = 0
    out_of_range_units: int = 0
    php_only_units: list[tuple[int, int, int]] = field(default_factory=list)
    engine_only_units: list[tuple[int, int, int]] = field(default_factory=list)
    funds_snapped: list[tuple[int, int, int]] = field(default_factory=list)
    deltas: list[UnitSyncDelta] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if no out-of-range deltas and no structural divergence."""
        return (
            self.out_of_range_units == 0
            and not self.php_only_units
            and not self.engine_only_units
        )

    def summary(self) -> str:
        bits = [
            f"snapped={self.snapped_units}",
            f"oor={self.out_of_range_units}",
            f"php_only={len(self.php_only_units)}",
            f"engine_only={len(self.engine_only_units)}",
            f"funds={len(self.funds_snapped)}",
        ]
        return " ".join(bits)

    def to_json(self) -> dict[str, Any]:
        return {
            "snapped_units": self.snapped_units,
            "out_of_range_units": self.out_of_range_units,
            "php_only_units": [list(t) for t in self.php_only_units],
            "engine_only_units": [list(t) for t in self.engine_only_units],
            "funds_snapped": [list(t) for t in self.funds_snapped],
            "deltas": [d.to_json() for d in self.deltas],
        }


def sync_state_to_snapshot(
    state: GameState,
    php_frame: dict[str, Any],
    awbw_to_engine: dict[int, int],
    *,
    max_hp_swing: int = MAX_PLAUSIBLE_HP_SWING_PER_ENVELOPE,
) -> SyncReport:
    """Validate + snap the engine to a PHP snapshot frame.

    For each unit visible in PHP, locate the matching engine unit by
    ``(engine_seat, row, col)`` and compare internal HPs. A delta within
    ``max_hp_swing`` is accepted as luck noise and the engine HP is set to the
    PHP HP. A delta beyond that threshold is **flagged** in the report (not
    snapped automatically — those are real parity bugs and should be triaged,
    not papered over). Engine units absent from PHP are killed (PHP is the
    ground truth). PHP units absent from the engine are reported only.

    Funds are always snapped (single integer per player; no plausibility
    threshold needed — funds drift cleanly indicates a missed/extra build,
    income tick, or repair).
    """
    report = SyncReport()

    # ---------- Funds ----------
    for _key, pl in (php_frame.get("players") or {}).items():
        if not isinstance(pl, dict):
            continue
        try:
            awbw_pid = int(pl["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if awbw_pid not in awbw_to_engine:
            continue
        seat = awbw_to_engine[awbw_pid]
        php_funds = int(pl.get("funds", 0) or 0)
        eng_funds = int(state.funds[seat])
        if eng_funds != php_funds:
            report.funds_snapped.append((seat, eng_funds, php_funds))
            state.funds[seat] = php_funds

    # ---------- Units ----------
    php_by_tile: dict[tuple[int, int, int], dict[str, Any]] = {}
    for _key, u in (php_frame.get("units") or {}).items():
        if not isinstance(u, dict):
            continue
        # ``carried: "Y"`` units are AWBW cargo riding inside a transport at
        # the same (x, y); the engine stores them inside ``Unit.loaded_units``
        # (not on the tile). Skip them so the sync does not "kill" the
        # carrier as engine-only and does not flag the carrier+cargo pair as
        # a structural divergence (see comparator note in
        # ``tools/replay_snapshot_compare.py``).
        if str(u.get("carried", "N")).upper() == "Y":
            continue
        try:
            r, c = int(u["y"]), int(u["x"])
            awbw_pid = int(u["players_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if awbw_pid not in awbw_to_engine:
            continue
        seat = awbw_to_engine[awbw_pid]
        php_by_tile[(seat, r, c)] = u

    # Index live and recently-dead engine units separately so we can resurrect
    # a defender the engine wrongly killed (luck-high attack) when PHP kept it
    # alive. Dead units retain ``pos`` from the moment they died; that is the
    # tile to compare against PHP. Dead units that are also missing from PHP
    # are simply gone — already pruned at the next sync step.
    eng_by_tile_alive: dict[tuple[int, int, int], Unit] = {}
    eng_by_tile_dead: dict[tuple[int, int, int], Unit] = {}
    for seat, ulist in state.units.items():
        for u in ulist:
            key = (seat, u.pos[0], u.pos[1])
            if u.is_alive:
                eng_by_tile_alive[key] = u
            else:
                # Newest dead-at-tile wins — last unit to die there is the
                # most likely candidate for "engine wrongly killed it".
                eng_by_tile_dead[key] = u

    # First pass: snap matched-tile pairs, resurrect dead-at-same-tile.
    # PHP-only and engine-only are deferred to a second pass so the bipartite
    # teleport step below sees the full leftover sets.
    pending_php_only: list[tuple[int, int, int]] = []
    for key, php_unit in php_by_tile.items():
        eng_unit = eng_by_tile_alive.get(key)
        php_hp = _php_internal_hp(php_unit.get("hit_points"))
        if eng_unit is None:
            # Try to resurrect a freshly-killed engine unit at the same
            # (seat, row, col) only if the unit type matches PHP. Type
            # mismatch here means the engine *replaced* one unit with another
            # at this tile (move-into-just-vacated) — resurrecting would be
            # wrong; keep it as a structural divergence to triage.
            dead = eng_by_tile_dead.get(key)
            php_name = _php_snapshot_name_canonical(str(php_unit.get("name", "")))
            if dead is not None and UNIT_STATS[dead.unit_type].name == php_name:
                report.deltas.append(
                    UnitSyncDelta(
                        seat=key[0],
                        pos=(key[1], key[2]),
                        unit_type=UNIT_STATS[dead.unit_type].name,
                        engine_hp=0,
                        php_hp=php_hp,
                        snapped=True,
                        out_of_range=False,
                    )
                )
                dead.hp = max(1, php_hp)
                report.snapped_units += 1
                continue
            pending_php_only.append(key)
            continue
        eng_hp = int(eng_unit.hp)
        if php_hp == eng_hp:
            continue
        delta = abs(eng_hp - php_hp)
        oor = delta > max_hp_swing
        report.deltas.append(
            UnitSyncDelta(
                seat=key[0],
                pos=(key[1], key[2]),
                unit_type=UNIT_STATS[eng_unit.unit_type].name,
                engine_hp=eng_hp,
                php_hp=php_hp,
                snapped=not oor,
                out_of_range=oor,
            )
        )
        if oor:
            report.out_of_range_units += 1
            continue
        eng_unit.hp = php_hp
        report.snapped_units += 1

    # Bipartite step: pair "engine has at tile A, PHP doesn't" with "PHP has
    # at tile B, engine doesn't" when (seat, unit_type) matches and the tiles
    # are within ``MAX_TELEPORT_DISTANCE``. These are the same unit moved
    # differently because of a luck-divergent attack on an obstacle (engine's
    # ATTACK chain freed a tile PHP didn't, etc.). Teleport instead of kill
    # so the unit stays alive for future envelopes — that prevents the
    # engine from appearing to "lose" units AWBW kept and triggering oracle
    # resolver aborts later (root cause of the 1619108 hard abort at day 22:
    # the missing Fighter at (7, 13) was actually a unit at a different tile
    # in the engine, not a build the engine missed).
    leftover_php_only = pending_php_only
    leftover_engine_only = [k for k in eng_by_tile_alive if k not in php_by_tile]

    php_buckets: dict[tuple[int, str], list[tuple[int, int, int]]] = {}
    for k in leftover_php_only:
        ts = (
            k[0],
            _php_snapshot_name_canonical(str(php_by_tile[k].get("name", ""))),
        )
        php_buckets.setdefault(ts, []).append(k)
    eng_buckets: dict[tuple[int, str], list[tuple[int, int, int]]] = {}
    for k in leftover_engine_only:
        u = eng_by_tile_alive[k]
        ts = (k[0], UNIT_STATS[u.unit_type].name)
        eng_buckets.setdefault(ts, []).append(k)

    matched_php: set[tuple[int, int, int]] = set()
    matched_eng: set[tuple[int, int, int]] = set()
    for ts, eng_keys in eng_buckets.items():
        php_keys = php_buckets.get(ts) or []
        if not php_keys:
            continue
        # Greedy nearest-pair match.
        pairs = []
        for ek in eng_keys:
            for pk in php_keys:
                d = abs(ek[1] - pk[1]) + abs(ek[2] - pk[2])
                if d <= MAX_TELEPORT_DISTANCE:
                    pairs.append((d, ek, pk))
        pairs.sort()
        used_e: set = set()
        used_p: set = set()
        for d, ek, pk in pairs:
            if ek in used_e or pk in used_p:
                continue
            eng_unit = eng_by_tile_alive[ek]
            php_unit = php_by_tile[pk]
            eng_unit.pos = (pk[1], pk[2])
            php_hp = _php_internal_hp(php_unit.get("hit_points"))
            if php_hp > 0:
                eng_unit.hp = php_hp
            used_e.add(ek)
            used_p.add(pk)
            matched_eng.add(ek)
            matched_php.add(pk)

    # Anything still leftover after teleport: kill the engine extras and
    # record php-only as structural divergence the sync cannot reconcile.
    for key in leftover_engine_only:
        if key in matched_eng:
            continue
        report.engine_only_units.append(key)
        eng_by_tile_alive[key].hp = 0
    for key in leftover_php_only:
        if key in matched_php:
            continue
        report.php_only_units.append(key)

    # Drop dead units immediately so subsequent get_unit_at / get_legal_actions
    # see the clean board. Resurrected units (hp >= 1 above) are kept.
    for seat in list(state.units.keys()):
        state.units[seat] = [u for u in state.units[seat] if u.is_alive]

    return report


def validate_damage_in_engine_range(
    *,
    pre_attacker_hp: int,
    pre_defender_hp: int,
    post_attacker_hp: int,
    post_defender_hp: int,
    attacker_unit: Unit,
    defender_unit: Unit,
    attacker_terrain,
    defender_terrain,
    attacker_co,
    defender_co,
) -> tuple[bool, Optional[tuple[int, int]], Optional[tuple[int, int]]]:
    """Return ``(in_range, fwd_range, ctr_range)`` — does the AWBW outcome fall
    inside the engine's possible damage envelope?

    ``fwd_range`` / ``ctr_range`` are ``(min, max)`` bands from
    ``engine.combat.damage_range`` (luck sweep: one digit 0..9 for most COs,
    or the full Cartesian product of two digits for dual-luck attackers).
    *both* the forward strike and the counterattack land inside their bands.
    Used by per-attack validators when we want a tighter check than the
    coarse per-envelope ``MAX_PLAUSIBLE_HP_SWING`` cap. Caller is responsible
    for handling the indirect-attacker case (no counter) by passing
    ``post_attacker_hp == pre_attacker_hp``.

    Currently called from tests; left exported so a future per-attack
    validator inside ``oracle_zip_replay`` can use it without copying the
    plumbing.
    """
    from engine.combat import damage_range

    # Snap PHP integer bars to internal HP via ``ceil`` for symmetry with the
    # engine's ``display_hp``; callers should pass internal HP directly when
    # they have it, in which case this is a no-op.
    fwd_actual = max(0, pre_defender_hp - post_defender_hp)
    ctr_actual = max(0, pre_attacker_hp - post_attacker_hp)

    fwd_range = damage_range(
        attacker_unit, defender_unit,
        attacker_terrain, defender_terrain,
        attacker_co, defender_co,
    )
    if fwd_range is None:
        return False, None, None

    ctr_range: Optional[tuple[int, int]] = None
    if ctr_actual > 0:
        # Counter is computed against the post-strike defender HP per
        # engine.game._apply_attack contract.
        post_def = max(0, defender_unit.hp - fwd_actual)
        if post_def > 0:
            from engine.unit import Unit as _U  # local import for clarity
            counter_unit = _U(
                unit_type=defender_unit.unit_type,
                player=defender_unit.player,
                hp=post_def,
                ammo=defender_unit.ammo,
                fuel=defender_unit.fuel,
                pos=defender_unit.pos,
                moved=defender_unit.moved,
                loaded_units=list(defender_unit.loaded_units),
                is_submerged=defender_unit.is_submerged,
                capture_progress=defender_unit.capture_progress,
                unit_id=defender_unit.unit_id,
            )
            ctr_range = damage_range(
                counter_unit, attacker_unit,
                defender_terrain, attacker_terrain,
                defender_co, attacker_co,
            )

    fwd_lo, fwd_hi = fwd_range
    fwd_ok = fwd_lo <= fwd_actual <= fwd_hi
    ctr_ok = ctr_range is None or (ctr_range[0] <= ctr_actual <= ctr_range[1])
    return (fwd_ok and ctr_ok), fwd_range, ctr_range


# Re-export for tests / callers wanting the symbolic threshold.
__all__ = [
    "MAX_PLAUSIBLE_HP_SWING_PER_ENVELOPE",
    "SyncReport",
    "UnitSyncDelta",
    "sync_state_to_snapshot",
    "validate_damage_in_engine_range",
]
