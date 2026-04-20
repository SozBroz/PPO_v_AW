"""Engine ⊂ AWBW legality probe.

Walks an in-progress ``GameState`` and validates that **every** action
returned by ``engine.action.get_legal_actions`` is also legal under AWBW
canonical rules. Permissive engine bugs (engine offers a move AWBW would
reject) surface as ``LegalityViolation`` records.

Why this exists
---------------
Replay-based validation already covers the AWBW → engine direction
(every recorded AWBW action must be reproducible by the engine). The
opposite direction — engine → AWBW — is much harder to test because we
have no AWBW reference implementation to query. This probe is the
practical compromise: instead of asking "would AWBW accept this exact
move?", it asks "does this move satisfy AWBW's documented invariants?".

Invariants checked
------------------
A1.  No ``BUILD`` is offered as a Stage-2 ACTION terminator.
B5.  No ``LOAD`` puts an incompatible cargo into a transport.
B4.  Every ``JOIN`` partners legal: same player, same type, neither carrying
     cargo, at least one of mover/partner below 100 HP.
G1.  Every ``CAPTURE`` targets an enemy- or neutral-owned property tile,
     and the unit's ``can_capture`` is True.
G2.  Every ``SELECT_UNIT`` (Stage-0) names a unit currently owned by the
     active player and not already moved.
G3.  Every Stage-1 SELECT_UNIT(move_pos=) has ``move_pos`` in the unit's
     ``compute_reachable_costs`` keyset and within fuel.
G4.  Every ``ATTACK`` target tile is within range of the attacker after
     the chosen move (range respects min/max + indirect-can't-move-and-fire),
     attacker has ammo (or uses 0-ammo machine-gun fallback), defender is
     enemy or pipe seam, defender visible (no fog modelled here, so always).
G5.  Every ``UNLOAD`` originates from a transport at ``move_pos`` carrying
     the named unit type; drop tile is empty (or the transport's pre-move
     tile) and passable for the cargo.
G6.  Every ``BUILD`` targets an empty owned base/airport/port; the unit
     type is producible there, and the player can afford the CO-adjusted cost.
G7.  Every ``REPAIR`` is issued by a ``BLACK_BOAT`` adjacent to an ally
     who can benefit (HP < 100 OR fuel/ammo below max).
G8.  Every ``DIVE_HIDE`` is issued by a ``can_dive`` unit (Sub / Stealth).

Runs
----
* ``probe-random N`` — play ``N`` random self-play turns from a fresh state
  (deterministic, seeded), invoking ``check_state`` before every action.
* ``probe-zip <zip>`` — replay a saved AWBW zip via ``oracle_zip_replay``
  and run ``check_state`` before each engine step (hooked through the
  engine-step pre-callback).

Exit code is non-zero iff at least one ``LegalityViolation`` is recorded.
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.action import (
    Action,
    ActionStage,
    ActionType,
    compute_reachable_costs,
    get_attack_targets,
    get_legal_actions,
    get_loadable_into,
    get_producible_units,
    units_can_join,
    _build_cost,
    _black_boat_repair_eligible,
)
from engine.game import GameState, make_initial_state
from engine.map_loader import load_map
from engine.terrain import (
    INF_PASSABLE,
    get_terrain,
)
from engine.unit import UNIT_STATS, UnitType
from engine.weather import effective_move_cost


# ---------------------------------------------------------------------------
# Violation record
# ---------------------------------------------------------------------------

@dataclass
class LegalityViolation:
    rule:    str
    action:  str
    detail:  str
    turn:    int
    player:  int
    stage:   str

    def __str__(self) -> str:  # pragma: no cover - human output
        return (
            f"[T{self.turn} P{self.player} {self.stage}] "
            f"{self.rule}: {self.action} — {self.detail}"
        )


# ---------------------------------------------------------------------------
# Per-action AWBW canonical predicate
# ---------------------------------------------------------------------------

def _violations_for_action(state: GameState, a: Action) -> Iterable[LegalityViolation]:
    """Yield violations the action triggers under AWBW canonical rules."""
    p = state.active_player
    t = int(state.turn)
    stage = state.action_stage.name

    def _v(rule: str, detail: str) -> LegalityViolation:
        return LegalityViolation(rule, repr(a), detail, t, p, stage)

    # ----- Stage 0 -----
    if state.action_stage == ActionStage.SELECT:
        if a.action_type == ActionType.SELECT_UNIT:
            u = state.get_unit_at(*a.unit_pos) if a.unit_pos else None
            if u is None:
                yield _v("G2", "SELECT_UNIT on empty tile")
            elif u.player != p:
                yield _v("G2", f"SELECT_UNIT on enemy unit (owner={u.player})")
            elif u.moved:
                yield _v("G2", "SELECT_UNIT on already-moved unit")

        elif a.action_type == ActionType.BUILD:
            # Stage-0 BUILD: factory direct.
            if a.unit_pos is not None:
                yield _v("A1", "BUILD with unit_pos set in SELECT stage")
            if a.move_pos is None or a.unit_type is None:
                yield _v("G6", "BUILD missing move_pos or unit_type")
            else:
                r, c = a.move_pos
                terr = get_terrain(state.map_data.terrain[r][c])
                prop = state.get_property_at(r, c)
                if prop is None or prop.owner != p:
                    yield _v("G6", f"BUILD on tile not owned by P{p}")
                elif not (terr.is_base or terr.is_airport or terr.is_port):
                    yield _v("G6", "BUILD on non-factory terrain")
                elif state.get_unit_at(r, c) is not None:
                    yield _v("G6", "BUILD on occupied factory")
                else:
                    if a.unit_type not in get_producible_units(terr, state.map_data.unit_bans):
                        yield _v("G6", f"BUILD {a.unit_type.name} not producible here")
                    cost = _build_cost(a.unit_type, state, p, a.move_pos)
                    if state.funds[p] < cost:
                        yield _v("G6", f"BUILD unaffordable (need {cost}, have {state.funds[p]})")

    # ----- Stage 1 -----
    elif state.action_stage == ActionStage.MOVE:
        if a.action_type == ActionType.SELECT_UNIT:
            u = state.selected_unit
            if u is None:
                yield _v("G3", "MOVE-stage SELECT_UNIT with no selected_unit")
            elif a.move_pos is None:
                yield _v("G3", "MOVE-stage missing move_pos")
            else:
                reach = compute_reachable_costs(state, u)
                if a.move_pos not in reach:
                    yield _v(
                        "G3",
                        f"move_pos={a.move_pos} not in reachable set for {u.unit_type.name}",
                    )

    # ----- Stage 2 -----
    elif state.action_stage == ActionStage.ACTION:
        if a.action_type == ActionType.BUILD:
            yield _v("A1", "BUILD leaked into Stage-2 ACTION")
            return

        u = state.selected_unit
        mp = state.selected_move_pos
        if u is None or mp is None:
            yield _v("G3", "ACTION-stage with no selected_unit / selected_move_pos")
            return
        stats = UNIT_STATS[u.unit_type]

        if a.action_type == ActionType.LOAD:
            tr = state.get_unit_at(*mp)
            if tr is None or tr is u or tr.player != p:
                yield _v("B5", "LOAD target not a friendly transport")
            else:
                cap = UNIT_STATS[tr.unit_type].carry_capacity
                if cap <= 0:
                    yield _v("B5", f"LOAD into non-transport {tr.unit_type.name}")
                elif u.unit_type not in get_loadable_into(tr.unit_type):
                    yield _v(
                        "B5",
                        f"LOAD {u.unit_type.name} into {tr.unit_type.name} forbidden by AWBW table",
                    )
                elif len(tr.loaded_units) >= cap:
                    yield _v("B5", f"LOAD into full transport ({len(tr.loaded_units)}/{cap})")

        elif a.action_type == ActionType.JOIN:
            partner = state.get_unit_at(*mp)
            if partner is None or partner is u:
                yield _v("B4", "JOIN target empty or self")
            elif not units_can_join(u, partner):
                # Mirror the canonical predicate; if the engine offers a JOIN
                # ``units_can_join`` rejects, that's a permissive bug elsewhere.
                yield _v("B4", "JOIN partner fails units_can_join predicate")

        elif a.action_type == ActionType.CAPTURE:
            if not stats.can_capture:
                yield _v("G1", f"{u.unit_type.name} cannot capture")
            else:
                terr = get_terrain(state.map_data.terrain[mp[0]][mp[1]])
                prop = state.get_property_at(*mp)
                if prop is None or not terr.is_property:
                    yield _v("G1", "CAPTURE on non-property tile")
                elif prop.owner == p:
                    yield _v("G1", f"CAPTURE on own property (owner=P{p})")

        elif a.action_type == ActionType.ATTACK:
            tgt = a.target_pos
            if tgt is None:
                yield _v("G4", "ATTACK missing target_pos")
            else:
                # Re-derive legal target set; engine offering target outside
                # this set is the canonical permissive bug.
                legal_targets = set(get_attack_targets(state, u, mp))
                if tgt not in legal_targets:
                    yield _v(
                        "G4",
                        f"ATTACK target {tgt} not in get_attack_targets({u.unit_type.name}, mp={mp})",
                    )

        elif a.action_type == ActionType.UNLOAD:
            if stats.carry_capacity <= 0 or not u.loaded_units:
                yield _v("G5", "UNLOAD from non-transport / empty transport")
            else:
                cargo_match = [c for c in u.loaded_units if c.unit_type == a.unit_type]
                if not cargo_match:
                    yield _v("G5", f"UNLOAD cargo type {a.unit_type} not aboard")
                if a.target_pos is None:
                    yield _v("G5", "UNLOAD missing target_pos")
                else:
                    tr, tc = a.target_pos
                    if not (0 <= tr < state.map_data.height and 0 <= tc < state.map_data.width):
                        yield _v("G5", f"UNLOAD target out of bounds: {a.target_pos}")
                    else:
                        drop_occ = state.get_unit_at(tr, tc)
                        if drop_occ is not None and drop_occ.pos != u.pos:
                            yield _v("G5", "UNLOAD onto occupied tile (not transport's source)")
                        if cargo_match:
                            cm0 = cargo_match[0]
                            tid = state.map_data.terrain[tr][tc]
                            if effective_move_cost(state, cm0, tid) >= INF_PASSABLE:
                                yield _v(
                                    "G5",
                                    f"UNLOAD onto impassable tile (terrain id {tid}) for {cm0.unit_type.name}",
                                )

        elif a.action_type == ActionType.REPAIR:
            if u.unit_type != UnitType.BLACK_BOAT:
                yield _v("G7", f"REPAIR by non-Black-Boat {u.unit_type.name}")
            elif a.target_pos is None:
                yield _v("G7", "REPAIR missing target_pos")
            else:
                tr, tc = a.target_pos
                if abs(tr - mp[0]) + abs(tc - mp[1]) != 1:
                    yield _v("G7", "REPAIR target not orthogonally adjacent")
                ally = state.get_unit_at(tr, tc)
                if ally is None or ally.player != p or ally is u:
                    yield _v("G7", "REPAIR target is not a friendly ally")
                elif not _black_boat_repair_eligible(state, ally):
                    yield _v("G7", "REPAIR target ineligible (full HP/fuel/ammo)")

        elif a.action_type == ActionType.DIVE_HIDE:
            if not stats.can_dive:
                yield _v("G8", f"DIVE_HIDE by non-diver {u.unit_type.name}")

        elif a.action_type == ActionType.WAIT:
            # WAIT has no further AWBW preconditions beyond reachability,
            # which Stage-1 already validated.
            pass


# ---------------------------------------------------------------------------
# State-level driver
# ---------------------------------------------------------------------------

def check_state(state: GameState, sink: list[LegalityViolation]) -> None:
    """Append every legality violation found in ``state``'s legal actions."""
    legal = get_legal_actions(state)
    for a in legal:
        for v in _violations_for_action(state, a):
            sink.append(v)


# ---------------------------------------------------------------------------
# Random self-play probe
# ---------------------------------------------------------------------------

def probe_random(
    map_id: int,
    seed: int = 0,
    turns: int = 200,
    co0: int = 1,
    co1: int = 1,
    map_pool: Path = ROOT / "data" / "gl_map_pool.json",
    maps_dir: Path = ROOT / "data" / "maps",
    verbose: bool = False,
) -> list[LegalityViolation]:
    rng = random.Random(seed)
    random.seed(seed)  # combat luck rolls also use module RNG
    m = load_map(map_id, map_pool, maps_dir)
    state = make_initial_state(m, co0, co1, tier_name="T2")

    violations: list[LegalityViolation] = []
    steps = 0
    max_steps = turns * 200
    while not state.done and steps < max_steps:
        check_state(state, violations)
        legal = get_legal_actions(state)
        if not legal:
            # Should never happen; engine always emits END_TURN at minimum.
            break
        a = rng.choice(legal)
        if verbose:
            print(f"step {steps}: {a}")
        try:
            state.step(a)
        except Exception as exc:
            violations.append(LegalityViolation(
                rule="STEP_RAISED",
                action=repr(a),
                detail=f"{type(exc).__name__}: {exc}",
                turn=int(state.turn),
                player=int(state.active_player),
                stage=state.action_stage.name,
            ))
            break
        steps += 1
        if state.turn > turns:
            break
    return violations


# ---------------------------------------------------------------------------
# Replay-zip probe (uses oracle replay, hooks pre-step)
# ---------------------------------------------------------------------------

def probe_zip(
    zip_path: Path,
    map_id: int,
    co0: int,
    co1: int,
    tier_name: str = "T2",
    map_pool: Path = ROOT / "data" / "gl_map_pool.json",
    maps_dir: Path = ROOT / "data" / "maps",
    seed: int = 0,
) -> list[LegalityViolation]:
    from tools.oracle_zip_replay import replay_oracle_zip

    random.seed(seed)
    violations: list[LegalityViolation] = []

    def _hook(state: GameState, action: Action) -> None:
        # Validate the *engine's* legal_actions before each step. We do not
        # validate ``action`` itself because oracle replay constructs actions
        # the engine's mask does not necessarily expose (oracle workarounds).
        check_state(state, violations)

    replay_oracle_zip(
        zip_path,
        map_pool=map_pool,
        maps_dir=maps_dir,
        map_id=map_id,
        co0=co0,
        co1=co1,
        tier_name=tier_name,
        before_engine_step=_hook,
    )
    return violations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(violations: list[LegalityViolation]) -> int:
    if not violations:
        print("OK — no legality violations.")
        return 0
    by_rule: dict[str, int] = {}
    for v in violations:
        by_rule[v.rule] = by_rule.get(v.rule, 0) + 1
    print(f"FAIL — {len(violations)} violations:")
    for rule, n in sorted(by_rule.items(), key=lambda x: -x[1]):
        print(f"  {rule:>8}: {n}")
    print("\nFirst 20:")
    for v in violations[:20]:
        print("  " + str(v))
    return 1


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Engine ⊂ AWBW legality probe.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("random", help="Random self-play probe.")
    rp.add_argument("--map-id", type=int, default=123858)
    rp.add_argument("--seed", type=int, default=0)
    rp.add_argument("--turns", type=int, default=50)
    rp.add_argument("--co0", type=int, default=1)
    rp.add_argument("--co1", type=int, default=1)
    rp.add_argument("--verbose", action="store_true")

    zp = sub.add_parser("zip", help="Replay-zip probe.")
    zp.add_argument("zip_path", type=Path)
    zp.add_argument("--map-id", type=int, required=True)
    zp.add_argument("--co0", type=int, required=True)
    zp.add_argument("--co1", type=int, required=True)
    zp.add_argument("--tier", default="T2")
    zp.add_argument("--seed", type=int, default=0)

    args = ap.parse_args(argv)

    if args.cmd == "random":
        v = probe_random(
            map_id=args.map_id,
            seed=args.seed,
            turns=args.turns,
            co0=args.co0,
            co1=args.co1,
            verbose=args.verbose,
        )
    elif args.cmd == "zip":
        v = probe_zip(
            zip_path=args.zip_path,
            map_id=args.map_id,
            co0=args.co0,
            co1=args.co1,
            tier_name=args.tier,
            seed=args.seed,
        )
    else:  # pragma: no cover
        ap.error(f"unknown cmd {args.cmd}")
        return 2
    return _print_summary(v)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
