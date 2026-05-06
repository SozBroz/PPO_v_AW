"""
AWBW GameState: complete mutable game state and transition logic.

step(action) → (state, reward, done)
  reward: +1.0 (active player wins) | -1.0 (active player loses) | 0.0 (ongoing/draw)
"""
from __future__ import annotations

import copy as _copy_mod
import random as _random_mod
import sys
from dataclasses import dataclass, field
from collections import Counter
from typing import Callable, Optional, Union

from engine.unit import Unit, UnitType, UNIT_STATS, idle_start_of_day_fuel_drain
from engine.unit_cap import alive_owned_unit_count
from engine.terrain import get_terrain, property_terrain_id_after_owner_change, TERRAIN_TABLE
from engine.co import COState, make_co_state_safe
from engine.map_loader import MapData, PropertyState
from engine.predeployed import specs_to_initial_units
from engine.action import (
    Action, ActionType, ActionStage,
    get_legal_actions, get_producible_units, _build_cost,
    get_reachable_tiles, compute_reachable_costs, get_loadable_into,
    get_attack_targets,
    units_can_join,
)
from engine.combat import (
    calculate_damage, calculate_counterattack, calculate_seam_damage,
)
from engine.spirit_pressure import SpiritState, maybe_spirit_after_end_turn

# Pipe seam constants (AWBW canonical).
SEAM_TERRAIN_IDS: tuple[int, int] = (113, 114)       # HPipe Seam / VPipe Seam
SEAM_BROKEN_IDS:  dict[int, int]  = {113: 115, 114: 116}  # → HPipe Rubble / VPipe Rubble
SEAM_MAX_HP: int = 99

MAX_TURNS = 100   # after this, winner = player with more properties; tie if equal


def _rounded_div_half_up(n: int, d: int) -> int:
    """ nonnegative n,d>0: round n/d to nearest int, half-up."""
    if d <= 0:
        raise ValueError("d must be positive")
    return (n + d // 2) // d


# AWBW Wiki: https://awbw.fandom.com/wiki/Machine_Gun
# Units listed below carry both a primary weapon (cannon / bazooka / missile)
# and a secondary Machine Gun. The MG is unlimited (no ammo cost) and is
# AWBW's default weapon when the defender is Infantry or Mech.
# Used by ``_apply_attack`` to decide whether the strike consumes one round
# of primary ammo.
_MG_SECONDARY_USERS: frozenset[UnitType] = frozenset({
    UnitType.MECH,
    UnitType.TANK,
    UnitType.MED_TANK,
    UnitType.NEO_TANK,
    UnitType.MEGA_TANK,
    UnitType.B_COPTER,
})
_MG_SECONDARY_TARGETS: frozenset[UnitType] = frozenset({
    UnitType.INFANTRY,
    UnitType.MECH,
})


class IllegalActionError(ValueError):
    """Raised by ``GameState.step`` when an action is not in
    ``get_legal_actions(state)`` and the call did not opt into ``oracle_mode``.

    STEP-GATE invariant — see
    ``.cursor/plans/desync_purge_engine_harden_d85bd82c.plan.md`` Phase 3
    Thread STEP-GATE and ``docs/oracle_exception_audit/phase3_step_gate.md``.
    """
    pass

# Dense shaping for CAPTURE (acting-player frame). Kept small vs terminal ±1.0.
# Coefficients raised after the P0-skew investigation (plan p0-capture-architecture-fix):
# the previous values were dominated by the per-step property-diff penalty in
# rl/env.py, leaving the learner with no positive discovery signal.
#
# NOTE (plan rl_capture-combat_recalibration): when env-var
# AWBW_REWARD_SHAPING=phi is set, ``_apply_capture`` returns 0.0 and these
# constants are ignored — the env's potential-based shaping (Φ_cap term)
# subsumes them and adds a free refund-on-reset via Φ telescoping.
_CAPTURE_SHAPING_PROGRESS: float = 0.04  # per 20 capture_points reduced toward flip
_CAPTURE_SHAPING_COMPLETE: float = 0.20  # bonus when ownership flips to capturer
# One-shot bonus the first time a given unit attempts CAPTURE in an episode.
# Rewards the *behavior* (issuing CAPTURE) so the SELECT->MOVE->CAPTURE chain
# gets a positive credit-assignment signal even before the tile flips.
_CAPTURE_FIRST_ATTEMPT_BONUS: float = 0.01

# Read-once gate so engine-side capture shaping can be disabled when the env
# uses potential-based shaping. Module-level read keeps SubprocVecEnv workers
# consistent (each child process inherits the value at import time).
import os as _os
_PHI_SHAPING_ACTIVE: bool = (
    _os.environ.get("AWBW_REWARD_SHAPING", "level").strip().lower() == "phi"
)

# Sami (CO 8): AWBW footsoldier capture points per capture action by display HP.
# https://awbw.fandom.com/wiki/Sami — COP/SCOP do not stack extra capture beyond D2D
# except Victory March (instant), handled separately.
_SAMI_AW_CAPTURE_D2D: dict[int, int] = {
    10: 15, 9: 13, 8: 12, 7: 10, 6: 9, 5: 7, 4: 6, 3: 4, 2: 3, 1: 1,
}


def _property_day_repair_gold(internal_heal: int, unit_type: UnitType) -> int:
    """Gold for owned-tile day repair: 20% of deployment cost per +20 internal HP (+2 bars).

    Scales linearly for partial heals (e.g. only +10 internal to reach max HP → 10% of cost).
    """
    if internal_heal <= 0:
        return 0
    listed = UNIT_STATS[unit_type].cost
    if listed <= 0:
        return 0
    return max(1, (internal_heal * listed) // 100)


# ---------------------------------------------------------------------------
# GameState
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    map_data:          MapData
    units:             dict[int, list[Unit]]   # player → units
    funds:             list[int]               # [p0, p1]
    co_states:         list[COState]           # [p0_co, p1_co]
    properties:        list[PropertyState]     # mutable property list
    turn:              int                     # 1-indexed; increments after P1 ends turn
    active_player:     int                     # 0 or 1
    action_stage:      ActionStage
    selected_unit:     Optional[Unit]
    selected_move_pos: Optional[tuple[int, int]]
    done:              bool
    winner:            Optional[int]           # 0, 1, or -1 (draw)
    win_reason:        Optional[str]           # e.g. hq_capture; max_days_draw; max_days_tiebreak
    game_log:          list[dict]              # append-only action history (resolved actions only)
    tier_name:         str
    max_turns:         int = field(default=MAX_TURNS)  # calendar day cap (end-inclusive; see _end_turn)
    full_trace:        list[dict] = field(default_factory=list)  # every action incl. SELECT/END_TURN

    # Luck rolls for combat (0–9) must use this RNG, not the module-level ``random``
    # module, so parallel games / subprocesses cannot steal each other's sequence
    # (and ``ai_vs_ai --seed`` reproduces full games including combat).
    luck_rng:          _random_mod.Random = field(default_factory=_random_mod.Random)

    # Economic tracking (Phase A logging requirements)
    gold_spent:        list[int] = field(default_factory=lambda: [0, 0])  # [p0, p1] cumulative
    losses_hp:         list[int] = field(default_factory=lambda: [0, 0])  # [p0, p1] HP lost
    losses_units:      list[int] = field(default_factory=lambda: [0, 0])  # [p0, p1] units destroyed

    # Monotonic unit id allocator. Every new Unit gets the next value and keeps
    # it for life — required for stable identity in AWBW replay exports.
    next_unit_id:      int = 1

    # Remaining hit points per intact pipe seam tile, keyed by (row, col).
    # Seams start at SEAM_MAX_HP (99); attacks chip it down. When HP hits 0
    # the seam terrain flips to Broken Seam (115/116) in ``map_data.terrain``
    # and the entry is removed. See ``_apply_attack`` seam branch.
    seam_hp:           dict[tuple[int, int], int] = field(default_factory=dict)

    # Weather state.  Supported values: "clear", "rain", "snow".
    # ``default_weather`` is the baseline restored once a CO power expires.
    # ``co_weather_segments_remaining`` counts _end_turn calls until expiry;
    # 0 means no active CO override.  Each CO power that triggers weather sets
    # this to 2 (activator's remaining turn + opponent's full turn = 1 AW day).
    weather:                        str = "clear"
    default_weather:                str = "clear"
    co_weather_segments_remaining:  int = 0

    # Per-episode set of unit ids that have already attempted CAPTURE at least
    # once. Used by `_apply_capture` to grant a one-shot first-attempt bonus
    # (acting-player frame) so the multi-stage SELECT -> MOVE -> CAPTURE chain
    # has a positive credit signal even when the tile does not flip on the
    # first attempt. Not persisted across episodes; reset implicitly via
    # `make_initial_state`.
    capture_attempted_unit_ids: set[int] = field(default_factory=set)

    # Optional spirit-broken heuristic (``AWBW_SPIRIT_BROKEN``); see ``engine/spirit_pressure``.
    spirit: SpiritState = field(default_factory=SpiritState)
    # When ``require_std_map`` is on, must be ``True`` for spirit to run. ``AWBWEnv`` sets from pool.
    spirit_map_is_std: Optional[bool] = None

    # Oracle replay channel: when set to ``(dmg, counter)`` immediately before
    # ``step(ActionType.ATTACK)``, ``_apply_attack`` consumes these instead of
    # rolling ``calculate_damage`` / ``calculate_counterattack`` (which use a
    # random luck roll, see ``engine/combat.py``). The oracle computes them
    # from AWBW ``combatInfoVision.attacker.units_hit_points`` and
    # ``defender.units_hit_points``, which are the **post-strike** display HP
    # AWBW actually rolled. Snapping engine HP to AWBW values here removes
    # the dominant non-determinism in audit / replay (game state diverged
    # within ~3 turns from luck alone, cascading into the Capt / Move / Fire
    # "no unit" drift cluster). The override is one-shot: cleared by
    # ``_apply_attack`` after consumption so a normal RL ``step`` always rolls
    # luck. Internal HP scale is 0–100; ``None`` slots fall back to the
    # rolled value (e.g. seam attacks have no defender unit).
    _oracle_combat_damage_override: Optional[tuple[Optional[int], Optional[int]]] = None

    # Phase 11J-MISSILE-AOE-CANON — Von Bolt SCOP "Ex Machina" AOE override.
    #
    # Tier 1 — AWBW CO Chart ``https://awbw.amarriner.com/co.php`` (Von Bolt):
    # *"A 2-range missile deals 3 HP damage and prevents all affected units
    # from acting next turn."*
    #
    # Tier 2 — AWBW Fandom Wiki Interface Guide (Missile Silos): 2-range blast =
    # Manhattan distance ≤ 2 from center, 13 tiles (5-wide diamond).
    #
    # Tier 3 — PHP gid 1622328 env 28 ``unitReplace``: seven damaged enemies all
    # at Manhattan ≤ 2 from ``missileCoords`` at pre-strike positions; oracle
    # pin in ``tools/oracle_zip_replay.py`` expands the center into that diamond.
    # Pre-fix engine subtracted 30 HP from every enemy globally (see
    # ``phase11j_funds_deep.md`` / ``phase11j_cluster_b_ship.md``).
    #
    # The Power action JSON itself carries ``missileCoords: [{x, y}]`` —
    # the AOE center — so the oracle can pin the affected tile set without
    # any external scrape. ``_oracle_power_aoe_positions`` is the set of
    # engine ``(row, col)`` tiles that should take the 30 HP loss; when
    # ``None`` the engine falls back to its prior global behavior so the RL
    # / non-oracle path is unchanged. One-shot: cleared by
    # ``_apply_power_effects`` after consumption.
    #
    # Phase 11J-RACHEL-SCOP-COVERING-FIRE-SHIP — type widened to also accept
    # a ``Counter[(row, col)] -> hit_count`` so Rachel's 3-missile SCOP can
    # encode per-tile multiplicity (overlapping missiles stack their 30 HP
    # damage). Von Bolt's branch continues to pin a plain ``set`` (1
    # missile, multiplicity always 1) and its ``u.pos in aoe`` consumer
    # works identically on both ``set`` and ``Counter``.
    _oracle_power_aoe_positions: Optional[
        Union[set[tuple[int, int]], Counter]
    ] = None

    # Phase 11J-CLOSE-1624082 — oracle War Bonds payout pin.
    #
    # AWBW canon (Tier 1, AWBW CO Chart Sasha row,
    # https://awbw.amarriner.com/co.php):
    #   *"War Bonds — Returns 50% of damage dealt as funds (subject to a 9HP
    #   cap)."*
    #
    # PHP emits the per-Fire War Bonds payout in the same combatInfo block
    # that pins post-strike HP — ``combatInfoVision.global.combatInfo
    # .gainedFunds`` is a dict keyed by AWBW player id, value = payout for
    # that player from this strike (None when the player gained nothing).
    # Sasha's primary-attack credit lives under the activator's id; her
    # counter-attack credit lives under the defender's id when she is
    # defending.
    #
    # Why this override exists: the engine computes War Bonds from
    # ``display_hp`` deltas of the engine board's defender / attacker. When
    # state-mismatch leaves an engine unit at a different pre-strike HP
    # than PHP's matching unit (combat-info HP override pulls them to the
    # *same* post-HP, but pre-HP can still differ), the engine's WB credit
    # diverges from PHP by ``(display_pre_engine - display_pre_PHP) *
    # cost(target) // 20`` per fire. Game ``1624082`` env 33 (Sasha
    # day-17) is the empirical anchor: 4 of 14 Sasha primary fires diverge
    # on pre-HP, total drift −500 g, surviving 150 g shortfall on a
    # NEO_TANK build at (13,3) after Sasha's intermediate ANTI_AIR build
    # consumes 8 000 g. Pinning the WB payout from PHP's ``gainedFunds``
    # field makes the engine's treasury match PHP's even under HP
    # state-mismatch, the same way ``_oracle_combat_damage_override``
    # makes the engine's post-HP match PHP's even under random-roll
    # divergence.
    #
    # Format: ``dict[engine_player_id, payout_gold]``. Each
    # ``_apply_war_bonds_payout`` call pops its dealer entry; ``_apply_attack``
    # clears the field after the primary + counter pair (one-shot per Fire,
    # mirrors ``_oracle_combat_damage_override``). ``None`` means "no
    # oracle pin" and the engine falls back to the formula-based payout
    # — the RL / non-oracle path is unchanged.
    _oracle_war_bonds_payout_override: Optional[dict[int, int]] = None

    # Phase 11K-FIRE-FRAC-COUNTER-SHIP — fractional-HP counter-damage pin.
    #
    # AWBW canon (Wars Wiki Damage_Formula and AWBW Fandom Damage Formula):
    # combat damage is computed as a FLOAT and the defender's internal HP
    # is reduced by that float. Display HP is ``ceil(internal_hp / 10)`` so
    # sub-display-HP damage (1–9 internal HP) leaves the display unchanged.
    # The fire envelope's ``combatInfoVision.global.combatInfo
    # .{attacker,defender}.units_hit_points`` records only the integer
    # display HP, which silently rounds away every counter ≤ 9 internal
    # HP. ``_oracle_set_combat_damage_override_from_combat_info``
    # historically used display × 10 → integer internal, so the engine
    # recorded counter = 0 every time the attacker's display didn't tick.
    # Cumulative drift on long-range / counter-rich replays (gid 1635679
    # env 10 RECON @ (3,4): engine kept hp=100 vs PHP hp=97, cascading
    # into +1600 g over-repair at env 25 day 13 and a missed 22 000 g
    # NEO_TANK build at env 32) is documented in
    # ``docs/oracle_exception_audit/phase11k_fire_frac_counter.md``.
    #
    # Recovery: AWBW snapshots after the envelope (``frames[env_i + 1]``)
    # carry the precise float ``hit_points``. When the audit / oracle
    # caller plumbs that into the state via this pin
    # (``dict[awbw_units_id, internal_hp_int]``, where
    # ``internal_hp_int = round(float_hit_points * 10)``), the override
    # consults the pin first for the attacker's post-counter HP, falling
    # back to the integer display × 10 only when the unit is missing
    # (post-fire dead, batched out, or pin not provided). The pin is
    # envelope-scoped (set/cleared by ``apply_oracle_action_json``'s caller)
    # and the RL / non-oracle path is unchanged — ``None`` means "no pin".
    _oracle_post_envelope_units_by_id: Optional[dict[int, int]] = None

    # Phase 11K-FIRE-FRAC-COUNTER-SHIP — defender side of the same pin.
    # The post-frame for a defender is unambiguous ONLY when this defender
    # is hit by exactly one ``Fire`` row in the envelope. When struck
    # multiple times (e.g., two Hawke counter-pieces stack on one Sturm
    # tile), the post-frame conflates all hits and per-fire ``combatInfo``
    # remains the only ground truth. The audit caller pre-scans the
    # envelope and stamps every multi-hit AWBW ``units_id`` here so the
    # override can opt out cleanly.
    _oracle_post_envelope_multi_hit_defenders: Optional[set[int]] = None

    def _allocate_unit_id(self) -> int:
        uid = self.next_unit_id
        self.next_unit_id += 1
        return uid

    @property
    def day(self) -> int:
        """1-based calendar day; alias of :attr:`turn` (full day = P0 + P1 player-turns)."""
        return self.turn

    @property
    def max_days(self) -> int:
        """End-inclusive calendar day limit; alias of :attr:`max_turns`."""
        return self.max_turns

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_unit_at(self, row: int, col: int) -> Optional[Unit]:
        for player_units in self.units.values():
            for u in player_units:
                if u.pos == (row, col) and u.is_alive:
                    return u
        return None

    def get_unit_at_oracle_id(
        self, row: int, col: int, select_unit_id: Optional[int]
    ) -> Optional[Unit]:
        """``get_unit_at`` unless ``select_unit_id`` pins a specific ``Unit.unit_id`` at tile."""
        if select_unit_id is None:
            return self.get_unit_at(row, col)
        want = int(select_unit_id)
        for player_units in self.units.values():
            for u in player_units:
                if (
                    u.is_alive
                    and u.pos == (row, col)
                    and int(u.unit_id) == want
                ):
                    return u
        return self.get_unit_at(row, col)

    def get_property_at(self, row: int, col: int) -> Optional[PropertyState]:
        for prop in self.properties:
            if prop.row == row and prop.col == col:
                return prop
        return None

    def count_properties(self, player: int) -> int:
        return sum(1 for p in self.properties if p.owner == player)

    def count_income_properties(self, player: int) -> int:
        """Owned properties that produce per-turn funds. Comm towers and labs do not."""
        return sum(
            1 for p in self.properties
            if p.owner == player and not p.is_comm_tower and not p.is_lab
        )

    def _refresh_comm_towers(self) -> None:
        """Sync per-CO property counts from current ownership.

        Tracks both ``comm_towers`` (Javier) and ``urban_props`` (Kindle
        SCOP +3% ATK per owned urban property). "Urban" here = the same
        set listed on the AWBW CO Chart under Kindle's D2D: HQs, bases,
        airports, ports, cities, labs, comm towers. Every entry in
        ``self.properties`` is by construction one of those urban tiles,
        so a plain owner-filtered count gives the Kindle rider its input.
        """
        for player in (0, 1):
            co = self.co_states[player]
            co.comm_towers = sum(
                1 for p in self.properties if p.owner == player and p.is_comm_tower
            )
            co.urban_props = sum(
                1 for p in self.properties if p.owner == player
            )

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(
        self,
        action: Action,
        *,
        oracle_mode: bool = False,
        oracle_strict: bool = False,
    ) -> tuple[GameState, float, bool]:
        """
        Apply action in-place, return (self, reward, done).
        reward is from the perspective of active_player at the time of the call.

        STEP-GATE: when ``oracle_mode`` is False (the default — RL, agent, tests,
        any normal caller), ``action`` must appear in ``get_legal_actions(self)``.
        Violations raise :class:`IllegalActionError`. ``get_legal_actions`` is
        the single source of truth for legality; this gate makes ``step``
        enforce it. Oracle / replay tooling that intentionally crafts actions
        outside the mask (e.g. AWBW zip replay reconstructing a game from
        recorded action envelopes) opts out by passing ``oracle_mode=True``.

        ``oracle_strict`` (default False) threads an audit-only lane: when True,
        some ``_apply_*`` paths that would silently no-op under oracle replay
        raise :class:`IllegalActionError` instead, surfacing AWBW disagreements.
        Does not affect the default oracle zip consumer, which leaves it False.

        See ``.cursor/plans/desync_purge_engine_harden_d85bd82c.plan.md``
        Phase 3 Thread STEP-GATE and
        ``docs/oracle_exception_audit/phase3_step_gate.md``.
        """
        if not oracle_mode:
            legal = get_legal_actions(self)
            if action not in legal:
                raise IllegalActionError(
                    f"Action {action!r} not in get_legal_actions() at "
                    f"turn={self.turn} active_player={self.active_player} "
                    f"action_stage={self.action_stage.name}; "
                    f"mask size={len(legal)}"
                )

        acting_player = self.active_player

        # Full trace — record every action for replay export.
        # NOTE: use `is not None` comparisons. `UnitType.INFANTRY == 0` is
        # truthy-false under IntEnum, so `if action.unit_type` silently
        # dropped the type on every BUILD INFANTRY. Same concern for
        # unit_pos/move_pos which are tuples (empty tuples are falsy).
        self.full_trace.append({
            "type":   action.action_type.name,
            "player": acting_player,
            "turn":   self.turn,
            "stage":  self.action_stage.name,
            "unit_pos":   list(action.unit_pos)   if action.unit_pos   is not None else None,
            "move_pos":   list(action.move_pos)   if action.move_pos   is not None else None,
            "target_pos": list(action.target_pos) if action.target_pos is not None else None,
            "unit_type":  action.unit_type.name   if action.unit_type  is not None else None,
            "select_unit_id": int(action.select_unit_id)
            if action.select_unit_id is not None
            else None,
        })

        capture_shaping = 0.0

        if action.action_type == ActionType.RESIGN:
            # Active player forfeits; opponent wins. Used by oracle zip replay (AWBW ``Resign``).
            if not self.done:
                self.done = True
                self.winner = 1 - acting_player
                self.win_reason = "resign"
                self.selected_unit = None
                self.selected_move_pos = None
                self.action_stage = ActionStage.SELECT
                self.game_log.append({
                    "type": "resign",
                    "player": acting_player,
                })

        elif action.action_type == ActionType.END_TURN:
            self._end_turn()

        elif action.action_type == ActionType.ACTIVATE_COP:
            self._activate_power(cop=True)

        elif action.action_type == ActionType.ACTIVATE_SCOP:
            self._activate_power(cop=False)

        elif action.action_type == ActionType.SELECT_UNIT:
            if self.action_stage == ActionStage.SELECT:
                unit = self.get_unit_at_oracle_id(
                    *action.unit_pos, action.select_unit_id
                )
                self.selected_unit = unit
                self.action_stage  = ActionStage.MOVE
            elif self.action_stage == ActionStage.MOVE:
                self.selected_move_pos = action.move_pos
                self.action_stage      = ActionStage.ACTION

        elif action.action_type == ActionType.ATTACK:
            self._apply_attack(
                action,
                oracle_mode=oracle_mode,
                oracle_strict=oracle_strict,
            )

        elif action.action_type == ActionType.CAPTURE:
            capture_shaping = self._apply_capture(action)

        elif action.action_type == ActionType.WAIT:
            self._apply_wait(action, oracle_mode=oracle_mode)

        elif action.action_type == ActionType.DIVE_HIDE:
            self._apply_dive_hide(action)

        elif action.action_type == ActionType.LOAD:
            self._apply_load(
                action,
                oracle_strict=oracle_strict,
                oracle_mode=oracle_mode,
            )

        elif action.action_type == ActionType.JOIN:
            self._apply_join(action, oracle_strict=oracle_strict)

        elif action.action_type == ActionType.UNLOAD:
            self._apply_unload(action, oracle_strict=oracle_strict)

        elif action.action_type == ActionType.BUILD:
            self._apply_build(action, oracle_strict=oracle_strict)

        elif action.action_type == ActionType.REPAIR:
            self._apply_repair(action, oracle_strict=oracle_strict)

        reward = self._check_win_conditions(acting_player) + capture_shaping
        return self, reward, self.done

    # ------------------------------------------------------------------
    # End turn
    # ------------------------------------------------------------------

    def _end_turn(self):
        player   = self.active_player
        opponent = 1 - player

        # Deactivate any power that was active
        co = self.co_states[player]
        co.cop_active  = False
        co.scop_active = False

        # Phase 11J-VONBOLT-SCOP-SHIP — clear Von Bolt "Ex Machina" stun on
        # the units of the player whose turn just ended. AWBW canon: the
        # stun "prevents all affected enemy units from acting next turn"
        # (https://awbw.fandom.com/wiki/Von_Bolt). Ex Machina fires on
        # Player A's turn → stun set on Player B's units → Player B's next
        # turn (the one we are now ending) is the served turn → clearing
        # here returns Player B to normal for their next-next turn while
        # leaving any *fresh* stun applied later this same envelope chain
        # untouched (Ex Machina cannot fire during the opponent's turn —
        # SCOPs only fire on the activator's own turn).
        for u in self.units[player]:
            u.is_stunned = False

        # Phase 11J-SASHA-WARBONDS-SHIP: Sasha's SCOP "War Bonds" persists past
        # her own end-turn so her counter-attacks during the opponent's
        # intervening turn still accumulate. The accumulated payout is
        # credited to her treasury HERE — at the end of the opponent's
        # intervening turn, immediately before Sasha's next turn begins.
        # PHP defers the credit to the start-of-next-turn settlement (see
        # `pending_war_bonds_funds` docstring on COState for empirical
        # grounding from game `1624082`).
        opp_co = self.co_states[opponent]
        if opp_co.co_id == 19 and opp_co.war_bonds_active:
            payout = opp_co.pending_war_bonds_funds
            if payout > 0:
                self.funds[opponent] = min(
                    999_999, self.funds[opponent] + payout
                )
            opp_co.pending_war_bonds_funds = 0
            opp_co.war_bonds_active = False

        # Tick CO-induced weather.  Each end-turn consumes one segment; when the
        # counter reaches 0 the weather reverts to the map default.
        if self.co_weather_segments_remaining > 0:
            self.co_weather_segments_remaining -= 1
        # Defensive: always sync weather to default when segments exhausted,
        # even if the decrement happened elsewhere or the initial state was inconsistent.
        if self.co_weather_segments_remaining == 0:
            self.weather = self.default_weather

        # Reset capture progress for enemy units on opponent's properties
        # (AWBW rule: capture resets if unit moves off tile — handled in _apply_capture)

        self.active_player     = opponent
        self.action_stage      = ActionStage.SELECT
        self.selected_unit     = None
        self.selected_move_pos = None

        if opponent == 0:
            self.turn += 1
            if self.turn > self.max_turns:
                self.done = True
                p0_props = self.count_properties(0)
                p1_props = self.count_properties(1)
                if p0_props > p1_props:
                    self.winner = 0
                    self.win_reason = "max_days_tiebreak"
                elif p1_props > p0_props:
                    self.winner = 1
                    self.win_reason = "max_days_tiebreak"
                else:
                    self.winner = -1
                    self.win_reason = "max_days_draw"
                return

        # Start of opponent's turn: reset moved flags, consume idle fuel, crash units.
        # https://awbw.fandom.com/wiki/Units#Fuel (Sub dive / Stealth hide, Eagle air).
        # Naval / air units that hit 0 fuel sink/crash UNLESS they are sitting on
        # their own port (naval) or airport (air/copter): AWBW refuels those at
        # the start of the turn, so they survive even if last turn ended at 0
        # fuel. Skipping that exemption made Black Boats sink mid-replay while
        # AWBW kept them alive (see desync_audit ``move_no_unit`` cluster on
        # game 1629104 — fuel-starved Black Boat parked on its own port).
        # AWBW empirically skips per-turn idle fuel drain on units that *moved*
        # during their owner's previous turn — drain only applies to units that
        # spent the whole turn idle. Game 1631302 was the first traced case
        # (a P0 Lander shuttling cargo every day): with the universal drain the
        # engine sank the Lander on day 19 from fuel exhaustion, while AWBW
        # kept it alive at fuel=3 through day 21. The path-cost the unit pays
        # on its Move action effectively replaces the idle drain.
        opp_co_id = self.co_states[opponent].co_id
        for unit in list(self.units[opponent]):
            moved_previous_turn = unit.moved
            unit.moved = False
            stats = UNIT_STATS[unit.unit_type]
            drain = idle_start_of_day_fuel_drain(unit, opp_co_id)
            if moved_previous_turn and drain > 0:
                drain = 0
            unit.fuel = max(0, unit.fuel - drain)
            # AWBW: naval units on water (sea/shoal) get +1 fuel per day.
            # This happens at the start of the owner's turn.
            if stats.unit_class == "naval":
                tid = self.map_data.terrain[unit.pos[0]][unit.pos[1]]
                tinfo = TERRAIN_TABLE.get(tid)
                if tinfo and ("Sea" in tinfo.name or "Shoal" in tinfo.name):
                    unit.fuel = min(stats.max_fuel, unit.fuel + 1)
            if unit.fuel == 0 and stats.unit_class in ("air", "copter", "naval"):
                prop = self.get_property_at(*unit.pos)
                refuel_exempt = (
                    prop is not None
                    and prop.owner == opponent
                    and (
                        (stats.unit_class == "naval" and prop.is_port)
                        or (stats.unit_class in ("air", "copter") and prop.is_airport)
                    )
                )
                # AWBW: naval units on water (sea/shoal) also survive
                if not refuel_exempt and stats.unit_class == "naval":
                    tid = self.map_data.terrain[unit.pos[0]][unit.pos[1]]
                    tinfo = TERRAIN_TABLE.get(tid)
                    if tinfo and ("Sea" in tinfo.name or "Shoal" in tinfo.name):
                        refuel_exempt = True
                if not refuel_exempt:
                    unit.hp = 0   # crash / sink

        # Remove units that crashed on fuel starvation
        self.units[opponent] = [u for u in self.units[opponent] if u.is_alive]

        # Resupply units on APC-adjacent tiles: handled in _apply_wait.
        #
        # AWBW start-of-turn ordering — Phase 11J-FUNDS-SHIP (R1):
        # **income FIRST, then property-day repair.** The opponent collects
        # per-property funds before the heal pass spends any of them.
        #
        # Sources:
        #   - User-confirmed AWBW canon (Imperator, Phase 11J-FUNDS-SHIP):
        #     "the ordering is at the start of the turn you get your daily
        #     income then you spend on repairs. In AWBW that's how it works."
        #     This overrides the Phase 11A Kindle precedent block for R1.
        #   - PHP-snapshot empirical proof: 69/69 Tier-3 income-before-repair
        #     matches in the 100-game corpus (Phase 11J-FUNDS-CORPUS,
        #     docs/oracle_exception_audit/phase11j_funds_corpus_derivation.md
        #     §3) and 37/39 NEITHER rows match income-before-repair to the
        #     gold under the IBR hypothetical (Phase 11J-FUNDS-DEEP,
        #     docs/oracle_exception_audit/phase11j_funds_deep.md §3.4).
        #   - Vanilla Advance Wars wiki "Turn" article (supplementary, not
        #     AWBW-specific): "In the beginning of a turn, a side earns
        #     funds for every property it controls, and units at allied
        #     properties are repaired by 2HP."
        #     https://advancewars.fandom.com/wiki/Turn
        #
        # The prior RBI order stranded heals the engine could not afford
        # because the heal pass ran on pre-income funds (typically 0g);
        # those skipped heals then accumulated as upstream funds drift
        # into every subsequent turn-roll.
        self._grant_income(opponent)
        self._resupply_on_properties(opponent)

        # Refresh tower counts now that ownership may have changed this turn
        self._refresh_comm_towers()

        if not self.done:
            maybe_spirit_after_end_turn(self, player)

    def _grant_income(self, player: int) -> None:
        """
        Apply per-turn income to ``player``'s treasury using AWBW rules:
        1000g per owned income-property (HQ/base/city/airport/port), excluding
        comm towers and labs.

        CO modifiers applied here:
          * **Colin** (co_id 15) "Gold Rush" DTD — +100g per income-property.
          * **Sasha** (co_id 19) "Market Crash" DTD — +100g per income-property
            (mirrors Colin's per-property bonus; AWBW Sasha also gains funds on
            damage dealt, handled in combat — not here). Without this, mid- and
            late-game Sasha games consistently drift the engine treasury below
            AWBW's by ~100g × props × turn, surfacing as ``Build no-op
            (insufficient funds)`` at every Tank build (game ``1623012`` was
            the first traced case; the Sasha bucket dominates this cluster).

        Kindle (co_id 23) is **deliberately not branched here.** Phase 11A
        attempted a +50% city-income bonus per ``data/co_data.json`` but
        ``tools/_phase10n_drilldown.py`` on game ``1628546`` (Kindle vs Max,
        map 159501) showed PHP rejecting the bonus on the very first Kindle
        city capture (turn 4 grant: PHP +4000 / engine +4500), pulling the
        first funds mismatch from envelope 11 (pre-fix) up to envelope 5
        (post-fix, +500 to engine). This matches the AWBW CO Chart
        (https://awbw.amarriner.com/co.php) which is **silent** on Kindle
        income — the +50% line in ``co_data.json`` and the community wiki is
        a discrepancy already flagged in
        ``docs/oracle_exception_audit/phase10t_co_income_audit.md`` Section
        3, and the live PHP oracle sides with the chart. See
        ``phase11a_kindle_hachi_canon.md`` for the rollback record.

        SCOP / COP income multipliers (Colin's "Power of Money" funds ×1.5 of
        base income, Sasha's "Market Crash" funds drain on opponent) are
        modeled in ``_apply_power_effects`` / ``_apply_attack`` rather than
        here so the per-turn baseline stays clean.
        """
        n = self.count_income_properties(player)
        income = n * 1000

        co = self.co_states[player]
        if co.co_id == 15:  # Colin: base +100 per prop (DTD), COP: ×1.5 of base income
            income += n * 100
        elif co.co_id == 19:  # Sasha: +100g per income-property DTD (War Bonds)
            income += n * 100

        self.funds[player] = min(self.funds[player] + income, 999_999)

    # ------------------------------------------------------------------
    # Power activation
    # ------------------------------------------------------------------

    def _activate_power(self, cop: bool):
        co = self.co_states[self.active_player]
        if cop:
            co.cop_active  = True
            co.scop_active = False
        else:
            co.scop_active = True
            co.cop_active  = False
        # Consume only the COP/SCOP segment; remainder stays on the bar (AWBW).
        # Threshold is stars * 9000 + uses * 1800, but we only subtract
        # stars * 9000 (the star cost), NOT the threshold (which includes uses).
        # AWBW: meter keeps the "uses * 1800" portion after activation.
        stars = co.cop_stars if cop else co.scop_stars
        thresh = stars * 9000
        co.power_bar = max(0, co.power_bar - thresh)
        co.power_uses += 1  # raises COP/SCOP threshold by +1800/star next time

        self._apply_power_effects(self.active_player, cop)
        self._apply_weather_from_power(co, cop)
        self.action_stage = ActionStage.SELECT

        self.game_log.append({
            "type": "power",
            "player": self.active_player,
            "kind": "cop" if cop else "scop",
            "co": co.name,
        })

    def _apply_weather_from_power(self, co: "COState", cop: bool) -> None:
        """Set global weather from CO power activation.

        Olaf (co_id 9) — COP Blizzard and SCOP Winter Fury both cause snow.
        Drake (co_id 5) — SCOP Typhoon causes rain; COP Tsunami does not.
        Lasts 2 end-turns (activator's remainder + opponent's full turn = 1 AW day).
        Last activation always wins (overrides any previous CO weather).
        """
        if co.co_id == 9:          # Olaf: COP + SCOP → snow
            self.weather = "snow"
            self.co_weather_segments_remaining = 2
        elif co.co_id == 5 and not cop:   # Drake: SCOP only → rain
            self.weather = "rain"
            self.co_weather_segments_remaining = 2

    def _apply_power_effects(self, player: int, cop: bool):
        """Apply immediate on-activation effects for special COs.

        **Flat HP loss** (Drake, Olaf SCOP, Hawke, Von Bolt): AWBW uses integer
        "display HP" steps on the usual 10× internal scale (1 HP = 10 points).
        These are *not* run through the combat damage formula (no luck, no
        ceil-to-0.05-then-floor). Units cannot be destroyed: remaining HP is
        floored at **1 internal**, matching wiki wording (~0.1 display HP minimum).
        """
        co       = self.co_states[player]
        opponent = 1 - player

        # Andy: COP heal +2HP (20pts), SCOP heal +5HP (50pts)
        if co.co_id == 1:
            heal = 20 if cop else 50
            for u in self.units[player]:
                u.hp = min(100, u.hp + heal)

        # Hawke (co_id 12) — power heal/damage canon.
        #
        #   * COP "Black Wave"  : friends +1 HP (+10 internal), enemies -1 HP
        #     (-10 internal, floored at 1 internal).
        #   * SCOP "Black Storm": friends +2 HP (+20 internal), enemies -2 HP
        #     (-20 internal, floored at 1 internal).
        #
        # Sources (Phase 11J-FINAL-HAWKE-CLUSTER, 2026-04-21):
        #   - AWBW CO Chart (Tier 1, amarriner.com canonical):
        #     https://awbw.amarriner.com/co.php Hawke row —
        #     Black Wave: "All units gain +1 HP. All enemy units take 1 HP damage."
        #     Black Storm: "All units gain +2 HP. All enemy units take 2 HP damage."
        #   - AWBW Fandom Wiki (Tier 2, supporting):
        #     https://awbw.fandom.com/wiki/Hawke
        #   - Wars Wiki (Tier 2, vanilla AW cross-check):
        #     https://warswiki.org/wiki/Hawke
        #   - PHP ground truth (Tier 3) — drilled gid 1635846 env 30 (Hawke COP
        #     "Black Wave" day 16) via tools/_phase11j_hawke_cop_drill.py: own
        #     non-combat units showed clean +1.0 display HP heals (Artillery
        #     7.1->8.1, Infantry 7.0->8.0, Mech 5.6->6.6, Mech 8.5->9.5),
        #     enemy non-combat units showed -1.0 baseline.
        #
        # Pre-fix bug: engine ALWAYS healed friends +20 internal HP regardless
        # of cop/scop, over-healing on Black Wave by +10 internal (+1 display
        # bar) per fire. data/co_data.json description for Black Wave
        # ("All enemy units take 1 HP damage; all own units recover 2 HP")
        # was also inconsistent with the AWBW chart and is corrected in the
        # same closeout doc (phase11j_final_hawke_cluster.md).
        elif co.co_id == 12:
            friend_heal = 10 if cop else 20
            enemy_loss = 10 if cop else 20
            for u in self.units[player]:
                u.hp = min(100, u.hp + friend_heal)
            for u in self.units[opponent]:
                u.hp = max(1, u.hp - enemy_loss)

        # Sensei COP: spawn Infantry on every owned city without a unit
        # SCOP: spawn Mech on every owned city without a unit
        # AWBW canon (Tier 1, https://awbw.amarriner.com/co.php Sensei row):
        #   COP "Copter Command" — *"9 HP unwaited infantry are placed on
        #        every owned, empty city."*
        #   SCOP "Airborne Assault" — *"9 HP unwaited mechs are placed on
        #        every owned, empty city."*
        # Note: "city" (not base/airport) per canon; AWBW Fandom wiki agrees.
        elif co.co_id == 13:
            if cop:
                spawn_type = UnitType.INFANTRY
            else:
                spawn_type = UnitType.MECH
            for prop in self.properties:
                if alive_owned_unit_count(self.units[player]) >= self.map_data.unit_limit:
                    break
                if prop.owner == player and prop.is_city:
                    if self.get_unit_at(prop.row, prop.col) is None:
                        unit = Unit(
                            unit_type=spawn_type,
                            player=player,
                            hp=90,   # 9 HP (AWBW canon: "9 HP unwaited")
                            ammo=UNIT_STATS[spawn_type].max_ammo,
                            fuel=UNIT_STATS[spawn_type].max_fuel,
                            pos=(prop.row, prop.col),
                            moved=True,
                            loaded_units=[],
                            is_submerged=False,
                            capture_progress=20,
                            unit_id=self._allocate_unit_id(),
                        )
                        self.units[player].append(unit)

        # Drake: deal HP damage + fuel drain to enemy air units
        elif co.co_id == 5:
            dmg  = 10 if cop else 20
            fuel = 10 if cop else 20
            for u in self.units[opponent]:
                u.hp = max(1, u.hp - dmg)
                if UNIT_STATS[u.unit_type].unit_class in ("air", "copter"):
                    u.fuel = max(0, u.fuel - fuel)

        # Olaf SCOP: 2HP damage to all enemies (blizzard)
        elif co.co_id == 9 and not cop:
            for u in self.units[opponent]:
                u.hp = max(1, u.hp - 20)

        # Jess (co_id 14) COP "Turbo Charge" / SCOP "Overdrive": refill ammo
        # and fuel for ALL of her units (not just vehicles — AWBW wiki Jess
        # entry: "Refills the fuel and ammunition of all units"). Vehicle atk
        # / mov bonuses are handled in combat / action move-range. Without the
        # refuel, naval / air units that drained low pre-power can no longer
        # execute the Move that AWBW emits the same envelope (game 1632380:
        # P1 Drake's BB at fuel=3 → AWBW shows fuel=60 after SCOP → engine
        # raised "Illegal move: Black Boat ... fuel=0 is not reachable").
        elif co.co_id == 14:
            for u in self.units[player]:
                stats = UNIT_STATS[u.unit_type]
                u.fuel = stats.max_fuel
                u.ammo = stats.max_ammo

        # Colin (co_id 15) — Phase 11J-COLIN-IMPL-SHIP.
        #
        # AWBW canon (Tier 1, both AWBW canonicals agree — see
        # docs/oracle_exception_audit/phase11y_colin_scrape.md §0.2, §0.3, §7):
        #
        #   * COP "Gold Rush" — *"Funds are multiplied by 1.5x."*
        #     Sources: https://awbw.amarriner.com/co.php (Colin row) and
        #     https://awbw.fandom.com/wiki/Colin
        #
        #   * Rounding: AWBW uses **round-half-up** (PHP's default ``round()``
        #     mode) on the ``× 1.5`` product. Both wikis are silent on
        #     rounding; the PHP-payload empirical drill (scrape §7.3,
        #     `tools/_colin_gold_rush_drill_strict.py`) confirmed
        #     round-half-up on **15 / 15** sub=0 COP envelopes carrying
        #     ``playerReplace.players_funds`` (3 boundary cases on the .5
        #     mark all matched ``round_half_up``, NOT ``int()`` floor).
        #     Using ``int()`` would silently desync ~20 % of Colin COP fires.
        #     Funds are clamped to the engine's universal 999 999 cap.
        #
        #   * SCOP "Power of Money" — funds snapshot only. The +(3 * funds /
        #     1000)% attack rider is computed in
        #     ``engine/combat.py::_colin_atk_rider`` from the snapshot field
        #     ``COState.colin_pom_funds_snapshot``. Snapshotting at activation
        #     (rather than reading live during each attack) keeps the bonus
        #     stable across mid-turn 80%-cost builds — the AW design intent
        #     for one-turn power durations.
        #
        # Closure: zero Colin gids in the 936-zip GL std corpus
        # (`logs/desync_register_post_phase11j_v2_936.jsonl`) so direct
        # gid validation is unavailable; ships the canon for the 15 ingested
        # Colin replays in `data/amarriner_gl_colin_batch.json` (12 of which
        # carry COP envelopes, see scrape §5).
        elif co.co_id == 15:
            if cop:
                # round_half_up(funds * 1.5) via pure integer arithmetic:
                #   (3 * funds + 1) // 2  for funds >= 0.
                # Examples (PHP-payload anchors from scrape §7.3):
                #   50 835 → 76 253; 48 533 → 72 800; 23 331 → 34 997.
                pre = self.funds[player]
                self.funds[player] = min(999_999, (3 * pre + 1) // 2)
            else:
                co.colin_pom_funds_snapshot = self.funds[player]

        # Sasha COP "Market Crash" — Phase 11J-SASHA-MARKETCRASH-FIX.
        #
        # AWBW canon (Tier 1, AWBW CO Chart, Sasha row):
        #   *"Market Crash -- Reduces enemy power bar(s) by
        #   (10 * Funds / 5000)% of their maximum power bar."*
        #   https://awbw.amarriner.com/co.php
        #
        # The drain is proportional to Sasha's current treasury, NOT her
        # property count. The pre-fix formula was
        # ``count_properties(player) * 9000`` — wildly wrong magnitude
        # (e.g. 14 properties × 9000 = 126,000 power-bar drain on an opp
        # with a max bar of ~54,000 → effectively always full-clear).
        #
        # "Maximum power bar" = the opponent's SCOP charge ceiling, which
        # mirrors ``_scop_threshold`` in engine/co.py: the bar visually
        # maxes at SCOP, and that ceiling rises by +1800 per star per
        # prior power use (AWBW changelog rev 139, 2018-06-30).
        #
        # Math: ``(10 * funds / 5000)%`` simplifies to
        # ``funds / 50000`` as a fraction, so
        # ``drain = max_bar * funds // 50000``. Floored at 0.
        #
        # Closure: ≥2 oracle_gap rows (1626284, 1628953) per
        # docs/oracle_exception_audit/phase11j_co_mechanics_survey.md.
        elif co.co_id == 19 and cop:
            opp_co = self.co_states[opponent]
            sasha_funds = self.funds[player]
            opp_max_bar = opp_co.scop_stars * (9000 + opp_co.power_uses * 1800)
            drain = (opp_max_bar * sasha_funds) // 50000
            opp_co.power_bar = max(0, opp_co.power_bar - drain)

        # Sasha SCOP "War Bonds" — Phase 11J-SASHA-WARBONDS-SHIP.
        #
        # AWBW canon (Tier 1, AWBW CO Chart, Sasha row):
        #   *"War Bonds — Returns 50% of damage dealt as funds (subject to
        #   a 9HP cap)."*  https://awbw.amarriner.com/co.php
        #
        # Mechanics implemented in `_apply_attack` →
        # `_apply_war_bonds_payout`:
        #   payout = min(damage_display_hp, 9) * unit_cost(target_type) // 20
        # Hybrid crediting (Phase 11J-L1-WAVE-2-SHIP):
        #   * **Own SCOP-turn attacks** (damage_dealer == active_player):
        #     credited IMMEDIATELY to Sasha's treasury so in-turn builds
        #     can spend the bonds. PHP behaves the same — five 936-cohort
        #     BUILD-FUNDS-RESIDUAL gids (1624082, 1626284, 1628953,
        #     1634267, 1634893) all show SCOP at action [0] followed by
        #     ≥4 attacks then builds totalling more than pre-SCOP funds.
        #   * **Counter-attacks during opp's intervening turn**
        #     (damage_dealer != active_player): accumulated into
        #     ``pending_war_bonds_funds`` and credited at the END of opp's
        #     turn (preserves the 1624082 env-22 −200g anchor where the
        #     delta only materialises after opp's intervening turn ends).
        # Active window = activator's remainder + opponent's full intervening
        # turn (cleared in `_end_turn` when the opponent finishes — see the
        # `war_bonds_active` clear block there).
        #
        # The earlier "all-real-time" experiment regressed 23/100 GL std
        # games (mid-turn spending-power drift in opp-counter scenarios);
        # the hybrid keeps deferred crediting on opp's turn so that
        # regression does not return.
        elif co.co_id == 19 and not cop:
            co.war_bonds_active = True
            co.pending_war_bonds_funds = 0

        # Kindle COP "Urban Blight" — Phase 11J-L1-BUILD-FUNDS-SHIP.
        #
        # AWBW canon (Tier 1, AWBW CO Chart https://awbw.amarriner.com/co.php
        # Kindle row): *"Urban Blight -- All enemy units lose -3 HP on urban
        # terrain."* Urban = HQs, bases, airports, ports, cities, labs,
        # comtowers — every tile type the terrain registry flags as
        # ``is_property``.
        #
        # SCOP "High Society" deals NO area damage per the same Tier-1
        # chart; its effect is purely the +130 urban ATK rider + 3/prop
        # global rider, both handled in ``engine/combat.py``. Do not add
        # a SCOP branch here.
        #
        # Flat HP loss — no luck, no damage formula, no terrain / CO DEF.
        # Floored at 1 internal (~0.1 display) like Hawke / Olaf / Von Bolt
        # AOE COP-paths above.
        #
        # Cluster: five BUILD-FUNDS-RESIDUAL oracle_gap rows in the Phase
        # 11J-L1 25-gid set where Kindle is the opponent CO — enemy units
        # remained healthier in the engine than in PHP because the -3 HP
        # AOE was unmodeled, which cascaded into lower PHP-side repair
        # costs vs engine (and the engine-side build refusals after).
        elif co.co_id == 23 and cop:
            for u in self.units[opponent]:
                tinfo = get_terrain(self.map_data.terrain[u.pos[0]][u.pos[1]])
                if tinfo.is_property:
                    u.hp = max(1, u.hp - 30)

        # Von Bolt SCOP (Ex Machina): AOE damage + stun on affected enemy
        # units. AWBW canon (Tier 1, AWBW Wiki Von Bolt page; mirrored on
        # the AWBW CO Chart https://awbw.amarriner.com/co.php Von Bolt row):
        #   *"Ex Machina — A 2-range missile deals 3 HP damage and prevents
        #   all affected enemy units from acting next turn. The missile
        #   targets the opponents' greatest accumulation of unit value."*
        # Damage is a flat 30 internal HP / 3 display HP (no luck, no terrain
        # / CO defense), floored at 1 internal (~0.1 display) — same model
        # as Hawke / Olaf flat-loss SCOPs handled above.
        #
        # When the oracle has pinned the AOE tile set via
        # ``_oracle_power_aoe_positions`` (parsed from the Power action's
        # ``missileCoords``), apply the loss + stun only to enemy units
        # inside that set. ``None`` keeps the historical global behaviour
        # for the RL / non-oracle path (no missile targeter implemented;
        # global enemy stun is the safest legality posture). One-shot:
        # cleared after consumption.
        #
        # AOE shape: 13 tiles, Manhattan ≤ 2 from ``missileCoords`` (Tier 1–3
        # citations on the oracle Von Bolt pin). Engine is membership-only on
        # ``_oracle_power_aoe_positions``.
        #
        # Stun (Phase 11J-VONBOLT-SCOP-SHIP):
        #   - Set ``Unit.is_stunned = True`` on every affected enemy.
        #   - The flag is read by ``engine/action.py::_get_select_actions``
        #     (no SELECT_UNIT emitted), the STEP-GATE in ``step``, and
        #     ``_apply_attack`` (counter-attack skipped against a stunned
        #     defender). Cleared in ``_end_turn`` on the units of the
        #     player whose turn just ended — the stunned army serves the
        #     stun across exactly one of its own turns.
        #   - Own units are not stunned (wiki: "all affected *enemy*
        #     units") even though the AWBW CO Chart shorthand says "all
        #     affected units"; the wiki is the more specific source and
        #     the cluster-B engine drift is consistent with enemy-only.
        #
        # Diagnostic source: phase11j_funds_deep.md §5.2 + drill on 1622328
        # env 28 (PHP diamond vs historical global engine -30).
        elif co.co_id == 30 and not cop:
            aoe = self._oracle_power_aoe_positions
            self._oracle_power_aoe_positions = None
            for u in self.units[opponent]:
                if aoe is None or u.pos in aoe:
                    u.hp = max(1, u.hp - 30)
                    u.is_stunned = True

        # Sturm (co_id 29) COP "Meteor Strike" / SCOP "Meteor Strike II" —
        # Phase 11J-FINAL-STURM-SCOP-SHIP. The Sturm SCOP freeze imposed
        # in Phase 11J-FINAL-BUILD-NO-OP-RESIDUALS was lifted by the
        # imperator on 2026-04-21 once the build no-op residual on gid
        # 1635679 was attributed to this exact missing branch.
        #
        # AWBW canon (Tier 1, AWBW CO Chart https://awbw.amarriner.com/co.php
        # Sturm row, fetched 2026-04-21):
        #   *"Meteor Strike -- A 2-range missile deals 4 HP damage. The
        #     missile targets an enemy unit located at the greatest
        #     accumulation of unit value."*
        #   *"Meteor Strike II -- A 2-range missile deals 8 HP damage. The
        #     missile targets an enemy unit located at the greatest
        #     accumulation of unit value."*
        # Site ``powerName`` for SCOP is ``"Meteor Strike II"`` (not the
        # ``co_data.json`` lore name "Fury Storm").
        #
        # Damage: flat HP loss — 40 internal (4 display) on COP, 80 internal
        # (8 display) on SCOP. No luck, no terrain / CO defense; floored at
        # 1 internal (~0.1 display) — same flat-loss SCOP family as Hawke,
        # Olaf SCOP, Drake, Von Bolt SCOP. Enemy units only (chart text
        # targets "an enemy unit").
        #
        # AOE shape: 13-tile Manhattan diamond (M<=2) around the
        # ``missileCoords`` center. Confirmed empirically against PHP
        # ``unitReplace`` ground truth on three envelopes:
        #   * gid 1615143 env 33 (COP center=(8,7)): exactly 5 enemies at
        #     M<=2 in engine pre-state, 5 affected in unitReplace.
        #   * gid 1615143 env 57 (COP center=(12,17)): exactly 2 (Fighter
        #     + Infantry), 2 affected — confirms air units are hit.
        #   * gid 1635679 env 28 (SCOP center=(6,9)): post-disp HPs all
        #     land at 1-2 from pre-disp 3-10 — consistent with -80 HP
        #     internal + 1-internal clamp.
        #
        # Oracle pin: ``_oracle_power_aoe_positions`` is filled by
        # ``tools/oracle_zip_replay.py`` from ``missileCoords`` before the
        # ``ACTIVATE_COP``/``ACTIVATE_SCOP`` step. ``None`` keeps the
        # historical no-op behaviour for the RL / non-oracle path (no
        # missile targeter implemented; global enemy -40/-80 would massively
        # over-damage). One-shot: cleared after consumption.
        #
        # Cluster: gid 1635679 BUILD-NO-OP residual (Sturm vs Hawke);
        # confirmed cohort 8 Sturm power activations across 3 zips
        # (1615143, 1635679, 1637200) per
        # ``tools/_phase11j_sturm_cohort.py``.
        elif co.co_id == 29:
            aoe = self._oracle_power_aoe_positions
            self._oracle_power_aoe_positions = None
            dmg = 40 if cop else 80
            if aoe is not None:
                for u in self.units[opponent]:
                    if u.pos in aoe:
                        u.hp = max(1, u.hp - dmg)

        # Rachel SCOP "Covering Fire" — Phase 11J-RACHEL-SCOP-COVERING-FIRE-SHIP.
        #
        # AWBW canon (Tier 1, AWBW CO Chart https://awbw.amarriner.com/co.php
        # Rachel row): *"Covering Fire — Three 2-range missiles deal 3 HP
        # damage each. The missiles target the opponents' greatest accumulation
        # of footsoldier HP, unit value, and unit HP (in that order)."*
        #
        # Damage: flat 30 internal HP / 3 display HP per missile, floored at
        # 1 internal (~0.1 display) — same flat-loss SCOP family as Hawke,
        # Olaf SCOP, Von Bolt SCOP. Enemy units only (chart text targets
        # "the opponents'" accumulations; mirrors Von Bolt's wiki-anchored
        # enemy-only convention).
        #
        # Multiplicity: Rachel fires three independent missiles, and the
        # player MAY aim two of them at the same tile (drilled on gid
        # 1622501 env 20: missileCoords = [(11,20), (4,9), (11,20)] — the
        # (11,20) cluster is hit by two missiles). The oracle pin is a
        # ``Counter[(y, x)] -> hit_count`` (see oracle_zip_replay.py Rachel
        # branch). A unit at a tile with count=2 takes 60 HP, count=3 takes
        # 90 HP. We multiply 30 × count and floor once.
        #
        # AOE shape: 5x5 Manhattan diamond (2-range) per missile — AWBW canon.
        # Phase 11J-RACHEL-FUNDS-DRIFT-SHIP closed the prior 3x3 Chebyshev box
        # gap by widening the oracle pin in ``oracle_zip_replay.py`` Rachel
        # branch. The engine consumer here is shape-agnostic (Counter lookup
        # by ``u.pos``); the only behavioural change is that more enemy tiles
        # appear in the Counter. Canon source + drill cited in
        # ``docs/oracle_exception_audit/phase11j_rachel_funds_drift_ship.md``.
        #
        # No oracle pin: the engine alone cannot decide where Rachel's
        # missiles land (the targeter chases enemy-cluster heuristics that
        # are not modeled in the engine). Falling back to a global -90 HP
        # would massively over-damage; falling back to no-op preserves the
        # RL / non-oracle path's prior silent behavior. We choose no-op.
        #
        # Cluster: five BUILD-FUNDS-RESIDUAL oracle_gap rows in the Phase
        # 11J-L1 25-gid set where Rachel is active (1622501, 1630669,
        # 1634146, 1635164, 1635658) — engine units stayed healthier than
        # PHP because the missile damage was unmodeled, cascading into
        # lower PHP-side repair costs vs engine.
        elif co.co_id == 28 and not cop:
            aoe = self._oracle_power_aoe_positions
            self._oracle_power_aoe_positions = None
            if isinstance(aoe, Counter):
                for u in self.units[opponent]:
                    hits = aoe.get(u.pos, 0)
                    if hits > 0:
                        u.hp = max(1, u.hp - 30 * hits)

        # Eagle (10) SCOP "Lightning Strike" — AWBW wiki
        # (https://awbw.fandom.com/wiki/Eagle): all non-footsoldier own units
        # may move and fire again. Footsoldiers (infantry, mech) are unchanged
        # so they cannot gain a second activation from this refresh.
        elif co.co_id == 10 and not cop:
            for u in self.units[player]:
                if not u.is_alive:
                    continue
                cls = UNIT_STATS[u.unit_type].unit_class
                if cls in ("infantry", "mech"):
                    continue
                u.moved = False

        # Prune dead units from power effects
        for p in (0, 1):
            self.units[p] = [u for u in self.units[p] if u.is_alive]

    # ------------------------------------------------------------------
    # Attack
    # ------------------------------------------------------------------

    def _apply_attack(
        self,
        action: Action,
        *,
        oracle_mode: bool = False,
        oracle_strict: bool = False,
    ):
        # Phase 11J P-COLO-ATTACKER: if STEP-GATE has already pinned the actor
        # via ``selected_unit`` and that unit is alive on ``action.unit_pos``,
        # prefer it over ``get_unit_at`` — the latter returns the *first*
        # match on a co-occupied tile, which can pick a stationary same-tile
        # unit (e.g. cargo or prior-turn arrival) instead of the actually
        # selected mover. Falls through to ``get_unit_at`` for legacy paths
        # (tests, seam attacks) where ``selected_unit`` is None.
        attacker: Optional[Unit] = None
        sel = self.selected_unit
        if sel is not None and sel.is_alive and sel.pos == action.unit_pos:
            attacker = sel
        if attacker is None:
            # Phase 11J-F5-OCCUPANCY: prefer oracle id when set, defends against duplicate-position oracle states.
            attacker = self.get_unit_at_oracle_id(
                *action.unit_pos, action.select_unit_id
            )
        if attacker is None:
            raise ValueError(
                f"_apply_attack: no attacker at {action.unit_pos}"
            )

        # Phase 3 ATTACK-INV defense-in-depth (mirrors get_attack_targets /
        # get_legal_actions; redundant when STEP-GATE is enforced, fires only
        # if step() is called with a crafted action that bypassed the mask).
        # Seam attacks (defender is None) are owned by SEAM thread and routed
        # through _apply_seam_attack below, so we only police the unit-vs-unit
        # branch here.
        defender_pre = (
            self.get_unit_at(*action.target_pos)
            if action.target_pos is not None
            else None
        )
        if defender_pre is not None and defender_pre.player == attacker.player:
            raise ValueError(
                f"_apply_attack: friendly fire from player {attacker.player} "
                f"on {defender_pre.unit_type.name} at {action.target_pos}"
            )
        # Phase 11J P-AMMO override-bypass: when the oracle has pinned the
        # post-strike HPs via ``_oracle_combat_damage_override``, AWBW already
        # decided the strike was legal (e.g. Mech/B-Copter using their
        # secondary MG with primary ammo=0 — ``get_attack_targets`` shorts
        # out at ``ammo == 0`` even though the MG is unmetered). Trust the
        # oracle and skip this defense-in-depth check; the override is
        # consumed at L684 below so any subsequent step is gated normally.
        #
        # Replay export / zip rebuild calls ``step(..., oracle_mode=True)``
        # after IllegalActionError: the live ``full_trace`` can disagree with
        # ``get_attack_targets`` on the rebuilt timeline (benign drift, or
        # indirect move_pos vs unit.pos edge cases). Non-strict oracle trusts
        # the envelope like the HP override path; ``oracle_strict`` keeps this
        # mirror check for audit builds.
        oracle_pinned = self._oracle_combat_damage_override is not None
        skip_attack_inv = oracle_pinned or (
            oracle_mode and not oracle_strict
        )
        if defender_pre is not None and not skip_attack_inv:
            atk_from = action.move_pos if action.move_pos is not None else attacker.pos
            if action.target_pos not in get_attack_targets(self, attacker, atk_from):
                raise ValueError(
                    f"_apply_attack: target {action.target_pos} not in attack "
                    f"range for {attacker.unit_type.name} from {atk_from} "
                    f"(unit_pos={action.unit_pos})"
                )

        self._move_unit(attacker, action.move_pos)

        defender = self.get_unit_at(*action.target_pos)
        if defender is None:
            # Empty tile — check for intact pipe seam (AWBW: seams are legal
            # attack targets without a defender unit). Bailing early here was
            # the historical gap the seam-targetable-check addresses.
            if self._apply_seam_attack(attacker, action):
                return
            self._finish_action(attacker)
            return

        att_terrain = get_terrain(self.map_data.terrain[action.move_pos[0]][action.move_pos[1]])
        def_terrain = get_terrain(self.map_data.terrain[action.target_pos[0]][action.target_pos[1]])
        att_co      = self.co_states[attacker.player]
        def_co      = self.co_states[defender.player]

        # Oracle-supplied AWBW ground truth bypasses the random luck roll in
        # ``calculate_damage`` / ``calculate_counterattack``. See the
        # ``_oracle_combat_damage_override`` docstring on ``GameState``.
        override = self._oracle_combat_damage_override
        self._oracle_combat_damage_override = None
        override_dmg = override[0] if override is not None else None
        override_counter = override[1] if override is not None else None

        # Primary attack
        # Phase 11J-SASHA-WARBONDS-SHIP: capture defender display HP BEFORE
        # damage so we can compute display-HP loss for the War Bonds payout.
        # CO meter likewise uses defender display buckets before/after hp change.
        pre_def_disp = defender.display_hp
        dmg = 0  # Default: no damage
        if override_dmg is not None:
            dmg = max(0, int(override_dmg))
        else:
            dmg = calculate_damage(
                attacker, defender,
                att_terrain, def_terrain,
                att_co, def_co,
                luck_rng=self.luck_rng,
            )
        if dmg is not None and dmg > 0:
            # AWBW: 9000 funds damage = 1 star = 9000 power_bar units
            # dmg is display HP lost (1-10) from oracle (pre/post are internal_HP/10)
            # Convert to internal HP (1-100) for engine unit state
            internal_dmg = dmg * 10
            defender.hp = max(0, defender.hp - internal_dmg)
            self.losses_hp[defender.player] += internal_dmg  # Track HP lost
            if defender.hp == 0:
                self.losses_units[defender.player] += 1  # Track unit destroyed
            # Pass internal HP to CO meter function
            self._apply_co_meter_from_internal_hp_lost(attacker, defender, internal_dmg)
            self._apply_war_bonds_payout(attacker, defender, pre_def_disp)

        # Counterattack (only if defender survived and attacker is direct)
        att_stats = UNIT_STATS[attacker.unit_type]
        # Phase 11J-VONBOLT-SCOP-SHIP — stunned defenders do not counter.
        # AWBW canon (https://awbw.fandom.com/wiki/Von_Bolt): Ex Machina
        # *"prevents all affected enemy units from acting next turn."* A
        # counter-attack is an act; PHP correctly emits no counter from a
        # stunned defender. The pre-fix engine ran the counter regardless,
        # which (a) damaged the attacker that PHP left untouched and (b)
        # consumed defender ammo PHP did not — both surfacing as drift in
        # the cluster-B funds gids (1621434, 1621898, 1622328: see
        # ``docs/oracle_exception_audit/phase11j_vonbolt_scop_ship.md`` §3
        # for the drill).
        defender_can_counter = (
            defender.is_alive
            and not att_stats.is_indirect
            and not defender.is_stunned
        )
        if defender_can_counter:
            # Phase 11J-SASHA-WARBONDS-SHIP: capture attacker display HP BEFORE
            # the counter so the defender's War Bonds payout (when defender
            # is Sasha) reflects the correct display-HP loss.
            pre_atk_disp = attacker.display_hp
            if override_counter is not None:
                counter = max(0, int(override_counter))
            else:
                counter = calculate_counterattack(
                    attacker, defender,
                    att_terrain, def_terrain,
                    att_co, def_co,
                    attack_damage=dmg,
                    luck_rng=self.luck_rng,
                )
            if counter is not None and counter > 0:
                attacker.hp = max(0, attacker.hp - counter)
                self.losses_hp[attacker.player] += counter  # Track counterattack HP lost
                if attacker.hp == 0:
                    self.losses_units[attacker.player] += 1  # Track unit destroyed
                # Use internal HP lost (counter) for CO meter
                # AWBW: 9000 funds of damage = 1 star
                # Formula: internal_hp_lost × cost / 10 = funds value
                self._apply_co_meter_from_internal_hp_lost(defender, attacker, counter)
                # War Bonds payout for Sasha when she's defending and her
                # unit deals counter-damage. Roles flipped: defender is the
                # damage-dealer, attacker is the recipient of the counter.
                self._apply_war_bonds_payout(defender, attacker, pre_atk_disp)

        # Phase 11J-CLOSE-1624082 — clear the oracle War Bonds pin once
        # the primary + counter pair has run (one-shot per Fire, mirrors
        # ``_oracle_combat_damage_override`` consumption above).
        self._oracle_war_bonds_payout_override = None

        # Consume attacker ammo. AWBW canon: the secondary Machine Gun is
        # **unlimited** and does NOT draw from the unit's primary ammo
        # magazine; only the primary weapon (cannon, bazooka, missile)
        # consumes one round per shot. Per the AWBW Wiki Machine_Gun page
        # (https://awbw.fandom.com/wiki/Machine_Gun) the MG is the
        # secondary weapon for Mech, Tank, Md.Tank, Neotank, Mega Tank and
        # B-Copter, and is used only against Infantry and Mech defenders.
        # The pre-fix engine consumed primary ammo on every strike, which
        # falsely zeroed B-Copter / Mech / Mega Tank magazines several
        # turns earlier than AWBW (47 GL std-tier engine_bug rows in
        # logs/desync_register_post_phase9.jsonl bottomed out at ammo=0).
        att_stats = UNIT_STATS[attacker.unit_type]
        used_secondary_mg = (
            attacker.unit_type in _MG_SECONDARY_USERS
            and defender.unit_type in _MG_SECONDARY_TARGETS
        )
        if att_stats.max_ammo > 0 and not used_secondary_mg:
            attacker.ammo = max(0, attacker.ammo - 1)

        # If a transport just died, all units it was carrying go down with it.
        # Record their HP and unit losses against the cargo's owner before
        # zeroing them so the loss tallies stay consistent.
        for p in (0, 1):
            for u in self.units[p]:
                if not u.is_alive and u.loaded_units:
                    for cargo in u.loaded_units:
                        if cargo.is_alive:
                            self.losses_hp[cargo.player]    += cargo.hp
                            self.losses_units[cargo.player] += 1
                            cargo.hp = 0
                    u.loaded_units = []

        # AWBW capture-progress reset on death: when the unit holding mid-capture
        # is killed (attacker counter-killed on the tile it just stepped onto, or
        # defender killed while capturing its own property), the partial capture
        # resets to 20. ``_move_unit`` already covers tile-vacated cases.
        if not attacker.is_alive:
            apos = attacker.pos
            apr  = self.get_property_at(*apos)
            if apr is not None and apr.capture_points < 20:
                apr.capture_points = 20
        if defender is not None and not defender.is_alive:
            dpos = defender.pos
            dpr  = self.get_property_at(*dpos)
            if dpr is not None and dpr.capture_points < 20:
                dpr.capture_points = 20

        # Prune dead units
        for p in (0, 1):
            self.units[p] = [u for u in self.units[p] if u.is_alive]

        # Army wipe only after combat (AWBW: not when opponent has never had units / build races)
        self._evaluate_army_wipe_after_combat()

        self._finish_action(attacker)
        self.game_log.append({
            "type":   "attack",
            "player": attacker.player,
            "from":   action.unit_pos,
            "to":     list(action.move_pos),
            "target": list(action.target_pos),
            "dmg":    dmg,
        })

    def _grant_co_meter_credit(self, seat: int, credit: int) -> None:
        """Add CO-meter credit capped at SCOP ceiling."""
        if credit <= 0:
            return
        co = self.co_states[int(seat)]
        ceiling = co._scop_threshold
        co.power_bar = min(ceiling, co.power_bar + int(credit))

    def _apply_co_meter_from_internal_hp_lost(
        self,
        striker_unit: Unit,
        victim_unit: Unit,
        internal_hp_lost: int,
    ) -> None:
        """Award CO-meter for one combat swing from internal HP lost.

        AWBW formula: CO meter charges based on funds value of damage dealt.
        9000 funds damage = 1 star = 9000 power_bar units.

        Args:
            internal_hp_lost: Internal HP lost (1-100), where 10 internal = 1 display HP.
        
        For internal HP lost D and unit cost C:
        - Victim seat credit: (D/10) × C = D × C / 10 (funds value)
        - Striker seat credit: 50% of victim credit (per AWBW)
        
        Example: 90 internal HP (9 display) of 1000-cost unit = 9000 funds = 1 star.

        Repairs and non-combat HP changes never call this hook.
        """
        if internal_hp_lost <= 0:
            return
        # AWBW canon: "Real unit cost is also factored into the calculation,
        # so COs with cost-affecting powers (Hachi, Colin, Kanbei) will
        # charge their powers more slowly / more quickly on a per-unit basis."
        base_cv = UNIT_STATS[victim_unit.unit_type].cost
        base_cs = UNIT_STATS[striker_unit.unit_type].cost
        # Apply victim's own CO cost modifier to victim cost
        victim_co = self.co_states[int(victim_unit.player)]
        mod_v = victim_co.unit_cost_modifier_for_unit(victim_unit.unit_type)
        # Apply striker's own CO cost modifier to striker cost
        striker_co = self.co_states[int(striker_unit.player)]
        mod_s = striker_co.unit_cost_modifier_for_unit(striker_unit.unit_type)
        # modifier is %: 20 means +20% → cost * 1.20; -10 → cost * 0.90
        cv = int(base_cv * (100 + mod_v) / 100)
        cs = int(base_cs * (100 + mod_s) / 100)
        # Victim credit: internal_HP_lost × cost / 10 = funds value
        # 90 internal HP × 1000 cost / 10 = 9000 funds = 1 star = 9000 power_bar
        credit_v = internal_hp_lost * cv // 10
        # Striker credit: 50% of victim (per AWBW)
        credit_s = internal_hp_lost * cs // 20
        self._grant_co_meter_credit(int(victim_unit.player), credit_v)
        self._grant_co_meter_credit(int(striker_unit.player), credit_s)
    def _apply_war_bonds_payout(
        self,
        damage_dealer: Unit,
        damage_target: Unit,
        target_pre_display_hp: int,
    ) -> None:
        """Sasha SCOP "War Bonds" funds payout — Phase 11J-SASHA-WARBONDS-SHIP.

        AWBW canon (Tier 1, AWBW CO Chart Sasha row):
          *"War Bonds — Returns 50% of damage dealt as funds (subject to a
          9HP cap)."*  https://awbw.amarriner.com/co.php

        Formula:
          ``payout = min(display_hp_loss, 9) * unit_cost(target_type) // 20``

        ``display_hp_loss`` is computed from the **display** HP scale (1–10),
        i.e. ``ceil(internal_hp / 10)``. The 9 HP cap matches the Tier-1
        chart text and is a per-attack cap (not cumulative). All AWBW unit
        costs are multiples of 1000, so ``cost // 20`` is always integer
        and the payout never has rounding ambiguity.

        Hybrid crediting model (Phase 11J-L1-WAVE-2-SHIP):

        * **Activator's OWN attacks during her SCOP turn** —
          ``damage_dealer.player == self.active_player`` — credit the
          payout IMMEDIATELY to her treasury so the in-turn builds /
          repairs that follow can spend the bonds. PHP empirically does
          the same: in five 936-cohort BUILD-FUNDS-RESIDUAL games
          (1624082, 1626284, 1628953, 1634267, 1634893) Sasha activates
          SCOP at envelope action ``[0]``, attacks ≥4 enemy units, then
          builds vehicles whose total cost exceeds her pre-SCOP funds —
          the only way PHP allows those builds is by crediting bonds in
          real time during the activator's own SCOP turn.
        * **Counter-attacks while Sasha is the defender** (i.e. the
          activator is not the active player — opp's intervening turn)
          — accumulate into ``pending_war_bonds_funds`` and defer the
          credit to ``_end_turn`` at the end of the opponent's turn.
          This preserves the empirical anchor of game ``1624082`` env 22
          where the −200g delta only materialises at the env-22 snapshot
          (after opp's intervening turn ends), not earlier.

        Does nothing when the dealer is not Sasha (``co_id == 19``) or
        her War Bonds window is inactive.

        Phase 11J-CLOSE-1624082 — oracle pin: when
        ``_oracle_war_bonds_payout_override`` carries an entry for this
        dealer's player, use that pinned PHP-side payout instead of the
        formula. See the field docstring on ``GameState`` for the
        empirical grounding (gid ``1624082`` env 33).
        """
        co = self.co_states[damage_dealer.player]
        if co.co_id != 19 or not co.war_bonds_active:
            return
        oracle_pinned: Optional[int] = None
        if self._oracle_war_bonds_payout_override is not None:
            oracle_pinned = self._oracle_war_bonds_payout_override.pop(
                int(damage_dealer.player), None
            )
        if oracle_pinned is not None:
            payout = max(0, int(oracle_pinned))
        else:
            post_disp = damage_target.display_hp
            damage_disp = max(0, target_pre_display_hp - post_disp)
            if damage_disp <= 0:
                return
            damage_capped = min(damage_disp, 9)
            target_cost = UNIT_STATS[damage_target.unit_type].cost
            payout = damage_capped * (target_cost // 20)
        if payout <= 0:
            return
        # Hybrid: own SCOP-turn attacks credit immediately; counter-attacks
        # during opp's intervening turn defer to end-of-opp-turn settlement.
        if damage_dealer.player == self.active_player:
            self.funds[damage_dealer.player] = min(
                999_999, self.funds[damage_dealer.player] + payout
            )
        else:
            co.pending_war_bonds_funds += payout

    def _evaluate_army_wipe_after_combat(self) -> None:
        """One side has no units left; other side wins (checked only after an attack resolves)."""
        if self.done:
            return
        for p in (0, 1):
            opp = 1 - p
            if len(self.units[opp]) == 0 and len(self.units[p]) > 0:
                self.done       = True
                self.winner     = p
                self.win_reason = "army_wipe"
                return

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _apply_capture(self, action: Action) -> float:
        """
        Apply capture; return dense shaping reward in the **capturing unit's**
        (acting player's) frame. No shaping when the tile is already owned by
        that player (non-contest).
        """
        if action.move_pos is None:
            raise ValueError("_apply_capture: action.move_pos is required")
        unit = self.get_unit_at(*action.unit_pos)
        if unit is None:
            raise ValueError(f"_apply_capture: no unit at {action.unit_pos}")
        if not UNIT_STATS[unit.unit_type].can_capture:
            raise ValueError(
                f"_apply_capture: {unit.unit_type.name} cannot capture"
            )
        prop = self.get_property_at(*action.move_pos)
        if prop is None:
            raise ValueError(
                f"_apply_capture: no capturable property at {action.move_pos}"
            )
        if prop.owner == unit.player:
            raise ValueError(
                f"_apply_capture: property at {action.move_pos} already owned "
                f"by player {unit.player}"
            )

        self._move_unit(unit, action.move_pos)
        old_owner = prop.owner
        old_cp = prop.capture_points
        contest = old_owner is None or old_owner != unit.player

        co = self.co_states[unit.player]

        # Sami SCOP: instant capture
        if co.co_id == 8 and co.scop_active:
            prop.capture_points = 0
        else:
            stats = UNIT_STATS[unit.unit_type]
            if co.co_id == 8 and stats.unit_class in ("infantry", "mech"):
                capture_amount = _SAMI_AW_CAPTURE_D2D.get(unit.display_hp, unit.display_hp)
            else:
                capture_amount = unit.display_hp
            prop.capture_points = max(0, prop.capture_points - capture_amount)

        shaping = 0.0
        # First-attempt bonus: rewards the *behavior* of issuing CAPTURE once
        # per (unit, episode) — survives credit-assignment across the 3-stage
        # action chain even when the property does not flip on this attempt.
        # Track set-membership unconditionally (used by other code paths /
        # logging); only emit shaping when phi mode is OFF.
        uid = int(getattr(unit, "unit_id", 0) or 0)
        first_attempt = uid > 0 and uid not in self.capture_attempted_unit_ids
        if uid > 0:
            self.capture_attempted_unit_ids.add(uid)
        if not _PHI_SHAPING_ACTIVE:
            if first_attempt:
                shaping += _CAPTURE_FIRST_ATTEMPT_BONUS
            if contest:
                reduced = float(old_cp - max(prop.capture_points, 0))
                shaping += _CAPTURE_SHAPING_PROGRESS * (reduced / 20.0)

        capture_flip_completed = False
        if prop.capture_points <= 0:
            capture_flip_completed = True
            if contest and not _PHI_SHAPING_ACTIVE:
                shaping += _CAPTURE_SHAPING_COMPLETE
            prop.owner = unit.player
            prop.capture_points = 20
            if prop.is_comm_tower:
                self._refresh_comm_towers()
            old_tid = self.map_data.terrain[prop.row][prop.col]
            new_tid = property_terrain_id_after_owner_change(
                old_tid, unit.player, self.map_data.country_to_player
            )
            if new_tid is not None:
                self.map_data.terrain[prop.row][prop.col] = new_tid
                prop.terrain_id = new_tid

        self._finish_action(unit)
        # game_log: cp_remaining is capture progress *on the contested tile* before the
        # post-flip reset (0 = this action completed the flip). Legacy rows used 20 here
        # after reset; rl.env._log_finished_game accepts both.
        self.game_log.append({
            "type":         "capture",
            "player":       unit.player,
            "from":         action.unit_pos,
            "to":           list(action.move_pos),
            "cp_remaining": 0 if capture_flip_completed else prop.capture_points,
        })
        return shaping

    # ------------------------------------------------------------------
    # Wait
    # ------------------------------------------------------------------

    def _apply_wait(self, action: Action, *, oracle_mode: bool = False):
        # Phase 11J-FINAL (T3 escalation, kept): mirror _apply_attack — prefer
        # selected_unit, then oracle-id pin, before falling back to tile lookup.
        # Closes gid 1636707 (extras catalog: P0 Infantry + P1 Md.Tank shared
        # tile, plain get_unit_at returned wrong seat).
        unit = self.selected_unit
        if unit is None or unit.pos != action.unit_pos:
            unit = self.get_unit_at_oracle_id(*action.unit_pos, action.select_unit_id)
        if unit is None:
            return

        # Defense in depth: WAIT must never leave a unit standing on top of a
        # friendly transport that could load it. The legal-action generator
        # filters this case out, but `step` is also reachable from hand-built
        # actions (tools, tests, scripted opponents).
        occupant = self.get_unit_at(*action.move_pos)
        if (
            occupant is not None
            and occupant is not unit
            and occupant.player == unit.player
            and UNIT_STATS[occupant.unit_type].carry_capacity > 0
            and unit.unit_type in get_loadable_into(occupant.unit_type)
            and len(occupant.loaded_units) < UNIT_STATS[occupant.unit_type].carry_capacity
        ):
            # Same oracle WAIT->LOAD reroute pattern as the JOIN guard below
            # (Phase 11J-FINAL): mask emits LOAD, AWBW envelope encodes Wait.
            if oracle_mode:
                self._apply_load(
                    Action(
                        ActionType.LOAD,
                        unit_pos=action.unit_pos,
                        move_pos=action.move_pos,
                        select_unit_id=action.select_unit_id,
                    ),
                    oracle_mode=True,
                )
                return
            raise ValueError(
                f"Illegal WAIT: {unit.unit_type.name} cannot stop on friendly "
                f"transport {occupant.unit_type.name} at {action.move_pos}; "
                f"use LOAD instead."
            )

        if (
            occupant is not None
            and occupant is not unit
            and occupant.player == unit.player
            and units_can_join(unit, occupant)
        ):
            # Phase 11J-FINAL (T3 follow-up): the engine's get_legal_actions
            # mask emits ONLY ActionType.JOIN at this destination — WAIT is
            # never legal here. AWBW's recorded envelope can still encode the
            # tail as `Wait` (the upstream paths.global lands on a same-type
            # ally), so in oracle_mode we silently route to JOIN — same end
            # state as the AWBW player would have produced via the JOIN menu.
            # RL / non-oracle callers retain the strict raise (defense in depth).
            if oracle_mode:
                self._apply_join(
                    Action(
                        ActionType.JOIN,
                        unit_pos=action.unit_pos,
                        move_pos=action.move_pos,
                        select_unit_id=action.select_unit_id,
                    )
                )
                return
            raise ValueError(
                f"Illegal WAIT: {unit.unit_type.name} cannot idle on injured "
                f"same-type ally at {action.move_pos}; use JOIN to merge."
            )

        # AWBW allows WAIT on a capturable enemy property (the player can
        # decline to capture). Engine ⊂ AWBW: do *not* raise here. The
        # ``get_legal_actions`` mask still hides WAIT in this case for RL
        # shaping; ``step`` accepts hand-built / oracle-replay WAITs.

        self._move_unit(unit, action.move_pos)

        # APC resupply: adjacent allies
        if unit.unit_type == UnitType.APC:
            self._apc_resupply(unit)

        # Black Boat's repair/resupply is NOT auto on WAIT. It is an explicit
        # ActionType.REPAIR targeting one adjacent ally (AWBW "Repair" cmd).
        # See ``_apply_repair`` and ``engine.action._black_boat_repair_eligible``.

        self._finish_action(unit)
        self.game_log.append({
            "type":   "wait",
            "player": unit.player,
            "from":   action.unit_pos,
            "to":     list(action.move_pos),
        })

    def _apply_dive_hide(self, action: Action):
        """Toggle Sub dive / Stealth hide after moving (AWBW Fandom Sub + Stealth pages)."""
        unit = self.get_unit_at(*action.unit_pos)
        if unit is None:
            return
        if not UNIT_STATS[unit.unit_type].can_dive:
            return

        occupant = self.get_unit_at(*action.move_pos)
        if (
            occupant is not None
            and occupant is not unit
            and occupant.player == unit.player
            and UNIT_STATS[occupant.unit_type].carry_capacity > 0
            and unit.unit_type in get_loadable_into(occupant.unit_type)
            and len(occupant.loaded_units) < UNIT_STATS[occupant.unit_type].carry_capacity
        ):
            raise ValueError(
                f"Illegal DIVE_HIDE: {unit.unit_type.name} cannot stop on friendly "
                f"transport {occupant.unit_type.name} at {action.move_pos}; use LOAD."
            )

        if (
            occupant is not None
            and occupant is not unit
            and occupant.player == unit.player
            and units_can_join(unit, occupant)
        ):
            raise ValueError(
                f"Illegal DIVE_HIDE: {unit.unit_type.name} cannot idle on injured "
                f"same-type ally at {action.move_pos}; use JOIN."
            )

        # Sub/Stealth cannot capture, so the "CAPTURE available" raise is
        # unreachable; preserved as a comment alongside the WAIT change so the
        # engine ⊂ AWBW guarantee is uniform.

        self._move_unit(unit, action.move_pos)
        unit.is_submerged = not unit.is_submerged

        self._finish_action(unit)
        self.game_log.append({
            "type":       "dive_hide",
            "player":     unit.player,
            "from":       action.unit_pos,
            "to":         list(action.move_pos),
            "submerged":  unit.is_submerged,
        })

    def _apc_resupply(self, apc: Unit):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            adj = self.get_unit_at(apc.pos[0] + dr, apc.pos[1] + dc)
            if adj and adj.player == apc.player:
                stats     = UNIT_STATS[adj.unit_type]
                adj.fuel  = stats.max_fuel
                if stats.max_ammo > 0:
                    adj.ammo = stats.max_ammo

    # ------------------------------------------------------------------
    # Black Boat REPAIR (AWBW "Repair" command)
    # ------------------------------------------------------------------

    def _black_boat_heal_cost(self, target_type: UnitType) -> int:
        """AWBW Black Boat heal cost for a single +10 HP tick.

        10% of the target's listed deployment cost, floor-divided, with a
        hard floor of 1 so even $0 units (e.g. Oozium) still cost something
        conceptually — the engine never charges negative / free heals.
        """
        listed = UNIT_STATS[target_type].cost
        return max(1, listed // 10)

    def _apply_repair(self, action: Action, *, oracle_strict: bool = False):
        """Resolve a Black Boat REPAIR action on an orthogonally adjacent ally.

        AWBW rules (see ``engine.action._black_boat_repair_eligible`` and
        https://awbw.fandom.com/wiki/Black_Boat):

        - One target per REPAIR (the agent chooses). Adjacency-only — the
          legal-action mask already filters this; the defense-in-depth
          ``adj is boat`` / player-match guards live here too in case a
          hand-crafted action slips in.
        - HP heal = +10 internal (1 AWBW bar). Applied only if the boat's
          player can afford the 10% heal cost AND the target is below
          max HP. On successful heal, funds decrease by the cost.
        - Resupply (fuel + ammo) always fires if the action is legal, even
          when the HP heal is skipped (full HP, unaffordable, or chip
          damage below a bar). Matches the wiki: "Black Boats resupply
          adjacent units regardless of funds."
        - Funds are clamped at 999_999 on write (AWBW treasury cap).
        """
        boat = self.get_unit_at(*action.unit_pos)
        if boat is None or boat.unit_type != UnitType.BLACK_BOAT:
            if oracle_strict:
                raise IllegalActionError(
                    "REPAIR: not a Black Boat or unit missing",
                )
            # Close ACTION/MOVE stage drift: old path returned without
            # finalizing; mirror _finish_action state transitions.
            if boat is not None:
                self._finish_action(boat)
            else:
                self.action_stage = ActionStage.SELECT
                self.selected_unit = None
                self.selected_move_pos = None
            return

        # Move first (boat may have a move_pos different from unit_pos).
        if action.move_pos is not None:
            self._move_unit(boat, action.move_pos)

        if action.target_pos is None:
            self._finish_action(boat)
            return

        target = self.get_unit_at(*action.target_pos)
        if target is None or target.player != boat.player:
            self._finish_action(boat)
            return
        if int(target.unit_id) == int(boat.unit_id):
            # Explicit self-repair guard (``adj is boat``) — cannot happen via
            # orthogonal adjacency in a valid state, but scripted actions can
            # still craft the case and we refuse to heal the boat itself.
            self._finish_action(boat)
            return

        # Adjacency validator: REPAIR only reaches Manhattan-1 neighbours.
        br, bc = boat.pos
        tr, tc = target.pos
        if abs(br - tr) + abs(bc - tc) != 1:
            self._finish_action(boat)
            return

        stats = UNIT_STATS[target.unit_type]
        did_heal = False
        # Display-cap parity with property-day repair (R4 in
        # ``_resupply_on_properties``): internal 91–99 still has display HP 10
        # (bar maxed). AWBW charges **no** gold for a Black Boat HP tick in
        # that band, but still tops internal HP up to 100 (residue under the
        # maxed bar). GL **1635742** env 38: Md.Tank @ 97 internal, PHP
        # ``Repair.funds`` stays 2600g while ``repaired`` shows post-HP display
        # 10; pre-fix engine charged ``_black_boat_heal_cost`` (1600g) and
        # denied the following INF×2 build chain.
        display_hp = (target.hp + 9) // 10
        if target.hp < 100 and display_hp >= 10:
            target.hp = 100
            did_heal = True
        elif target.hp < 100:
            cost = self._black_boat_heal_cost(target.unit_type)
            if self.funds[boat.player] >= cost:
                target.hp = min(100, target.hp + 10)
                self.funds[boat.player] = max(
                    0, min(999_999, self.funds[boat.player] - cost),
                )
                did_heal = True

        # Resupply always fires (wiki rule), even when HP heal is skipped.
        target.fuel = stats.max_fuel
        if stats.max_ammo > 0:
            target.ammo = stats.max_ammo

        self._finish_action(boat)
        self.game_log.append({
            "type":    "repair",
            "player":  boat.player,
            "from":    list(action.unit_pos) if action.unit_pos else None,
            "to":      list(action.move_pos) if action.move_pos else None,
            "target":  list(action.target_pos),
            "healed":  did_heal,
            "new_hp":  target.hp,
        })

    # ------------------------------------------------------------------
    # Pipe seam attack
    # ------------------------------------------------------------------

    def _apply_seam_attack(self, attacker: Unit, action: Action) -> bool:
        """Resolve an ATTACK whose target tile holds an intact pipe seam.

        Returns True when the action was handled as a seam strike (terminator
        already consumed); False when the tile is *not* a seam, leaving the
        caller to continue the normal "no defender" no-op path.

        AWBW seam rules:
        - Seam has 99 HP; tracked in ``GameState.seam_hp``. Missing entries
          default to SEAM_MAX_HP so pre-existing maps don't need init order
          to match.
        - Damage computed by ``calculate_seam_damage`` (no luck, Neo-on-0★
          defense profile; CO/tower ATK still applies).
        - Attacker always consumes one ammo (AWBW); no counterattack is
          possible (seams do not fire back).
        - When HP drops to 0 the tile flips to Broken Seam (115 / 116,
          preserving horizontal vs vertical orientation) and the seam_hp
          entry is cleared. Piperunner traversal rules then treat the tile
          as Plains via ``terrain.py`` lookup (move costs already reflect
          this in ``_plain_costs``).
        """
        target_pos = action.target_pos
        if target_pos is None:
            self._finish_action(attacker)
            return True

        tr, tc = target_pos
        tid = self.map_data.terrain[tr][tc]
        # Broken HPipe/VPipe rubble (115/116): AWBW replays can still list AttackSeam
        # vs rubble many times. Consume ammo and log seam-style damage, but **do not**
        # flip terrain to plain(1) — rubble stays 115/116 (plains-like move costs already
        # apply). Clearing to plain desynced replays that fire again on the same cell.
        if tid in (115, 116):
            att_terrain = get_terrain(self.map_data.terrain[action.move_pos[0]][action.move_pos[1]])
            att_co = self.co_states[attacker.player]
            dmg = calculate_seam_damage(attacker, att_terrain, att_co)
            if dmg is None:
                dmg = 0
            att_stats = UNIT_STATS[attacker.unit_type]
            if att_stats.max_ammo > 0:
                attacker.ammo = max(0, attacker.ammo - 1)
            self._finish_action(attacker)
            self.game_log.append({
                "type":    "attack_seam_rubble",
                "player":  attacker.player,
                "from":    list(action.unit_pos) if action.unit_pos else None,
                "to":      list(action.move_pos) if action.move_pos else None,
                "target":  list(target_pos),
                "dmg":     dmg,
            })
            return True

        if tid not in SEAM_TERRAIN_IDS:
            return False

        att_terrain = get_terrain(self.map_data.terrain[action.move_pos[0]][action.move_pos[1]])
        att_co = self.co_states[attacker.player]

        dmg = calculate_seam_damage(attacker, att_terrain, att_co)
        if dmg is None:
            dmg = 0

        current_hp = self.seam_hp.get(target_pos, SEAM_MAX_HP)
        new_hp = max(0, current_hp - dmg)

        if new_hp <= 0:
            self.map_data.terrain[tr][tc] = SEAM_BROKEN_IDS[tid]
            self.seam_hp.pop(target_pos, None)
        else:
            self.seam_hp[target_pos] = new_hp

        att_stats = UNIT_STATS[attacker.unit_type]
        if att_stats.max_ammo > 0:
            attacker.ammo = max(0, attacker.ammo - 1)

        self._finish_action(attacker)
        self.game_log.append({
            "type":    "attack_seam",
            "player":  attacker.player,
            "from":    list(action.unit_pos) if action.unit_pos else None,
            "to":      list(action.move_pos) if action.move_pos else None,
            "target":  list(target_pos),
            "dmg":     dmg,
            "seam_hp": new_hp,
            "broken":  new_hp <= 0,
        })
        return True

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _apply_load(
        self,
        action: Action,
        *,
        oracle_strict: bool = False,
        oracle_mode: bool = False,
    ):
        unit      = self.get_unit_at(*action.unit_pos)
        transport = self.get_unit_at(*action.move_pos)
        if unit is None or transport is None:
            if oracle_strict:
                raise IllegalActionError(
                    "_apply_load: mover or transport missing at unit_pos or move_pos "
                    f"(unit={unit!r}, transport={transport!r}, action={action!r})"
                )
            return

        # Capacity + compatibility guards (the engine should never produce an
        # illegal LOAD via get_legal_actions, but `step` accepts hand-crafted
        # actions too).
        cap = UNIT_STATS[transport.unit_type].carry_capacity
        if cap <= 0 or unit.unit_type not in get_loadable_into(transport.unit_type):
            raise ValueError(
                f"Illegal LOAD: {unit.unit_type.name} cannot board "
                f"{transport.unit_type.name}."
            )
        if len(transport.loaded_units) >= cap:
            raise ValueError(
                f"Illegal LOAD: transport {transport.unit_type.name} at "
                f"{transport.pos} is full ({len(transport.loaded_units)}/{cap})."
            )

        # Route through _move_unit so reachability validation and fuel deduction
        # apply for normal play. Oracle zips can still encode LOAD after path /
        # occupancy drift (e.g. GL 1629588: tank boards lander on shoal) where
        # ``compute_reachable_costs`` no longer contains ``transport.pos`` even
        # though AWBW committed the board — mirror JOIN/WAIT oracle routing.
        try:
            self._move_unit(unit, transport.pos)
        except ValueError:
            if not oracle_mode:
                raise
            self._move_unit_forced(unit, transport.pos)
        self.units[unit.player].remove(unit)
        transport.loaded_units.append(unit)
        unit.moved = True

        self.action_stage      = ActionStage.SELECT
        self.selected_unit     = None
        self.selected_move_pos = None

        self.game_log.append({
            "type":   "load",
            "player": unit.player,
            "unit":   unit.unit_type.name,
            "into":   transport.unit_type.name,
            "pos":    list(transport.pos),
        })

    # ------------------------------------------------------------------
    # Join (merge same-type allies)
    # ------------------------------------------------------------------

    def _apply_join(self, action: Action, *, oracle_strict: bool = False):
        """Move ``mover`` onto ``partner`` and merge (AWBW join).

        Partner (occupant of ``move_pos``) must be injured; combined HP caps at
        100 internal. Overflow in **display** bars (1–10 each) converts to funds:
        ``(unit_cost // 10) * max(0, d_mover + d_partner - 10)``.
        Fuel and ammo take the max of the two (capped at stats). The mover is
        removed; ``partner`` keeps its ``unit_id`` and tile.
        """
        # Phase 11J-FINAL: mirror _apply_attack/_apply_wait — prefer
        # selected_unit, then oracle-id pin, before falling back to tile lookup.
        # Without this, when two friendly same-type units share a tile via
        # oracle drift (e.g. gid 1632226: two P1 TANKs at (12,6)), get_unit_at
        # returns the first-in-roster for BOTH mover and partner → silent
        # return → action_stage never advances → settle loop infinite-loops.
        mover = self.selected_unit
        if mover is None or mover.pos != action.unit_pos:
            mover = self.get_unit_at_oracle_id(*action.unit_pos, action.select_unit_id)
        partner = None
        if action.move_pos:
            for lst in self.units.values():
                for u in lst:
                    if (
                        u.is_alive
                        and u.pos == action.move_pos
                        and u is not mover
                        and (mover is None or u.player == mover.player)
                    ):
                        partner = u
                        break
                if partner is not None:
                    break
        if mover is None or partner is None or mover is partner:
            if oracle_strict:
                raise IllegalActionError("JOIN: no merge partner at target")
            return
        if not units_can_join(mover, partner):
            raise ValueError(
                f"Illegal JOIN: {mover.unit_type.name} cannot merge with unit at "
                f"{action.move_pos}."
            )

        self._move_unit(mover, partner.pos)

        stats = UNIT_STATS[mover.unit_type]
        combined = min(100, mover.hp + partner.hp)
        excess_bars = max(0, mover.display_hp + partner.display_hp - 10)
        gold_gain = (stats.cost // 10) * excess_bars
        self.funds[mover.player] = min(999_999, self.funds[mover.player] + gold_gain)

        partner.hp = combined
        partner.fuel = min(stats.max_fuel, max(mover.fuel, partner.fuel))
        if stats.max_ammo > 0:
            partner.ammo = min(stats.max_ammo, max(mover.ammo, partner.ammo))
        else:
            partner.ammo = 0

        self.units[mover.player].remove(mover)
        partner.moved = True
        self.action_stage      = ActionStage.SELECT
        self.selected_unit     = None
        self.selected_move_pos = None

        self.game_log.append({
            "type":        "join",
            "player":      mover.player,
            "from":        list(action.unit_pos),
            "to":          list(action.move_pos),
            "gold_gained": gold_gain,
            "hp_after":    partner.hp,
        })

    # ------------------------------------------------------------------
    # Unload
    # ------------------------------------------------------------------

    def _apply_unload(self, action: Action, *, oracle_strict: bool = False):
        """
        Drop one cargo unit from a transport onto an adjacent legal tile.

        ``action.unit_pos``   = transport's pre-move position
        ``action.move_pos``   = transport's destination after its move
        ``action.target_pos`` = drop tile (must be 4-adjacent to ``move_pos``)
        ``action.unit_type``  = which cargo to drop (first matching slot).
                                 If ``None`` and the transport carries a single
                                 cargo, that one is dropped.

        If cargo remains in the transport after the unload, the action stage
        stays ``ACTION`` so the player can drop another cargo or finalize with
        WAIT. With no cargo left, the transport's turn ends here.
        """
        transport = self.get_unit_at(*action.unit_pos)
        if transport is None or action.move_pos is None or action.target_pos is None:
            return

        if not transport.loaded_units:
            return

        # Resolve which cargo to drop.
        cargo_idx: Optional[int] = None
        for i, c in enumerate(transport.loaded_units):
            if action.unit_type is None or c.unit_type == action.unit_type:
                cargo_idx = i
                break
        if cargo_idx is None:
            return

        # Move the transport first (deducts fuel via _move_unit). No-op if
        # the transport is already at move_pos (e.g. on a multi-drop turn).
        if transport.pos != action.move_pos:
            self._move_unit(transport, action.move_pos)

        drop_pos = action.target_pos
        # Drop tile must be 4-adjacent to the transport's current position,
        # legally walkable for the cargo, and empty.
        dr = abs(drop_pos[0] - transport.pos[0])
        dc = abs(drop_pos[1] - transport.pos[1])
        if dr + dc != 1:
            if oracle_strict:
                raise IllegalActionError(
                    "_apply_unload: drop tile not orthogonally adjacent to transport "
                    f"after move (transport={transport!r}, action={action!r})"
                )
            return
        if not (0 <= drop_pos[0] < self.map_data.height and 0 <= drop_pos[1] < self.map_data.width):
            if oracle_strict:
                raise IllegalActionError(
                    "_apply_unload: drop position out of bounds "
                    f"(transport={transport!r}, action={action!r})"
                )
            return
        if self.get_unit_at(*drop_pos) is not None:
            if oracle_strict:
                raise IllegalActionError(
                    "_apply_unload: drop tile occupied "
                    f"(transport={transport!r}, action={action!r})"
                )
            return
        from engine.terrain import INF_PASSABLE
        from engine.weather import effective_move_cost
        cargo_unit = transport.loaded_units[cargo_idx]
        tid = self.map_data.terrain[drop_pos[0]][drop_pos[1]]
        if effective_move_cost(self, cargo_unit, tid) >= INF_PASSABLE:
            if oracle_strict:
                raise IllegalActionError(
                    "_apply_unload: drop terrain impassable for cargo "
                    f"(transport={transport!r}, action={action!r})"
                )
            return

        cargo = transport.loaded_units.pop(cargo_idx)
        cargo.pos = drop_pos
        cco = self.co_states[cargo.player]
        cargo_cls = UNIT_STATS[cargo.unit_type].unit_class
        # Eagle SCOP: unloaded non-footsoldiers may still act this turn (wiki).
        if (
            cco.co_id == 10
            and cco.scop_active
            and cargo_cls not in ("infantry", "mech")
        ):
            cargo.moved = False
        else:
            cargo.moved = True  # dropped units cannot act again this turn
        self.units[cargo.player].append(cargo)

        self.game_log.append({
            "type":   "unload",
            "player": transport.player,
            "from":   list(transport.pos),
            "to":     list(drop_pos),
            "unit":   cargo.unit_type.name,
        })

        # If more cargo aboard, keep the transport "selected" so the player
        # can issue another UNLOAD or finalize with WAIT. Otherwise finish.
        if transport.loaded_units:
            self.selected_unit     = transport
            self.selected_move_pos = transport.pos
            self.action_stage      = ActionStage.ACTION
        else:
            self._finish_action(transport)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _apply_build(self, action: Action, *, oracle_strict: bool = False):
        player = self.active_player
        ut     = action.unit_type
        if ut is None or action.move_pos is None:
            if oracle_strict:
                raise IllegalActionError("BUILD: missing unit_type or move_pos")
            return

        # Defense-in-depth: `get_legal_actions` already filters BUILD to owned
        # factories, but `step` is also callable with hand-constructed Actions
        # (tests, scripted opponents, tools). Refuse to build unless the target
        # tile is a base/airport/port owned by the *active* player — so one
        # player can never place units on the opponent's factories.
        from engine.terrain import get_terrain
        prop = self.get_property_at(*action.move_pos)
        if prop is None or prop.owner != player:
            if oracle_strict:
                raise IllegalActionError(
                    "BUILD: tile is not an owned factory property",
                )
            return
        terrain = get_terrain(self.map_data.terrain[prop.row][prop.col])
        if not (terrain.is_base or terrain.is_airport or terrain.is_port):
            if oracle_strict:
                raise IllegalActionError("BUILD: terrain is not base/airport/port")
            return

        # Unit class must match terrain type: naval on port, air on airport,
        # ground/pipe on base. `get_producible_units` is the canonical rule
        # set; reuse it so we never drift out of sync with action generation.
        if ut not in get_producible_units(terrain, self.map_data.unit_bans):
            if oracle_strict:
                raise IllegalActionError(
                    "BUILD: unit type not producible on this terrain",
                )
            return

        cost = _build_cost(ut, self, player, action.move_pos)
        if self.funds[player] < cost:
            if oracle_strict:
                raise IllegalActionError("BUILD: insufficient funds")
            return

        # Verify factory is empty (important for direct factory builds)
        if self.get_unit_at(*action.move_pos) is not None:
            if oracle_strict:
                raise IllegalActionError("BUILD: factory tile occupied")
            return

        # Unit limit includes cargo aboard transports (see ``alive_owned_unit_count``).
        if alive_owned_unit_count(self.units[player]) >= self.map_data.unit_limit:
            if oracle_strict:
                raise IllegalActionError("BUILD: unit limit reached")
            return

        self.funds[player] -= cost
        self.gold_spent[player] += cost  # Track spending
        stats    = UNIT_STATS[ut]
        co       = self.co_states[player]
        # Eagle SCOP "Lightning Strike": non-footsoldier builds may move the same
        # turn (AWBW wiki). Infantry/mech still spawn exhausted like normal.
        lightning_build_moves_now = (
            co.co_id == 10
            and co.scop_active
            and stats.unit_class not in ("infantry", "mech")
        )
        new_unit = Unit(
            unit_type=ut,
            player=player,
            hp=100,
            ammo=stats.max_ammo if stats.max_ammo > 0 else 0,
            fuel=stats.max_fuel,
            pos=action.move_pos,
            moved=not lightning_build_moves_now,
            loaded_units=[],
            is_submerged=False,
            capture_progress=20,
            unit_id=self._allocate_unit_id(),
        )
        self.units[player].append(new_unit)

        self.action_stage      = ActionStage.SELECT
        self.selected_unit     = None
        self.selected_move_pos = None

        self.game_log.append({
            "type":   "build",
            "player": player,
            "unit":   ut.name,
            "pos":    list(action.move_pos),
            "cost":   cost,
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _move_unit(self, unit: Unit, new_pos: tuple[int, int]):
        """
        Teleport ``unit`` to ``new_pos`` after validating it is a legal
        destination given current terrain, move-type, and fuel rules. Raises
        ``ValueError`` on illegal moves so stray actions (e.g. artillery
        onto a mountain tile) fail loudly rather than corrupting state.

        Deducts the path's movement-point cost from ``unit.fuel``. Movement
        points and fuel use the same scale, so a Lander that spends 4 MP to
        cross 4 sea tiles loses 4 fuel.
        """
        if new_pos == unit.pos:
            return
        costs = compute_reachable_costs(self, unit)
        if new_pos not in costs:
            stats = UNIT_STATS[unit.unit_type]
            tid = self.map_data.terrain[new_pos[0]][new_pos[1]]
            raise ValueError(
                f"Illegal move: {stats.name} (move_type={stats.move_type}) "
                f"from {unit.pos} to {new_pos} (terrain id={tid}, fuel={unit.fuel}) "
                f"is not reachable."
            )
        old_prop = self.get_property_at(*unit.pos)
        if old_prop is not None and old_prop.capture_points < 20:
            old_prop.capture_points = 20
        unit.pos  = new_pos
        unit.fuel = max(0, unit.fuel - costs[new_pos])

    def _move_unit_forced(self, unit: Unit, new_pos: tuple[int, int]):
        """Move ``unit`` to ``new_pos`` **without** reachability validation.

        Only intended for trace-replay tools that need to keep unit positions
        consistent with recorded actions even when the re-executed state has
        drifted (e.g. a blocking enemy unit shifted due to earlier divergence).
        Never call this from the game engine's own action handlers.
        """
        if new_pos == unit.pos:
            return
        old_prop = self.get_property_at(*unit.pos)
        if old_prop is not None and old_prop.capture_points < 20:
            old_prop.capture_points = 20
        unit.pos = new_pos

    def _finish_action(self, unit: Unit):
        unit.moved             = True
        self.action_stage      = ActionStage.SELECT
        self.selected_unit     = None
        self.selected_move_pos = None

    def _resupply_on_properties(self, player: int):
        """Units standing on owned properties are resupplied at start of turn.

        Day repair on valid tiles (HQ / base / city for ground, airport for air,
        port for sea): up to +2 displayed HP (``+20`` internal on the 0–100
        scale). Costs **20% of the unit's deployment cost** for a full +20 HP;
        partial heals (capped by max HP or by insufficient funds) cost the same
        fraction per internal HP (integer gold, minimum 1 when listed cost
        > 0). Labs and comm towers do not grant this heal. CO power heals are
        separate and do not use this path.

        CO modifiers applied here:
          * **Rachel** (co_id 28) D2D — heal **+3 displayed HP** (``+30``
            internal) per property-day instead of +2. Sources (all in
            agreement; PHP wins on disagreement per the Phase 11A Kindle
            precedent):

              - AWBW CO Chart https://awbw.amarriner.com/co.php Rachel row:
                *"Units repair +1 additional HP (note: liable for costs)."*
              - AWBW Fandom Wiki https://awbw.fandom.com/wiki/Rachel
                Day-to-Day: *"Units repair +1 additional HP on properties
                (note: liable for costs)."* (The amarriner.com/wiki/ path
                returns HTTP 404; the Fandom wiki is the canonical
                community wiki.)
              - AWBW Fandom Wiki repair-step rule
                https://awbw.fandom.com/wiki/Changes_in_AWBW: *"Repairs
                will only take place in increments of exactly 20 hitpoints,
                or 2 full visual hitpoints."* Combined with the Rachel
                +1 line ⇒ Rachel heals exactly +30 internal HP (3 visual
                bars).
              - Repair cost rule https://advancewars.fandom.com/wiki/Repairing:
                *"10% cost per 10% health, or 1HP."* ⇒ +30 internal HP =
                30% of deployment cost. The existing helper
                ``_property_day_repair_gold`` is already linear, so no cost
                rework is needed.
              - **PHP cross-check** (``tools/_phase11y_rachel_php_check.py``):
                across 7 Rachel-bearing zips and 5 Andy-control zips from
                the 936-zip GL std pool, AWBW PHP snapshot bar-deltas at
                Rachel turn boundaries are: **43 of 48 positive heals at
                exactly +3 bars** (the remaining 5 explained by HP-cap or
                post-heal combat damage). Andy control: **39 of 39 = +2**.
                Funds parity post-fix: ``funds_delta_by_seat == {0, 0}``
                on 4 of 5 drilled Rachel zips (1622501 / 1623772 / 1624181
                / 1625211), confirming PHP also charges 30% of unit cost
                for the full Rachel band.

            ``data/co_data.json`` Rachel entry (only mentions luck) is
            **not** trusted for D2D repair (chart + wiki + PHP all win).

            Recon: ``docs/oracle_exception_audit/phase11y_co_wave_2.md`` §4
            (10-zip drill showing engine-overstated funds pre-fix) and
            ``phase11y_rachel_impl.md`` (this fix, full PHP audit).
        """
        co = self.co_states[player]
        property_heal = 30 if co.co_id == 28 else 20  # Rachel: +3 bars, others +2

        # Phase 11J-FUNDS-SHIP (R3) — Deterministic iteration order.
        #
        # Sort eligible units by (prop.col, prop.row) ascending —
        # column-major-from-left. With R2 below enforcing all-or-nothing
        # per-unit repair, the iteration order becomes observable when the
        # treasury is exactly straddling the cost of a single full step:
        # the engine and PHP can otherwise pick different units to heal,
        # producing a funds delta even though both spend the same total
        # ±1 step. Without this ordering the Rachel game ``1622501``
        # regresses on R1 + R2 alone (see
        # docs/oracle_exception_audit/phase11j_funds_deep.md §6).
        #
        # Citation (Tier-4, supporting only — not amarriner / not the AWBW
        # Wiki): RPGHQ AWBW Q&A — *"Repair priority is checked by columns
        # (top to bottom) starting from the left. Units which the player
        # doesn't have sufficient funds to repair are skipped."* Documented
        # in docs/oracle_exception_audit/phase11j_f2_koal_fu_oracle_funds.md
        # §2 (Tier-4 supporting note). Required to keep the 100-game gate
        # green under R1 + R2.
        eligible: list[tuple[Unit, PropertyState]] = []
        for unit in self.units[player]:
            prop = self.get_property_at(*unit.pos)
            if prop is None or prop.owner != player:
                continue
            eligible.append((unit, prop))
        eligible.sort(key=lambda up: (up[1].col, up[1].row))

        for unit, prop in eligible:
            stats = UNIT_STATS[unit.unit_type]
            cls   = stats.unit_class

            is_city = not (
                prop.is_hq or prop.is_lab or prop.is_comm_tower
                or prop.is_base or prop.is_airport or prop.is_port
            )
            qualifies_heal = False
            if cls in ("infantry", "mech", "vehicle", "pipe"):
                qualifies_heal = prop.is_hq or prop.is_base or is_city
            elif cls in ("air", "copter"):
                qualifies_heal = prop.is_airport
            elif cls == "naval":
                qualifies_heal = prop.is_port

            if (
                qualifies_heal
                and not prop.is_lab
                and not prop.is_comm_tower
                and unit.hp < 100
            ):
                # Phase 11J-FUNDS-SHIP (R2) — All-or-nothing per-unit step.
                #
                # Compute the FULL step (capped only by the unit's max HP):
                # +20 internal HP for standard COs, +30 for Rachel
                # (co_id 28). If the player can pay for it in full, heal the
                # full step. Otherwise skip the unit entirely — NO partial
                # heals, even if the player can afford a smaller increment.
                # The previous ``while h > 0: h -= 1`` partial-degrade loop
                # violated AWBW canon and silently masked the R1 ordering
                # bug at funds-tight boundaries.
                #
                # Sources (Tier 2, AWBW Wiki):
                #   - "Units" article, *Repairing and Resupplying* /
                #     *Transports* section: *"If a unit is not specifically
                #     at 9HP, repair costs will be calculated only in
                #     increments of 2HP. This can create a fringe scenario
                #     where a unit that is at 8 or less with <20% of the
                #     unit's full value available (such as an 8HP Fighter
                #     on an Airport with less than 4000 funds) will not be
                #     repaired, even if a 1HP repair is technically
                #     affordable."* https://awbw.fandom.com/wiki/Units
                #   - "Advance Wars Overview" Economy section: *"Repairs
                #     are handled similarly [to builds], with money being
                #     deducted depending on the base price of the unit —
                #     if the repairs cannot be afforded, no repairs will
                #     take place."*
                #     https://awbw.fandom.com/wiki/Advance_Wars_Overview
                #   - "Units" article, Black-Boat repair bullet: *"This
                #     repair is liable for costs - if the player cannot
                #     afford the cost to repair the unit, it will only be
                #     resupplied and no repairs will be given."*
                #     https://awbw.fandom.com/wiki/Units
                #
                # Recon: docs/oracle_exception_audit/phase11j_funds_deep.md
                # §4 R2 collected the three cites above; the prior
                # partial-loop matched canon outcome only at the funds=0
                # boundary (cluster A) and diverged everywhere else.
                #
                # Phase 11J-BUILD-NO-OP-CLUSTER-CLOSE (R4, 2026-04-21) —
                # Display-cap repair cost canon for non-Rachel COs.
                #
                # Internal HP scale 0–100 has display HP = ceil(internal/10).
                # AWBW PHP repair canon (per the AWBW Wiki "Units" article
                # already cited above) operates on DISPLAY HP, not internal
                # HP:
                #   * display 10 (internal 91–99): NO REPAIR — bar is
                #     already maxed; PHP refuses and charges 0g. The
                #     pre-R4 engine charged ``(100-hp)% × unit_cost`` for
                #     a phantom +1..+9 internal heal that never appears
                #     on the AWBW HP bar (e.g. Tank at HP 94 → engine
                #     charged 420g for +6 internal HP).
                #   * display 9 (internal 81–90): exactly +1 display bar
                #     (+10 internal HP), cost = 10% of unit cost. The
                #     pre-R4 engine charged ``min(20,100-hp)%`` (e.g.
                #     Tank at HP 85 → engine charged 1050g for +15
                #     internal; canon is 700g for +10 internal to
                #     display 10).
                #   * display ≤ 8 (internal ≤ 80): +2 display bars
                #     (+20 internal HP), cost = 20% of unit cost. The
                #     existing engine path already matches here; full
                #     cost is charged even when capping internal HP at
                #     100.
                #
                # Rachel (co_id 28) is left on the prior internal-cap
                # path — empirically PHP-matched in Phase 11Y across
                # 7 Rachel zips (see Rachel D2D commentary above).
                #
                # PHP cross-check for the new branch — five
                # Sonja-bearing gids in the GL std corpus
                # (1627563/1632289/1634961/1634980/1637338) showed funds
                # drift exactly equal to the predicted display-based
                # delta on the failing turn boundaries. See
                # docs/oracle_exception_audit/phase11j_build_no_op_cluster_close.md
                # for the per-gid breakdown.
                listed = UNIT_STATS[unit.unit_type].cost
                repair_morning_initial_hp = int(unit.hp)
                display_hp = (unit.hp + 9) // 10  # ceil(internal/10)
                if display_hp >= 10:
                    # Display bar already maxed — PHP does not repair
                    # (Rachel included; ``TestR4DisplayCapRepairCanon``).
                    cost = 0
                    step = 0
                elif co.co_id != 28:
                    # Display-8 internal 71–80: PHP uses one +10 tick at 10% first
                    # (gid 1624307 env 36, 73 internal; top of display-8 band 79–80).
                    if display_hp == 9:
                        display_step = 1
                    elif display_hp == 8 and 71 <= unit.hp <= 80:
                        display_step = 1
                    else:
                        display_step = 2
                    cost = max(1, (display_step * 10 * listed) // 100) if listed > 0 else 0
                    step = min(display_step * 10, 100 - unit.hp)
                else:
                    # Phase 11J-FINAL-LASTMILE — Rachel D2D display-based
                    # repair cost canon for **display ≤ 9** (display 10
                    # internal 91–99 is skipped above — same bar-maxed rule
                    # as non-Rachel R4; see ``TestR4DisplayCapRepairCanon``).
                    # Anchor gids / narrative: ``phase11j_build_no_op_cluster_close.md``,
                    # ``phase11j_funds_drift_extermination.md``, Rachel D2D
                    # commentary at top of ``_resupply_on_properties``.
                    display_step = min(3, 10 - display_hp)
                    cost = max(1, (display_step * 10 * listed) // 100) if listed > 0 else 0
                    step = min(display_step * 10, 100 - unit.hp)
                if step > 0 and self.funds[player] >= cost:
                    unit.hp = min(100, unit.hp + step)
                    self.funds[player] = max(
                        0, min(999_999, self.funds[player] - cost),
                    )
                    self.gold_spent[player] += cost
                # Same-morning second +10 when PHP finishes display-8 band (71–80)
                # with a follow-up display-9 tick (80→90→100). Does not run for 73→83
                # (morning ends at 83 internal — not 90).
                if (
                    co.co_id != 28
                    and 71 <= repair_morning_initial_hp <= 80
                    and unit.hp == 90
                    and unit.hp < 100
                ):
                    dh2 = (unit.hp + 9) // 10
                    if dh2 == 9:
                        listed2 = UNIT_STATS[unit.unit_type].cost
                        cost2 = (
                            max(1, (10 * listed2) // 100) if listed2 > 0 else 0
                        )
                        step2 = min(10, 100 - unit.hp)
                        if step2 > 0 and self.funds[player] >= cost2:
                            unit.hp = min(100, unit.hp + step2)
                            self.funds[player] = max(
                                0,
                                min(999_999, self.funds[player] - cost2),
                            )
                            self.gold_spent[player] += cost2

            # Resupply (fuel/ammo) runs even when the heal is skipped — the
            # AWBW canon line above explicitly preserves resupply on the
            # all-or-nothing branch ("it will only be resupplied and no
            # repairs will be given").
            unit.fuel = stats.max_fuel
            if stats.max_ammo > 0:
                unit.ammo = stats.max_ammo

    # ------------------------------------------------------------------
    # Win condition checks
    # ------------------------------------------------------------------

    def _check_win_conditions(self, acting_player: int) -> float:
        """
        Evaluate all win conditions.
        Returns reward from acting_player's perspective.
        0 = ongoing | +1 = acting_player wins | -1 = acting_player loses.
        """
        if self.done:
            if self.winner == -1:
                return 0.0
            return 1.0 if self.winner == acting_player else -1.0

        # Unit wipe is evaluated in _evaluate_army_wipe_after_combat (attack path only).

        # Cap limit
        for p in (0, 1):
            if self.count_properties(p) >= self.map_data.cap_limit:
                self.done       = True
                self.winner     = p
                self.win_reason = "cap_limit"
                return 1.0 if p == acting_player else -1.0

        # HQ / Lab capture
        if self.map_data.objective_type == "hq":
            for p in (0, 1):
                opp          = 1 - p
                starting_hqs = self.map_data.hq_positions.get(p, [])
                if starting_hqs and all(
                    (pr := self.get_property_at(*pos)) is not None and pr.owner == opp
                    for pos in starting_hqs
                ):
                    self.done       = True
                    self.winner     = opp
                    self.win_reason = "hq_capture"
                    return 1.0 if opp == acting_player else -1.0

        elif self.map_data.objective_type == "lab":
            for p in (0, 1):
                opp           = 1 - p
                starting_labs = self.map_data.lab_positions.get(p, [])
                if starting_labs and all(
                    (pr := self.get_property_at(*pos)) is not None and pr.owner == opp
                    for pos in starting_labs
                ):
                    self.done       = True
                    self.winner     = opp
                    self.win_reason = "lab_capture"
                    return 1.0 if opp == acting_player else -1.0

        return 0.0

    # ------------------------------------------------------------------
    # Legal action proxy
    # ------------------------------------------------------------------

    def legal_actions(self) -> list[Action]:
        return get_legal_actions(self)

    # ------------------------------------------------------------------
    # ASCII renderer
    # ------------------------------------------------------------------

    def render_ascii(self) -> str:
        TERRAIN_CHARS: dict[int, str] = {
            1: '.', 2: '^', 3: 'T',
            28: '~', 29: 's', 30: 'r',
        }
        for tid in range(4, 28):
            if tid not in TERRAIN_CHARS:
                TERRAIN_CHARS[tid] = '='   # roads / rivers / bridges

        UNIT_CHARS = {
            'infantry': 'i', 'mech': 'm', 'vehicle': 'v',
            'copter':   'c', 'air':  'a', 'naval':   'n',
            'pipe':     'p',
        }

        header = (
            f"Turn {self.turn} | Player {self.active_player} | "
            f"Stage {self.action_stage.name} | "
            f"Funds: {self.funds[0]:,} / {self.funds[1]:,}"
        )
        lines = [header, "-" * len(header)]

        for r in range(self.map_data.height):
            row_str = ""
            for c in range(self.map_data.width):
                unit = self.get_unit_at(r, c)
                if unit:
                    cls = UNIT_STATS[unit.unit_type].unit_class
                    ch  = UNIT_CHARS.get(cls, '?')
                    row_str += ch.upper() if unit.player == 0 else ch
                else:
                    tid     = self.map_data.terrain[r][c]
                    prop    = self.get_property_at(r, c)
                    if prop is not None:
                        if prop.owner == 0:
                            row_str += 'O'
                        elif prop.owner == 1:
                            row_str += 'X'
                        else:
                            row_str += 'P'
                    else:
                        row_str += TERRAIN_CHARS.get(tid, '?')
            lines.append(row_str)

        lines.append(
            f"P0 units: {len(self.units[0])} | P1 units: {len(self.units[1])} | "
            f"P0 props: {self.count_properties(0)} | P1 props: {self.count_properties(1)}"
        )
        return "\n".join(lines)

    def _mcts_trace_pre_action(self, action: Optional[Action]) -> dict:
        """Capture cheap pre-step metadata for MCTS stochastic-risk tracing.

        This is intentionally advisory telemetry: it must not affect game rules.
        ``apply_full_turn(return_trace=True)`` uses it to classify combat/capture
        threshold events without forcing every normal engine step to deep-copy.
        """
        pre: dict = {
            "action_type": action.action_type.name if action is not None else None,
            "active_player": int(self.active_player),
            "turn": int(self.turn),
            "stage": self.action_stage.name,
        }
        if action is None:
            return pre
        if action.action_type == ActionType.ATTACK:
            attacker = None
            if self.selected_unit is not None and self.selected_unit.is_alive:
                attacker = self.selected_unit
            elif action.unit_pos is not None:
                attacker = self.get_unit_at(*action.unit_pos)
            defender = self.get_unit_at(*action.target_pos) if action.target_pos is not None else None
            pre.update({
                "attacker_id": int(attacker.unit_id) if attacker is not None else None,
                "attacker_player": int(attacker.player) if attacker is not None else None,
                "attacker_type": attacker.unit_type.name if attacker is not None else None,
                "attacker_pre_hp": int(attacker.hp) if attacker is not None else None,
                "attacker_pre_display_hp": int(attacker.display_hp) if attacker is not None else None,
                "defender_id": int(defender.unit_id) if defender is not None else None,
                "defender_player": int(defender.player) if defender is not None else None,
                "defender_type": defender.unit_type.name if defender is not None else None,
                "defender_pre_hp": int(defender.hp) if defender is not None else None,
                "defender_pre_display_hp": int(defender.display_hp) if defender is not None else None,
                "target_pos": list(action.target_pos) if action.target_pos is not None else None,
                "move_pos": list(action.move_pos) if action.move_pos is not None else None,
            })
            if defender is not None:
                prop = self.get_property_at(*defender.pos)
                pre["defender_property_capture_points"] = int(prop.capture_points) if prop is not None else None
            if attacker is not None:
                prop = self.get_property_at(*attacker.pos)
                pre["attacker_property_capture_points"] = int(prop.capture_points) if prop is not None else None
        elif action.action_type == ActionType.CAPTURE:
            unit = self.selected_unit
            if unit is None and action.move_pos is not None:
                unit = self.get_unit_at(*action.move_pos)
            prop = self.get_property_at(*(action.move_pos or unit.pos)) if unit is not None else None
            pre.update({
                "capturer_id": int(unit.unit_id) if unit is not None else None,
                "capturer_pre_hp": int(unit.hp) if unit is not None else None,
                "capturer_pre_display_hp": int(unit.display_hp) if unit is not None else None,
                "capture_points_pre": int(prop.capture_points) if prop is not None else None,
                "property_owner_pre": int(prop.owner) if prop is not None and prop.owner is not None else None,
            })
        return pre

    def _mcts_trace_post_action(
        self,
        action: Action,
        pre: dict,
        reward: float,
        done: bool,
    ) -> dict:
        item = dict(pre)
        item.update({"reward": float(reward), "done": bool(done)})
        critical = False
        if action.action_type == ActionType.ATTACK:
            attacker_id = pre.get("attacker_id")
            defender_id = pre.get("defender_id")
            attacker = None
            defender = None
            for player_units in self.units.values():
                for u in player_units:
                    if attacker_id is not None and int(u.unit_id) == int(attacker_id):
                        attacker = u
                    if defender_id is not None and int(u.unit_id) == int(defender_id):
                        defender = u
            attacker_post_hp = int(attacker.hp) if attacker is not None else 0 if attacker_id is not None else None
            defender_post_hp = int(defender.hp) if defender is not None else 0 if defender_id is not None else None
            defender_killed = bool(defender_id is not None and defender_post_hp == 0)
            attacker_killed = bool(attacker_id is not None and attacker_post_hp == 0)
            dmg = None
            counter = None
            if pre.get("defender_pre_hp") is not None and defender_post_hp is not None:
                dmg = max(0, int(pre["defender_pre_hp"]) - int(defender_post_hp))
            if pre.get("attacker_pre_hp") is not None and attacker_post_hp is not None:
                counter = max(0, int(pre["attacker_pre_hp"]) - int(attacker_post_hp))
            capture_interrupted = False
            if defender_killed and action.target_pos is not None:
                prop = self.get_property_at(*action.target_pos)
                capture_interrupted = bool(prop is not None and pre.get("defender_property_capture_points") is not None and int(pre["defender_property_capture_points"]) < 20 and prop.capture_points == 20)
            if attacker_killed and pre.get("move_pos") is not None:
                prop = self.get_property_at(*tuple(pre["move_pos"]))
                capture_interrupted = capture_interrupted or bool(prop is not None and pre.get("attacker_property_capture_points") is not None and int(pre["attacker_property_capture_points"]) < 20 and prop.capture_points == 20)
            survived_low = (
                (defender_id is not None and defender_post_hp is not None and 0 < defender_post_hp <= 10)
                or (attacker_id is not None and attacker_post_hp is not None and 0 < attacker_post_hp <= 10)
            )
            killed_from_low_margin = defender_killed or attacker_killed
            critical = bool(survived_low or killed_from_low_margin or capture_interrupted)
            item.update({
                "attack_damage_roll": dmg,
                "counter_damage_roll": counter,
                "defender_post_hp": defender_post_hp,
                "attacker_post_hp": attacker_post_hp,
                "defender_killed": defender_killed,
                "attacker_killed": attacker_killed,
                "capture_interrupted": capture_interrupted,
                "critical_threshold_event": critical,
            })
        elif action.action_type == ActionType.CAPTURE:
            unit_id = pre.get("capturer_id")
            unit = None
            for player_units in self.units.values():
                for u in player_units:
                    if unit_id is not None and int(u.unit_id) == int(unit_id):
                        unit = u
                        break
            prop = self.get_property_at(*(action.move_pos or unit.pos)) if unit is not None else None
            item.update({
                "capture_points_post": int(prop.capture_points) if prop is not None else None,
                "property_owner_post": int(prop.owner) if prop is not None and prop.owner is not None else None,
                "capture_interrupted": False,
                "critical_threshold_event": False,
            })
        else:
            item.update({
                "capture_interrupted": False,
                "critical_threshold_event": False,
            })
        return item

    def apply_full_turn(
        self,
        plan_or_policy: Union[list[Action], Callable[["GameState"], Action]],
        *,
        copy: bool = True,
        max_actions: int = 10_000,
        rng_seed: Optional[int] = None,
        on_step: Optional[Callable[[GameState, Action, float, bool], None]] = None,
        return_trace: bool = False,
    ) -> tuple[GameState, list[Action], float, bool] | tuple[GameState, list[Action], float, bool, list[dict]]:
        """
        Apply one full turn for ``self.active_player`` and return the resulting
        state at the start of the next turn (or terminal).

        Phase 11a: foundation for MCTS turn-level rollouts.

        Args:
            plan_or_policy:
                - ``list[Action]``: a fixed plan; consumed in order. If the plan
                  exhausts before the turn ends, raises ``RuntimeError``.
                - ``Callable[[GameState], Action]``: invoked at each sub-step with
                  the current state; must return a legal action (validated against
                  ``get_legal_actions``).
            copy:
                If True (default), the input state is deep-copied; the caller's
                state is NOT mutated. If False, this method mutates ``self`` in
                place — the caller must own the state and be ready for it to
                change. Set False inside MCTS for hot-path performance.
            max_actions:
                Hard cap on sub-steps inside the turn. Prevents infinite loops if
                a buggy policy gets stuck. Default 10_000 is generous for AWBW
                (a turn is typically 5-50 sub-steps).
            rng_seed:
                If not None, ``random.seed(rng_seed)`` is called BEFORE the rollout
                and ``state.luck_rng`` is replaced with ``random.Random(rng_seed)``
                so combat luck matches the seed (combat does not use the module
                RNG). Saved-and-restored so the global RNG state is preserved
                across the call.
            on_step:
                Optional callback ``(state_after, action, reward, done)`` invoked
                after each sub-step. Use for tree-search bookkeeping or tracing.

        Returns:
            ``(final_state, actions_taken, total_reward, done)``:
                * ``final_state``: the resulting state. If ``copy=True``, this is
                  a different object from the input; if ``copy=False``, it IS
                  ``self``.
                * ``actions_taken``: list of Actions actually applied.
                * ``total_reward``: sum of per-step rewards from ``self.active_player``'s
                  perspective at the START of the turn (the player whose turn
                  this rollout simulated). Note: ``state.step`` returns reward
                  from the *active* player at the time of the call; we accumulate
                  with the SIGN that the starting player would see (we're always
                  the active player during this rollout, so just sum directly).
                * ``done``: True if the game ended during this turn.

        Raises:
            ValueError: if the input state's ``action_stage`` is not SELECT
                        (we only support starting at a clean turn boundary).
            RuntimeError: if a plan exhausts before the turn ends, or if
                          ``max_actions`` is exceeded.
            IllegalActionError: propagated from ``state.step`` if the policy
                                picks an illegal action (caller must catch).
        """
        if self.action_stage != ActionStage.SELECT:
            raise ValueError(
                f"apply_full_turn requires action_stage==SELECT at entry, "
                f"got {self.action_stage.name}"
            )

        state = _copy_mod.deepcopy(self) if copy else self
        starting_player = state.active_player

        saved_rng_state: Optional[object] = None
        if rng_seed is not None:
            saved_rng_state = _random_mod.getstate()
            _random_mod.seed(rng_seed)
            # Combat luck draws from ``state.luck_rng``, not the module RNG.
            state.luck_rng = _random_mod.Random(int(rng_seed))

        actions_taken: list[Action] = []
        total_reward = 0.0
        done = False
        turn_trace: list[dict] = []

        if isinstance(plan_or_policy, list):
            plan_iter = iter(plan_or_policy)

            def get_action(_s: GameState) -> Optional[Action]:
                return next(plan_iter, None)
        else:
            policy = plan_or_policy

            def get_action(s: GameState) -> Action:
                return policy(s)

        try:
            for _ in range(max_actions):
                if done:
                    break
                if state.active_player != starting_player:
                    break
                action = get_action(state)
                trace_pre: dict | None = None
                if return_trace:
                    trace_pre = state._mcts_trace_pre_action(action)
                if action is None:
                    raise RuntimeError(
                        "plan exhausted before turn ended; "
                        f"actions_taken={len(actions_taken)} "
                        f"active_player still {state.active_player}, "
                        f"stage={state.action_stage.name}"
                    )
                _, reward, done = state.step(action)
                actions_taken.append(action)
                total_reward += float(reward)
                if return_trace and trace_pre is not None:
                    turn_trace.append(state._mcts_trace_post_action(action, trace_pre, float(reward), done))
                if on_step is not None:
                    on_step(state, action, float(reward), done)
                if done or state.active_player != starting_player:
                    break
            else:
                raise RuntimeError(
                    f"apply_full_turn exceeded max_actions={max_actions}; "
                    f"actions_taken={len(actions_taken)} active={state.active_player}"
                )
        finally:
            if saved_rng_state is not None:
                _random_mod.setstate(saved_rng_state)

        if return_trace:
            return state, actions_taken, total_reward, done, turn_trace
        return state, actions_taken, total_reward, done


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_initial_state(
    map_data: MapData,
    p0_co_id: int,
    p1_co_id: int,
    starting_funds: int = 0,
    tier_name: str = "T2",
    default_weather: str = "clear",
    *,
    replay_first_mover: Optional[int] = None,
    max_turns: Optional[int] = None,
    max_days: Optional[int] = None,
    luck_seed: Optional[int] = None,
    spirit_map_is_std: Optional[bool] = True,
) -> GameState:

    # Treasuries always start at 0g in AWBW; the opening player receives income
    # at the start of their first turn via _grant_income below. ``starting_funds``
    # stays as a parameter only for non-AWBW experiments (tests/handicaps).
    """
    Create a fresh GameState for a new game.

    Uses make_co_state_safe so it works even before co_data.json is generated.

    ``max_days`` (preferred) or ``max_turns`` (deprecated alias) set the end-inclusive
    calendar day cap (default ``MAX_TURNS``). Pass at most one of them; must be >= 1.

    ``luck_seed`` seeds :attr:`GameState.luck_rng` for combat luck when not ``None``;
    otherwise a fresh :class:`random.Random` is used (isolated from other games).

    ``spirit_map_is_std`` gates spirit-broken when ``AWBW_SPIRIT_REQUIRE_STD`` is on;
    :class:`AWBWEnv` overwrites from the map pool. Default ``True`` for standalone engine/tests.
    """
    if max_turns is not None and max_days is not None:
        if int(max_turns) != int(max_days):
            raise ValueError("make_initial_state: pass only one of max_turns and max_days")
        mt = int(max_turns)
    elif max_days is not None:
        mt = int(max_days)
    elif max_turns is not None:
        mt = int(max_turns)
    else:
        mt = MAX_TURNS
    if mt < 1:
        raise ValueError(f"max_days/max_turns must be >= 1, got {mt}")

    props = _copy_mod.deepcopy(map_data.properties)

    units = specs_to_initial_units(map_data.predeployed_specs)

    # Deep-copy the mutable pieces of ``map_data`` that the engine writes to
    # during play. Terrain flips on seam break (113/114 → 115/116) must not
    # leak across games that share the same loaded MapData instance.
    # Properties are already cloned above; the terrain grid is the only other
    # in-place mutation the engine performs.
    map_data = _copy_mod.copy(map_data)
    map_data.terrain = [row[:] for row in map_data.terrain]

    # Initialise seam HP (99 per intact HPipe/VPipe seam tile).
    seam_hp: dict[tuple[int, int], int] = {}
    for r, row in enumerate(map_data.terrain):
        for c, tid in enumerate(row):
            if tid in (113, 114):
                seam_hp[(r, c)] = 99

    # AWBW opening rule (skirmish / GL default): asymmetric predeploy — the
    # empty side opens. Tie → P0. When replaying a site zip, pass
    # ``replay_first_mover=0|1`` so the first ``p:`` envelope matches
    # ``active_player`` (hosted games may not follow the predeploy heuristic).
    if replay_first_mover is not None:
        if replay_first_mover not in (0, 1):
            raise ValueError(f"replay_first_mover must be 0 or 1, got {replay_first_mover}")
        opening = replay_first_mover
    else:
        n0, n1 = len(units[0]), len(units[1])
        if n0 == 0 and n1 > 0:
            opening = 0
        elif n1 == 0 and n0 > 0:
            opening = 1
        else:
            opening = 0

    luck0 = (
        _random_mod.Random(int(luck_seed))
        if luck_seed is not None
        else _random_mod.Random()
    )
    state = GameState(
        map_data=map_data,
        units=units,
        funds=[starting_funds, starting_funds],
        co_states=[
            make_co_state_safe(p0_co_id),
            make_co_state_safe(p1_co_id),
        ],
        properties=props,
        turn=1,
        active_player=opening,
        action_stage=ActionStage.SELECT,
        selected_unit=None,
        selected_move_pos=None,
        done=False,
        winner=None,
        win_reason=None,
        game_log=[],
        tier_name=tier_name,
        max_turns=mt,
        full_trace=[],
        seam_hp=seam_hp,
        weather=default_weather,
        default_weather=default_weather,
        co_weather_segments_remaining=0,
        luck_rng=luck0,
        spirit=SpiritState(),
        spirit_map_is_std=spirit_map_is_std,
    )
    state._refresh_comm_towers()

    # Stamp every predeployed unit with a stable id. Iterate in a deterministic
    # order so the assignment is reproducible across runs / replays.
    for player in (0, 1):
        for u in state.units[player]:
            u.unit_id = state._allocate_unit_id()

    # Day 1 income goes to whoever opens; the other player collects on their
    # first turn via the normal _end_turn path.
    state._grant_income(opening)

    return state