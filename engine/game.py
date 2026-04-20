"""
AWBW GameState: complete mutable game state and transition logic.

step(action) → (state, reward, done)
  reward: +1.0 (active player wins) | -1.0 (active player loses) | 0.0 (ongoing/draw)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from engine.unit import Unit, UnitType, UNIT_STATS, idle_start_of_day_fuel_drain
from engine.terrain import get_terrain, property_terrain_id_after_owner_change
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

# Pipe seam constants (AWBW canonical).
SEAM_TERRAIN_IDS: tuple[int, int] = (113, 114)       # HPipe Seam / VPipe Seam
SEAM_BROKEN_IDS:  dict[int, int]  = {113: 115, 114: 116}  # → HPipe Rubble / VPipe Rubble
SEAM_MAX_HP: int = 99

MAX_TURNS = 100   # after this, winner = player with more properties; tie if equal

# Dense shaping for CAPTURE (acting-player frame). Kept small vs terminal ±1.0.
# Coefficients raised after the P0-skew investigation (plan p0-capture-architecture-fix):
# the previous values were dominated by the per-step property-diff penalty in
# rl/env.py, leaving the learner with no positive discovery signal.
_CAPTURE_SHAPING_PROGRESS: float = 0.04  # per 20 capture_points reduced toward flip
_CAPTURE_SHAPING_COMPLETE: float = 0.20  # bonus when ownership flips to capturer
# One-shot bonus the first time a given unit attempts CAPTURE in an episode.
# Rewards the *behavior* (issuing CAPTURE) so the SELECT->MOVE->CAPTURE chain
# gets a positive credit-assignment signal even before the tile flips.
_CAPTURE_FIRST_ATTEMPT_BONUS: float = 0.01

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
    win_reason:        Optional[str]           # why the game ended (set when done becomes True)
    game_log:          list[dict]              # append-only action history (resolved actions only)
    tier_name:         str
    full_trace:        list[dict] = field(default_factory=list)  # every action incl. SELECT/END_TURN
    
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

    def _allocate_unit_id(self) -> int:
        uid = self.next_unit_id
        self.next_unit_id += 1
        return uid

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
        """Sync each CO's comm_towers count from current property ownership."""
        for player in (0, 1):
            self.co_states[player].comm_towers = sum(
                1 for p in self.properties if p.owner == player and p.is_comm_tower
            )

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self, action: Action) -> tuple[GameState, float, bool]:
        """
        Apply action in-place, return (self, reward, done).
        reward is from the perspective of active_player at the time of the call.
        """
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
            self._apply_attack(action)

        elif action.action_type == ActionType.CAPTURE:
            capture_shaping = self._apply_capture(action)

        elif action.action_type == ActionType.WAIT:
            self._apply_wait(action)

        elif action.action_type == ActionType.DIVE_HIDE:
            self._apply_dive_hide(action)

        elif action.action_type == ActionType.LOAD:
            self._apply_load(action)

        elif action.action_type == ActionType.JOIN:
            self._apply_join(action)

        elif action.action_type == ActionType.UNLOAD:
            self._apply_unload(action)

        elif action.action_type == ActionType.BUILD:
            self._apply_build(action)

        elif action.action_type == ActionType.REPAIR:
            self._apply_repair(action)

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

        # Tick CO-induced weather.  Each end-turn consumes one segment; when the
        # counter reaches 0 the weather reverts to the map default.
        if self.co_weather_segments_remaining > 0:
            self.co_weather_segments_remaining -= 1
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
            if self.turn > MAX_TURNS:
                self.done = True
                p0_props = self.count_properties(0)
                p1_props = self.count_properties(1)
                if p0_props > p1_props:
                    self.winner = 0
                    self.win_reason = "max_turns_tiebreak"
                elif p1_props > p0_props:
                    self.winner = 1
                    self.win_reason = "max_turns_tiebreak"
                else:
                    self.winner = -1
                    self.win_reason = "max_turns_draw"
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
                if not refuel_exempt:
                    unit.hp = 0   # crash / sink

        # Remove units that crashed on fuel starvation
        self.units[opponent] = [u for u in self.units[opponent] if u.is_alive]

        # Resupply units on APC-adjacent tiles: handled in _apply_wait
        # Resupply on ports/airports
        self._resupply_on_properties(opponent)

        # Collect income
        self._grant_income(opponent)

        # Refresh tower counts now that ownership may have changed this turn
        self._refresh_comm_towers()

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
        co.power_bar = 0
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

        # Hawke: Black Wave (COP) 1 enemy HP / Black Storm (SCOP) 2 enemy HP;
        # both heal own units +2 HP (co_data.json / AWBW wiki).
        elif co.co_id == 12:
            enemy_loss = 10 if cop else 20
            for u in self.units[player]:
                u.hp = min(100, u.hp + 20)
            for u in self.units[opponent]:
                u.hp = max(1, u.hp - enemy_loss)

        # Sensei COP: spawn Mech on every owned base without a unit
        elif co.co_id == 13 and cop:
            for prop in self.properties:
                if prop.owner == player and prop.is_base:
                    if self.get_unit_at(prop.row, prop.col) is None:
                        mech = Unit(
                            unit_type=UnitType.MECH,
                            player=player,
                            hp=100,
                            ammo=UNIT_STATS[UnitType.MECH].max_ammo,
                            fuel=UNIT_STATS[UnitType.MECH].max_fuel,
                            pos=(prop.row, prop.col),
                            moved=True,
                            loaded_units=[],
                            is_submerged=False,
                            capture_progress=20,
                            unit_id=self._allocate_unit_id(),
                        )
                        self.units[player].append(mech)

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

        # Sasha COP: drain power bar of enemy CO
        elif co.co_id == 19 and cop:
            self.co_states[opponent].power_bar = max(
                0, self.co_states[opponent].power_bar - (self.count_properties(player) * 9000)
            )

        # Von Bolt SCOP (Ex Machina): simplified global 3 HP to all enemies
        # (live AWBW is an AOE strike; stun not modeled here).
        elif co.co_id == 30 and not cop:
            for u in self.units[opponent]:
                u.hp = max(1, u.hp - 30)

        # Prune dead units from power effects
        for p in (0, 1):
            self.units[p] = [u for u in self.units[p] if u.is_alive]

    # ------------------------------------------------------------------
    # Attack
    # ------------------------------------------------------------------

    def _apply_attack(self, action: Action):
        attacker = self.get_unit_at(*action.unit_pos)
        if attacker is None:
            return

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
        if override_dmg is not None:
            dmg = max(0, int(override_dmg))
        else:
            dmg = calculate_damage(
                attacker, defender,
                att_terrain, def_terrain,
                att_co, def_co,
            )
        if dmg is not None:
            defender.hp = max(0, defender.hp - dmg)
            self.losses_hp[defender.player] += dmg  # Track HP lost
            if defender.hp == 0:
                self.losses_units[defender.player] += 1  # Track unit destroyed
            self._charge_power(attacker.player, defender.player, dmg)
        else:
            dmg = 0

        # Counterattack (only if defender survived and attacker is direct)
        att_stats = UNIT_STATS[attacker.unit_type]
        if defender.is_alive and not att_stats.is_indirect:
            if override_counter is not None:
                counter = max(0, int(override_counter))
            else:
                counter = calculate_counterattack(
                    attacker, defender,
                    att_terrain, def_terrain,
                    att_co, def_co,
                    attack_damage=dmg,
                )
            if counter is not None and counter > 0:
                attacker.hp = max(0, attacker.hp - counter)
                self.losses_hp[attacker.player] += counter  # Track counterattack HP lost
                if attacker.hp == 0:
                    self.losses_units[attacker.player] += 1  # Track unit destroyed
                self._charge_power(defender.player, attacker.player, counter)

        # Consume attacker ammo
        att_stats = UNIT_STATS[attacker.unit_type]
        if att_stats.max_ammo > 0:
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

    def _charge_power(self, attacker_player: int, defender_player: int, damage: int):
        """Charge CO power bars. Attacker gains less than defender (approx AWBW rates)."""
        self.co_states[attacker_player].power_bar += damage * 18
        self.co_states[defender_player].power_bar += damage * 27

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
        unit = self.get_unit_at(*action.unit_pos)
        if unit is None:
            return 0.0

        self._move_unit(unit, action.move_pos)
        prop = self.get_property_at(*action.move_pos)
        if prop is None:
            self._finish_action(unit)
            return 0.0

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
        uid = int(getattr(unit, "unit_id", 0) or 0)
        if uid > 0 and uid not in self.capture_attempted_unit_ids:
            self.capture_attempted_unit_ids.add(uid)
            shaping += _CAPTURE_FIRST_ATTEMPT_BONUS
        if contest:
            reduced = float(old_cp - max(prop.capture_points, 0))
            shaping += _CAPTURE_SHAPING_PROGRESS * (reduced / 20.0)

        if prop.capture_points <= 0:
            if contest:
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
        self.game_log.append({
            "type":         "capture",
            "player":       unit.player,
            "from":         action.unit_pos,
            "to":           list(action.move_pos),
            "cp_remaining": prop.capture_points,
        })
        return shaping

    # ------------------------------------------------------------------
    # Wait
    # ------------------------------------------------------------------

    def _apply_wait(self, action: Action):
        unit = self.get_unit_at(*action.unit_pos)
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

    def _apply_repair(self, action: Action):
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
        if target.hp < 100:
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

    def _apply_load(self, action: Action):
        unit      = self.get_unit_at(*action.unit_pos)
        transport = self.get_unit_at(*action.move_pos)
        if unit is None or transport is None:
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

        # Route through _move_unit so the standard reachability validation
        # (and fuel deduction) applies.
        self._move_unit(unit, transport.pos)
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

    def _apply_join(self, action: Action):
        """Move ``mover`` onto ``partner`` and merge (AWBW join).

        Partner (occupant of ``move_pos``) must be injured; combined HP caps at
        100 internal. Overflow in **display** bars (1–10 each) converts to funds:
        ``(unit_cost // 10) * max(0, d_mover + d_partner - 10)``.
        Fuel and ammo take the max of the two (capped at stats). The mover is
        removed; ``partner`` keeps its ``unit_id`` and tile.
        """
        mover = self.get_unit_at(*action.unit_pos)
        partner = self.get_unit_at(*action.move_pos) if action.move_pos else None
        if mover is None or partner is None or mover is partner:
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

    def _apply_unload(self, action: Action):
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
            return
        if not (0 <= drop_pos[0] < self.map_data.height and 0 <= drop_pos[1] < self.map_data.width):
            return
        if self.get_unit_at(*drop_pos) is not None:
            return
        from engine.terrain import INF_PASSABLE
        from engine.weather import effective_move_cost
        cargo_unit = transport.loaded_units[cargo_idx]
        tid = self.map_data.terrain[drop_pos[0]][drop_pos[1]]
        if effective_move_cost(self, cargo_unit, tid) >= INF_PASSABLE:
            return

        cargo = transport.loaded_units.pop(cargo_idx)
        cargo.pos   = drop_pos
        cargo.moved = True   # dropped units cannot act again this turn
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

    def _apply_build(self, action: Action):
        player = self.active_player
        ut     = action.unit_type
        if ut is None or action.move_pos is None:
            return

        # Defense-in-depth: `get_legal_actions` already filters BUILD to owned
        # factories, but `step` is also callable with hand-constructed Actions
        # (tests, scripted opponents, tools). Refuse to build unless the target
        # tile is a base/airport/port owned by the *active* player — so one
        # player can never place units on the opponent's factories.
        from engine.terrain import get_terrain
        prop = self.get_property_at(*action.move_pos)
        if prop is None or prop.owner != player:
            return
        terrain = get_terrain(self.map_data.terrain[prop.row][prop.col])
        if not (terrain.is_base or terrain.is_airport or terrain.is_port):
            return

        # Unit class must match terrain type: naval on port, air on airport,
        # ground/pipe on base. `get_producible_units` is the canonical rule
        # set; reuse it so we never drift out of sync with action generation.
        if ut not in get_producible_units(terrain, self.map_data.unit_bans):
            return

        cost = _build_cost(ut, self, player, action.move_pos)
        if self.funds[player] < cost:
            return

        # Verify factory is empty (important for direct factory builds)
        if self.get_unit_at(*action.move_pos) is not None:
            return

        self.funds[player] -= cost
        self.gold_spent[player] += cost  # Track spending
        stats    = UNIT_STATS[ut]
        new_unit = Unit(
            unit_type=ut,
            player=player,
            hp=100,
            ammo=stats.max_ammo if stats.max_ammo > 0 else 0,
            fuel=stats.max_fuel,
            pos=action.move_pos,
            moved=True,  # Newly built units can't move this turn
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
        """
        property_heal = 20  # +2 display HP
        for unit in self.units[player]:
            prop = self.get_property_at(*unit.pos)
            if prop is None or prop.owner != player:
                continue

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
                desired = min(property_heal, 100 - unit.hp)
                funds   = self.funds[player]
                h       = desired
                while h > 0:
                    cost = _property_day_repair_gold(h, unit.unit_type)
                    if funds >= cost:
                        unit.hp = min(100, unit.hp + h)
                        self.funds[player] = max(
                            0, min(999_999, self.funds[player] - cost),
                        )
                        self.gold_spent[player] += cost
                        break
                    h -= 1

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
) -> GameState:
    # Treasuries always start at 0g in AWBW; the opening player receives income
    # at the start of their first turn via _grant_income below. ``starting_funds``
    # stays as a parameter only for non-AWBW experiments (tests/handicaps).
    """
    Create a fresh GameState for a new game.

    Uses make_co_state_safe so it works even before co_data.json is generated.
    """
    props = copy.deepcopy(map_data.properties)
    units = specs_to_initial_units(map_data.predeployed_specs)

    # Deep-copy the mutable pieces of ``map_data`` that the engine writes to
    # during play. Terrain flips on seam break (113/114 → 115/116) must not
    # leak across games that share the same loaded MapData instance.
    # Properties are already cloned above; the terrain grid is the only other
    # in-place mutation the engine performs.
    map_data = copy.copy(map_data)
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
        full_trace=[],
        seam_hp=seam_hp,
        weather=default_weather,
        default_weather=default_weather,
        co_weather_segments_remaining=0,
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
