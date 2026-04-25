"""Regression: attacker post-envelope HP pin vs same-envelope join (gid 1611364).

``_oracle_post_envelope_units_by_id`` is built from ``frames[env_i+1]`` (end of
envelope).  When the attacker is damaged mid-envelope then later merged to full
HP, the pin must not replace per-fire ``combatInfo`` attacker HP — that silently
zeroed counter damage and inflated join scrap (+2800g vs PHP for 1611364).

Sub-bar refinement (``abs(pin - combatInfo×10) < 10``) remains for Phase 11K
fire-frac anchors (e.g. 1635679).
"""
from __future__ import annotations

from engine.action import ActionStage
from engine.co import make_co_state_safe
from engine.game import GameState
from engine.map_loader import MapData
from engine.unit import Unit, UnitType, UNIT_STATS

from tools.oracle_zip_replay import _oracle_set_combat_damage_override_from_combat_info

PLAIN = 1


def _tiny_state() -> GameState:
    md = MapData(
        map_id=0,
        name="pin_guard",
        map_type="std",
        terrain=[[PLAIN] * 8 for _ in range(8)],
        height=8,
        width=8,
        cap_limit=99,
        unit_limit=50,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=[],
        hq_positions={0: [], 1: []},
        lab_positions={0: [], 1: []},
        country_to_player={},
    )
    return GameState(
        map_data=md,
        units={0: [], 1: []},
        funds=[0, 0],
        co_states=[make_co_state_safe(0), make_co_state_safe(0)],
        properties=[],
        turn=1,
        active_player=0,
        action_stage=ActionStage.ACTION,
        selected_unit=None,
        selected_move_pos=None,
        done=False,
        winner=None,
        win_reason=None,
        game_log=[],
        tier_name="T2",
        full_trace=[],
    )


def _spawn(
    state: GameState,
    *,
    ut: UnitType,
    player: int,
    pos: tuple[int, int],
    unit_id: int,
    hp: int,
) -> Unit:
    stats = UNIT_STATS[ut]
    u = Unit(
        unit_type=ut,
        player=player,
        hp=hp,
        ammo=stats.max_ammo if stats.max_ammo > 0 else 0,
        fuel=stats.max_fuel,
        pos=pos,
        moved=False,
        loaded_units=[],
        is_submerged=False,
        capture_progress=20,
        unit_id=unit_id,
    )
    state.units[player].append(u)
    return u


def test_attacker_pin_ignored_when_end_frame_hp_disagrees_by_full_bar():
    state = _tiny_state()
    att_uid, def_uid = 192052226, 191763703
    attacker = _spawn(
        state, ut=UnitType.TANK, player=1, pos=(7, 1), unit_id=att_uid, hp=100
    )
    _spawn(state, ut=UnitType.TANK, player=0, pos=(6, 1), unit_id=def_uid, hp=50)
    # End-of-envelope snapshot has the attacker back at full HP after a later Join;
    # mid-envelope Fire combatInfo says 6 display (60 internal).
    state._oracle_post_envelope_units_by_id = {att_uid: 100, def_uid: 50}
    state._oracle_post_envelope_multi_hit_defenders = set()

    fire_blk = {
        "combatInfoVision": {
            "global": {
                "combatInfo": {
                    "attacker": {
                        "units_id": att_uid,
                        "units_hit_points": 6,
                    },
                    "defender": {
                        "units_id": def_uid,
                        "units_hit_points": 5,
                    },
                }
            }
        }
    }
    _oracle_set_combat_damage_override_from_combat_info(
        state,
        fire_blk,
        envelope_awbw_player_id=3725811,
        attacker=attacker,
        defender_pos=(6, 1),
        awbw_to_engine=None,
    )
    dmg, counter = state._oracle_combat_damage_override
    assert dmg == 0
    assert counter == 40


def test_attacker_pin_refines_sub_display_counter_within_same_bar():
    state = _tiny_state()
    att_uid, def_uid = 192721109, 192721110
    attacker = _spawn(
        state, ut=UnitType.RECON, player=0, pos=(1, 1), unit_id=att_uid, hp=80
    )
    _spawn(state, ut=UnitType.TANK, player=1, pos=(1, 2), unit_id=def_uid, hp=100)
    state._oracle_post_envelope_units_by_id = {att_uid: 63, def_uid: 50}
    state._oracle_post_envelope_multi_hit_defenders = set()

    fire_blk = {
        "combatInfoVision": {
            "global": {
                "combatInfo": {
                    "attacker": {
                        "units_id": att_uid,
                        "units_hit_points": 6,
                    },
                    "defender": {
                        "units_id": def_uid,
                        "units_hit_points": 5,
                    },
                }
            }
        }
    }
    _oracle_set_combat_damage_override_from_combat_info(
        state,
        fire_blk,
        envelope_awbw_player_id=None,
        attacker=attacker,
        defender_pos=(1, 2),
        awbw_to_engine=None,
    )
    _dmg, counter = state._oracle_combat_damage_override
    assert counter == 80 - 63


def test_defender_pin_rejects_post_repair_inflation_from_end_frame():
    """End-frame pin can reflect day-start heal; combatInfo is still post-strike."""
    state = _tiny_state()
    att_uid, def_uid = 192052226, 191763703
    attacker = _spawn(
        state, ut=UnitType.TANK, player=1, pos=(7, 1), unit_id=att_uid, hp=100
    )
    _spawn(state, ut=UnitType.TANK, player=0, pos=(6, 1), unit_id=def_uid, hp=100)
    state._oracle_post_envelope_units_by_id = {att_uid: 60, def_uid: 60}
    state._oracle_post_envelope_multi_hit_defenders = set()
    fire_blk = {
        "combatInfoVision": {
            "global": {
                "combatInfo": {
                    "attacker": {
                        "units_id": att_uid,
                        "units_hit_points": 6,
                    },
                    "defender": {
                        "units_id": def_uid,
                        "units_hit_points": 5,
                    },
                }
            }
        }
    }
    _oracle_set_combat_damage_override_from_combat_info(
        state,
        fire_blk,
        envelope_awbw_player_id=None,
        attacker=attacker,
        defender_pos=(6, 1),
        awbw_to_engine=None,
    )
    dmg, _counter = state._oracle_combat_damage_override
    assert dmg == 50


def test_defender_pin_tight_zip_sub_display_gid1631520():
    """Tight-zip post-frame pin refines lossy display-1 defender HP (true 7 internal).

    GL **1631520** env 24: ``combatInfo`` defender ``units_hit_points: 1`` is
    display bars (×10 → 10 internal) but PHP ``hit_points`` 0.7 is 7 internal.
    Without ``_oracle_post_envelope_units_by_id`` the oracle under-applies
    strike damage (86→10 instead of 86→7).
    """
    state = _tiny_state()
    att_uid, def_uid = 192445080, 192414462
    attacker = _spawn(
        state,
        ut=UnitType.ANTI_AIR,
        player=0,
        pos=(6, 14),
        unit_id=att_uid,
        hp=51,
    )
    _spawn(
        state,
        ut=UnitType.B_COPTER,
        player=1,
        pos=(6, 15),
        unit_id=def_uid,
        hp=86,
    )
    state._oracle_post_envelope_units_by_id = {att_uid: 48, def_uid: 7}
    state._oracle_post_envelope_multi_hit_defenders = set()
    fire_blk = {
        "combatInfoVision": {
            "global": {
                "combatInfo": {
                    "attacker": {
                        "units_id": att_uid,
                        "units_hit_points": 5,
                    },
                    "defender": {
                        "units_id": def_uid,
                        "units_hit_points": 1,
                    },
                }
            }
        }
    }
    _oracle_set_combat_damage_override_from_combat_info(
        state,
        fire_blk,
        envelope_awbw_player_id=None,
        attacker=attacker,
        defender_pos=(6, 15),
        awbw_to_engine=None,
    )
    dmg, counter = state._oracle_combat_damage_override
    assert dmg == 79
    assert counter == 3

