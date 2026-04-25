"""Phase 11J-FINAL — phantom-degenerate ``Move`` silent-skip helper.

Covers ``_oracle_phantom_degenerate_move_is_safe_skip`` and the gated
``allow_phantom_degenerate_skip`` path in ``_apply_move_paths_then_terminator``.

Source pattern: GL **1626236** — AWBW exporter writes a per-day ``Move`` for
a Black Boat (hp=1, ``paths.global`` length 1) the engine has already sunk.
Skipping it is benign.  Anything richer (real movement, nested terminator,
or an alive engine same-type unit) MUST still raise.
"""

from __future__ import annotations

import unittest

from engine.action import ActionStage
from engine.game import make_initial_state
from engine.map_loader import load_map
from engine.unit import UNIT_STATS, Unit, UnitType
from server.play_human import MAPS_DIR, POOL_PATH
from tools.oracle_zip_replay import (
    UnsupportedOracleAction,
    _oracle_phantom_degenerate_move_is_safe_skip,
    apply_oracle_action_json,
)


def _move_action(
    *,
    awbw_pid: int,
    units_id: int,
    units_name: str,
    y: int,
    x: int,
    hp: int | str = 1,
    path: list[tuple[int, int]] | None = None,
) -> dict:
    """Build a minimal AWBW Move envelope with ``unit.global`` and ``paths.global``."""
    if path is None:
        path = [(y, x)]
    return {
        "action": "Move",
        "unit": {
            "global": {
                "units_id": units_id,
                "units_players_id": awbw_pid,
                "units_name": units_name,
                "units_y": y,
                "units_x": x,
                "units_movement_points": 7,
                "units_vision": 1,
                "units_fuel": 50,
                "units_fuel_per_turn": 0,
                "units_sub_dive": "N",
                "units_ammo": 0,
                "units_short_range": 0,
                "units_long_range": 0,
                "units_second_weapon": "N",
                "units_symbol": "S",
                "units_cost": 7500,
                "units_movement_type": "L",
                "units_moved": 1,
                "units_capture": 0,
                "units_fired": 0,
                "units_hit_points": hp,
                "units_cargo1_units_id": 0,
                "units_cargo2_units_id": 0,
                "units_carried": "N",
                "countries_code": "uw",
            }
        },
        "paths": {"global": [{"y": yy, "x": xx} for yy, xx in path]},
    }


class TestPhantomDegenerateMoveSilentSkip(unittest.TestCase):
    AWBW_PID = 3758345
    GHOST_UID = 191895057
    TILE_RC = (8, 10)

    def _empty_state(self) -> object:
        m = load_map(77060, POOL_PATH, MAPS_DIR)
        s = make_initial_state(m, 1, 1, tier_name="T3", starting_funds=0)
        s.units[0] = []
        s.units[1] = []
        s.active_player = 0
        s.action_stage = ActionStage.SELECT
        return s

    def test_positive_skip_when_length1_no_engine_mover_no_same_type_unit(self) -> None:
        """The 1626236 shape — engine has 0 BLACK_BOAT, paths.global length 1.

        The skip helper records the event on ``state._oracle_phantom_mover_skips``
        and ``apply_oracle_action_json`` returns without raising or mutating the
        roster.  This is the **only** way the bare-Move call site is allowed to
        swallow a ``Move: mover not found`` envelope.
        """
        s = self._empty_state()
        before_units = {seat: list(lst) for seat, lst in s.units.items()}
        envelope = _move_action(
            awbw_pid=self.AWBW_PID,
            units_id=self.GHOST_UID,
            units_name="Black Boat",
            y=self.TILE_RC[0],
            x=self.TILE_RC[1],
            hp=1,
            path=[self.TILE_RC],
        )
        apply_oracle_action_json(
            s,
            envelope,
            {self.AWBW_PID: 0, self.AWBW_PID + 1: 1},
            envelope_awbw_player_id=self.AWBW_PID,
        )
        self.assertEqual({seat: list(lst) for seat, lst in s.units.items()}, before_units)
        skips = getattr(s, "_oracle_phantom_mover_skips", None)
        self.assertIsInstance(skips, list)
        self.assertEqual(len(skips), 1)
        rec = skips[0]
        self.assertEqual(rec["awbw_units_id"], self.GHOST_UID)
        self.assertEqual(rec["declared_mover_type"], UnitType.BLACK_BOAT.name)
        self.assertEqual(rec["engine_player"], 0)
        self.assertEqual(tuple(rec["tile_rc"]), self.TILE_RC)

    def test_negative_does_not_skip_when_path_length_gt_1(self) -> None:
        """1628722 / 1632825 shape — non-degenerate path stays INHERENT.

        Even with no engine mover and no same-type unit, length > 1 means a
        real movement was intended.  The helper refuses to swallow it and the
        original ``Move: mover not found`` raise must surface so the audit
        keeps the divergence on the books.
        """
        s = self._empty_state()
        envelope = _move_action(
            awbw_pid=self.AWBW_PID,
            units_id=self.GHOST_UID,
            units_name="Md.Tank",
            y=8,
            x=2,
            hp="?",
            path=[(5, 3), (6, 3), (7, 3), (8, 3), (8, 2)],
        )
        with self.assertRaises(UnsupportedOracleAction) as cm:
            apply_oracle_action_json(
                s,
                envelope,
                {self.AWBW_PID: 0, self.AWBW_PID + 1: 1},
                envelope_awbw_player_id=self.AWBW_PID,
            )
        self.assertIn("Move: mover not found", str(cm.exception))
        self.assertFalse(getattr(s, "_oracle_phantom_mover_skips", []))

    def test_helper_refuses_when_engine_has_same_type_unit_alive(self) -> None:
        """Direct helper test: same-type unit on the seat → refuse skip.

        Integration coverage is hard to express here because the upstream
        resolver chain (lone-BB hatch / ``_pick_same_type_mover_by_path_reachability``)
        already succeeds when any same-type unit exists — there is no
        ``Move: mover not found`` raise to gate.  The skip helper itself is
        the safety net: even if a future regression bypassed the resolver's
        same-type fallback, the helper must still refuse to silently swallow
        the action while a candidate of the declared type lives.
        """
        s = self._empty_state()
        bbs = UNIT_STATS[UnitType.BLACK_BOAT]
        s.units[0].append(
            Unit(
                UnitType.BLACK_BOAT, 0, 100, bbs.max_ammo, bbs.max_fuel,
                (4, 4), False, [], False, bbs.vision, 998,
            )
        )
        sr, sc = self.TILE_RC
        ok = _oracle_phantom_degenerate_move_is_safe_skip(
            s,
            eng=0,
            declared_mover_type=UnitType.BLACK_BOAT,
            uid=self.GHOST_UID,
            paths=[{"y": sr, "x": sc}],
            sr=sr,
            sc=sc,
            er=sr,
            ec=sc,
        )
        self.assertFalse(ok)

    def test_helper_refuses_when_uid_collides_with_dead_engine_unit(self) -> None:
        """uid match on a *dead* engine unit also refuses skip.

        The directive's gate #3 reads "no live unit on the engine map matches
        the AWBW units_id"; we tighten it to "any unit (alive or dead)".  A
        dead unit with the same ``unit_id`` could indicate stale roster drift
        the engine should learn about — silent-skip would obscure that signal.
        """
        s = self._empty_state()
        ist = UNIT_STATS[UnitType.INFANTRY]
        ghost = Unit(
            UnitType.INFANTRY, 0, 0, ist.max_ammo, ist.max_fuel,
            (-1, -1), False, [], False, ist.vision, self.GHOST_UID,
        )
        ghost.hp = 0
        s.units[0].append(ghost)
        self.assertFalse(ghost.is_alive)
        sr, sc = self.TILE_RC
        ok = _oracle_phantom_degenerate_move_is_safe_skip(
            s,
            eng=0,
            declared_mover_type=UnitType.BLACK_BOAT,
            uid=self.GHOST_UID,
            paths=[{"y": sr, "x": sc}],
            sr=sr,
            sc=sc,
            er=sr,
            ec=sc,
        )
        self.assertFalse(ok)

    def test_negative_does_not_skip_when_friendly_unit_sits_on_path_tile(self) -> None:
        """Length-1 path but path tile already holds a friendly (different-type) unit.

        AWBW would have addressed *that* unit (or stacked-tile lookup would
        have); silent-skipping risks losing a real interaction.  The helper
        refuses when ``state.get_unit_at(sr, sc)`` is friendly — even if the
        type does not match the AWBW declared type.
        """
        s = self._empty_state()
        ist = UNIT_STATS[UnitType.INFANTRY]
        s.units[0].append(
            Unit(
                UnitType.INFANTRY,
                0,
                100,
                ist.max_ammo,
                ist.max_fuel,
                self.TILE_RC,
                False,
                [],
                False,
                ist.vision,
                51,
            )
        )
        s.units[0][0].moved = True
        envelope = _move_action(
            awbw_pid=self.AWBW_PID,
            units_id=self.GHOST_UID,
            units_name="Black Boat",
            y=self.TILE_RC[0],
            x=self.TILE_RC[1],
            hp=1,
            path=[self.TILE_RC],
        )
        with self.assertRaises(UnsupportedOracleAction):
            apply_oracle_action_json(
                s,
                envelope,
                {self.AWBW_PID: 0, self.AWBW_PID + 1: 1},
                envelope_awbw_player_id=self.AWBW_PID,
            )


if __name__ == "__main__":
    unittest.main()
