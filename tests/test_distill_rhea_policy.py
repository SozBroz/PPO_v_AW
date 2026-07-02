"""Smoke tests for RHEA corpus policy distillation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from engine.action import Action, ActionType
from engine.game import make_initial_state
from engine.map_loader import MapData
from engine.predeployed import PredeployedUnitSpec
from engine.unit import UnitType
from rl.game_corpus import build_corpus_row
from rl.network import AWBWNet
from scripts.distill_rhea_policy import run_distillation

WOOD = 3
TINY_MAP_ID = 9_000_043


@pytest.fixture
def tiny_map_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    def _loader(map_id: int, pool: Path, maps_dir: Path) -> MapData:
        if int(map_id) == TINY_MAP_ID:
            return _tiny_combat_map()
        raise ValueError(f"unexpected map_id {map_id}")

    monkeypatch.setattr("rl.game_corpus.load_map", _loader)
    monkeypatch.setattr("engine.map_loader.load_map", _loader)


def _tiny_combat_map() -> MapData:
    return MapData(
        map_id=TINY_MAP_ID,
        name="distill_combat_tiny",
        map_type="std",
        terrain=[[WOOD, WOOD], [WOOD, WOOD]],
        height=2,
        width=2,
        cap_limit=99,
        unit_limit=50,
        unit_bans=[],
        tiers=[],
        objective_type=None,
        properties=[],
        hq_positions={0: [], 1: []},
        lab_positions={0: [], 1: []},
        country_to_player={},
        predeployed_specs=[
            PredeployedUnitSpec(row=0, col=0, player=0, unit_type=UnitType.TANK),
            PredeployedUnitSpec(row=0, col=1, player=1, unit_type=UnitType.INFANTRY),
        ],
    )


def _play_scripted_combat_game(*, luck_seed: int) -> object:
    md = _tiny_combat_map()
    st = make_initial_state(
        md,
        14,
        14,
        starting_funds=0,
        tier_name="T2",
        replay_first_mover=0,
        luck_seed=luck_seed,
    )
    tank = st.get_unit_at(0, 0)
    inf = st.get_unit_at(0, 1)
    assert tank is not None and inf is not None
    scripted = [
        Action(ActionType.SELECT_UNIT, unit_pos=tank.pos),
        Action(ActionType.SELECT_UNIT, unit_pos=tank.pos, move_pos=tank.pos),
        Action(
            ActionType.ATTACK,
            unit_pos=tank.pos,
            move_pos=tank.pos,
            target_pos=inf.pos,
        ),
    ]
    for act in scripted:
        st.step(act)
    return st


def _write_tiny_corpus(path: Path) -> None:
    rows = []
    for seed in (51_001, 51_002):
        st = _play_scripted_combat_game(luck_seed=seed)
        rows.append(
            build_corpus_row(
                st,
                map_id=TINY_MAP_ID,
                map_name="distill_combat_tiny",
                tier="T2",
                p0_co_id=14,
                p1_co_id=14,
                luck_seed=seed,
                replay_first_mover=0,
            )
        )
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")


def test_distill_rhea_policy_smoke(
    tmp_path: Path, tiny_map_loader: None
) -> None:
    corpus = tmp_path / "tiny_corpus.jsonl"
    out_ckpt = tmp_path / "distill.pt"
    _write_tiny_corpus(corpus)

    result = run_distillation(
        corpus=corpus,
        epochs=4,
        batch_size=2,
        lr=1e-3,
        device="cpu",
        max_games=2,
        out=out_ckpt,
        value_loss_weight=0.25,
        val_fraction=0.2,
        seed=7,
    )

    assert result.games_trained == 2
    assert result.steps_trained >= 6
    assert result.train_loss_final_epoch < result.train_loss_first_epoch
    assert out_ckpt.is_file()
    assert result.sidecar_path.is_file()

    payload = json.loads(result.sidecar_path.read_text(encoding="utf-8"))
    assert payload["games_trained"] == 2
    assert payload["val_top1_accuracy"] == pytest.approx(result.val_top1_accuracy)

    ckpt = torch.load(out_ckpt, map_location="cpu", weights_only=True)
    net = AWBWNet()
    net.load_state_dict(ckpt["state_dict"])
    assert next(net.parameters()).shape[0] > 0
