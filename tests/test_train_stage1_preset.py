"""train.py --stage1-narrow preset."""
from __future__ import annotations

def test_stage1_narrow_fills_defaults() -> None:
    from train import _apply_stage1_narrow_defaults, build_train_argument_parser

    p = build_train_argument_parser()
    args = p.parse_args(["--stage1-narrow"])
    _apply_stage1_narrow_defaults(args)
    assert args.map_id == 123858
    assert args.tier == "T3"
    assert args.co_p0 == 1
    assert args.co_p1 == 1
    assert args.curriculum_tag == "stage1-misery-andy"


def test_stage1_narrow_respects_explicit_overrides() -> None:
    from train import _apply_stage1_narrow_defaults, build_train_argument_parser

    p = build_train_argument_parser()
    args = p.parse_args(
        ["--stage1-narrow", "--map-id", "999999", "--curriculum-tag", "custom"]
    )
    _apply_stage1_narrow_defaults(args)
    assert args.map_id == 999999
    assert args.curriculum_tag == "custom"
    assert args.tier == "T3"


def test_stage1_narrow_noop_without_flag() -> None:
    from train import _apply_stage1_narrow_defaults, build_train_argument_parser

    p = build_train_argument_parser()
    args = p.parse_args([])
    _apply_stage1_narrow_defaults(args)
    assert args.map_id is None
    assert args.curriculum_tag is None
