"""Regression tests for ``desync_subtype`` prefix routing (`tools/cluster_desync_register.py`)."""

import json
from pathlib import Path

from tools.cluster_desync_register import cluster, desync_subtype, load_jsonl


def test_oracle_gap_build_supply_power_buckets():
    assert (
        desync_subtype(
            {
                "class": "oracle_gap",
                "message": "Build no-op at tile (4,10) unit=INFANTRY for engine P0: engine refused BUILD (x)",
            }
        )
        == "oracle_build"
    )
    assert (
        desync_subtype(
            {
                "class": "oracle_gap",
                "message": "Supply (no path): no WAIT/DIVE_HIDE at (3,4); legal=[WAIT]",
            }
        )
        == "oracle_supply"
    )
    assert (
        desync_subtype(
            {
                "class": "oracle_gap",
                "message": "Power without playerID",
            }
        )
        == "oracle_power"
    )
    assert (
        desync_subtype(
            {
                "class": "oracle_gap",
                "message": "Join without nested Move dict",
            }
        )
        == "oracle_join"
    )
    assert (
        desync_subtype(
            {
                "class": "oracle_gap",
                "message": "Load without nested Move dict",
            }
        )
        == "oracle_load"
    )
    assert (
        desync_subtype(
            {
                "class": "oracle_gap",
                "message": "Hide without nested Move dict",
            }
        )
        == "oracle_hide"
    )


def test_active_player_wins_over_supply_prefix():
    assert (
        desync_subtype(
            {
                "class": "oracle_gap",
                "message": "Supply (no path) for engine P0 but active_player=1",
            }
        )
        == "oracle_turn_active_player"
    )


def test_unknown_unit_lowercase():
    assert (
        desync_subtype(
            {
                "class": "oracle_gap",
                "message": "unknown AWBW unit name 'FooBar'",
            }
        )
        == "oracle_unknown_unit"
    )


def test_unsupported_kind_stable():
    assert (
        desync_subtype(
            {
                "class": "oracle_gap",
                "message": "UnsupportedOracleAction: unsupported oracle action 'FooKind'",
            }
        )
        == "oracle_unsupported_kind"
    )


def test_desync_subtype_tolerates_schema_v2_fields():
    """Phase 11d schema_version 2 added ``machine_id`` / ``recorded_at`` /
    ``schema_version``. ``desync_subtype`` only reads ``class`` + ``message``
    via ``row.get`` so the new keys must be transparent."""
    row = {
        "schema_version": 2,
        "machine_id": "pc-b",
        "recorded_at": "2026-04-23T12:00:00Z",
        "class": "engine_bug",
        "message": "AssertionError boom",
        "games_id": 7,
    }
    assert desync_subtype(row) == "engine_other"


def test_cluster_handles_mixed_legacy_and_v2_rows(tmp_path: Path) -> None:
    """``cluster`` indexes by ``games_id`` and routes via ``class``/``message``;
    rows missing the new attribution fields must coexist with ones that
    have them, both in-memory and via ``load_jsonl``."""
    reg = tmp_path / "mixed.jsonl"
    rows = [
        # legacy (no schema_version, no machine_id, no recorded_at)
        {"games_id": 100, "class": "ok"},
        # v2 row
        {
            "games_id": 200,
            "schema_version": 2,
            "machine_id": "pc-b",
            "recorded_at": "2026-04-23T12:00:00Z",
            "class": "engine_bug",
            "message": "kaboom",
        },
        # v2 row, different machine
        {
            "games_id": 300,
            "schema_version": 2,
            "machine_id": "pc-c",
            "recorded_at": "2026-04-23T12:01:00Z",
            "class": "oracle_gap",
            "message": "Build no-op",
        },
    ]
    reg.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    loaded = load_jsonl(reg)
    assert len(loaded) == 3
    clusters = cluster(loaded)
    assert clusters["ok"] == [100]
    assert clusters["engine_other"] == [200]
    assert clusters["oracle_build"] == [300]
