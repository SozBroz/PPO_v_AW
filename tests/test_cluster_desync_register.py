"""Regression tests for ``desync_subtype`` prefix routing (`tools/cluster_desync_register.py`)."""

from tools.cluster_desync_register import desync_subtype


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
