from pathlib import Path

from tools.amarriner_live_meta import resolve_games_meta


def test_resolve_games_meta_finds_current_list() -> None:
    root = Path(__file__).resolve().parents[1]
    row = resolve_games_meta(1620036, repo_root=root)
    assert row is not None
    assert row["games_id"] == 1620036
    assert "map_id" in row
    assert row.get("co_p0_id") is not None and row.get("co_p1_id") is not None
