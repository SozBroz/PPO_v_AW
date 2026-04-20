"""
Computes win-rate matrices per (CO, map_id, tier) from game_log.jsonl.
Uses Wilson score confidence interval for ranking.
Outputs data/co_rankings.json.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
GAME_LOG_PATH = ROOT / "logs" / "game_log.jsonl"
POOL_PATH = ROOT / "data" / "gl_map_pool.json"
OUT_PATH = ROOT / "data" / "co_rankings.json"


def wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    """Wilson score interval lower bound (95% confidence)."""
    if n == 0:
        return 0.0
    p_hat = wins / n
    denominator = 1 + z**2 / n
    centre = p_hat + z**2 / (2 * n)
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return (centre - margin) / denominator


def load_game_log() -> list[dict]:
    if not GAME_LOG_PATH.exists():
        return []
    games = []
    with open(GAME_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    games.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return games


def compute_rankings() -> dict:
    """
    For each (map_id, tier) combination, rank COs by Wilson lower bound win rate.

    Returns:
    {
      "12345": {
        "map_name": "...",
        "by_tier": {
          "T2": {
            "games_played": 412,
            "co_rankings": [
              {"co_id": 7, "co_name": "Max", "win_rate": 0.68, "ci_lower": 0.63, "games": 87}
            ]
          }
        }
      }
    }
    """
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    games = load_game_log()

    with open(POOL_PATH) as f:
        pool = json.load(f)
    map_names = {str(m["map_id"]): m["name"] for m in pool}

    co_data_path = ROOT / "data" / "co_data.json"
    co_names: dict[int, str] = {}
    if co_data_path.exists():
        with open(co_data_path) as f:
            data = json.load(f)
        co_names = {int(k): v["name"] for k, v in data.get("cos", {}).items()}

    # Accumulate stats: (map_id, tier, co_id) -> {wins, games}
    stats: dict[tuple[str, str, int], dict[str, int]] = defaultdict(
        lambda: {"wins": 0, "games": 0}
    )
    total_by_map_tier: dict[tuple[str, str], int] = defaultdict(int)

    for game in games:
        map_id = str(game.get("map_id", ""))
        tier = game.get("tier", "")
        p0_co = game.get("p0_co_id", game.get("p0_co"))
        p1_co = game.get("p1_co_id", game.get("p1_co"))
        winner = game.get("winner", -1)

        if p0_co is None or p1_co is None:
            continue

        total_by_map_tier[(map_id, tier)] += 1

        key0 = (map_id, tier, p0_co)
        stats[key0]["games"] += 1
        if winner == 0:
            stats[key0]["wins"] += 1

        key1 = (map_id, tier, p1_co)
        stats[key1]["games"] += 1
        if winner == 1:
            stats[key1]["wins"] += 1

    # Group map_id -> set of tiers seen
    map_tiers: dict[str, set[str]] = defaultdict(set)
    for map_id, tier, _co_id in stats.keys():
        map_tiers[map_id].add(tier)

    results: dict[str, dict] = {}

    for map_id, tiers in map_tiers.items():
        by_tier: dict[str, dict] = {}
        for tier in tiers:
            co_keys = [
                (m, t, co_id)
                for (m, t, co_id) in stats.keys()
                if m == map_id and t == tier
            ]

            co_rankings = []
            for key in co_keys:
                _, _, co_id = key
                s = stats[key]
                wins, n = s["wins"], s["games"]
                win_rate = wins / n if n > 0 else 0.0
                ci_lower = wilson_lower_bound(wins, n)
                co_rankings.append(
                    {
                        "co_id": co_id,
                        "co_name": co_names.get(co_id, f"CO#{co_id}"),
                        "win_rate": round(win_rate, 4),
                        "ci_lower": round(ci_lower, 4),
                        "games": n,
                        "wins": wins,
                    }
                )

            co_rankings.sort(key=lambda x: x["ci_lower"], reverse=True)

            by_tier[tier] = {
                "games_played": total_by_map_tier[(map_id, tier)],
                "co_rankings": co_rankings,
            }

        results[map_id] = {
            "map_name": map_names.get(map_id, f"Map#{map_id}"),
            "by_tier": by_tier,
        }

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"[co_ranker] Rankings written to {OUT_PATH}")
    print(f"[co_ranker] Total games processed: {len(games)}")

    return results


if __name__ == "__main__":
    results = compute_rankings()
    for map_id, data in results.items():
        for tier, td in data["by_tier"].items():
            if td["co_rankings"]:
                top = td["co_rankings"][0]
                print(
                    f"  {data['map_name']} [{tier}]: {top['co_name']} "
                    f"({top['win_rate']:.1%} WR, {top['games']} games)"
                )
