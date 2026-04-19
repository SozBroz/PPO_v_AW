"""
Head-to-head CO matchup analysis (Phase B2).

Analyzes game_log.jsonl to compute win rates for each CO matchup
per (map_id, tier) combination.
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse

ROOT = Path(__file__).parent.parent
GAME_LOG_PATH = ROOT / "data" / "game_log.jsonl"
OUTPUT_PATH = ROOT / "data" / "co_matchups.json"


def load_game_log() -> List[dict]:
    """Load all game records from game_log.jsonl."""
    if not GAME_LOG_PATH.exists():
        return []
    
    games = []
    with open(GAME_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    games.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return games


def compute_h2h_stats(
    games: List[dict],
    min_games: int = 5
) -> Dict[Tuple[int, str], Dict]:
    """
    Compute head-to-head statistics per (map_id, tier).
    
    Returns:
        {
            (map_id, tier): {
                "games_played": int,
                "matchups": {
                    (co_a, co_b): {
                        "games": int,
                        "co_a_wins": int,  # wins when co_a is P0
                        "co_b_wins": int,  # wins when co_b is P0
                        "co_a_as_p0": int,  # times co_a played as P0
                        "co_b_as_p0": int,  # times co_b played as P0
                        "co_a_win_rate": float,  # overall win rate for co_a
                        "co_b_win_rate": float,  # overall win rate for co_b
                    }
                }
            }
        }
    """
    # Group games by (map_id, tier)
    strata = defaultdict(list)
    for game in games:
        if game.get("winner") is None:
            continue  # Skip draws or incomplete games
        
        map_id = game.get("map_id")
        tier = game.get("tier")
        if map_id is None or tier is None:
            continue
        
        strata[(map_id, tier)].append(game)
    
    # Compute stats per stratum
    results = {}
    for (map_id, tier), stratum_games in strata.items():
        matchup_stats = defaultdict(lambda: {
            "games": 0,
            "co_a_wins": 0,
            "co_b_wins": 0,
            "co_a_as_p0": 0,
            "co_b_as_p0": 0,
        })
        
        for game in stratum_games:
            p0_co = game.get("p0_co_id", game.get("p0_co"))
            p1_co = game.get("p1_co_id", game.get("p1_co"))
            winner = game.get("winner")
            
            if p0_co is None or p1_co is None or winner is None:
                continue
            
            # Create ordered pair (smaller CO ID first for consistency)
            co_a, co_b = (p0_co, p1_co) if p0_co <= p1_co else (p1_co, p0_co)
            pair = (co_a, co_b)
            
            matchup_stats[pair]["games"] += 1
            
            # Track which CO was P0 and who won
            if p0_co == co_a:
                matchup_stats[pair]["co_a_as_p0"] += 1
                if winner == 0:
                    matchup_stats[pair]["co_a_wins"] += 1
                elif winner == 1:
                    matchup_stats[pair]["co_b_wins"] += 1
            else:  # p0_co == co_b
                matchup_stats[pair]["co_b_as_p0"] += 1
                if winner == 0:
                    matchup_stats[pair]["co_b_wins"] += 1
                elif winner == 1:
                    matchup_stats[pair]["co_a_wins"] += 1
        
        # Compute win rates and filter by min_games
        filtered_matchups = {}
        for pair, stats in matchup_stats.items():
            if stats["games"] < min_games:
                continue
            
            co_a_total_wins = stats["co_a_wins"]
            co_b_total_wins = stats["co_b_wins"]
            total_games = stats["games"]
            
            stats["co_a_win_rate"] = co_a_total_wins / total_games if total_games > 0 else 0.0
            stats["co_b_win_rate"] = co_b_total_wins / total_games if total_games > 0 else 0.0
            
            # Convert tuple keys to strings for JSON serialization
            filtered_matchups[f"{pair[0]}_vs_{pair[1]}"] = stats
        
        if filtered_matchups:
            results[(map_id, tier)] = {
                "games_played": len(stratum_games),
                "matchups": filtered_matchups,
            }
    
    return results


def format_output(stats: Dict) -> Dict:
    """Format stats for JSON output with readable keys."""
    output = {}
    for (map_id, tier), data in stats.items():
        key = f"map_{map_id}_tier_{tier}"
        output[key] = {
            "map_id": map_id,
            "tier": tier,
            "games_played": data["games_played"],
            "matchups": data["matchups"],
        }
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Analyze head-to-head CO matchup statistics"
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=5,
        help="Minimum games required to include a matchup (default: 5)",
    )
    parser.add_argument(
        "--map-id",
        type=int,
        help="Filter to specific map ID",
    )
    parser.add_argument(
        "--tier",
        type=str,
        help="Filter to specific tier name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_PATH),
        help="Output JSON file path",
    )
    
    args = parser.parse_args()
    
    print(f"Loading games from {GAME_LOG_PATH}...")
    games = load_game_log()
    print(f"Loaded {len(games)} game records")
    
    # Apply filters
    if args.map_id is not None:
        games = [g for g in games if g.get("map_id") == args.map_id]
        print(f"Filtered to {len(games)} games on map {args.map_id}")
    
    if args.tier is not None:
        games = [g for g in games if g.get("tier") == args.tier]
        print(f"Filtered to {len(games)} games in tier {args.tier}")
    
    print(f"\nComputing head-to-head statistics (min {args.min_games} games)...")
    stats = compute_h2h_stats(games, min_games=args.min_games)
    
    output = format_output(stats)
    
    print(f"\nFound {len(output)} map/tier combinations with matchup data")
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print(f"Wrote results to {output_path}")
    
    # Print summary
    total_matchups = sum(len(data["matchups"]) for data in output.values())
    print(f"\nTotal matchups analyzed: {total_matchups}")


if __name__ == "__main__":
    main()

# Made with Bob
