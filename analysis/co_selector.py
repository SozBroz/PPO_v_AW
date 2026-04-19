"""
Tier-aware staged CO selection for training.

Stages:
  Cold:  < MIN_GAMES_FOR_WARM — uniform random from enabled pool
  Warm:  >= MIN_GAMES_FOR_WARM — weighted by Wilson lower bound
  Eval:  fixed — returns best-ranked CO for the (map, tier)
"""
import json
import random
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
POOL_PATH = ROOT / "data" / "gl_map_pool.json"
RANKINGS_PATH = ROOT / "data" / "co_rankings.json"

MIN_GAMES_FOR_WARM = 50


def load_pool() -> list[dict]:
    with open(POOL_PATH) as f:
        return json.load(f)


def load_rankings() -> dict:
    if not RANKINGS_PATH.exists():
        return {}
    with open(RANKINGS_PATH) as f:
        return json.load(f)


def get_enabled_tiers(map_meta: dict) -> list[dict]:
    """Return enabled tiers with at least one CO for a map."""
    return [t for t in map_meta.get("tiers", []) if t.get("enabled") and t.get("co_ids")]


def select_co(map_id: int, tier_name: str, mode: str = "auto") -> int:
    """
    Select a CO for the given map and tier.

    mode:
      "auto"  — cold if insufficient data, else warm
      "cold"  — uniform random
      "warm"  — weighted by Wilson lower bound
      "eval"  — best-ranked CO

    Returns co_id (int).
    """
    pool = load_pool()
    map_meta = next((m for m in pool if m["map_id"] == map_id), None)
    if map_meta is None:
        raise ValueError(f"Map {map_id} not found in pool")

    tier = next((t for t in map_meta.get("tiers", []) if t["tier_name"] == tier_name), None)
    if tier is None or not tier.get("co_ids"):
        enabled = get_enabled_tiers(map_meta)
        if enabled:
            tier = random.choice(enabled)
        else:
            # Last resort: use whatever tier exists
            tiers = map_meta.get("tiers", [])
            tier = tiers[-1] if tiers else {"co_ids": []}

    eligible_co_ids: list[int] = tier.get("co_ids", [])
    if not eligible_co_ids:
        raise ValueError(f"No eligible COs for map {map_id} tier {tier_name}")

    if mode == "cold":
        return random.choice(eligible_co_ids)

    rankings = load_rankings()
    map_data = rankings.get(str(map_id), {})
    tier_data = map_data.get("by_tier", {}).get(tier_name, {})
    co_rankings: list[dict] = tier_data.get("co_rankings", [])
    games_played: int = tier_data.get("games_played", 0)

    if mode == "eval":
        for co_rank in co_rankings:
            if co_rank["co_id"] in eligible_co_ids:
                return co_rank["co_id"]
        return random.choice(eligible_co_ids)

    # Auto mode: cold start until we have enough data
    if games_played < MIN_GAMES_FOR_WARM or not co_rankings:
        return random.choice(eligible_co_ids)

    # Warm: weight by Wilson lower bound
    eligible_set = set(eligible_co_ids)
    eligible_ranked = [r for r in co_rankings if r["co_id"] in eligible_set]
    ranked_ids = {r["co_id"] for r in eligible_ranked}
    unranked = [co_id for co_id in eligible_co_ids if co_id not in ranked_ids]

    if not eligible_ranked:
        return random.choice(eligible_co_ids)

    ids: list[int] = []
    weights: list[float] = []
    for r in eligible_ranked:
        ids.append(r["co_id"])
        weights.append(max(r["ci_lower"], 0.01))
    for co_id in unranked:
        ids.append(co_id)
        weights.append(0.1)  # exploration bonus for never-seen COs

    return random.choices(ids, weights=weights, k=1)[0]


def select_game_config(
    map_id: Optional[int] = None,
    tier_name: Optional[str] = None,
    mode: str = "auto",
) -> tuple[int, str, int, int]:
    """
    Select a full game configuration: (map_id, tier_name, p0_co_id, p1_co_id).
    """
    pool = load_pool()

    if map_id is None:
        meta = random.choice(pool)
        map_id = meta["map_id"]
    else:
        meta = next((m for m in pool if m["map_id"] == map_id), None)
        if meta is None:
            meta = random.choice(pool)
            map_id = meta["map_id"]

    if tier_name is None:
        enabled = get_enabled_tiers(meta)
        tier_name = random.choice(enabled)["tier_name"] if enabled else "T2"

    p0_co = select_co(map_id, tier_name, mode=mode)
    p1_co = select_co(map_id, tier_name, mode=mode)

    return map_id, tier_name, p0_co, p1_co


def print_pool_summary() -> None:
    """Print summary of map pool and tier availability."""
    pool = load_pool()
    rankings = load_rankings()

    print(f"\nMap Pool Summary ({len(pool)} maps)")
    print("-" * 80)
    for meta in pool:
        enabled = get_enabled_tiers(meta)
        tier_summary = ", ".join(t["tier_name"] for t in enabled)
        map_rank_data = rankings.get(str(meta["map_id"]), {})
        games_total = sum(
            td.get("games_played", 0)
            for td in map_rank_data.get("by_tier", {}).values()
        )
        print(
            f"  [{meta.get('type', '?'):3s}] {meta['map_id']:6} | "
            f"{meta['name'][:35]:<35} | "
            f"Tiers: {tier_summary:<20} | Games: {games_total}"
        )


if __name__ == "__main__":
    print_pool_summary()
    print("\nExample selection (auto mode):")
    for _ in range(5):
        mid, tier, p0, p1 = select_game_config()
        print(f"  Map {mid}, Tier {tier}, P0_CO={p0}, P1_CO={p1}")
