#!/usr/bin/env python3
"""
Analyze live AWBW games using incremental action tracking to avoid unnecessary requests.
"""
import argparse
import json
import os
import re
import sys
import time
import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.amarriner_list_your_games import _login, HEADERS, BASE_URL

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ROOT / "tools" / "analyze_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("analyze_live_games")

# Cache directory for storing replay positions
CACHE_DIR = ROOT / "tools" / "replay_cache"
if not CACHE_DIR.exists():
    CACHE_DIR.mkdir(parents=True)

MAX_ACTIONS_PER_GAME = 10  # Maximum actions to process per game
REPLAY_URL_TO_DEBUGFILE = {
    1638496: CACHE_DIR / "1638496_replay_html.html"
}


def get_last_position(games_id: int) -> int:
    """Get last processed action position for a game."""
    cache_file = CACHE_DIR / f"{games_id}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f).get("last_position", -1)
        except Exception as e:
            logger.error(f"Error reading cache for {games_id}: {e}")
    return -1


def save_position(games_id: int, position: int):
    """Save last processed action position for a game."""
    cache_file = CACHE_DIR / f"{games_id}.json"
    try:
        with open(cache_file, "w") as f:
            json.dump({"last_position": position}, f)
    except Exception as e:
        logger.error(f"Error saving position for {games_id}: {e}")


def parse_minimal_game_info(html: str, games_id: int) -> Dict[str, Any]:
    """Fast parser for minimal game info (day and map)."""
    result = {
        'games_id': games_id,
        'day': 1,
        'map': 'Unknown'
    }
    
    # Extract day using regex
    day_match = re.search(r'Day (\d+)', html)
    if day_match:
        result['day'] = int(day_match.group(1))
    
    # Extract map name
    map_match = re.search(r'Game Started:.*? - ([^-()]+?)\s*\(', html)
    if map_match:
        result['map'] = map_match.group(1).strip()
    
    return result


def fetch_replay_page(session: requests.Session, games_id: int) -> Optional[str]:
    """Fetch replay page HTML for position tracking."""
    try:
        # Try to access the replay viewer page (ndx=0)
        replay_url = f"{BASE_URL}/game.php?games_id={games_id}&ndx=0"
        response = session.get(replay_url, headers=HEADERS)
        response.raise_for_status()
        
        # Save debug HTML where needed
        if games_id in REPLAY_URL_TO_DEBUGFILE:
            with open(REPLAY_URL_TO_DEBUGFILE[games_id], "w", encoding="utf-8") as f:
                f.write(response.text)
                
        return response.text
    except Exception as e:
        logger.error(f"Error fetching replay page for {games_id}: {e}")
        return None


def get_max_position_from_replay(html: str) -> Optional[int]:
    """Extract maxPosition from replay viewer HTML."""
    # More flexible regex patterns to match JavaScript variables
    patterns = [
        r"maxPosition\s*[:=]\s*(\d+)",
        r"var\s+maxPosition\s*=\s*(\d+)",
        r'id="maxPosition"\s+value="(\d+)"'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                continue
    return None


def fetch_incremental_actions(session: requests.Session, games_id: int) -> List[Dict[str, Any]]:
    """Fetch replay actions incrementally using AWBW's replay API."""
    try:
        # Get maximum position from replay viewer if possible
        replay_html = fetch_replay_page(session, games_id)
        max_position = get_max_position_from_replay(replay_html) if replay_html else None

        current_position = get_last_position(games_id)
        new_positions = []
        
        if max_position is not None and current_position < max_position:
            # We know the total action count - calculate what's new
            new_positions = range(current_position + 1, min(current_position + MAX_ACTIONS_PER_GAME + 1, max_position + 1))
        elif current_position == -1 and max_position is None:
            # First run and no position info - start at beginning
            new_positions = [0]
        else:
            # Fallback: iterate until we hit an error or limit
            new_positions = []
            next_pos = current_position + 1
            for _ in range(MAX_ACTIONS_PER_GAME):
                new_positions.append(next_pos)
                next_pos += 1

        if not new_positions:
            return []

        logger.info(f"Game {games_id}: Fetching {len(new_positions)} actions")
        
        # Fetch new actions
        new_actions = []
        for i, position in enumerate(new_positions):
            actions_url = f"{BASE_URL}/api/game/get_actions.php?games_id={games_id}&ndx={position}"
            
            # Random delays to avoid DDoSing
            jitter = 0.3 if random.random() > 0.7 else 0.8  # Occasional faster requests
            time.sleep(jitter + random.random() * 0.5)  # 0.3-1.3 or 0.8-1.3 seconds
            
            try:
                actions_response = session.get(actions_url, headers=HEADERS)
                actions_response.raise_for_status()
                
                action_data = actions_response.json()
                if not action_data or not isinstance(action_data, dict):
                    continue
                
                # Extract action details
                action_type = action_data.get("type", "unknown")
                action_info = {"type": action_type, "index": position}
                
                # Handle different action types
                if action_type == "move":
                    action_info.update({
                        "unit_id": action_data.get("unitId"),
                        "from": action_data.get("fromPos", {"x": -1, "y": -1}),
                        "to": action_data.get("toPos", {"x": -1, "y": -1}),
                        "path": action_data.get("path", [])
                    })
                elif action_type == "attack":
                    action_info.update({
                        "attacker_id": action_data.get("attackerUnitId"),
                        "defender_id": action_data.get("defenderUnitId"),
                        "damage": action_data.get("damageDealt", -1)
                    })
                elif action_type == "capture":
                    action_info.update({
                        "unit_id": action_data.get("unitId"),
                        "position": action_data.get("position", {"x": -1, "y": -1}),
                        "capture_hp": action_data.get("captureHp", -1)
                    })
                
                new_actions.append(action_info)
                save_position(games_id, position)
                
            except Exception as e:
                logger.warning(f"Stopping on failed action {position}: {e}")
                break  # Stop on error
        
        return new_actions
    
    except Exception as e:
        logger.error(f"Error in incremental fetching: {e}", exc_info=True)
        return []


def fetch_game_page(session: requests.Session, games_id: int) -> str:
    """Fetch game page HTML."""
    url = f"{BASE_URL}/game.php?games_id={games_id}"
    response = session.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.text


def analyze_games(games_ids: List[int]) -> List[Dict[str, Any]]:
    """Analyze multiple games using incremental action tracking."""
    # Create session and login
    session = requests.Session()
    with open(ROOT / "secrets.txt", "r") as f:
        lines = f.readlines()
        username = lines[0].strip()
        password = lines[1].strip()
    
    if not _login(session, username, password):
        logger.error("Login failed")
        return []
    
    results = []
    for games_id in games_ids:
        logger.info(f"\n{'='*80}")
        logger.info(f"Analyzing game {games_id}")
        logger.info(f"{'='*80}")
        
        try:
            # Get minimal game info
            html = fetch_game_page(session, games_id)
            game_info = parse_minimal_game_info(html, games_id)
            
            # Only fetch incremental actions
            new_actions = fetch_incremental_actions(session, games_id)
            
            # Combine results
            result = {
                "game_info": game_info,
                "new_actions": new_actions
            }
            results.append(result)
            
            # Print summary
            logger.info(f"Game ID: {games_id}")
            logger.info(f"Day: {game_info['day']}")
            logger.info(f"Map: {game_info['map']}")
            
            if new_actions:
                logger.info(f"Fetched {len(new_actions)} new actions")
                for action in new_actions[:5]:  # Show first 5
                    logger.info(f"  - {action.get('type', 'unknown')} (index {action['index']})")
                if len(new_actions) > 5:
                    logger.info(f"  ... and {len(new_actions)-5} more")
            else:
                logger.info("No new actions since last check")
            
            # Sleep between games to avoid server overload
            time.sleep(1 + random.random())  # 1-2 seconds
            
        except Exception as e:
            logger.error(f"Error analyzing game {games_id}: {e}", exc_info=True)
    
    return results


def main() -> int:
    """Main function."""

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--games-id',
        type=int,
        action='append',
        help='Games ID to analyze (can be repeated)'
    )
    ap.add_argument(
        '--from-list',
        action='store_true',
        help='Get games from yourgames.php'
    )
    ap.add_argument(
        '--reset-cache',
        action='store_true',
        help='Reset cached replay positions'
    )
    ap.add_argument(
        '--debug-html',
        type=int,
        help='Game ID to write debug HTML for'
    )
    
    args = ap.parse_args()
    
    # Handle debug HTML request
    if args.debug_html:
        if args.debug_html not in REPLAY_URL_TO_DEBUGFILE:
            REPLAY_URL_TO_DEBUGFILE[args.debug_html] = CACHE_DIR / f"{args.debug_html}_replay_html.html"
    
    # Reset cache if requested
    if args.reset_cache and CACHE_DIR.exists():
        for file in CACHE_DIR.glob("*.json"):
            file.unlink()
        logger.info("Cache reset complete")
    
    games_ids = []
    if args.games_id:
        games_ids = args.games_id
    elif args.from_list:
        # Get games from yourgames.php
        from tools.amarriner_list_your_games import list_your_games_ids
        
        session = requests.Session()
        with open(ROOT / "secrets.txt", "r") as f:
            lines = f.readline()
            username = lines[0].strip()
            password = lines[1].strip()
        
        if not _login(session, username, password):
            logger.error("Login failed")
            return 1
        
        games_ids = list_your_games_ids(session)
        logger.info(f"Found {len(games_ids)} games: {games_ids}")
    
    if not games_ids:
        logger.error("No games IDs specified. Use --games-id or --from-list")
        return 1
    
    analyze_games(games_ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())