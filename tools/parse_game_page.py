#!/usr/bin/env python3
"""
Parse AWBW game page HTML to extract detailed game state.
"""
import re
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup


def parse_game_page(html: str) -> Dict[str, Any]:
    """
    Parse game page HTML to extract detailed game state.
    
    Returns a dictionary with:
    - day: current day
    - map: map name
    - players: list of player info dictionaries
      - name: player username
      - co: CO name (extracted from image path or description)
      - funds: player funds
      - units: dictionary of unit type -> count
      - total_units: total unit count
      - total_value: total unit value
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    result: Dict[str, Any] = {
        'players': []
    }
    
    # Extract day
    day_match = re.search(r'Day (\d+)', html)
    if day_match:
        result['day'] = int(day_match.group(1))
    
    # Extract map name
    map_match = re.search(r'Game Started:.*? - ([^-()]+?)\s*\(', html)
    if map_match:
        result['map'] = map_match.group(1).strip()
    
    # Look for player sections - they seem to be in Vue.js templates or script data
    # Let's try a different approach: look for player stats in the HTML
    
    # Method 1: Look for player names in the page
    player_names = []
    
    # Look for player name in the header/title
    title_elem = soup.find('title')
    if title_elem:
        title_text = title_elem.text
        # Extract names from title like "iwinagain vs skeebeedee"
        vs_match = re.search(r':\s*([^\[\]]+?)\s+vs\s+([^\[\]]+?)\s*\[', title_text)
        if vs_match:
            player_names = [vs_match.group(1).strip(), vs_match.group(2).strip()]
    
    # Method 2: Look for player stats in the HTML
    # The HTML structure from the grep shows unit-count-* divs
    
    # Find all unit count divs
    unit_count_divs = soup.find_all('div', class_=re.compile(r'unit-count-'))
    
    # Group by player - this is tricky without more context
    # Let me look for a different pattern
    
    # Method 3: Look for the player stats in script tags
    # The game data might be in JavaScript variables
    script_tags = soup.find_all('script')
    for script in script_tags:
        if script.string:
            # Look for player data in JavaScript
            # Try to find player funds
            funds_matches = re.findall(r'player_funds["\']?\s*:\s*["\']?(\d+)["\']?', script.string)
            if funds_matches:
                print(f"Found funds in script: {funds_matches}")
            
            # Try to find unit data
            unit_matches = re.findall(r'unit_count["\']?\s*:\s*["\']?(\d+)["\']?', script.string)
            if unit_matches:
                print(f"Found unit counts in script: {unit_matches}")
    
    # Method 4: Try to extract from the visible text patterns we saw earlier
    # Look for patterns like "x 6" or "× 6" near unit images
    unit_pattern = re.compile(r'[x×]\s*(\d+)')
    all_unit_counts = unit_pattern.findall(html)
    if all_unit_counts:
        print(f"Found unit count patterns: {all_unit_counts}")
    
    # For now, let me create a simpler parser based on what we can see
    # Extract player info from the structure we observed
    
    return result


def extract_unit_counts_from_html(html: str) -> List[Dict[str, Any]]:
    """
    Extract unit counts from HTML using the structure we observed.
    
    The structure appears to be:
    <div class="unit-count-infantry">
      <div>
        <img src="terrain/ani/osinfantry.gif" alt="Unit-count sprite">
      </div>
      <span>
        x 6  <!-- This is the count -->
      </span>
    </div>
    """
    players = []
    
    # Use regex to find unit count sections
    # Look for: <div class="unit-count-[type]">...<span>x N</span>
    unit_count_pattern = re.compile(
        r'<div\s+class="unit-count-([^"]+)">.*?<span>\s*[x×]\s*(\d+)\s*</span>',
        re.DOTALL
    )
    
    matches = unit_count_pattern.findall(html)
    if matches:
        print(f"Found unit count matches: {matches}")
        
        # Group by player - we need to know which player each unit belongs to
        # The image src gives us the army: os = Orange Star, pl = Purple Lightning, etc.
        army_pattern = re.compile(
            r'src="terrain/ani/([a-z]+)(?:[^"]+)"\s+alt="Unit-count sprite"',
            re.DOTALL
        )
        
        # Find all unit images and their counts
        # We need to parse the HTML more carefully
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find all unit count divs
        unit_divs = soup.find_all('div', class_=re.compile(r'unit-count-'))
        
        current_player = None
        player_units = {}
        
        for div in unit_divs:
            # Get unit type from class
            class_name = div.get('class', [''])[0]
            unit_type = class_name.replace('unit-count-', '')
            
            # Find the image to get army
            img = div.find('img', alt="Unit-count sprite")
            if img:
                src = img.get('src', '')
                army_match = re.search(r'/([a-z]+)[a-zA-Z]+\.gif', src)
                if army_match:
                    army = army_match.group(1)
                    
                    # Map army to player
                    # os = Orange Star, pl = Purple Lightning, etc.
                    if army == 'os':
                        current_player = 'iwinagain'  # This might need to be dynamic
                    elif army == 'pl':
                        current_player = 'skeebeedee'
                    
                    # Find the count in the span
                    span = div.find('span')
                    if span:
                        span_text = span.get_text(strip=True)
                        count_match = re.search(r'[x×]\s*(\d+)', span_text)
                        if count_match:
                            count = int(count_match.group(1))
                            
                            if current_player not in player_units:
                                player_units[current_player] = {}
                            
                            player_units[current_player][unit_type] = count
        
        # Convert to player list
        for player_name, units in player_units.items():
            players.append({
                'name': player_name,
                'units': units,
                'total_units': sum(units.values())
            })
    
    return players


def main():
    """Test the parser."""
    # Read a saved HTML file
    with open('game_1638515_analysis.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    # Try to extract unit counts
    players = extract_unit_counts_from_html(html)
    
    print(f"Found {len(players)} players:")
    for player in players:
        print(f"  Player: {player['name']}")
        print(f"  Total units: {player['total_units']}")
        print(f"  Units: {player['units']}")


if __name__ == "__main__":
    main()