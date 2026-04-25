"""Fetch game.php metadata for Colin game ids."""
from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GIDS = [
    1636107,
    1558571,
    1637153,
    1638360,
    1637096,
    1638136,
    1636411,
    1620117,
    1628024,
    1629555,
    1637705,
    1358720,
    1636108,
    1619141,
    1637200,
]


def fetch(gid: int) -> str:
    url = f"https://awbw.amarriner.com/game.php?games_id={gid}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read().decode("utf-8", "replace")


def parse(html: str, gid: int) -> dict:
    title_m = re.search(r"<title>\s*Game\s*-\s*([^<]+?)\s*-\s*AWBW", html, re.I)
    title = title_m.group(1).strip() if title_m else None
    pcm = re.search(r"players_co_id\[(\d+)\]\s*:\s*(\d+)", html)
    # Vue may use players_co_id: [15, 3] — try bracket list
    co_list = [int(x) for x in re.findall(r"players_co_id[\"']?\s*:\s*\[([^\]]+)\]", html)]
    if co_list:
        # wrong - need full match
        pass
    players_co = [int(x) for x in re.findall(r"players_co_id[\"']?\s*:\s*(\d+)", html)]
    # single assignments per line
    lines_co = re.findall(r"players_co_id\[(\d+)\]\s*=\s*(\d+)", html)
    if lines_co:
        cos = [int(v) for _, v in sorted(lines_co, key=lambda x: int(x[0]))]
    else:
        m2 = re.search(r"players_co_id\s*:\s*\[([^\]]+)\]", html)
        if m2:
            cos = [int(x.strip()) for x in m2.group(1).split(",") if x.strip().isdigit()]
        else:
            cos = []
    map_m = re.search(r"prevmaps\.php\?maps_id=(\d+)", html)
    map_id = int(map_m.group(1)) if map_m else None
    # tier: guess from title GL ... [Tx]
    tier_m = re.search(r"\[T(\d+)\]", title or "")
    tier = f"T{tier_m.group(1)}" if tier_m else None
    return {
        "games_id": gid,
        "tier": tier,
        "matchup": title,
        "map_id": map_id,
        "map_name": None,
        "co_p0_id": cos[0] if len(cos) > 0 else None,
        "co_p1_id": cos[1] if len(cos) > 1 else None,
    }


def main() -> None:
    ts = datetime.now(timezone.utc).isoformat()
    games: dict[str, dict] = {}
    for gid in GIDS:
        html = fetch(gid)
        row = parse(html, gid)
        row["list_index"] = None
        row["source_start"] = None
        row["scraped_at"] = ts
        games[str(gid)] = row
        print(row)
    out = {
        "meta": {
            "source": "gamescompleted.php global crawl + game.php metadata",
            "scraped_at": ts,
            "n_games": len(games),
            "note": "Colin (co_id 15) games from global completed listing; T0/FOG/Live mixes.",
        },
        "games": games,
    }
    p = ROOT / "data" / "amarriner_gl_colin_batch.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
