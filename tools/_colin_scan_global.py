"""Scan gamescompleted.php?start= for games where P0 or P1 uses co_id 15."""
from __future__ import annotations

import re
import sys
import time
import urllib.request

BASE = "https://awbw.amarriner.com"
_RE_GAME_ANCHOR = re.compile(r'<a class="anchor" name="game_(\d+)"></a>', re.I)
_RE_PLAYER_ROW_CO = re.compile(
    r'id="do-game-player-row"[^>]*>.*?class=.co_portrait.*?\.png\?v=(\d+)\?v=',
    re.I | re.S,
)
# legacy GL listing (two portraits in matchup, not do-game-player-row)
_RE_LEGACY_CO = re.compile(
    r"class='co_portrait'\s+src=terrain/co-portraits/[^/]+/[^?]+\.png\?v=(\d+)\?v=",
    re.I,
)


def fetch(url: str) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (compatible; Colin-global/1.0)"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8", "replace")


def co_pair_from_chunk(chunk: str) -> tuple[int | None, int | None]:
    row_ids = [int(x) for x in _RE_PLAYER_ROW_CO.findall(chunk)]
    if len(row_ids) >= 2:
        return row_ids[0], row_ids[1]
    legacy = [int(x) for x in _RE_LEGACY_CO.findall(chunk)]
    # Legacy: first two in matchup row (may include ban strip — GL catalog uses same heuristic)
    if len(legacy) >= 2:
        return legacy[0], legacy[1]
    if len(legacy) == 1:
        return legacy[0], None
    return None, None


def main() -> None:
    hits: list[tuple[int, int | None, int | None]] = []
    max_pages = 80
    for page in range(max_pages):
        start = 50 * page + 1
        url = f"{BASE}/gamescompleted.php?start={start}"
        html = fetch(url)
        matches = list(_RE_GAME_ANCHOR.finditer(html))
        if not matches:
            print(f"no anchors start={start}", file=sys.stderr)
            break
        for i, m in enumerate(matches):
            gid = int(m.group(1))
            s = m.start()
            e = matches[i + 1].start() if i + 1 < len(matches) else len(html)
            chunk = html[s:e]
            p0, p1 = co_pair_from_chunk(chunk)
            if p0 == 15 or p1 == 15:
                hits.append((gid, p0, p1))
        print(
            f"page {page+1} start={start} games={len(matches)} colin_hits_total={len(hits)}",
            flush=True,
        )
        if len(matches) < 50:
            break
        time.sleep(0.3)
        if len(hits) >= 15:
            break
    print("HITS:")
    for t in hits:
        print(t)


if __name__ == "__main__":
    main()
