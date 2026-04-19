#!/usr/bin/env python3
"""
Download map + unit PNGs from upstream AWBW Replay Player texture paths on GitHub.

Writes under server/static/awbw_textures/ and generates manifest.json for board.js.

Tiles.json / Units.json are loaded from ``third_party/AWBW-Replay-Player/.../Json`` when
present; otherwise they are fetched from raw.githubusercontent.com (no local clone).

  python tools/sync_awbw_textures.py

Upstream: https://github.com/DeamonHunter/AWBW-Replay-Player (MIT).
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TP = ROOT / "third_party" / "AWBW-Replay-Player" / "AWBWApp.Resources" / "Json"
OUT = ROOT / "server" / "static" / "awbw_textures"
RAW = "https://raw.githubusercontent.com/DeamonHunter/AWBW-Replay-Player/master/AWBWApp.Resources/Textures"
_JSON_BASE = (
    "https://raw.githubusercontent.com/DeamonHunter/AWBW-Replay-Player/"
    "master/AWBWApp.Resources/Json"
)

# sys.path for `import engine` when running as script
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# AWBW terrain IDs used by engine/maps but absent from viewer Tiles.json (or no Clear texture).
# Values are paths under Map/Classic/ matching other manifest entries.
EXTRA_TERRAIN_BY_AWBW_ID: dict[int, str] = {
    111: "Map/Classic/Plain.png",  # full missile silo — viewer JSON has no dedicated tile
    113: "Map/Classic/Pipe/H.png",  # H pipe seam
    114: "Map/Classic/Pipe/V.png",  # V pipe seam
}

# Engine UnitType order (0..26) -> Units.json object keys (IdleAnimation.Texture stem)
UNIT_JSON_KEYS = [
    "Infantry", "Mech", "Recon", "Tank", "Md.Tank", "Neotank", "Mega Tank", "APC",
    "Artillery", "Rocket", "Anti-Air", "Missile", "Fighter", "Bomber", "Stealth",
    "B-Copter", "T-Copter", "Battleship", "Carrier", "Sub", "Cruiser", "Lander",
    "Lander",  # GUNBOAT — not in viewer Units.json; Lander idle is the closest hull
    "Black Boat", "Black Bomb", "Piperunner",
    None,  # engine 26 OOZIUM — no sheet in viewer JSON; reuse Infantry sprite
]


def _strip_json_comments(text: str) -> str:
    return re.sub(r"//[^\n]*", "", text)


# engine.engine.terrain.TerrainInfo.country_id -> Replay Player Map/AW2 folder (idle -0 frame)
_COUNTRY_ID_TO_AW2_FOLDER: dict[int | None, str] = {
    None: "Neutral",
    1: "OrangeStar",
    2: "BlueMoon",
    3: "GreenEarth",
    4: "YellowComet",
    5: "BlackHole",
    6: "RedFire",
    7: "GreySky",
    8: "BrownDesert",
    9: "AmberBlossom",
    10: "JadeSun",
    11: "CobaltIce",
    12: "PinkCosmos",
    13: "TealGalaxy",
    14: "PurpleLightning",
    15: "AcidRain",
    16: "WhiteNova",
    17: "AzureAsteroid",
    18: "NoirEclipse",
    19: "SilverClaw",
    20: "UmberWilds",
}


def _aw2_building_texture_rels() -> dict[int, str]:
    """Map AWBW terrain IDs for capturable / factory tiles to Map/AW2/.../*-0.png paths."""
    from engine.terrain import TERRAIN_TABLE

    out: dict[int, str] = {}
    for tid, info in TERRAIN_TABLE.items():
        if not info.is_property:
            continue
        folder = _COUNTRY_ID_TO_AW2_FOLDER.get(info.country_id)
        if folder is None:
            continue
        if info.is_hq:
            stem = "HQ"
        elif info.is_lab:
            stem = "Lab"
        elif info.is_comm_tower:
            stem = "ComTower"
        elif info.is_base:
            stem = "Base"
        elif info.is_airport:
            stem = "Airport"
        elif info.is_port:
            stem = "Port"
        else:
            stem = "City"
        rel = f"Map/AW2/{folder}/{stem}-0.png"
        out[int(tid)] = rel
    return out


def _fetch_utf8(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "AWBW-RL-texture-sync"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read().decode("utf-8")


def _load_json_text(filename: str) -> str:
    local = TP / filename
    if local.is_file():
        return local.read_text(encoding="utf-8")
    url = f"{_JSON_BASE}/{filename}"
    print(f"Fetching {url} …", file=sys.stderr)
    return _fetch_utf8(url)


def _download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "AWBW-RL-texture-sync"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return True
    except Exception as exc:
        print(f"  FAIL {url}: {exc}", file=sys.stderr)
        return False


def main() -> None:
    try:
        tiles = json.loads(_strip_json_comments(_load_json_text("Tiles.json")))
        units = json.loads(_strip_json_comments(_load_json_text("Units.json")))
    except Exception as exc:
        print(f"Failed to load Tiles.json / Units.json: {exc}", file=sys.stderr)
        sys.exit(1)

    terrain_by_id: dict[str, str] = {}
    rel_paths: list[str] = []

    for _name, tile in tiles.items():
        tid = tile.get("AWBWID")
        if tid is None:
            continue
        tex = (tile.get("Textures") or {}).get("Clear")
        if not tex:
            continue
        rel = f"Map/Classic/{tex}.png"
        terrain_by_id[str(int(tid))] = rel
        if rel not in rel_paths:
            rel_paths.append(rel)

    for aid, rel in EXTRA_TERRAIN_BY_AWBW_ID.items():
        k = str(int(aid))
        if k not in terrain_by_id:
            terrain_by_id[k] = rel
        if rel not in rel_paths:
            rel_paths.append(rel)

    for tid, rel in _aw2_building_texture_rels().items():
        k = str(int(tid))
        if k not in terrain_by_id:
            terrain_by_id[k] = rel
        if rel not in rel_paths:
            rel_paths.append(rel)

    # Unit idle first frame: Units/{OrangeStar|BlueMoon}/{Texture}-0.png
    unit_rows: list[dict] = []
    for engine_id, ukey in enumerate(UNIT_JSON_KEYS):
        if ukey is None:
            stem = "Infantry"
        else:
            entry = units.get(ukey)
            if not entry:
                print(f"WARN: missing Units.json key {ukey!r}", file=sys.stderr)
                stem = "Infantry"
            else:
                stem = (entry.get("IdleAnimation") or {}).get("Texture") or "Infantry"
        for country in ("OrangeStar", "BlueMoon"):
            rel = f"Units/{country}/{stem}-0.png"
            unit_rows.append(
                {
                    "engineTypeId": engine_id,
                    "player": 0 if country == "OrangeStar" else 1,
                    "rel": rel,
                }
            )
            if rel not in rel_paths:
                rel_paths.append(rel)

    print(f"Downloading {len(rel_paths)} textures to {OUT} …")
    ok = 0
    for i, rel in enumerate(rel_paths):
        url = f"{RAW}/{rel.replace(chr(92), '/')}"
        dest = OUT / rel
        if dest.is_file() and dest.stat().st_size > 0:
            ok += 1
            continue
        if _download(url, dest):
            ok += 1
        time.sleep(0.05)  # be gentle on raw.githubusercontent.com
        if (i + 1) % 25 == 0:
            print(f"  … {i + 1}/{len(rel_paths)}")

    manifest = {
        "source": "DeamonHunter/AWBW-Replay-Player",
        "terrainByAwbwId": terrain_by_id,
        "units": unit_rows,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. Wrote manifest and {ok}/{len(rel_paths)} files under {OUT}")


if __name__ == "__main__":
    main()
