#!/usr/bin/env python3
"""
Patch rl/encoder.py to add CO-specific tile attack bonus planes.

Why this is a script instead of a plain unified diff:
    Some checked-in files in this repo are compacted onto very long physical
    lines, so a normal line-oriented patch is brittle. This patcher performs
    guarded string rewrites and refuses to run if it cannot find the expected
    anchors.

Adds two spatial planes:
    co_tile_attack_bonus_me
    co_tile_attack_bonus_enemy

Values are normalized attack bonus percentages:
    +10% -> 0.10
    +40% -> 0.40
    +130% -> 1.30

For normal COs both planes are zero. The four terrain-sensitive COs are:
    Lash   (co_id 16): +10% per terrain star; SCOP doubles to +20%/star.
                       This plane represents ground-unit potential. Air units
                       do not receive Lash's terrain-star attack bonus in the
                       combat rules, but the map-level plane still tells the NN
                       which ground tiles are valuable for Lash.
    Koal   (co_id 21): roads +10%, COP +20%, SCOP +30%.
    Jake   (co_id 22): plains +10%, COP +20%, SCOP +40%.
    Kindle (co_id 23): properties/buildings except silos +40%, COP +80%,
                       SCOP +130%.

Run from repo root:
    python scripts/apply_co_tile_bonus_encoding_patch.py
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENCODER = ROOT / "rl" / "encoder.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"[co-tile-patch] expected exactly one {label} anchor; found {count}")
    return text.replace(old, new, 1)


def insert_after_once(text: str, anchor: str, insertion: str, label: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise SystemExit(f"[co-tile-patch] expected exactly one {label} anchor; found {count}")
    return text.replace(anchor, anchor + insertion, 1)


HELPERS = r'''

def _co_tile_attack_bonus_for_category(
    category: int,
    defense_norm: float,
    co_state,
) -> float:
    """Return terrain/property attack bonus as a normalized percentage.

    This is a map-position feature, not a replacement for combat math.
    It tells the NN how attractive a tile is for terrain-sensitive COs.
    """
    co_id = int(getattr(co_state, "co_id", -1))
    cop_active = bool(getattr(co_state, "cop_active", False))
    scop_active = bool(getattr(co_state, "scop_active", False))

    # Lash: +10% per defense star from the attacker's tile. SCOP doubles
    # attack terrain-star value. COP only doubles defensive terrain in AWBW,
    # so attack remains D2D while COP is active.
    if co_id == 16:
        stars = max(0.0, float(defense_norm)) * 4.0
        per_star = 0.20 if scop_active else 0.10
        return stars * per_star

    # Koal: roads.
    if co_id == 21:
        if category == TERRAIN_CATEGORIES["road"]:
            if scop_active:
                return 0.30
            if cop_active:
                return 0.20
            return 0.10
        return 0.0

    # Jake: plains/fields.
    if co_id == 22:
        if category == TERRAIN_CATEGORIES["plain"]:
            if scop_active:
                return 0.40
            if cop_active:
                return 0.20
            return 0.10
        return 0.0

    # Kindle: urban/property tiles, excluding missile silos. The encoder's
    # main terrain categories do not include silos as properties, so every
    # property category here is a valid urban bonus tile.
    if co_id == 23:
        if category in (
            TERRAIN_CATEGORIES["city"],
            TERRAIN_CATEGORIES["base"],
            TERRAIN_CATEGORIES["airport"],
            TERRAIN_CATEGORIES["port"],
            TERRAIN_CATEGORIES["hq"],
            TERRAIN_CATEGORIES["lab"],
        ):
            if scop_active:
                return 1.30
            if cop_active:
                return 0.80
            return 0.40
        return 0.0

    return 0.0


def _fill_co_tile_attack_bonus_planes(
    spatial: np.ndarray,
    state: GameState,
    observer: int,
    terrain_block: np.ndarray,
    defense_block: np.ndarray,
) -> None:
    """Fill observer-relative CO tile attack bonus planes.

    Channel 0 of this block is for observer/me. Channel 1 is for enemy.
    """
    base = N_CO_TILE_ATTACK_CHANNEL_BASE
    seats = (int(observer), 1 - int(observer))
    for rel_idx, seat in enumerate(seats):
        co_state = state.co_states[seat]
        if int(getattr(co_state, "co_id", -1)) not in (16, 21, 22, 23):
            continue
        out_ch = base + rel_idx
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                # Empty padded rows/cols have all-zero terrain channels.
                if not np.any(terrain_block[r, c, :]):
                    continue
                category = int(np.argmax(terrain_block[r, c, :]))
                spatial[r, c, out_ch] = _co_tile_attack_bonus_for_category(
                    category,
                    float(defense_block[r, c]),
                    co_state,
                )
'''


def main() -> None:
    text = ENCODER.read_text(encoding="utf-8")

    if "N_CO_TILE_ATTACK_CHANNELS" in text:
        print("[co-tile-patch] rl/encoder.py already has CO tile bonus channels; nothing to do")
        return

    text = text.replace(
        "Total: 28 + 2 + 15 + 15 + 3 + 6 + 1 + 7 = 77 spatial channels",
        "Total: 28 + 2 + 15 + 15 + 3 + 6 + 1 + 2 + 7 = 79 spatial channels",
        1,
    )
    text = text.replace(
        "The 70→77 / 17→16 unit-modifier migration",
        "The 70→77 / 17→16 unit-modifier migration and 77→79 CO-tile-bonus migration",
        1,
    )

    # Handle multi-line constants - we need to insert N_CO_TILE_ATTACK_CHANNELS after N_DEFENSE_STARS_CHANNELS
    # and update N_SPATIAL_CHANNELS calculation
    text = replace_once(
        text,
        "N_DEFENSE_STARS_CHANNELS = 1\nN_UNIT_MODIFIER_CHANNELS = 7",
        "N_DEFENSE_STARS_CHANNELS = 1\nN_CO_TILE_ATTACK_CHANNELS = 2\nN_UNIT_MODIFIER_CHANNELS = 7",
        "spatial-channel constants",
    )
    
    # Update N_SPATIAL_CHANNELS to include N_CO_TILE_ATTACK_CHANNELS
    text = replace_once(
        text,
        """N_SPATIAL_CHANNELS = (
    N_UNIT_CHANNELS
    + N_HP_CHANNELS
    + N_TERRAIN_CHANNELS
    + N_PROPERTY_CHANNELS
    + N_CAPTURE_EXTRA_CHANNELS
    + N_INFLUENCE_CHANNELS
    + N_DEFENSE_STARS_CHANNELS
    + N_UNIT_MODIFIER_CHANNELS
)  # 77""",
        """N_SPATIAL_CHANNELS = (
    N_UNIT_CHANNELS
    + N_HP_CHANNELS
    + N_TERRAIN_CHANNELS
    + N_PROPERTY_CHANNELS
    + N_CAPTURE_EXTRA_CHANNELS
    + N_INFLUENCE_CHANNELS
    + N_DEFENSE_STARS_CHANNELS
    + N_CO_TILE_ATTACK_CHANNELS
    + N_UNIT_MODIFIER_CHANNELS
)  # 79""",
        "spatial-channel calculation",
    )

    # Update channel base constants
    text = replace_once(
        text,
        "N_DEFENSE_STARS_CHANNEL = N_INFLUENCE_CHANNEL_BASE + N_INFLUENCE_CHANNELS\nN_UNIT_MODIFIER_CHANNEL_BASE = N_DEFENSE_STARS_CHANNEL + N_DEFENSE_STARS_CHANNELS\nUNIT_MODIFIER_CHANNEL_NAMES: tuple[str, ...] = (",
        "N_DEFENSE_STARS_CHANNEL = N_INFLUENCE_CHANNEL_BASE + N_INFLUENCE_CHANNELS\nN_CO_TILE_ATTACK_CHANNEL_BASE = N_DEFENSE_STARS_CHANNEL + N_DEFENSE_STARS_CHANNELS\nN_UNIT_MODIFIER_CHANNEL_BASE = N_CO_TILE_ATTACK_CHANNEL_BASE + N_CO_TILE_ATTACK_CHANNELS\nCO_TILE_ATTACK_CHANNEL_NAMES: tuple[str, ...] = (\n    \"co_tile_attack_bonus_me\",\n    \"co_tile_attack_bonus_enemy\",\n)\nUNIT_MODIFIER_CHANNEL_NAMES: tuple[str, ...] = (",
        "channel-base constants",
    )

    text = insert_after_once(
        text,
        'def _get_terrain_category(terrain_id: int) -> int:\n    """Map a raw terrain tile id to a TERRAIN_CATEGORIES index."""\n    cached = _TERRAIN_CATEGORY_TABLE.get(terrain_id)\n    if cached is not None:\n        return cached\n    return _compute_terrain_category_uncached(terrain_id)',
        HELPERS,
        "terrain-category helper",
    )

    text = replace_once(
        text,
        "# ── Unit modifier planes (70..76) ───────────────────────────────────────\n    # Kept in Python even when Cython is enabled so we reuse engine movement and\n    # combat helpers directly instead of duplicating CO-specific rules in C++.",
        "# ── Unit modifier planes (72..78) ───────────────────────────────────────\n    # Kept in Python even when Cython is enabled so we reuse engine movement and\n    # combat helpers directly instead of duplicating CO-specific rules in C++.",
        "unit-modifier channel comment",
    )

    text = replace_once(
        text,
        "# ── Map-static defense stars (channel 69) ────────────────────────────────\n    np.copyto(spatial[:, :, N_DEFENSE_STARS_CHANNEL], defense_block)\n\n    # ── Scalar features (ego-centric) ───────────────────────────────────────",
        "# ── Map-static defense stars (channel 69) ────────────────────────────────\n    np.copyto(spatial[:, :, N_DEFENSE_STARS_CHANNEL], defense_block)\n    \n    # ── CO-specific tile attack bonuses (channels 70..71) ────────────────────\n    _fill_co_tile_attack_bonus_planes(spatial, state, observer, terrain_block, defense_block)\n\n    # ── Scalar features (ego-centric) ───────────────────────────────────────",
        "defense-star copy site",
    )

    text = replace_once(
        text,
        "# 69:     defense stars           (/4, map-static)\n# 70-76:  unit modifiers          (move/atk/def/luck/indirect)\n\nCHANNEL_GROUPS = {",
        "# 69:     defense stars           (/4, map-static)\n# 70-71:  CO tile attack bonus planes (me/enemy)\n# 72-78:  unit modifiers          (move/atk/def/luck/indirect)\n\nCHANNEL_GROUPS = {",
        "channel-layout comment",
    )

    # Update CHANNEL_GROUPS entries
    text = replace_once(
        text,
        '    "defense_stars":    (69, 70),\n    "unit_modifiers":   (70, 77),',
        '    "defense_stars":    (69, 70),\n    "co_tile_attack_bonus": (70, 72),\n    "unit_modifiers":   (72, 79),',
        "CHANNEL_GROUPS entries",
    )

    ENCODER.write_text(text, encoding="utf-8")
    print("[co-tile-patch] patched rl/encoder.py: N_SPATIAL_CHANNELS 77 -> 79")
    print("[co-tile-patch] added planes: co_tile_attack_bonus_me/enemy")
    print("[co-tile-patch] CO ids: Lash=16, Koal=21, Jake=22, Kindle=23")


if __name__ == "__main__":
    main()