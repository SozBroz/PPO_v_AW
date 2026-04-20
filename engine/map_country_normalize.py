"""
Remap competitive-map terrain from arbitrary AWBW country pair to **Orange Star (1) + Blue Moon (2)**.

Engine ``country_id`` and ``gl_map_pool.json`` ``p0_country_id`` must stay in sync; lobby
choices are cosmetic on AWBW but **seating + terrain art** must match or BUILD/capture
drift. For debugging, standardizing every map to OS/BM + ``p0_country_id: 1`` removes
one variable when comparing replays.

Use :func:`remap_property_terrain_id_to_os_bm` per tile, or :func:`normalize_terrain_grid`
for a full CSV grid. After rewriting ``data/maps/<id>.csv``, set pool ``p0_country_id`` to ``1``.
Sidecar ``*_units.json`` is reconciled by :func:`tools.normalize_map_to_os_bm.run_normalize_map_to_os_bm`
(non-dry-run) via :func:`engine.map_loader.load_map` so predeploy rows match the new grid.
In-replay **built** units are only in replay zips / exports — not rewritten by that map tool.
"""
from __future__ import annotations

from engine.terrain import (
    get_terrain,
    is_property,
    property_terrain_id_for_country_and_kind,
)


def remap_property_terrain_id_to_os_bm(
    terrain_id: int,
    *,
    engine_p0_country_id: int,
    engine_p1_country_id: int,
) -> int:
    """
    Map a single property ``terrain_id`` to OS (1) or BM (2) **same building kind**.

    Non-properties and neutral properties (no ``country_id``) are returned unchanged.
    Tiles whose country is neither **engine P0** nor **engine P1** country are unchanged
    (third factions on weird maps — leave for manual review).
    """
    t = get_terrain(terrain_id)
    if not t.is_property:
        return terrain_id
    cid = t.country_id
    if cid is None:
        return terrain_id
    if cid == engine_p0_country_id:
        target = 1
    elif cid == engine_p1_country_id:
        target = 2
    else:
        return terrain_id

    new_tid = property_terrain_id_for_country_and_kind(
        target,
        is_hq=t.is_hq,
        is_lab=t.is_lab,
        is_comm_tower=t.is_comm_tower,
        is_base=t.is_base,
        is_airport=t.is_airport,
        is_port=t.is_port,
    )
    if new_tid is None:
        return terrain_id
    return int(new_tid)


def normalize_terrain_grid(
    terrain: list[list[int]],
    *,
    engine_p0_country_id: int,
    engine_p1_country_id: int,
) -> list[list[int]]:
    """Return a new grid with OS/BM property tiles; non-matching tiles copied."""
    out: list[list[int]] = []
    for row in terrain:
        out.append(
            [
                remap_property_terrain_id_to_os_bm(
                    tid,
                    engine_p0_country_id=engine_p0_country_id,
                    engine_p1_country_id=engine_p1_country_id,
                )
                for tid in row
            ]
        )
    return out


def infer_two_country_ids_from_grid(terrain: list[list[int]]) -> tuple[int, int] | None:
    """
    Row-major scan of property tiles — first two distinct ``country_id`` values
    (same rule as :func:`engine.map_loader.load_map` before ``p0_country_id`` remap).
    Returns ``None`` if fewer than two competitive countries appear.
    """
    from engine.terrain import get_country

    seen: list[int] = []
    for row in terrain:
        for tid in row:
            t = get_terrain(tid)
            if not t.is_property:
                continue
            c = get_country(tid)
            if c is None:
                continue
            if c not in seen:
                seen.append(c)
            if len(seen) >= 2:
                return (seen[0], seen[1])
    return None


def engine_seat_country_pair(
    terrain: list[list[int]],
    p0_country_id: int,
) -> tuple[int, int] | None:
    """
    Return ``(awbw_country_for_engine_p0, awbw_country_for_engine_p1)`` for this map
    given pool ``p0_country_id`` (which AWBW country sits in engine P0).

    Uses the two distinct property countries on the grid; returns ``None`` if that pair
    cannot be resolved (e.g. not exactly two countries, or ``p0_country_id`` not among them).
    """
    pair = infer_two_country_ids_from_grid(terrain)
    if pair is None:
        return None
    a, b = pair
    if p0_country_id not in (a, b):
        return None
    other = b if p0_country_id == a else a
    return (p0_country_id, other)
