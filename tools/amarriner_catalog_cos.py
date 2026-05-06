"""
GL catalog rows must identify **both** COs. A game without ``co_p0_id`` and
``co_p1_id`` cannot be loaded into the engine (``make_initial_state`` requires
two real ``co_id`` values).

Listing HTML sometimes omits a portrait (e.g. old ``ds/``-only regex); fix the
scraper in ``amarriner_gl_catalog.py`` and re-run ``build``, or patch the JSON
after manual inspection — do not silently substitute ``-1``.
"""
from __future__ import annotations

from typing import Any


def catalog_row_has_both_cos(g: dict[str, Any]) -> bool:
    """True if both CO columns are present (non-null) in the cached listing row."""
    return g.get("co_p0_id") is not None and g.get("co_p1_id") is not None


def pair_catalog_cos_ids(g: dict[str, Any]) -> tuple[int, int]:
    """
    Return ``(co_p0_id, co_p1_id)`` as ints.

    Raises
    -----
    ValueError
        If either id is missing or not an integer (e.g. 'N' placeholder from fogged listings).
    """
    a, b = g.get("co_p0_id"), g.get("co_p1_id")
    if a is None or b is None:
        gid = g.get("games_id", "?")
        raise ValueError(
            f"games_id={gid}: catalog missing co_p0_id and/or co_p1_id (got co_p0_id={a!r}, co_p1_id={b!r}). "
            "Re-run `python tools/amarriner_gl_catalog.py build` after fixing portrait parsing, "
            "or edit data/amarriner_gl_std_catalog.json when the listing is wrong."
        )
    # Guard against 'N' placeholders from fogged/unknown CO listings
    try:
        a_int = int(a)
        b_int = int(b)
    except (TypeError, ValueError) as exc:
        gid = g.get("games_id", "?")
        raise ValueError(
            f"games_id={gid}: invalid co_p0_id={a!r} or co_p1_id={b!r} (not an integer). "
            f"Original error: {exc}. "
            "Re-run `python tools/amarriner_gl_catalog.py build` after fixing portrait parsing."
        ) from exc
    return a_int, b_int
