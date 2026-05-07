"""Canonical unit-name resolver — Phase 11Z whack-a-mole closeout.

One backing table, one normalize step. Every consumer (oracle replay,
state-mismatch comparator, PHP/JSON exporter, predeployed-units
fetcher) routes through ``to_unit_type`` / ``from_unit_type`` instead
of carrying its own alias dict.

Audit + design: ``docs/oracle_exception_audit/phase11z_unit_naming_canon_audit.md``.
"""
from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Mapping

from engine.unit import UnitType

__all__ = (
    "UnitNameSurface",
    "UnknownUnitName",
    "to_unit_type",
    "from_unit_type",
    "is_known_alias",
    "all_known_aliases",
    "normalize_alias_key",
)


class UnitNameSurface(str, Enum):
    """Where a unit-type string lives in the wild.

    Each surface owns the canonical render for its venue plus any
    accepted aliases for inbound resolution. The first entry of each
    tuple in ``_TABLE`` is the canonical render returned by
    ``from_unit_type``; the rest are inbound-only aliases.
    """

    ENGINE = "engine"
    """``engine.unit.UNIT_STATS[ut].name`` — internal Python canonical."""

    AWBW_PHP = "awbw_php"
    """``tools/export_awbw_replay._AWBW_UNIT_NAMES`` — what the engine
    emits into PHP-serialized snapshots and into JSON action streams.
    Kept stable to preserve replay-zip backwards compatibility."""

    AWBW_VIEWER = "awbw_viewer"
    """Upstream C# AWBW Replay Player, ``AWBWApp.Resources/Json/Units.json``
    keys (commit 3ccbc60, 2025-12-30). Live AWBW site
    ``replay_download.php`` payloads also use this canonical
    spelling — empirical sample of 951 GL zips at audit time agreed
    100%."""

    AWBW_DAMAGE_PHP = "awbw_damage_php"
    """The 27 short labels used in ``data/damage_table.json::unit_order``
    (also visible on https://awbw.amarriner.com/damage.php as row
    headers). Documentation-only at runtime today; we recognize them
    so any future consumer can route through the canon."""

    DISPLAY = "display"
    """Human-friendly render (``Mega Tank`` not ``MEGA_TANK``,
    ``Medium Tank`` not ``Md.Tank``). Used for log lines / error
    messages where readability beats venue fidelity."""


class UnknownUnitName(ValueError):
    """Raised when an alias does not resolve to a known UnitType.

    Carries the original input plus the surfaces that were searched so
    callers can produce diagnostic messages. Does **not** inherit from
    any oracle-replay exception; consumers that need the oracle error
    contract must catch and re-raise (see
    ``tools/oracle_zip_replay._name_to_unit_type``).
    """

    def __init__(self, name: str, tried_surfaces: tuple["UnitNameSurface", ...]):
        self.name = name
        self.tried_surfaces = tried_surfaces
        super().__init__(
            f"unknown unit name {name!r} (tried surfaces: "
            f"{', '.join(s.value for s in tried_surfaces)})"
        )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def normalize_alias_key(name: str) -> str:
    """Lowercase, strip whitespace, drop ``-``, ``.``, ``_``, ``space``.

    Mirrors ``tools/desync_audit._canonicalize_unit_type_name``'s shape
    (the one that already worked for the cosmetic comparator). Keep
    the punctuation set tight — ``apostrophe``/``slash``/digits are
    not stripped because no AWBW unit name contains them, and we want
    a future garbage input ``"Tank/2"`` to fail rather than collide
    with ``Tank``.
    """
    s = str(name).strip().lower()
    for ch in (" ", "-", ".", "_"):
        s = s.replace(ch, "")
    return s


# ---------------------------------------------------------------------------
# Backing table — single source of truth
# ---------------------------------------------------------------------------
# Per (UnitType, surface): the first string is the canonical render the
# surface uses; subsequent strings are accepted aliases for inbound
# resolution (``to_unit_type``). Keep the canonical entry **first** —
# ``from_unit_type`` returns ``_TABLE[ut][surface][0]``.
#
# Add aliases by extending the appropriate tuple. Run the corpus
# regression and canon tests; do not add a parallel dict elsewhere.
_TABLE_RAW: dict[UnitType, dict[UnitNameSurface, tuple[str, ...]]] = {
    UnitType.INFANTRY: {
        UnitNameSurface.ENGINE: ("Infantry",),
        UnitNameSurface.AWBW_PHP: ("Infantry",),
        UnitNameSurface.AWBW_VIEWER: ("Infantry",),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("Infantry",),
        UnitNameSurface.DISPLAY: ("Infantry",),
    },
    UnitType.MECH: {
        UnitNameSurface.ENGINE: ("Mech",),
        UnitNameSurface.AWBW_PHP: ("Mech",),
        UnitNameSurface.AWBW_VIEWER: ("Mech",),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("Mech",),
        UnitNameSurface.DISPLAY: ("Mech",),
    },
    UnitType.RECON: {
        UnitNameSurface.ENGINE: ("Recon",),
        UnitNameSurface.AWBW_PHP: ("Recon",),
        UnitNameSurface.AWBW_VIEWER: ("Recon",),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("Recon",),
        UnitNameSurface.DISPLAY: ("Recon",),
    },
    UnitType.TANK: {
        UnitNameSurface.ENGINE: ("Tank",),
        UnitNameSurface.AWBW_PHP: ("Tank",),
        UnitNameSurface.AWBW_VIEWER: ("Tank",),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("Tank",),
        UnitNameSurface.DISPLAY: ("Tank",),
    },
    UnitType.MED_TANK: {
        UnitNameSurface.ENGINE: ("Medium Tank", "Med Tank", "MedTank"),
        UnitNameSurface.AWBW_PHP: ("Md. Tank", "Md.Tank"),
        UnitNameSurface.AWBW_VIEWER: ("Md.Tank", "Md. Tank"),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("MedTank",),
        UnitNameSurface.DISPLAY: ("Medium Tank",),
    },
    UnitType.NEO_TANK: {
        UnitNameSurface.ENGINE: ("Neotank",),
        UnitNameSurface.AWBW_PHP: ("Neo Tank", "Neotank"),
        UnitNameSurface.AWBW_VIEWER: ("Neotank", "Neo Tank"),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("NeoTank",),
        UnitNameSurface.DISPLAY: ("Neotank",),
    },
    UnitType.MEGA_TANK: {
        UnitNameSurface.ENGINE: ("Megatank",),
        UnitNameSurface.AWBW_PHP: ("Mega Tank", "Megatank"),
        UnitNameSurface.AWBW_VIEWER: ("Mega Tank", "Megatank"),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("MegaTank",),
        UnitNameSurface.DISPLAY: ("Mega Tank",),
    },
    UnitType.APC: {
        UnitNameSurface.ENGINE: ("APC",),
        UnitNameSurface.AWBW_PHP: ("APC",),
        UnitNameSurface.AWBW_VIEWER: ("APC",),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("APC",),
        UnitNameSurface.DISPLAY: ("APC",),
    },
    UnitType.ARTILLERY: {
        UnitNameSurface.ENGINE: ("Artillery",),
        UnitNameSurface.AWBW_PHP: ("Artillery",),
        UnitNameSurface.AWBW_VIEWER: ("Artillery",),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("Artillery",),
        UnitNameSurface.DISPLAY: ("Artillery",),
    },
    UnitType.ROCKET: {
        UnitNameSurface.ENGINE: ("Rocket",),
        UnitNameSurface.AWBW_PHP: ("Rockets", "Rocket"),
        UnitNameSurface.AWBW_VIEWER: ("Rockets", "Rocket"),  # C# viewer expects plural
        UnitNameSurface.AWBW_DAMAGE_PHP: ("Rocket",),
        UnitNameSurface.DISPLAY: ("Rocket",),
    },
    UnitType.ANTI_AIR: {
        UnitNameSurface.ENGINE: ("Anti-Air", "Anti Air"),
        UnitNameSurface.AWBW_PHP: ("Anti-Air", "Anti Air"),
        UnitNameSurface.AWBW_VIEWER: ("Anti-Air",),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("AntiAir",),
        UnitNameSurface.DISPLAY: ("Anti-Air",),
    },
    UnitType.MISSILES: {
        UnitNameSurface.ENGINE: ("Missiles",),
        # Live AWBW site emits singular ``Missile`` (matching the C#
        # viewer); ``damage.php`` row label is plural ``Missiles``.
        # Both are accepted on inbound; outbound stays plural for
        # exporter backwards compatibility.
        UnitNameSurface.AWBW_PHP: ("Missiles", "Missile"),
        UnitNameSurface.AWBW_VIEWER: ("Missile", "Missiles"),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("Missiles",),
        UnitNameSurface.DISPLAY: ("Missiles",),
    },
    UnitType.FIGHTER: {
        UnitNameSurface.ENGINE: ("Fighter",),
        UnitNameSurface.AWBW_PHP: ("Fighter",),
        UnitNameSurface.AWBW_VIEWER: ("Fighter",),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("Fighter",),
        UnitNameSurface.DISPLAY: ("Fighter",),
    },
    UnitType.BOMBER: {
        UnitNameSurface.ENGINE: ("Bomber",),
        UnitNameSurface.AWBW_PHP: ("Bomber",),
        UnitNameSurface.AWBW_VIEWER: ("Bomber",),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("Bomber",),
        UnitNameSurface.DISPLAY: ("Bomber",),
    },
    UnitType.STEALTH: {
        UnitNameSurface.ENGINE: ("Stealth",),
        UnitNameSurface.AWBW_PHP: ("Stealth",),
        UnitNameSurface.AWBW_VIEWER: ("Stealth",),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("Stealth",),
        UnitNameSurface.DISPLAY: ("Stealth",),
    },
    UnitType.B_COPTER: {
        UnitNameSurface.ENGINE: ("B-Copter", "B Copter", "BCopter"),
        UnitNameSurface.AWBW_PHP: ("B-Copter", "B Copter"),
        UnitNameSurface.AWBW_VIEWER: ("B-Copter",),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("BCopter",),
        UnitNameSurface.DISPLAY: ("B-Copter",),
    },
    UnitType.T_COPTER: {
        UnitNameSurface.ENGINE: ("T-Copter", "T Copter", "TCopter"),
        UnitNameSurface.AWBW_PHP: ("T-Copter", "T Copter"),
        UnitNameSurface.AWBW_VIEWER: ("T-Copter",),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("TCopter",),
        UnitNameSurface.DISPLAY: ("T-Copter",),
    },
    UnitType.BATTLESHIP: {
        UnitNameSurface.ENGINE: ("Battleship",),
        UnitNameSurface.AWBW_PHP: ("Battleship",),
        UnitNameSurface.AWBW_VIEWER: ("Battleship",),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("Battleship",),
        UnitNameSurface.DISPLAY: ("Battleship",),
    },
    UnitType.CARRIER: {
        UnitNameSurface.ENGINE: ("Carrier",),
        UnitNameSurface.AWBW_PHP: ("Carrier",),
        UnitNameSurface.AWBW_VIEWER: ("Carrier",),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("Carrier",),
        UnitNameSurface.DISPLAY: ("Carrier",),
    },
    UnitType.SUBMARINE: {
        UnitNameSurface.ENGINE: ("Submarine",),
        UnitNameSurface.AWBW_PHP: ("Sub", "Submarine"),
        UnitNameSurface.AWBW_VIEWER: ("Sub", "Submarine"),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("Submarine",),
        UnitNameSurface.DISPLAY: ("Submarine",),
    },
    UnitType.CRUISER: {
        UnitNameSurface.ENGINE: ("Cruiser",),
        UnitNameSurface.AWBW_PHP: ("Cruiser",),
        UnitNameSurface.AWBW_VIEWER: ("Cruiser",),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("Cruiser",),
        UnitNameSurface.DISPLAY: ("Cruiser",),
    },
    UnitType.LANDER: {
        UnitNameSurface.ENGINE: ("Lander",),
        UnitNameSurface.AWBW_PHP: ("Lander",),
        UnitNameSurface.AWBW_VIEWER: ("Lander",),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("Lander",),
        UnitNameSurface.DISPLAY: ("Lander",),
    },
    UnitType.GUNBOAT: {
        UnitNameSurface.ENGINE: ("Gunboat",),
        UnitNameSurface.AWBW_PHP: ("Gunboat",),
        # Not on AWBW / not in C# ``Units.json`` (Advance Wars: Days of Ruin).
        # Engine retains stats for damage oracle; ports cannot build it (``action.py``).
        # Desktop replay zip must never contain this ``units_name`` — no viewer key.
        UnitNameSurface.AWBW_VIEWER: ("Gunboat",),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("Gunboat",),
        UnitNameSurface.DISPLAY: ("Gunboat",),
    },
    UnitType.BLACK_BOAT: {
        UnitNameSurface.ENGINE: ("Black Boat",),
        UnitNameSurface.AWBW_PHP: ("Black Boat",),
        UnitNameSurface.AWBW_VIEWER: ("Black Boat",),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("BlackBoat",),
        UnitNameSurface.DISPLAY: ("Black Boat",),
    },
    UnitType.BLACK_BOMB: {
        UnitNameSurface.ENGINE: ("Black Bomb",),
        UnitNameSurface.AWBW_PHP: ("Black Bomb",),
        UnitNameSurface.AWBW_VIEWER: ("Black Bomb",),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("BlackBomb",),
        UnitNameSurface.DISPLAY: ("Black Bomb",),
    },
    UnitType.PIPERUNNER: {
        UnitNameSurface.ENGINE: ("Piperunner",),
        UnitNameSurface.AWBW_PHP: ("Piperunner",),
        UnitNameSurface.AWBW_VIEWER: ("Piperunner",),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("Piperunner",),
        UnitNameSurface.DISPLAY: ("Piperunner",),
    },
    UnitType.OOZIUM: {
        UnitNameSurface.ENGINE: ("Oozium",),
        UnitNameSurface.AWBW_PHP: ("Oozium",),
        # Campaign-only on DS; not in C# ``Units.json``. Map ban ``Oozium`` is typical.
        UnitNameSurface.AWBW_VIEWER: ("Oozium",),
        UnitNameSurface.AWBW_DAMAGE_PHP: ("Oozium",),
        UnitNameSurface.DISPLAY: ("Oozium",),
    },
}


def _freeze() -> Mapping[UnitType, Mapping[UnitNameSurface, tuple[str, ...]]]:
    inner: dict[UnitType, Mapping[UnitNameSurface, tuple[str, ...]]] = {}
    for ut, sd in _TABLE_RAW.items():
        inner[ut] = MappingProxyType(dict(sd))
    return MappingProxyType(inner)


_TABLE: Mapping[UnitType, Mapping[UnitNameSurface, tuple[str, ...]]] = _freeze()


# ---------------------------------------------------------------------------
# Reverse index for to_unit_type — built once, normalized
# ---------------------------------------------------------------------------
def _build_reverse_index() -> Mapping[str, tuple[UnitType, UnitNameSurface]]:
    """Index: normalize_alias_key(alias) → (UnitType, surface).

    Surface attribution is best-effort (first surface that defined the
    alias wins). Used only for diagnostic surface-tracing in
    ``UnknownUnitName``; ``to_unit_type`` itself only cares about the
    UnitType.
    """
    rev: dict[str, tuple[UnitType, UnitNameSurface]] = {}
    # Also accept the engine enum *member* name (e.g. ``MED_TANK``) as a
    # universal alias — historical tests rely on this.
    for ut in UnitType:
        rev.setdefault(normalize_alias_key(ut.name), (ut, UnitNameSurface.ENGINE))
    for ut, sd in _TABLE_RAW.items():
        for surface, names in sd.items():
            for nm in names:
                rev.setdefault(normalize_alias_key(nm), (ut, surface))
    # Cross-check: every UnitType is reachable
    seen_uts = {ut for ut, _ in rev.values()}
    missing = set(UnitType) - seen_uts
    if missing:  # pragma: no cover — defensive
        raise RuntimeError(
            f"unit_naming canon is missing entries for: "
            f"{sorted(u.name for u in missing)}"
        )
    return MappingProxyType(rev)


_REVERSE: Mapping[str, tuple[UnitType, UnitNameSurface]] = _build_reverse_index()


_ALL_SURFACES: tuple[UnitNameSurface, ...] = tuple(UnitNameSurface)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def to_unit_type(
    name: "str | UnitType",
    *,
    surface: UnitNameSurface | None = None,
) -> UnitType:
    """Resolve any known string to a UnitType.

    ``name`` may already be a UnitType (returned unchanged) or any
    recognized alias from any surface. ``surface`` is advisory: when
    given, the resolution still consults the global reverse index but
    the resulting UnitType is verified to be present in
    ``_TABLE[ut][surface]``. This catches programmer errors like
    asking for ``Sub`` from the ``ENGINE`` surface.
    """
    if isinstance(name, UnitType):
        return name
    key = normalize_alias_key(name)
    hit = _REVERSE.get(key)
    if hit is None:
        raise UnknownUnitName(str(name), _ALL_SURFACES)
    ut, _hit_surface = hit
    if surface is not None:
        if surface not in _TABLE[ut]:  # pragma: no cover — defensive
            raise UnknownUnitName(str(name), (surface,))
        if key not in {normalize_alias_key(a) for a in _TABLE[ut][surface]} \
                and key != normalize_alias_key(ut.name):
            raise UnknownUnitName(str(name), (surface,))
    return ut


def from_unit_type(ut: UnitType, surface: UnitNameSurface) -> str:
    """Render ``ut`` for ``surface`` (the canonical/first entry)."""
    if not isinstance(ut, UnitType):  # pragma: no cover — defensive
        raise TypeError(f"expected UnitType, got {type(ut).__name__}")
    sd = _TABLE.get(ut)
    if sd is None or surface not in sd:
        raise ValueError(
            f"no canonical render for {ut.name} on surface {surface.value!r}"
        )
    return sd[surface][0]


def is_known_alias(name: str) -> bool:
    """True if ``name`` resolves to any UnitType on any surface."""
    if isinstance(name, UnitType):
        return True
    return normalize_alias_key(name) in _REVERSE


def all_known_aliases() -> Mapping[str, UnitType]:
    """Mapping of every recognized alias (original casing) → UnitType.

    Used by tests as a coverage gate. Multiple aliases may collide
    after ``normalize_alias_key``; this view shows the *original*
    spellings as authored, so test asserts of
    ``len(all_known_aliases())`` reflect the human-visible coverage.
    """
    out: dict[str, UnitType] = {}
    for ut in UnitType:
        out.setdefault(ut.name, ut)
    for ut, sd in _TABLE_RAW.items():
        for _surface, names in sd.items():
            for nm in names:
                out.setdefault(nm, ut)
    return MappingProxyType(out)
