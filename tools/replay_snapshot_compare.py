"""
Compare **AWBW Replay Player reference state** (PHP ``awbwGame`` lines) to ``GameState``.

**Reference side (what we diff against):** each frame is one gzipped line of
PHP-serialized ``awbwGame`` from the site ``.zip`` — the same payload the
open-source **C# AWBW Replay Player** loads and renders
(`github.com/DeamonHunter/AWBW-Replay-Player`). We parse those bytes with
``tools.diff_replay_zips.load_replay``; we do not run the desktop app in this
harness, but we are **not** comparing to the in-repo Flask ``/replay`` JSONL
viewer (that path is engine-only).

Site zips use one of two shapes (both are valid AWBW exports — not gameplay bugs):

- **Trailing snapshot** — ``N+1`` gzip lines for ``N`` ``p:`` envelopes: ``frame[0]``
  is the opening state; after envelope ``i``, compare engine to ``frame[i+1]``
  (the last line is the final board).
- **Tight** — ``N`` gzip lines for ``N`` envelopes: same pairing for ``i = 0 .. N-2``;
  there is **no** extra line after the last envelope (common when the match ends on
  that half-turn), so we do not snapshot-compare after the final envelope.

**Under test:** our engine after ``tools.oracle_zip_replay.apply_oracle_action_json``.
"""
from __future__ import annotations

import math
from typing import Any, Literal, Optional

from engine.game import GameState
from engine.unit import UNIT_STATS
from engine.unit_naming import UnknownUnitName, to_unit_type

PairingMode = Literal["trailing", "tight"]


def replay_snapshot_pairing(n_frames: int, n_envelopes: int) -> Optional[PairingMode]:
    """
    Return how PHP lines pair with ``p:`` envelopes, or ``None`` if unsupported.

    Both supported modes use the same step loop: after applying envelope ``i``,
    compare the engine to ``frame[i+1]`` **when** ``i + 1 < n_frames`` (tight mode
    skips only the comparison after the last envelope).
    """
    if n_frames <= 0 or n_envelopes < 0:
        return None
    if n_frames == n_envelopes + 1:
        return "trailing"
    if n_frames == n_envelopes:
        return "tight"
    return None


def frames_envelopes_aligned(n_frames: int, n_envelopes: int) -> bool:
    """True if ``replay_snapshot_pairing`` accepts this pair (trailing or tight)."""
    return replay_snapshot_pairing(n_frames, n_envelopes) is not None


def php_internal_from_snapshot_hit_points(php_raw: Any, engine_hp: int) -> int:
    """Map PHP ``hit_points`` to internal HP (1–100) for comparator vs ``Unit.hp``.

    Site exports are normally ``internal / 10`` as a float (``2.0`` = 20 HP). A
    **Global League / tight-zip** class stores an effectively **20× compressed**
    value: ``0.1``/``0.2``/… where ``round(php*10)`` is a bogus 1–2 but
    ``round(php*200)`` matches the **engine** (ground truth in audit). When the
    ×10 reading is far from the engine but ×200 is within 1 internal HP, trust
    the ×200 path (else keep ×10).

    Some exports round **aggressively** (e.g. ``0.1`` for true ``0.15`` =
    internal 30): ×200 reads 20, ×300 reads 30. When ``f`` is small and the
    ×10 reading is ``≤ 1``, we pick among ×10 / ×200 / ×300 candidates the
    value **closest** to ``engine_hp`` (tie-break: prefer ×200, then ×300, then
    ×10) so ``desync_audit`` SM rows do not false-positive on scale alone.

    This does **not** change engine or oracle sim — only snapshot diff hygiene.
    """
    try:
        f = float(php_raw)
    except (TypeError, ValueError):
        return 0
    a = int(round(f * 10.0))
    b = int(round(f * 200.0))
    c300 = int(round(f * 300.0))
    eng = int(engine_hp)
    if abs(b - eng) <= 1 and abs(b - eng) < abs(a - eng):
        return max(0, min(100, b))
    # Lossy GL floats: ``a <= 1`` avoids pulling in ambiguous 0.15-scale rows
    # where ``a == 2`` (still standard ``internal/10`` territory).
    if f > 0 and f < 0.25 and a <= 1:
        c_clamped = max(0, min(100, c300))
        best: tuple[int, int] = (10**9, 99)
        chosen = a
        for val, tie_pri in (
            (max(0, min(100, a)), 3),
            (max(0, min(100, b)), 1),
            (c_clamped, 2),
        ):
            dist = abs(val - eng)
            key = (dist, tie_pri)
            if key < best:
                best = key
                chosen = val
        return chosen
    return max(0, min(100, a))


def _php_unit_bars(
    u: dict[str, Any], *, engine_internal_hp: Optional[int] = None
) -> int:
    """
    AWBW snapshot ``hit_points`` is the **internal HP / 10** as a float
    (e.g. ``6.3`` = 63 internal HP). The displayed bar is the **ceiling** of
    that value, matching ``engine.unit.Unit.display_hp`` (``(hp + 9) // 10``).

    Using ``round`` here produced spurious bar mismatches against the engine
    whenever PHP stored a non-integer ``hit_points`` whose rounded value
    differed from its ceiling (e.g. PHP ``6.3`` → ``round`` 6 vs engine ceil
    7). Both sides should use ceiling so only true internal-HP drift surfaces.

    When ``engine_internal_hp`` is set, ``php_internal_from_snapshot_hit_points``
    rewrites lossy *200-scale* zip rows so bars match the engine where appropriate.
    """
    hp = u.get("hit_points")
    if hp is None:
        return 0
    if engine_internal_hp is not None:
        intern = php_internal_from_snapshot_hit_points(hp, engine_internal_hp)
        return max(0, min(10, (intern + 9) // 10))
    v = int(math.ceil(float(hp)))
    return max(0, min(10, v))


def compare_funds(
    php_frame: dict[str, Any],
    state: GameState,
    awbw_to_engine: dict[int, int],
) -> list[str]:
    out: list[str] = []
    players = php_frame.get("players") or {}
    for _k, pl in players.items():
        if not isinstance(pl, dict):
            continue
        pid = int(pl["id"])
        eng = awbw_to_engine[pid]
        php_f = int(pl.get("funds", 0) or 0)
        eng_f = int(state.funds[eng])
        if php_f != eng_f:
            out.append(f"P{eng} funds engine={eng_f} php_snapshot={php_f} (awbw_players_id={pid})")
    return out


def compare_units(
    php_frame: dict[str, Any],
    state: GameState,
    awbw_to_engine: dict[int, int],
) -> list[str]:
    """
    Match alive units by **tile + owner**, not by numeric ``units_id``: AWBW stores
    database ids (e.g. ``191637002``) while the engine allocates ``unit_id`` from a
    local counter starting at 1 for ``make_initial_state``.
    """
    out: list[str] = []

    # PHP: (engine_seat, row, col) -> snapshot row.
    # AWBW exports loaded cargo with the SAME (x, y) as their carrier and
    # marks them ``carried: "Y"`` — the engine stores cargo inside
    # ``Unit.loaded_units`` (not on the tile). Filtering these here is what
    # makes "transport + loaded passenger at the same tile" stop registering
    # as a `php duplicate unit` / `unit tile set mismatch` (root cause of the
    # 1619695 APC-vs-Infantry false positive at (19, 2)).
    php_by_tile: dict[tuple[int, int, int], dict[str, Any]] = {}
    for _k, u in (php_frame.get("units") or {}).items():
        if not isinstance(u, dict):
            continue
        if str(u.get("carried", "N")).upper() == "Y":
            continue
        col, row = int(u["x"]), int(u["y"])
        pid = int(u["players_id"])
        eng_seat = awbw_to_engine[pid]
        key = (eng_seat, row, col)
        if key in php_by_tile:
            out.append(f"php duplicate unit at P{eng_seat} (row={row},col={col})")
        php_by_tile[key] = u

    eng_by_tile: dict[tuple[int, int, int], Any] = {}
    for seat in (0, 1):
        for u in state.units[seat]:
            if u.is_alive:
                r, c = u.pos
                key = (seat, r, c)
                if key in eng_by_tile:
                    out.append(f"engine duplicate unit at P{seat} {u.pos}")
                eng_by_tile[key] = u

    if set(php_by_tile) != set(eng_by_tile):
        only_php = set(php_by_tile) - set(eng_by_tile)
        only_eng = set(eng_by_tile) - set(php_by_tile)
        if only_php or only_eng:
            out.append(
                f"unit tile set mismatch only_in_php={sorted(only_php)[:16]}"
                f"{'…' if len(only_php) > 16 else ''} only_in_engine={sorted(only_eng)[:16]}"
                f"{'…' if len(only_eng) > 16 else ''}"
            )

    for key in sorted(set(php_by_tile) & set(eng_by_tile)):
        pu, eu = php_by_tile[key], eng_by_tile[key]
        php_name = str(pu.get("name", "")).strip()
        eng_name = UNIT_STATS[eu.unit_type].name
        if php_name and eng_name != php_name:
            # Phase 11Z: route through ``engine.unit_naming``. Both
            # spellings must resolve to the same UnitType for the type
            # comparison to be considered cosmetic. Anything that
            # fails resolution falls through to a literal string
            # mismatch (preserves legacy diagnostic output).
            try:
                php_ut = to_unit_type(php_name)
            except UnknownUnitName:
                php_ut = None
            if php_ut != eu.unit_type:
                out.append(f"at {key} type engine={eng_name!r} php={php_name!r}")
        php_bars = _php_unit_bars(pu, engine_internal_hp=int(eu.hp))
        eng_bars = eu.display_hp
        if php_bars != eng_bars:
            out.append(
                f"at {key} hp_bars engine={eng_bars} (hp={eu.hp}) php_bars={php_bars} "
                f"php_id={pu.get('id')}"
            )
    return out


def compare_snapshot_to_engine(
    php_frame: dict[str, Any],
    state: GameState,
    awbw_to_engine: dict[int, int],
    *,
    check_funds: bool = True,
    check_units: bool = True,
    check_properties: bool = True,
    check_co_states: bool = True,
    check_weather: bool = True,
    check_turn: bool = True,
) -> list[str]:
    """Return a list of human-readable mismatch lines (empty => match on checked axes)."""
    out: list[str] = []
    if check_funds:
        out.extend(compare_funds(php_frame, state, awbw_to_engine))
    if check_units:
        out.extend(compare_units(php_frame, state, awbw_to_engine))
    if check_properties:
        out.extend(compare_properties(php_frame, state, awbw_to_engine))
    if check_co_states:
        out.extend(compare_co_states(php_frame, state, awbw_to_engine))
    if check_weather:
        out.extend(compare_weather(php_frame, state))
    if check_turn:
        out.extend(compare_turn(php_frame, state))
    return out


def compare_properties(
    php_frame: dict[str, Any],
    state: GameState,
    awbw_to_engine: dict[int, int],
) -> list[str]:
    """Compare property ownership and capture points.

    PHP buildings are keyed by database id; we match by (row, col) using
    the engine ``PropertyState`` grid position.  PHP ``capture`` is 0–99
    (99 = neutral/no owner, 20 = fully owned by the player whose
    ``countries_id`` matches).  Engine ``capture_points`` is 0–20
    (20 = fully owned).

    Neutral PHP buildings (``capture == 99``) are skipped — engine
    ``owner = None`` is correct.
    """
    out: list[str] = []
    php_buildings = php_frame.get("buildings") or {}
    # Build a map from (row, col) -> PHP building for matching
    php_by_pos: dict[tuple[int, int], dict[str, Any]] = {}
    for _k, b in php_buildings.items():
        if not isinstance(b, dict):
            continue
        try:
            r, c = int(b["y"]), int(b["x"])
        except (KeyError, ValueError):
            continue
        php_by_pos[(r, c)] = b

    for prop in state.properties:
        key = (prop.row, prop.col)
        pb = php_by_pos.get(key)
        if pb is None:
            out.append(f"property at ({prop.row},{prop.col}) tid={prop.terrain_id} not in PHP snapshot")
            continue
        php_capture = int(pb.get("capture", 0) or 0)
        php_last_capture = int(pb.get("last_capture", 20) or 20)
        # PHP capture=99 means neutral; skip ownership check
        if php_capture == 99:
            if prop.owner is not None:
                out.append(
                    f"property at ({prop.row},{prop.col}) tid={prop.terrain_id} "
                    f"engine_owner={prop.owner} php=neutral(capture=99)"
                )
            continue
        # Determine expected engine owner from PHP capture state
        # When capture < 20, the building is being captured by someone.
        # We use last_capture (20 = owned by the player who placed it) to determine owner.
        # This is approximate; AWBW's capture field doesn't directly encode owner.
        # Use the country_to_player mapping from the map data.
        expected_owner: Optional[int] = None
        if php_capture >= 20:
            # Fully owned — find which engine seat owns this country
            cid = pb.get("countries_id")
            if cid is not None:
                try:
                    cid_int = int(cid)
                    ctp = state.map_data.country_to_player
                    if ctp:
                        expected_owner = ctp.get(cid_int)
                except (ValueError, TypeError):
                    pass
        if expected_owner is not None and prop.owner != expected_owner:
            out.append(
                f"property at ({prop.row},{prop.col}) tid={prop.terrain_id} "
                f"engine_owner={prop.owner} php_expected={expected_owner}"
            )
        # Compare capture points
        # PHP capture is 0-20 (20 = full, matching engine scale).
        # Some older AWBW versions may use 0-99; we handle both:
        # If php_capture > 20, assume 0-99 scale and convert; otherwise use as-is.
        eng_cp = prop.capture_points
        if php_capture > 20:
            # Scale from 0-99 to 0-20
            php_cp_scaled = max(0, min(20, (php_capture * 20 + 49) // 99))
        else:
            # Already in 0-20 scale
            php_cp_scaled = php_capture
        # Only report divergence if difference is significant (> 2 to absorb minor drift)
        if eng_cp != php_cp_scaled and abs(eng_cp - php_cp_scaled) > 2:
            out.append(
                f"property at ({prop.row},{prop.col}) capture_points "
                f"engine={eng_cp} php_scaled={php_cp_scaled} (php_raw={php_capture})"
            )
    return out


def _php_int_optional(val, default=0):
    """Convert a PHP value to int, treating 'N' and None as the default."""
    if val is None or val == 'N':
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def compare_co_states(
    php_frame: dict[str, Any],
    state: GameState,
    awbw_to_engine: dict[int, int],
) -> list[str]:
    """Compare CO power meter and activation state.

    PHP ``players[]`` carries ``co_power`` (current charge, in units of 1000 per star),
    ``co_max_power`` (COP threshold * 1000), ``co_max_spower`` (SCOP threshold * 1000),
    and ``co_power_on`` (1 if a power is active this turn).

    Engine ``COState`` carries ``cop_stars``, ``scop_stars`` (threshold stars),
    ``cop_active``, ``scop_active``, and ``power_bar`` (current charge in 0..max).

    Comparison logic:
    - PHP ``co_power // 1000`` gives stars charged (2500 → 2 stars)
    - Engine ``power_bar // 1000`` gives stars charged (3000 → 3 stars)
    - We compare these normalized values, allowing a tolerance of 1 star
      to absorb minor timing differences (end-of-turn vs start-of-turn snapshots).
    """
    out: list[str] = []
    players = php_frame.get("players") or {}
    for _k, pl in players.items():
        if not isinstance(pl, dict):
            continue
        try:
            pid = int(pl["id"])
        except (KeyError, TypeError, ValueError):
            continue
        eng = awbw_to_engine.get(pid)
        if eng is None or eng < 0 or eng >= len(state.co_states):
            continue
        co_state = state.co_states[eng]

        # Compare power activation — 'N' means fogged, skip comparison
        php_power_on = _php_int_optional(pl.get("co_power_on"), 0)
        engine_power_active = 1 if (co_state.cop_active or co_state.scop_active) else 0
        if php_power_on != engine_power_active:
            power_type = "COP" if co_state.cop_active else ("SCOP" if co_state.scop_active else "none")
            out.append(
                f"P{eng} power_active engine={engine_power_active}({power_type}) php={php_power_on}"
            )

        # Compare charge meter (only when no power is active)
        if not co_state.cop_active and not co_state.scop_active:
            php_charge = _php_int_optional(pl.get("co_power"), 0)
            if php_charge > 0:
                # Both values are in units of 1000 per star; convert to integer stars
                php_stars = php_charge // 1000
                eng_stars = co_state.power_bar // 1000
                # Allow 1-star tolerance (timing: snapshot at end-of-turn vs start-of-turn)
                if abs(php_stars - eng_stars) > 1:
                    out.append(
                        f"P{eng} charge_mismatch: "
                        f"php={php_stars} stars (raw={php_charge}) "
                        f"engine={eng_stars} stars (power_bar={co_state.power_bar})"
                    )
    return out


def compare_weather(
    php_frame: dict[str, Any],
    state: GameState,
) -> list[str]:
    """Compare weather state.

    PHP: ``weather_type`` / ``weather_code`` (numeric code or string).
    Engine: ``state.weather`` is "clear", "rain", or "snow".
    """
    out: list[str] = []
    php_weather = php_frame.get("weather_type") or php_frame.get("weather_code")
    if php_weather is None:
        return out
    # Map PHP weather codes to engine strings
    # PHP: 1=clear, 2=rain, 3=snow (approximate based on AWBW source)
    weather_map = {1: "clear", 2: "rain", 3: "snow", "1": "clear", "2": "rain", "3": "snow"}
    php_str = weather_map.get(php_weather)
    if php_str is None:
        php_str = str(php_weather).strip().lower()
    eng_weather = state.weather
    if php_str and eng_weather and php_str != eng_weather:
        out.append(f"weather engine={eng_weather} php={php_str} (raw={php_weather})")
    return out


def compare_turn(
    php_frame: dict[str, Any],
    state: GameState,
) -> list[str]:
    """Compare turn/day number.

    PHP: ``day`` field (1-indexed day number).
    Engine: ``state.turn`` (1-indexed turn number, increments after P1 ends).
    """
    out: list[str] = []
    php_day = php_frame.get("day")
    if php_day is None:
        return out
    try:
        php_day_int = int(php_day)
    except (TypeError, ValueError):
        return out
    # Engine turn should match PHP day (both are 1-indexed day numbers)
    if php_day_int != state.turn:
        out.append(f"turn engine={state.turn} php_day={php_day_int}")
    return out
