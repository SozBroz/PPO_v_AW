# Phase 11Z — Unit-Name Canon Audit (whack-a-mole closeout)

Status: shipping with this slice. Replaces the architecture of
`phase11j_state_mismatch_name_normalize.md` (kept as a record of the
intermediate fix).

## 0. Why this exists

`tests/test_desync_eagle_gl_corpus.py` failed 4 subtests on `games_id`
1631554 / 1634482 with `unknown AWBW unit name 'Missile'`. Site JSON
emits singular `Missile`; the oracle resolver
(`tools/oracle_zip_replay._name_to_unit_type`) only had plural `Missiles`
in its forward dict. Meanwhile the comparator's
`_canonicalize_unit_type_name` (a different code path) already folded
`missiles → missile`. Four parallel name maps, no roundtrip test, no
shared resolver. Each fix has been one alias in one dict; the next new
spelling reopens it.

This audit catalogs every surface, then ships a single canonical
resolver in `engine/unit_naming.py` and routes every consumer through
it. The whack-a-mole stopper is a corpus regression test that walks
every `replays/amarriner_gl/*.zip` and asserts every emitted
`units_name` / `awbwUnit.name` is recognized.

## 1. Engine canonical (`engine/unit.py`)

`UnitType` is the integer enum (0–26). The string field is
`UNIT_STATS[ut].name`. Engine tables use enum identity; the string is
human-only.

| `UnitType.NAME` | `stats.name` |
|---|---|
| `INFANTRY`  | `Infantry` |
| `MECH`      | `Mech` |
| `RECON`     | `Recon` |
| `TANK`      | `Tank` |
| `MED_TANK`  | `Medium Tank` |
| `NEO_TANK`  | `Neotank` |
| `MEGA_TANK` | `Megatank` |
| `APC`       | `APC` |
| `ARTILLERY` | `Artillery` |
| `ROCKET`    | `Rocket` |
| `ANTI_AIR`  | `Anti-Air` |
| `MISSILES`  | `Missiles` |
| `FIGHTER`   | `Fighter` |
| `BOMBER`    | `Bomber` |
| `STEALTH`   | `Stealth` |
| `B_COPTER`  | `B-Copter` |
| `T_COPTER`  | `T-Copter` |
| `BATTLESHIP`| `Battleship` |
| `CARRIER`   | `Carrier` |
| `SUBMARINE` | `Submarine` |
| `CRUISER`   | `Cruiser` |
| `LANDER`    | `Lander` |
| `GUNBOAT`   | `Gunboat` |
| `BLACK_BOAT`| `Black Boat` |
| `BLACK_BOMB`| `Black Bomb` |
| `PIPERUNNER`| `Piperunner` |
| `OOZIUM`    | `Oozium` |

Cite: `engine/unit.py:84-328`.

The ENUM identity (`UnitType.X.name`, e.g. `MED_TANK`) is itself a
recognized form throughout the codebase (e.g.
`tests/test_state_mismatch_name_canon.py` asserts `MED_TANK` folds
to `Medium Tank`).

## 2. AWBW PHP / damage.php / replay-zip writes (`tools/export_awbw_replay.py`)

`_AWBW_UNIT_NAMES` (lines 110-143). This is the dictionary the export
pipeline writes into both the snapshot stream (`s:4:"name"` field of
`O:8:"awbwUnit"`) and the action stream (`units_name` JSON field). The
file's own comment at lines 125-127 calls out why `Missiles` is plural
and `Sub` is abbreviated.

Divergences vs. engine `stats.name`:

| `UnitType` | engine `stats.name` | exporter PHP form |
|---|---|---|
| `MED_TANK`  | `Medium Tank`   | `Md. Tank`  |
| `NEO_TANK`  | `Neotank`       | `Neo Tank`  |
| `MEGA_TANK` | `Megatank`      | `Mega Tank` |
| `ROCKET`    | `Rocket`        | `Rockets`   |
| `MISSILES`  | `Missiles`      | `Missiles`  |
| `SUBMARINE` | `Submarine`     | `Sub`       |

Empirical sample from `replays/amarriner_gl/*.zip` (951 zips, scanned
via `.tmp/survey_unit_names.py`): both surfaces (snapshot
`awbwUnit.name` and action `units_name`) emit exactly 25 distinct unit
names — one per non-Oozium UnitType. **All 25 match the C# viewer's
`Units.json` keys**, not the exporter's form (see §3). The exporter
diverges on `Rockets`/`Rocket`, `Missiles`/`Missile`,
`Md. Tank`/`Md.Tank`. The PHP `damage.php` page (the row labels for
the 27×27 table) does still appear to use `Missiles`/`Rockets`
plural — that is what motivated the exporter's choice originally —
but the live `replay_download.php` payloads use the singular C#-
viewer form. We preserve the existing exporter table (and call it
`AWBW_PHP`) to avoid behavior drift; the alias table covers the live
forms.

## 3. C# replay viewer canonical (`AWBWApp.Replay-Player`, master `3ccbc60`)

Upstream repo: https://github.com/DeamonHunter/AWBW-Replay-Player
Default branch: `master`
Latest commit at audit time: `3ccbc6001501adc52488b1e106f7a01d6f750a3a`
(2025-12-30, "Desert Road Fixes.")

Authoritative file: `AWBWApp.Resources/Json/Units.json`
Raw URL:
https://raw.githubusercontent.com/DeamonHunter/AWBW-Replay-Player/master/AWBWApp.Resources/Json/Units.json

The C# viewer keys its in-memory storage by the JSON object key (which
equals each entry's `Name` field). Empirical extraction (object keys
in source order):

```
Infantry, Mech, Md.Tank, Tank, Recon, APC, Artillery, Rocket, Anti-Air,
Missile, Fighter, Bomber, B-Copter, T-Copter, Battleship, Cruiser,
Lander, Sub, Black Boat, Carrier, Stealth, Neotank, Piperunner,
Black Bomb, Mega Tank
```

Divergences vs. our exporter:

| UnitType | C# / AWBW site | exporter |
|---|---|---|
| `MED_TANK`  | `Md.Tank`  (no space) | `Md. Tank` (with space) |
| `ROCKET`    | `Rocket`   (singular) | `Rockets`   (plural)   |
| `MISSILES`  | `Missile`  (singular) | `Missiles`  (plural)   |

(Note: the upstream Units.json has no `Oozium` entry — AWBW core game
does not surface that unit; engine carries it for campaign/SCOP work.)

## 4. Oracle alias dicts pre-refactor

Four parallel maps, each with its own coverage matrix:

### 4.1 `tools/oracle_zip_replay.py::_name_to_unit_type` (lines 3317-3350)

Forward: aliases dict + iterate `_AWBW_UNIT_NAMES`. Recognized:

- engine PHP forms (via `_AWBW_UNIT_NAMES`)
- `Md.Tank`, `md.tank`, `MD.TANK` → `Md. Tank`
- `Neotank`, `NeoTank`, `neo tank`, `NEO TANK` → `Neo Tank`
- `Megatank`, `mega tank`, `MEGA TANK` → `Mega Tank`
- `Rocket`, `rocket`, `ROCKET` → `Rockets`
- `Anti Air`, `anti air` → `Anti-Air`
- `B Copter`, `b copter` → `B-Copter`
- `T Copter`, `t copter` → `T-Copter`

**Misses** (the bleeders): `Missile` (singular), `Sub`, `Submarine`
(when site emits the abbrev), arbitrary case variations beyond the
listed ones, `Md.Tank` vs `MdTank` vs `MD_TANK`.

### 4.2 `tools/desync_audit.py::_PHP_NAME_ALIASES` + `_canonicalize_unit_type_name` (lines 374-413)

Phase 11J normalizer. Strips space/punct, lowercases, then folds
`missiles → missile`, `mediumtank|mdtank|medtank → mediumtank`,
`sub|submarine → submarine`. Used **only** for state-mismatch
comparator equality (not for resolving names to UnitType).

**Coverage matrix relative to live data**: handles the cosmetic pairs
(empirical: 21× `Megatank`/`Mega Tank`, 9× `Missiles`/`Missile`, 1×
`Sub`/`Submarine`). Does **not** resolve names to `UnitType`, so it
cannot stop `_name_to_unit_type` from raising on a never-seen alias.

### 4.3 `tools/replay_snapshot_compare.py::aliases` (lines 209-212)

Local dict inside `compare_units`: only `Md.Tank → Medium Tank`,
`Md. Tank → Medium Tank`. **Misses everything else.**

### 4.4 `tools/fetch_predeployed_units.py::AWBW_NAME_TO_UNIT_TYPE` (lines 62-91)

Predeployed-units fetcher. Hand-rolled dict keyed by site PHP names
to engine enum *member name strings* (e.g. `"Sub" → "SUBMARINE"`).
Already includes `"Missile" → "MISSILES"` (caught it earlier than
the oracle resolver did) — but not `Mega Tank` vs `Megatank`,
`Md.Tank` vs `Md. Tank`, or any case variation.

### 4.5 `engine/action.py::_BAN_MAP` (lines 590-595)

Map-level `unit_bans` keys (small set: `Black Bomb`, `Stealth`,
`Piperunner`, `Oozium`). Drives `get_producible_units` filtering. This
list is fed by `MapData.unit_bans` which originates in the map JSON
(authored), not from site replays. Routing through canon is safe but
optional; we leave it as an explicit short list and add a runtime
sanity assertion that each key resolves via the canon (so a typo here
fails fast rather than silently letting a banned unit through).

## 5. Predeployed-units / damage_table

- `data/maps/<id>_units.json` (written by `tools/fetch_predeployed_units.py`)
  stores **engine enum member names** (`"unit_type": "INFANTRY"`).
  Reader is `engine/predeployed.py`. Not a name-string surface for
  this audit beyond the PHP→enum map already documented in §4.4.
- `data/damage_table.json` has a `unit_order` list of 27 short labels
  (`MedTank`, `NeoTank`, `MegaTank`, `BCopter`, `TCopter`, `AntiAir`,
  `BlackBoat`, `BlackBomb`, …). At runtime these are **not** used:
  `engine/combat.py::get_base_damage` indexes the `table` field by
  `int(UnitType)` directly. `unit_order` is a documentation field
  only. Left untouched. We do recognize these short labels in the
  canon as a fourth surface (`AWBW_DAMAGE_PHP`) so any future
  consumer of `unit_order` works through the resolver.

## 6. Failure-mode analysis (whack-a-mole history)

Prior incidents traceable to a missing alias in one of the dicts above:

| Incident | Year-month | Symptom | Patched in | Could a roundtrip test have caught it? |
|---|---|---|---|---|
| `Md.Tank` vs `Md. Tank` site spacing | early 2026 | comparator type-mismatch | `replay_snapshot_compare` aliases | yes |
| `Megatank` (engine) vs `Mega Tank` (PHP) | 2026-04 | 21× state_mismatch_units rows | Phase 11J `_canonicalize_unit_type_name` | yes |
| `Missiles` (engine) vs `Missile` (PHP) | 2026-04 | 9× state_mismatch_units rows | Phase 11J `_canonicalize_unit_type_name` (cosmetic only) | yes |
| `Sub` (PHP) vs `Submarine` (engine) | 2026-04 | 1× cosmetic line | Phase 11J fold | yes |
| `Missile` (singular) at oracle resolver | 2026-04-23 (this slice) | 4× hard-fail subtests in `test_desync_eagle_gl_corpus`, gid 1631554 / 1634482 | this slice (canon resolver) | **yes — corpus regression test** |
| `Neotank`/`Neo Tank` site variant | 2025 | `unknown AWBW unit name` | local alias add in `_name_to_unit_type` | yes |
| `Megatank` site variant at oracle | 2025 | `unknown AWBW unit name` | local alias add | yes |

Pattern: every single instance was a new spelling appearing in a real
replay payload. Every fix added one entry to one dict. Roundtrip
coverage was nil.

## 7. Architectural diagnosis & fix

**Today**: each surface has its own private alias dict. There is no
single "is this a known unit name?" predicate, no roundtrip test, and
the engine writes one form while the site emits a different one with
no enforced reconciliation.

**Fix shipped this slice** (`engine/unit_naming.py`):

1. One backing table `_TABLE: dict[UnitType, dict[UnitNameSurface,
   tuple[str, ...]]]`. First entry per (UT, surface) is the canonical
   render; rest are aliases.
2. `to_unit_type(name, surface=None)` resolves any alias → UnitType
   (raises `UnknownUnitName`).
3. `from_unit_type(ut, surface)` renders for a specific surface.
4. `is_known_alias(name)` predicate.
5. `all_known_aliases()` for tests.
6. Frozen via `MappingProxyType`.

Routed:
- `tools/oracle_zip_replay._name_to_unit_type` → calls `to_unit_type`,
  re-raises as `UnsupportedOracleAction` to preserve the oracle-error
  contract.
- `tools/desync_audit._canonicalize_unit_type_name` → `to_unit_type` →
  enum `.name`; falls back to legacy normalize for unresolvable
  strings (so legacy diff lines still print).
- `tools/export_awbw_replay._AWBW_UNIT_NAMES` → derived from
  `from_unit_type(ut, AWBW_PHP)` for every UT. Symbol kept as a
  module-level dict for backwards compatibility with importers.
- `tools/replay_snapshot_compare` `aliases` dict: replaced with
  canon-based equality.
- `tools/fetch_predeployed_units.AWBW_NAME_TO_UNIT_TYPE`: rebuilt
  from the canon (kept as an exported dict for back-compat with any
  out-of-tree caller).

**Whack-a-mole stopper**: `tests/test_unit_naming_corpus_regression.py`
walks every zip under `replays/amarriner_gl/`, scans both snapshot
and action streams, and asserts `is_known_alias(name)` for every
observed value. Any future replay introducing a new spelling fails
this test before it reaches the audit corpus.

## 8. Operator instructions — adding a new unit / new alias

1. Edit `engine/unit_naming.py` only. Add the new `(UnitType, surface)`
   entry to `_TABLE`, or extend an existing tuple to add an alias.
   The first entry of each tuple is the canonical render for that
   surface; the rest are accepted aliases for `to_unit_type`.
2. Run:
   ```
   python -m pytest -q tests/test_unit_naming_canon.py tests/test_unit_naming_corpus_regression.py
   ```
3. No other file should change. If a test fails because a downstream
   consumer assumed a specific spelling, fix that consumer (don't
   add a second alias dict). The corpus regression test is the
   stopper — once it goes green, every replay in the local pool
   is covered.

## 9. Out of scope (intentional)

- Behaviour change of the export pipeline (we keep the existing PHP
  forms `Missiles`, `Rockets`, `Md. Tank`, `Mega Tank`, `Neo Tank`,
  `Sub` even though the live AWBW site uses singular forms — changing
  what we emit changes downstream consumers and is a separate audit).
- `engine/action.py::_BAN_MAP` (small fixed set; wired from authored
  map JSON, not site payloads). Documented above; not refactored.
- `data/damage_table.json::unit_order` (documentation field, unused
  at runtime).
- MCTS / fleet orchestrator code paths (per task envelope).

## 10. Predicted future incidents this audit cannot prevent

- A genuinely new unit (e.g. an AWBW campaign unit) introduced upstream
  will need a new `UnitType` enum member **and** entries in
  `_TABLE` for every surface. The corpus regression test will fail
  loudly when its zip arrives, and the operator instructions point
  at the one file to edit.
- Site PHP changes its emitted name (e.g. renames `Md.Tank` →
  `MdTank`) — the corpus regression test will fail on the next zip
  ingest, again pointing at one file.
- A consumer that bypasses the canon and hardcodes a string literal
  for comparison (e.g. `if name == "Missiles": ...`). The audit
  refactor removes the known instances; new ones can be added via
  ripgrep scrubs in CI. (Not added in this slice — consider adding
  a `tests/test_no_hardcoded_unit_strings.py` lint test in a
  follow-up.)
