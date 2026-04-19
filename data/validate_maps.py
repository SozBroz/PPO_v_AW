"""
Map CSV validation script.

Scans every CSV in data/maps/ and reports:
  - All distinct terrain IDs found (flags any unknown to terrain.py)
  - Distinct country IDs found per map
  - Exactly-2-countries assertion (warns on violations)
  - Total properties per country per map
  - HQ count per country (warns if != 1 for standard competitive maps)

Usage:
    python data/validate_maps.py [--verbose] [maps_dir]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.terrain import TERRAIN_TABLE, get_terrain, get_country, is_property


def validate_all(maps_dir: Path, *, verbose: bool = False) -> int:
    """
    Validate all CSVs in maps_dir.  Returns the number of violations found.
    """
    csvs = sorted(maps_dir.glob("*.csv"))
    if not csvs:
        print(f"[warn] No CSV files found in {maps_dir}")
        return 0

    all_unknown_ids: set[int] = set()
    total_violations = 0

    for csv_path in csvs:
        map_id = csv_path.stem
        rows = []
        for line in csv_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append([int(x) for x in line.split(",")])

        if not rows:
            print(f"[warn] {map_id}: empty CSV")
            continue

        all_ids: set[int] = set()
        for row in rows:
            all_ids.update(row)

        # ---- Unknown IDs ----
        unknown = sorted(tid for tid in all_ids if tid not in TERRAIN_TABLE)
        all_unknown_ids.update(unknown)

        # ---- Property scan ----
        country_props: dict[int | None, list[str]] = {}  # country_id → list of prop types
        country_hqs:   dict[int | None, int]       = {}

        for r, row in enumerate(rows):
            for c, tid in enumerate(row):
                info = get_terrain(tid)
                if not info.is_property:
                    continue
                cid = info.country_id
                if cid not in country_props:
                    country_props[cid] = []
                ptype = (
                    "HQ" if info.is_hq else
                    "Lab" if info.is_lab else
                    "Tower" if info.is_comm_tower else
                    "Airport" if info.is_airport else
                    "Base" if info.is_base else
                    "Port" if info.is_port else
                    "City"
                )
                country_props[cid].append(ptype)
                if info.is_hq:
                    country_hqs[cid] = country_hqs.get(cid, 0) + 1

        # Only non-neutral countries count toward the "2-country" requirement
        owned_countries = [cid for cid in country_props if cid is not None]

        violations: list[str] = []

        if unknown:
            violations.append(f"UNKNOWN IDs: {unknown}")
            total_violations += 1

        if len(owned_countries) != 2:
            violations.append(
                f"COUNTRY COUNT: expected 2 owned countries, found {len(owned_countries)}: {owned_countries}"
            )
            total_violations += 1

        for cid, hq_count in country_hqs.items():
            if hq_count > 1:
                # Multi-HQ maps are legal in AWBW (capture ALL opponent HQs to win).
                # Flag as info, not a hard violation.
                violations.append(
                    f"INFO/multi-HQ: country {cid} has {hq_count} HQs (valid multi-HQ map)"
                )

        # ---- Report ----
        if verbose or violations:
            print(f"\n{'='*60}")
            print(f"Map {map_id}")
            print(f"  Terrain IDs: {sorted(all_ids)}")
            if unknown:
                print(f"  !! Unknown IDs: {unknown}")

            neutral_props = country_props.get(None, [])
            if neutral_props:
                from collections import Counter
                nc = Counter(neutral_props)
                print(f"  Neutral props: {dict(nc)}")

            for cid in sorted(owned_countries):
                from collections import Counter
                pc = Counter(country_props[cid])
                print(f"  Country {cid:2d} props: {dict(pc)}")

            for v in violations:
                print(f"  !! {v}")
        else:
            # Terse one-liner
            countries_str = ", ".join(str(c) for c in sorted(owned_countries))
            total_owned = sum(len(v) for k, v in country_props.items() if k is not None)
            neutral_count = len(country_props.get(None, []))
            print(
                f"  OK  {map_id:10s}  countries=[{countries_str}]  "
                f"owned_props={total_owned}  neutral_props={neutral_count}"
            )

    # ---- Global summary ----
    print(f"\n{'='*60}")
    print(f"Validated {len(csvs)} maps.  Total violations: {total_violations}")
    if all_unknown_ids:
        print(f"Unknown terrain IDs across ALL maps: {sorted(all_unknown_ids)}")
    else:
        print("No unknown terrain IDs found. [OK]")

    return total_violations


def main():
    parser = argparse.ArgumentParser(description="Validate AWBW map CSV terrain data.")
    parser.add_argument(
        "maps_dir", nargs="?",
        default=str(Path(__file__).resolve().parent / "maps"),
        help="Directory containing map CSV files (default: data/maps/)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full detail for every map, not just violations.",
    )
    args = parser.parse_args()

    maps_dir = Path(args.maps_dir)
    if not maps_dir.is_dir():
        print(f"[error] Directory not found: {maps_dir}", file=sys.stderr)
        sys.exit(1)

    violations = validate_all(maps_dir, verbose=args.verbose)
    sys.exit(1 if violations else 0)


if __name__ == "__main__":
    main()
