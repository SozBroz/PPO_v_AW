"""Phase 11Z — whack-a-mole stopper.

Walk every ``replays/amarriner_gl/*.zip`` and assert every unit-name
string in both the snapshot stream (PHP-serialized ``awbwUnit.name``)
and the action stream (JSON ``units_name``) resolves via
``engine.unit_naming.is_known_alias``.

If a future replay introduces a new spelling, this test fails before
the audit corpus does — and the operator instructions in
``docs/oracle_exception_audit/phase11z_unit_naming_canon_audit.md`` §8
point at the one file to edit (``engine/unit_naming.py``).
"""
from __future__ import annotations

import gzip
import re
import sys
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.unit_naming import is_known_alias  # noqa: E402

REPLAYS = ROOT / "replays" / "amarriner_gl"

# Snapshot stream: ``s:4:"name";s:N:"VALUE"`` inside an ``O:8:"awbwUnit"`` object.
_AWBWUNIT_OPEN_RE = re.compile(r'O:8:"awbwUnit":\d+:\{')
_PHP_NAME_FIELD_RE = re.compile(r's:4:"name";s:(\d+):"')
# Action stream: ``"units_name":"VALUE"``.
_JSON_UNITS_NAME_RE = re.compile(r'"units_name"\s*:\s*"([^"]*)"')


def _scan_snapshot(txt: str) -> set[str]:
    out: set[str] = set()
    for m in _AWBWUNIT_OPEN_RE.finditer(txt):
        window = txt[m.end() : m.end() + 4096]
        nm = _PHP_NAME_FIELD_RE.search(window)
        if not nm:
            continue
        n = int(nm.group(1))
        start = nm.end()
        out.add(window[start : start + n])
    return out


def _scan_action(txt: str) -> set[str]:
    return {m.group(1) for m in _JSON_UNITS_NAME_RE.finditer(txt)}


def _gather_zip(zip_path: Path) -> tuple[set[str], set[str]]:
    snap: set[str] = set()
    act: set[str] = set()
    with zipfile.ZipFile(zip_path) as zf:
        for nm in zf.namelist():
            raw = zf.read(nm)
            try:
                txt = gzip.decompress(raw).decode("utf-8", "replace")
            except OSError:
                txt = raw.decode("utf-8", "replace")
            if nm.startswith("a"):
                act |= _scan_action(txt)
            else:
                snap |= _scan_snapshot(txt)
    return snap, act


_ZIPS = sorted(REPLAYS.glob("*.zip"))


@unittest.skipUnless(_ZIPS, "no replays/amarriner_gl/*.zip available")
class TestUnitNamingCorpusRegression(unittest.TestCase):
    """Empirical sanity over the entire local GL replay pool."""

    def test_every_observed_unit_name_resolves(self) -> None:
        snap_names: set[str] = set()
        act_names: set[str] = set()
        for z in _ZIPS:
            try:
                s, a = _gather_zip(z)
            except (zipfile.BadZipFile, OSError) as exc:
                self.fail(f"could not read {z.name}: {exc}")
            snap_names |= s
            act_names |= a

        all_names = sorted(snap_names | act_names)
        self.assertGreater(
            len(all_names),
            0,
            msg="no unit-name strings extracted — scanner likely broken",
        )
        unknown = sorted(n for n in all_names if not is_known_alias(n))
        self.assertEqual(
            unknown,
            [],
            msg=(
                f"unknown unit-name strings observed in corpus "
                f"(add them to engine/unit_naming.py): {unknown!r}"
            ),
        )

    def test_eagle_gl_bleeders_are_recognized(self) -> None:
        """Cite the specific failures that motivated this slice."""
        for spelling in ("Missile", "Missiles", "Sub", "Submarine",
                         "Md.Tank", "Md. Tank", "Mega Tank", "Megatank",
                         "Neotank", "Neo Tank", "Rocket", "Rockets",
                         "Anti-Air", "B-Copter", "T-Copter"):
            self.assertTrue(
                is_known_alias(spelling),
                msg=f"canonical resolver lost coverage for {spelling!r}",
            )


if __name__ == "__main__":
    unittest.main()
