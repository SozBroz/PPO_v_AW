"""Scan Colin replay zips for Power / Colin / COP (coPower Y)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.oracle_zip_replay import parse_p_envelopes_from_zip  # noqa: E402

GIDS = [
    1636107,
    1558571,
    1637153,
    1638360,
    1637096,
    1638136,
    1636411,
    1620117,
    1628024,
    1629555,
    1637705,
    1358720,
    1636108,
    1619141,
    1637200,
]


def scan_zip(zpath: Path) -> tuple[int, int, list[str]]:
    envs = parse_p_envelopes_from_zip(zpath)
    if not envs:
        return 0, 0, ["(no p: envelopes / RV1)"]
    lines: list[str] = []
    n = 0
    n_cop = 0
    for ei, (_pid, _day, actions) in enumerate(envs):
        for ai, a in enumerate(actions):
            if a.get("action") != "Power":
                continue
            if a.get("coName") != "Colin":
                continue
            cp = a.get("coPower")
            pn = a.get("powerName", "")
            n += 1
            if cp == "Y":
                n_cop += 1
            lines.append(f"  env={ei} sub={ai} coPower={cp!r} powerName={pn!r}")
    return n, n_cop, lines


def main() -> None:
    base = ROOT / "replays" / "amarriner_gl"
    zips_with_cop = 0
    for gid in GIDS:
        zpath = base / f"{gid}.zip"
        if not zpath.is_file():
            print(f"{gid} MISSING")
            continue
        cnt, n_cop, detail = scan_zip(zpath)
        print(f"\n{gid}.zip Colin Power rows: {cnt} (COP coPower=Y: {n_cop})")
        for ln in detail[:25]:
            print(ln)
        if n_cop:
            zips_with_cop += 1
        elif cnt:
            print("  (SCOP or other coPower only)")
    print(f"\nSummary: zips with at least one Colin COP (coPower Y): {zips_with_cop}")


if __name__ == "__main__":
    main()
