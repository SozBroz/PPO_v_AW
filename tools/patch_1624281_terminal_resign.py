"""
Close GL 1624281 in the action gzip: replace the final End with Resign (P0).

The amarriner mirror replay ends after P0's day-18 half-turn (35 p: envelopes, max
day 18) while PHP still shows active play (next mover P1). The engine never reaches
done/winner. For tooling that expects a terminal replay, P0 resigning matches a P1
victory when the live match ended with Bigou (P1) winning.
"""
from __future__ import annotations

import sys

import gzip
import io
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.oracle_zip_replay import extract_json_action_strings_from_envelope_line
ZIP_PATH = ROOT / "replays" / "amarriner_gl" / "1624281.zip"
MEMBER = "a1624281"
NEW_TAIL = '{"action":"Resign"}'


def main() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(ZIP_PATH, "r") as zin:
        raw = zin.read(MEMBER)
        txt = gzip.decompress(raw).decode("utf-8")
        lines = txt.split("\n")
        non_empty = [i for i, ln in enumerate(lines) if ln.strip()]
        if not non_empty:
            raise SystemExit("empty action stream")
        li = non_empty[-1]
        line = lines[li]
        blobs = extract_json_action_strings_from_envelope_line(line)
        if not blobs:
            raise SystemExit("no JSON blobs in last line")
        old = blobs[-1]
        try:
            last_action = json.loads(old).get("action")
        except json.JSONDecodeError as e:
            raise SystemExit(f"bad last JSON: {e}") from e
        if last_action == "Resign":
            print("already terminal (Resign); no change")
            return
        if last_action != "End":
            raise SystemExit(f"expected last action End, got {last_action!r}")

        j: int | None = None
        n_old: int | None = None
        body_start: int | None = None
        for m in re.finditer(r"s:(\d+):\"", line):
            n = int(m.group(1))
            bs = m.end()
            body = line[bs : bs + n]
            if not body.startswith('{"action":'):
                continue
            if body == old:
                j = m.start()
                n_old = n
                body_start = bs
        if j is None or n_old is None or body_start is None:
            raise SystemExit("could not locate s:len:\" wrapper for last action blob")
        k = line.find(":", j + 2)
        if line[k + 1] != '"' or line[k + 2 : k + 2 + n_old] != old:
            raise SystemExit("s: length prefix does not match last blob")
        old_end = k + 2 + n_old + 2
        new_seg = f's:{len(NEW_TAIL)}:"' + NEW_TAIL + '";'
        lines[li] = line[:j] + new_seg + line[old_end:]

        new_txt = "\n".join(lines)
        # Preserve trailing newline if original had it
        if txt.endswith("\n") and not new_txt.endswith("\n"):
            new_txt += "\n"
        out_gz = gzip.compress(new_txt.encode("utf-8"), compresslevel=6)

        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = out_gz if item.filename == MEMBER else zin.read(item.filename)
                zout.writestr(item, data)

    ZIP_PATH.write_bytes(buf.getvalue())
    print(f"patched {ZIP_PATH} member {MEMBER}: last action End -> Resign ({len(NEW_TAIL)} chars)")


if __name__ == "__main__":
    main()
