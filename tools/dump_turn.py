"""Quick dumper for a single turn's PHP blob from an AWBW replay zip."""
import gzip
import sys
import zipfile
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: dump_turn.py <zip> [turn_index] [chars]")
        return 2
    path = Path(sys.argv[1])
    turn = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
    chars = int(sys.argv[3]) if len(sys.argv) >= 4 else 6000

    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            blob = zf.read(name)
            if blob[:2] == b"\x1f\x8b":
                text = gzip.decompress(blob).decode("utf-8", errors="replace")
            else:
                text = blob.decode("utf-8", errors="replace")
            lines = text.split("\n")
            print(f"-- {name}: {len(lines)} lines, first char '{text[:1]}'")
            if turn < len(lines):
                print(lines[turn][:chars])
                print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
