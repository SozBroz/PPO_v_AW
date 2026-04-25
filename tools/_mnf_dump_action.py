"""Dump the raw action JSON at a given env/ai for inspection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tools.oracle_zip_replay import parse_p_envelopes_from_zip


def main() -> None:
    gid = int(sys.argv[1])
    target_env = int(sys.argv[2])
    target_ai = int(sys.argv[3])
    envs = parse_p_envelopes_from_zip(Path(f"replays/amarriner_gl/{gid}.zip"))
    pid, day, actions = envs[target_env]
    obj = actions[target_ai]
    print(f"env={target_env} ai={target_ai} pid={pid} day={day}")
    print(json.dumps(obj, indent=2, default=str))


if __name__ == "__main__":
    main()
