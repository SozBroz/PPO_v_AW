"""JSONL for train reconfig (soft) and hard-restart wall-time instrumentation.

Written by ``fleet_orchestrator`` and :class:`rl.self_play.SelfPlayTrainer`` under
``<shared_root>/logs/train_reconfig.jsonl``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def append_train_reconfig_line(shared_root: Path, record: dict[str, Any]) -> None:
    log = Path(shared_root).resolve() / "logs" / "train_reconfig.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    if "ts" not in record:
        record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
