"""
Re-fetch Amarriner live-game folders before ``train.py`` starts (fleet orchestrator).

Runs ``tools/amarriner_export_my_games_replays.py`` for each ``--live-games-id`` on the
trainer argv, writing into ``--live-snapshot-dir`` (same layout as
``_resolve_live_snapshot_pkl_path`` in ``rl.self_play`` — nested
``<dir>/<games_id>/engine_snapshot.pkl``).

That script pulls ``game.php`` HTML, the full ``load_replay.php`` envelope stream, and
rebuilds ``engine_snapshot.pkl`` when ``meta.json`` + map data are available — i.e. it is
the **site-to-disk** step ``train.py`` needs for up-to-date live PPO snapshots.

Caveats (unchanged by this wrapper):

- Requires repo ``secrets.txt`` and network reachability to ``awbw.amarriner.com``.
- If the site truncates envelopes, stale tail states can still occur (see live-audit tools).

Disable with env ``AWBW_ORCH_SKIP_LIVE_RESYNC=1`` (debug / air-gapped hosts).
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

_LOG = logging.getLogger(__name__)

SKIP_ENV = "AWBW_ORCH_SKIP_LIVE_RESYNC"
DEFAULT_EXPORT_TIMEOUT_S = 600.0


def parse_train_cmd_live_ppo(cmd: list[str]) -> tuple[list[int], Path | None]:
    """
    Parse repeated ``--live-games-id`` and optional ``--live-snapshot-dir`` from a
    ``train.py`` argv (first python, then ``train.py``, then flags).
    """
    ids: list[int] = []
    snap: Path | None = None
    i = 0
    while i < len(cmd):
        part = cmd[i]
        if part == "--live-games-id" and i + 1 < len(cmd):
            try:
                ids.append(int(cmd[i + 1]))
            except ValueError:
                pass
            i += 2
            continue
        if part == "--live-snapshot-dir" and i + 1 < len(cmd):
            snap = Path(cmd[i + 1])
            i += 2
            continue
        i += 1
    return ids, snap


def resync_live_games_for_train_cmd(
    repo_root: Path,
    cmd: list[str],
    *,
    cwd: Path | None = None,
    sleep_s: float = 0.35,
    timeout_s: float = DEFAULT_EXPORT_TIMEOUT_S,
) -> bool:
    """
    If *cmd* enables live PPO, run the Amarriner export for those ids into the snapshot dir.

    Returns ``True`` if skipped (no live games / skip env / no secrets) or export exited 0.
    Returns ``False`` if export was attempted and failed (train may still start with older pkls).
    """
    if os.environ.get(SKIP_ENV, "").strip() == "1":
        _LOG.info("skip live resync (%s=1)", SKIP_ENV)
        return True

    secrets = Path(repo_root) / "secrets.txt"
    if not secrets.is_file():
        _LOG.info("skip live resync: missing %s", secrets)
        return True

    live_ids, snap_opt = parse_train_cmd_live_ppo(cmd)
    if not live_ids:
        return True
    live_ids = list(dict.fromkeys(live_ids))

    out_dir = snap_opt if snap_opt is not None else Path(repo_root) / ".tmp" / "awbw_live_snapshot"
    if not out_dir.is_absolute():
        base = cwd if cwd is not None else Path(repo_root)
        out_dir = (base / out_dir).resolve()

    export_py = Path(repo_root) / "tools" / "amarriner_export_my_games_replays.py"
    if not export_py.is_file():
        _LOG.warning("live resync: missing %s", export_py)
        return False

    argv: list[str] = [
        sys.executable,
        str(export_py),
        "--out",
        str(out_dir),
        "--sleep",
        str(float(sleep_s)),
    ]
    for gid in live_ids:
        argv.extend(["--games-id", str(int(gid))])

    _LOG.info(
        "live resync: exporting games_id=%s -> %s (timeout_s=%s)",
        live_ids,
        out_dir,
        timeout_s,
    )
    try:
        proc = subprocess.run(
            argv,
            cwd=str(repo_root),
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        _LOG.error("live resync: timeout after %ss for games_id=%s", timeout_s, live_ids)
        return False
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()
        if len(tail) > 4000:
            tail = tail[:4000] + "…"
        _LOG.error(
            "live resync failed rc=%s games_id=%s stderr/stdout tail:\n%s",
            proc.returncode,
            live_ids,
            tail,
        )
        return False
    _LOG.info("live resync ok games_id=%s", live_ids)
    return True
