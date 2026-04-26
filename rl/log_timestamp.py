"""Wall timestamps for logs in US Eastern (IANA ``America/New_York`` = EST/EDT)."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

LOG_TZ = ZoneInfo("America/New_York")


def log_timestamp_iso(ts: float) -> str:
    """Epoch seconds -> ISO-8601 with milliseconds and explicit offset."""
    return datetime.fromtimestamp(ts, tz=LOG_TZ).isoformat(timespec="milliseconds")


def log_now_iso() -> str:
    """Current instant as ISO-8601 with milliseconds and explicit offset."""
    return datetime.now(tz=LOG_TZ).isoformat(timespec="milliseconds")


def log_now_wall() -> str:
    """``YYYY-MM-DD HH:MM:SS`` in LOG_TZ (no suffix; wall clock in that zone)."""
    return datetime.now(tz=LOG_TZ).strftime("%Y-%m-%d %H:%M:%S")
