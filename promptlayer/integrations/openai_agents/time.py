from __future__ import annotations

from datetime import datetime, timezone


def iso_to_unix_nano(timestamp: str | None) -> int | None:
    if not timestamp:
        return None

    normalized = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)

    return int(parsed.timestamp() * 1_000_000_000)
