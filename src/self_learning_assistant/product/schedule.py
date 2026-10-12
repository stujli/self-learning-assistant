"""Calendar windows use local wall time; duration limits use monotonic time."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import ProductError


@dataclass(frozen=True)
class Window:
    start: str
    end: str
    timezone: str
    enabled: bool = True

    @classmethod
    def parse(cls, value: Any) -> Window:
        if (
            not isinstance(value, dict)
            or set(value) != {"start", "end", "timezone", "enabled"}
            or type(value["enabled"]) is not bool
        ):
            raise ProductError("Window requires start, end, timezone, enabled")
        for key in ("start", "end"):
            if not isinstance(value[key], str) or not re.fullmatch(
                r"(?:[01]\d|2[0-3]):[0-5]\d", value[key]
            ):
                raise ProductError("Window time must be HH:MM")
        if value["start"] == value["end"]:
            raise ProductError("Learning window must have a distinct start and end")
        try:
            ZoneInfo(value["timezone"])
        except (ZoneInfoNotFoundError, TypeError, ValueError) as exc:
            raise ProductError(
                "Use an IANA timezone, for example Asia/Shanghai"
            ) from exc
        return cls(**value)

    def contains(self, instant: datetime) -> bool:
        if instant.tzinfo is None:
            raise ProductError("Schedule requires a timezone-aware clock")
        current = instant.astimezone(ZoneInfo(self.timezone)).strftime("%H:%M")
        return self.enabled and (
            self.start <= current < self.end
            if self.start < self.end
            else current >= self.start or current < self.end
        )

    def next_start(self, instant: datetime) -> str:
        if not self.enabled:
            return ""
        # Searching real instants correctly skips nonexistent wall times and handles folds.
        current = instant.astimezone(UTC).replace(second=0, microsecond=0)
        if self.contains(current):
            return current.isoformat()
        for minutes in range(1, 2881):
            candidate = current + timedelta(minutes=minutes)
            if self.contains(candidate):
                return candidate.isoformat()
        return ""

    def identifier(self, user: str, instant: datetime) -> str:
        local = instant.astimezone(ZoneInfo(self.timezone))
        date = local.date()
        if self.start > self.end and local.strftime("%H:%M") < self.end:
            date -= timedelta(days=1)
        return f"{user}:{date}:{self.start}:{self.end}:{self.timezone}"


def on_ac_power(root: Path = Path("/sys/class/power_supply")) -> bool:
    batteries = list(root.glob("*/type"))
    if not any(path.read_text().strip() == "Battery" for path in batteries):
        return True
    return any(
        path.exists() and path.read_text().strip() == "1"
        for path in root.glob("*/online")
    )
