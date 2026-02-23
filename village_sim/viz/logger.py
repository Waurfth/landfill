"""Structured event logging for narrative and debugging."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, TextIO


@dataclass
class LogEntry:
    """A single log entry."""

    day: int
    category: str
    message: str
    villager_ids: list[int] = field(default_factory=list)
    data: dict = field(default_factory=dict)


class SimLogger:
    """Structured logging with categories and verbosity control."""

    # Category constants
    ACTIVITY = "ACTIVITY"
    TRADE = "TRADE"
    SOCIAL = "SOCIAL"
    LIFECYCLE = "LIFECYCLE"
    MARRIAGE = "MARRIAGE"
    EVENT = "EVENT"
    KNOWLEDGE = "KNOWLEDGE"
    SENTIMENT = "SENTIMENT"

    def __init__(
        self,
        verbosity: int = 1,
        log_file: Optional[str] = None,
        stdout: bool = True,
    ) -> None:
        """
        verbosity levels:
            0 = only lifecycle and events
            1 = + activity summaries
            2 = + social, trade, knowledge
            3 = everything (debug)
        """
        self.verbosity = verbosity
        self._buffer: list[LogEntry] = []
        self._all_entries: list[LogEntry] = []
        self._file: Optional[TextIO] = None
        self._stdout = stdout

        if log_file:
            os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
            self._file = open(log_file, "w", encoding="utf-8")

    def log(
        self,
        category: str,
        message: str,
        villager_ids: Optional[list[int]] = None,
        day: int = 0,
        **data,
    ) -> None:
        """Log an event."""
        entry = LogEntry(
            day=day,
            category=category,
            message=message,
            villager_ids=villager_ids or [],
            data=data,
        )
        self._buffer.append(entry)

    def flush_day(self, day: int) -> None:
        """Write buffered logs for the day."""
        # Filter by verbosity
        _VERBOSITY_MAP = {
            self.LIFECYCLE: 0,
            self.MARRIAGE: 0,
            self.EVENT: 0,
            self.ACTIVITY: 1,
            self.TRADE: 2,
            self.SOCIAL: 2,
            self.KNOWLEDGE: 2,
            self.SENTIMENT: 3,
        }

        for entry in self._buffer:
            required_verbosity = _VERBOSITY_MAP.get(entry.category, 1)
            if required_verbosity <= self.verbosity:
                line = f"[Day {entry.day:>4}] [{entry.category:<10}] {entry.message}"
                if self._stdout:
                    print(line)
                if self._file:
                    self._file.write(line + "\n")

        self._all_entries.extend(self._buffer)
        self._buffer.clear()

        if self._file:
            self._file.flush()

    def get_narrative(self, day: int) -> str:
        """Generate a human-readable summary of a specific day."""
        day_entries = [e for e in self._all_entries if e.day == day]
        if not day_entries:
            return f"Day {day}: Nothing notable happened."

        lines = [f"=== Day {day} ==="]
        for entry in day_entries:
            lines.append(f"  [{entry.category}] {entry.message}")
        return "\n".join(lines)

    def export_json(self, filepath: str) -> None:
        """Export all log entries to JSON."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        data = [
            {
                "day": e.day,
                "category": e.category,
                "message": e.message,
                "villager_ids": e.villager_ids,
                "data": e.data,
            }
            for e in self._all_entries
        ]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def close(self) -> None:
        """Close the log file."""
        if self._file:
            self._file.close()
            self._file = None
