from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class InvalidLimitError(ValueError):
    pass


class InvalidSummaryError(ValueError):
    pass


@dataclass(frozen=True)
class MessageRecord:
    role: str
    content: str


@dataclass(frozen=True)
class SummaryRecord:
    content: str


@dataclass(frozen=True)
class SummaryInput:
    existing_summary: str | None
    older_messages: tuple[MessageRecord, ...]


@dataclass(frozen=True)
class MemoryState:
    messages: tuple[MessageRecord, ...]
    summary: SummaryRecord | None
    preferences: dict[str, Any]
    unresolved_tasks: tuple[str, ...]
