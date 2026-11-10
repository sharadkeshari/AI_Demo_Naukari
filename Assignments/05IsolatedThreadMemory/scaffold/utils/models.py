from __future__ import annotations

from dataclasses import dataclass


class InvalidThreadIdError(ValueError):
    """Raised when a thread ID cannot safely identify a conversation."""


class InvalidMessageError(ValueError):
    """Raised when a user message is not usable."""


class MissingAssistantReplyError(ValueError):
    """Raised when a graph result has no usable assistant response."""


@dataclass(frozen=True)
class ConversationReply:
    content: str
