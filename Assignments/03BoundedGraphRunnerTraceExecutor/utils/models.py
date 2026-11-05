from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import BaseMessage


OUTPUT_SNIPPET_LIMIT = 80


class RunStatus(str, Enum):
    COMPLETED = "completed"
    INVALID_INPUT = "invalid_input"
    EMPTY_OUTPUT = "empty_output"
    LIMIT_REACHED = "limit_reached"
    GRAPH_ERROR = "graph_error"


@dataclass(frozen=True)
class GraphRunInput:
    graph_input: dict[str, list[BaseMessage]]
    config: dict[str, int]


@dataclass(frozen=True)
class TraceStep:
    node_name: str
    message_type: str
    tool_names: tuple[str, ...] = ()
    tool_call_ids: tuple[str, ...] = ()
    output_snippet: str = ""


@dataclass(frozen=True)
class RunResult:
    status: RunStatus
    final_answer: str | None
    trace: tuple[TraceStep, ...]
    step_count: int
    message: str | None = None
