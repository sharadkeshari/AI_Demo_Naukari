from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from utils.models import OUTPUT_SNIPPET_LIMIT, TraceStep


def _message_content(message: BaseMessage) -> str:
    """Convert supported message content into stable display text."""
    content = message.content
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)


def _message_type(message: BaseMessage) -> str:
    if isinstance(message, AIMessage):
        return "ai"
    if isinstance(message, HumanMessage):
        return "human"
    if isinstance(message, ToolMessage):
        return "tool"
    return str(message.type)


def trace_step_from_update(node_name: str, update: Any) -> TraceStep:
    """Normalize one node update into the supplied compact trace type."""
    raw_messages = update.get("messages", []) if isinstance(update, Mapping) else []
    if isinstance(raw_messages, BaseMessage):
        messages = [raw_messages]
    else:
        messages = [message for message in raw_messages if isinstance(message, BaseMessage)]

    tool_names: list[str] = []
    tool_call_ids: list[str] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                if tool_call.get("name"):
                    tool_names.append(str(tool_call["name"]))
                if tool_call.get("id"):
                    tool_call_ids.append(str(tool_call["id"]))
        elif isinstance(message, ToolMessage):
            if message.name:
                tool_names.append(message.name)
            if message.tool_call_id:
                tool_call_ids.append(message.tool_call_id)

    latest = messages[-1] if messages else None
    output = _message_content(latest).strip() if latest else ""
    return TraceStep(
        node_name=str(node_name),
        message_type=_message_type(latest) if latest else "empty",
        tool_names=tuple(tool_names),
        tool_call_ids=tuple(tool_call_ids),
        output_snippet=output[:OUTPUT_SNIPPET_LIMIT],
    )


def final_answer_from_events(events: Iterable[Mapping[str, Any]]) -> str | None:
    """Return the latest non-tool-calling assistant text from completed events."""
    ordered_events = list(events)
    for event in reversed(ordered_events):
        for update in reversed(list(event.values())):
            if not isinstance(update, Mapping):
                continue
            raw_messages = update.get("messages", [])
            if isinstance(raw_messages, BaseMessage):
                messages = [raw_messages]
            else:
                messages = list(raw_messages)
            for message in reversed(messages):
                if isinstance(message, AIMessage) and not message.tool_calls:
                    answer = _message_content(message).strip()
                    if answer:
                        return answer
    return None
