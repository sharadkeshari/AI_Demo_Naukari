from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage


STUDY_TIPS = {
    "graph traversal": "Draw the frontier after each BFS or DFS step.",
    "recursion": "Write the base case before tracing recursive calls.",
    "dynamic programming": "Define the state and transition before coding.",
}


STUDY_TIP_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "lookup_study_tip",
        "description": "Look up a study tip from the supplied local guide.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic to study."}
            },
            "required": ["topic"],
            "additionalProperties": False,
        },
    },
}


def lookup_study_tip(topic: str) -> dict[str, str]:
    normalized = topic.strip().lower()
    return {
        "topic": normalized,
        "tip": STUDY_TIPS.get(normalized, "Break the topic into small examples."),
    }


def tools_node(state: dict[str, Any]) -> dict[str, list[ToolMessage]]:
    """Safely dispatch tool calls from the latest assistant message."""
    latest = state["messages"][-1]
    if not isinstance(latest, AIMessage):
        raise TypeError("The tools node requires an AIMessage.")

    results: list[ToolMessage] = []
    for tool_call in latest.tool_calls:
        name = tool_call["name"]
        arguments = tool_call["args"]
        if name != "lookup_study_tip":
            output = {"error": "unknown_tool", "tool_name": name}
        elif not isinstance(arguments, dict) or not isinstance(arguments.get("topic"), str):
            output = {"error": "invalid_arguments", "tool_name": name}
        else:
            output = lookup_study_tip(arguments["topic"])
        results.append(
            ToolMessage(
                content=json.dumps(output, sort_keys=True),
                tool_call_id=tool_call["id"],
                name=name,
            )
        )
    return {"messages": results}
