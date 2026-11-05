from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage


EVIDENCE_RECORDS = [
    {
        "source_id": "guide-battery",
        "content": "Battery runtime drops when screen brightness and background sync increase.",
    },
    {
        "source_id": "guide-heat",
        "content": "Overheating can be reduced by closing heavy applications and improving airflow.",
    },
    {
        "source_id": "guide-network",
        "content": "Intermittent Wi-Fi may improve after restarting the router and updating its firmware.",
    },
]


SEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_evidence",
        "description": "Search the supplied local troubleshooting evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The troubleshooting topic to search for.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def search_evidence(query: str) -> dict[str, Any]:
    """Return deterministic local evidence ordered by token overlap."""
    query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    ranked = []
    for record in EVIDENCE_RECORDS:
        content_tokens = set(re.findall(r"[a-z0-9]+", record["content"].lower()))
        overlap = len(query_tokens & content_tokens)
        if overlap:
            ranked.append((overlap, record["source_id"], record))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return {
        "query": query,
        "matches": [item[2] for item in ranked[:2]],
    }


def tools_node(state: dict[str, Any]) -> dict[str, list[ToolMessage]]:
    """Execute tool calls emitted by the latest assistant message."""
    latest = state["messages"][-1]
    if not isinstance(latest, AIMessage):
        raise TypeError("The tools node requires an AIMessage.")

    results: list[ToolMessage] = []
    for tool_call in latest.tool_calls:
        name = tool_call["name"]
        arguments = tool_call["args"]
        if name != "search_evidence":
            output = {"error": "unknown_tool", "tool_name": name}
        elif not isinstance(arguments, dict) or not isinstance(
            arguments.get("query"), str
        ):
            output = {"error": "invalid_arguments", "tool_name": name}
        else:
            output = search_evidence(arguments["query"])

        results.append(
            ToolMessage(
                content=json.dumps(output, sort_keys=True),
                tool_call_id=tool_call["id"],
                name=name,
            )
        )
    return {"messages": results}
