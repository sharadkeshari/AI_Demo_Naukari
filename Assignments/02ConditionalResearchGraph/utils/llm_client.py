from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

try:
    from groq import Groq
except ImportError:  # Allows evaluation to inject a mock without loading the SDK.
    Groq = None  # type: ignore[assignment]


DEFAULT_MODEL = "openai/gpt-oss-120b"


def _message_payload(message: BaseMessage) -> dict[str, Any]:
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": str(message.content)}
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": str(message.content)}
    if isinstance(message, ToolMessage):
        payload: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": str(message.content),
        }
        if message.name:
            payload["name"] = message.name
        return payload
    if isinstance(message, AIMessage):
        payload = {"role": "assistant", "content": str(message.content or "")}
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(tool_call["args"]),
                    },
                }
                for tool_call in message.tool_calls
            ]
        return payload
    raise TypeError(f"Unsupported message type: {type(message).__name__}")


class LLMClient:
    """Groq client that converts responses into LangChain AIMessage objects."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        resolved_key = api_key or os.getenv("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError("GROQ_API_KEY is required to call Groq.")
        if Groq is None:
            raise ImportError("Install the groq package before running the application.")

        self.client = Groq(api_key=resolved_key)
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)

    def complete(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Mapping[str, Any]],
    ) -> AIMessage:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[_message_payload(message) for message in messages],
            tools=list(tools),
            tool_choice="auto",
        )
        if not response.choices:
            raise RuntimeError("Groq returned no completion choices.")

        raw_message = response.choices[0].message
        normalized_calls = []
        for raw_call in list(raw_message.tool_calls or []):
            try:
                arguments = json.loads(raw_call.function.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            normalized_calls.append(
                {
                    "name": raw_call.function.name,
                    "args": arguments,
                    "id": raw_call.id,
                    "type": "tool_call",
                }
            )

        return AIMessage(
            content=raw_message.content or "",
            tool_calls=normalized_calls,
        )
