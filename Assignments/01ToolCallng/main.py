"""Safe Local Tool Dispatcher.

Scenario
--------
A parcel-support assistant uses Groq tool calling to look up orders, estimate
delivery times, and prepare support-ticket drafts. The supplied agent loop asks
the model what to do, executes any requested local tool, returns the tool result
to the model, and then returns the model's final answer.

Your task
---------
Complete only the three TODO functions in this file:

1. validate_call
2. execute_call
3. make_tool_result

The Groq client, tool schemas, local tools, data classes, argument-checking
helper, and agent loop are supplied. When the application is run normally,
LLMClient uses GROQ_API_KEY. During evaluation, pytest supplies a mocked
LLMClient, so no real API call, API key, or network access is used.

Required behavior
-----------------
validate_call:
- Return unknown_tool when the tool name is not registered.
- Otherwise call the supplied validate_arguments helper with the call
  arguments and the selected tool's parameter contract.

execute_call:
- Validate before execution. An invalid call executes nothing, copies the
  validation error into ExecutionOutcome, and reports 0 attempts.
- max_attempts includes the first call. A value below 1 returns
  invalid_attempt_limit and executes nothing.
- Retry only exceptions declared in ToolSpec.transient_errors.
- Return transient_failure when the last allowed transient attempt fails.
- Return tool_execution_failed immediately for any other Exception.
- Never place raw exception text in the returned outcome.

make_tool_result:
- Preserve tool_call_id and tool name.
- Copy the success flag and attempt count.
- On success, set output to the tool value and leave error fields empty.
- On failure, set output to None and copy the error code and safe message.

The evaluator checks these behaviors through the supplied agent loop as well as
direct dispatcher calls. It does not require a particular private implementation
style. Do not modify the supplied Groq client or agent-loop helpers.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from groq import Groq
except ImportError:  # Allows tests to mock the client before the SDK is installed.
    Groq = None  # type: ignore[assignment]

from utils.models import (
    ExecutionOutcome,
    ToolCall,
    ToolResult,
    ToolSpec,
    ValidationResult,
    validate_arguments,
)


DEFAULT_MODEL = "openai/gpt-oss-120b"


class LLMClient:
    """Small Groq client used by the supplied agent loop."""

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
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> Any:
        """Call Groq and return the first assistant message."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            tools=list(tools),
            tool_choice="auto",
        )
        if not response.choices:
            raise RuntimeError("Groq returned no completion choices.")
        return response.choices[0].message


def validate_call(
    call: ToolCall,
    registry: Mapping[str, ToolSpec],
) -> ValidationResult:
    """Validate one tool call without executing its handler.

    TODO 1 contract:
    1. Look up ``call.name`` in ``registry``.
    2. If the name is absent, return ``ValidationResult.invalid`` with
       ``error_code="unknown_tool"`` and a message that names the unknown tool.
    3. If the name is present, call the supplied ``validate_arguments`` helper
       with ``call.arguments`` and that tool specification's ``parameters``.
    4. Return the helper's result unchanged. Do not execute any tool here.
    """
    
    # TODO 1: Implement the complete validation contract described above.
    raise NotImplementedError


def execute_call(
    call: ToolCall,
    registry: Mapping[str, ToolSpec],
    max_attempts: int,
) -> ExecutionOutcome:
    """Validate and execute one call with bounded transient retries.

    TODO 2 contract:
    1. Call ``validate_call`` before selecting or invoking the handler.
    2. On validation failure, return ``ExecutionOutcome.failed`` with the same
       error code and message, ``attempts=0``, and no tool execution.
    3. Reject ``max_attempts < 1`` with ``error_code="invalid_attempt_limit"``
       and ``attempts=0``. The first handler call counts as attempt 1.
    4. Invoke the registered handler with the validated arguments.
    5. Return ``ExecutionOutcome.succeeded`` immediately when it succeeds.
    6. Retry only exceptions declared in ``ToolSpec.transient_errors``. If the
       final allowed transient attempt fails, return ``transient_failure``.
    7. For every other exception, stop immediately and return
       ``tool_execution_failed``. Never expose raw exception text in a result.
    """
    # TODO 2: Implement the complete execution and retry contract above.
    raise NotImplementedError


def make_tool_result(
    call: ToolCall,
    outcome: ExecutionOutcome,
) -> ToolResult:
    """Convert an execution outcome into the safe agent-facing ToolResult.

    TODO 3 contract:
    1. Preserve ``call.tool_call_id`` as ``tool_call_id`` and ``call.name`` as
       ``tool_name``. Copy ``outcome.ok`` and ``outcome.attempts``.
    2. On success, copy ``outcome.value`` into ``output`` and set both
       ``error_code`` and ``message`` to ``None``.
    3. On failure, set ``output`` to ``None`` and copy the safe error code and
       message from the outcome.
    4. Return a ``ToolResult``. Do not mutate either input object.
    """
    # TODO 3: Implement the complete result-normalization contract above.
    raise NotImplementedError


def dispatch_tool_call(
    call: ToolCall,
    registry: Mapping[str, ToolSpec],
    max_attempts: int = 2,
) -> ToolResult:
    """Complete supplied wrapper used by the agent loop."""
    outcome = execute_call(call, registry, max_attempts)
    return make_tool_result(call, outcome)


def build_tool_schemas(
    registry: Mapping[str, ToolSpec],
) -> list[dict[str, Any]]:
    """Convert the local registry into Groq-compatible tool schemas."""
    json_types = {str: "string", int: "integer", float: "number", bool: "boolean"}
    schemas: list[dict[str, Any]] = []
    for name, spec in registry.items():
        properties = {
            parameter_name: {"type": json_types[parameter_type]}
            for parameter_name, parameter_type in spec.parameters.items()
        }
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": spec.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": list(spec.parameters),
                        "additionalProperties": False,
                    },
                },
            }
        )
    return schemas


def _decode_arguments(raw_arguments: object) -> object:
    if not isinstance(raw_arguments, str):
        return raw_arguments
    try:
        return json.loads(raw_arguments)
    except json.JSONDecodeError:
        return raw_arguments


def _assistant_message_payload(message: Any) -> dict[str, Any]:
    tool_calls = []
    for emitted_call in list(getattr(message, "tool_calls", None) or []):
        tool_calls.append(
            {
                "id": emitted_call.id,
                "type": "function",
                "function": {
                    "name": emitted_call.function.name,
                    "arguments": emitted_call.function.arguments,
                },
            }
        )
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": getattr(message, "content", None),
    }
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return payload


def _tool_message(result: ToolResult) -> dict[str, Any]:
    if result.ok:
        content = {"ok": True, "output": result.output}
    else:
        content = {
            "ok": False,
            "error_code": result.error_code,
            "message": result.message,
        }
    return {
        "role": "tool",
        "tool_call_id": result.tool_call_id,
        "name": result.tool_name,
        "content": json.dumps(content, sort_keys=True),
    }


def run_agent_loop(
    user_message: str,
    registry: Mapping[str, ToolSpec],
    llm_client: LLMClient | None = None,
    max_steps: int = 3,
    max_attempts: int = 2,
) -> str:
    """Run the supplied model, tool, observation loop until a final answer."""
    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("user_message must not be blank.")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")

    client = llm_client or LLMClient()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_message.strip()}
    ]
    tools = build_tool_schemas(registry)

    for _ in range(max_steps):
        assistant_message = client.complete(list(messages), tools)
        emitted_calls = list(getattr(assistant_message, "tool_calls", None) or [])
        if not emitted_calls:
            return str(getattr(assistant_message, "content", "") or "")

        messages.append(_assistant_message_payload(assistant_message))
        for emitted_call in emitted_calls:
            call = ToolCall(
                name=emitted_call.function.name,
                arguments=_decode_arguments(emitted_call.function.arguments),
                tool_call_id=emitted_call.id,
            )
            result = dispatch_tool_call(call, registry, max_attempts=max_attempts)
            messages.append(_tool_message(result))

    return "The assistant stopped after reaching the step limit."
