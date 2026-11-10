"""Isolated Thread Memory.

Scenario
--------
A study assistant serves multiple learners. Each learner's conversation must
continue across messages without exposing or mixing another learner's history.

Your task
---------
Complete only these three TODO functions:

1. build_thread_config
2. send_message
3. extract_reply

The result type, validation errors, Groq-compatible client, local study tool,
conditional LangGraph agent, and in-memory checkpointer are supplied. The
application can use GROQ_API_KEY during normal execution. Pytest injects a
mocked LLM client, so evaluation makes no real API call and requires no API key
or network access.

Required behavior
-----------------
build_thread_config:
- Reject a non-string, empty, or whitespace-only thread_id by raising
  InvalidThreadIdError.
- Strip surrounding whitespace from a valid thread ID.
- Return the LangGraph configuration in this exact shape:
  {"configurable": {"thread_id": "the-clean-thread-id"}}

send_message:
- Reject a non-string, empty, or whitespace-only message by raising
  InvalidMessageError.
- Build the thread configuration by calling build_thread_config.
- Create a new HumanMessage containing the caller's exact message text.
- Invoke the supplied graph with a new {"messages": [...]} input dictionary
  and the thread configuration. Do not alter caller-provided data.
- Return the graph's result unchanged.

extract_reply:
- Inspect the result's "messages" sequence from newest to oldest.
- Return a ConversationReply containing only the latest nonblank AIMessage
  text. Do not include earlier human, assistant, or tool messages.
- Raise MissingAssistantReplyError when no nonblank assistant text exists.

The evaluator checks observable behavior and does not require a particular
private implementation style.

Not required
------------
You do not need to build graph nodes, implement a tool, create a checkpointer,
write prompts, design a database, simulate a process restart, call a real model
during evaluation, build a chat interface, or deploy an application.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from utils.models import (
    ConversationReply,
    InvalidMessageError,
    InvalidThreadIdError,
    MissingAssistantReplyError,
)


def build_thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    """Validate a thread ID and build the LangGraph persistence config."""
    # TODO 1: Reject non-string or blank IDs with InvalidThreadIdError. Strip a
    # valid ID and return exactly {"configurable": {"thread_id": clean_id}}.
    
    if not isinstance(thread_id, str):
        raise InvalidThreadIdError()
    clean_id = thread_id.strip()
    if not clean_id:
        raise InvalidThreadIdError()
    return {"configurable": {"thread_id": clean_id}}

def send_message(    graph: Any,    thread_id: str,    message: str,) -> Mapping[str, Any]:
    """Send one user message through the supplied checkpointed graph."""
    # TODO 2: Reject non-string or blank messages with InvalidMessageError.
    # Keep valid message text exactly as supplied, wrap it in a HumanMessage,
    # call build_thread_config, invoke the graph once, and return its result.
    if not isinstance(message, str) or not message.strip():
        raise InvalidMessageError()

    config = build_thread_config(thread_id)

    return graph.invoke({"messages": [HumanMessage(content=message)]}, config=config,)


def extract_reply(result: Mapping[str, Any]) -> ConversationReply:
    """Return only the latest nonblank assistant reply from a graph result."""
    # TODO 3: Scan result["messages"] newest first. Return only the latest
    # nonblank AIMessage text in ConversationReply. If none exists, raise
    # MissingAssistantReplyError.
    messages = result.get("messages")
    
    if not messages:
      raise MissingAssistantReplyError()
    
    for message in reversed(messages):
      if (
      isinstance(message, AIMessage)
      and isinstance(message.content, str)
      and message.content.strip()
      ):
        return ConversationReply(content=message.content)
      
    raise MissingAssistantReplyError()