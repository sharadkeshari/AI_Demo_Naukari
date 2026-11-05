from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph, add_messages

from utils.llm_client import LLMClient
from utils.local_tools import SEARCH_TOOL_SCHEMA, tools_node


class ResearchState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def make_agent_node(
    llm_client: LLMClient | None = None,
) -> Callable[[ResearchState], dict[str, list[AIMessage]]]:
    """Create a model node for real or mocked LLM execution."""
    client = llm_client or LLMClient()

    def agent_node(state: ResearchState) -> dict[str, list[AIMessage]]:
        message = client.complete(list(state["messages"]), [SEARCH_TOOL_SCHEMA])
        return {"messages": [message]}

    return agent_node


def route_after_agent(state: ResearchState) -> Literal["tools", "end"]:
    """Route a tool request to execution and a final response to completion."""
    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "tools"
    return "end"


def build_research_graph(llm_client: LLMClient | None = None) -> Any:
    """Build the supplied conditional agent graph."""
    builder = StateGraph(ResearchState)
    builder.add_node("agent", make_agent_node(llm_client))
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "end": END},
    )
    builder.add_edge("tools", "agent")
    return builder.compile()
