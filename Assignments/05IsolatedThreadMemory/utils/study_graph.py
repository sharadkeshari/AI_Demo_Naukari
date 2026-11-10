from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph, add_messages

from utils.llm_client import LLMClient
from utils.local_tools import STUDY_TIP_TOOL_SCHEMA, tools_node


class StudyState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def make_tutor_node(
    llm_client: LLMClient | None = None,
) -> Callable[[StudyState], dict[str, list[AIMessage]]]:
    """Create the supplied tutor node for real or mocked LLM execution."""
    client = llm_client or LLMClient()

    def tutor_node(state: StudyState) -> dict[str, list[AIMessage]]:
        reply = client.complete(list(state["messages"]), [STUDY_TIP_TOOL_SCHEMA])
        return {"messages": [reply]}

    return tutor_node


def route_after_tutor(state: StudyState) -> Literal["tools", "end"]:
    latest = state["messages"][-1]
    if isinstance(latest, AIMessage) and latest.tool_calls:
        return "tools"
    return "end"


def build_study_graph(
    llm_client: LLMClient | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Build the supplied checkpointed agent and conditional tool loop."""
    builder = StateGraph(StudyState)
    builder.add_node("tutor", make_tutor_node(llm_client))
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "tutor")
    builder.add_conditional_edges(
        "tutor",
        route_after_tutor,
        {"tools": "tools", "end": END},
    )
    builder.add_edge("tools", "tutor")
    return builder.compile(checkpointer=checkpointer or InMemorySaver())
