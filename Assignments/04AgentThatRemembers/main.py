"""Compact Long Conversation State.

Scenario
--------
A study assistant has a growing conversation history. It must summarize older
messages while retaining recent turns and structured learner state exactly.

Complete only these TODOs:
1. select_messages_to_compact
2. build_summary_input
3. compact_memory

The state records, summary-input type, validation errors, limits, and a
Groq-compatible summarizer are supplied. Pytest uses a mocked summarizer, so it
requires no API key or network access.

Required behavior
-----------------
- threshold and keep_recent must be positive non-boolean integers, with
  keep_recent no greater than threshold. Otherwise raise InvalidLimitError.
- Compact only when the message count is greater than threshold.
- Select every message except the exact latest keep_recent messages.
- SummaryInput must keep an existing nonblank summary separately from the
  ordered older messages. A blank existing summary becomes None.
- compact_memory calls the summarizer once only when compaction is needed.
- The returned state contains the new single SummaryRecord and exact recent
  message tail. Preferences and unresolved tasks remain unchanged.
- Do not mutate the input state, its message list, preferences, or tasks.
- Raise InvalidSummaryError if the summarizer does not return nonblank text.

Not required: writing a summarization prompt, judging summary prose, counting
tokens, implementing persistence, calling a real model during evaluation, or
building an interface.
"""

from __future__ import annotations

from collections.abc import Iterable

from utils.models import (
    InvalidLimitError,
    InvalidSummaryError,
    MemoryState,
    MessageRecord,
    SummaryInput,
    SummaryRecord,
)


def select_messages_to_compact(state: MemoryState, threshold: int, keep_recent: int,) -> tuple[MessageRecord, ...]:
    """Select the ordered older messages only when compaction is required."""
    # TODO 1: Validate both limits as positive non-boolean integers and require
    # keep_recent <= threshold. Compact only above the threshold, returning the
    # ordered prefix before the exact most-recent keep_recent messages.
    if isinstance(threshold,bool) or not isinstance(threshold, int) or threshold<1:
        raise InvalidLimitError("Invalid Threshold value") 
    if isinstance(keep_recent,bool) or not isinstance(keep_recent, int) or keep_recent<1:
        raise InvalidLimitError("Invalid keep_recent value")
    if(keep_recent> threshold):
        raise InvalidLimitError(" keep_recent cannot be greater than threshold")
    
    state_messages= state.messages
    # compact only when message count exceeds threshold
    if len(state_messages)<=threshold:
       return ()

    #return everything except the most recent keep_recent messages
    return tuple(state_messages[:-keep_recent])

    

def build_summary_input(
    existing_summary: SummaryRecord | None,
    older_messages: Iterable[MessageRecord],
) -> SummaryInput:
    """Build one non-nested input for the supplied summarizer."""
    # TODO 2: Return SummaryInput with the older messages as an ordered tuple.
    # Keep nonblank existing summary text in its separate field, and use None
    # when no summary exists or its text is blank.
    summary_content=None
    if (existing_summary is not None and existing_summary.content.strip()):
        summary_content=existing_summary.content
    return SummaryInput(existing_summary=summary_content, older_messages=tuple(older_messages))


def compact_memory(    state: MemoryState,
    summarizer,    threshold: int,    keep_recent: int,
) -> MemoryState:
    """Compact older messages without changing caller-owned state."""
    # TODO 3: If no compaction is needed, return an equivalent safe copy without
    # calling summarizer. Otherwise call it exactly once, reject blank output,
    # and return one new SummaryRecord plus the exact recent tail. Copy all
    # structured fields so the caller's state cannot be mutated.
    older_messages= select_messages_to_compact(threshold=threshold, keep_recent=keep_recent, state=state)
    if not older_messages:
        return state
    
    summary_input = build_summary_input(existing_summary=state.summary, older_messages=older_messages)
    summary_text = summarizer(summary_input)

    if not isinstance(summary_text, str) or not summary_text.strip():
        raise InvalidSummaryError("Summary in invalid")

    recent_tail= tuple(state.messages[len(older_messages):])

    return MemoryState(
    messages=recent_tail,
    summary=SummaryRecord(content=summary_text),
    preferences=dict(state.preferences),
    unresolved_tasks=tuple(state.unresolved_tasks),
    )
