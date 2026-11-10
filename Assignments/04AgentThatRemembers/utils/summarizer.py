from __future__ import annotations

import os

from utils.models import SummaryInput

try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore[assignment]


DEFAULT_MODEL = "openai/gpt-oss-120b"


class GroqSummarizer:
    """Real application summarizer, replaced by a mock during evaluation."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        resolved_key = api_key or os.getenv("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError("GROQ_API_KEY is required to call Groq.")
        if Groq is None:
            raise ImportError("Install the groq package before running the application.")
        self.client = Groq(api_key=resolved_key)
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)

    def __call__(self, summary_input: SummaryInput) -> str:
        prior = summary_input.existing_summary or "No earlier summary."
        transcript = "\n".join(
            f"{message.role}: {message.content}"
            for message in summary_input.older_messages
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Update the conversation summary without repeating facts.",
                },
                {
                    "role": "user",
                    "content": f"Existing summary:\n{prior}\n\nOlder messages:\n{transcript}",
                },
            ],
        )
        if not response.choices:
            raise RuntimeError("Groq returned no completion choices.")
        return response.choices[0].message.content or ""
