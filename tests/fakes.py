"""Synthetic model responses for tests; no real model is called."""

from __future__ import annotations

from collections.abc import Sequence

from self_learning_assistant.domain import ChatMessage


class RecordingModel:
    def __init__(self) -> None:
        self.calls: list[list[ChatMessage]] = []

    def generate(
        self, messages: Sequence[ChatMessage], *, max_new_tokens: int = 128
    ) -> str:
        self.calls.append(list(messages))
        return "Test response"
