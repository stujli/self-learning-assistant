from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..domain import ChatMessage


class ChatModel(Protocol):
    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int = 128,
    ) -> str: ...


__all__ = ["ChatModel"]
