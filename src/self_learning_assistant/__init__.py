"""Personal local assistant with user-controlled memory and knowledge."""

from .domain import ChatMessage, MemoryScope, WebEvidence
from .model import ChatModel

__all__ = ["ChatMessage", "ChatModel", "MemoryScope", "WebEvidence"]
