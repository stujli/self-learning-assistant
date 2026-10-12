"""Read-only tools that let MCP clients inspect the public web."""

from self_learning_assistant.tools.config import WebToolsConfig
from self_learning_assistant.tools.fetch import SafeWebFetcher
from self_learning_assistant.tools.search import SearchRouter

__all__ = ["SafeWebFetcher", "SearchRouter", "WebToolsConfig"]
