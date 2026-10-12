"""Small serializable domain objects shared by the web tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class WebToolError(RuntimeError):
    """A safe error that can be returned to an MCP client."""


class UnsafeRequestError(WebToolError):
    """The requested query or URL violates the tool's safety boundary."""


class BackendUnavailableError(WebToolError):
    """The requested search backend is unavailable or not configured."""


@dataclass(frozen=True, slots=True)
class SearchHit:
    title: str
    url: str
    snippet: str
    provider: str
    published_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchResponse:
    query: str
    backend: str
    results: tuple[SearchHit, ...]
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "backend": self.backend,
            "fallback_used": self.fallback_used,
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True, slots=True)
class FetchedPage:
    url: str
    final_url: str
    title: str
    text: str
    retrieved_at: str
    content_hash: str
    content_type: str
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
