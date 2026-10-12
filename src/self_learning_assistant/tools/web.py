"""Compose search and safe fetching for the product verification workflow."""

from __future__ import annotations

from self_learning_assistant.tools.config import WebToolsConfig
from self_learning_assistant.tools.domain import (
    FetchedPage,
    SearchResponse,
)
from self_learning_assistant.tools.fetch import SafeWebFetcher, build_web_fetcher
from self_learning_assistant.tools.search import (
    BraveProvider,
    SearchRouter,
    SearxngProvider,
)


class WebResearchService:
    """Adapt the bounded web tools to the chat research protocol."""

    def __init__(self, router: SearchRouter, fetcher: SafeWebFetcher) -> None:
        self._router = router
        self._fetcher = fetcher

    def search(self, query: str, *, max_results: int = 4) -> SearchResponse:
        return self._router.search(query, max_results=max_results)

    def fetch(self, url: str, *, max_chars: int = 20_000) -> FetchedPage:
        return self._fetcher.fetch(url, max_chars=max_chars)


def build_web_research_service(
    config: WebToolsConfig | None = None,
) -> WebResearchService:
    settings = config or WebToolsConfig.from_env()
    searxng = SearxngProvider(
        settings.searxng_url,
        timeout=settings.request_timeout_seconds,
    )
    brave = (
        BraveProvider(
            settings.brave_api_key,
            timeout=settings.request_timeout_seconds,
        )
        if settings.brave_api_key
        else None
    )
    return WebResearchService(
        SearchRouter(searxng, brave),
        build_web_fetcher(settings),
    )


__all__ = ["WebResearchService", "build_web_research_service"]
