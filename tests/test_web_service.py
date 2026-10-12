from __future__ import annotations

import socket

import pytest

from self_learning_assistant.tools.config import WebToolsConfig
from self_learning_assistant.tools.domain import SearchHit, WebToolError
from self_learning_assistant.tools.fetch import Download, SafeWebFetcher
from self_learning_assistant.tools.search import SearchRouter
from self_learning_assistant.tools.web import (
    WebResearchService,
    build_web_research_service,
)


def public_resolver(
    host: str, port: int | None
) -> list[tuple[int, int, int, str, tuple[str, int]]]:
    del host
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 80))]


class FixedProvider:
    name = "fixture"

    def search(
        self,
        query: str,
        *,
        max_results: int,
        language: str,
        time_range: str | None,
    ) -> tuple[SearchHit, ...]:
        del query, max_results, language, time_range
        return (
            SearchHit(
                "Result",
                "https://example.com/page",
                "Snippet",
                self.name,
            ),
        )


def test_web_service_returns_search_hits_and_hashed_pages() -> None:
    fetcher = SafeWebFetcher(
        resolver=public_resolver,
        downloader=lambda url: Download(
            b"<title>Page</title><p>Body</p>",
            url,
            "text/html",
        ),
    )
    service = WebResearchService(SearchRouter(FixedProvider()), fetcher)

    results = service.search("query")
    page = service.fetch(results.results[0].url)

    assert results.results[0].snippet == "Snippet"
    assert page.title == "Page"
    assert page.text == "Body"
    assert page.content_hash.startswith("sha256:")


def test_web_service_exposes_fetch_failure_without_fabricated_evidence() -> None:
    def fail(url: str) -> Download:
        del url
        raise WebToolError("offline")

    fetcher = SafeWebFetcher(resolver=public_resolver, downloader=fail)
    service = WebResearchService(SearchRouter(FixedProvider()), fetcher)

    results = service.search("query")
    assert results.results[0].snippet == "Snippet"
    with pytest.raises(WebToolError, match="offline"):
        service.fetch(results.results[0].url)


def test_research_service_builder_uses_web_configuration() -> None:
    service = build_web_research_service(
        WebToolsConfig(searxng_url="http://127.0.0.1:8080")
    )
    assert isinstance(service, WebResearchService)
