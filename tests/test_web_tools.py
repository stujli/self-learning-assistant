from __future__ import annotations

import socket
from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request

import pytest

from self_learning_assistant.tools.config import WebToolsConfig
from self_learning_assistant.tools.domain import (
    BackendUnavailableError,
    SearchHit,
    UnsafeRequestError,
    WebToolError,
)
from self_learning_assistant.tools.fetch import Download, SafeWebFetcher
from self_learning_assistant.tools.search import (
    BraveProvider,
    SearchRouter,
    SearxngProvider,
)
from self_learning_assistant.tools.security import (
    ensure_safe_query,
    validate_public_http_url,
    validate_search_endpoint,
)


def public_resolver(
    host: str, port: int | None
) -> list[tuple[int, int, int, str, tuple[str, int]]]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 80))]


class FakeProvider:
    def __init__(
        self,
        name: str,
        results: tuple[SearchHit, ...] = (),
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.results = results
        self.error = error
        self.calls: list[tuple[str, int, str, str | None]] = []

    def search(
        self,
        query: str,
        *,
        max_results: int,
        language: str,
        time_range: str | None,
    ) -> tuple[SearchHit, ...]:
        self.calls.append((query, max_results, language, time_range))
        if self.error:
            raise self.error
        return self.results


def test_config_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLA_SEARXNG_URL", "http://search:8080/")
    monkeypatch.setenv("SLA_BRAVE_API_KEY", "secret")
    monkeypatch.setenv("SLA_WEB_TIMEOUT_SECONDS", "3.5")
    monkeypatch.setenv("SLA_WEB_MAX_DOWNLOAD_BYTES", "1234")
    monkeypatch.setenv("SLA_WEB_MAX_PAGE_CHARS", "4321")

    config = WebToolsConfig.from_env()

    assert config == WebToolsConfig(
        searxng_url="http://search:8080/",
        brave_api_key="secret",
        request_timeout_seconds=3.5,
        max_download_bytes=1234,
        max_page_chars=4321,
    )


@pytest.mark.parametrize(
    "query",
    ["", "x" * 501, "api_key=secret-value", "-----BEGIN PRIVATE KEY-----"],
)
def test_search_query_rejects_empty_oversized_or_secret_text(query: str) -> None:
    with pytest.raises(UnsafeRequestError):
        ensure_safe_query(query)


def test_search_query_is_trimmed() -> None:
    assert ensure_safe_query("  current docs  ") == "current docs"


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/admin",
        "http://127.0.0.1/admin",
        "http://10.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data",
        "https://user:pass@example.com/",
    ],
)
def test_public_url_validation_rejects_unsafe_targets(url: str) -> None:
    with pytest.raises(UnsafeRequestError):
        validate_public_http_url(url, public_resolver)


def test_public_url_validation_accepts_public_target() -> None:
    assert (
        validate_public_http_url("https://example.com/a", public_resolver)
        == "https://example.com/a"
    )


def test_search_endpoint_allows_localhost_but_rejects_credentials() -> None:
    assert validate_search_endpoint("http://127.0.0.1:8080/") == "http://127.0.0.1:8080"
    with pytest.raises(UnsafeRequestError):
        validate_search_endpoint("http://user:pass@127.0.0.1:8080")


def test_searxng_provider_builds_request_and_parses_results() -> None:
    captured: list[Request] = []

    def requester(request: Request, timeout: float) -> Mapping[str, Any]:
        captured.append(request)
        assert timeout == 2
        return {
            "results": [
                {
                    "title": "Official docs",
                    "url": "https://example.com/docs",
                    "content": "A result",
                    "publishedDate": "2026-01-01",
                }
            ]
        }

    provider = SearxngProvider("http://localhost:8080/", timeout=2, requester=requester)
    results = provider.search(
        "mcp tools", max_results=5, language="zh-CN", time_range="month"
    )

    query = parse_qs(urlsplit(captured[0].full_url).query)
    assert query["q"] == ["mcp tools"]
    assert query["format"] == ["json"]
    assert query["time_range"] == ["month"]
    assert results[0].provider == "searxng"
    assert results[0].published_at == "2026-01-01"


def test_searxng_provider_rejects_bad_results_shape() -> None:
    provider = SearxngProvider(
        "http://localhost:8080", requester=lambda request, timeout: {"results": {}}
    )
    with pytest.raises(WebToolError):
        provider.search("q", max_results=1, language="all", time_range=None)


def test_brave_provider_sends_key_in_header_and_parses_results() -> None:
    captured: list[Request] = []

    def requester(request: Request, timeout: float) -> Mapping[str, Any]:
        captured.append(request)
        return {
            "web": {
                "results": [
                    {
                        "title": "Result",
                        "url": "https://example.com",
                        "description": "text",
                        "page_age": "1 day ago",
                    }
                ]
            }
        }

    provider = BraveProvider("top-secret", requester=requester)
    results = provider.search("query", max_results=3, language="en", time_range="month")

    assert captured[0].get_header("X-subscription-token") == "top-secret"
    assert "top-secret" not in captured[0].full_url
    assert parse_qs(urlsplit(captured[0].full_url).query)["freshness"] == ["pm"]
    assert results[0].snippet == "text"
    assert results[0].provider == "brave"


def test_router_prefers_searxng() -> None:
    hit = SearchHit("title", "https://example.com", "text", "searxng")
    searxng = FakeProvider("searxng", (hit,))
    brave = FakeProvider("brave", (hit,))

    response = SearchRouter(searxng, brave).search("query")

    assert response.backend == "searxng"
    assert not response.fallback_used
    assert brave.calls == []


@pytest.mark.parametrize("searxng_error", [None, OSError("offline")])
def test_router_falls_back_on_empty_or_unavailable_searxng(
    searxng_error: Exception | None,
) -> None:
    brave_hit = SearchHit("title", "https://example.com", "text", "brave")
    searxng = FakeProvider("searxng", error=searxng_error)
    brave = FakeProvider("brave", (brave_hit,))

    response = SearchRouter(searxng, brave).search("query")

    assert response.backend == "brave"
    assert response.fallback_used


def test_router_does_not_fallback_when_backend_is_explicit() -> None:
    searxng = FakeProvider("searxng", error=OSError("offline"))
    brave = FakeProvider("brave")
    with pytest.raises(OSError):
        SearchRouter(searxng, brave).search("query", backend="searxng")
    assert brave.calls == []


def test_router_reports_missing_brave_and_validates_options() -> None:
    router = SearchRouter(FakeProvider("searxng"))
    with pytest.raises(BackendUnavailableError):
        router.search("query", backend="brave")
    with pytest.raises(WebToolError):
        router.search("query", backend="unknown")
    with pytest.raises(WebToolError):
        router.search("query", max_results=0)
    with pytest.raises(WebToolError):
        router.search("query", time_range="decade")


def test_fetcher_extracts_readable_html_and_metadata() -> None:
    body = b"""
        <html><head><title> Example title </title><style>hidden</style></head>
        <body><nav>menu</nav><main><h1>Heading</h1><p>Useful text.</p>
        <script>ignore()</script></main></body></html>
    """
    fetcher = SafeWebFetcher(
        resolver=public_resolver,
        downloader=lambda url: Download(body, url, "text/html", "utf-8"),
    )

    page = fetcher.fetch("https://example.com/article", max_chars=1_000)

    assert page.title == "Example title"
    assert "Heading" in page.text and "Useful text." in page.text
    assert "menu" not in page.text and "ignore" not in page.text
    assert page.content_hash.startswith("sha256:")
    assert page.retrieved_at.endswith("+00:00")
    assert not page.truncated


def test_fetcher_handles_plain_text_and_truncates() -> None:
    body = ("x" * 1_200).encode()
    fetcher = SafeWebFetcher(
        resolver=public_resolver,
        downloader=lambda url: Download(body, url, "text/plain"),
    )
    page = fetcher.fetch("https://example.com/text", max_chars=1_000)
    assert len(page.text) == 1_000
    assert page.truncated


def test_fetcher_rejects_bad_type_redirect_and_character_limit() -> None:
    bad_type = SafeWebFetcher(
        resolver=public_resolver,
        downloader=lambda url: Download(b"pdf", url, "application/pdf"),
    )
    with pytest.raises(WebToolError):
        bad_type.fetch("https://example.com/file", max_chars=1_000)

    private_redirect = SafeWebFetcher(
        resolver=public_resolver,
        downloader=lambda url: Download(b"text", "http://127.0.0.1", "text/plain"),
    )
    with pytest.raises(UnsafeRequestError):
        private_redirect.fetch("https://example.com/file", max_chars=1_000)

    with pytest.raises(WebToolError):
        bad_type.fetch("https://example.com/file", max_chars=999)
