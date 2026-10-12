"""SearXNG-first search with an optional Brave Search fallback."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from self_learning_assistant.tools.domain import (
    BackendUnavailableError,
    SearchHit,
    SearchResponse,
    WebToolError,
)
from self_learning_assistant.tools.security import (
    ensure_safe_query,
    validate_search_endpoint,
)

JsonRequester = Callable[[Request, float], Mapping[str, Any]]


class SearchProvider(Protocol):
    name: str

    def search(
        self,
        query: str,
        *,
        max_results: int,
        language: str,
        time_range: str | None,
    ) -> tuple[SearchHit, ...]: ...


def _request_json(request: Request, timeout: float) -> Mapping[str, Any]:
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = response.read(2_000_001)
    if len(payload) > 2_000_000:
        raise WebToolError("搜索服务响应过大")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WebToolError("搜索服务返回了无效 JSON") from exc
    if not isinstance(value, dict):
        raise WebToolError("搜索服务返回格式不正确")
    return value


class SearxngProvider:
    name = "searxng"

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        requester: JsonRequester = _request_json,
    ) -> None:
        self._base_url = validate_search_endpoint(base_url)
        self._timeout = timeout
        self._requester = requester

    def search(
        self,
        query: str,
        *,
        max_results: int,
        language: str,
        time_range: str | None,
    ) -> tuple[SearchHit, ...]:
        params = {
            "q": query,
            "format": "json",
            "language": language,
            "categories": "general",
        }
        if time_range:
            params["time_range"] = time_range
        request = Request(
            f"{self._base_url}/search?{urlencode(params)}",
            headers={
                "Accept": "application/json",
                "User-Agent": "self-learning-assistant/0.3.0",
            },
        )
        payload = self._requester(request, self._timeout)
        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list):
            raise WebToolError("SearXNG results 字段格式不正确")
        return tuple(
            SearchHit(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                snippet=str(item.get("content", "")),
                provider=self.name,
                published_at=_optional_text(
                    item.get("publishedDate") or item.get("published_date")
                ),
            )
            for item in raw_results[:max_results]
            if isinstance(item, dict) and item.get("url")
        )


class BraveProvider:
    name = "brave"

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 10.0,
        requester: JsonRequester = _request_json,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Brave API key must not be empty")
        self._api_key = api_key
        self._timeout = timeout
        self._requester = requester

    def search(
        self,
        query: str,
        *,
        max_results: int,
        language: str,
        time_range: str | None,
    ) -> tuple[SearchHit, ...]:
        params = {"q": query, "count": str(max_results), "safesearch": "moderate"}
        if language != "all":
            params["search_lang"] = language
        if time_range:
            params["freshness"] = {
                "day": "pd",
                "month": "pm",
                "year": "py",
            }[time_range]
        request = Request(
            "https://api.search.brave.com/res/v1/web/search?" + urlencode(params),
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self._api_key,
                "User-Agent": "self-learning-assistant/0.3.0",
            },
        )
        payload = self._requester(request, self._timeout)
        web = payload.get("web", {})
        raw_results = web.get("results", []) if isinstance(web, dict) else []
        if not isinstance(raw_results, list):
            raise WebToolError("Brave results 字段格式不正确")
        return tuple(
            SearchHit(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                snippet=str(item.get("description", "")),
                provider=self.name,
                published_at=_optional_text(item.get("page_age")),
            )
            for item in raw_results[:max_results]
            if isinstance(item, dict) and item.get("url")
        )


class SearchRouter:
    def __init__(
        self,
        searxng: SearchProvider,
        brave: SearchProvider | None = None,
    ) -> None:
        self._searxng = searxng
        self._brave = brave

    def search(
        self,
        query: str,
        *,
        backend: str = "auto",
        max_results: int = 8,
        language: str = "all",
        time_range: str | None = None,
    ) -> SearchResponse:
        query = ensure_safe_query(query)
        if backend not in {"auto", "searxng", "brave"}:
            raise WebToolError("backend 必须是 auto、searxng 或 brave")
        if not 1 <= max_results <= 20:
            raise WebToolError("max_results 必须在 1 到 20 之间")
        if time_range not in {None, "day", "month", "year"}:
            raise WebToolError("time_range 必须是 day、month 或 year")

        if backend == "brave":
            return self._search_brave(
                query, max_results=max_results, language=language, time_range=time_range
            )
        if backend == "searxng":
            results = self._call(
                self._searxng, query, max_results, language, time_range
            )
            return SearchResponse(query, "searxng", results)

        try:
            results = self._call(
                self._searxng, query, max_results, language, time_range
            )
            if results or self._brave is None:
                return SearchResponse(query, "searxng", results)
        except (OSError, WebToolError):
            if self._brave is None:
                raise BackendUnavailableError(
                    "SearXNG 不可用，且未配置 Brave"
                ) from None
        response = self._search_brave(
            query, max_results=max_results, language=language, time_range=time_range
        )
        return SearchResponse(
            query, response.backend, response.results, fallback_used=True
        )

    def status(self) -> dict[str, Any]:
        return {
            "default_backend": "searxng",
            "searxng_configured": True,
            "brave_configured": self._brave is not None,
        }

    def _search_brave(
        self, query: str, *, max_results: int, language: str, time_range: str | None
    ) -> SearchResponse:
        if self._brave is None:
            raise BackendUnavailableError("未配置 Brave Search API Key")
        results = self._call(self._brave, query, max_results, language, time_range)
        return SearchResponse(query, "brave", results)

    @staticmethod
    def _call(
        provider: SearchProvider,
        query: str,
        max_results: int,
        language: str,
        time_range: str | None,
    ) -> tuple[SearchHit, ...]:
        return provider.search(
            query,
            max_results=max_results,
            language=language,
            time_range=time_range,
        )


def _optional_text(value: object) -> str | None:
    return str(value) if value not in {None, ""} else None
