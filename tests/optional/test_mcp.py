import asyncio

from mcp_types import CallToolResult

from self_learning_assistant.tools.config import WebToolsConfig
from self_learning_assistant.tools.domain import SearchHit
from self_learning_assistant.tools.fetch import Download, SafeWebFetcher
from self_learning_assistant.tools.search import SearchRouter
from self_learning_assistant.tools.server import build_server
from tests.test_web_tools import FakeProvider, public_resolver


def test_mcp_server_exposes_and_calls_read_only_tools() -> None:
    hit = SearchHit("title", "https://example.com", "text", "searxng")
    router = SearchRouter(FakeProvider("searxng", (hit,)))
    fetcher = SafeWebFetcher(
        resolver=public_resolver,
        downloader=lambda url: Download(b"hello", url, "text/plain"),
    )
    server = build_server(WebToolsConfig(), router=router, fetcher=fetcher)

    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == {
        "search_web",
        "fetch_web_page",
        "web_tools_status",
    }
    assert all(tool.annotations and tool.annotations.read_only_hint for tool in tools)

    result = asyncio.run(server.call_tool("search_web", {"query": "query"}))
    assert isinstance(result, CallToolResult)
    assert result.structured_content
    assert result.structured_content["backend"] == "searxng"
    status = asyncio.run(server.call_tool("web_tools_status", {}))
    assert isinstance(status, CallToolResult)
    assert status.structured_content == {
        "default_backend": "searxng",
        "searxng_configured": True,
        "brave_configured": False,
    }
