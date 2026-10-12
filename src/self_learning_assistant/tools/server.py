"""stdio MCP server exposing the read-only Self-Learning Assistant web tools."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

from self_learning_assistant.prompts import (
    FETCH_TOOL_DESCRIPTION,
    MCP_INSTRUCTIONS,
    SEARCH_TOOL_DESCRIPTION,
    STATUS_TOOL_DESCRIPTION,
)
from self_learning_assistant.tools.config import WebToolsConfig
from self_learning_assistant.tools.fetch import SafeWebFetcher, build_web_fetcher
from self_learning_assistant.tools.search import (
    BraveProvider,
    SearchRouter,
    SearxngProvider,
)

_READ_ONLY_OPEN_WORLD = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=True,
)


def build_server(
    config: WebToolsConfig | None = None,
    *,
    router: SearchRouter | None = None,
    fetcher: SafeWebFetcher | None = None,
) -> MCPServer[None]:
    settings = config or WebToolsConfig.from_env()
    if router is None:
        searxng = SearxngProvider(
            settings.searxng_url, timeout=settings.request_timeout_seconds
        )
        brave = (
            BraveProvider(
                settings.brave_api_key, timeout=settings.request_timeout_seconds
            )
            if settings.brave_api_key
            else None
        )
        router = SearchRouter(searxng, brave)
    if fetcher is None:
        fetcher = build_web_fetcher(settings)

    server: MCPServer[None] = MCPServer(
        "self-learning-assistant-web",
        title="Self-Learning Assistant Web Tools",
        description="Read-only public web search and bounded page retrieval.",
        instructions=MCP_INSTRUCTIONS,
        version="0.1.0",
    )

    @server.tool(
        title="搜索网页",
        description=(SEARCH_TOOL_DESCRIPTION),
        annotations=_READ_ONLY_OPEN_WORLD,
        structured_output=True,
    )
    async def search_web(
        query: str,
        backend: str = "auto",
        max_results: int = 8,
        language: str = "all",
        time_range: str | None = None,
    ) -> dict[str, Any]:
        return router.search(
            query,
            backend=backend,
            max_results=max_results,
            language=language,
            time_range=time_range,
        ).to_dict()

    @server.tool(
        title="读取网页正文",
        description=(FETCH_TOOL_DESCRIPTION),
        annotations=_READ_ONLY_OPEN_WORLD,
        structured_output=True,
    )
    async def fetch_web_page(
        url: str,
        max_chars: int = 20_000,
    ) -> dict[str, Any]:
        return fetcher.fetch(url, max_chars=max_chars).to_dict()

    @server.tool(
        title="查看搜索后端配置",
        description=STATUS_TOOL_DESCRIPTION,
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
        structured_output=True,
    )
    async def web_tools_status() -> dict[str, Any]:
        return router.status()

    return server


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
