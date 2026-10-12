"""Bounded, non-JavaScript page fetching for evidence inspection."""

from __future__ import annotations

import hashlib
import re
import socket
import ssl
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from http.client import HTTPConnection, HTTPMessage, HTTPSConnection
from typing import IO, Any
from urllib.error import HTTPError
from urllib.request import (
    HTTPHandler,
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from self_learning_assistant.tools.config import WebToolsConfig
from self_learning_assistant.tools.dns import GoogleDohResolver
from self_learning_assistant.tools.domain import FetchedPage, WebToolError
from self_learning_assistant.tools.security import (
    Resolver,
    default_resolver,
    validate_public_http_url,
)


@dataclass(frozen=True, slots=True)
class Download:
    body: bytes
    final_url: str
    content_type: str
    charset: str | None = None


Downloader = Callable[[str], Download]


def _pinned_connection(
    kind: type[HTTPConnection], host: str, resolver: Resolver, **options: Any
) -> HTTPConnection:
    connection = kind(host, **options)

    def connect(address: tuple[str, int], *args: Any, **kwargs: Any) -> socket.socket:
        hostname, port = address
        label = f"[{hostname}]" if ":" in hostname else hostname
        records = resolver(hostname, port)
        validate_public_http_url(
            f"https://{label}:{port}/", lambda _host, _port: records
        )
        last_error: OSError | None = None
        for record in records:
            try:
                return socket.create_connection(
                    (str(record[4][0]), port), *args, **kwargs
                )
            except OSError as exc:
                last_error = exc
        raise OSError("Cannot connect to validated public address") from last_error

    # HTTPSConnection keeps the original hostname for SNI and certificate
    # checking; only the underlying TCP destination is pinned to public DNS.
    connection._create_connection = connect  # type: ignore[attr-defined]
    return connection


class _PinnedHTTPHandler(HTTPHandler):
    def __init__(self, resolver: Resolver):
        super().__init__()
        self.resolver = resolver

    def http_open(self, req: Request) -> Any:
        return self.do_open(
            lambda host, **opts: _pinned_connection(
                HTTPConnection, host, self.resolver, **opts
            ),
            req,
        )


class _PinnedHTTPSHandler(HTTPSHandler):
    def __init__(self, resolver: Resolver):
        self.ssl_context = ssl.create_default_context()
        super().__init__(context=self.ssl_context)
        self.resolver = resolver

    def https_open(self, req: Request) -> Any:
        return self.do_open(
            lambda host, **opts: _pinned_connection(
                HTTPSConnection, host, self.resolver, **opts
            ),
            req,
            context=self.ssl_context,
        )


class _SafeRedirectHandler(HTTPRedirectHandler):
    def __init__(self, resolver: Resolver) -> None:
        self._resolver = resolver

    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Request | None:
        validate_public_http_url(newurl, self._resolver)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class SafeWebFetcher:
    _ALLOWED_TYPES = {"text/html", "text/plain", "application/xhtml+xml"}

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        max_download_bytes: int = 2_000_000,
        max_page_chars: int = 20_000,
        resolver: Resolver = default_resolver,
        downloader: Downloader | None = None,
        pin_addresses: bool = False,
    ) -> None:
        self._timeout = timeout
        self._max_download_bytes = max_download_bytes
        self._max_page_chars = max_page_chars
        self._resolver = resolver
        self._downloader = downloader or self._download
        self._pin_addresses = pin_addresses

    def fetch(self, url: str, *, max_chars: int | None = None) -> FetchedPage:
        validate_public_http_url(url, self._resolver)
        limit = max_chars if max_chars is not None else self._max_page_chars
        if not 1_000 <= limit <= 50_000:
            raise WebToolError("max_chars 必须在 1000 到 50000 之间")
        download = self._downloader(url)
        validate_public_http_url(download.final_url, self._resolver)
        media_type = download.content_type.partition(";")[0].strip().lower()
        if media_type not in self._ALLOWED_TYPES:
            raise WebToolError(f"不支持的页面类型：{media_type or 'unknown'}")
        encoding = download.charset or "utf-8"
        text = download.body.decode(encoding, errors="replace")
        if media_type in {"text/html", "application/xhtml+xml"}:
            title, text = _extract_html(text)
        else:
            title = ""
            text = _normalize(text)
        truncated = len(text) > limit
        text = text[:limit]
        return FetchedPage(
            url=url,
            final_url=download.final_url,
            title=title,
            text=text,
            retrieved_at=datetime.now(UTC).isoformat(),
            content_hash="sha256:" + hashlib.sha256(download.body).hexdigest(),
            content_type=media_type,
            truncated=truncated,
        )

    def _download(self, url: str) -> Download:
        handlers: list[Any] = [_SafeRedirectHandler(self._resolver)]
        if self._pin_addresses:
            handlers.extend(
                [
                    ProxyHandler({}),
                    _PinnedHTTPHandler(self._resolver),
                    _PinnedHTTPSHandler(self._resolver),
                ]
            )
        opener = build_opener(*handlers)
        request = Request(
            url,
            headers={
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
                "User-Agent": "self-learning-assistant/0.3.0",
            },
        )
        try:
            with opener.open(request, timeout=self._timeout) as response:
                length = response.headers.get("Content-Length")
                if length and int(length) > self._max_download_bytes:
                    raise WebToolError("页面超过下载大小限制")
                body = response.read(self._max_download_bytes + 1)
                if len(body) > self._max_download_bytes:
                    raise WebToolError("页面超过下载大小限制")
                return Download(
                    body=body,
                    final_url=response.geturl(),
                    content_type=response.headers.get_content_type(),
                    charset=response.headers.get_content_charset(),
                )
        except HTTPError as exc:
            raise WebToolError(f"页面请求失败：HTTP {exc.code}") from exc


def build_web_fetcher(settings: WebToolsConfig) -> SafeWebFetcher:
    return SafeWebFetcher(
        timeout=settings.request_timeout_seconds,
        max_download_bytes=settings.max_download_bytes,
        max_page_chars=settings.max_page_chars,
        resolver=GoogleDohResolver(timeout=settings.request_timeout_seconds)
        if settings.dns_mode == "doh"
        else default_resolver,
        pin_addresses=settings.dns_mode == "doh",
    )


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg", "nav"}:
            self._ignored_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "br", "li", "article", "section", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "nav"} and self._ignored_depth:
            self._ignored_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if not self._ignored_depth and not self._in_title:
            self.parts.append(data)


def _extract_html(html: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(html)
    return _normalize(" ".join(parser.title_parts)), _normalize(" ".join(parser.parts))


def _normalize(text: str) -> str:
    lines = (re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)
