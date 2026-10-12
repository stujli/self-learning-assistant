"""Input validation and SSRF protection for model-callable web tools."""

from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Callable, Sequence
from typing import cast
from urllib.parse import urlsplit

from self_learning_assistant.tools.domain import UnsafeRequestError

type SocketAddress = tuple[str, int] | tuple[str, int, int, int]
type AddressRecord = tuple[int, int, int, str, SocketAddress]
type Resolver = Callable[[str, int | None], Sequence[AddressRecord]]

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.I),
    re.compile(r"\b(?:password|passwd|api[_ -]?key|access[_ -]?token)\s*[:=]", re.I),
    re.compile(r"\b(?:sk|ghp|github_pat)_[A-Za-z0-9_-]{16,}\b"),
)


def ensure_safe_query(query: str) -> str:
    cleaned = query.strip()
    if not cleaned:
        raise UnsafeRequestError("搜索词不能为空")
    if len(cleaned) > 500:
        raise UnsafeRequestError("搜索词不能超过 500 个字符")
    if any(pattern.search(cleaned) for pattern in _SECRET_PATTERNS):
        raise UnsafeRequestError("搜索词疑似包含凭据或私钥，已拒绝发送")
    return cleaned


def validate_search_endpoint(url: str) -> str:
    """Validate a trusted, configured endpoint; localhost is allowed here."""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafeRequestError("搜索端点必须是有效的 HTTP(S) URL")
    if parsed.username or parsed.password:
        raise UnsafeRequestError("搜索端点不能在 URL 中包含凭据")
    return url.rstrip("/")


def default_resolver(hostname: str, port: int | None) -> Sequence[AddressRecord]:
    records = socket.getaddrinfo(hostname, port)
    return [
        (int(family), int(kind), protocol, canonname, cast(SocketAddress, address))
        for family, kind, protocol, canonname, address in records
    ]


def validate_public_http_url(url: str, resolver: Resolver = default_resolver) -> str:
    """Reject URLs that could reach local, private, special, or non-HTTP targets."""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafeRequestError("只允许完整的公共 HTTP(S) URL")
    if parsed.username or parsed.password:
        raise UnsafeRequestError("URL 不能包含用户名或密码")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
        raise UnsafeRequestError("不允许访问本机或局域网主机")

    try:
        addresses = _resolve_addresses(hostname, parsed.port, resolver)
    except (OSError, ValueError) as exc:
        raise UnsafeRequestError("无法安全解析目标主机") from exc
    if not addresses:
        raise UnsafeRequestError("目标主机没有可用地址")
    if any(not _is_public_address(address) for address in addresses):
        raise UnsafeRequestError("不允许访问内网、回环或特殊用途地址")
    return url


def _resolve_addresses(
    hostname: str, port: int | None, resolver: Resolver
) -> set[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        return {ipaddress.ip_address(hostname)}
    except ValueError:
        records = resolver(hostname, port)
        return {ipaddress.ip_address(str(record[4][0])) for record in records}


def _is_public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return address.is_global and not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )
