"""Environment-backed configuration for the web tools process."""

from __future__ import annotations

from dataclasses import dataclass

from ..environment import environment


@dataclass(frozen=True, slots=True)
class WebToolsConfig:
    searxng_url: str = "http://127.0.0.1:8080"
    brave_api_key: str | None = None
    request_timeout_seconds: float = 10.0
    max_download_bytes: int = 2_000_000
    max_page_chars: int = 20_000
    dns_mode: str = "system"

    def __post_init__(self) -> None:
        if self.dns_mode not in {"system", "doh"}:
            raise ValueError("SLA_WEB_DNS_MODE must be system or doh")

    @classmethod
    def from_env(cls) -> WebToolsConfig:
        return cls(
            searxng_url=environment("SEARXNG_URL", "http://127.0.0.1:8080"),
            brave_api_key=environment("BRAVE_API_KEY") or None,
            request_timeout_seconds=float(environment("WEB_TIMEOUT_SECONDS", "10")),
            max_download_bytes=int(environment("WEB_MAX_DOWNLOAD_BYTES", "2000000")),
            max_page_chars=int(environment("WEB_MAX_PAGE_CHARS", "20000")),
            dns_mode=environment("WEB_DNS_MODE", "system"),
        )
