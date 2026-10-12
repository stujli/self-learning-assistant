"""Optional public DNS lookup for hosts behind a transparent fake-IP proxy."""

from __future__ import annotations

import ipaddress
import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .security import AddressRecord


class GoogleDohResolver:
    """Resolve A and AAAA over HTTPS; callers still reject non-public addresses."""

    def __init__(self, *, timeout: float = 10):
        self.timeout = timeout
        self.cache: dict[str, tuple[float, list[tuple[int, str]]]] = {}

    def __call__(self, hostname: str, port: int | None) -> list[AddressRecord]:
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            addresses = self._lookup(hostname.encode("idna").decode())
        else:
            addresses = [
                (
                    socket.AF_INET if address.version == 4 else socket.AF_INET6,
                    str(address),
                )
            ]
        return [
            (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port or 0))
            for family, ip in addresses
        ]

    def _lookup(self, hostname: str) -> list[tuple[int, str]]:
        cached = self.cache.get(hostname)
        if cached and cached[0] > time.monotonic():
            return cached[1]
        addresses: list[tuple[int, str]] = []
        ttl = 60
        deadline = time.monotonic() + self.timeout
        executor = ThreadPoolExecutor(max_workers=2)
        try:
            futures = [
                executor.submit(self._query, hostname, kind, family, deadline)
                for kind, family in ((1, socket.AF_INET), (28, socket.AF_INET6))
            ]
            # Both families must succeed. Never cache or validate a partial
            # answer that could hide a private address in the other family.
            for future in futures:
                records, lifetime = future.result(
                    timeout=max(0, deadline - time.monotonic())
                )
                addresses.extend(records)
                ttl = min(ttl, lifetime)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        if len(self.cache) >= 256:
            self.cache.clear()
        self.cache[hostname] = (time.monotonic() + ttl, addresses)
        return addresses

    def _query(
        self, hostname: str, kind: int, family: int, deadline: float
    ) -> tuple[list[tuple[int, str]], int]:
        query = urlencode(
            {"name": hostname, "type": kind, "edns_client_subnet": "0.0.0.0/0"}
        )
        request = Request(
            "https://dns.google/resolve?" + query,
            headers={"Accept": "application/json"},
        )
        for attempt in range(3):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Public DNS deadline exceeded")
            try:
                with urlopen(request, timeout=min(self.timeout, remaining)) as response:
                    raw = response.read(65537)
                break
            except OSError:
                if attempt == 2:
                    raise
        if len(raw) > 65536:
            raise OSError("Public DNS response exceeds limit")
        records: list[tuple[int, str]] = []
        ttl = 60
        try:
            payload = json.loads(raw)
            if payload.get("Status") != 0 or payload.get("TC"):
                raise OSError("Public DNS lookup failed")
            for item in payload.get("Answer", []):
                if item.get("type") == kind:
                    ip = ipaddress.ip_address(item["data"])
                    if ip.version != (4 if kind == 1 else 6):
                        raise ValueError("Incorrect DNS record family")
                    records.append((family, str(ip)))
                    ttl = min(ttl, max(0, int(item.get("TTL", 0))))
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            raise OSError("Invalid public DNS response") from exc
        return records, ttl
