import io
import json
import socket
import threading
import time
from http.client import HTTPSConnection
from urllib.parse import parse_qs, urlsplit

import pytest

from self_learning_assistant.tools import dns
from self_learning_assistant.tools.config import WebToolsConfig
from self_learning_assistant.tools.domain import UnsafeRequestError
from self_learning_assistant.tools.fetch import _pinned_connection, build_web_fetcher
from self_learning_assistant.tools.security import validate_public_http_url


def answers(kind, *, private_ipv6=False):
    ip = (
        "93.184.216.34"
        if kind == 1
        else "fd00::1"
        if private_ipv6
        else "2001:4860:4860::8888"
    )
    return {"Status": 0, "Answer": [{"type": kind, "data": ip, "TTL": 60}]}


def test_public_dns_checks_both_address_families_and_caches_by_host(monkeypatch):
    requests = []

    def request(req, *, timeout):
        params = parse_qs(urlsplit(req.full_url).query)
        requests.append(params)
        assert params["edns_client_subnet"] == ["0.0.0.0/0"]
        return io.BytesIO(json.dumps(answers(int(params["type"][0]))).encode())

    monkeypatch.setattr(dns, "urlopen", request)
    resolver = dns.GoogleDohResolver()
    first = resolver("example.com", 443)
    second = resolver("example.com", 8443)
    assert len(requests) == 2
    assert {item[0] for item in first} == {socket.AF_INET, socket.AF_INET6}
    assert {item[4][1] for item in second} == {8443}
    assert validate_public_http_url("https://example.com/", resolver)


@pytest.mark.parametrize(
    "raw",
    [
        b"x" * 65537,
        b"not JSON",
        b"[]",
        b'{"Status": 2}',
        b'{"Status": 0, "TC": true}',
        b'{"Status": 0, "Answer": [{"type": 1, "data": "invalid"}]}',
        b'{"Status": 0, "Answer": [{"type": 1, "data": "::1"}]}',
    ],
)
def test_invalid_public_dns_results_fail_closed(monkeypatch, raw):
    monkeypatch.setattr(dns, "urlopen", lambda *args, **kw: io.BytesIO(raw))
    with pytest.raises(OSError):
        dns.GoogleDohResolver()("example.com", 443)


def test_public_dns_does_not_allow_private_ipv6_hidden_among_public_records(
    monkeypatch,
):
    def request(req, *, timeout):
        kind = int(parse_qs(urlsplit(req.full_url).query)["type"][0])
        return io.BytesIO(json.dumps(answers(kind, private_ipv6=True)).encode())

    monkeypatch.setattr(dns, "urlopen", request)
    resolver = dns.GoogleDohResolver()
    for url in ("https://example.com/", "http://127.0.0.1/", "http://10.0.0.1/"):
        with pytest.raises(UnsafeRequestError):
            validate_public_http_url(url, resolver)


def test_public_dns_retries_transient_network_errors(monkeypatch):
    attempts = 0

    def request(req, *, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("TLS connection dropped")
        return io.BytesIO(b'{"Status": 0, "Answer": []}')

    monkeypatch.setattr(dns, "urlopen", request)
    assert dns.GoogleDohResolver()("example.com", 443) == []
    assert attempts == 3


def test_public_dns_queries_both_families_concurrently(monkeypatch):
    barrier = threading.Barrier(2)

    def request(req, *, timeout):
        barrier.wait(timeout=1)
        kind = int(parse_qs(urlsplit(req.full_url).query)["type"][0])
        return io.BytesIO(json.dumps(answers(kind)).encode())

    monkeypatch.setattr(dns, "urlopen", request)
    records = dns.GoogleDohResolver(timeout=2)("example.com", 443)
    assert {record[0] for record in records} == {socket.AF_INET, socket.AF_INET6}


def test_public_dns_deadline_rejects_partial_answers_without_caching(monkeypatch):
    completed = threading.Event()

    def request(req, *, timeout):
        kind = int(parse_qs(urlsplit(req.full_url).query)["type"][0])
        if kind == 28:
            time.sleep(0.2)
            completed.set()
        return io.BytesIO(json.dumps(answers(kind)).encode())

    monkeypatch.setattr(dns, "urlopen", request)
    resolver = dns.GoogleDohResolver(timeout=0.02)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        resolver("example.com", 443)
    assert time.monotonic() - started < 0.15
    assert "example.com" not in resolver.cache
    # A late IPv6 result must not populate the cache after the caller timed out.
    assert completed.wait(1)
    assert "example.com" not in resolver.cache


def test_pinned_https_connects_to_checked_ip_and_keeps_certificate_hostname(
    monkeypatch,
):
    targets = []
    marker = object()

    def connect(target, *args, **kwargs):
        targets.append(target)
        return marker

    monkeypatch.setattr(socket, "create_connection", connect)
    records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
    connection = _pinned_connection(
        HTTPSConnection, "example.com", lambda h, p: records
    )
    assert connection._create_connection(("example.com", 443)) is marker
    assert targets == [("93.184.216.34", 443)]
    assert connection.host == "example.com"


def test_pinned_connection_rechecks_addresses_before_any_network_connection(
    monkeypatch,
):
    records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]
    connection = _pinned_connection(
        HTTPSConnection, "example.com", lambda h, p: records
    )
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: pytest.fail("Unsafe connection")
    )
    with pytest.raises(UnsafeRequestError):
        connection._create_connection(("example.com", 443))


def test_doh_fetcher_is_explicit_and_bad_configuration_is_rejected(monkeypatch):
    default = build_web_fetcher(WebToolsConfig())
    assert default._pin_addresses is False
    monkeypatch.setenv("SLA_WEB_DNS_MODE", "doh")
    configured = build_web_fetcher(WebToolsConfig.from_env())
    assert configured._pin_addresses is True
    assert isinstance(configured._resolver, dns.GoogleDohResolver)
    with pytest.raises(ValueError):
        WebToolsConfig(dns_mode="disable-security")
