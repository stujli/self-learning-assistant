"""Host-controlled source discovery, ranking and fair page budgets."""

from __future__ import annotations

import ipaddress
import re
import socket
import ssl
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit, urlunsplit

from ..domain import WebEvidence
from ..tools.domain import SearchHit, UnsafeRequestError
from ..tools.security import ensure_safe_query
from ..tools.web import WebResearchService
from .config import ProductError
from .store import tokens

STOP_WORDS = frozenset(
    "a an the is are was were be been being do does did can could may might must "
    "should would will shall of at in on to from for by with and or not no once "
    "when where why how what which who that this these those as it its every "
    "each again only than then under all any such into about whether".split()
)


def terms(text: str) -> set[str]:
    return set(tokens(text).split()) - STOP_WORDS


def hostname(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().rstrip(".")


def source_host(value: str) -> str | None:
    """Suggestions are search hints, never URLs or a grant of trust."""
    try:
        value = value.rstrip(".").encode("idna").decode().lower()
        if len(value) > 253 or "." not in value:
            return None
        if any(
            not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
            for label in value.split(".")
        ) or value.endswith((".local", ".localhost")):
            return None
        try:
            ipaddress.ip_address(value)
        except ValueError:
            return value
    except (ValueError, UnicodeError):
        pass
    return None


def canonical_url(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Invalid public page URL")
    host = parsed.hostname.encode("idna").decode().lower().rstrip(".")
    authority = f"[{host}]" if ":" in host else host
    if parsed.port and parsed.port != (443 if parsed.scheme == "https" else 80):
        authority += f":{parsed.port}"
    # Keep path and query semantics; fragments do not change the HTTP resource.
    return urlunsplit(
        (parsed.scheme.lower(), authority, parsed.path or "/", parsed.query, "")
    )


def failure(exc: Exception) -> dict[str, Any]:
    """Use typed errors/causes; do not infer an HTTP status from arbitrary text."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and len(chain) < 6:
        chain.append(current)
        current = current.__cause__
    status = next((item.code for item in chain if isinstance(item, HTTPError)), None)
    if status is not None:
        return {
            "error": type(exc).__name__,
            "error_kind": "http",
            "status_code": status,
        }
    for item in chain:
        reason = item.reason if isinstance(item, URLError) else item
        if isinstance(reason, ssl.SSLError):
            return {"error": type(exc).__name__, "error_kind": "tls"}
        if isinstance(reason, socket.gaierror):
            return {"error": type(exc).__name__, "error_kind": "dns"}
        if isinstance(reason, TimeoutError):
            return {"error": type(exc).__name__, "error_kind": "timeout"}
    kind = (
        "unsafe_address"
        if isinstance(exc, UnsafeRequestError)
        else "deadline"
        if isinstance(exc, ProductError) and "deadline" in str(exc).lower()
        else "unknown"
    )
    return {"error": type(exc).__name__, "error_kind": kind}


@dataclass
class Candidate:
    url: str
    hit: SearchHit
    provider_rank: int
    order: int
    purposes: set[str] = field(default_factory=set)
    query_ids: set[int] = field(default_factory=set)
    coverage: int = 0
    score: int = 0
    relevant: bool = False
    matched_terms: set[str] = field(default_factory=set)
    authority: str = "general"
    state: str = "queued"
    skip_reason: str = ""

    def priority(self) -> tuple[int, int, int, int, int]:
        return (
            int(self.relevant),
            {"trusted": 2, "preferred": 1, "general": 0}[self.authority],
            self.score,
            -self.provider_rank,
            -self.order,
        )

    def diagnostic(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.hit.title,
            "purposes": sorted(self.purposes),
            "query_ids": sorted(self.query_ids),
            "authority": self.authority,
            "term_overlap": self.coverage,
            "relevance_score": self.score,
            "relevant": self.relevant,
            "matched_terms": sorted(self.matched_terms),
            "provider_rank": self.provider_rank,
            "state": self.state,
            "skip_reason": self.skip_reason,
        }


class Retrieval:
    def __init__(
        self,
        web: WebResearchService,
        claim: str,
        plan: dict[str, Any],
        trusted_hosts: set[str],
        deadline: float,
        bounded: Callable[..., Any],
    ) -> None:
        self.web, self.claim, self.deadline, self.bounded = (
            web,
            claim,
            deadline,
            bounded,
        )
        self.plan = plan
        self.trusted = trusted_hosts
        suggested = [
            host for value in plan["preferred_hosts"] if (host := source_host(value))
        ]
        self.preferred = list(dict.fromkeys([*sorted(trusted_hosts), *suggested]))
        self.topic = terms(claim)
        self.minimum_overlap = min(len(self.topic), max(2, (len(self.topic) + 1) // 2))
        self.events: list[dict[str, Any]] = []
        self.candidates: dict[str, Candidate] = {}
        self.sources: dict[str, WebEvidence] = {}
        self.hashes: set[str] = set()
        self.searched: set[str] = set()
        self.targeted: set[str] = set()
        self.attempts: Counter[str] = Counter()
        self.failures: Counter[str] = Counter()
        self.lanes: Counter[str] = Counter()
        self.source_purposes: dict[str, set[str]] = {}
        self.source_relevance: dict[str, bool] = {}
        self.counter_search = False
        self.search_count = 0
        self.fetch_count = 0

    def _target(self, query: str, host: str) -> str | None:
        combined = f"site:{host} {query}"
        return combined if len(combined) <= 500 else None

    def search(self, purpose: str, query: str, *, target: str | None = None) -> None:
        query = ensure_safe_query(query)
        key = " ".join(query.casefold().split())
        if (
            key in self.searched
            or self.search_count >= 4
            or time.monotonic() >= self.deadline
        ):
            return
        self.searched.add(key)
        if target:
            self.targeted.add(target)
        self.search_count += 1
        event: dict[str, Any] = {
            "phase": "search",
            "query_id": self.search_count,
            "purpose": purpose,
            "query": query,
            "target_host": target,
        }
        try:
            response = self.bounded(
                lambda: self.web.search(query, max_results=8),
                min(self.deadline, time.monotonic() + 20),
            )
            event.update(
                ok=True,
                results=len(response.results),
                backend=response.backend,
                fallback_used=response.fallback_used,
            )
            self.counter_search |= purpose == "counter"
            for rank, hit in enumerate(response.results, 1):
                try:
                    url = canonical_url(hit.url)
                except (ValueError, UnicodeError):
                    self.events.append(
                        {
                            "phase": "skip",
                            "reason": "invalid_url",
                            "query_id": self.search_count,
                        }
                    )
                    continue
                candidate = self.candidates.get(url)
                if candidate is None:
                    candidate = Candidate(url, hit, rank, len(self.candidates))
                    self.candidates[url] = candidate
                candidate.purposes.add(purpose)
                candidate.query_ids.add(self.search_count)
                # Keep the strongest metadata for duplicate results, plus best rank.
                if self._score(hit) > self._score(candidate.hit):
                    candidate.hit = hit
                candidate.provider_rank = min(candidate.provider_rank, rank)
                candidate.coverage, candidate.score = self._score(candidate.hit)
                candidate.relevant = candidate.coverage >= self.minimum_overlap
                candidate.matched_terms = (
                    terms(candidate.hit.title)
                    | terms(candidate.hit.snippet)
                    | terms(unquote(urlsplit(candidate.url).path))
                ) & self.topic
                host = hostname(url)
                candidate.authority = (
                    "trusted"
                    if host in self.trusted
                    else "preferred"
                    if host in self.preferred
                    else "general"
                )
        except Exception as exc:
            event.update(ok=False, **failure(exc))
        self.events.append(event)

    def _score(self, hit: SearchHit) -> tuple[int, int]:
        title = terms(hit.title) & self.topic
        snippet = terms(hit.snippet) & self.topic
        path = terms(unquote(urlsplit(hit.url).path)) & self.topic
        return len(title | snippet | path), 3 * len(title) + 2 * len(snippet) + len(
            path
        )

    def ranked(self) -> list[Candidate]:
        return sorted(
            self.candidates.values(), key=lambda item: item.priority(), reverse=True
        )

    def fetch(self, lane: str) -> bool:
        if self.fetch_count >= 8 or time.monotonic() >= self.deadline:
            return False
        for candidate in self.ranked():
            if candidate.state != "queued" or (
                lane != "supplement" and lane not in candidate.purposes
            ):
                continue
            host = hostname(candidate.url)
            cap = 3 if candidate.authority != "general" else 2
            skip = (
                "host_failures"
                if self.failures[host] >= 2
                else "host_budget"
                if self.attempts[host] >= cap
                else ""
            )
            if skip:
                candidate.state, candidate.skip_reason = "skipped", skip
                continue
            self.fetch_count += 1
            self.attempts[host] += 1
            self.lanes[lane] += 1
            limit = 50_000 if candidate.authority != "general" else 20_000
            event: dict[str, Any] = {
                "phase": "fetch",
                "url": candidate.url,
                "lane": lane,
                "attempt": self.fetch_count,
                "max_chars": limit,
            }
            try:
                page = self.bounded(
                    partial(self.web.fetch, candidate.url, max_chars=limit),
                    min(self.deadline, time.monotonic() + 20),
                )
                self.failures[host] = 0
                event.update(
                    ok=True,
                    final_url=page.final_url,
                    truncated=page.truncated,
                    characters=len(page.text),
                )
                candidate.state = "fetched"
                if page.content_hash in self.hashes:
                    candidate.skip_reason = "duplicate_content"
                else:
                    self.hashes.add(page.content_hash)
                    identifier = f"S{len(self.sources) + 1}"
                    self.sources[identifier] = WebEvidence(
                        page.title or candidate.hit.title,
                        page.final_url,
                        "",
                        page.text,
                        page.retrieved_at,
                        page.content_hash,
                        page.truncated,
                    )
                    self.source_purposes[identifier] = candidate.purposes.copy()
                    self.source_relevance[identifier] = candidate.relevant
                    event["source_id"] = identifier
            except Exception as exc:
                self.failures[host] += 1
                candidate.state = "failed"
                event.update(ok=False, **failure(exc))
            self.events.append(event)
            return True
        return False

    def _usable(self, source: WebEvidence) -> bool:
        return (
            bool(source.text.strip())
            and not source.truncated
            and len(terms(source.text) & self.topic) >= self.minimum_overlap
        )

    def _supplement_query(self) -> tuple[str, str, str | None] | None:
        has_authority = any(
            hostname(source.url) in self.preferred
            and self.source_relevance[identifier]
            and self._usable(source)
            for identifier, source in self.sources.items()
        )
        has_counter = any(
            "counter" in self.source_purposes[identifier] and self._usable(source)
            for identifier, source in self.sources.items()
        )
        options: list[tuple[str, str, str | None]] = []
        if self.preferred and not has_authority:
            for host in self.preferred:
                if host not in self.targeted:
                    query = self._target(self.plan["neutral"][0], host)
                    if query:
                        options.append(("neutral", query, host))
            for query in self.plan["neutral"][1:]:
                targeted = self._target(query, self.preferred[0])
                if targeted:
                    options.append(("neutral", targeted, self.preferred[0]))
        if not has_counter:
            options.extend(
                ("counter", query, None) for query in self.plan["counter"][1:]
            )
        options.extend(("neutral", query, None) for query in self.plan["neutral"])
        options.extend(("counter", query, None) for query in self.plan["counter"])
        return next(
            (
                item
                for item in options
                if " ".join(item[1].casefold().split()) not in self.searched
            ),
            None,
        )

    def run(self) -> None:
        neutral, counter = self.plan["neutral"], self.plan["counter"]
        first = self._target(neutral[0], self.preferred[0]) if self.preferred else None
        self.search(
            "neutral", first or neutral[0], target=self.preferred[0] if first else None
        )
        # An unrestricted query can find independent sources even with authority hints.
        self.search("neutral", neutral[0] if first else neutral[-1])
        counter_query = counter[0]
        if " ".join(counter_query.casefold().split()) in self.searched:
            expanded = counter_query + " counterexamples exceptions"
            counter_query = expanded if len(expanded) <= 500 else counter_query
        self.search("counter", counter_query)
        for lane in ("neutral", "counter", "neutral", "counter", "neutral", "neutral"):
            self.fetch(lane)
        supplement = self._supplement_query()
        if supplement:
            self.search(supplement[0], supplement[1], target=supplement[2])
        # Supplementary search can refill a previously empty lane. Honor its
        # unused quota before transferring the remainder to global ranking.
        for lane in ("counter", "neutral", "counter", "neutral", "neutral", "neutral"):
            if self.lanes[lane] < (2 if lane == "counter" else 4):
                self.fetch(lane)
        while self.fetch("supplement"):
            pass
        for candidate in self.candidates.values():
            if candidate.state == "queued":
                candidate.state = "skipped"
                candidate.skip_reason = (
                    "deadline" if time.monotonic() >= self.deadline else "fetch_budget"
                )

    def passages(self) -> dict[str, tuple[str, str]]:
        by_source: dict[str, list[str]] = {}
        for identifier, source in self.sources.items():
            spans = [
                m.group(0).strip()
                # HTML text often wraps a sentence across lines. Keep those
                # lines and decimal/version dots in the exact captured span.
                for m in re.finditer(
                    r"[\s\S]+?(?:[。！？]|[.!?](?=\s|$)|\Z)", source.text
                )
            ]
            spans = list(
                dict.fromkeys(span for span in spans if 12 <= len(span) <= 1000)
            )
            spans.sort(key=lambda span: len(terms(span) & self.topic), reverse=True)
            by_source[identifier] = spans
        chosen: list[tuple[str, str]] = [
            (identifier, spans[0]) for identifier, spans in by_source.items() if spans
        ]
        extra = [
            (identifier, span)
            for identifier, spans in by_source.items()
            for span in spans[1:]
        ]
        extra.sort(key=lambda pair: len(terms(pair[1]) & self.topic), reverse=True)
        chosen.extend(extra[: 32 - len(chosen)])
        counts: Counter[str] = Counter()
        passages = {}
        for identifier, span in chosen:
            counts[identifier] += 1
            passages[f"{identifier}P{counts[identifier]}"] = (identifier, span)
        return passages

    def diagnostic(self) -> dict[str, Any]:
        return {
            "search_limit": 4,
            "fetch_limit": 8,
            "searches_used": self.search_count,
            "fetches_used": self.fetch_count,
            "allocation": {"neutral": 4, "counter": 2, "supplement": 2},
            "attempts_by_lane": dict(self.lanes),
            "minimum_term_overlap": self.minimum_overlap,
            "preferred_hosts": self.preferred,
            "trusted_hosts": sorted(self.trusted),
            "model_suggestions": self.plan["preferred_hosts"],
            "candidates": [
                {"rank": rank, **item.diagnostic()}
                for rank, item in enumerate(self.ranked(), 1)
            ],
        }
