from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from ..model import ChatModel
from ..prompts import (
    RESEARCH_SOURCE_POLICY,
    research_query_prompt,
    research_report_prompt,
)
from ..tools.config import WebToolsConfig
from ..tools.security import ensure_safe_query
from ..tools.web import WebResearchService, build_web_research_service
from ..verification import EvidenceReportGate
from .config import ProductError, Settings
from .model import structured
from .retrieval import Retrieval

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "neutral": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 2,
        },
        "counter": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 2,
        },
        "preferred_hosts": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 2,
        },
    },
    "required": ["neutral", "counter", "preferred_hosts"],
    "additionalProperties": False,
}
REPORT: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "claim": {"type": "string"},
        "verdict": {
            "type": "string",
            "enum": ["supported", "refuted", "insufficient", "conflicting"],
        },
        "reason": {"type": "string"},
        "conditions": {"type": "string"},
        "unresolved": {"type": "array", "items": {"type": "string"}},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "passage_id": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": ["supports", "refutes", "irrelevant"],
                    },
                },
                "required": ["source_id", "passage_id", "relation"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["claim", "verdict", "reason", "conditions", "unresolved", "evidence"],
}


def bounded[T](
    operation: Callable[[], T],
    deadline: float,
    *,
    cancel: Callable[[], None] | None = None,
    stopped: Callable[[], bool] = lambda: False,
) -> T:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ProductError("Operation deadline exceeded")
    results: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            results.put((True, operation()))
        except Exception as exc:
            results.put((False, exc))

    threading.Thread(target=run, daemon=True).start()
    while True:
        if stopped():
            if cancel is not None:
                cancel()
            raise InterruptedError("Operation paused")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if cancel is not None:
                cancel()
            raise ProductError("Operation deadline exceeded")
        try:
            success, value = results.get(timeout=min(remaining, 0.25))
            break
        except queue.Empty:
            continue
    if not success:
        raise value
    return value  # type: ignore[no-any-return]


def web_service(settings: Settings) -> WebResearchService:
    options = settings.effective()["search"]
    environment = WebToolsConfig.from_env()
    return build_web_research_service(
        WebToolsConfig(
            searxng_url=options["searxng_url"],
            brave_api_key=environment.brave_api_key,
            request_timeout_seconds=environment.request_timeout_seconds,
            dns_mode=environment.dns_mode,
        )
    )


class Researcher:
    """Finite host-controlled research; model output never supplies page contents."""

    def __init__(
        self,
        model: ChatModel,
        web: WebResearchService,
        *,
        trusted_hosts: list[str] | None = None,
        timeout: float = 300,
        reserve: float = 60,
    ) -> None:
        if timeout <= 0 or reserve <= 0 or reserve >= timeout:
            raise ProductError(
                "Research timeout must exceed its positive report reserve"
            )
        self.model, self.web = model, web
        self.timeout, self.reserve = timeout, reserve
        self.gate = EvidenceReportGate(trusted_hosts=trusted_hosts or [])

    def verify(self, claim: str) -> dict[str, Any]:
        claim = ensure_safe_query(claim)
        started = time.monotonic()
        end = started + self.timeout
        retrieval_end = end - self.reserve
        retrieval: Retrieval | None = None
        sources = {}
        passages: dict[str, tuple[str, str]] = {}
        counter = False
        try:
            plan = bounded(
                lambda: structured(
                    self.model,
                    [
                        {
                            "role": "user",
                            "content": research_query_prompt(claim),
                        }
                    ],
                    PLAN_SCHEMA,
                    max_new_tokens=768,
                ),
                retrieval_end,
                cancel=getattr(self.model, "cancel_current", None),
            )
            retrieval = Retrieval(
                self.web, claim, plan, self.gate.trusted_hosts, retrieval_end, bounded
            )
            sources = retrieval.sources
            retrieval.run()
            passages = retrieval.passages()
            counter = retrieval.counter_search
            payload = {
                "claim": claim,
                "counter_search_completed": counter,
                "trusted_hosts": sorted(self.gate.trusted_hosts),
                "source_policy": RESEARCH_SOURCE_POLICY,
                "sources": [
                    {
                        "source_id": identifier,
                        "title": source.title,
                        "url": source.url,
                        "truncated": source.truncated,
                    }
                    for identifier, source in sources.items()
                ],
                "passages": [
                    {"passage_id": identifier, "source_id": source_id, "text": text}
                    for identifier, (source_id, text) in passages.items()
                ],
            }
            report = bounded(
                lambda: structured(
                    self.model,
                    [
                        {
                            "role": "user",
                            "content": research_report_prompt(payload),
                        }
                    ],
                    REPORT,
                    max_new_tokens=1536,
                ),
                end,
                cancel=getattr(self.model, "cancel_current", None),
            )
            entries = report.get("evidence")
            if not isinstance(entries, list):
                raise ProductError("Invalid report evidence")
            citations = []
            for entry in entries:
                if not isinstance(entry, dict) or set(entry) != {
                    "source_id",
                    "passage_id",
                    "relation",
                }:
                    raise ProductError("Invalid passage reference")
                passage = passages.get(entry["passage_id"])
                if not passage or passage[0] != entry["source_id"]:
                    raise ProductError("Unknown or mismatched passage reference")
                citations.append(
                    {
                        "source_id": entry["source_id"],
                        "quote": passage[1],
                        "relation": entry["relation"],
                    }
                )
            report["evidence"] = citations
            result = self.gate.verify(
                claim, report, sources, counter_search_completed=counter
            )
            result["execution"] = {
                "report_origin": "model",
                "failed": False,
                "seconds": time.monotonic() - started,
            }
        except Exception as exc:
            result = {
                "accepted": False,
                "verdict": "insufficient",
                "reason": str(exc)[:500],
                "execution": {
                    "report_origin": "host_fallback",
                    "failed": True,
                    "seconds": time.monotonic() - started,
                },
            }
        result["captured_sources"] = {
            identifier: source.to_dict() for identifier, source in sources.items()
        }
        result["events"] = retrieval.events if retrieval else []
        result["retrieval"] = retrieval.diagnostic() if retrieval else {}
        return result
