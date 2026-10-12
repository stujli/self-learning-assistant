"""Check cited research reports; relevance alone never authorizes learning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import urlsplit

from .domain import WebEvidence

VERDICTS = ("supported", "refuted", "insufficient", "conflicting")
REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "claim": {"type": "string"},
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "reason": {"type": "string"},
        "conditions": {"type": "string"},
        "unresolved": {"type": "array", "items": {"type": "string"}},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "source_id": {"type": "string"},
                    "quote": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": ["supports", "refutes", "irrelevant"],
                    },
                },
                "required": ["source_id", "quote", "relation"],
            },
        },
    },
    "required": ["claim", "verdict", "reason", "conditions", "unresolved", "evidence"],
}


def _normalized(text: str) -> str:
    return " ".join(text.split())


class EvidenceReportGate:
    """Validate provenance and reject incomplete or internally inconsistent reports.

    This checks citations, not semantic truth. The model's evidence-relation
    judgments must be evaluated independently before unattended training.
    """

    def __init__(self, *, trusted_hosts: Sequence[str] = ()) -> None:
        self.trusted_hosts = {host.lower().strip(".") for host in trusted_hosts}

    def verify(
        self,
        claim: str,
        report: Mapping[str, object],
        sources: Mapping[str, WebEvidence],
        *,
        counter_search_completed: bool,
    ) -> dict[str, object]:
        errors: list[str] = []
        if set(report) != set(REPORT_SCHEMA["required"]):
            errors.append("invalid report fields")
        verdict = report.get("verdict")
        if verdict not in VERDICTS:
            errors.append("invalid verdict")
        if report.get("claim") != claim:
            errors.append("claim changed")
        reason, conditions = report.get("reason"), report.get("conditions")
        if not isinstance(reason, str) or not reason.strip():
            errors.append("missing reason")
        if not isinstance(conditions, str) or not conditions.strip():
            errors.append("missing conditions")
        unresolved = report.get("unresolved")
        if not isinstance(unresolved, list) or unresolved:
            errors.append("unresolved issues")
        if not counter_search_completed:
            errors.append("counter-evidence search not completed")

        citations: list[dict[str, object]] = []
        supporting: list[WebEvidence] = []
        refuting = False
        entries = report.get("evidence")
        if not isinstance(entries, list) or not entries:
            errors.append("missing evidence")
            entries = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {
                "source_id",
                "quote",
                "relation",
            }:
                errors.append("invalid evidence entry")
                continue
            source_id, quote, relation = (
                entry.get("source_id"),
                entry.get("quote"),
                entry.get("relation"),
            )
            source = sources.get(source_id) if isinstance(source_id, str) else None
            if (
                source is None
                or not source.text.strip()
                or not source.content_hash.startswith("sha256:")
                or source.truncated
                or not isinstance(quote, str)
                or len(quote.strip()) < 12
                or _normalized(quote) not in _normalized(source.text)
                or not isinstance(relation, str)
                or relation not in {"supports", "refutes", "irrelevant"}
            ):
                errors.append("invalid or fabricated citation")
                continue
            citations.append(
                {
                    "source_id": source_id,
                    "url": source.url,
                    "quote": quote,
                    "relation": relation,
                    "content_hash": source.content_hash,
                }
            )
            if relation == "supports":
                supporting.append(source)
            if relation == "refutes":
                refuting = True
        if verdict == "supported" and refuting:
            errors.append("supported verdict contains counter-evidence")
        if verdict == "supported" and not self._source_policy(supporting):
            errors.append("requires trusted primary source or corroborating sources")
        return {
            "accepted": verdict == "supported" and not errors,
            "verdict": verdict if verdict in VERDICTS else "insufficient",
            "reason": "; ".join(errors) if errors else str(report.get("reason", "")),
            "conditions": report.get("conditions", ""),
            "sources": citations,
            "report": dict(report),
            "checks": {"counter_search_completed": counter_search_completed},
        }

    def _source_policy(self, sources: Sequence[WebEvidence]) -> bool:
        unique: list[WebEvidence] = []
        for source in sources:
            host = (urlsplit(source.url).hostname or "").lower()
            if host in self.trusted_hosts:
                return True
            if any(
                host == (urlsplit(other.url).hostname or "").lower()
                or source.content_hash == other.content_hash
                or SequenceMatcher(
                    None,
                    _normalized(source.text),
                    _normalized(other.text),
                    autojunk=True,
                ).ratio()
                >= 0.85
                for other in unique
            ):
                continue
            unique.append(source)
        # Different hosts are corroboration candidates, not proof of independent
        # authorship. Reprints with the same or near-identical content are excluded.
        return len(unique) >= 2
