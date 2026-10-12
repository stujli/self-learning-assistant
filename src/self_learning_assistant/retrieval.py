from __future__ import annotations

import re
from datetime import UTC, datetime

from .domain import MemoryLifecycle, ProjectFactRecord, RetrievedMemory
from .memory.repository import MemoryRepository


class MemoryRetriever:
    def __init__(
        self,
        memory: MemoryRepository,
        *,
        minimum_relevance: float = 0.2,
    ) -> None:
        self.memory = memory
        self.minimum_relevance = minimum_relevance

    def retrieve(
        self,
        query: str,
        *,
        project_limit: int = 5,
    ) -> list[RetrievedMemory]:
        memories = self._project_facts(query, project_limit)
        self.memory.record_access(
            [(item.kind, item.memory_id or item.subject) for item in memories]
        )
        return memories

    def _project_facts(
        self,
        query: str,
        limit: int,
    ) -> list[RetrievedMemory]:
        ranked: list[RetrievedMemory] = []
        for record in self.memory.list_project_facts():
            relevance = _project_relevance(query, record)
            relevance *= _lifecycle_weight(record.lifecycle)
            if relevance < self.minimum_relevance:
                continue
            candidate = record.candidate
            ranked.append(
                RetrievedMemory(
                    kind="project_fact",
                    subject=candidate.subject,
                    label=_memory_label(candidate.subject, candidate.subject_text),
                    value=candidate.object,
                    relevance=round(relevance, 3),
                    timestamp=record.updated_at,
                    memory_scope=record.memory_scope,
                    lifecycle=record.lifecycle,
                    memory_id=candidate.subject,
                )
            )
        ranked.sort(
            key=lambda item: (item.relevance, item.timestamp),
            reverse=True,
        )
        return ranked[: max(0, limit)]


def _project_relevance(query: str, record: ProjectFactRecord) -> float:
    normalized_query = _normalize(query)
    if not normalized_query:
        return 0.0

    candidates = (
        record.candidate.subject_text,
        record.candidate.subject.replace("_", " "),
    )
    best = 0.0
    query_bigrams = _bigrams(normalized_query)
    for candidate in candidates:
        normalized_candidate = _normalize(candidate)
        if not normalized_candidate:
            continue
        if (
            normalized_candidate in normalized_query
            or normalized_query in normalized_candidate
        ):
            best = max(best, 1.0)
            continue
        candidate_bigrams = _bigrams(normalized_candidate)
        if not candidate_bigrams:
            continue
        overlap = len(query_bigrams & candidate_bigrams)
        if overlap >= 2:
            best = max(best, overlap / len(candidate_bigrams))
    return best


def _memory_label(subject: str, subject_text: str) -> str:
    return subject_text.strip() or subject.replace("_", " ").strip()


def _normalize(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower(), flags=re.UNICODE)


def _bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[index : index + 2] for index in range(len(text) - 1)}


def _lifecycle_weight(lifecycle: MemoryLifecycle) -> float:
    if not lifecycle.valid_until:
        return 1.0
    try:
        valid_until = datetime.fromisoformat(
            lifecycle.valid_until.replace("Z", "+00:00")
        )
        verified_at = datetime.fromisoformat(
            lifecycle.last_verified_at.replace("Z", "+00:00")
        )
    except ValueError:
        return 0.0
    total = (valid_until - verified_at).total_seconds()
    remaining = (valid_until - datetime.now(UTC)).total_seconds()
    if total <= 0 or remaining <= 0:
        return 0.0
    return max(0.2, min(1.0, remaining / total))
