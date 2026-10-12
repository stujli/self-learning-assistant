from __future__ import annotations

import json
import sqlite3
from typing import Any, cast

from ..domain import (
    CandidateFact,
    DecisionAction,
    EvidenceRecord,
    EvidenceSourceType,
    EvidenceStatus,
    EvidenceTargetKind,
    KnowledgeAssessment,
    MemoryLifecycle,
    MemoryScope,
    MemoryStatus,
    ProjectFactRecord,
    ScoreResult,
    Sensitivity,
)


def project_fact_from_row(row: sqlite3.Row) -> ProjectFactRecord:
    return ProjectFactRecord(
        candidate=_candidate_from_row(row),
        score=_score_from_row(row),
        updated_at=str(row["updated_at"]),
        assessment=_assessment_from_row(row),
        memory_scope=_scope_from_row(row),
        lifecycle=_lifecycle_from_row(row),
    )


def evidence_from_row(row: sqlite3.Row) -> EvidenceRecord:
    metadata = json.loads(str(row["metadata_json"]))
    if not isinstance(metadata, dict):
        metadata = {}
    return EvidenceRecord(
        evidence_id=str(row["evidence_id"]),
        source_type=cast(EvidenceSourceType, row["source_type"]),
        locator=str(row["locator"]),
        content_hash=str(row["content_hash"]),
        captured_at=str(row["captured_at"]),
        expires_at=str(row["expires_at"]),
        status=cast(EvidenceStatus, row["status"]),
        strength=float(row["strength"]),
        target_kind=cast(EvidenceTargetKind, row["target_kind"]),
        target_id=str(row["target_id"]),
        metadata=cast(dict[str, Any], metadata),
        memory_scope=_scope_from_row(row),
    )


def _scope_from_row(row: sqlite3.Row) -> MemoryScope:
    return MemoryScope(
        user_id=str(row["user_id"]),
        project_id=str(row["project_id"]),
        workspace_id=str(row["workspace_id"]),
    )


def _lifecycle_from_row(row: sqlite3.Row) -> MemoryLifecycle:
    return MemoryLifecycle(
        status=cast(MemoryStatus, row["memory_status"]),
        valid_until=str(row["valid_until"]),
        last_verified_at=str(row["last_verified_at"]),
        last_accessed_at=str(row["last_accessed_at"]),
        access_count=int(row["access_count"]),
    )


def _candidate_from_row(row: sqlite3.Row) -> CandidateFact:
    return CandidateFact(
        subject=str(row["subject"]),
        relation=str(row["relation"]),
        object=str(row["value"]),
        scope=str(row["scope"]),
        source=str(row["source"]),
        raw_text=str(row["raw_text"]),
        subject_text=str(row["subject_text"]),
    )


def _score_from_row(row: sqlite3.Row) -> ScoreResult:
    return ScoreResult(
        credibility=float(row["credibility"]),
        novelty=float(row["novelty"]),
        conflict_risk=float(row["conflict_risk"]),
        reason=str(row["score_reason"]),
    )


def _assessment_from_row(row: sqlite3.Row) -> KnowledgeAssessment | None:
    if "assessment_json" not in row.keys():
        return None
    value = json.loads(str(row["assessment_json"]))
    if not isinstance(value, dict) or not value:
        return None
    action = value.get("recommended_action")
    if action not in {"accept", "hold", "reject"}:
        return None
    evidence = value.get("evidence_needed", [])
    if not isinstance(evidence, list):
        return None
    try:
        return KnowledgeAssessment(
            clarity=float(value["clarity"]),
            plausibility=float(value["plausibility"]),
            source_reliability=float(value["source_reliability"]),
            evidence_strength=float(value["evidence_strength"]),
            conflict_risk=float(value["conflict_risk"]),
            temporal_stability=float(value["temporal_stability"]),
            recommended_action=cast(DecisionAction, action),
            reason=str(value["reason"]),
            evidence_needed=tuple(str(item) for item in evidence),
            judge=str(value.get("judge", "unknown")),
            importance=float(value.get("importance", 0.5)),
            sensitivity=cast(Sensitivity, value.get("sensitivity", "none")),
        )
    except (KeyError, TypeError, ValueError):
        return None
