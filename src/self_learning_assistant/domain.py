from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal, TypedDict

DecisionAction = Literal["accept", "hold", "reject"]
Sensitivity = Literal["none", "personal", "secret"]
MemoryKind = Literal["project_fact"]
MemoryStatus = Literal["active", "stale", "archived"]
MemoryEventType = Literal[
    "created",
    "accessed",
    "marked_stale",
    "revalidated",
    "archived",
]
ChatRole = Literal["system", "user", "assistant"]
EvidenceSourceType = Literal[
    "user_statement",
    "project_file",
    "web",
    "manual_confirmation",
    "system",
    "migration",
]
EvidenceStatus = Literal["unverified", "verified", "rejected", "stale"]
EvidenceTargetKind = Literal["project_fact", "hold"]


@dataclass(frozen=True, slots=True)
class MemoryScope:
    """Identity boundary applied to every memory read and write."""

    user_id: str = "default_user"
    project_id: str = "default_project"
    workspace_id: str = "default_workspace"

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            normalized = value.strip()
            if not normalized:
                raise ValueError(f"{name} must not be empty")
            if len(normalized) > 200:
                raise ValueError(f"{name} must be at most 200 characters")
            object.__setattr__(self, name, normalized)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    evidence_id: str
    source_type: EvidenceSourceType
    locator: str
    content_hash: str
    captured_at: str
    status: EvidenceStatus
    strength: float
    target_kind: EvidenceTargetKind
    target_id: str
    expires_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_scope: MemoryScope = field(default_factory=MemoryScope)

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise ValueError("evidence_id must not be empty")
        if not self.locator.strip():
            raise ValueError("evidence locator must not be empty")
        if len(self.locator) > 2048:
            raise ValueError("evidence locator must be at most 2048 characters")
        if not self.captured_at.strip():
            raise ValueError("evidence captured_at must not be empty")
        if not self.target_id.strip():
            raise ValueError("evidence target_id must not be empty")
        if len(self.target_id) > 200:
            raise ValueError("evidence target_id must be at most 200 characters")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("evidence strength must be between 0 and 1")
        if self.source_type != "migration" and not _is_sha256(self.content_hash):
            raise ValueError("evidence content_hash must be a sha256 digest")
        _validate_timestamp(
            self.captured_at,
            "captured_at",
            require_timezone=self.source_type != "migration",
        )
        if self.expires_at:
            _validate_timestamp(self.expires_at, "expires_at", require_timezone=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MemoryLifecycle:
    status: MemoryStatus = "active"
    valid_until: str = ""
    last_verified_at: str = ""
    last_accessed_at: str = ""
    access_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_sha256(value: str) -> bool:
    prefix = "sha256:"
    digest = value.removeprefix(prefix)
    return (
        value.startswith(prefix)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def _validate_timestamp(
    value: str,
    field_name: str,
    *,
    require_timezone: bool,
) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"evidence {field_name} must be an ISO-8601 timestamp"
        ) from exc
    if require_timezone and parsed.utcoffset() is None:
        raise ValueError(f"evidence {field_name} must include a timezone")


class ChatMessage(TypedDict):
    role: ChatRole
    content: str


@dataclass(frozen=True, slots=True)
class CandidateFact:
    subject: str
    relation: str
    object: str
    scope: str
    source: str
    raw_text: str
    subject_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScoreResult:
    credibility: float
    novelty: float
    conflict_risk: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class KnowledgeAssessment:
    clarity: float
    plausibility: float
    source_reliability: float
    evidence_strength: float
    conflict_risk: float
    temporal_stability: float
    recommended_action: DecisionAction
    reason: str
    evidence_needed: tuple[str, ...] = ()
    judge: str = "unknown"
    importance: float = 0.5
    sensitivity: Sensitivity = "none"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence_needed"] = list(self.evidence_needed)
        return value


@dataclass(frozen=True, slots=True)
class ProjectFactRecord:
    candidate: CandidateFact
    score: ScoreResult
    updated_at: str
    assessment: KnowledgeAssessment | None = None
    memory_scope: MemoryScope = field(default_factory=MemoryScope)
    evidence: tuple[EvidenceRecord, ...] = ()
    lifecycle: MemoryLifecycle = field(default_factory=MemoryLifecycle)


@dataclass(frozen=True, slots=True)
class RetrievedMemory:
    kind: MemoryKind
    subject: str
    label: str
    value: str
    relevance: float
    timestamp: str = ""
    memory_scope: MemoryScope = field(default_factory=MemoryScope)
    lifecycle: MemoryLifecycle = field(default_factory=MemoryLifecycle)
    memory_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ConflictConstraint:
    subject: str
    unaccepted_value: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WebEvidence:
    title: str
    url: str
    snippet: str
    text: str = ""
    retrieved_at: str = ""
    content_hash: str = ""
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
