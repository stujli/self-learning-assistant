from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..domain import MemoryKind, MemoryScope, MemoryStatus, ProjectFactRecord


class MemoryStoreError(RuntimeError):
    """Base error for legacy database compatibility."""


class UnsupportedSchemaError(MemoryStoreError):
    """Raised when an existing database schema is unsupported."""


class MemoryRepository(Protocol):
    scope: MemoryScope

    def list_project_facts(
        self, status: MemoryStatus | None = "active"
    ) -> list[ProjectFactRecord]: ...

    def record_access(self, memories: Sequence[tuple[MemoryKind, str]]) -> None: ...
