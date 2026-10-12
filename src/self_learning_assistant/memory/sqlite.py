"""Compatibility reader and migrations for existing project-memory databases.

Research writers are archived. Existing tables, evidence and audit data remain intact.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from ..domain import (
    EvidenceRecord,
    EvidenceTargetKind,
    MemoryEventType,
    MemoryKind,
    MemoryScope,
    MemoryStatus,
    ProjectFactRecord,
)
from .codec import evidence_from_row, project_fact_from_row
from .repository import UnsupportedSchemaError
from .schema import (
    MIGRATION_1_TO_2,
    MIGRATION_2_TO_3,
    MIGRATION_3_TO_4,
    MIGRATION_4_TO_5,
    MIGRATION_5_TO_6,
    MIGRATION_6_TO_7,
    SCHEMA_SQL,
    SCHEMA_VERSION,
)

_INITIALIZE_LOCK = threading.Lock()


class SqliteMemoryRepository:
    def __init__(
        self,
        database_path: Path | str,
        *,
        scope: MemoryScope | None = None,
    ) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.scope = scope or MemoryScope()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with _INITIALIZE_LOCK:
            self._initialize()

    def get_project_fact(
        self,
        subject: str,
        *,
        include_inactive: bool = False,
    ) -> ProjectFactRecord | None:
        self._refresh_expired()
        status_clause = "" if include_inactive else " AND memory_status = 'active'"
        with self._connect() as connection:
            row = connection.execute(
                f"""SELECT * FROM project_facts
                WHERE user_id = ? AND project_id = ? AND workspace_id = ?
                  AND subject = ?{status_clause}""",  # noqa: S608
                (*self._scope_values(), subject),
            ).fetchone()
        return self._project_fact_record(row) if row else None

    def list_project_facts(
        self,
        status: MemoryStatus | None = "active",
    ) -> list[ProjectFactRecord]:
        self._refresh_expired()
        status_clause = "" if status is None else " AND memory_status = ?"
        parameters: tuple[str, ...] = self._scope_values()
        if status is not None:
            parameters = (*parameters, status)
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT * FROM project_facts
                WHERE user_id = ? AND project_id = ? AND workspace_id = ?
                {status_clause}
                ORDER BY updated_at DESC""",  # noqa: S608
                parameters,
            ).fetchall()
        return [self._project_fact_record(row) for row in rows]

    def record_access(self, memories: Sequence[tuple[MemoryKind, str]]) -> None:
        unique = tuple(dict.fromkeys(memories))
        if not unique:
            return
        now = _now_iso()
        with self._transaction() as connection:
            for kind, memory_id in unique:
                table, identifier = _memory_table(kind)
                cursor = connection.execute(
                    f"""UPDATE {table}
                    SET access_count = access_count + 1, last_accessed_at = ?
                    WHERE user_id = ? AND project_id = ? AND workspace_id = ?
                      AND {identifier} = ? AND memory_status = 'active'""",  # noqa: S608
                    (now, *self._scope_values(), memory_id),
                )
                if cursor.rowcount:
                    self._insert_event(
                        connection,
                        kind,
                        memory_id,
                        "accessed",
                        "included in retrieval context",
                        occurred_at=now,
                    )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            version = cast(
                int,
                connection.execute("PRAGMA user_version").fetchone()[0],
            )
            if version == 1:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    for statement in MIGRATION_1_TO_2:
                        connection.execute(statement)
                    connection.execute("PRAGMA user_version = 2")
                except Exception:
                    connection.rollback()
                    raise
                else:
                    connection.commit()
                version = 2
            if version == 2:
                connection.commit()
                connection.execute("PRAGMA foreign_keys = OFF")
                try:
                    connection.executescript(
                        f"BEGIN IMMEDIATE;{MIGRATION_2_TO_3}"
                        "PRAGMA user_version = 3;COMMIT;"
                    )
                except Exception:
                    connection.rollback()
                    raise
                finally:
                    connection.execute("PRAGMA foreign_keys = ON")
                version = 3
            if version == 3:
                try:
                    connection.executescript(
                        f"BEGIN IMMEDIATE;{MIGRATION_3_TO_4}"
                        "PRAGMA user_version = 4;COMMIT;"
                    )
                except Exception:
                    connection.rollback()
                    raise
                version = 4
            if version == 4:
                try:
                    connection.executescript(
                        f"BEGIN IMMEDIATE;{MIGRATION_4_TO_5}"
                        "PRAGMA user_version = 5;COMMIT;"
                    )
                except Exception:
                    connection.rollback()
                    raise
                version = 5
            if version == 5:
                try:
                    connection.executescript(
                        f"BEGIN IMMEDIATE;{MIGRATION_5_TO_6}"
                        "PRAGMA user_version = 6;COMMIT;"
                    )
                except Exception:
                    connection.rollback()
                    raise
                version = 6
            if version == 6:
                try:
                    connection.executescript(
                        f"BEGIN IMMEDIATE;{MIGRATION_6_TO_7}"
                        "PRAGMA user_version = 7;COMMIT;"
                    )
                except Exception:
                    connection.rollback()
                    raise
                version = 7
            if version not in {0, SCHEMA_VERSION}:
                raise UnsupportedSchemaError(
                    f"Unsupported database schema version {version}; "
                    f"expected {SCHEMA_VERSION}."
                )
            connection.executescript(SCHEMA_SQL)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _refresh_expired(self) -> None:
        now = datetime.now(UTC)
        with self._transaction() as connection:
            evidence_rows = connection.execute(
                """SELECT evidence_id, expires_at FROM evidence_records
                WHERE user_id = ? AND project_id = ? AND workspace_id = ?
                  AND status IN ('unverified', 'verified') AND expires_at != ''""",
                self._scope_values(),
            ).fetchall()
            expired_evidence = [
                str(row["evidence_id"])
                for row in evidence_rows
                if _is_expired(str(row["expires_at"]), now)
            ]
            for evidence_id in expired_evidence:
                connection.execute(
                    "UPDATE evidence_records SET status = 'stale' WHERE evidence_id = ?",
                    (evidence_id,),
                )

            rows = connection.execute(
                """SELECT subject, valid_until FROM project_facts
                WHERE user_id = ? AND project_id = ? AND workspace_id = ?
                  AND memory_status = 'active' AND valid_until != ''""",
                self._scope_values(),
            ).fetchall()
            for row in rows:
                if not _is_expired(str(row["valid_until"]), now):
                    continue
                subject = str(row["subject"])
                connection.execute(
                    """UPDATE project_facts SET memory_status = 'stale'
                    WHERE user_id = ? AND project_id = ? AND workspace_id = ?
                      AND subject = ? AND memory_status = 'active'""",
                    (*self._scope_values(), subject),
                )
                self._insert_event(
                    connection,
                    "project_fact",
                    subject,
                    "marked_stale",
                    "evidence validity period expired",
                )

    def _insert_event(
        self,
        connection: sqlite3.Connection,
        memory_kind: MemoryKind,
        memory_id: str,
        event_type: MemoryEventType,
        reason: str,
        *,
        occurred_at: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        connection.execute(
            """INSERT INTO memory_events (
                event_id, user_id, project_id, workspace_id, memory_kind,
                memory_id, event_type, reason, occurred_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                *self._scope_values(),
                memory_kind,
                memory_id,
                event_type,
                reason[:500],
                occurred_at or _now_iso(),
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )

    def _project_fact_record(self, row: sqlite3.Row) -> ProjectFactRecord:
        record = project_fact_from_row(row)
        return replace(
            record,
            evidence=tuple(
                self._load_evidence("project_fact", record.candidate.subject)
            ),
        )

    def _load_evidence(
        self,
        target_kind: EvidenceTargetKind,
        target_id: str,
    ) -> list[EvidenceRecord]:
        with self._connect() as connection:
            return self._evidence_for_target(connection, target_kind, target_id)

    def _evidence_for_target(
        self,
        connection: sqlite3.Connection,
        target_kind: EvidenceTargetKind,
        target_id: str,
    ) -> list[EvidenceRecord]:
        rows = connection.execute(
            """SELECT * FROM evidence_records
            WHERE user_id = ? AND project_id = ? AND workspace_id = ?
              AND target_kind = ? AND target_id = ?
            ORDER BY captured_at, created_at, evidence_id""",
            (*self._scope_values(), target_kind, target_id),
        ).fetchall()
        return [evidence_from_row(row) for row in rows]

    def _scope_values(self) -> tuple[str, str, str]:
        return (
            self.scope.user_id,
            self.scope.project_id,
            self.scope.workspace_id,
        )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _is_expired(value: str, now: datetime) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return True
    if parsed.utcoffset() is None:
        return True
    return parsed <= now


def _memory_table(kind: MemoryKind) -> tuple[str, str]:
    if kind != "project_fact":
        raise ValueError(f"unsupported memory kind: {kind}")
    return "project_facts", "subject"
