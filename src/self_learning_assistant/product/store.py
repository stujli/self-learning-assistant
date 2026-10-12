from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import ProductError

SCHEMA = """
CREATE TABLE IF NOT EXISTS product_schema(version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS personal_memories(
 id TEXT PRIMARY KEY, user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
 kind TEXT NOT NULL, status TEXT NOT NULL, source_id TEXT NOT NULL, quote TEXT NOT NULL,
 updated_at TEXT NOT NULL, UNIQUE(user_id,key));
CREATE TABLE IF NOT EXISTS memory_exclusions(
 user_id TEXT NOT NULL,key TEXT NOT NULL,PRIMARY KEY(user_id,key));
CREATE TABLE IF NOT EXISTS product_messages(
 id TEXT PRIMARY KEY,user_id TEXT NOT NULL,session_id TEXT NOT NULL,
 role TEXT NOT NULL,text TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS product_jobs(
 id TEXT PRIMARY KEY,user_id TEXT NOT NULL,kind TEXT NOT NULL,prompt TEXT NOT NULL,
 source_id TEXT NOT NULL,state TEXT NOT NULL,result TEXT NOT NULL DEFAULT '{}',
 error TEXT NOT NULL DEFAULT '',attempts INTEGER NOT NULL DEFAULT 0,
 created_at TEXT NOT NULL,lease_until TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS product_documents(
 id TEXT PRIMARY KEY,user_id TEXT NOT NULL,kind TEXT NOT NULL,title TEXT NOT NULL,
 locator TEXT NOT NULL,sha256 TEXT NOT NULL,raw BLOB NOT NULL,
 metadata TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(user_id,sha256));
CREATE TABLE IF NOT EXISTS product_chunks(
 id TEXT PRIMARY KEY,document_id TEXT NOT NULL REFERENCES product_documents(id) ON DELETE CASCADE,
 user_id TEXT NOT NULL,text TEXT NOT NULL,location TEXT NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS product_fts USING fts5(
 id UNINDEXED,user_id UNINDEXED,document_id UNINDEXED,tokens);
CREATE TABLE IF NOT EXISTS product_learning_messages(
 message_id TEXT PRIMARY KEY REFERENCES product_messages(id) ON DELETE CASCADE,
 user_id TEXT NOT NULL,eligible INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS product_training_runs(
 id TEXT PRIMARY KEY,user_id TEXT NOT NULL,model_fingerprint TEXT NOT NULL,
 state TEXT NOT NULL,path TEXT NOT NULL,sources TEXT NOT NULL,
 result TEXT NOT NULL DEFAULT '{}',created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS product_active_adapters(
 user_id TEXT NOT NULL,model_fingerprint TEXT NOT NULL,run_id TEXT NOT NULL,
 PRIMARY KEY(user_id,model_fingerprint));
"""


def now() -> str:
    return datetime.now(UTC).isoformat()


def tokens(text: str) -> str:
    parts = re.findall(r"[a-zA-Z0-9_]+|[\u3400-\u9fff]+", text.casefold())
    result: list[str] = []
    for part in parts:
        if re.fullmatch(r"[\u3400-\u9fff]+", part):
            result.extend(part[i : i + 2] for i in range(max(1, len(part) - 1)))
        else:
            result.append(part)
    return " ".join(result)


@contextmanager
def database_lock(database: Path, *, exclusive: bool = False) -> Iterator[None]:
    lock_path = database.with_name(database.name + ".lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield


class Store:
    """User-bound product tables; legacy schemas and project scopes stay intact."""

    def __init__(self, database: Path, user: str = "default_user") -> None:
        if not user.strip() or len(user) > 200:
            raise ProductError("Invalid user identifier")
        self.database = database.expanduser().resolve()
        self.user = user
        self.database.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Preserve legacy tables through the compatibility migration reader.
        from ..memory.sqlite import SqliteMemoryRepository

        with database_lock(self.database):
            SqliteMemoryRepository(self.database)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            rows = connection.execute("SELECT version FROM product_schema").fetchall()
            if not rows:
                connection.execute("INSERT INTO product_schema VALUES(2)")
            elif len(rows) != 1 or rows[0][0] not in {1, 2}:
                raise ProductError("Unsupported product database version")
            else:
                connection.execute("UPDATE product_schema SET version=2")
        os.chmod(self.database, 0o600)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        with database_lock(self.database):
            connection = sqlite3.connect(self.database, timeout=10)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            try:
                with connection:
                    yield connection
            finally:
                connection.close()

    def messages(self, session: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM product_messages WHERE user_id=? AND session_id=? ORDER BY rowid",
                    (self.user, session),
                )
            ]

    def message(self, session: str, role: str, text: str) -> str:
        if (
            role not in {"user", "assistant"}
            or not text.strip()
            or len(text) > 100_000
            or not session
            or len(session) > 200
        ):
            raise ProductError("Invalid message or session")
        identifier = uuid.uuid4().hex
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO product_messages VALUES(?,?,?,?,?,?)",
                (identifier, self.user, session, role, text, now()),
            )
        return identifier

    def allow_learning(self, message: str, eligible: bool) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO product_learning_messages VALUES(?,?,?)",
                (message, self.user, int(eligible)),
            )

    def eligible_message(self, message: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT eligible FROM product_learning_messages WHERE user_id=? AND message_id=?",
                (self.user, message.removeprefix("manual:")),
            ).fetchone()
        return bool(row and row[0])

    def memories(self, status: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM personal_memories WHERE user_id=? AND (? IS NULL OR status=?) ORDER BY updated_at DESC,id",
                    (self.user, status, status),
                )
            ]

    def memory(self, identifier: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM personal_memories WHERE user_id=? AND id=?",
                (self.user, identifier),
            ).fetchone()
        if not row:
            raise ProductError("Personal memory not found")
        return dict(row)

    def remember(
        self,
        key: str,
        value: str,
        kind: str,
        status: str,
        source: str,
        quote: str,
        *,
        explicit: bool = False,
    ) -> str | None:
        with self.connect() as connection:
            return self._remember(
                connection, key, value, kind, status, source, quote, explicit=explicit
            )

    def remember_many(self, items: list[dict[str, Any]], source: str) -> list[str]:
        result = []
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for item in items:
                self_report = bool(
                    re.search(r"我|本人|\b(?:I|my|I'm)\b", item["quote"], re.I)
                )
                identifier = self._remember(
                    connection,
                    item["key"],
                    item["value"],
                    item["kind"],
                    "active" if item["explicit"] and self_report else "pending",
                    source,
                    item["quote"],
                )
                if identifier:
                    result.append(identifier)
        return result

    def _remember(
        self,
        connection: sqlite3.Connection,
        key: str,
        value: str,
        kind: str,
        status: str,
        source: str,
        quote: str,
        *,
        explicit: bool = False,
    ) -> str | None:
        key = " ".join(key.casefold().split())
        if (
            not key
            or len(key) > 200
            or not value.strip()
            or len(value) > 2000
            or kind not in {"preference", "background", "goal"}
            or status not in {"active", "pending"}
        ):
            raise ProductError("Invalid personal memory")
        identifier = uuid.uuid4().hex
        if explicit:
            connection.execute(
                "DELETE FROM memory_exclusions WHERE user_id=? AND key=?",
                (self.user, key),
            )
        if connection.execute(
            "SELECT 1 FROM memory_exclusions WHERE user_id=? AND key=?",
            (self.user, key),
        ).fetchone():
            return None
        existing = connection.execute(
            "SELECT * FROM personal_memories WHERE user_id=? AND key=?",
            (self.user, key),
        ).fetchone()
        if existing:
            identifier = str(existing["id"])
            # Inferred candidates must never overwrite a confirmed preference.
            if status == "pending" and existing["status"] == "active":
                return identifier
        connection.execute(
            "INSERT INTO personal_memories VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(user_id,key) DO UPDATE SET value=excluded.value,kind=excluded.kind,status=excluded.status,source_id=excluded.source_id,quote=excluded.quote,updated_at=excluded.updated_at",
            (identifier, self.user, key, value, kind, status, source, quote, now()),
        )
        return identifier

    def change_memory(
        self, identifier: str, *, value: str | None = None, status: str | None = None
    ) -> dict[str, Any]:
        record = self.memory(identifier)
        if value is not None and (not value.strip() or len(value) > 2000):
            raise ProductError("Invalid memory value")
        if status is not None and status not in {"active", "dismissed"}:
            raise ProductError("Invalid memory status")
        with self.connect() as connection:
            connection.execute(
                "UPDATE personal_memories SET value=?,status=?,updated_at=? WHERE user_id=? AND id=?",
                (
                    value if value is not None else record["value"],
                    status or "active",
                    now(),
                    self.user,
                    identifier,
                ),
            )
            if value is not None:
                connection.execute(
                    "UPDATE personal_memories SET source_id=?,quote=? WHERE user_id=? AND id=?",
                    (
                        "manual:" + record["source_id"].removeprefix("manual:"),
                        value,
                        self.user,
                        identifier,
                    ),
                )
        return self.memory(identifier)

    def delete_memory(self, identifier: str, *, purge_source: bool = False) -> None:
        record = self.memory(identifier)
        source_id = record["source_id"].removeprefix("manual:")
        with self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO memory_exclusions VALUES(?,?)",
                (self.user, record["key"]),
            )
            connection.execute(
                "DELETE FROM personal_memories WHERE user_id=? AND id=?",
                (self.user, identifier),
            )
            # Do not reprocess a source after deleting one of its memories.
            connection.execute(
                "DELETE FROM product_jobs WHERE user_id=? AND source_id=?",
                (self.user, source_id),
            )
            if purge_source:
                connection.execute(
                    "DELETE FROM product_messages WHERE user_id=? AND id=?",
                    (self.user, source_id),
                )
                connection.execute(
                    "UPDATE personal_memories SET quote='',source_id='' WHERE user_id=? AND source_id=?",
                    (self.user, source_id),
                )

    def enqueue(self, kind: str, prompt: str, source: str = "") -> str:
        if (
            kind not in {"personal", "topic"}
            or not prompt.strip()
            or len(prompt) > 100_000
        ):
            raise ProductError("Invalid learning job")
        identifier = uuid.uuid4().hex
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO product_jobs(id,user_id,kind,prompt,source_id,state,created_at) VALUES(?,?,?,?,?,'queued',?)",
                (identifier, self.user, kind, prompt, source, now()),
            )
        return identifier

    def jobs(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            records = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM product_jobs WHERE user_id=? ORDER BY created_at,id",
                    (self.user,),
                )
            ]
        for item in records:
            item["result"] = json.loads(item["result"])
        return records

    def claim(self, *, limit: int = 10) -> list[dict[str, Any]]:
        if not 1 <= limit <= 100:
            raise ProductError("Job limit must be between 1 and 100")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE product_jobs SET state='queued' WHERE user_id=? AND state='processing' AND lease_until<?",
                (self.user, now()),
            )
            rows = connection.execute(
                "SELECT * FROM product_jobs WHERE user_id=? AND state='queued' ORDER BY created_at LIMIT ?",
                (self.user, limit),
            ).fetchall()
            lease = (datetime.now(UTC) + timedelta(minutes=30)).isoformat()
            for row in rows:
                connection.execute(
                    "UPDATE product_jobs SET state='processing',lease_until=?,attempts=attempts+1 WHERE id=?",
                    (lease, row["id"]),
                )
        return [dict(row) for row in rows]

    def finish(
        self, identifier: str, state: str, result: dict[str, Any], error: str = ""
    ) -> None:
        if state not in {"done", "verified", "deferred", "failed"}:
            raise ProductError("Invalid job state")
        with self.connect() as connection:
            connection.execute(
                "UPDATE product_jobs SET state=?,result=?,error=?,lease_until='' WHERE user_id=? AND id=?",
                (
                    state,
                    json.dumps(result, ensure_ascii=False),
                    error[:1000],
                    self.user,
                    identifier,
                ),
            )

    def retry(self) -> int:
        with self.connect() as connection:
            return connection.execute(
                "UPDATE product_jobs SET state='queued',error='' WHERE user_id=? AND state IN ('failed','deferred')",
                (self.user,),
            ).rowcount

    def documents(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT id,title,kind,locator,sha256,metadata,created_at FROM product_documents WHERE user_id=? ORDER BY created_at",
                    (self.user,),
                )
            ]

    def document(self, identifier: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id,title,kind,locator,sha256,metadata,created_at FROM product_documents WHERE user_id=? AND id=?",
                (self.user, identifier),
            ).fetchone()
            if not row:
                raise ProductError("Document not found")
            result = dict(row)
            result["metadata"] = json.loads(result["metadata"])
            result["chunks"] = [
                dict(item)
                for item in connection.execute(
                    "SELECT id,text,location FROM product_chunks WHERE document_id=? AND user_id=? ORDER BY rowid",
                    (identifier, self.user),
                )
            ]
        return result

    def add_document(
        self,
        title: str,
        locator: str,
        raw: bytes,
        chunks: list[tuple[str, str]],
        *,
        kind: str = "document",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if (
            not chunks
            or sum(len(text) for text, _ in chunks) > 2_000_000
            or len(raw) > 20_000_000
        ):
            raise ProductError("Document empty or exceeds extraction limits")
        digest = hashlib.sha256(raw).hexdigest()
        identifier = uuid.uuid4().hex
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT id FROM product_documents WHERE user_id=? AND sha256=?",
                (self.user, digest),
            ).fetchone()
            if existing:
                return str(existing[0])
            connection.execute(
                "INSERT INTO product_documents VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    self.user,
                    kind,
                    title,
                    locator,
                    digest,
                    raw,
                    json.dumps(
                        metadata or {"verification": "provided"}, ensure_ascii=False
                    ),
                    now(),
                ),
            )
            for text, location in chunks:
                chunk_id = uuid.uuid4().hex
                connection.execute(
                    "INSERT INTO product_chunks VALUES(?,?,?,?,?)",
                    (chunk_id, identifier, self.user, text, location),
                )
                connection.execute(
                    "INSERT INTO product_fts VALUES(?,?,?,?)",
                    (chunk_id, self.user, identifier, tokens(text)),
                )
        return identifier

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        terms = list(dict.fromkeys(tokens(query).split()))[:40]
        if not terms:
            return []
        expression = " OR ".join('"' + term + '"' for term in terms)
        with self.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT c.id,c.document_id,c.text,c.location,d.title,d.locator,d.kind,d.metadata,bm25(product_fts) AS rank FROM product_fts JOIN product_chunks c ON c.id=product_fts.id JOIN product_documents d ON d.id=c.document_id WHERE product_fts MATCH ? AND c.user_id=? ORDER BY rank,c.id LIMIT ?",
                    (expression, self.user, min(max(limit, 1), 20)),
                )
            ]

    def delete_document(self, identifier: str) -> None:
        self.document(identifier)
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM product_fts WHERE user_id=? AND document_id=?",
                (self.user, identifier),
            )
            connection.execute(
                "DELETE FROM product_documents WHERE user_id=? AND id=?",
                (self.user, identifier),
            )

    def import_file(self, path: Path) -> str:
        path = path.expanduser().resolve()
        if not path.is_file() or path.stat().st_size > 20_000_000:
            raise ProductError("Import requires a file of at most 20 MB")
        raw = path.read_bytes()
        chunks: list[tuple[str, str]] = []
        if path.suffix.casefold() == ".pdf":
            from io import BytesIO

            from pypdf import PdfReader

            try:
                reader = PdfReader(BytesIO(raw))
                if reader.is_encrypted or len(reader.pages) > 500:
                    raise ProductError(
                        "Encrypted PDFs and PDFs over 500 pages are unsupported"
                    )
                for index, page in enumerate(reader.pages, 1):
                    text = page.extract_text() or ""
                    chunks.extend((part, f"page {index}") for part in split_text(text))
            except ProductError:
                raise
            except Exception as exc:
                raise ProductError("Cannot extract this PDF") from exc
            if not chunks:
                raise ProductError(
                    "PDF contains no extractable text; OCR is not supported"
                )
        elif path.suffix.casefold() in {".txt", ".md", ".markdown"}:
            try:
                text = raw.decode("utf-8")
            except UnicodeError as exc:
                raise ProductError("Text imports must be UTF-8") from exc
            if "\x00" in text:
                raise ProductError("Binary input is unsupported")
            line = 1
            for part in split_text(text):
                count = part.count("\n")
                chunks.append((part, f"lines {line}-{line + count}"))
                line += count
        else:
            raise ProductError("Supported imports: .txt, .md, .markdown, text .pdf")
        return self.add_document(path.name, str(path), raw, chunks)


def split_text(text: str) -> list[str]:
    return [
        text[start : start + 1500]
        for start in range(0, len(text), 1500)
        if text[start : start + 1500].strip()
    ]
