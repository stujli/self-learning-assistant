import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from self_learning_assistant.domain import MemoryScope
from self_learning_assistant.memory.repository import UnsupportedSchemaError
from self_learning_assistant.memory.schema import SCHEMA_SQL
from self_learning_assistant.memory.sqlite import SqliteMemoryRepository
from self_learning_assistant.product.config import ProductError, Settings
from self_learning_assistant.product.data import backup, migrate, restore
from self_learning_assistant.product.store import Store
from self_learning_assistant.retrieval import MemoryRetriever


def insert_fact(
    database: Path, scope: MemoryScope, subject="api_base", status="active", valid=""
):
    with sqlite3.connect(database) as connection:
        connection.execute(
            """INSERT INTO project_facts
            (user_id,project_id,workspace_id,subject,relation,value,scope,source,raw_text,
             subject_text,credibility,novelty,conflict_risk,score_reason,updated_at,
             memory_status,valid_until,last_verified_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                scope.user_id,
                scope.project_id,
                scope.workspace_id,
                subject,
                "is",
                "/v2",
                "project",
                "legacy",
                "API base /v2",
                "API base",
                0.9,
                1.0,
                0.0,
                "legacy",
                "2026-01-01T00:00:00+00:00",
                status,
                valid,
                "2026-01-01T00:00:00+00:00",
            ),
        )


def test_legacy_facts_remain_scope_bound_and_access_is_audited(tmp_path):
    database = tmp_path / "legacy.db"
    owner = MemoryScope("alice", "project", "workspace")
    repo = SqliteMemoryRepository(database, scope=owner)
    insert_fact(database, owner)
    insert_fact(database, owner, "archived", "archived")
    for scope in [
        MemoryScope("bob", "project", "workspace"),
        MemoryScope("alice", "other", "workspace"),
        MemoryScope("alice", "project", "other"),
    ]:
        assert not SqliteMemoryRepository(database, scope=scope).list_project_facts()
    assert repo.get_project_fact("missing") is None
    assert repo.get_project_fact("archived") is None
    assert repo.get_project_fact("archived", include_inactive=True) is not None
    assert len(repo.list_project_facts(None)) == 2
    result = MemoryRetriever(repo).retrieve("API base")
    assert len(result) == 1 and result[0].value == "/v2"
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT access_count FROM project_facts WHERE subject='api_base'"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT event_type FROM memory_events"
        ).fetchall() == [("accessed",)]
    assert MemoryRetriever(repo).retrieve("unrelated words") == []
    assert MemoryRetriever(repo).retrieve("") == []


def test_expired_legacy_fact_is_excluded_without_deleting_history(tmp_path):
    database = tmp_path / "legacy.db"
    repo = SqliteMemoryRepository(database)
    expired = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    insert_fact(database, MemoryScope(), valid=expired)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """INSERT INTO evidence_records
            (evidence_id,user_id,project_id,workspace_id,target_kind,target_id,source_type,
             locator,content_hash,captured_at,expires_at,status,strength,created_at)
            VALUES ('e','default_user','default_project','default_workspace','project_fact',
                    'api_base','web','https://example.org',?,?,?, 'verified',.9,?)""",
            (
                "sha256:" + "a" * 64,
                "2026-01-01T00:00:00+00:00",
                expired,
                "2026-01-01T00:00:00+00:00",
            ),
        )
    assert repo.list_project_facts() == []
    fact = repo.get_project_fact("api_base", include_inactive=True)
    assert fact.lifecycle.status == "stale" and fact.evidence[0].status == "stale"
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT event_type FROM memory_events"
        ).fetchall() == [("marked_stale",)]
        assert connection.execute(
            "SELECT COUNT(*) FROM evidence_records"
        ).fetchone() == (1,)


def test_legacy_research_rows_survive_migration_backup_and_restore(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SLA_HOME", str(tmp_path / "home"))
    source = tmp_path / "old.db"
    SqliteMemoryRepository(source)
    insert_fact(source, MemoryScope())
    with sqlite3.connect(source) as connection:
        connection.execute(
            """INSERT INTO training_samples
            (sample_id,user_id,project_id,workspace_id,prompt,target,sample_type,source_subject,metadata_json,created_at)
            VALUES ('sample','default_user','default_project','default_workspace','old prompt',
                    'old target','legacy','api_base','{}','2026-01-01')"""
        )
    before = source.read_bytes()
    settings = Settings()
    store = Store(settings.paths.database)
    migrate(settings, store, source)
    assert source.read_bytes() == before
    archive = backup(settings, store, tmp_path / "backup.zip")
    with store.connect() as connection:
        connection.execute("DELETE FROM training_samples")
    restore(settings, store, archive)
    with store.connect() as connection:
        assert connection.execute(
            "SELECT prompt,target FROM training_samples"
        ).fetchone()[:] == ("old prompt", "old target")
    assert (
        MemoryRetriever(SqliteMemoryRepository(store.database))
        .retrieve("API base")[0]
        .value
        == "/v2"
    )


def test_unknown_legacy_schema_is_rejected(tmp_path):
    database = tmp_path / "future.db"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version=99")
    with pytest.raises(UnsupportedSchemaError):
        SqliteMemoryRepository(database)


def test_migration_does_not_overwrite_target_containing_only_archived_records(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SLA_HOME", str(tmp_path / "home"))
    source = tmp_path / "source.db"
    SqliteMemoryRepository(source)
    settings = Settings()
    store = Store(settings.paths.database)
    with store.connect() as connection:
        connection.execute(
            """INSERT INTO training_samples
            (sample_id,user_id,project_id,workspace_id,prompt,target,sample_type,
             source_subject,metadata_json,created_at)
            VALUES ('sample','default_user','default_project','default_workspace',
                    'saved prompt','saved target','legacy','api_base','{}','2026-01-01')"""
        )
    with pytest.raises(ProductError, match="contains data"):
        migrate(settings, store, source)
    with store.connect() as connection:
        assert (
            connection.execute("SELECT prompt FROM training_samples").fetchone()[0]
            == "saved prompt"
        )


def test_legacy_fact_assessment_can_still_be_decoded(tmp_path):
    database = tmp_path / "legacy.db"
    repo = SqliteMemoryRepository(database)
    insert_fact(database, MemoryScope())
    assessment = dict(
        clarity=0.9,
        plausibility=0.9,
        source_reliability=0.9,
        evidence_strength=0.9,
        conflict_risk=0.0,
        temporal_stability=0.9,
        recommended_action="accept",
        reason="legacy",
        evidence_needed=[],
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE project_facts SET assessment_json=?", (json.dumps(assessment),)
        )
    assert repo.get_project_fact("api_base").assessment.recommended_action == "accept"


def _schema_v2() -> str:
    schema = SCHEMA_SQL.split("CREATE TABLE IF NOT EXISTS evidence_records", 1)[0]
    lifecycle_columns = (
        "    memory_status TEXT NOT NULL DEFAULT 'active'\n"
        "        CHECK(memory_status IN ('active', 'stale', 'archived')),\n"
        "    valid_until TEXT NOT NULL DEFAULT '',\n"
        "    last_verified_at TEXT NOT NULL DEFAULT '',\n"
        "    last_accessed_at TEXT NOT NULL DEFAULT '',\n"
        "    access_count INTEGER NOT NULL DEFAULT 0 CHECK(access_count >= 0),\n"
    )
    schema = schema.replace(lifecycle_columns, "")
    schema = schema.replace(
        "    user_id TEXT NOT NULL,\n"
        "    project_id TEXT NOT NULL,\n"
        "    workspace_id TEXT NOT NULL,\n",
        "",
    )
    schema = (
        schema.replace(
            "    subject TEXT NOT NULL,\n    relation TEXT NOT NULL,",
            "    subject TEXT PRIMARY KEY,\n    relation TEXT NOT NULL,",
            1,
        )
        .replace(
            "    PRIMARY KEY (user_id, project_id, workspace_id, subject)\n",
            "",
        )
        .replace(
            "    updated_at TEXT NOT NULL,\n);", "    updated_at TEXT NOT NULL\n);"
        )
    )
    return schema.replace(
        "UNIQUE(user_id, project_id, workspace_id, subject, value)",
        "UNIQUE(subject, value)",
    ).replace(
        "ON holds(user_id, project_id, workspace_id, subject, value)",
        "ON holds(subject, value)",
    )


@pytest.mark.parametrize("version", [1, 2])
def test_earliest_legacy_schema_migrates_with_original_fact_and_audit(
    tmp_path, version
):
    database = tmp_path / "legacy.db"
    schema = _schema_v2()
    if version == 1:
        schema = schema.replace("    assessment_json TEXT NOT NULL DEFAULT '{}',\n", "")
        schema = schema.replace(
            "    resolution_reason TEXT NOT NULL DEFAULT '',\n    assessment_json TEXT NOT NULL DEFAULT '{}'",
            "    resolution_reason TEXT NOT NULL DEFAULT ''",
        )
    with sqlite3.connect(database) as connection:
        connection.executescript(schema)
        columns = "subject,relation,value,scope,source,raw_text,subject_text,credibility,novelty,conflict_risk,score_reason,updated_at"
        connection.execute(
            f"INSERT INTO project_facts ({columns}) VALUES ('api_base','is','/v2','project','legacy','raw','API base',.9,1.,0.,'legacy','2026-01-01')"
        )
        connection.execute(f"PRAGMA user_version={version}")
    repo = SqliteMemoryRepository(database)
    fact = repo.get_project_fact("api_base")
    assert fact.candidate.object == "/v2" and fact.memory_scope == MemoryScope()
    assert fact.evidence[0].source_type == "migration"
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (7,)
        assert connection.execute(
            "SELECT event_type FROM memory_events"
        ).fetchall() == [("created",)]
