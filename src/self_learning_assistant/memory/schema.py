SCHEMA_VERSION = 7

MIGRATION_1_TO_2 = (
    "ALTER TABLE project_facts ADD COLUMN assessment_json TEXT NOT NULL DEFAULT '{}'",
    "ALTER TABLE user_preferences ADD COLUMN assessment_json TEXT NOT NULL DEFAULT '{}'",
    "ALTER TABLE holds ADD COLUMN assessment_json TEXT NOT NULL DEFAULT '{}'",
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS project_facts (
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    relation TEXT NOT NULL,
    value TEXT NOT NULL,
    scope TEXT NOT NULL,
    source TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    subject_text TEXT NOT NULL,
    credibility REAL NOT NULL,
    novelty REAL NOT NULL,
    conflict_risk REAL NOT NULL,
    score_reason TEXT NOT NULL,
    assessment_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL,
    memory_status TEXT NOT NULL DEFAULT 'active'
        CHECK(memory_status IN ('active', 'stale', 'archived')),
    valid_until TEXT NOT NULL DEFAULT '',
    last_verified_at TEXT NOT NULL DEFAULT '',
    last_accessed_at TEXT NOT NULL DEFAULT '',
    access_count INTEGER NOT NULL DEFAULT 0 CHECK(access_count >= 0),
    PRIMARY KEY (user_id, project_id, workspace_id, subject)
);

-- Legacy storage retained only so schema v1-v7 databases can migrate safely.
-- Runtime code no longer reads from or writes to this table.
CREATE TABLE IF NOT EXISTS user_preferences (
    preference_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    relation TEXT NOT NULL,
    value TEXT NOT NULL,
    scope TEXT NOT NULL,
    source TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    subject_text TEXT NOT NULL,
    credibility REAL NOT NULL,
    novelty REAL NOT NULL,
    conflict_risk REAL NOT NULL,
    score_reason TEXT NOT NULL,
    assessment_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    memory_status TEXT NOT NULL DEFAULT 'active'
        CHECK(memory_status IN ('active', 'stale', 'archived')),
    valid_until TEXT NOT NULL DEFAULT '',
    last_verified_at TEXT NOT NULL DEFAULT '',
    last_accessed_at TEXT NOT NULL DEFAULT '',
    access_count INTEGER NOT NULL DEFAULT 0 CHECK(access_count >= 0),
    UNIQUE(user_id, project_id, workspace_id, subject, value)
);

CREATE TABLE IF NOT EXISTS holds (
    hold_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    relation TEXT NOT NULL,
    value TEXT NOT NULL,
    scope TEXT NOT NULL,
    source TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    subject_text TEXT NOT NULL,
    credibility REAL NOT NULL,
    novelty REAL NOT NULL,
    conflict_risk REAL NOT NULL,
    score_reason TEXT NOT NULL,
    decision_action TEXT NOT NULL,
    decision_reason TEXT NOT NULL,
    knowledge_type TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'accepted', 'rejected')),
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    resolution_reason TEXT NOT NULL DEFAULT '',
    assessment_json TEXT NOT NULL DEFAULT '{}'
);

CREATE UNIQUE INDEX IF NOT EXISTS holds_pending_candidate
ON holds(user_id, project_id, workspace_id, subject, value)
WHERE status = 'pending';

CREATE TABLE IF NOT EXISTS training_samples (
    sample_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    target TEXT NOT NULL,
    sample_type TEXT NOT NULL,
    source_subject TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    source_hold_id TEXT UNIQUE REFERENCES holds(hold_id),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_records (
    evidence_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    target_kind TEXT NOT NULL
        CHECK(target_kind IN ('project_fact', 'user_preference', 'hold')),
    target_id TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK(source_type IN (
        'user_statement', 'project_file', 'web', 'manual_confirmation',
        'system', 'migration'
    )),
    locator TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    expires_at TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL
        CHECK(status IN ('unverified', 'verified', 'rejected', 'stale')),
    strength REAL NOT NULL CHECK(strength >= 0.0 AND strength <= 1.0),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (
        user_id, project_id, workspace_id, target_kind, target_id,
        source_type, locator, content_hash
    )
);

CREATE INDEX IF NOT EXISTS evidence_by_target
ON evidence_records(user_id, project_id, workspace_id, target_kind, target_id);

CREATE TABLE IF NOT EXISTS memory_events (
    event_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    memory_kind TEXT NOT NULL
        CHECK(memory_kind IN ('project_fact', 'user_preference')),
    memory_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK(event_type IN (
        'created', 'accessed', 'marked_stale', 'revalidated', 'archived'
    )),
    reason TEXT NOT NULL DEFAULT '',
    occurred_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS memory_events_by_target
ON memory_events(
    user_id, project_id, workspace_id, memory_kind, memory_id, occurred_at
);

CREATE TABLE IF NOT EXISTS skill_versions (
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK(version > 0),
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    steps_json TEXT NOT NULL,
    applicability_json TEXT NOT NULL DEFAULT '[]',
    required_tools_json TEXT NOT NULL DEFAULT '[]',
    permissions_json TEXT NOT NULL DEFAULT '[]',
    source TEXT NOT NULL,
    source_text TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL CHECK(status IN (
        'candidate', 'evaluated', 'active', 'deprecated', 'rejected'
    )),
    created_at TEXT NOT NULL,
    evaluated_at TEXT NOT NULL DEFAULT '',
    promoted_at TEXT NOT NULL DEFAULT '',
    deprecated_at TEXT NOT NULL DEFAULT '',
    status_reason TEXT NOT NULL DEFAULT '',
    evaluation_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (user_id, project_id, workspace_id, skill_id, version)
);

CREATE UNIQUE INDEX IF NOT EXISTS skill_one_active_version
ON skill_versions(user_id, project_id, workspace_id, skill_id)
WHERE status = 'active';

CREATE TABLE IF NOT EXISTS skill_executions (
    execution_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('started', 'succeeded', 'failed')),
    input_summary TEXT NOT NULL DEFAULT '',
    output_summary TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (user_id, project_id, workspace_id, skill_id, version)
        REFERENCES skill_versions(user_id, project_id, workspace_id, skill_id, version)
);

CREATE INDEX IF NOT EXISTS skill_executions_by_skill
ON skill_executions(
    user_id, project_id, workspace_id, skill_id, started_at
);

CREATE TABLE IF NOT EXISTS skill_events (
    event_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    event_type TEXT NOT NULL CHECK(event_type IN (
        'proposed', 'evaluated', 'promoted', 'deprecated',
        'execution_started', 'execution_finished'
    )),
    reason TEXT NOT NULL DEFAULT '',
    occurred_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS skill_events_by_skill
ON skill_events(
    user_id, project_id, workspace_id, skill_id, version, occurred_at
);

CREATE TABLE IF NOT EXISTS conversation_sessions (
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, project_id, workspace_id, session_id)
);

CREATE INDEX IF NOT EXISTS conversation_sessions_by_scope
ON conversation_sessions(user_id, project_id, workspace_id, updated_at);

CREATE TABLE IF NOT EXISTS conversation_messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    learning_status TEXT NOT NULL CHECK(learning_status IN (
        'pending', 'processing', 'processed', 'failed', 'skipped'
    )),
    learning_attempts INTEGER NOT NULL DEFAULT 0 CHECK(learning_attempts >= 0),
    processed_at TEXT NOT NULL DEFAULT '',
    processing_error TEXT NOT NULL DEFAULT '',
    processor_version TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (user_id, project_id, workspace_id, session_id)
        REFERENCES conversation_sessions(
            user_id, project_id, workspace_id, session_id
        )
);

CREATE INDEX IF NOT EXISTS conversation_messages_learning_queue
ON conversation_messages(
    user_id, project_id, workspace_id, role, learning_status, created_at
);

CREATE INDEX IF NOT EXISTS conversation_messages_by_session
ON conversation_messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS learning_jobs (
    job_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('conversation', 'topic')),
    prompt TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL CHECK(state IN (
        'queued', 'researching', 'verified', 'training', 'evaluating',
        'promoted', 'deferred', 'rejected', 'failed'
    )),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
    available_at TEXT NOT NULL,
    lease_until TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    result_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS learning_jobs_queue
ON learning_jobs(user_id, project_id, workspace_id, state, available_at);

CREATE TABLE IF NOT EXISTS knowledge_claims (
    claim_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES learning_jobs(job_id),
    claim TEXT NOT NULL,
    normalized_claim TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('candidate', 'accepted', 'rejected', 'deferred')),
    verification_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(job_id, normalized_claim)
);
CREATE INDEX IF NOT EXISTS knowledge_claims_by_job
ON knowledge_claims(job_id, status);

CREATE TABLE IF NOT EXISTS research_sources (
    source_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES learning_jobs(job_id),
    claim_id TEXT NOT NULL REFERENCES knowledge_claims(claim_id),
    url TEXT NOT NULL,
    domain TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    excerpt TEXT NOT NULL DEFAULT '',
    UNIQUE(claim_id, url, content_hash)
);
CREATE INDEX IF NOT EXISTS research_sources_by_claim
ON research_sources(claim_id, domain);

CREATE TABLE IF NOT EXISTS learning_artifacts (
    artifact_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES learning_jobs(job_id),
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
"""

# Rebuilds v2 tables because their primary and unique keys did not contain a namespace.
# Existing records are intentionally assigned to the historical default namespace.
MIGRATION_2_TO_3 = """
ALTER TABLE project_facts RENAME TO project_facts_v2;
ALTER TABLE user_preferences RENAME TO user_preferences_v2;
ALTER TABLE holds RENAME TO holds_v2;
ALTER TABLE training_samples RENAME TO training_samples_v2;
DROP INDEX IF EXISTS holds_pending_candidate;

CREATE TABLE project_facts (
    user_id TEXT NOT NULL, project_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
    subject TEXT NOT NULL, relation TEXT NOT NULL, value TEXT NOT NULL,
    scope TEXT NOT NULL, source TEXT NOT NULL, raw_text TEXT NOT NULL,
    subject_text TEXT NOT NULL, credibility REAL NOT NULL, novelty REAL NOT NULL,
    conflict_risk REAL NOT NULL, score_reason TEXT NOT NULL,
    assessment_json TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, project_id, workspace_id, subject)
);
INSERT INTO project_facts
SELECT 'default_user', 'default_project', 'default_workspace', subject, relation,
       value, scope, source, raw_text, subject_text, credibility, novelty,
       conflict_risk, score_reason, assessment_json, updated_at
FROM project_facts_v2;

CREATE TABLE user_preferences (
    preference_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL, subject TEXT NOT NULL, relation TEXT NOT NULL,
    value TEXT NOT NULL, scope TEXT NOT NULL, source TEXT NOT NULL,
    raw_text TEXT NOT NULL, subject_text TEXT NOT NULL, credibility REAL NOT NULL,
    novelty REAL NOT NULL, conflict_risk REAL NOT NULL, score_reason TEXT NOT NULL,
    assessment_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
    UNIQUE(user_id, project_id, workspace_id, subject, value)
);
INSERT INTO user_preferences
SELECT preference_id, 'default_user', 'default_project', 'default_workspace',
       subject, relation, value, scope, source, raw_text, subject_text,
       credibility, novelty, conflict_risk, score_reason, assessment_json, created_at
FROM user_preferences_v2;

CREATE TABLE holds (
    hold_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL, subject TEXT NOT NULL, relation TEXT NOT NULL,
    value TEXT NOT NULL, scope TEXT NOT NULL, source TEXT NOT NULL,
    raw_text TEXT NOT NULL, subject_text TEXT NOT NULL, credibility REAL NOT NULL,
    novelty REAL NOT NULL, conflict_risk REAL NOT NULL, score_reason TEXT NOT NULL,
    decision_action TEXT NOT NULL, decision_reason TEXT NOT NULL,
    knowledge_type TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'accepted', 'rejected')),
    created_at TEXT NOT NULL, resolved_at TEXT,
    resolution_reason TEXT NOT NULL DEFAULT '',
    assessment_json TEXT NOT NULL DEFAULT '{}'
);
INSERT INTO holds
SELECT hold_id, 'default_user', 'default_project', 'default_workspace', subject,
       relation, value, scope, source, raw_text, subject_text, credibility, novelty,
       conflict_risk, score_reason, decision_action, decision_reason, knowledge_type,
       status, created_at, resolved_at, resolution_reason, assessment_json
FROM holds_v2;
CREATE UNIQUE INDEX holds_pending_candidate
ON holds(user_id, project_id, workspace_id, subject, value)
WHERE status = 'pending';

CREATE TABLE training_samples (
    sample_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL, prompt TEXT NOT NULL, target TEXT NOT NULL,
    sample_type TEXT NOT NULL, source_subject TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    source_hold_id TEXT UNIQUE REFERENCES holds(hold_id), created_at TEXT NOT NULL
);
INSERT INTO training_samples
SELECT sample_id, 'default_user', 'default_project', 'default_workspace', prompt,
       target, sample_type, source_subject, metadata_json, source_hold_id, created_at
FROM training_samples_v2;

DROP TABLE training_samples_v2;
DROP TABLE holds_v2;
DROP TABLE user_preferences_v2;
DROP TABLE project_facts_v2;
"""

MIGRATION_3_TO_4 = """
CREATE TABLE evidence_records (
    evidence_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    target_kind TEXT NOT NULL
        CHECK(target_kind IN ('project_fact', 'user_preference', 'hold')),
    target_id TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK(source_type IN (
        'user_statement', 'project_file', 'web', 'manual_confirmation',
        'system', 'migration'
    )),
    locator TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    expires_at TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL
        CHECK(status IN ('unverified', 'verified', 'rejected', 'stale')),
    strength REAL NOT NULL CHECK(strength >= 0.0 AND strength <= 1.0),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (
        user_id, project_id, workspace_id, target_kind, target_id,
        source_type, locator, content_hash
    )
);
CREATE INDEX evidence_by_target
ON evidence_records(user_id, project_id, workspace_id, target_kind, target_id);

INSERT INTO evidence_records
SELECT lower(hex(randomblob(16))), user_id, project_id, workspace_id,
       'project_fact', subject, 'migration', 'schema-v3', '', updated_at, '',
       'unverified', 0.0, '{"reason":"provenance unavailable before schema v4"}',
       updated_at
FROM project_facts;

INSERT INTO evidence_records
SELECT lower(hex(randomblob(16))), user_id, project_id, workspace_id,
       'user_preference', preference_id, 'migration', 'schema-v3', '', created_at,
       '', 'unverified', 0.0,
       '{"reason":"provenance unavailable before schema v4"}', created_at
FROM user_preferences;

INSERT INTO evidence_records
SELECT lower(hex(randomblob(16))), user_id, project_id, workspace_id,
       'hold', hold_id, 'migration', 'schema-v3', '', created_at, '',
       'unverified', 0.0, '{"reason":"provenance unavailable before schema v4"}',
       created_at
FROM holds;
"""

MIGRATION_4_TO_5 = """
ALTER TABLE project_facts ADD COLUMN memory_status TEXT NOT NULL DEFAULT 'active'
    CHECK(memory_status IN ('active', 'stale', 'archived'));
ALTER TABLE project_facts ADD COLUMN valid_until TEXT NOT NULL DEFAULT '';
ALTER TABLE project_facts ADD COLUMN last_verified_at TEXT NOT NULL DEFAULT '';
ALTER TABLE project_facts ADD COLUMN last_accessed_at TEXT NOT NULL DEFAULT '';
ALTER TABLE project_facts ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0
    CHECK(access_count >= 0);

ALTER TABLE user_preferences ADD COLUMN memory_status TEXT NOT NULL DEFAULT 'active'
    CHECK(memory_status IN ('active', 'stale', 'archived'));
ALTER TABLE user_preferences ADD COLUMN valid_until TEXT NOT NULL DEFAULT '';
ALTER TABLE user_preferences ADD COLUMN last_verified_at TEXT NOT NULL DEFAULT '';
ALTER TABLE user_preferences ADD COLUMN last_accessed_at TEXT NOT NULL DEFAULT '';
ALTER TABLE user_preferences ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0
    CHECK(access_count >= 0);

UPDATE project_facts SET last_verified_at = updated_at;
UPDATE user_preferences SET last_verified_at = created_at;

CREATE TABLE memory_events (
    event_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    memory_kind TEXT NOT NULL
            CHECK(memory_kind IN ('project_fact', 'user_preference')),
    memory_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK(event_type IN (
        'created', 'accessed', 'marked_stale', 'revalidated', 'archived'
    )),
    reason TEXT NOT NULL DEFAULT '',
    occurred_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX memory_events_by_target
ON memory_events(
    user_id, project_id, workspace_id, memory_kind, memory_id, occurred_at
);

INSERT INTO memory_events
SELECT lower(hex(randomblob(16))), user_id, project_id, workspace_id,
       'project_fact', subject, 'created', 'imported by schema v5 migration',
       updated_at, '{"migration":true}'
FROM project_facts;
INSERT INTO memory_events
SELECT lower(hex(randomblob(16))), user_id, project_id, workspace_id,
       'user_preference', preference_id, 'created',
       'imported by schema v5 migration', created_at, '{"migration":true}'
FROM user_preferences;
"""

MIGRATION_5_TO_6 = """
CREATE TABLE skill_versions (
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK(version > 0),
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    steps_json TEXT NOT NULL,
    applicability_json TEXT NOT NULL DEFAULT '[]',
    required_tools_json TEXT NOT NULL DEFAULT '[]',
    permissions_json TEXT NOT NULL DEFAULT '[]',
    source TEXT NOT NULL,
    source_text TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL CHECK(status IN (
        'candidate', 'evaluated', 'active', 'deprecated', 'rejected'
    )),
    created_at TEXT NOT NULL,
    evaluated_at TEXT NOT NULL DEFAULT '',
    promoted_at TEXT NOT NULL DEFAULT '',
    deprecated_at TEXT NOT NULL DEFAULT '',
    status_reason TEXT NOT NULL DEFAULT '',
    evaluation_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (user_id, project_id, workspace_id, skill_id, version)
);
CREATE UNIQUE INDEX skill_one_active_version
ON skill_versions(user_id, project_id, workspace_id, skill_id)
WHERE status = 'active';

CREATE TABLE skill_executions (
    execution_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('started', 'succeeded', 'failed')),
    input_summary TEXT NOT NULL DEFAULT '',
    output_summary TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (user_id, project_id, workspace_id, skill_id, version)
        REFERENCES skill_versions(user_id, project_id, workspace_id, skill_id, version)
);
CREATE INDEX skill_executions_by_skill
ON skill_executions(
    user_id, project_id, workspace_id, skill_id, started_at
);

CREATE TABLE skill_events (
    event_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    event_type TEXT NOT NULL CHECK(event_type IN (
        'proposed', 'evaluated', 'promoted', 'deprecated',
        'execution_started', 'execution_finished'
    )),
    reason TEXT NOT NULL DEFAULT '',
    occurred_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX skill_events_by_skill
ON skill_events(
    user_id, project_id, workspace_id, skill_id, version, occurred_at
);
"""

MIGRATION_6_TO_7 = """
CREATE TABLE IF NOT EXISTS conversation_sessions (
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, project_id, workspace_id, session_id)
);
CREATE INDEX IF NOT EXISTS conversation_sessions_by_scope
ON conversation_sessions(user_id, project_id, workspace_id, updated_at);

CREATE TABLE IF NOT EXISTS conversation_messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    learning_status TEXT NOT NULL CHECK(learning_status IN (
        'pending', 'processing', 'processed', 'failed', 'skipped'
    )),
    learning_attempts INTEGER NOT NULL DEFAULT 0 CHECK(learning_attempts >= 0),
    processed_at TEXT NOT NULL DEFAULT '',
    processing_error TEXT NOT NULL DEFAULT '',
    processor_version TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (user_id, project_id, workspace_id, session_id)
        REFERENCES conversation_sessions(
            user_id, project_id, workspace_id, session_id
        )
);
CREATE INDEX IF NOT EXISTS conversation_messages_learning_queue
ON conversation_messages(
    user_id, project_id, workspace_id, role, learning_status, created_at
);
CREATE INDEX IF NOT EXISTS conversation_messages_by_session
ON conversation_messages(session_id, created_at);
"""

MIGRATION_7_TO_8 = """
CREATE TABLE IF NOT EXISTS learning_jobs (
    job_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('conversation', 'topic')),
    prompt TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL CHECK(state IN (
        'queued', 'researching', 'verified', 'training', 'evaluating',
        'promoted', 'deferred', 'rejected', 'failed'
    )),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
    available_at TEXT NOT NULL,
    lease_until TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    result_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS learning_jobs_queue
ON learning_jobs(user_id, project_id, workspace_id, state, available_at);
CREATE TABLE IF NOT EXISTS knowledge_claims (
    claim_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES learning_jobs(job_id),
    claim TEXT NOT NULL,
    normalized_claim TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('candidate', 'accepted', 'rejected', 'deferred')),
    verification_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(job_id, normalized_claim)
);
CREATE INDEX IF NOT EXISTS knowledge_claims_by_job ON knowledge_claims(job_id, status);
CREATE TABLE IF NOT EXISTS research_sources (
    source_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES learning_jobs(job_id),
    claim_id TEXT NOT NULL REFERENCES knowledge_claims(claim_id),
    url TEXT NOT NULL,
    domain TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    excerpt TEXT NOT NULL DEFAULT '',
    UNIQUE(claim_id, url, content_hash)
);
CREATE INDEX IF NOT EXISTS research_sources_by_claim ON research_sources(claim_id, domain);
CREATE TABLE IF NOT EXISTS learning_artifacts (
    artifact_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES learning_jobs(job_id),
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
"""
