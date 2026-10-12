# Privacy and data

Conversations, memories, and documents are stored locally, and personal content is isolated by user. File permissions protect application data, but disk encryption is not provided. An external model service's logging and networking behavior depends on that service's configuration.

## Session controls

| Option | Saving and learning behavior |
| --- | --- |
| `--no-learn` | Saves the conversation but excludes personal extraction and parameter training for this turn; verified public knowledge can still be saved |
| `--private` | Uses an ephemeral session and the base model; does not read personal memories or the knowledge base, save messages or tasks, or use a personal adapter; response preferences still apply |
| `--offline` | Disables networking for this operation without changing saving or learning eligibility |
| `--web-query QUERY` | Explicitly supplies a query that may be sent publicly; private mode also requires networking permission for this operation |

```bash
uv run sla chat --private --offline
uv run sla chat "Save this turn, but do not learn from it" --no-learn
```

Private mode sends no queries by default. To explicitly verify public facts, use `--web --web-query QUERY`. Automatic verification in normal chat keeps personal questions local; see [Web verification](web-verification.md).

## Data locations

| Default location | Content |
| --- | --- |
| `~/.config/self-learning-assistant/settings.json` | Configuration |
| `~/.local/share/self-learning-assistant/` | Database, document copies, base models, and adapters |
| `~/.local/state/self-learning-assistant/` | Runtime state |
| `~/.cache/self-learning-assistant/` | Rebuildable cache |

XDG environment variables can override the root directories; relative paths are ignored. `SLA_HOME=/absolute/path` creates config/data/state/cache beneath the specified root, useful for isolated demonstrations and tests. Directory permissions are 0700; databases, configuration, and exports use 0600. Use `uv run sla doctor --json` to inspect the actual paths.

`SLA_MODEL`, `SLA_ONLINE`, `SLA_LEARNING`, and `SLA_SEARXNG_URL` can override configuration. New `SLA_*` variables take precedence over legacy `AUTO_EVOLVE_*` variables. Existing data paths and backups from older installations remain compatible.

## Export, backup, and restore

```bash
uv run sla data export /absolute/path/to/user.json
uv run sla data backup /absolute/path/to/backup.zip
uv run sla data restore /absolute/path/to/backup.zip --yes
uv run sla data migrate /absolute/path/to/old.db
```

| Operation | Scope |
| --- | --- |
| export | Memories, messages, and documents for the current user, as JSON |
| backup | Database, configuration, valid personal adapters, and source manifests for all users, as ZIP |
| restore | Validates the manifest, SHA-256, database, and versions, then stops background work and creates a safety backup before replacing data |
| migrate | Copies a legacy database into an empty destination without adapters; existing parameter versions are revoked |

Export and backup refuse to overwrite existing destinations and use permissions 0600. Backups exclude base model weights, search credentials, and runtime logs, and older backups remain supported. Backups contain private content and should be kept outside public repositories.

Deleting memories or documents affects only application-managed data. It does not remove original user files, existing external backups, or exports. See [Personal memory](personal-memory.md#common-commands) for source deletion and prevention of relearning, and [Background learning](background-learning.md#version-management) for adapter invalidation.
