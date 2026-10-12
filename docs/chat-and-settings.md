# Chat and settings

The assistant establishes response preferences through conversation, then uses session history, personal memories, and relevant documents to answer. See [Installation and running](installation.md) for installation and model preparation.

## First-time setup

```bash
uv run sla
uv run sla setup
```

The first interactive session collects your language, tone, response length, and daily learning window. You can provide multiple settings in one sentence; the window must include start and end times and a time zone. For example:

> Please use Chinese and keep answers brief; allow learning every day from midnight to 6 a.m., in the Asia/Shanghai time zone.

The wizard asks separately for network access and background learning permission. Draft settings can be resumed after an interruption, and changing one item preserves the others already entered.

`init` only initializes configuration and the database. JSON and non-interactive commands do not open the wizard; non-interactive `setup` returns exit code 2. First-time setup with `chat --offline` does not configure search or download a model, so an existing local model is required.

## Chat and configuration

```bash
uv run sla chat "Explain connection pools"
uv run sla chat --session SESSION_ID "Continue the previous topic"
uv run sla profile set length "brief"
uv run sla profile show
uv run sla config show
```

`chat` starts an interactive session when no message is supplied. `--session ID` resumes a session; `--model NAME` or `--model-dir PATH` selects a model, and `--json` is intended for scripts. Answers are displayed after the entire response has been generated. Initial model loading or switching back from training to inference requires a wait.

| Interactive command | Purpose |
| --- | --- |
| `/help`, `/info` | Help and configuration |
| `/clear`, `/reset` | Start a new short-term session without deleting long-term memories |
| `/setup` | Change settings |
| `/learning` | Learning status |
| `/exit`, `/quit` | Exit |

`profile show/set` manages name, tone, length, and language, saved separately by user-id. `config set KEY VALUE` supports online, learning, max_new_tokens, search.searxng_url, and search.trusted_hosts; boolean values use true/false. See [Web verification](web-verification.md) for networking options and [Privacy and data](privacy-and-data.md#session-controls) for private mode and learning exclusions.

Place global arguments before the subcommand, for example: `uv run sla --user-id alice --project-id demo --database /tmp/demo.db chat "Hello"`. Personal memories and the knowledge base are isolated by user and shared across projects; legacy project facts are additionally isolated by project/workspace. Use `uv run sla COMMAND --help` for all arguments.

## Answers and sessions

Answers use the current question, the most recent six conversation rounds, active personal memories, and relevant knowledge passages. An explicit language request in the current message takes precedence, followed by the configured language; without a configured preference, answers follow the input language. Original quotations remain unchanged.

See [Personal memory](personal-memory.md) to view, correct, or delete long-term memories, and [Knowledge base](knowledge-base.md) to import and cite documents. Starting a new session does not delete this content.

Command errors are written to stderr. Exit codes are 0 for success, 1 for execution failure, 2 for configuration/input errors, 3 for verification that is deferred, refuted, or conflicting, and 130 for user interruption.
