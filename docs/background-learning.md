# Background learning

During permitted idle periods, background learning extracts personal memories, processes topic tasks, and trains a personal QLoRA adapter when conditions are met. Base model weights remain unchanged. Completing a task or reducing training loss does not mean a candidate has passed evaluation.

## Background service

First-time setup asks separately for background learning and networking permission. The learning window includes start and end times and a time zone; see [Chat and settings](chat-and-settings.md#first-time-setup).

```bash
uv run sla daemon install
uv run sla daemon status
uv run sla learn pause
uv run sla learn resume
```

daemon install/uninstall manages the application's systemd user service. Service files are stored in `systemd/user/` under XDG_CONFIG_HOME, and names are distinguished by configuration directory. The service continues after the terminal closes. Continuing after logout requires enabling linger yourself; check it with `loginctl show-user "$USER" -p Linger`. Without systemd, run `uv run sla daemon run` and keep the process running. Reinstall the service after changing the installation location or virtual environment. Uninstalling the service preserves personal data.

Background work runs inside the learning window, on AC power, and after at least two minutes of chat inactivity. Learning windows support crossing midnight and daylight saving time. Each window has a cumulative budget of at most one hour, preserved across service restarts. Work is not made up outside the window and does not wake a sleeping computer. Starting a chat requests a training pause, saves a checkpoint, and releases the GPU; training resumes when the conditions are met again.

## Learning queue

```bash
uv run sla learn status
uv run sla learn topic "Learn about a public topic"
uv run sla learn run --limit 10
uv run sla learn retry
```

The queue contains [personal memory](personal-memory.md) extraction and [web verification](web-verification.md) topic tasks. run processes the queue manually, and topic tasks require networking to be enabled. retry requeues failed/deferred tasks; interrupted tasks can recover after their lease expires. A topic marked verified means at least one claim passed; inspect the individual reports.

pause/resume controls both the queue and background parameter learning. Manually processing the queue does not start adapter training.

## Training and evaluation

Training a personal adapter requires at least five independent sources and twenty samples. Sources include active personal information permitted for learning after parameter learning was enabled, configured preferences, and verified stable knowledge. Pending memories, unverified private documents, excessively long content, and volatile facts are not automatically used for training. See [Privacy and data](privacy-and-data.md#session-controls) for `--no-learn` and private-session exclusions. Parameter training requires NVIDIA CUDA and a local model; API backends do not train parameters.

Each round starts from the base model, with checkpoints available for resuming within that round. A candidate must improve held-out target loss by at least 2% relative to the current version and pass checks for Chinese, English, explicit language requests, arithmetic, JSON, and document citations. With knowledge retrieval disabled, it must accurately recall at least one item of knowledge, achieve at least 80% personal-fact recall, and retain facts it could already recall. Candidates that fail do not replace the current version.

## Version management

```bash
uv run sla learn history
uv run sla learn rollback VERSION_ID
uv run sla daemon status --json
```

The two most recent valid accepted versions are retained. rollback selects only a valid version for the current user and current base model. States include pending, paused, accepted, rejected, failed, revoked, and retired.

Editing or deleting sources, or changing relevant preferences, revokes versions that depend on the old content and removes training materials. Source snapshots, model fingerprints, and file checksums are checked before loading. The status field `runtime.inference.generation` reports loading/generation times, token counts, and speed without conversation content.
