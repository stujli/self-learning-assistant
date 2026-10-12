# Self-Learning Assistant

A personal assistant that learns over time. It learns your preferences through conversation, stores memories and documents, and trains personal adapters during idle periods.

## Getting started

These steps are for local inference and parameter training on an NVIDIA GPU. Before you begin, prepare:

- **System and tools**: Linux and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- **Project source**: Clone this repository or download and extract it, then open a terminal in the project root containing `pyproject.toml`.
- **GPU and drivers**: A CUDA-capable NVIDIA GPU with working drivers. Check your setup with `nvidia-smi`.
- **Model and storage**: Use an existing model directory, or download the default Qwen3.5-4B model on first launch. The setup wizard estimates about 9 GB for the weights; allow additional space for dependencies, caches, and personal adapters. Downloading dependencies and models requires network access.

Install and launch from the project root:

```bash
uv sync
uv run sla
```

On first launch, select an existing model directory or approve downloading Qwen3.5-4B. Describe your preferences in the setup conversation, then start chatting. Use `sla setup` or `/setup` in chat to change your settings.

## Features

| Module | Features and guide |
| --- | --- |
| Chat and settings | Set response preferences and manage conversations and session history. [Guide](docs/chat-and-settings.md) |
| Personal memory | Extract preferences, background, and goals from your messages; confirm, correct, or delete memories. [Guide](docs/personal-memory.md) |
| Knowledge base | Import documents, search them locally, and cite original passages in answers. [Guide](docs/knowledge-base.md) |
| Web verification | Check public claims with your permission and record sources and evidence reports. [Guide](docs/web-verification.md) |
| Background learning | Process queued tasks during allowed idle periods, train and evaluate personal adapters, and pause or roll back learning. [Guide](docs/background-learning.md) |
| Privacy and data | Control what is saved and learned, use private sessions, and export, back up, or restore data. [Guide](docs/privacy-and-data.md) |

See [Installation and running](docs/installation.md) for model management, installation options, support scope, upgrades, and removal. Each guide covers usage, configuration, and limitations.

## Common commands

Run these commands from the project root. Replace example paths and IDs with your own values:

```bash
uv run sla memory list
uv run sla knowledge import /absolute/path/to/notes.md
uv run sla learn history
uv run sla chat --private --offline
uv run sla data backup /absolute/path/to/backup.zip
```
