# Personal memory

Personal memories store your preferences, background, and goals to maintain continuity across conversations. Unlike whole documents in the [knowledge base](knowledge-base.md), each memory has a stable key, value, status, and source, and can be confirmed, edited, or deleted individually.

## Extraction and confirmation

Conversations are saved locally. With managed background learning enabled, user messages permitted for learning are queued and processed during the specified window. Without background management, extraction runs after the answer with bounded execution. See [Background learning](background-learning.md) for conditions and pause controls.

Explicit self-reported statements can become active memories; inferences require confirmation. Extracted memories must include an exact quotation from the original message. Invalid formats or quotations are not saved as memories, and failed tasks remain available for retry. Pending inferences cannot overwrite an existing active memory.

## Common commands

```bash
uv run sla memory list
uv run sla memory show MEMORY_ID
uv run sla memory edit MEMORY_ID "Corrected content"
uv run sla memory confirm MEMORY_ID
uv run sla memory dismiss MEMORY_ID
uv run sla memory delete MEMORY_ID
```

Replace MEMORY_ID with an actual ID returned by list. edit updates the value and makes the memory active, confirm accepts a candidate, and dismiss deactivates a candidate. These operations are restricted to the current user.

Deletion removes the memory and pending learning tasks from its source, and records a key exclusion to prevent relearning the same content from old tasks. `memory delete MEMORY_ID --purge-source` also removes the original user message. External backups and exported files must be managed separately; see [Privacy and data](privacy-and-data.md).

Editing or deleting a source invalidates adapters that depend on its old content and removes derived training materials. See [Background learning](background-learning.md#version-management) for version handling.

## Memory and parameter learning

After a memory is saved, the assistant can use that record in later answers. Check learning history and evaluation results to determine whether it has also been learned through parameter training in a personal adapter. Saving a memory and completing parameter learning are different processes.
