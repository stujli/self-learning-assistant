from __future__ import annotations

from collections.abc import Sequence

from .domain import (
    ChatMessage,
    ConflictConstraint,
    RetrievedMemory,
    WebEvidence,
)
from .prompts import (
    BASE_SYSTEM_PROMPT,
    conflict_context,
    project_memory_context,
    research_error_context,
    web_evidence_context,
)


def build_chat_messages(
    query: str,
    memories: Sequence[RetrievedMemory],
    *,
    history: Sequence[ChatMessage] = (),
    conflict_constraints: Sequence[ConflictConstraint] = (),
    web_evidence: Sequence[WebEvidence] = (),
    research_error: str = "",
) -> list[ChatMessage]:
    system_prompt = _build_system_prompt(
        memories,
        conflict_constraints,
        web_evidence,
        research_error,
    )
    messages: list[ChatMessage] = [{"role": "system", "content": system_prompt}]
    messages.extend(_valid_history(history))
    messages.append({"role": "user", "content": query})
    return messages


def _build_system_prompt(
    memories: Sequence[RetrievedMemory],
    conflict_constraints: Sequence[ConflictConstraint],
    web_evidence: Sequence[WebEvidence],
    research_error: str,
) -> str:
    memory_payload = {
        "project_facts": [
            {"subject": item.label, "value": item.value}
            for item in memories
            if item.kind == "project_fact"
        ],
    }
    system_prompt = BASE_SYSTEM_PROMPT + project_memory_context(memory_payload)
    if web_evidence:
        system_prompt += web_evidence_context([item.to_dict() for item in web_evidence])
    elif research_error:
        system_prompt += research_error_context(research_error)
    if conflict_constraints:
        system_prompt += conflict_context(
            [item.to_dict() for item in conflict_constraints]
        )
    return system_prompt


def _valid_history(history: Sequence[ChatMessage]) -> list[ChatMessage]:
    return [
        {"role": message["role"], "content": message["content"]}
        for message in history
        if message["role"] in {"user", "assistant"}
    ]
