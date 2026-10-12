from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from typing import Any

from ..domain import ChatMessage, MemoryScope
from ..model import ChatModel
from ..prompting import build_chat_messages
from ..prompts import (
    personal_context,
    personal_memory_prompt,
    personal_retry_prompt,
    profile_context,
    research_error_context,
    research_result_context,
    topic_claims_prompt,
    web_route_prompt,
)
from ..retrieval import MemoryRetriever
from .config import ProductError, Settings
from .model import structured
from .research import Researcher, bounded, web_service
from .store import Store

MEMORY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["preference", "background", "goal"],
                    },
                    "explicit": {"type": "boolean"},
                    "quote": {"type": "string"},
                },
                "required": ["key", "value", "kind", "explicit", "quote"],
            },
        }
    },
    "required": ["items"],
}
CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {"type": "string"}}},
    "required": ["claims"],
    "additionalProperties": False,
}
ROUTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"needed": {"type": "boolean"}, "query": {"type": "string"}},
    "required": ["needed", "query"],
    "additionalProperties": False,
}


def learn_personal(
    store: Store,
    model: ChatModel,
    prompt: str,
    source_id: str,
    *,
    stopped: Callable[[], bool] = lambda: False,
) -> list[str]:
    import time

    messages: list[ChatMessage] = [
        {
            "role": "user",
            "content": personal_memory_prompt(
                [
                    {"key": item["key"], "value": item["value"]}
                    for item in store.memories()
                ],
                prompt,
            ),
        }
    ]
    deadline = time.monotonic() + 120
    for attempt in range(3):
        result = bounded(
            lambda: structured(model, messages, MEMORY_SCHEMA),
            deadline,
            cancel=getattr(model, "cancel_current", None),
            stopped=stopped,
        )
        try:
            items = validate_personal(result, prompt)
        except ProductError:
            if attempt == 2:
                raise
            messages.append({"role": "user", "content": personal_retry_prompt(result)})
            continue
        if stopped():
            raise InterruptedError("Personal extraction paused")
        return store.remember_many(items, source_id)
    raise ProductError("Personal memory extraction did not converge")


def validate_personal(result: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    items = result.get("items")
    if not isinstance(items, list) or len(items) > 10:
        raise ProductError("Invalid personal memory extraction")
    # Validate the entire selection before writing any candidate.
    for item in items:
        if not isinstance(item, dict) or set(item) != {
            "key",
            "value",
            "kind",
            "explicit",
            "quote",
        }:
            raise ProductError("Invalid personal memory fields")
        if (
            any(
                not isinstance(item[key], str) or not item[key].strip()
                for key in ("key", "value", "kind", "quote")
            )
            or type(item["explicit"]) is not bool
        ):
            raise ProductError("Invalid personal memory values")
        if (
            item["quote"] not in prompt
            or len(item["quote"]) < 4
            or len(item["key"]) > 200
            or len(item["value"]) > 2000
            or item["kind"] not in {"preference", "background", "goal"}
        ):
            raise ProductError("Personal memory lacks a valid user source")
    return items


class Assistant:
    def __init__(
        self,
        settings: Settings,
        store: Store | None,
        model: ChatModel,
        *,
        user: str = "default_user",
        scope: MemoryScope | None = None,
    ) -> None:
        self.settings, self.store, self.model = settings, store, model
        self.user = user
        self.scope = scope or MemoryScope(user_id=user)
        self.history: list[ChatMessage] = []

    def respond(
        self,
        query: str,
        *,
        session: str | None = None,
        private: bool = False,
        learn: bool = True,
        online: bool | None = None,
        web_query: str | None = None,
        resume: bool = False,
        defer_learning: bool = False,
    ) -> dict[str, Any]:
        if not query.strip() or len(query) > 100_000:
            raise ProductError("Message must contain 1 to 100000 characters")
        config = self.settings.effective()
        session = session or uuid.uuid4().hex
        if private and self.store is not None:
            raise ProductError("Private conversations require an ephemeral assistant")
        if resume and self.store and not self.history:
            self.history = [
                {"role": item["role"], "content": item["text"]}
                for item in self.store.messages(session)[-12:]
            ]
        memories = self.store.memories("active")[:20] if self.store else []
        chunks = self.store.search(query) if self.store else []
        legacy = []
        if self.store and not private:
            from ..memory.sqlite import SqliteMemoryRepository

            legacy = MemoryRetriever(
                SqliteMemoryRepository(self.store.database, scope=self.scope)
            ).retrieve(query)
        messages = build_chat_messages(query, legacy, history=self.history[-12:])
        messages[0]["content"] += profile_context(self.settings.profile(self.user))
        messages[0]["content"] += personal_context(memories, chunks)
        research: dict[str, Any] | None = None
        enabled = config["online"] if online is None else online
        research_error = ""
        if enabled and private and web_query is None:
            enabled = False
            research_error = "Private mode requires an explicit public query; this turn stayed offline."
        if enabled:
            route_query = web_query
            if route_query is None:
                try:
                    route = structured(
                        self.model,
                        [{"role": "user", "content": web_route_prompt(query)}],
                        ROUTE_SCHEMA,
                        max_new_tokens=256,
                    )
                except Exception as exc:
                    route = {}
                    research_error = (
                        f"Verification routing unavailable: {type(exc).__name__}"
                    )
                if route.get("needed") is True and isinstance(route.get("query"), str):
                    route_query = route["query"]
                    # Auto-routing must not invent a disclosure from private context.
                    if route_query.strip() and re.search(
                        r"我|本人|\b(?:i|my|mine|our|ours)\b", query, re.I
                    ):
                        route_query = None
                        research_error = "A personal question was kept local; provide a public --web-query to verify external facts."
            if route_query:
                try:
                    research = Researcher(
                        self.model,
                        web_service(self.settings),
                        trusted_hosts=config["search"].get("trusted_hosts", []),
                    ).verify(route_query)
                except Exception as exc:
                    research_error = f"Verification unavailable: {type(exc).__name__}"
                if research is None:
                    research = {
                        "accepted": False,
                        "verdict": "insufficient",
                        "reason": research_error,
                    }
                messages[0]["content"] += research_result_context(
                    {
                        key: research.get(key)
                        for key in (
                            "accepted",
                            "verdict",
                            "reason",
                            "conditions",
                            "sources",
                            "execution",
                        )
                    }
                )
        if research_error:
            messages[0]["content"] += research_error_context(research_error)
        source = self.store.message(session, "user", query) if self.store else ""
        if self.store:
            self.store.allow_learning(
                source,
                bool(
                    learn
                    and config["learning"]
                    and config["parameter_learning"]
                    and self.user in config["learning_since"]
                ),
            )
        answer = self.model.generate(messages, max_new_tokens=config["max_new_tokens"])
        if self.store:
            self.store.message(session, "assistant", answer)
        saved: list[str] = []
        learning_error = ""
        if self.store and learn and config["learning"]:
            job = self.store.enqueue("personal", query, source)
            if defer_learning:
                pass
            else:
                try:
                    saved = learn_personal(self.store, self.model, query, source)
                    self.store.finish(job, "done", {"memories": saved})
                except Exception as exc:
                    learning_error = str(exc)[:500]
                    self.store.finish(job, "failed", {}, learning_error)
        if self.store and research and research.get("accepted"):
            save_research(
                self.store,
                research,
                parameter_eligible=bool(
                    learn and config["learning"] and config["parameter_learning"]
                ),
            )
        self.history.extend(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer},
            ]
        )
        self.history = self.history[-12:]
        return {
            "session_id": session,
            "answer": answer,
            "memories_used": memories,
            "sources": chunks,
            "learned": saved,
            "learning_error": learning_error,
            "research": research,
            "research_error": research_error,
            "private": private,
        }


def save_research(
    store: Store, report: dict[str, Any], *, parameter_eligible: bool = False
) -> str:
    if not report.get("accepted"):
        raise ProductError("Only accepted research can enter verified knowledge")
    raw = json.dumps(report, ensure_ascii=False, sort_keys=True).encode()
    citations = report.get("sources", [])
    chunks = [(str(item["quote"]), str(item["url"])) for item in citations]
    return store.add_document(
        str(report["report"]["claim"]),
        "web-research",
        raw,
        chunks,
        kind="web",
        metadata={
            "verification": "supported",
            "conditions": report.get("conditions", ""),
            "sources": citations,
            "parameter_eligible": parameter_eligible,
        },
    )


def run_jobs(
    settings: Settings,
    store: Store,
    model: ChatModel,
    *,
    limit: int = 10,
    researcher: Researcher | None = None,
    stopped: Callable[[], bool] = lambda: False,
) -> list[dict[str, Any]]:
    config = settings.effective()
    if not config["learning"]:
        return [{"state": "paused"}]
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ProductError("Job limit must be between 1 and 100")
    results: list[dict[str, Any]] = []
    for _ in range(limit):
        if stopped():
            break
        claimed = store.claim(limit=1)
        if not claimed:
            break
        job = claimed[0]
        payload: dict[str, Any]
        try:
            if job["kind"] == "personal":
                payload = {
                    "memories": learn_personal(
                        store, model, job["prompt"], job["source_id"], stopped=stopped
                    )
                }
                state = "done"
            else:
                if not config["online"]:
                    raise ProductError(
                        "Online learning is disabled; enable it explicitly"
                    )
                claims = structured(
                    model,
                    [
                        {
                            "role": "user",
                            "content": topic_claims_prompt(job["prompt"]),
                        }
                    ],
                    CLAIM_SCHEMA,
                    max_new_tokens=512,
                ).get("claims")
                if (
                    not isinstance(claims, list)
                    or not claims
                    or len(claims) > 3
                    or any(
                        not isinstance(claim, str) or not claim.strip()
                        for claim in claims
                    )
                ):
                    raise ProductError("No valid atomic claims extracted")
                research = researcher or Researcher(
                    model,
                    web_service(settings),
                    trusted_hosts=config["search"].get("trusted_hosts", []),
                )
                reports = [research.verify(claim) for claim in claims]
                if stopped():
                    raise InterruptedError("Topic research paused")
                documents = [
                    save_research(
                        store,
                        report,
                        parameter_eligible=bool(config["parameter_learning"]),
                    )
                    for report in reports
                    if report.get("accepted")
                ]
                payload = {"reports": reports, "documents": documents}
                state = (
                    "verified"
                    if documents
                    else "failed"
                    if any(report["execution"]["failed"] for report in reports)
                    else "deferred"
                )
            store.finish(job["id"], state, payload)
        except InterruptedError:
            state, payload = "queued", {}
            with store.connect() as connection:
                connection.execute(
                    "UPDATE product_jobs SET state='queued',attempts=max(attempts-1,0),lease_until='',error='' WHERE id=? AND user_id=?",
                    (job["id"], store.user),
                )
        except Exception as exc:
            state, payload = "failed", {"error": str(exc)[:500]}
            store.finish(job["id"], state, payload, payload["error"])
        results.append({"id": job["id"], "state": state, **payload})
    return results
