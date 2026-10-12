"""Loopback chat and Responses service for local text models.

Only text and the three research functions are supported. Hosted search, shell,
custom tools and multimodal inputs are deliberately not exposed to the model.
"""

from __future__ import annotations

import json
import re
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

from ..domain import ChatMessage
from ..prompts import json_instruction

ALLOWED_TOOLS = {"search_web", "fetch_web_page", "assess_evidence"}
Generator = Callable[[list[dict[str, Any]], list[dict[str, Any]]], str]


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in {
                "input_text",
                "output_text",
                "text",
            }:
                raise ValueError("local Responses bridge only supports text content")
            parts.append(str(part.get("text", "")))
        return "\n".join(parts)
    raise ValueError("invalid message content")


def prepare_request(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tools = []
    for item in (
        [] if payload.get("tool_choice") == "none" else payload.get("tools", [])
    ):
        if item.get("type") == "function" and item.get("name") in ALLOWED_TOOLS:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "description": item.get("description", ""),
                        "parameters": item.get("parameters", {}),
                    },
                }
            )
    instructions = str(payload.get("instructions") or "")
    schema = payload.get("text", {}).get("format", {}).get("schema")
    if schema and json.dumps(schema, ensure_ascii=False) not in instructions:
        instructions += "\n" + json_instruction(schema)
    messages: list[dict[str, Any]] = [{"role": "system", "content": instructions}]
    inputs = payload.get("input", [])
    if isinstance(inputs, str):
        inputs = [{"role": "user", "content": inputs}]
    for item in inputs:
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role", "user")
            text = _text(item.get("content", ""))
            if role in {"system", "developer"}:
                messages[0]["content"] += "\n" + text
            elif role in {"user", "assistant"}:
                messages.append({"role": role, "content": text})
            else:
                raise ValueError("unsupported message role")
        elif kind == "function_call":
            if item.get("name") not in ALLOWED_TOOLS:
                raise ValueError("unsupported function in history")
            arguments = item.get("arguments", "{}")
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": item["name"],
                                "arguments": json.loads(arguments),
                            }
                        }
                    ],
                }
            )
        elif kind == "function_call_output":
            messages.append({"role": "tool", "content": _text(item.get("output", ""))})
        elif kind != "reasoning":
            raise ValueError(f"unsupported Responses input item: {kind}")
    if not any(message["role"] == "user" for message in messages):
        raise ValueError("request requires a user message")
    return messages, tools


def parse_generation(raw: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    matches = list(re.finditer(r"<tool_call>(.*?)</tool_call>", raw, re.DOTALL))
    if not matches:
        if "<tool_call" in raw or "<function=" in raw:
            raise ValueError("incomplete Qwen tool call")
        return [
            {
                "type": "message",
                "id": "msg_" + uuid.uuid4().hex,
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": raw, "annotations": []}],
            }
        ]
    if raw.count("<tool_call>") != len(matches) or raw.count("</tool_call>") != len(
        matches
    ):
        raise ValueError("incomplete Qwen tool call")
    if raw[matches[-1].end() :].strip():
        raise ValueError("unexpected suffix after Qwen tool call")
    definitions = {tool["function"]["name"]: tool["function"] for tool in tools}
    items = []
    for match in matches:
        function = re.fullmatch(
            r"\s*<function=([\w.-]+)>(.*?)</function>\s*", match[1], re.DOTALL
        )
        if function is None or function[1] not in definitions:
            raise ValueError("malformed or unauthorized Qwen tool call")
        arguments: dict[str, Any] = {}
        parameters = list(
            re.finditer(
                r"<parameter=([\w.-]+)>(.*?)</parameter>", function[2], re.DOTALL
            )
        )
        if re.sub(
            r"<parameter=[\w.-]+>.*?</parameter>", "", function[2], flags=re.DOTALL
        ).strip():
            raise ValueError("malformed Qwen parameters")
        schema = definitions[function[1]]["parameters"]
        for parameter in parameters:
            name, value = parameter[1], parameter[2].strip()
            definition = schema.get("properties", {}).get(name)
            if definition is None or name in arguments:
                raise ValueError("unknown or repeated Qwen parameter")
            if definition.get("type") != "string":
                value = json.loads(value)
            if "enum" in definition and value not in definition["enum"]:
                raise ValueError("invalid Qwen parameter value")
            arguments[name] = value
        if not set(schema.get("required", [])) <= arguments.keys():
            raise ValueError("missing required Qwen parameter")
        items.append(
            {
                "type": "function_call",
                "id": "fc_" + uuid.uuid4().hex,
                "call_id": "call_" + uuid.uuid4().hex,
                "name": function[1],
                "arguments": json.dumps(arguments, ensure_ascii=False),
                "status": "completed",
            }
        )
    return items


def response_events(response: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield {
        "type": "response.created",
        "response": {**response, "status": "in_progress", "output": []},
    }
    for index, item in enumerate(response["output"]):
        yield {
            "type": "response.output_item.added",
            "output_index": index,
            "item": {**item, "status": "in_progress"},
        }
        if item["type"] == "message":
            yield {
                "type": "response.output_text.delta",
                "item_id": item["id"],
                "output_index": index,
                "content_index": 0,
                "delta": item["content"][0]["text"],
            }
        else:
            yield {
                "type": "response.function_call_arguments.delta",
                "item_id": item["id"],
                "output_index": index,
                "delta": item["arguments"],
            }
        yield {"type": "response.output_item.done", "output_index": index, "item": item}
    yield {"type": "response.completed", "response": response}


class LocalResponsesServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        generator: Generator,
        *,
        trace_path: Path | None = None,
    ):
        self.generator = generator
        self.generation_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.active_research_id: str | None = None
        self.trace_path = trace_path
        if trace_path:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(address, ResponsesHandler)

    def trace(self, value: dict[str, Any]) -> None:
        if self.trace_path:
            with self.state_lock, self.trace_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {"at": datetime.now(UTC).isoformat(), **value},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def cancel(self, research_id: str) -> bool:
        with self.state_lock:
            cancel = getattr(self.generator, "cancel_current", None)
            if (
                research_id
                and research_id == self.active_research_id
                and callable(cancel)
            ):
                cancel()
                return True
            return False

    def generation_diagnostics(self) -> dict[str, Any]:
        if self.trace_path:
            snapshot = getattr(self.generator, "diagnostics", None)
            if callable(snapshot):
                try:
                    return cast(dict[str, Any], snapshot())
                except Exception:
                    return {"diagnostics_unavailable": True}
        return {}


class ResponsesHandler(BaseHTTPRequestHandler):
    server: LocalResponsesServer

    def log_message(self, format: str, *args: object) -> None:
        del format, args  # Request bodies can include private local conversations.

    def _json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ready", "backend": "local"})
        elif self.path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "self-learning-assistant-local",
                            "object": "model",
                            "owned_by": "local",
                        }
                    ],
                },
            )
        else:
            self._json(404, {"error": "unknown endpoint"})

    def do_POST(self) -> None:
        if self.path == "/v1/cancel":
            self._json(
                200,
                {
                    "cancelled": self.server.cancel(
                        self.headers.get("X-AE-Research-ID", "")
                    )
                },
            )
            return
        chat_completion = self.path == "/v1/chat/completions"
        if self.path not in {"/v1/responses", "/v1/chat/completions"}:
            self._json(404, {"error": "unknown endpoint"})
            return
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        research_id = self.headers.get("X-AE-Research-ID", "")
        diagnostic: dict[str, Any] = {
            "request_id": request_id,
            "research_id": research_id,
            "stage": self.headers.get("X-AE-Stage", "search"),
        }
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2_000_000:
                raise ValueError("request body exceeds limit or is missing")
            payload = json.loads(self.rfile.read(length))
            if chat_completion:
                if payload.get("stream") or payload.get("tools"):
                    raise ValueError(
                        "Local chat endpoint supports non-streaming text only"
                    )
                schema = (
                    payload.get("response_format", {})
                    .get("json_schema", {})
                    .get("schema")
                )
                translated = {
                    "input": payload.get("messages", []),
                    "instructions": "",
                    "tools": [],
                }
                if schema:
                    schema_text = json.dumps(schema, ensure_ascii=False)
                    inputs = payload.get("messages", [])
                    existing = (
                        "\n".join(
                            _text(item.get("content", ""))
                            for item in inputs
                            if isinstance(item, dict)
                            and item.get("role") in {"system", "developer"}
                        )
                        if isinstance(inputs, list)
                        else ""
                    )
                    if schema_text not in existing:
                        translated["instructions"] = json_instruction(schema)
                messages, tools = prepare_request(translated)
            else:
                messages, tools = prepare_request(payload)
            self.server.trace(
                {**diagnostic, "event": "request", "messages": messages, "tools": tools}
            )
            with self.server.generation_lock:
                generation_started = time.monotonic()
                diagnostic["resources_before"] = self.server.generation_diagnostics()
                with self.server.state_lock:
                    begin = getattr(self.server.generator, "begin_generation", None)
                    if callable(begin):
                        begin()
                    self.server.active_research_id = research_id
                try:
                    chat_generate = getattr(
                        self.server.generator, "generate_chat", None
                    )
                    if chat_completion and callable(chat_generate):
                        raw = chat_generate(
                            messages, int(payload.get("max_tokens", 512))
                        )
                    else:
                        raw = self.server.generator(messages, tools)
                finally:
                    diagnostic.update(
                        queue_seconds=round(generation_started - started, 3),
                        generation_seconds=round(
                            time.monotonic() - generation_started, 3
                        ),
                        resources_after=self.server.generation_diagnostics(),
                    )
                    with self.server.state_lock:
                        self.server.active_research_id = None
                diagnostic["raw_output"] = raw
            output = parse_generation(raw, tools)
            response = {
                "id": "resp_" + uuid.uuid4().hex,
                "object": "response",
                "status": "completed",
                "model": "self-learning-assistant-local",
                "output": output,
            }
            if chat_completion:
                if any(item["type"] != "message" for item in output):
                    raise ValueError("Unexpected tool output from text endpoint")
                response = {
                    "id": "chatcmpl_" + uuid.uuid4().hex,
                    "object": "chat.completion",
                    "model": "self-learning-assistant-local",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "".join(
                                    part["text"]
                                    for item in output
                                    for part in item["content"]
                                ),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            self.server.trace(
                {
                    **diagnostic,
                    "event": "invalid_output",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            )
            self._json(
                400,
                {
                    "error": {
                        "message": "Invalid local Responses request or tool output"
                    }
                },
            )
            return
        except Exception as exc:
            self.server.trace(
                {
                    **diagnostic,
                    "event": "generation_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            )
            self._json(500, {"error": {"message": "Local model generation failed"}})
            return
        self.server.trace(
            {
                **diagnostic,
                "event": "response",
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        )
        try:
            if not payload.get("stream"):
                self._json(200, response)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for event in response_events(response):
                self.wfile.write(
                    ("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode()
                )
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class TransformersGenerator:
    def __init__(self, model: Any, max_new_tokens: int):
        self.model, self.max_new_tokens = model, max_new_tokens
        self.cancel_event = threading.Event()
        self.prepared = False

    def begin_generation(self) -> None:
        # Reset under the server's state lock, before publishing the owner.
        self.cancel_event.clear()
        self.prepared = True

    def cancel_current(self) -> None:
        self.cancel_event.set()

    def diagnostics(self) -> dict[str, Any]:
        snapshot = getattr(self.model, "diagnostics", None)
        return cast(dict[str, Any], snapshot()) if callable(snapshot) else {}

    def generate_chat(self, messages: list[dict[str, Any]], tokens: int) -> str:
        if not 1 <= tokens <= 8192:
            raise ValueError("Invalid chat token budget")
        self.prepared = False
        return cast(
            str,
            self.model.generate_with_options(
                cast(list[ChatMessage], messages),
                max_new_tokens=min(tokens, self.max_new_tokens),
                enable_thinking=False,
                tools=[],
                cancel_event=self.cancel_event,
            ),
        )

    def __call__(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> str:
        if not self.prepared:
            self.cancel_event.clear()
        self.prepared = False
        return cast(
            str,
            self.model.generate_with_options(
                cast(list[ChatMessage], messages),
                max_new_tokens=self.max_new_tokens,
                enable_thinking=False,
                tools=tools,
                cancel_event=self.cancel_event,
            ),
        )


def transformers_generator(model: Any, *, max_new_tokens: int) -> TransformersGenerator:
    return TransformersGenerator(model, max_new_tokens)
