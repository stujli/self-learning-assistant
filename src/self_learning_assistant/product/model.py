from __future__ import annotations

import ipaddress
import json
import threading
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from jsonschema import Draft202012Validator

from ..domain import ChatMessage
from ..model import ChatModel
from ..prompts import json_instruction as json_instruction
from .config import ProductError, Settings


def validate_endpoint(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ProductError(
            "Model URL must be a loopback HTTP(S) endpoint without credentials"
        )
    host = parsed.hostname.lower()
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback:
        raise ProductError("Only a local loopback model service is supported")
    # Pin localhost rather than trusting DNS; never use proxy environment settings.
    if host == "localhost":
        url = url.replace("localhost", "127.0.0.1", 1)
    return url.rstrip("/")


class LocalAPIModel:
    def __init__(
        self,
        url: str,
        model: str,
        *,
        timeout: float = 120,
        client: httpx.Client | None = None,
    ) -> None:
        self.url = validate_endpoint(url)
        self.model = model
        self.timeout = timeout
        self.client = client
        self._lock = threading.Lock()
        self._active_id: str | None = None

    def cancel_current(self) -> None:
        identifier = self._active_id
        if identifier is None:
            return

        def cancel() -> None:
            # The bundled service supports owner-scoped cancellation. Other
            # local servers may ignore it and remain bounded by HTTP timeout.
            try:
                with httpx.Client(trust_env=False, timeout=2) as client:
                    client.post(
                        self.url + "/cancel",
                        headers={"X-AE-Research-ID": identifier},
                    )
            except httpx.HTTPError:
                pass

        threading.Thread(target=cancel, daemon=True).start()

    def generate(
        self, messages: Sequence[ChatMessage], *, max_new_tokens: int = 128
    ) -> str:
        return self._request(messages, max_new_tokens)

    def generate_json(
        self,
        messages: Sequence[ChatMessage],
        schema: dict[str, Any],
        *,
        max_new_tokens: int = 1024,
    ) -> dict[str, Any]:
        prepared = list(messages)
        prepared.insert(
            0,
            {
                "role": "system",
                "content": json_instruction(schema),
            },
        )
        return decode_object(self._request(prepared, max_new_tokens, schema))

    def _request(
        self,
        messages: Sequence[ChatMessage],
        tokens: int,
        schema: dict[str, Any] | None = None,
    ) -> str:
        with self._lock:
            self._active_id = uuid.uuid4().hex
            try:
                return self._send(messages, tokens, schema, self._active_id)
            finally:
                self._active_id = None

    def _send(
        self,
        messages: Sequence[ChatMessage],
        tokens: int,
        schema: dict[str, Any] | None,
        identifier: str,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": tokens,
            "stream": False,
            "temperature": 0,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": True, "schema": schema},
            }
        try:
            if self.client:
                response = self.client.post(
                    self.url + "/chat/completions",
                    json=payload,
                    headers={"X-AE-Research-ID": identifier},
                    timeout=self.timeout,
                    follow_redirects=False,
                )
            else:
                with httpx.Client(trust_env=False, follow_redirects=False) as client:
                    response = client.post(
                        self.url + "/chat/completions",
                        json=payload,
                        headers={"X-AE-Research-ID": identifier},
                        timeout=self.timeout,
                    )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("empty or non-text response")
            return raw
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProductError(
                f"Local model request failed: {type(exc).__name__}"
            ) from exc


def decode_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```json") and raw.endswith("```"):
        raw = raw[7:-3].strip()
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise ProductError("Model returned invalid JSON; nothing was learned") from exc
    if not isinstance(value, dict):
        raise ProductError("Model JSON must be an object")
    return value


class DirectoryModel:
    """Serialize direct inference and cooperatively stop expired generations."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self._lock = threading.Lock()
        self._cancel = threading.Event()

    def warmup(self) -> None:
        with self._lock:
            self.backend.load()

    def generate(
        self, messages: Sequence[ChatMessage], *, max_new_tokens: int = 128
    ) -> str:
        with self._lock:
            self._cancel.clear()
            result = self.backend.generate_with_options(
                messages,
                max_new_tokens=max_new_tokens,
                enable_thinking=False,
                cancel_event=self._cancel,
            )
            if self._cancel.is_set():
                raise ProductError("Model generation cancelled")
            return str(result)

    def cancel_current(self) -> None:
        self._cancel.set()


def structured(
    model: ChatModel,
    messages: Sequence[ChatMessage],
    schema: dict[str, Any],
    *,
    max_new_tokens: int = 1024,
) -> dict[str, Any]:
    method = getattr(model, "generate_json", None)
    if method is not None:
        result: dict[str, Any] = method(messages, schema, max_new_tokens=max_new_tokens)
    else:
        prepared: list[ChatMessage] = [
            {
                "role": "system",
                "content": json_instruction(schema),
            },
            *messages,
        ]
        result = decode_object(model.generate(prepared, max_new_tokens=max_new_tokens))
    if not isinstance(result, dict) or list(
        Draft202012Validator(schema).iter_errors(result)
    ):
        raise ProductError("Model output failed schema validation; nothing was learned")
    return result


def create_model(
    settings: Settings,
    *,
    name: str | None = None,
    directory: Path | None = None,
    device: str = "auto",
    dtype: str = "auto",
    user: str = "default_user",
    personal: bool = True,
    database: Path | None = None,
) -> ChatModel:
    config = settings.effective()
    if directory:
        spec: dict[str, Any] = {"backend": "local", "path": str(directory)}
    else:
        selected = name or config["selected_model"]
        if selected not in config["models"]:
            raise ProductError(
                "No model configured. Use sla model add and sla model use first."
            )
        spec = config["models"][selected]
    if spec["backend"] == "api":
        return LocalAPIModel(
            spec["url"], spec["model"], timeout=float(spec.get("timeout", 120))
        )
    try:
        from ..model.transformers import TransformersModel
    except ImportError as exc:
        raise ProductError(
            "Model dependencies are unavailable. Run uv sync in the project directory."
        ) from exc
    path = Path(spec["path"]).expanduser().resolve()
    adapter = None
    from ..model.artifacts import resolve_active_model_artifact

    artifact = resolve_active_model_artifact(path)
    path, adapter = artifact.base_model_path, artifact.adapter_path
    if not personal:
        adapter = None
    elif config["parameter_learning"]:
        from .adaptation import active, model_fingerprint
        from .store import Store

        adapter = active(
            settings,
            Store(database or settings.paths.database, user),
            model_fingerprint(path),
        )
    return DirectoryModel(
        TransformersModel(
            path,
            adapter_dir=adapter,
            device=spec.get("device", device),
            dtype=spec.get("dtype", dtype),
            load_in_4bit=bool(spec.get("load_in_4bit", False)),
            enable_thinking=False,
        )
    )
