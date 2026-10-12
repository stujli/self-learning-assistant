from __future__ import annotations

import copy
import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..environment import environment, variable_name


class ProductError(RuntimeError):
    """A user-facing configuration or application error."""


def _base(variable: str, fallback: Path) -> Path:
    value = os.environ.get(variable, "")
    return Path(value) if value and Path(value).is_absolute() else fallback


@dataclass(frozen=True)
class Paths:
    config: Path
    data: Path
    state: Path
    cache: Path

    @classmethod
    def discover(cls) -> Paths:
        root = environment("HOME")
        if root:
            if not Path(root).is_absolute():
                raise ProductError(f"{variable_name('HOME')} must be an absolute path")
            return cls(
                *(Path(root) / part for part in ("config", "data", "state", "cache"))
            )
        home = Path.home()
        bases = (
            _base("XDG_CONFIG_HOME", home / ".config"),
            _base("XDG_DATA_HOME", home / ".local/share"),
            _base("XDG_STATE_HOME", home / ".local/state"),
            _base("XDG_CACHE_HOME", home / ".cache"),
        )
        current = cls(*(base / "self-learning-assistant" for base in bases))
        previous = cls(*(base / "auto-evolve" for base in bases))
        # Reuse the whole existing layout, so configuration and memory stay together.
        if not current.has_data() and previous.has_data():
            return previous
        return current

    def has_data(self) -> bool:
        return (self.config / "settings.json").is_file() or self.database.is_file()

    @property
    def database(self) -> Path:
        current = self.data / "self_learning_assistant.db"
        previous = self.data / "auto_evolve.db"
        return previous if not current.exists() and previous.is_file() else current

    def initialize(self) -> None:
        for directory in (self.config, self.data, self.state, self.cache):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)


DEFAULTS: dict[str, Any] = {
    "version": 2,
    "selected_model": "",
    "models": {},
    "profiles": {},
    "learning": True,
    "online": False,
    "max_new_tokens": 512,
    "search": {"searxng_url": "http://127.0.0.1:8080", "trusted_hosts": []},
    "onboarding": {},
    "schedules": {},
    "managed": False,
    "parameter_learning": False,
    "learning_since": {},
}
PROFILE = {
    "name": "自我学习助手",
    "tone": "友好、清晰",
    "length": "适中",
    "language": "",
}


class Settings:
    def __init__(self, paths: Paths | None = None) -> None:
        self.paths = paths or Paths.discover()
        self.file = self.paths.config / "settings.json"

    def read(self) -> dict[str, Any]:
        result = copy.deepcopy(DEFAULTS)
        if self.file.exists():
            try:
                value = json.loads(self.file.read_text(encoding="utf-8"))
                result = self.upgrade(value)
                self.validate(result)
            except (ValueError, TypeError) as exc:
                raise ProductError(f"Invalid settings: {exc}") from exc
        return result

    @staticmethod
    def upgrade(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict) or value.get("version") not in {1, 2}:
            raise ProductError("Unsupported settings version")
        if value["version"] == 2:
            Settings.validate(value)
            return copy.deepcopy(value)
        legacy = set(DEFAULTS) - {
            "onboarding",
            "schedules",
            "managed",
            "parameter_learning",
            "learning_since",
        }
        if set(value) - legacy:
            raise ProductError("Unsupported legacy settings fields")
        result = copy.deepcopy(DEFAULTS)
        result.update(value)
        result["version"] = 2
        Settings.validate(result)
        return result

    @staticmethod
    def validate(value: dict[str, Any]) -> None:
        if set(value) != set(DEFAULTS) or value.get("version") != 2:
            raise ProductError("Unsupported settings fields or version")
        for key in ("learning", "online", "managed", "parameter_learning"):
            if not isinstance(value.get(key), bool):
                raise ProductError(f"{key} must be true or false")
        for key in (
            "models",
            "profiles",
            "search",
            "onboarding",
            "schedules",
            "learning_since",
        ):
            if not isinstance(value.get(key), dict):
                raise ProductError(f"{key} must be an object")
        tokens = value.get("max_new_tokens")
        if type(tokens) is not int or not 32 <= tokens <= 8192:
            raise ProductError("max_new_tokens must be an integer between 32 and 8192")
        if not isinstance(value.get("selected_model"), str):
            raise ProductError("selected_model must be a string")
        for name, spec in value["models"].items():
            if (
                not isinstance(name, str)
                or not name.strip()
                or not isinstance(spec, dict)
            ):
                raise ProductError("Invalid model configuration")
            backend = spec.get("backend")
            if backend == "api":
                if not {"backend", "url", "model"} <= set(spec) or set(spec) - {
                    "backend",
                    "url",
                    "model",
                    "timeout",
                }:
                    raise ProductError("Invalid API model fields")
                from .model import validate_endpoint

                if not isinstance(spec["url"], str):
                    raise ProductError("Model URL must be a string")
                validate_endpoint(spec["url"])
                if not isinstance(spec["model"], str) or not spec["model"].strip():
                    raise ProductError("API model ID must be a nonempty string")
                timeout = spec.get("timeout", 120)
                if type(timeout) not in {int, float} or not 1 <= timeout <= 300:
                    raise ProductError("Model HTTP timeout must be between 1 and 300")
            elif backend == "local":
                if not {"backend", "path"} <= set(spec) or set(spec) - {
                    "backend",
                    "path",
                    "device",
                    "dtype",
                    "load_in_4bit",
                }:
                    raise ProductError("Invalid local model fields")
                if (
                    not isinstance(spec["path"], str)
                    or not Path(spec["path"]).is_absolute()
                ):
                    raise ProductError("Model path must be absolute")
                if spec.get("device", "auto") not in {"auto", "cpu", "cuda", "mps"}:
                    raise ProductError("Invalid model device")
                if spec.get("dtype", "auto") not in {
                    "auto",
                    "float32",
                    "float16",
                    "bfloat16",
                }:
                    raise ProductError("Invalid model dtype")
                if type(spec.get("load_in_4bit", False)) is not bool:
                    raise ProductError("load_in_4bit must be boolean")
            else:
                raise ProductError("Unknown model backend")
        for user, profile in value["profiles"].items():
            if (
                not isinstance(user, str)
                or not isinstance(profile, dict)
                or set(profile) - set(PROFILE)
            ):
                raise ProductError("Invalid profile fields")
            if any(
                not isinstance(item, str) or not item.strip() or len(item) > 500
                for item in profile.values()
            ):
                raise ProductError(
                    "Profile values must be nonempty strings up to 500 characters"
                )
        search = value["search"]
        if set(search) != {"searxng_url", "trusted_hosts"}:
            raise ProductError(
                "Unsupported search fields; credentials belong in environment variables"
            )
        from urllib.parse import urlsplit

        if not isinstance(search["searxng_url"], str):
            raise ProductError("SearXNG URL must be a string")
        url = urlsplit(search["searxng_url"])
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ProductError(
                "SearXNG URL must be HTTP(S) without credentials, query or fragment"
            )
        hosts = search["trusted_hosts"]
        if not isinstance(hosts, list) or any(
            not isinstance(host, str)
            or not host.strip()
            or any(char in host for char in "/:@ ")
            for host in hosts
        ):
            raise ProductError("trusted_hosts must contain host names")
        from .schedule import Window

        for user, window in value["schedules"].items():
            if not isinstance(user, str) or not user.strip():
                raise ProductError("Invalid schedule user")
            Window.parse(window)
        for user, state in value["onboarding"].items():
            if (
                not isinstance(user, str)
                or not isinstance(state, dict)
                or set(state) != {"complete", "draft"}
                or type(state["complete"]) is not bool
                or not isinstance(state["draft"], dict)
            ):
                raise ProductError("Invalid onboarding state")
            if set(state["draft"]) - {
                "language",
                "tone",
                "length",
                "start",
                "end",
                "timezone",
            } or any(
                not isinstance(item, str) or len(item) > 500
                for item in state["draft"].values()
            ):
                raise ProductError("Invalid onboarding draft")
        from datetime import datetime

        for user, since in value["learning_since"].items():
            if (
                not isinstance(user, str)
                or not isinstance(since, str)
                or datetime.fromisoformat(since).tzinfo is None
            ):
                raise ProductError("Invalid learning start timestamp")

    @contextmanager
    def edit(self) -> Iterator[dict[str, Any]]:
        self.paths.initialize()
        with (self.paths.config / "settings.lock").open("a", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            value = self.read()
            yield value
            self.validate(value)
            descriptor, name = tempfile.mkstemp(dir=self.paths.config)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(value, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(name, self.file)
            finally:
                if Path(name).exists():
                    Path(name).unlink()

    def profile(self, user: str) -> dict[str, str]:
        return {**PROFILE, **self.read()["profiles"].get(user, {})}

    def effective(self) -> dict[str, Any]:
        value = self.read()
        for key in ("online", "learning"):
            variable = variable_name(key.upper())
            raw = environment(key.upper())
            if raw is not None:
                raw = raw.lower()
                if raw not in {"true", "false", "1", "0"}:
                    raise ProductError(f"{variable} must be true/false or 1/0")
                value[key] = raw in {"true", "1"}
        model = environment("MODEL")
        if model is not None:
            value["selected_model"] = model
        search_url = environment("SEARXNG_URL")
        if search_url is not None:
            value["search"]["searxng_url"] = search_url
        return value
