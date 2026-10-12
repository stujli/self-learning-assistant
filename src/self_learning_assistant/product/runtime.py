"""Local model owner, foreground priority and window-bound training workers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import adaptation
from .config import ProductError, Settings
from .finetune import write_json
from .model import create_model, structured
from .schedule import Window, on_ac_power
from .store import Store

MAX_RPC = 8 * 1024 * 1024


def socket_path(settings: Settings) -> Path:
    identity = hashlib.sha256(
        str(settings.paths.config.resolve()).encode()
    ).hexdigest()[:16]
    root = Path("/tmp") / f"sla-{os.getuid()}-{identity}"
    if root.exists() and (root.is_symlink() or root.stat().st_uid != os.getuid()):
        raise ProductError("Unsafe runtime socket directory")
    root.mkdir(mode=0o700, exist_ok=True)
    os.chmod(root, 0o700)
    return root / "runtime.sock"


@contextmanager
def gpu_slot(*, root: Path | None = None) -> Iterator[None]:
    root = root or Path("/tmp") / f"sla-gpu-{os.getuid()}"
    if root.exists() and (root.is_symlink() or root.stat().st_uid != os.getuid()):
        raise ProductError("Unsafe GPU lock directory")
    root.mkdir(mode=0o700, exist_ok=True)
    os.chmod(root, 0o700)
    with (root / "cuda.lock").open("a") as handle:
        os.chmod(handle.name, 0o600)
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def rpc(settings: Settings, request: dict[str, Any], timeout: float = 600) -> Any:
    raw = json.dumps(request, ensure_ascii=False).encode() + b"\n"
    if len(raw) > MAX_RPC:
        raise ProductError("Runtime request exceeds limit")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(socket_path(settings)))
        client.sendall(raw)
        with client.makefile("rb") as reader:
            response = reader.readline(MAX_RPC + 1)
    if len(response) > MAX_RPC:
        raise ProductError("Runtime response exceeds limit")
    result = json.loads(response)
    if "error" in result:
        raise ProductError(result["error"])
    return result["result"]


def running(settings: Settings) -> bool:
    try:
        return bool(rpc(settings, {"operation": "ping"}, timeout=1))
    except (OSError, ProductError, ValueError):
        return False


def ensure(settings: Settings) -> None:
    if running(settings):
        return
    settings.paths.initialize()
    log = settings.paths.state / "runtime.log"
    descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a") as handle:
        subprocess.Popen(
            [sys.executable, "-m", "self_learning_assistant", "daemon", "run"],
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=handle,
            start_new_session=True,
        )
    for _ in range(100):
        if running(settings):
            return
        time.sleep(0.1)
    raise ProductError("Local runtime did not start; inspect sla daemon status")


class RuntimeModel:
    def __init__(self, settings: Settings, user: str, *, personal: bool = True) -> None:
        self.settings, self.user, self.personal = settings, user, personal

    def warmup(self) -> None:
        try:
            rpc(
                self.settings,
                {"operation": "warmup", "user": self.user, "personal": self.personal},
            )
        except ProductError as exc:
            if str(exc) != "Unknown runtime operation":
                raise
            # Compatibility with an already running daemon from an older version.
            self.generate(
                [{"role": "user", "content": "Reply with OK."}], max_new_tokens=1
            )

    def generate(self, messages: Any, *, max_new_tokens: int = 128) -> str:
        result = rpc(
            self.settings,
            {
                "operation": "generate",
                "user": self.user,
                "personal": self.personal,
                "messages": messages,
                "tokens": max_new_tokens,
            },
        )
        return str(result)

    def generate_json(
        self, messages: Any, schema: dict[str, Any], *, max_new_tokens: int = 1024
    ) -> dict[str, Any]:
        result = rpc(
            self.settings,
            {
                "operation": "generate",
                "user": self.user,
                "personal": self.personal,
                "messages": messages,
                "tokens": max_new_tokens,
                "schema": schema,
            },
        )
        if not isinstance(result, dict):
            raise ProductError("Runtime returned invalid structured output")
        return result

    def cancel_current(self) -> None:
        try:
            rpc(self.settings, {"operation": "cancel"}, timeout=2)
        except (OSError, ProductError):
            pass


class Runtime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.lock = threading.RLock()
        self.model: Any = None
        self.model_key = ""
        self.slot: Any = None
        self.worker: subprocess.Popen[bytes] | None = None
        self.last_chat = time.monotonic()
        self.last_scheduled_user = ""
        self.closed = threading.Event()
        self.pause = settings.paths.state / "pause-training"
        self.status: dict[str, Any] = {"state": "idle"}
        self.window_started = 0.0
        self.window_key = ""

    def budget(self, key: str) -> float:
        path = self.settings.paths.state / "learning-budget.json"
        value = json.loads(path.read_text()) if path.exists() else {}
        entry = value.get(key, {"used": 0.0})
        if entry.get("started") is not None:
            entry["used"] += max(0, time.time() - entry.pop("started"))
            value[key] = entry
            write_json(path, value)
        return max(0.0, 3600.0 - float(entry["used"]))

    def begin_budget(self, key: str) -> None:
        self.window_key = key
        self.window_started = time.monotonic()
        path = self.settings.paths.state / "learning-budget.json"
        value = json.loads(path.read_text()) if path.exists() else {}
        value[key] = {"used": 3600 - self.budget(key), "started": time.time()}
        write_json(path, dict(list(value.items())[-20:]))

    def end_budget(self) -> None:
        path = self.settings.paths.state / "learning-budget.json"
        if path.exists() and self.window_key:
            value = json.loads(path.read_text())
            entry = value.get(self.window_key)
            if entry and "started" in entry:
                entry["used"] += max(0, time.monotonic() - self.window_started)
                entry.pop("started")
                write_json(path, value)

    def unload(self) -> None:
        if self.model is not None:
            backend = getattr(self.model, "backend", None)
            if backend is not None:
                backend.unload()
            self.model = None
            self.model_key = ""
        if self.slot is not None:
            self.slot.__exit__(None, None, None)
            self.slot = None

    def load(self, user: str, personal: bool) -> Any:
        value = self.settings.effective()
        spec = value["models"].get(value["selected_model"], {})
        store = Store(self.settings.paths.database, user) if personal else None
        # Revalidate provenance before every use, including already-loaded adapters.
        if personal:
            assert store is not None
            adaptation.invalidate(self.settings, store)
        record_key = (
            [(item["id"], item["state"]) for item in adaptation.runs(store)]
            if store is not None
            else []
        )
        pointers = []
        if store is not None:
            with store.connect() as connection:
                pointers = [
                    tuple(row)
                    for row in connection.execute(
                        "SELECT model_fingerprint,run_id FROM product_active_adapters WHERE user_id=? ORDER BY model_fingerprint",
                        (user,),
                    )
                ]
        key = json.dumps(
            [user, personal, spec, record_key, pointers, value["profiles"].get(user)]
        )
        if key != self.model_key:
            self.unload()
            if spec.get("backend") == "local" and spec.get("device") != "cpu":
                self.slot = gpu_slot()
                self.slot.__enter__()
            try:
                self.model = create_model(self.settings, user=user, personal=personal)
                self.model_key = key
            except Exception:
                self.unload()
                raise
        return self.model

    def foreground(self) -> None:
        self.last_chat = time.monotonic()
        self.pause.touch(mode=0o600)
        if self.status.get("state") == "extracting" and self.model is not None:
            self.model.cancel_current()
        if self.worker and self.worker.poll() is None:
            try:
                self.worker.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.worker.terminate()
                try:
                    self.worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.worker.kill()
                    self.worker.wait(timeout=5)
            self.end_budget()

    def request(self, request: dict[str, Any]) -> Any:
        operation = request.get("operation")
        if operation == "ping":
            return True
        if operation == "status":
            return self.status
        if operation == "shutdown":
            self.closed.set()
            self.foreground()
            return True
        if operation == "cancel":
            if self.model:
                self.model.cancel_current()
            return True
        if operation not in {"generate", "warmup"}:
            raise ProductError("Unknown runtime operation")
        user = request.get("user", "default_user")
        if (
            not isinstance(user, str)
            or not user.strip()
            or len(user) > 200
            or type(request.get("personal", True)) is not bool
        ):
            raise ProductError("Invalid runtime user")
        if operation == "generate" and (
            type(request.get("tokens")) is not int
            or not 1 <= request["tokens"] <= 8192
            or not isinstance(request.get("messages"), list)
        ):
            raise ProductError("Invalid runtime generation")
        self.foreground()
        with self.lock:
            self.status = {"state": "loading" if operation == "warmup" else "chatting"}
            model = None
            try:
                model = self.load(user, request.get("personal", True))
                if operation == "warmup":
                    method = getattr(model, "warmup", None)
                    if callable(method):
                        method()
                    return True
                if "schema" in request:
                    return structured(
                        model,
                        request["messages"],
                        request["schema"],
                        max_new_tokens=request["tokens"],
                    )
                return model.generate(
                    request["messages"], max_new_tokens=request["tokens"]
                )
            finally:
                self.last_chat = time.monotonic()
                self.status = {"state": "idle"}
                backend = getattr(model, "backend", None)
                if backend is not None and callable(
                    getattr(backend, "diagnostics", None)
                ):
                    try:
                        diagnostics = backend.diagnostics()
                        if isinstance(diagnostics, dict):
                            self.status["inference"] = diagnostics
                    except (OSError, RuntimeError):
                        self.status["inference_unavailable"] = True

    def tick(self) -> None:
        if self.worker and self.worker.poll() is None:
            return
        config = self.settings.effective()
        instant = datetime.now(UTC)
        if (
            not config["learning"]
            or not config["managed"]
            or time.monotonic() - self.last_chat < 120
        ):
            return
        if not on_ac_power():
            self.status = {"state": "waiting_for_ac_power"}
            return
        windows = list(config["schedules"].items())
        previous = next(
            (
                index
                for index, (user, _) in enumerate(windows)
                if user == self.last_scheduled_user
            ),
            -1,
        )
        windows = windows[previous + 1 :] + windows[: previous + 1]
        for user, raw in windows:
            window = Window.parse(raw)
            if not window.contains(instant):
                continue
            with self.lock:
                if time.monotonic() - self.last_chat < 120:
                    return
                self.last_scheduled_user = user
                self.pause.unlink(missing_ok=True)
                store = Store(self.settings.paths.database, user)
                window_id = window.identifier(user, instant)
                remaining = self.budget(window_id)
                if remaining <= 0:
                    self.status = {"state": "window_budget_exhausted"}
                    return
                from .service import run_jobs

                if any(item["state"] == "queued" for item in store.jobs()):
                    self.status = {"state": "extracting", "user": user}
                    self.begin_budget(window_id)

                    def extraction_stopped(
                        owner_user: str = user, available_seconds: float = remaining
                    ) -> bool:
                        current = self.settings.effective()
                        raw_window = current["schedules"].get(owner_user)
                        return (
                            self.pause.exists()
                            or self.closed.is_set()
                            or not current["learning"]
                            or not current["managed"]
                            or raw_window is None
                            or not Window.parse(raw_window).contains(datetime.now(UTC))
                            or not on_ac_power()
                            or time.monotonic() - self.window_started
                            >= available_seconds
                        )

                    try:
                        results = run_jobs(
                            self.settings,
                            store,
                            self.load(user, False),
                            limit=1,
                            stopped=extraction_stopped,
                        )
                    finally:
                        self.end_budget()
                    if self.pause.exists():
                        with store.connect() as connection:
                            for result in results:
                                if result["state"] == "failed":
                                    connection.execute(
                                        "UPDATE product_jobs SET state='queued',attempts=max(attempts-1,0),error='' WHERE id=? AND user_id=?",
                                        (result["id"], user),
                                    )
                    self.status = {"state": "processing_queue"}
                    return
                spec = config["models"].get(config["selected_model"], {})
                if (
                    not config["parameter_learning"]
                    or spec.get("backend") != "local"
                    or spec.get("device") == "cpu"
                ):
                    self.status = {
                        "state": "parameter_training_unavailable",
                        "user": user,
                    }
                    return
                self.unload()
                model_id = adaptation.model_fingerprint(Path(spec["path"]))
                adaptation.invalidate(self.settings, store)
                record = adaptation.start(self.settings, store, model_id)
                if record is None:
                    self.status = {
                        "state": "accumulating",
                        "user": user,
                        "sources": len(adaptation.sources(self.settings, store)),
                    }
                    return
                self.begin_budget(window_id)
                self.status = {"state": "training", "run": record["id"], "user": user}
                self.worker = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "self_learning_assistant",
                        "--user-id",
                        user,
                        "daemon",
                        "worker",
                        record["id"],
                        "--seconds",
                        str(remaining),
                    ]
                )
                return

    def schedule(self) -> None:
        while not self.closed.wait(1):
            try:
                if self.worker and self.worker.poll() is not None:
                    self.end_budget()
                    self.worker = None
                self.tick()
            except Exception as exc:
                self.status = {"state": "failed", "error": type(exc).__name__}
            self.closed.wait(
                1 if self.status.get("state") == "processing_queue" else 59
            )


def run(settings: Settings) -> None:
    settings.paths.initialize()
    path = socket_path(settings)
    with (settings.paths.state / "daemon.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProductError("Runtime is already running") from exc
        path.unlink(missing_ok=True)
        runtime = Runtime(settings)

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                self.connection.settimeout(600)
                try:
                    raw = self.rfile.readline(MAX_RPC + 1)
                    if len(raw) > MAX_RPC:
                        raise ProductError("Runtime request exceeds limit")
                    request = json.loads(raw)
                    if not isinstance(request, dict):
                        raise ProductError("Runtime request must be an object")
                    result = {"result": runtime.request(request)}
                except Exception as exc:
                    result = {
                        "error": str(exc)
                        if isinstance(exc, ProductError)
                        else type(exc).__name__
                    }
                self.wfile.write(
                    json.dumps(result, ensure_ascii=False).encode() + b"\n"
                )

        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True

        server = Server(str(path), Handler)
        os.chmod(path, 0o600)
        server.timeout = 1
        signal.signal(signal.SIGTERM, lambda *_: runtime.closed.set())
        threading.Thread(target=runtime.schedule, daemon=True).start()
        try:
            while not runtime.closed.is_set():
                server.handle_request()
        finally:
            runtime.closed.set()
            runtime.foreground()
            runtime.end_budget()
            with runtime.lock:
                runtime.unload()
            server.server_close()
            path.unlink(missing_ok=True)


def worker(
    settings: Settings, user: str, identifier: str, *, seconds: float = 3600
) -> None:
    from .evaluation import compare
    from .finetune import train

    store = Store(settings.paths.database, user)
    record = next(
        (item for item in adaptation.runs(store) if item["id"] == identifier), None
    )
    if (
        record is None
        or record["state"] not in {"paused", "pending"}
        or not adaptation.valid_sources(record, settings, store)
    ):
        raise ProductError("Training round is missing or revoked")
    config = settings.read()
    spec = config["models"][config["selected_model"]]
    if (
        spec["backend"] != "local"
        or adaptation.model_fingerprint(Path(spec["path"]))
        != record["model_fingerprint"]
    ):
        raise ProductError("Training model changed")
    path = Path(record["path"])
    root = (settings.paths.data / "adapters").resolve()
    if not path.resolve().is_relative_to(root):
        raise ProductError("Training path is outside the adapter store")
    data = json.loads((path / "dataset.json").read_text())
    started = time.monotonic()
    interrupted = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: interrupted.set())

    def stopped() -> bool:
        value = settings.effective()
        raw = value["schedules"].get(user)
        return (
            interrupted.is_set()
            or (settings.paths.state / "pause-training").exists()
            or not value["learning"]
            or not value["parameter_learning"]
            or raw is None
            or not Window.parse(raw).contains(datetime.now(UTC))
            or not on_ac_power()
            or time.monotonic() - started >= min(3600, seconds)
            or not adaptation.valid_sources(record, settings, store)
        )

    try:
        with gpu_slot():
            baseline = adaptation.active(settings, store, record["model_fingerprint"])
            result = train(
                Path(spec["path"]),
                path,
                data["training"],
                evaluation=data["evaluation"],
                baseline_adapter=baseline,
                stopped=stopped,
            )
            if result["state"] == "trained" and not stopped():
                probes = adaptation.evaluation_probes(data["evaluation"])
                if baseline:
                    previous = next(
                        item
                        for item in adaptation.runs(store)
                        if Path(item["path"]).resolve() == baseline
                    )
                    retained = [
                        item
                        for item in adaptation.sources(settings, store)
                        if item["id"] in previous["sources"]
                        and item["id"] not in record["sources"]
                    ]
                    _, old_evaluation = adaptation.dataset(retained)
                    probes += adaptation.evaluation_probes(old_evaluation)
                result["checks"] = compare(
                    Path(spec["path"]),
                    path,
                    probes,
                    baseline,
                    stopped=stopped,
                )
            if stopped():
                result["state"] = "paused"
        adaptation.finish(settings, store, record, result)
    except InterruptedError:
        adaptation.finish(settings, store, record, {"state": "paused"})
    except Exception as exc:
        adaptation.finish(
            settings, store, record, {"state": "failed", "error": type(exc).__name__}
        )
        raise


def unit_name(settings: Settings) -> str:
    return (
        "sla-"
        + hashlib.sha256(str(settings.paths.config.resolve()).encode()).hexdigest()[:12]
        + ".service"
    )


def install(settings: Settings) -> str:
    if not shutil_which_systemctl():
        raise ProductError("systemd user services are unavailable; use sla daemon run")
    root = (
        Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
        / "systemd/user"
    )
    root.mkdir(parents=True, exist_ok=True)

    def quote(value: object) -> str:
        return (
            '"'
            + str(value)
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("%", "%%")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            + '"'
        )

    environment = []
    for key in (
        "SLA_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "XDG_CACHE_HOME",
    ):
        if os.environ.get(key):
            environment.append("Environment=" + quote(key + "=" + os.environ[key]))
    name = unit_name(settings)
    content = (
        "[Unit]\nDescription=Self-Learning Assistant local runtime\n[Service]\nType=simple\nExecStart="
        + quote(sys.executable)
        + " -m self_learning_assistant daemon run\n"
        + "\n".join(environment)
        + "\nRestart=on-failure\nRestartSec=5\nUMask=0077\n[Install]\nWantedBy=default.target\n"
    )
    (root / name).write_text(content)
    (root / name).chmod(0o600)
    if running(settings):
        rpc(settings, {"operation": "shutdown"})
        for _ in range(100):
            if not running(settings):
                break
            time.sleep(0.1)
    for args in (["daemon-reload"], ["enable", "--now", name]):
        result = subprocess.run(
            ["systemctl", "--user", *args], capture_output=True, check=False
        )
        if result.returncode:
            raise ProductError(
                "Could not enable systemd user runtime; use sla daemon run or inspect systemctl --user"
            )
    return name


def shutil_which_systemctl() -> str | None:
    import shutil

    return shutil.which("systemctl")


def uninstall(settings: Settings) -> None:
    name = unit_name(settings)
    if shutil_which_systemctl():
        subprocess.run(["systemctl", "--user", "disable", "--now", name], check=False)
    (
        Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
        / "systemd/user"
        / name
    ).unlink(missing_ok=True)
    if shutil_which_systemctl():
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    if running(settings):
        rpc(settings, {"operation": "shutdown"})


def status(settings: Settings, user: str) -> dict[str, Any]:
    config = settings.effective()
    raw = config["schedules"].get(user)
    window = Window.parse(raw) if raw else None
    return {
        "running": running(settings),
        "unit": unit_name(settings),
        "next_learning": window.next_start(datetime.now(UTC)) if window else "",
        "parameter_learning": config["parameter_learning"],
        "runtime": rpc(settings, {"operation": "status"})
        if running(settings)
        else {"state": "stopped"},
        "history": adaptation.runs(Store(settings.paths.database, user)),
    }
