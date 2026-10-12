from __future__ import annotations

import json
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest

from self_learning_assistant.cli.main import main
from self_learning_assistant.product.config import Paths, ProductError, Settings
from self_learning_assistant.product.data import (
    backup,
    export_user,
    migrate,
    restore,
    validate_archive,
)
from self_learning_assistant.product.model import (
    LocalAPIModel,
    create_model,
    decode_object,
    structured,
    validate_endpoint,
)
from self_learning_assistant.product.research import Researcher, bounded
from self_learning_assistant.product.service import (
    Assistant,
    learn_personal,
    run_jobs,
    save_research,
)
from self_learning_assistant.product.store import Store, split_text, tokens
from self_learning_assistant.tools.domain import FetchedPage, SearchHit, SearchResponse

CLAIM = "Python default arguments are evaluated once at function definition."
BODY = "Python default arguments are evaluated once at function definition, not again on each call."
URL = "https://docs.python.org/3/tutorial/controlflow.html"


class Model:
    def __init__(self) -> None:
        self.memory: list[dict[str, Any]] = []
        self.report: dict[str, Any] | None = None
        self.messages: list[Any] = []
        self.fail = False

    def generate(self, messages: Any, *, max_new_tokens: int = 128) -> str:
        self.messages.append(messages)
        return "Test response"

    def generate_json(
        self, messages: Any, schema: Any, *, max_new_tokens: int = 1024
    ) -> dict[str, Any]:
        text = json.dumps(messages, ensure_ascii=False)
        if self.fail:
            raise ProductError("test model unavailable")
        if "PERSONAL_MEMORY_EXTRACTION" in text:
            return {"items": self.memory}
        if "RESEARCH_QUERY_PLAN" in text:
            return {
                "neutral": ["Python defaults evaluation"],
                "counter": ["Python defaults exceptions"],
                "preferred_hosts": [],
            }
        if "RESEARCH_REPORT" in text:
            return self.report or {
                "claim": CLAIM,
                "verdict": "supported",
                "reason": "documented",
                "conditions": "Python",
                "unresolved": [],
                "evidence": [
                    {"source_id": "S1", "passage_id": "S1P1", "relation": "supports"}
                ],
            }
        if "TOPIC_CLAIMS" in text:
            return {"claims": [CLAIM]}
        if "WEB_ROUTE" in text:
            return {"needed": False, "query": ""}
        return {"ok": True}


class Web:
    def __init__(self) -> None:
        self.searches: list[str] = []
        self.fetches: list[str] = []
        self.fail_search = False
        self.fail_fetch = False
        self.body = BODY
        self.truncated = False

    def search(self, query: str, *, max_results: int = 4) -> SearchResponse:
        self.searches.append(query)
        if self.fail_search:
            raise OSError("offline")
        return SearchResponse(
            query, "fixture", (SearchHit("Python", URL, "", "fixture"),)
        )

    def fetch(self, url: str, *, max_chars: int = 20_000) -> FetchedPage:
        self.fetches.append(url)
        if self.fail_fetch:
            raise OSError("offline")
        return FetchedPage(
            url,
            url,
            "Python",
            self.body,
            "2026-10-09T00:00:00+00:00",
            "sha256:" + "a" * 64,
            "text/plain",
            self.truncated,
        )


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("SLA_HOME", str(tmp_path / "application"))
    for key in (
        "SLA_MODEL",
        "SLA_ONLINE",
        "SLA_LEARNING",
        "SLA_SEARXNG_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    return Settings()


@pytest.fixture
def store(settings: Settings) -> Store:
    return Store(settings.paths.database, "alice")


def preference() -> dict[str, Any]:
    return {
        "key": "answer_length",
        "value": "short",
        "kind": "preference",
        "explicit": True,
        "quote": "I prefer short answers",
    }


def test_xdg_and_config_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLA_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative")
    paths = Paths.discover()
    assert paths.data == tmp_path / "data/self-learning-assistant"
    assert paths.config == Path.home() / ".config/self-learning-assistant"
    monkeypatch.setenv("SLA_HOME", "relative")
    with pytest.raises(ProductError):
        Paths.discover()


def test_settings_atomic_validation_and_environment(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert settings.read()["online"] is False
    with settings.edit() as config:
        config["online"] = True
        config["profiles"]["alice"] = {"length": "short"}
    before = settings.file.read_bytes()
    with pytest.raises(ProductError), settings.edit() as config:
        config["max_new_tokens"] = -1
    assert settings.file.read_bytes() == before
    assert settings.profile("alice")["length"] == "short"
    assert settings.profile("bob")["length"] != "short"
    monkeypatch.setenv("SLA_ONLINE", "0")
    monkeypatch.setenv("SLA_MODEL", "selected")
    monkeypatch.setenv("SLA_SEARXNG_URL", "http://127.0.0.1:8088")
    assert not settings.effective()["online"]
    assert settings.effective()["selected_model"] == "selected"
    monkeypatch.setenv("SLA_LEARNING", "bad")
    with pytest.raises(ProductError):
        settings.effective()
    settings.file.write_text('{"version":2}')
    with pytest.raises(ProductError):
        settings.read()


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/v1",
        "http://localhost.evil/v1",
        "file:///tmp/model",
        "http://user:pass@127.0.0.1/v1",
        "http://127.0.0.1/v1?key=secret",
    ],
)
def test_model_endpoints_are_local(url: str) -> None:
    with pytest.raises(ProductError):
        validate_endpoint(url)


def test_api_model_schema_no_redirect_and_errors() -> None:
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"ok":true}'}}]}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model = LocalAPIModel("http://localhost:11434/v1", "local", client=client)
        assert model.url.startswith("http://127.0.0.1")
        assert model.generate_json(
            [{"role": "user", "content": "test"}], {"type": "object"}
        ) == {"ok": True}
        assert model.generate([{"role": "user", "content": "test"}]) == '{"ok":true}'
    assert captured[0]["response_format"]["type"] == "json_schema"
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                302, headers={"Location": "https://example.com"}
            )
        )
    ) as client:
        with pytest.raises(ProductError):
            LocalAPIModel("http://127.0.0.1/v1", "local", client=client).generate([])
    assert decode_object('```json\n{"ok":true}\n```') == {"ok": True}
    for raw in ("[]", "not JSON"):
        with pytest.raises(ProductError):
            decode_object(raw)

    class Plain:
        def generate(self, messages: Any, *, max_new_tokens: int = 128) -> str:
            return '{"ok":true}'

    assert structured(Plain(), [], {}) == {"ok": True}


def test_model_factory(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ProductError):
        create_model(settings)
    with settings.edit() as config:
        config["selected_model"] = "api"
        config["models"]["api"] = {
            "backend": "api",
            "url": "http://127.0.0.1:11434/v1",
            "model": "local",
        }
    assert isinstance(create_model(settings), LocalAPIModel)


def test_personal_learning_is_user_bound_and_source_checked(store: Store) -> None:
    model = Model()
    model.memory = [preference()]
    source = store.message("session", "user", "I prefer short answers")
    identifiers = learn_personal(store, model, "I prefer short answers", source)
    assert len(identifiers) == 1
    assert store.memory(identifiers[0])["status"] == "active"
    bob = Store(store.database, "bob")
    assert bob.memories() == []
    with pytest.raises(ProductError):
        bob.memory(identifiers[0])
    model.memory = [preference(), {**preference(), "quote": "fabricated"}]
    before = store.memories()
    with pytest.raises(ProductError):
        learn_personal(store, model, "I prefer short answers", source)
    assert store.memories() == before


def test_personal_extraction_repairs_quote_without_relaxing_validation(
    store: Store,
) -> None:
    """Chinese quotes must retain the exact original characters and whitespace."""

    class RepairModel(Model):
        attempts = 0

        def generate_json(
            self, messages: Any, schema: Any, *, max_new_tokens: int = 1024
        ) -> dict[str, Any]:
            self.attempts += 1
            self.messages.append(messages)
            quote = "我正在学习Python" if self.attempts == 1 else "我正在学习 Python"
            return {
                "items": [
                    {
                        **preference(),
                        "key": "learning_goal",
                        "value": "Python",
                        "kind": "goal",
                        "quote": quote,
                    }
                ]
            }

    prompt = "我正在学习 Python，目标是编写自动化脚本。"
    source = store.message("session", "user", prompt)
    model = RepairModel()
    identifiers = learn_personal(store, model, prompt, source)
    assert model.attempts == 2
    assert len(identifiers) == 1
    assert store.memory(identifiers[0])["quote"] == "我正在学习 Python"
    assert "character-for-character" in model.messages[-1][-1]["content"]


def test_confirm_correct_delete_and_do_not_relearn(store: Store) -> None:
    identifier = store.remember("tool", "Python", "preference", "pending", "", "test")
    assert identifier
    assert store.change_memory(identifier, status="active")["status"] == "active"
    assert store.change_memory(identifier, value="Rust")["value"] == "Rust"
    store.remember("tool", "C", "preference", "pending", "", "test")
    assert store.memory(identifier)["value"] == "Rust"
    store.delete_memory(identifier)
    assert store.remember("tool", "Rust", "preference", "active", "", "test") is None
    assert store.remember(
        "tool", "Rust", "preference", "active", "manual", "test", explicit=True
    )
    with pytest.raises(ProductError):
        store.change_memory(identifier)


def test_source_purge_removes_pending_jobs(store: Store) -> None:
    source = store.message("session", "user", "I prefer short answers")
    identifier = store.remember(
        "length", "short", "preference", "active", source, "I prefer short answers"
    )
    store.enqueue("personal", "I prefer short answers", source)
    assert identifier
    store.change_memory(identifier, value="detailed")
    store.delete_memory(identifier, purge_source=True)
    assert store.messages("session") == []
    assert store.jobs() == []


def test_documents_chinese_search_deduplicate_and_delete(
    store: Store, tmp_path: Path
) -> None:
    """Chinese text exercises segmentation and search alongside English text."""
    path = tmp_path / "notes.md"
    path.write_text(
        "# 项目\n数据库连接池设置为32。\nPython defaults are evaluated at definition."
    )
    identifier = store.import_file(path)
    assert store.import_file(path) == identifier
    assert store.search("连接池")[0]["document_id"] == identifier
    assert store.search("Python")[0]["location"].startswith("lines")
    assert Store(store.database, "bob").search("Python") == []
    assert store.document(identifier)["metadata"]["verification"] == "provided"
    assert store.search("!!!") == []
    store.delete_document(identifier)
    assert store.search("Python") == []
    assert path.exists()
    with pytest.raises(ProductError):
        store.document(identifier)
    assert "连接" in tokens("数据库连接池")
    assert len(split_text("x" * 4000)) == 3


@pytest.mark.parametrize(
    "suffix,raw",
    [(".txt", b"\xff"), (".md", b"x\x00x"), (".exe", b"text"), (".pdf", b"invalid")],
)
def test_import_errors(store: Store, tmp_path: Path, suffix: str, raw: bytes) -> None:
    path = tmp_path / ("input" + suffix)
    path.write_bytes(raw)
    with pytest.raises(ProductError):
        store.import_file(path)


def test_pdf_no_text_and_extraction(
    store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pypdf import PdfWriter

    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(path)
    with pytest.raises(ProductError, match="OCR"):
        store.import_file(path)
    from pypdf._page import PageObject

    monkeypatch.setattr(
        PageObject, "extract_text", lambda self: "Knowledge from PDF page."
    )
    identifier = store.import_file(path)
    assert store.search("Knowledge")[0]["location"] == "page 1"
    assert store.document(identifier)["chunks"]


def test_assistant_restarts_personalizes_and_private_writes_nothing(
    settings: Settings, store: Store
) -> None:
    model = Model()
    model.memory = [preference()]
    first = Assistant(settings, store, model, user="alice").respond(
        "I prefer short answers", session="s"
    )
    assert first["learned"]
    model.memory = []
    assistant = Assistant(settings, Store(store.database, "alice"), model, user="alice")
    second = assistant.respond("Hello", session="s", resume=True)
    assert second["memories_used"][0]["value"] == "short"
    assert "short" in model.messages[-1][0]["content"]
    assert len(store.messages("s")) == 4
    snapshot = store.database.read_bytes()
    private = Assistant(settings, None, model, user="alice")
    assert private.respond("Private message", private=True)["private"]
    assert store.database.read_bytes() == snapshot
    before = len(store.jobs())
    assistant.respond("Hello", learn=False)
    assert len(store.jobs()) == before


def test_learning_failure_does_not_block_answer_and_pause(
    settings: Settings, store: Store
) -> None:
    model = Model()
    model.fail = True
    result = Assistant(settings, store, model, user="alice").respond(
        "I prefer short answers"
    )
    assert result["answer"] and result["learning_error"]
    assert store.jobs()[0]["state"] == "failed"
    assert store.retry() == 1
    with settings.edit() as config:
        config["learning"] = False
    assert run_jobs(settings, store, model) == [{"state": "paused"}]
    assert store.jobs()[0]["state"] == "queued"
    with pytest.raises(ProductError):
        Assistant(settings, store, model).respond("secret", private=True)
    private = Assistant(settings, None, model).respond(
        "secret", private=True, online=True
    )
    assert private["answer"] and private["research"] is None
    assert "stayed offline" in private["research_error"]


def test_jobs_leases_and_isolation(store: Store) -> None:
    identifier = store.enqueue("personal", "some user message")
    assert Store(store.database, "bob").claim() == []
    assert len(store.claim()) == 1
    assert store.claim() == []
    with store.connect() as connection:
        connection.execute(
            "UPDATE product_jobs SET lease_until='2000-01-01' WHERE id=?", (identifier,)
        )
    assert len(store.claim()) == 1
    store.finish(identifier, "failed", {}, "error")
    assert store.retry() == 1
    with pytest.raises(ProductError):
        store.claim(limit=0)


def test_research_exact_passages_and_verified_storage(store: Store) -> None:
    model, web = Model(), Web()
    report = Researcher(model, web, trusted_hosts=["docs.python.org"]).verify(CLAIM)
    assert report["accepted"], report
    assert report["sources"][0]["quote"] in BODY
    assert len(web.fetches) == 1
    identifier = save_research(store, report)
    assert store.document(identifier)["metadata"]["verification"] == "supported"
    assert store.search("Python")


@pytest.mark.parametrize(
    "change",
    [
        "unknown_passage",
        "changed_claim",
        "unresolved",
        "refuted",
        "conflicting",
        "insufficient",
        "counter_evidence",
        "truncated",
        "single_source",
    ],
)
def test_research_never_accepts_invalid_or_insufficient_evidence(change: str) -> None:
    model, web = Model(), Web()
    model.report = {
        "claim": CLAIM,
        "verdict": "supported",
        "reason": "documented",
        "conditions": "Python",
        "unresolved": [],
        "evidence": [{"source_id": "S1", "passage_id": "S1P1", "relation": "supports"}],
    }
    trusted = ["docs.python.org"]
    if change == "unknown_passage":
        model.report["evidence"][0]["passage_id"] = "invented"
    elif change == "changed_claim":
        model.report["claim"] = "replacement"
    elif change == "unresolved":
        model.report["unresolved"] = ["missing condition"]
    elif change in {"refuted", "conflicting", "insufficient"}:
        model.report["verdict"] = change
    elif change == "counter_evidence":
        model.report["evidence"][0]["relation"] = "refutes"
    elif change == "truncated":
        web.truncated = True
    else:
        trusted = []
    report = Researcher(model, web, trusted_hosts=trusted).verify(CLAIM)
    assert report["accepted"] is False


def test_research_network_failures_and_deadline() -> None:
    web = Web()
    web.fail_search = True
    report = Researcher(Model(), web, trusted_hosts=["docs.python.org"]).verify(CLAIM)
    assert not report["accepted"]
    web.fail_search, web.fail_fetch = False, True
    assert not Researcher(Model(), web).verify(CLAIM)["accepted"]
    with pytest.raises(ProductError):
        bounded(lambda: time.sleep(0.1), time.monotonic() + 0.01)
    with pytest.raises(ProductError):
        bounded(lambda: None, time.monotonic() - 1)
    with pytest.raises(ProductError):
        Researcher(Model(), web, timeout=10, reserve=20)


def test_topic_and_personal_job_runner(settings: Settings, store: Store) -> None:
    model = Model()
    model.memory = [preference()]
    store.enqueue("personal", "I prefer short answers")
    assert run_jobs(settings, store, model)[0]["state"] == "done"
    store.enqueue("topic", CLAIM)
    assert run_jobs(settings, store, model)[0]["state"] == "failed"
    store.retry()
    with settings.edit() as config:
        config["online"] = True
    researcher = Researcher(model, Web(), trusted_hosts=["docs.python.org"])
    assert (
        run_jobs(settings, store, model, researcher=researcher)[0]["state"]
        == "verified"
    )
    assert store.documents()


def test_backup_restore_checksums_and_user_export(
    settings: Settings, store: Store, tmp_path: Path
) -> None:
    with settings.edit() as config:
        config["profiles"]["alice"] = {"name": "Ada"}
    identifier = store.remember(
        "language", "Python", "preference", "active", "", "test"
    )
    document = store.add_document(
        "Notes", "file", b"Python notes", [("Python notes", "line 1")]
    )
    store.message("s", "user", "hello")
    archive = backup(settings, store, tmp_path / "backup.zip")
    assert validate_archive(archive)
    assert identifier
    store.delete_memory(identifier)
    store.delete_document(document)
    safety = restore(settings, store, archive)
    assert safety.exists()
    assert store.memory(identifier)["value"] == "Python"
    assert store.search("Python")
    assert export_user(store)["messages"][0]["text"] == "hello"
    assert export_user(Store(store.database, "bob"))["memories"] == []
    with pytest.raises(ProductError):
        backup(settings, store, archive)
    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as handle:
        handle.writestr("../data.sqlite", "bad")
    before = store.database.read_bytes()
    with pytest.raises(ProductError):
        restore(settings, store, unsafe)
    assert store.database.read_bytes() == before
    with zipfile.ZipFile(archive, "a") as handle:
        handle.writestr("manifest.json", '{"version":2}')
    with pytest.raises(ProductError):
        validate_archive(archive)


def test_migration_is_copy_and_preserves_source(
    settings: Settings, store: Store, tmp_path: Path
) -> None:
    from self_learning_assistant.memory.sqlite import SqliteMemoryRepository

    source = tmp_path / "old.sqlite"
    SqliteMemoryRepository(source)
    before = source.read_bytes()
    assert migrate(settings, store, source).exists()
    assert source.read_bytes() == before
    assert store.memories() == []
    with pytest.raises(ProductError):
        migrate(settings, store, store.database)


def cli(settings: Settings, capsys: Any, arguments: list[str]) -> Any:
    assert main(arguments + ["--json"]) == 0
    output = capsys.readouterr()
    return json.loads(output.out)


def test_cli_setup_models_profiles_memory_knowledge_data(
    settings: Settings, tmp_path: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli(settings, capsys, ["init"])
    assert cli(settings, capsys, ["doctor"])["integrity"] == "ok"
    cli(settings, capsys, ["profile", "set", "name", "Ada"])
    assert cli(settings, capsys, ["profile", "show"])["name"] == "Ada"
    cli(settings, capsys, ["config", "set", "max_new_tokens", "256"])
    cli(
        settings,
        capsys,
        ["config", "set", "search.searxng_url", "http://127.0.0.1:8080"],
    )
    cli(settings, capsys, ["config", "set", "search.trusted_hosts", "docs.python.org"])
    assert cli(settings, capsys, ["config", "show"])["max_new_tokens"] == 256
    cli(
        settings,
        capsys,
        [
            "model",
            "add",
            "api",
            "--url",
            "http://127.0.0.1:11434/v1",
            "--api-model",
            "local",
        ],
    )
    cli(settings, capsys, ["model", "add", "direct", "--path", str(tmp_path)])
    cli(settings, capsys, ["model", "use", "api"])
    assert cli(settings, capsys, ["model", "list"])["selected"] == "api"
    model = Model()
    model.memory = [preference()]
    monkeypatch.setattr(
        "self_learning_assistant.cli.product.create_model",
        lambda *args, **kwargs: model,
    )
    cli(settings, capsys, ["model", "check"])
    assert cli(settings, capsys, ["doctor", "--check-model"])["model"]["structured"]
    first = cli(settings, capsys, ["chat", "I prefer short answers", "--session", "s"])
    identifier = first["learned"][0]
    assert cli(settings, capsys, ["memory", "list"])[0]["id"] == identifier
    cli(settings, capsys, ["memory", "edit", identifier, "detailed"])
    cli(settings, capsys, ["memory", "confirm", identifier])
    assert cli(settings, capsys, ["memory", "show", identifier])["value"] == "detailed"
    cli(settings, capsys, ["memory", "dismiss", identifier])
    cli(settings, capsys, ["memory", "delete", identifier, "--purge-source"])
    path = tmp_path / "notes.txt"
    path.write_text("Python knowledge from a private file.")
    document = cli(settings, capsys, ["knowledge", "import", str(path)])["id"]
    assert cli(settings, capsys, ["knowledge", "search", "Python"])
    assert cli(settings, capsys, ["knowledge", "list"])
    cli(settings, capsys, ["knowledge", "show", document])
    cli(settings, capsys, ["data", "export", str(tmp_path / "export.json")])
    cli(settings, capsys, ["data", "backup", str(tmp_path / "backup.zip")])
    cli(settings, capsys, ["knowledge", "delete", document])
    cli(settings, capsys, ["data", "restore", str(tmp_path / "backup.zip"), "--yes"])
    cli(settings, capsys, ["learn", "topic", CLAIM])
    assert cli(settings, capsys, ["learn", "status"])["jobs"]
    cli(settings, capsys, ["learn", "pause"])
    assert cli(settings, capsys, ["learn", "run"]) == [{"state": "paused"}]
    cli(settings, capsys, ["learn", "resume"])
    cli(settings, capsys, ["learn", "retry"])


@pytest.mark.parametrize(
    "arguments",
    [
        ["config", "set", "online", "bad"],
        ["config", "set", "unknown", "x"],
        ["profile", "set", "unknown", "x"],
        ["model", "use", "missing"],
        ["model", "add", "x", "--url", "http://127.0.0.1/v1"],
        ["model", "add", "x", "--path", "/no-such-model"],
        ["memory", "show", "missing"],
        ["knowledge", "show", "missing"],
        ["research", CLAIM],
        ["data", "restore", "/missing.zip"],
    ],
)
def test_cli_user_errors_have_json_and_nonzero_exit(
    settings: Settings, capsys: Any, arguments: list[str]
) -> None:
    assert main(arguments + ["--json"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]


def test_private_cli_creates_no_data(
    settings: Settings, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "self_learning_assistant.cli.product.create_model",
        lambda *args, **kwargs: Model(),
    )
    assert not settings.paths.data.exists()
    cli(settings, capsys, ["chat", "secret", "--private"])
    assert not settings.paths.data.exists()


def test_cli_interactive_resume_and_controls(
    settings: Settings, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "self_learning_assistant.cli.product.create_model",
        lambda *args, **kwargs: Model(),
    )
    inputs = iter(["/help", "/info", "/clear", "hello", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    assert main(["chat", "--no-learn", "--no-color"]) == 0
    assert "Test response" in capsys.readouterr().out


def test_direct_model_deadline_cancels_generation() -> None:
    from self_learning_assistant.product.model import DirectoryModel

    stopped = threading.Event()

    class Slow:
        def generate_with_options(self, messages: Any, **kwargs: Any) -> str:
            kwargs["cancel_event"].wait(1)
            stopped.set()
            return "incomplete"

    model = DirectoryModel(Slow())
    with pytest.raises(ProductError, match="deadline"):
        bounded(
            lambda: model.generate([]),
            time.monotonic() + 0.01,
            cancel=model.cancel_current,
        )
    assert stopped.wait(0.5)


def test_settings_reject_nested_credentials(settings: Settings) -> None:
    with pytest.raises(ProductError), settings.edit() as config:
        config["models"]["bad"] = {
            "backend": "api",
            "url": "http://127.0.0.1/v1",
            "model": "local",
            "api_key": "secret",
        }
    assert "bad" not in settings.read()["models"]


def test_global_equals_arguments_use_product_cli(
    settings: Settings, capsys: Any
) -> None:
    result = cli(settings, capsys, ["--user-id=alice", "init"])
    assert result["database"] == str(settings.paths.database)


def test_invalid_application_home_has_json_error(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setenv("SLA_HOME", "relative")
    assert main(["init", "--json"]) == 2
    assert "absolute" in json.loads(capsys.readouterr().err)["error"]


def test_first_person_english_auto_verification_keeps_personal_details_local(
    settings, store, monkeypatch
):
    class Routed(Model):
        def generate_json(self, messages, schema, **kwargs):
            return {"needed": True, "query": "private employer payroll"}

    def forbidden(*args):
        raise AssertionError("Personal details must not reach search")

    monkeypatch.setattr(
        "self_learning_assistant.product.service.web_service", forbidden
    )
    result = Assistant(settings, store, Routed(), user=store.user).respond(
        "I work at a private employer; is this salary fair?", online=True, learn=False
    )
    assert result["answer"] and result["research"] is None
    assert "kept local" in result["research_error"]
