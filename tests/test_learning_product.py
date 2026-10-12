from __future__ import annotations

import copy
import json
import threading
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from self_learning_assistant.cli.main import main
from self_learning_assistant.product import adaptation, runtime
from self_learning_assistant.product.config import DEFAULTS, ProductError, Settings
from self_learning_assistant.product.data import backup, restore, validate_archive
from self_learning_assistant.product.evaluation import check_model
from self_learning_assistant.product.finetune import fingerprint, write_json
from self_learning_assistant.product.onboarding import confirm, setup
from self_learning_assistant.product.schedule import Window, on_ac_power
from self_learning_assistant.product.service import Assistant
from self_learning_assistant.product.store import Store
from tests.test_product import Model


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("SLA_HOME", str(tmp_path / "app"))
    for key in ("SLA_MODEL", "SLA_ONLINE", "SLA_LEARNING", "AUTO_EVOLVE_HOME"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings()
    settings.paths.initialize()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"base weights")
    with settings.edit() as value:
        value["models"]["test"] = {
            "backend": "local",
            "path": str(model),
            "device": "cpu",
        }
        value["selected_model"] = "test"
        value["learning_since"]["alice"] = "2026-10-11T00:00:00+00:00"
        value["parameter_learning"] = True
        value["profiles"]["alice"] = {
            "language": "English",
            "length": "concise",
            "tone": "gentle",
        }
    return settings, Store(settings.paths.database, "alice"), model


def seed(store, count=4):
    for index in range(count):
        source = store.message("session", "user", f"I prefer option {index}")
        store.allow_learning(source, True)
        store.remember(
            f"option_{index}",
            str(index),
            "preference",
            "active",
            source,
            f"I prefer option {index}",
        )


def candidate(settings, store, model):
    seed(store)
    record = adaptation.start(settings, store, adaptation.model_fingerprint(model))
    assert record
    path = Path(record["path"])
    (path / "adapter_config.json").write_text("{}")
    (path / "adapter_model.safetensors").write_bytes(b"personal parameters")
    return record


def accept(settings, store, record):
    return adaptation.finish(
        settings,
        store,
        record,
        {
            "state": "trained",
            "improvement": 0.1,
            "checks": {"core_no_regression": True, "facts_no_regression": True},
        },
    )


def test_old_configuration_and_database_upgrade_preserves_data(app):
    settings, store, _ = app
    source = store.message("old", "user", "old conversation")
    old = {
        key: value
        for key, value in settings.read().items()
        if key
        not in {
            "onboarding",
            "schedules",
            "managed",
            "parameter_learning",
            "learning_since",
        }
    }
    old["version"] = 1
    settings.file.write_text(json.dumps(old))
    with store.connect() as connection:
        connection.execute("UPDATE product_schema SET version=1")
    migrated = Store(store.database, "alice")
    assert settings.read()["version"] == 2
    assert settings.read()["parameter_learning"] is False
    assert migrated.messages("old")[0]["id"] == source
    assert not migrated.eligible_message(source)
    with migrated.connect() as connection:
        assert (
            connection.execute("SELECT version FROM product_schema").fetchone()[0] == 2
        )


@pytest.mark.parametrize(
    "update",
    [
        {"managed": "yes"},
        {"parameter_learning": 1},
        {"schedules": []},
        {"schedules": {"alice": {"start": "25:00"}}},
        {"onboarding": {"alice": {"complete": "yes", "draft": {}}}},
        {"onboarding": {"alice": {"complete": False, "draft": {"command": "rm"}}}},
        {"learning_since": {"alice": "2026-01-01"}},
    ],
)
def test_invalid_new_configuration_rejected(update):
    config = copy.deepcopy(DEFAULTS)
    config.update(update)
    with pytest.raises((ProductError, ValueError)):
        Settings.validate(config)


@pytest.mark.parametrize(
    "start,end,instant,expected",
    [
        ("00:00", "06:00", "2026-10-10T16:00:00+00:00", True),
        ("00:00", "06:00", "2026-10-10T22:00:00+00:00", False),
        ("23:00", "06:00", "2026-10-10T15:00:00+00:00", True),
        ("23:00", "06:00", "2026-10-10T18:00:00+00:00", True),
        ("23:00", "06:00", "2026-10-10T06:00:00+00:00", False),
    ],
)
def test_learning_window_boundaries(start, end, instant, expected):
    window = Window.parse(
        {"start": start, "end": end, "timezone": "Asia/Shanghai", "enabled": True}
    )
    assert window.contains(datetime.fromisoformat(instant)) == expected
    assert window.next_start(datetime.fromisoformat(instant))


@pytest.mark.parametrize(
    "update",
    [
        {"start": "1:00"},
        {"end": "00:00"},
        {"timezone": "Mars"},
        {"enabled": "true"},
        {"surprise": True},
    ],
)
def test_window_validation(update):
    value = {"start": "00:00", "end": "06:00", "timezone": "UTC", "enabled": True}
    value.update(update)
    with pytest.raises(ProductError):
        Window.parse(value)


def test_dst_window_uses_real_instants():
    window = Window("01:00", "02:00", "America/New_York")
    assert window.contains(datetime.fromisoformat("2026-11-01T05:30:00+00:00"))
    assert window.contains(datetime.fromisoformat("2026-11-01T06:30:00+00:00"))
    skipped = Window("02:00", "03:00", "America/New_York")
    assert skipped.next_start(
        datetime.fromisoformat("2026-03-08T06:59:00+00:00")
    ).startswith("2026-03-09")
    assert Window("00:00", "01:00", "UTC", False).next_start(datetime.now(UTC)) == ""
    with pytest.raises(ProductError):
        window.contains(datetime(2026, 1, 1))


def test_power_detection(tmp_path):
    assert on_ac_power(tmp_path)
    battery = tmp_path / "battery"
    battery.mkdir()
    (battery / "type").write_text("Battery")
    assert not on_ac_power(tmp_path)
    ac = tmp_path / "ac"
    ac.mkdir()
    (ac / "online").write_text("1")
    assert on_ac_power(tmp_path)


class SetupModel:
    def __init__(self, fields):
        self.fields = iter(fields)

    def generate_json(self, messages, schema, **kwargs):
        return {"fields": next(self.fields), "reply": "Understood."}


def test_conversational_setup_and_resume(app):
    """Chinese preferences must survive interrupted onboarding and resume."""
    settings, _, _ = app
    model = SetupModel([{"language": "中文"}, {"tone": "温柔", "length": "简短"}])
    answers = iter(["中文", "温柔简短", "/exit"])
    with pytest.raises(ProductError, match="saved"):
        setup(
            settings,
            "alice",
            model=model,
            services=False,
            ask=lambda _: next(answers),
            say=lambda _: None,
        )
    assert settings.read()["onboarding"]["alice"]["draft"]["length"] == "简短"
    answers = iter(["凌晨零点到六点，北京时间", "y"])
    model = SetupModel(
        [{"start": "00:00", "end": "06:00", "timezone": "Asia/Shanghai"}]
    )
    setup(
        settings,
        "alice",
        model=model,
        services=False,
        ask=lambda _: next(answers),
        say=lambda _: None,
    )
    assert settings.read()["onboarding"]["alice"]["complete"]
    assert settings.profile("alice")["language"] == "中文"
    assert settings.read()["schedules"]["alice"]["end"] == "06:00"
    assert not settings.read()["managed"]


def test_setup_combined_answer_and_revision(app):
    settings, _, _ = app
    fields = {
        "language": "English",
        "tone": "gentle",
        "length": "short",
        "start": "23:00",
        "end": "06:00",
        "timezone": "Asia/Shanghai",
    }
    answers = iter(["Apply all settings", "y"])
    setup(
        settings,
        "alice",
        model=SetupModel([fields]),
        services=False,
        ask=lambda _: next(answers),
        say=lambda _: None,
    )
    answers = iter(["Use detailed answers", "y"])
    setup(
        settings,
        "alice",
        model=SetupModel([{"length": "detailed"}]),
        services=False,
        ask=lambda _: next(answers),
        say=lambda _: None,
    )
    assert settings.profile("alice")["length"] == "detailed"
    # Chinese confirmation is an intentionally supported input.
    assert confirm("ok", lambda _: "确认")
    assert not confirm("ok", lambda _: "no")


def test_setup_rejects_ambiguous_time_and_reasks(app):
    settings, _, _ = app
    fields = {
        "language": "English",
        "tone": "gentle",
        "length": "short",
        "start": "25:00",
        "end": "06:00",
        "timezone": "UTC",
    }
    answers = iter(["settings", "corrected", "y"])
    setup(
        settings,
        "alice",
        model=SetupModel(
            [fields, {"start": "00:00", "end": "06:00", "timezone": "UTC"}]
        ),
        services=False,
        ask=lambda _: next(answers),
        say=lambda _: None,
    )
    assert settings.read()["schedules"]["alice"]["start"] == "00:00"


def test_setup_english_copy_and_single_preference_question(app, monkeypatch):
    import locale

    settings, _, _ = app
    monkeypatch.setattr(locale, "getlocale", lambda: ("zh_CN", "UTF-8"))
    model = SetupModel(
        [
            {"language": "English"},
            {"tone": "gentle", "length": "concise"},
            {"start": "00:00", "end": "06:00", "timezone": "UTC"},
        ]
    )
    prompts, output = [], []
    answers = iter(
        ["English", "gentle and concise", "midnight to six UTC", "y", "n", "n"]
    )

    def ask(question):
        prompts.append(question)
        return next(answers)

    setup(settings, "alice", model=model, ask=ask, say=output.append)
    combined = "\n".join(output + prompts)
    assert output[0] == "Which language would you like us to use?"
    assert combined.count("What tone and answer length do you prefer") == 1
    assert "Confirm these settings?" in combined
    assert "Allow public web queries" in combined
    assert "Enable local background learning" in combined
    assert not any("\u3400" <= character <= "\u9fff" for character in combined)


def test_setup_passes_current_question_and_skips_known_length(app):
    settings, _, _ = app
    supplied = iter(
        [
            {"language": "English", "length": "short"},
            {"tone": "gentle"},
            {"start": "00:00", "end": "06:00", "timezone": "UTC"},
        ]
    )
    inputs, output = [], []

    class CaptureModel:
        def generate_json(self, messages, schema, **kwargs):
            inputs.append(messages[0]["content"])
            return {"fields": next(supplied), "reply": "What tone do you prefer?"}

    answers = iter(["English, short", "gentle", "midnight to six UTC", "y"])
    setup(
        settings,
        "alice",
        model=CaptureModel(),
        services=False,
        ask=lambda _: next(answers),
        say=output.append,
    )
    question = "What tone do you prefer, for example gentle or direct?"
    assert question in inputs[1]
    assert output.count(question) == 1
    assert "What tone do you prefer?" not in output
    assert not any("tone and answer length" in line for line in output)


def test_setup_summary_revision_preserves_preferences(app):
    settings, _, _ = app
    fields = {
        "language": "English",
        "tone": "gentle",
        "length": "short",
        "start": "00:00",
        "end": "06:00",
        "timezone": "UTC",
    }
    answers = iter(["all settings", "n", "make answers detailed", "y"])
    output = []
    setup(
        settings,
        "alice",
        model=SetupModel([fields, {"length": "detailed"}]),
        services=False,
        ask=lambda _: next(answers),
        say=output.append,
    )
    assert settings.profile("alice")["tone"] == "gentle"
    assert settings.profile("alice")["length"] == "detailed"
    assert not any("What tone" in line for line in output)


def test_setup_clarification_replaces_host_question(app):
    settings, _, _ = app
    outputs, responses = (
        [],
        iter(
            [
                {"fields": {}, "reply": "Which conversation language do you mean?"},
                {
                    "fields": {
                        "language": "English",
                        "tone": "gentle",
                        "length": "short",
                        "start": "00:00",
                        "end": "06:00",
                        "timezone": "UTC",
                    },
                    "reply": "",
                },
            ]
        ),
    )

    class ClarifyingModel:
        def generate_json(self, *args, **kwargs):
            return next(responses)

    answers = iter(["unclear", "all explicit settings", "y"])
    setup(
        settings,
        "alice",
        model=ClarifyingModel(),
        services=False,
        ask=lambda _: next(answers),
        say=outputs.append,
    )
    assert outputs.count("Which language would you like us to use?") == 1
    assert outputs.count("Which conversation language do you mean?") == 1


def test_model_preparation_defaults_to_english(app):
    from self_learning_assistant.product.onboarding import prepare_model

    settings, _, _ = app
    with settings.edit() as config:
        config["selected_model"] = ""
    output, prompts = [], []

    def ask(question):
        prompts.append(question)
        raise EOFError

    with pytest.raises(EOFError):
        prepare_model(settings, ask=ask, say=output.append)
    assert "Prepare a local model" in output[0]
    assert "absolute path" in prompts[0]
    assert not any("\u3400" <= char <= "\u9fff" for char in "".join(output + prompts))


def test_source_eligibility_and_training_split(app):
    settings, store, _ = app
    old = store.message("old", "user", "old personal data")
    store.remember("old", "excluded", "background", "active", old, "old personal data")
    seed(store)
    pending = store.message("pending", "user", "I may prefer another option")
    store.allow_learning(pending, True)
    store.remember(
        "pending",
        "excluded",
        "preference",
        "pending",
        pending,
        "I may prefer another option",
    )
    items = adaptation.sources(settings, store)
    assert len(items) == 5
    train, evaluation = adaptation.dataset(items)
    assert 20 <= len(train) <= 500 and len(evaluation) >= 10
    assert not {item["prompt"] for item in train} & {
        item["prompt"] for item in evaluation
    }
    grounded = [item for item in train if item.get("context")]
    assert grounded and all(item["kind"] == "rehearsal" for item in grounded)
    assert all("evaluation-doc" not in json.dumps(item) for item in grounded)
    assert all("[rehearsal-" in item["target"] for item in grounded)
    assert all("excluded" not in item["text"] for item in items)
    assert adaptation.sources(settings, Store(store.database, "bob")) == []


def test_verified_world_only_and_volatile_exclusion(app):
    settings, store, _ = app
    for title, kind, metadata in [
        (
            "Stable verified fact",
            "web",
            {
                "verification": "supported",
                "parameter_eligible": True,
                "conditions": "under condition A",
            },
        ),
        ("Provided fact", "document", {"verification": "provided"}),
        (
            "Unverified fact",
            "web",
            {"verification": "insufficient", "parameter_eligible": True},
        ),
        (
            "Today's price",
            "web",
            {"verification": "supported", "parameter_eligible": True},
        ),
    ]:
        store.add_document(
            title,
            "test",
            title.encode(),
            [(title, "line 1")],
            kind=kind,
            metadata=metadata,
        )
    items = adaptation.sources(settings, store)
    assert len(items) == 2 and "condition A" in items[1]["text"]


def test_deferred_chat_no_learn_and_private(app):
    settings, store, _ = app
    model = Model()
    assistant = Assistant(settings, store, model, user="alice")
    result = assistant.respond("I prefer quiet answers", defer_learning=True)
    assert result["answer"] and not result["learned"]
    assert store.jobs()[0]["state"] == "queued"
    assert store.eligible_message(store.messages(result["session_id"])[0]["id"])
    result = assistant.respond("do not learn this", learn=False, defer_learning=True)
    assert not store.eligible_message(store.messages(result["session_id"])[0]["id"])
    before = store.jobs()
    private = Assistant(settings, None, model, user="alice").respond(
        "secret", private=True, online=True
    )
    assert private["research"] is None and store.jobs() == before


@pytest.mark.parametrize("change", ["memory", "delete", "profile", "document"])
def test_adapter_revocation_follows_source_changes(app, change):
    settings, store, model = app
    if change == "document":
        store.add_document(
            "Confirmed knowledge",
            "url",
            b"source",
            [("source", "url")],
            kind="web",
            metadata={"verification": "supported", "parameter_eligible": True},
        )
    record = candidate(settings, store, model)
    assert accept(settings, store, record) == "accepted"
    assert adaptation.active(settings, store, record["model_fingerprint"]) == Path(
        record["path"]
    )
    if change == "memory":
        store.change_memory(store.memories()[0]["id"], value="corrected")
    elif change == "delete":
        store.delete_memory(store.memories()[0]["id"], purge_source=True)
    elif change == "profile":
        with settings.edit() as value:
            value["profiles"]["alice"]["length"] = "detailed"
    else:
        store.delete_document(store.documents()[0]["id"])
    assert adaptation.invalidate(settings, store) == 1
    assert adaptation.active(settings, store, record["model_fingerprint"]) is None
    assert not Path(record["path"]).exists()
    assert not Path(record["path"]).exists()
    with pytest.raises(ProductError):
        adaptation.rollback(settings, store, record["id"])


@pytest.mark.parametrize(
    "result",
    [
        {
            "state": "trained",
            "improvement": 0.01,
            "checks": {"core_no_regression": True, "facts_no_regression": True},
        },
        {
            "state": "trained",
            "improvement": 0.2,
            "checks": {"core_no_regression": False, "facts_no_regression": True},
        },
        {"state": "trained", "improvement": 0.2, "checks": {}},
    ],
)
def test_failed_evaluation_never_activates(app, result):
    settings, store, model = app
    record = candidate(settings, store, model)
    assert adaptation.finish(settings, store, record, result) == "rejected"
    assert adaptation.active(settings, store, record["model_fingerprint"]) is None


def test_pause_resume_rollback_and_checksum(app):
    settings, store, model = app
    assert (
        adaptation.start(settings, store, adaptation.model_fingerprint(model)) is None
    )
    record = candidate(settings, store, model)
    assert adaptation.finish(settings, store, record, {"state": "paused"}) == "paused"
    assert (
        adaptation.start(settings, store, record["model_fingerprint"])["id"]
        == record["id"]
    )
    assert accept(settings, store, record) == "accepted"
    assert adaptation.start(settings, store, record["model_fingerprint"]) is None
    assert adaptation.rollback(settings, store, record["id"])["active"] == record["id"]
    (Path(record["path"]) / "adapter_model.safetensors").write_bytes(b"corrupted")
    with pytest.raises(ProductError, match="checksum"):
        adaptation.rollback(settings, store, record["id"])
    assert adaptation.active(settings, store, record["model_fingerprint"]) is None


def test_deletion_acceptance_after_background_revocation(app):
    from scripts.learning_acceptance import deletion_state

    settings, store, model = app
    record = candidate(settings, store, model)
    accept(settings, store, record)
    memory = next(
        item for item in store.memories() if "memory:" + item["id"] in record["sources"]
    )
    store.delete_memory(memory["id"])
    assert adaptation.invalidate(settings, store) == 1
    assert adaptation.invalidate(settings, store) == 0
    result = deletion_state(settings, store, record)
    assert result["files_removed"] is True and result["state"] == "revoked"


def test_adapter_backup_and_restore_remaps_paths(app, tmp_path):
    settings, store, model = app
    record = candidate(settings, store, model)
    accept(settings, store, record)
    path = backup(settings, store, tmp_path / "backup.zip")
    assert len(validate_archive(path)) == 5
    store.delete_memory(store.memories()[0]["id"])
    adaptation.invalidate(settings, store)
    restore(settings, store, path)
    restored = adaptation.runs(store)[0]
    assert adaptation.active(settings, store, record["model_fingerprint"]) == Path(
        restored["path"]
    )


def test_atomic_json_permissions_and_model_fingerprint(app, tmp_path):
    _, _, model = app
    output = tmp_path / "output.json"
    write_json(output, {"message": "Original text"})
    assert output.stat().st_mode & 0o777 == 0o600
    before = adaptation.model_fingerprint(model)
    (model / "model.safetensors").write_bytes(b"different base")
    assert adaptation.model_fingerprint(model) != before
    with pytest.raises(ProductError):
        fingerprint(tmp_path / "missing")


def test_runtime_foreground_priority_and_controls(app, monkeypatch):
    settings, _, _ = app
    owner = runtime.Runtime(settings)
    worker = Mock()
    worker.poll.return_value = None
    owner.worker = worker
    owner.foreground()
    assert owner.pause.exists()
    worker.wait.assert_called_once_with(timeout=60)
    assert owner.request({"operation": "ping"})
    assert owner.request({"operation": "status"}) == {"state": "idle"}
    with pytest.raises(ProductError):
        owner.request({"operation": "unknown"})
    with pytest.raises(ProductError):
        owner.request(
            {"operation": "generate", "user": "", "tokens": 5, "messages": []}
        )
    with pytest.raises(ProductError):
        owner.request({"operation": "generate", "tokens": 0, "messages": []})
    model = Model()
    monkeypatch.setattr(owner, "load", lambda user, personal: model)
    assert (
        owner.request({"operation": "generate", "tokens": 5, "messages": []})
        == "Test response"
    )
    assert owner.request(
        {"operation": "generate", "tokens": 5, "messages": [], "schema": {}}
    ) == {"ok": True}
    owner.worker = None
    assert owner.request({"operation": "shutdown"})
    assert owner.closed.is_set()


def test_scheduler_respects_disabled_recent_activity_and_battery(app, monkeypatch):
    settings, _, _ = app
    owner = runtime.Runtime(settings)
    owner.tick()
    with settings.edit() as value:
        value["managed"] = True
    owner.tick()
    owner.last_chat = -1000
    monkeypatch.setattr(runtime, "on_ac_power", lambda: False)
    owner.tick()
    assert owner.status["state"] == "waiting_for_ac_power"


def test_runtime_socket_and_status_without_service(app):
    settings, _, _ = app
    path = runtime.socket_path(settings)
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert not runtime.running(settings)
    assert runtime.status(settings, "alice")["runtime"]["state"] == "stopped"
    with runtime.gpu_slot(root=settings.paths.state / "test-gpu-lock"):
        pass


def test_new_cli_noninteractive_setup_and_history(app, capsys):
    assert main(["setup", "--json"]) == 2
    assert "interactive" in capsys.readouterr().err
    assert main(["--user-id", "alice", "learn", "history", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert main(["--user-id", "alice", "learn", "rollback", "missing", "--json"]) == 2
    capsys.readouterr()
    assert main(["daemon", "status", "--json"]) == 0
    assert not json.loads(capsys.readouterr().out)["running"]


def test_fixed_evaluation_and_interruption():
    """These Chinese responses satisfy the evaluator's multilingual probes."""

    class Evaluated:
        def generate(self, messages, **kwargs):
            query = messages[-1]["content"]
            if query.startswith("What is 2"):
                return "4"
            if "资料" in query:
                return "连接池大小是 32。[evaluation-doc:lines 1-2]"
            if "英语" in query or query.startswith("Explain"):
                return "A connection pool reuses connections."
            if "中文" in query:
                return "连接池复用已有连接。"
            return "favorite_color: blue\ncondition A"

        def generate_json(self, *args, **kwargs):
            return {"ok": True}

    result = check_model(
        Evaluated(),
        [{"prompt": "recall", "target": "favorite_color: blue\ncondition A"}],
    )
    assert all(result["core"].values()) and result["facts"] == [True]
    with pytest.raises(InterruptedError):
        check_model(Evaluated(), [], stopped=lambda: True)


def test_real_unix_runtime_rpc_and_shutdown(app, monkeypatch):
    settings, _, _ = app
    monkeypatch.setattr(runtime.signal, "signal", lambda *args: None)
    monkeypatch.setattr(runtime, "create_model", lambda *args, **kwargs: Model())
    owner = threading.Thread(target=runtime.run, args=(settings,))
    owner.start()
    try:
        for _ in range(100):
            if runtime.running(settings):
                break
            time.sleep(0.01)
        assert runtime.running(settings)
        model = runtime.RuntimeModel(settings, "alice")
        assert model.generate([]) == "Test response"
        assert model.generate_json([], {}) == {"ok": True}
        model.cancel_current()
        assert runtime.rpc(settings, {"operation": "status"})["state"] == "idle"
        with pytest.raises(ProductError, match="Unknown"):
            runtime.rpc(settings, {"operation": "unsupported"})
        with pytest.raises(ProductError, match="limit"):
            runtime.rpc(
                settings, {"operation": "ping", "content": "x" * runtime.MAX_RPC}
            )
        runtime.rpc(settings, {"operation": "shutdown"})
    finally:
        owner.join(timeout=10)
    assert not owner.is_alive() and not runtime.running(settings)


def test_runtime_model_reload_after_revocation(app, monkeypatch):
    settings, store, model = app
    record = candidate(settings, store, model)
    accept(settings, store, record)
    created = []

    def factory(*args, **kwargs):
        created.append(kwargs)
        return Model()

    monkeypatch.setattr(runtime, "create_model", factory)
    owner = runtime.Runtime(settings)
    first = owner.load("alice", True)
    store.delete_memory(store.memories()[0]["id"])
    second = owner.load("alice", True)
    assert first is not second and len(created) == 2
    assert owner.load("alice", False) is not second


def test_runtime_warmup_loads_without_generating_or_saving(app, monkeypatch):
    from self_learning_assistant.product.model import DirectoryModel

    settings, store, _ = app
    backend = Mock()
    backend.diagnostics.return_value = {"loaded": True, "generation": {}}
    model = DirectoryModel(backend)
    monkeypatch.setattr(runtime, "create_model", lambda *args, **kwargs: model)
    owner = runtime.Runtime(settings)
    request = {"operation": "warmup", "user": "alice", "personal": True}
    assert owner.request(request) is True
    backend.load.assert_called_once()
    backend.generate_with_options.assert_not_called()
    assert owner.status["inference"]["loaded"] is True
    assert store.jobs() == []


def test_chat_warms_model_before_first_input(app, monkeypatch, capsys):
    from self_learning_assistant.cli import product

    settings, _, _ = app
    with settings.edit() as config:
        config["onboarding"]["alice"] = {"complete": True, "draft": {}}
        config["profiles"]["alice"] = {"language": "English"}
    events = []

    class WarmModel(Model):
        def warmup(self):
            events.append("warmup")

    monkeypatch.setattr(product, "create_model", lambda *args, **kwargs: WarmModel())
    monkeypatch.setattr(product.sys.stdin, "isatty", lambda: True)

    def ask(question):
        events.append("input")
        assert "Model ready." in capsys.readouterr().out
        return "/exit"

    monkeypatch.setattr("builtins.input", ask)
    assert main(["--user-id", "alice", "chat"]) == 0
    assert events == ["warmup", "input"]


def test_systemd_unit_generation_is_scoped_and_escaped(app, tmp_path, monkeypatch):
    settings, _, _ = app
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "user config"))
    monkeypatch.setenv("SLA_HOME", '/tmp/a%name"with space')
    monkeypatch.setattr(runtime, "shutil_which_systemctl", lambda: "/usr/bin/systemctl")
    monkeypatch.setattr(runtime, "running", lambda _: False)
    calls = []

    def command(args, **kwargs):
        calls.append(args)
        return Mock(returncode=0)

    monkeypatch.setattr(runtime.subprocess, "run", command)
    name = runtime.install(settings)
    path = tmp_path / "user config/systemd/user" / name
    text = path.read_text()
    assert (
        "Restart=on-failure" in text and "a%%name" in text and '\\"with space' in text
    )
    assert calls[-1] == ["systemctl", "--user", "enable", "--now", name]
    runtime.uninstall(settings)
    assert not path.exists()


def test_window_budget_persists_and_midnight_keeps_same_window(app, monkeypatch):
    settings, _, _ = app
    owner = runtime.Runtime(settings)
    window = Window("23:00", "06:00", "UTC")
    before = window.identifier(
        "alice", datetime.fromisoformat("2026-10-11T23:30:00+00:00")
    )
    after = window.identifier(
        "alice", datetime.fromisoformat("2026-10-12T02:30:00+00:00")
    )
    assert before == after
    clock = [100.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])
    owner.begin_budget(before)
    clock[0] += 1800
    owner.end_budget()
    assert owner.budget(before) == 1800
    restarted = runtime.Runtime(settings)
    assert restarted.budget(after) == 1800


@pytest.mark.parametrize("state", ["paused", "trained"])
def test_training_worker_gate_and_resume(app, monkeypatch, state):
    settings, store, model = app
    record = candidate(settings, store, model)
    with settings.edit() as config:
        config["schedules"]["alice"] = {
            "start": "00:00",
            "end": "23:59",
            "timezone": "UTC",
            "enabled": True,
        }
    monkeypatch.setattr(runtime.signal, "signal", lambda *args: None)
    monkeypatch.setattr(runtime, "gpu_slot", lambda: nullcontext())
    monkeypatch.setattr(Window, "contains", lambda *args: True)
    monkeypatch.setattr(runtime, "on_ac_power", lambda: True)
    from self_learning_assistant.product import evaluation, finetune

    monkeypatch.setattr(
        finetune, "train", lambda *args, **kwargs: {"state": state, "improvement": 0.1}
    )
    monkeypatch.setattr(
        evaluation,
        "compare",
        lambda *args, **kwargs: {
            "core_no_regression": True,
            "facts_no_regression": True,
        },
    )
    runtime.worker(settings, "alice", record["id"])
    assert adaptation.runs(store)[0]["state"] == (
        "accepted" if state == "trained" else "paused"
    )


def test_scheduler_extracts_then_handles_unavailable_training(app, monkeypatch):
    settings, store, _ = app
    with settings.edit() as config:
        config["managed"] = True
        config["schedules"]["alice"] = {
            "start": "00:00",
            "end": "23:59",
            "timezone": "UTC",
            "enabled": True,
        }
    monkeypatch.setattr(Window, "contains", lambda *args: True)
    monkeypatch.setattr(runtime, "on_ac_power", lambda: True)
    owner = runtime.Runtime(settings)
    owner.last_chat = -1000
    monkeypatch.setattr(owner, "load", lambda *args: Model())
    store.enqueue("personal", "I prefer concise answers")
    owner.tick()
    assert owner.status["state"] == "processing_queue"
    assert store.jobs()[0]["state"] == "done"
    owner.tick()
    assert owner.status["state"] == "parameter_training_unavailable"


def test_personal_extraction_window_end_requeues_without_writing(app):
    from self_learning_assistant.product.service import run_jobs

    settings, store, _ = app
    halted = threading.Event()

    class InterruptedModel(Model):
        def generate_json(self, *args, **kwargs):
            halted.set()
            return {
                "items": [
                    {
                        "key": "answer_length",
                        "value": "concise",
                        "kind": "preference",
                        "explicit": True,
                        "quote": "I prefer concise",
                    }
                ]
            }

    source = store.message("session", "user", "I prefer concise answers")
    store.enqueue("personal", "I prefer concise answers", source)
    result = run_jobs(settings, store, InterruptedModel(), stopped=halted.is_set)
    assert result[0]["state"] == "queued"
    assert store.jobs()[0]["attempts"] == 0
    assert store.memories() == []


def test_offline_setup_skips_search_and_download(app, monkeypatch):
    from self_learning_assistant.product import onboarding

    settings, _, _ = app
    search = Mock(side_effect=AssertionError("Offline setup cannot probe search"))
    monkeypatch.setattr(onboarding, "prepare_search", search)
    fields = {
        "language": "English",
        "tone": "gentle",
        "length": "short",
        "start": "00:00",
        "end": "06:00",
        "timezone": "UTC",
    }
    answers = iter(["English, gentle and short; 00:00 to 06:00 UTC", "y", "n"])
    setup(
        settings,
        "alice",
        model=SetupModel([fields]),
        offline=True,
        ask=lambda _: next(answers),
        say=lambda _: None,
    )
    search.assert_not_called()
    assert not settings.read()["online"]
    with settings.edit() as value:
        value["selected_model"] = ""
    with pytest.raises(ProductError, match="Offline setup"):
        onboarding.prepare_model(
            settings, ask=lambda _: "", say=lambda _: None, offline=True
        )


def test_adapter_remains_valid_when_other_sources_are_added(app):
    settings, store, model = app
    record = candidate(settings, store, model)
    accept(settings, store, record)
    for index in range(260):
        source = store.message("new", "user", f"I prefer new option {index}")
        store.allow_learning(source, True)
        store.remember(
            f"new_{index}", str(index), "preference", "active", "I prefer", source
        )
    assert adaptation.valid_sources(record, settings, store)
    assert adaptation.invalidate(settings, store) == 0


def test_world_holdout_covers_conditions_without_disclosing_answer():
    items = [
        {
            "id": "p",
            "origin": "p",
            "kind": "profile",
            "language": "English",
            "text": "language: English",
        }
    ]
    items += [
        {
            "id": str(i),
            "origin": str(i),
            "kind": "personal",
            "language": "English",
            "text": f"option_{i}: blue",
        }
        for i in range(15)
    ]
    claim = "Python default arguments are evaluated once at function definition."
    items.append(
        {
            "id": "world",
            "origin": "world",
            "kind": "world",
            "language": "English",
            "text": claim + "\nConditions: when defined",
        }
    )
    _, evaluation = adaptation.dataset(items)
    probes = adaptation.evaluation_probes(evaluation)
    assert len(probes) == 10
    world = next(item for item in probes if item["kind"] == "world")
    assert claim not in world["prompt"]
    assert "Conditions: when defined" in world["target"]


@pytest.mark.parametrize("state", ["failed", "rejected"])
def test_deleted_source_cleans_failed_and_rejected_training_material(app, state):
    settings, store, model = app
    record = candidate(settings, store, model)
    adaptation.finish(settings, store, record, {"state": state})
    store.delete_memory(store.memories()[0]["id"])
    assert adaptation.invalidate(settings, store) == 1
    assert not Path(record["path"]).exists()
    assert adaptation.runs(store)[0]["result"] == {}


def test_incomplete_adapter_manifest_is_rejected(app):
    settings, store, model = app
    record = candidate(settings, store, model)
    accept(settings, store, record)
    path = Path(record["path"])
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["files"] = {}
    write_json(path / "manifest.json", manifest)
    assert adaptation.active(settings, store, record["model_fingerprint"]) is None


def test_runtime_reload_after_rollback_of_two_valid_versions(app, monkeypatch):
    settings, store, model = app
    first = candidate(settings, store, model)
    accept(settings, store, first)
    source = store.message("additional", "user", "I prefer blue")
    store.allow_learning(source, True)
    store.remember(
        "favorite_color", "blue", "preference", "active", source, "I prefer blue"
    )
    second = adaptation.start(settings, store, adaptation.model_fingerprint(model))
    assert second
    path = Path(second["path"])
    (path / "adapter_config.json").write_text("{}")
    (path / "adapter_model.safetensors").write_bytes(b"second version")
    accept(settings, store, second)
    factory = Mock(side_effect=lambda *args, **kwargs: Model())
    monkeypatch.setattr(runtime, "create_model", factory)
    owner = runtime.Runtime(settings)
    owner.load("alice", True)
    owner.load("alice", True)
    assert factory.call_count == 1
    adaptation.rollback(settings, store, first["id"])
    owner.load("alice", True)
    assert factory.call_count == 2


def test_private_runtime_model_does_not_open_database(app, monkeypatch):
    settings, _, _ = app
    owner = runtime.Runtime(settings)
    forbidden = Mock(side_effect=AssertionError("Private inference cannot open SQLite"))
    monkeypatch.setattr(runtime, "Store", forbidden)
    model = Model()
    factory = Mock(return_value=model)
    monkeypatch.setattr(runtime, "create_model", factory)
    assert owner.load("alice", False) is model
    forbidden.assert_not_called()
    assert factory.call_args.kwargs["personal"] is False


@pytest.mark.parametrize(
    "baseline,candidate,expected",
    [
        ([False, False], [False, False], False),
        ([False, False], [True, False], False),
        ([False, False], [True, True], True),
        ([True, False], [False, True], False),
    ],
)
def test_recall_gate_requires_learning_and_preserves_each_fact(
    baseline, candidate, expected
):
    from self_learning_assistant.product.evaluation import assess

    probes = [
        {
            "kind": "personal",
            "prompt": "What is your user's occupation?",
            "target": "engineer",
        },
        {
            "kind": "personal",
            "prompt": "What answer length does your user prefer?",
            "target": "concise",
        },
    ]
    before = {"core": {"maths": True}, "facts": baseline}
    after = {"core": {"maths": True}, "facts": candidate}
    checked = assess(before, after, probes)
    assert checked["facts_no_regression"] is expected
    assert checked["baseline"] == before and checked["candidate"] == after


def test_large_training_budget_covers_world_and_tracks_only_used_sources(app):
    settings, store, model = app
    seed(store, count=110)
    store.add_document(
        "Python default arguments are evaluated at definition.",
        "public",
        b"fixture",
        [("Verified source", "lines 1-1")],
        kind="web",
        metadata={
            "verification": "supported",
            "parameter_eligible": True,
            "conditions": "standard Python",
        },
    )
    record = adaptation.start(settings, store, adaptation.model_fingerprint(model))
    assert record
    data = json.loads((Path(record["path"]) / "dataset.json").read_text())
    assert len(data["training"]) <= 500
    assert any(item["kind"] == "world" for item in data["training"])
    assert len(record["sources"]) == 25  # 24 sources and the training protocol.


def test_scheduler_round_robin_reaches_other_users(app, monkeypatch):
    settings, _, _ = app
    with settings.edit() as value:
        value["managed"] = True
        for user in ("alice", "bob"):
            value["schedules"][user] = {
                "start": "00:00",
                "end": "23:59",
                "timezone": "UTC",
                "enabled": True,
            }
    monkeypatch.setattr(Window, "contains", lambda *args: True)
    monkeypatch.setattr(runtime, "on_ac_power", lambda: True)
    owner = runtime.Runtime(settings)
    owner.last_chat = -1000
    monkeypatch.setattr(owner, "load", lambda *args: Model())
    bob = Store(settings.paths.database, "bob")
    bob.enqueue("personal", "I prefer concise answers")
    owner.tick()
    assert owner.last_scheduled_user == "alice"
    owner.tick()
    assert owner.last_scheduled_user == "bob"
    assert bob.jobs()[0]["state"] == "done"


@pytest.mark.parametrize(
    "reply,expected", [("320", False), ("132", False), ("32.5", False), ("32", True)]
)
def test_exact_numeric_recall_rejects_longer_numbers(reply, expected):
    class Numeric(Model):
        def generate(self, messages, **kwargs):
            if messages[-1]["content"] == "Recall the stored number":
                return reply
            return super().generate(messages, **kwargs)

    result = check_model(
        Numeric(), [{"prompt": "Recall the stored number", "target": "32"}]
    )
    assert result["facts"] == [expected]


def test_worker_evaluates_previously_learned_sources_outside_current_budget(
    app, monkeypatch
):
    from self_learning_assistant.product import evaluation, finetune

    settings, store, model = app
    seed(store, count=35)
    fingerprint = adaptation.model_fingerprint(model)
    first = adaptation.start(settings, store, fingerprint)
    for record in [first]:
        path = Path(record["path"])
        (path / "adapter_config.json").write_text("{}")
        (path / "adapter_model.safetensors").write_bytes(b"first version")
    accept(settings, store, first)
    source = store.message("new", "user", "I prefer blue")
    store.allow_learning(source, True)
    store.remember("color", "blue", "preference", "active", source, "I prefer blue")
    second = adaptation.start(settings, store, fingerprint)
    assert second
    path = Path(second["path"])
    (path / "adapter_config.json").write_text("{}")
    (path / "adapter_model.safetensors").write_bytes(b"second version")
    dropped = set(first["sources"]) - set(second["sources"])
    assert dropped
    _, old_eval = adaptation.dataset(
        [item for item in adaptation.sources(settings, store) if item["id"] in dropped]
    )
    with settings.edit() as value:
        value["schedules"]["alice"] = {
            "start": "00:00",
            "end": "23:59",
            "timezone": "UTC",
            "enabled": True,
        }
    monkeypatch.setattr(Window, "contains", lambda *args: True)
    monkeypatch.setattr(runtime, "on_ac_power", lambda: True)
    monkeypatch.setattr(runtime, "gpu_slot", nullcontext)
    monkeypatch.setattr(runtime.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        finetune,
        "train",
        lambda *args, **kwargs: {"state": "trained", "improvement": 0.1},
    )
    captured = []

    def compare(*args, **kwargs):
        captured.extend(args[2])
        return {"core_no_regression": True, "facts_no_regression": True}

    monkeypatch.setattr(evaluation, "compare", compare)
    runtime.worker(settings, "alice", second["id"])
    assert any(item["target"] == old_eval[0]["target"] for item in captured)
