"""Exercise real-model personal memory, retrieval and privacy in isolated data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from self_learning_assistant.product.config import Paths, Settings
from self_learning_assistant.product.data import backup, restore
from self_learning_assistant.product.model import create_model, validate_endpoint
from self_learning_assistant.product.service import Assistant
from self_learning_assistant.product.store import Store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument("--url")
    backend.add_argument("--path", type=Path)
    parser.add_argument("--api-model", default="self-learning-assistant-local")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()
    if not args.app_home.is_absolute():
        parser.error("app-home must be absolute")
    settings = Settings(
        Paths(*(args.app_home / name for name in ("config", "data", "state", "cache")))
    )
    if settings.paths.database.exists() or settings.file.exists():
        parser.error("Use a fresh isolated app-home; this test creates and edits data")
    spec = (
        {
            "backend": "api",
            "url": validate_endpoint(args.url),
            "model": args.api_model,
        }
        if args.url
        else {
            "backend": "local",
            "path": str(args.path.resolve()),
            "device": args.device,
            "load_in_4bit": args.load_in_4bit,
        }
    )
    with settings.edit() as config:
        config["models"]["real"] = spec
        config["selected_model"] = "real"
        config["profiles"]["smoke"] = {"name": "小知", "length": "简短"}
    store = Store(settings.paths.database, "smoke")
    model = create_model(settings)
    results = {}
    assistant = Assistant(settings, store, model, user="smoke")
    results["initial"] = assistant.respond(
        "我喜欢简短回答，请记住这个长期偏好。", session="s"
    )
    assert results["initial"]["learned"], results["initial"]["learning_error"]
    # Recreate both objects to exercise durable state rather than Python history.
    store = Store(settings.paths.database, "smoke")
    assistant = Assistant(Settings(settings.paths), store, model, user="smoke")
    results["recall"] = assistant.respond("我喜欢什么样的回答？", learn=False)
    assert "简短" in results["recall"]["answer"], results["recall"]["answer"]
    identifier = results["initial"]["learned"][0]
    store.change_memory(identifier, value="必要时详细解释")
    results["corrected"] = Assistant(settings, store, model, user="smoke").respond(
        "我的回答偏好是什么？", learn=False
    )
    assert "详细" in results["corrected"]["answer"], results["corrected"]["answer"]
    source = args.app_home / "notes.md"
    source.write_text("数据库连接池大小为 32。\n", encoding="utf-8")
    document = store.import_file(source)
    results["retrieval"] = Assistant(settings, store, model, user="smoke").respond(
        "资料里的数据库连接池大小是多少？请标明出处。", learn=False
    )
    assert "32" in results["retrieval"]["answer"]
    assert results["retrieval"]["sources"]
    archive = backup(settings, store, args.app_home / "backup.zip")
    store.delete_document(document)
    assert not store.search("连接池")
    restore(settings, store, archive)
    assert store.search("连接池")
    before = hashlib.sha256(store.database.read_bytes()).hexdigest()
    results["private"] = Assistant(settings, None, model, user="smoke").respond(
        "你好", private=True
    )
    assert before == hashlib.sha256(store.database.read_bytes()).hexdigest()
    store.delete_memory(identifier, purge_source=True)
    assert not store.memories("active")
    assert all(item["role"] != "user" for item in store.messages("s"))
    results["checks"] = {
        "memory_restart": True,
        "correction": True,
        "retrieval": True,
        "backup_restore": True,
        "private_database_unchanged": True,
        "delete": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results["checks"]))


if __name__ == "__main__":
    main()
