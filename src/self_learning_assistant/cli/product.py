from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

from ..domain import MemoryScope
from ..memory.repository import MemoryStoreError
from ..model.roles import RoleModelError
from ..product.config import PROFILE, ProductError, Settings
from ..product.model import create_model, structured, validate_endpoint
from ..product.service import Assistant, run_jobs, save_research
from ..product.store import Store
from ..product.ui import ui_text
from ..prompts import MODEL_CHECK_JSON, MODEL_CHECK_TEXT


def remaining_arguments(arguments: list[str]) -> list[str]:
    rest = arguments[:]
    globals_ = {
        "--database",
        "--user-id",
        "--project-id",
        "--workspace-id",
    }
    while rest:
        if rest[0] in globals_:
            if len(rest) < 2:
                return rest
            rest = rest[2:]
        elif rest[0].split("=", 1)[0] in globals_ and "=" in rest[0]:
            rest = rest[1:]
        else:
            break
    return rest


def _check_archived(arguments: list[str], root: argparse.ArgumentParser) -> None:
    rest = remaining_arguments(arguments)
    retired = {"evolution", "eval", "samples", "skills", "self-learn"}
    actions = {
        "memory": {
            "facts",
            "holds",
            "accept",
            "reject",
            "review-queue",
            "events",
            "forget",
            "backup",
            "revalidate",
        },
    }
    if rest and (
        rest[0] in retired or len(rest) > 1 and rest[1] in actions.get(rest[0], set())
    ):
        root.error(
            "This research command is archived and no longer executable. "
            "See README.md for current commands and documentation."
        )
    retired_options = {
        "--smart-routing",
        "--harness-command",
        "--harness-model",
        "--research-backend",
        "--tool-budget",
        "--no-train",
        "--model-url",
        "--no-progress-timeout",
        "--no-web-research",
    }
    for token in arguments:
        if token == "--":
            break
        if token.split("=", 1)[0] in retired_options:
            root.error(
                f"{token.split('=', 1)[0]} belongs to the archived prototype. "
                "Use model add/use, research --web or learn topic/run; "
                "see README.md for current commands and documentation."
            )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="sla",
        description="Self-Learning Assistant · 自我学习助手: local chat, memory and knowledge.",
    )
    root.add_argument("--database", type=Path)
    root.add_argument("--user-id", default="default_user")
    root.add_argument("--project-id", default="default_project")
    root.add_argument("--workspace-id", default="default_workspace")
    commands = root.add_subparsers(dest="command", required=True)
    from .serve import configure_serve

    configure_serve(commands.add_parser("serve", help="Serve an existing local model"))
    for name in ("init", "doctor"):
        command = commands.add_parser(name)
        command.add_argument("--json", action="store_true")
        if name == "doctor":
            command.add_argument("--check-model", action="store_true")
            command.add_argument("--check-search", action="store_true")
    setup_parser = commands.add_parser(
        "setup", help="Conversational first-run settings"
    )
    setup_parser.add_argument("--json", action="store_true")
    daemon = commands.add_parser("daemon").add_subparsers(dest="action", required=True)
    for name in ("install", "uninstall", "run", "status", "worker"):
        action = daemon.add_parser(name)
        if name == "worker":
            action.add_argument("id")
            action.add_argument("--seconds", type=float, default=3600)
        action.add_argument("--json", action="store_true")
    for name in ("config", "profile"):
        command = commands.add_parser(name)
        actions = command.add_subparsers(dest="action", required=True)
        show = actions.add_parser("show")
        show.add_argument("--json", action="store_true")
        set_ = actions.add_parser("set")
        set_.add_argument("key")
        set_.add_argument("value")
        set_.add_argument("--json", action="store_true")
    model = commands.add_parser("model").add_subparsers(dest="action", required=True)
    add = model.add_parser("add")
    add.add_argument("name")
    backend = add.add_mutually_exclusive_group(required=True)
    backend.add_argument("--path", type=Path)
    backend.add_argument("--url")
    add.add_argument("--api-model")
    add.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    add.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto"
    )
    add.add_argument("--load-in-4bit", action="store_true")
    add.add_argument("--json", action="store_true")
    for name in ("list", "use", "check"):
        action = model.add_parser(name)
        action.add_argument(
            "name", nargs="?" if name == "check" else (None if name == "use" else "?")
        )
        action.add_argument("--json", action="store_true")
    chat = commands.add_parser("chat")
    chat.add_argument("message", nargs="?")
    chat.add_argument("--session")
    chat.add_argument("--model")
    chat.add_argument("--model-dir", type=Path)
    chat.add_argument("--private", action="store_true")
    chat.add_argument("--no-learn", action="store_true")
    chat.add_argument("--no-color", action="store_true")
    network = chat.add_mutually_exclusive_group()
    network.add_argument("--offline", action="store_true")
    network.add_argument("--web", action="store_true")
    chat.add_argument("--web-query")
    chat.add_argument("--json", action="store_true")
    memory = commands.add_parser("memory").add_subparsers(dest="action", required=True)
    for name in ("list", "show", "edit", "confirm", "dismiss", "delete"):
        action = memory.add_parser(name)
        if name != "list":
            action.add_argument("id")
        if name == "edit":
            action.add_argument("value")
        if name == "delete":
            action.add_argument("--purge-source", action="store_true")
        action.add_argument("--json", action="store_true")
    knowledge = commands.add_parser("knowledge").add_subparsers(
        dest="action", required=True
    )
    for name in ("import", "list", "search", "show", "delete"):
        action = knowledge.add_parser(name)
        if name in {"import", "search", "show", "delete"}:
            action.add_argument("value")
        action.add_argument("--json", action="store_true")
    learn = commands.add_parser("learn").add_subparsers(dest="action", required=True)
    for name in (
        "topic",
        "run",
        "status",
        "retry",
        "pause",
        "resume",
        "history",
        "rollback",
    ):
        action = learn.add_parser(name)
        if name == "topic":
            action.add_argument("text")
        if name == "run":
            action.add_argument("--limit", type=int, default=10)
        if name == "rollback":
            action.add_argument("version")
        action.add_argument("--json", action="store_true")
    research = commands.add_parser("research")
    research.add_argument("claim")
    research.add_argument(
        "--web",
        action="store_true",
        help="Explicitly enable networking for this operation",
    )
    research.add_argument("--timeout", type=float, default=300)
    research.add_argument("--report-reserve", type=float, default=60)
    research.add_argument("--trusted-host", action="append", default=[])
    research.add_argument("--output", type=Path)
    research.add_argument("--json", action="store_true")
    data = commands.add_parser("data").add_subparsers(dest="action", required=True)
    for name in ("export", "backup", "restore", "migrate"):
        action = data.add_parser(name)
        action.add_argument("path", type=Path)
        if name == "restore":
            action.add_argument(
                "--yes",
                action="store_true",
                help="Replace application data after validation and safety backup",
            )
        action.add_argument("--json", action="store_true")
    return root


def emit(value: Any, args: argparse.Namespace) -> None:
    if getattr(args, "json", False):
        print(json.dumps(value, ensure_ascii=False))
    elif isinstance(value, dict) and "answer" in value:
        language = getattr(args, "ui_language", "")
        print(value["answer"])
        if value.get("learned"):
            count = len(value["learned"])
            print(
                "\n"
                + ui_text(
                    language,
                    f"Remembered {count} personal facts.",
                    f"记住了 {count} 条个人信息。",
                )
            )
        if value.get("learning_error"):
            print(
                "\n"
                + ui_text(
                    language,
                    "Learning is incomplete; the task was saved for retry.",
                    "学习未完成，任务已保存供重试。",
                ),
                file=sys.stderr,
            )
        print("\n" + ui_text(language, "Session: ", "会话：") + value["session_id"])
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2))


def execute(arguments: list[str]) -> int:
    root = parser()
    _check_archived(arguments, root)
    args = root.parse_args(
        arguments if remaining_arguments(arguments) else [*arguments, "chat"]
    )
    try:
        settings = Settings()
        return dispatch(args, settings)
    except (ProductError, MemoryStoreError, RoleModelError) as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        else:
            print(
                ui_text(getattr(args, "ui_language", ""), "Error: ", "错误：")
                + str(exc),
                file=sys.stderr,
            )
        return 2
    except (OSError, sqlite3.Error, ImportError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def dispatch(args: argparse.Namespace, settings: Settings) -> int:
    args.ui_language = settings.profile(args.user_id).get("language", "")
    command = args.command
    if command == "serve":
        from .serve import run_serve

        return run_serve(args, settings)
    database = args.database or settings.paths.database
    if command == "setup":
        if args.json or not sys.stdin.isatty():
            raise ProductError("Conversational setup requires an interactive terminal")
        from ..product.onboarding import setup

        setup(settings, args.user_id, offline=getattr(args, "offline", False))
        return 0
    if command == "daemon":
        from ..product import runtime

        if args.database:
            raise ProductError(
                "The managed runtime uses the configured application database"
            )
        if args.action == "run":
            runtime.run(settings)
        elif args.action == "install":
            emit({"unit": runtime.install(settings)}, args)
        elif args.action == "uninstall":
            runtime.uninstall(settings)
            emit({"uninstalled": True}, args)
        elif args.action == "worker":
            runtime.worker(settings, args.user_id, args.id, seconds=args.seconds)
        else:
            emit(runtime.status(settings, args.user_id), args)
        return 0
    if command == "init":
        with settings.edit():
            pass
        Store(database, args.user_id)
        emit(
            {
                "config": str(settings.file),
                "database": str(database),
                "next": "sla model add NAME --url http://127.0.0.1:11434/v1 --api-model MODEL",
            },
            args,
        )
        return 0
    if command in {"config", "profile"}:
        if args.action == "show":
            emit(
                settings.effective()
                if command == "config"
                else settings.profile(args.user_id),
                args,
            )
            return 0
        with settings.edit() as value:
            if command == "profile":
                if (
                    args.key not in PROFILE
                    or not args.value.strip()
                    or len(args.value) > 500
                ):
                    raise ProductError(
                        "Profile keys: name, tone, length; values must contain 1-500 characters"
                    )
                value["profiles"].setdefault(args.user_id, {})[args.key] = args.value
            elif args.key in {"online", "learning"}:
                if args.value not in {"true", "false"}:
                    raise ProductError("Use true or false")
                value[args.key] = args.value == "true"
            elif args.key == "max_new_tokens":
                value[args.key] = int(args.value)
            elif args.key == "search.searxng_url":
                from ..tools.security import validate_search_endpoint

                value["search"]["searxng_url"] = validate_search_endpoint(args.value)
            elif args.key == "search.trusted_hosts":
                value["search"]["trusted_hosts"] = [
                    host.strip().lower()
                    for host in args.value.split(",")
                    if host.strip()
                ]
            else:
                raise ProductError(
                    "Unsupported configuration key; credentials belong in environment variables"
                )
        emit({"updated": args.key}, args)
        if command == "profile":
            from ..product.adaptation import invalidate

            invalidate(settings, Store(database, args.user_id))
        return 0
    if command == "model":
        return models(args, settings)
    if command == "doctor":
        return doctor(args, settings, database)
    if command == "chat":
        return chat(args, settings, database)
    store = Store(database, args.user_id)
    if command == "memory":
        if args.action == "list":
            result: Any = store.memories()
        elif args.action == "show":
            result = store.memory(args.id)
        elif args.action == "edit":
            result = store.change_memory(args.id, value=args.value)
        elif args.action in {"confirm", "dismiss"}:
            result = store.change_memory(
                args.id, status="active" if args.action == "confirm" else "dismissed"
            )
        else:
            store.delete_memory(args.id, purge_source=args.purge_source)
            result = {"deleted": args.id, "source_purged": args.purge_source}
        if args.action not in {"list", "show"}:
            from ..product.adaptation import invalidate

            invalidate(settings, store)
        emit(result, args)
    elif command == "knowledge":
        if args.action == "import":
            result = {"id": store.import_file(Path(args.value))}
        elif args.action == "list":
            result = store.documents()
        elif args.action == "search":
            result = store.search(args.value)
        elif args.action == "show":
            result = store.document(args.value)
        else:
            store.delete_document(args.value)
            result = {"deleted": args.value, "original_file_deleted": False}
            from ..product.adaptation import invalidate

            invalidate(settings, store)
        emit(result, args)
    elif command == "learn":
        if args.action == "topic":
            result = {"id": store.enqueue("topic", args.text), "state": "queued"}
        elif args.action == "status":
            from ..product.runtime import status

            result = {
                "learning": settings.effective()["learning"],
                "jobs": store.jobs(),
                "background": status(settings, args.user_id),
            }
        elif args.action == "history":
            from ..product.adaptation import runs

            result = runs(store)
        elif args.action == "rollback":
            from ..product.adaptation import rollback

            result = rollback(settings, store, args.version)
        elif args.action == "retry":
            result = {"retried": store.retry()}
        elif args.action in {"pause", "resume"}:
            with settings.edit() as value:
                value["learning"] = args.action == "resume"
            result = {"learning": args.action == "resume"}
        else:
            if not settings.effective()["learning"]:
                result = [{"state": "paused"}]
            else:
                result = run_jobs(
                    settings, store, create_model(settings), limit=args.limit
                )
            emit(result, args)
            return (
                1
                if any(item["state"] == "failed" for item in result)
                else 3
                if any(item["state"] == "deferred" for item in result)
                else 0
            )
        emit(result, args)
    elif command == "research":
        from ..product.research import Researcher, web_service

        if not args.web and not settings.effective()["online"]:
            raise ProductError(
                "Online research is disabled; use --web or sla config set online true"
            )
        model = create_model(settings)
        report = Researcher(
            model,
            web_service(settings),
            trusted_hosts=args.trusted_host
            or settings.effective()["search"].get("trusted_hosts", []),
            timeout=args.timeout,
            reserve=args.report_reserve,
        ).verify(args.claim)
        job = store.enqueue("topic", args.claim)
        state = (
            "verified"
            if report["accepted"]
            else "failed"
            if report["execution"]["failed"]
            else "deferred"
        )
        store.finish(job, state, report)
        if report["accepted"]:
            report["document_id"] = save_research(
                store,
                report,
                parameter_eligible=bool(
                    settings.effective()["learning"]
                    and settings.effective()["parameter_learning"]
                ),
            )
        if args.output:
            if args.output.resolve() in {database.resolve(), settings.file.resolve()}:
                raise ProductError("Report cannot overwrite application data")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.chmod(args.output, 0o600)
        emit(report, args)
        return 0 if report["accepted"] else 1 if report["execution"]["failed"] else 3
    elif command == "data":
        from ..product.data import backup, export_user, migrate, restore

        if args.action == "backup":
            result = {"backup": str(backup(settings, store, args.path))}
        elif args.action == "restore":
            if not args.yes:
                raise ProductError(
                    "Restore replaces data. Inspect the backup and supply --yes; a safety backup is always created."
                )
            result = {"safety_backup": str(restore(settings, store, args.path))}
        elif args.action == "migrate":
            result = {"safety_backup": str(migrate(settings, store, args.path))}
        else:
            if args.path.exists() or args.path.resolve() in {
                database.resolve(),
                settings.file.resolve(),
            }:
                raise ProductError("Export destination already exists")
            args.path.parent.mkdir(parents=True, exist_ok=True)
            args.path.write_text(
                json.dumps(export_user(store), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.chmod(args.path, 0o600)
            result = {"export": str(args.path)}
        emit(result, args)
    return 0


def models(args: argparse.Namespace, settings: Settings) -> int:
    if args.action == "add":
        if not args.name.strip() or len(args.name) > 100:
            raise ProductError("Invalid model name")
        if args.path:
            path = args.path.expanduser().resolve()
            if not path.is_dir():
                raise ProductError("Model directory does not exist")
            spec = {
                "backend": "local",
                "path": str(path),
                "device": args.device,
                "dtype": args.dtype,
                "load_in_4bit": args.load_in_4bit,
            }
        else:
            if not args.api_model:
                raise ProductError("API models require --api-model")
            spec = {
                "backend": "api",
                "url": validate_endpoint(args.url),
                "model": args.api_model,
            }
        with settings.edit() as config:
            if args.name in config["models"]:
                raise ProductError("Model already exists; choose a different name")
            config["models"][args.name] = spec
            if not config["selected_model"]:
                config["selected_model"] = args.name
        emit({"name": args.name, **spec}, args)
    elif args.action == "list":
        config = settings.effective()
        emit({"selected": config["selected_model"], "models": config["models"]}, args)
    elif args.action == "use":
        with settings.edit() as config:
            if args.name not in config["models"]:
                raise ProductError("Unknown model")
            config["selected_model"] = args.name
        emit({"selected": args.name}, args)
    else:
        emit(check_model(settings, args.name), args)
    return 0


def check_model(settings: Settings, name: str | None = None) -> dict[str, bool]:
    model = create_model(settings, name=name)
    answer = model.generate(
        [{"role": "user", "content": MODEL_CHECK_TEXT}], max_new_tokens=32
    )
    result = structured(
        model,
        [{"role": "user", "content": MODEL_CHECK_JSON}],
        {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        max_new_tokens=64,
    )
    if not answer.strip() or result != {"ok": True}:
        raise ProductError("Model failed the structured output check")
    return {"chat": True, "structured": True}


def doctor(args: argparse.Namespace, settings: Settings, database: Path) -> int:
    config = settings.effective()
    settings.paths.initialize()
    store = Store(database, args.user_id)
    with store.connect() as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    result: dict[str, Any] = {
        "database": str(database),
        "integrity": integrity,
        "config": str(settings.file),
        "online": config["online"],
        "learning": config["learning"],
        "selected_model": config["selected_model"],
        "private_permissions": oct(database.stat().st_mode & 0o777),
    }
    if args.check_model:
        result["model"] = check_model(settings)
    if args.check_search:
        if not config["online"]:
            raise ProductError("Enable online before running a search diagnostic")
        from ..product.research import web_service

        response = web_service(settings).search("Python documentation", max_results=1)
        result["search"] = {
            "backend": response.backend,
            "results": len(response.results),
        }
    emit(result, args)
    return 0 if integrity == "ok" else 1


def chat(args: argparse.Namespace, settings: Settings, database: Path) -> int:
    config = settings.read()
    interactive = not args.message and not args.json and sys.stdin.isatty()
    if (
        interactive
        and not args.private
        and not config["onboarding"].get(args.user_id, {}).get("complete")
    ):
        from ..product.onboarding import setup

        setup(settings, args.user_id, offline=args.offline)
        config = settings.read()
    language = settings.profile(args.user_id).get("language", "")
    args.ui_language = language
    managed = (
        config["managed"]
        and not args.model
        and not args.model_dir
        and database.resolve() == settings.paths.database.resolve()
    )
    if managed:
        from ..product.runtime import RuntimeModel, ensure

        ensure(settings)
        model: Any = RuntimeModel(settings, args.user_id, personal=not args.private)
    else:
        model = create_model(
            settings,
            name=args.model,
            directory=args.model_dir,
            user=args.user_id,
            personal=not args.private,
            database=database,
        )
    if interactive:
        warmup = getattr(model, "warmup", None)
        if callable(warmup):
            print(
                ui_text(
                    language,
                    "Loading the local model; the first load may take a moment…",
                    "正在加载本地模型，首次加载可能需要一些时间……",
                ),
                flush=True,
            )
            warmup()
            print(ui_text(language, "Model ready.", "模型已就绪。"), flush=True)
    store = None if args.private else Store(database, args.user_id)
    assistant = Assistant(
        settings,
        store,
        model,
        user=args.user_id,
        scope=MemoryScope(args.user_id, args.project_id, args.workspace_id),
    )
    online = False if args.offline else True if args.web else None
    import uuid

    session = args.session or uuid.uuid4().hex
    if args.message:
        emit(
            assistant.respond(
                args.message,
                session=session,
                private=args.private,
                learn=not args.no_learn,
                online=online,
                web_query=args.web_query,
                resume=bool(args.session),
                defer_learning=managed,
            ),
            args,
        )
        return 0
    if not args.json:
        name = ui_text(language, "Self-Learning Assistant", "自我学习助手")
        hint = ui_text(
            language, "/help for commands, /exit to quit", "/help 查看命令，/exit 退出"
        )
        print(
            f"{name} · session {session} · {'private' if args.private else 'local memory'}\n{hint}"
        )
    while True:
        try:
            message = input("› " if sys.stdin.isatty() and not args.json else "")
        except EOFError:
            break
        if message in {"/exit", "/quit"}:
            break
        if message == "/help":
            print(
                "/help /info /clear /reset /setup /learning /exit; memory: sla memory list; learning: sla learn status"
            )
            continue
        if message == "/learning":
            from ..product.runtime import status

            emit(status(settings, args.user_id), args)
            continue
        if message == "/setup":
            if args.private or not interactive:
                raise ProductError("Use sla setup in an ordinary interactive session")
            from ..product.onboarding import setup

            if getattr(model, "backend", None) is not None:
                model.backend.unload()
            setup(settings, args.user_id, offline=args.offline)
            return chat(args, settings, database)
        if message == "/info":
            emit(
                {
                    "session": session,
                    "private": args.private,
                    "online": settings.effective()["online"]
                    if online is None
                    else online,
                },
                args,
            )
            continue
        if message in {"/clear", "/reset"}:
            assistant.history.clear()
            session = uuid.uuid4().hex
            continue
        if message.strip():
            emit(
                assistant.respond(
                    message,
                    session=session,
                    private=args.private,
                    learn=not args.no_learn,
                    online=online,
                    web_query=args.web_query,
                    resume=bool(args.session),
                    defer_learning=managed,
                ),
                args,
            )
    return 0
