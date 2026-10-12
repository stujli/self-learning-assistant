"""Real-model end-to-end acceptance in a fresh application home; web bodies are fixtures."""

from __future__ import annotations

import argparse
import os
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from unittest.mock import patch

from self_learning_assistant.product import adaptation, runtime
from self_learning_assistant.product.config import Settings
from self_learning_assistant.product.finetune import write_json
from self_learning_assistant.product.model import DirectoryModel, create_model
from self_learning_assistant.product.onboarding import setup
from self_learning_assistant.product.service import Assistant
from self_learning_assistant.product.store import Store
from self_learning_assistant.tools.domain import FetchedPage, SearchHit, SearchResponse

CLAIM = "Python default arguments are evaluated once at function definition."
URL = "https://docs.python.org/3/tutorial/controlflow.html"


def deletion_state(
    settings: Settings, store: Store, record: dict[str, Any]
) -> dict[str, Any]:
    """Verify effects even when the runtime revoked the version first."""
    current = next(
        item for item in adaptation.runs(store) if item["id"] == record["id"]
    )
    remaining = {"memory:" + item["id"] for item in store.memories()}
    removed = [
        key
        for key in record["sources"]
        if key.startswith("memory:") and key not in remaining
    ]
    assert removed and current["state"] == "revoked"
    assert adaptation.active(settings, store, record["model_fingerprint"]) is None
    assert not Path(record["path"]).exists()
    return {
        "version": record["id"],
        "state": "revoked",
        "removed_sources": removed,
        "files_removed": True,
    }


class FixtureWeb:
    def search(self, query: str, *, max_results: int = 4) -> SearchResponse:
        return SearchResponse(
            query, "fixture", (SearchHit("Python documentation", URL, "", "fixture"),)
        )

    def fetch(self, url: str, *, max_chars: int = 20000) -> FetchedPage:
        return FetchedPage(
            url,
            url,
            "Python documentation",
            CLAIM + " They are not evaluated again on each function call.",
            datetime.now(UTC).isoformat(),
            "sha256:" + "a" * 64,
            "text/plain",
            False,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--app-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted or rejected run using its matching isolated home and report",
    )
    args = parser.parse_args()
    if (
        not args.app_home.is_absolute()
        or (not args.resume and (args.app_home.exists() or args.output.exists()))
        or (args.resume and (not args.app_home.exists() or not args.output.exists()))
    ):
        parser.error("Use a fresh absolute application home and unused report path")
    os.environ["SLA_HOME"] = str(args.app_home)
    settings = Settings()
    store = Store(settings.paths.database, "acceptance")
    if not args.resume:
        with settings.edit() as value:
            value["models"]["acceptance"] = {
                "backend": "local",
                "path": str(args.model_dir.resolve()),
                "device": "cuda",
                "dtype": "bfloat16",
                "load_in_4bit": True,
            }
            value["selected_model"] = "acceptance"
            value["max_new_tokens"] = 160
            value["search"]["trusted_hosts"] = ["docs.python.org"]
    import torch

    report: dict[str, Any] = {
        "web": "synthetic page bodies; real routing/report model",
        "environment": {
            "app_home": str(args.app_home.resolve()),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "model": str(args.model_dir.resolve()),
            "gpu": torch.cuda.get_device_name(),
            "versions": {
                name: version(name)
                for name in ("torch", "transformers", "peft", "bitsandbytes")
            },
        },
        "stages": {},
    }

    def stage(name: str, value: Any) -> None:
        report["stages"][name] = value
        write_json(args.output, report)
        print(name, flush=True)

    if args.resume:
        import json

        prior = json.loads(args.output.read_text())
        config = settings.read()
        prior_run = prior["stages"].get("training_failure") or prior["stages"].get(
            "training"
        )
        if (
            not (
                prior["environment"].get("app_home") == str(args.app_home.resolve())
                or (
                    prior_run
                    and Path(prior_run["path"])
                    .resolve()
                    .is_relative_to(args.app_home.resolve())
                )
            )
            or config["selected_model"] != "acceptance"
            or prior["environment"]["model"] != str(args.model_dir.resolve())
            or "private" not in prior["stages"]
        ):
            parser.error(
                "Resume requires the report and isolated home from the same completed initial stages"
            )
        if prior.get("passed"):
            print("This acceptance already passed; choose a fresh home to rerun.")
            return
        report = prior
        if (
            "personalized_chat" in report["stages"]
            and "parameter_recall_without_knowledge_base" in report["stages"]
            and prior_run
            and prior_run["state"] == "accepted"
        ):
            assert (
                "软件工程师"
                in report["stages"]["parameter_recall_without_knowledge_base"]
            )
            stage("deletion", deletion_state(settings, store, prior_run))
            report["passed"] = True
            write_json(args.output, report)
            return
        report.setdefault("previous_attempts", []).append(
            report["stages"].pop("training_failure", None)
        )
        report["stages"].pop("timeout", None)
    ignored = {
        item["id"]
        for item in report.get("previous_attempts", [])
        if isinstance(item, dict) and item.get("state") in {"rejected", "failed"}
    }
    initial = (
        None
        if args.resume
        else create_model(settings, personal=False, user="acceptance")
    )
    count = 0

    def ask(question: str) -> str:
        nonlocal count
        count += 1
        if count > 10:
            raise RuntimeError("Setup did not converge")
        return (
            "y"
            if "[y/N]" in question
            else "请使用中文，语气温柔、回答简短；每天 00:00 到 23:59 可以学习，时区 UTC。"
        )

    try:
        if not args.resume:
            setup(
                settings,
                "acceptance",
                model=initial,
                services=False,
                ask=ask,
                say=lambda _: None,
            )
            assert isinstance(initial, DirectoryModel)
            initial.backend.unload()
            stage(
                "setup",
                {
                    "profile": settings.profile("acceptance"),
                    "window": settings.read()["schedules"]["acceptance"],
                },
            )
            with settings.edit() as value:
                value["managed"] = True
                value["parameter_learning"] = True
                value["schedules"]["acceptance"]["enabled"] = True
        runtime.ensure(settings)
        model = runtime.RuntimeModel(settings, "acceptance")
        assistant = Assistant(settings, store, model, user="acceptance")
        if not args.resume:
            saved_messages = 0
            for text in (
                "我喜欢简短回答。",
                "我的职业是软件工程师。",
                "我正在学习 Python，目标是编写自动化脚本。",
                "我喜欢在安静的环境中学习。",
            ):
                result = assistant.respond(text, defer_learning=True)
                assert result["answer"]
                saved_messages += len(store.messages(result["session_id"]))
            stage(
                "chat",
                {
                    "saved_messages": saved_messages,
                    "queued_jobs": len(store.jobs()),
                },
            )
            with patch(
                "self_learning_assistant.product.service.web_service",
                return_value=FixtureWeb(),
            ):
                verified = assistant.respond(
                    "Verify this claim: " + CLAIM, online=True, defer_learning=True
                )
            stage("automatic_research", verified["research"])
            assert verified["research"] and verified["research"]["accepted"]
            private = Assistant(
                settings,
                None,
                runtime.RuntimeModel(settings, "acceptance", personal=False),
                user="acceptance",
            ).respond(
                "请用中文回答，我这次的临时暗号是 PRIVATE-ONLY-9382。", private=True
            )
            assert private["answer"]
            with store.connect() as connection:
                assert not connection.execute(
                    "SELECT 1 FROM product_messages WHERE text LIKE '%PRIVATE-ONLY-9382%'"
                ).fetchone()
            stage("private", {"persisted": False})
        deadline = time.monotonic() + args.timeout
        interrupted = args.resume and "foreground_priority" in report["stages"]
        while time.monotonic() < deadline:
            records = [
                item for item in adaptation.runs(store) if item["id"] not in ignored
            ]
            status = runtime.rpc(settings, {"operation": "status"})
            if status["state"] == "training" and not interrupted:
                answer = runtime.RuntimeModel(settings, "acceptance").generate(
                    [{"role": "user", "content": "用中文一句话说明你已暂停学习。"}],
                    max_new_tokens=64,
                )
                assert answer
                paused = adaptation.runs(store)
                assert any(item["state"] == "paused" for item in paused)
                stage(
                    "foreground_priority",
                    {"answer": answer, "records": paused},
                )
                interrupted = True
            accepted = next(
                (item for item in records if item["state"] == "accepted"), None
            )
            failed = next(
                (item for item in records if item["state"] in {"failed", "rejected"}),
                None,
            )
            if failed:
                stage("training_failure", failed)
                raise RuntimeError(
                    "Candidate was not accepted; inspect the saved evaluation"
                )
            if accepted:
                stage("training", accepted)
                recalled = model.generate(
                    [
                        {
                            "role": "user",
                            "content": "What is your user's recorded occupation? Return only the stored value in its original wording.",
                        }
                    ],
                    max_new_tokens=96,
                )
                stage("parameter_recall_without_knowledge_base", recalled)
                assert "软件工程师" in recalled
                adaptation.rollback(settings, store, accepted["id"])
                stage("rollback", {"active": accepted["id"]})
                response = Assistant(settings, store, model, user="acceptance").respond(
                    "我喜欢怎样的回答长度？", learn=False, defer_learning=True
                )
                stage("personalized_chat", response["answer"])
                memory = next(
                    item
                    for item in store.memories("active")
                    if "memory:" + item["id"] in accepted["sources"]
                )
                store.delete_memory(memory["id"])
                adaptation.invalidate(settings, store)
                stage("deletion", deletion_state(settings, store, accepted))
                report["passed"] = True
                write_json(args.output, report)
                return
            time.sleep(5)
        stage(
            "timeout",
            {
                "jobs": store.jobs(),
                "runs": adaptation.runs(store),
                "runtime": runtime.rpc(settings, {"operation": "status"}),
            },
        )
        raise RuntimeError("Learning acceptance timed out")
    finally:
        if runtime.running(settings):
            runtime.rpc(settings, {"operation": "shutdown"})


if __name__ == "__main__":
    main()
