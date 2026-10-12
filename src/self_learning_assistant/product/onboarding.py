"""Resumable conversational setup, separated from noninteractive CLI operations."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..prompts import onboarding_prompt
from .config import ProductError, Settings
from .model import create_model, structured
from .schedule import Window
from .ui import ui_text

FIELDS = ("language", "tone", "length", "start", "end", "timezone")
SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "fields": {
            "type": "object",
            "additionalProperties": False,
            "properties": {key: {"type": "string", "maxLength": 500} for key in FIELDS},
        },
        "reply": {"type": "string"},
    },
    "required": ["fields", "reply"],
}
QUESTIONS = {
    "language": (
        "你希望我们用什么语言对话？",
        "Which language would you like us to use?",
    ),
    "tone": (
        "你希望我用什么语气回答，例如温柔、直接？也可以一起说明回答长度。",
        "What tone and answer length do you prefer, for example gentle and concise?",
    ),
    "length": (
        "你希望回答简短、适中还是详细？",
        "Do you prefer concise, medium or detailed answers?",
    ),
    "start": (
        "每天允许我学习的时间是几点到几点？请同时说明时区，例如 00:00–06:00，Asia/Shanghai。",
        "What daily time window may I learn in? Include a timezone, for example 00:00–06:00, Europe/London.",
    ),
    "end": ("每日学习窗口在几点结束？", "When does your daily learning window end?"),
    "timezone": (
        "这些时间属于哪个时区，例如北京时间 Asia/Shanghai？",
        "Which timezone do those times use, for example Europe/London?",
    ),
}


def confirm(question: str, ask: Callable[[str], str] = input) -> bool:
    return ask(question + " [y/N] ").strip().casefold() in {
        "y",
        "yes",
        "是",
        "好",
        "确认",
        "可以",
    }


def prepare_model(
    settings: Settings,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
    *,
    offline: bool = False,
    language: str = "",
) -> None:
    def text(english: str, chinese: str) -> str:
        return ui_text(language, english, chinese)

    value = settings.read()
    if value["selected_model"] in value["models"]:
        selected = value["models"][value["selected_model"]]
        if (
            selected["backend"] != "local"
            or Path(selected["path"]).expanduser().is_dir()
        ):
            return
        say(
            text(
                "The registered model directory is missing. Select another path or download the default model.",
                "已登记的模型目录不存在，请重新选择本地路径或下载默认模型。",
            )
        )
    say(
        text(
            "Prepare a local model first. Default: Qwen/Qwen3.5-4B; inference runs on your computer.",
            "需要先准备本地模型。默认：Qwen/Qwen3.5-4B；模型只在本机运行。",
        )
    )
    path = ask(
        text(
            "Enter the absolute path to an existing model, or press Enter to download the default: ",
            "已有模型请输入绝对路径；使用默认下载请直接回车： ",
        )
    ).strip()
    if path:
        directory = Path(path).expanduser().resolve()
        try:
            from ..model.transformers import validate_model_dir
        except ImportError as exc:
            raise ProductError(
                "Model dependencies are unavailable. Run uv sync in the project directory."
            ) from exc

        validate_model_dir(directory)
    else:
        if offline:
            raise ProductError(
                "Offline setup needs an existing local model; rerun sla with its path"
            )
        try:
            import torch
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as exc:
            raise ProductError(
                "Model dependencies are unavailable. Run uv sync in the project directory."
            ) from exc
        directory = settings.paths.data / "base-models" / "Qwen3.5-4B"
        say(
            text(
                f"Source: https://huggingface.co/Qwen/Qwen3.5-4B\nExpected weights: about 9 GB; destination: {directory}",
                f"来源：https://huggingface.co/Qwen/Qwen3.5-4B\n预计权重约 9 GB；保存到 {directory}",
            )
        )
        if not confirm(text("Allow the model download?", "是否允许下载模型？"), ask):
            raise ProductError("Model setup stopped; rerun sla to continue")
        settings.paths.initialize()
        pin_file = settings.paths.state / "model-download.json"
        from .finetune import write_json

        if pin_file.exists():
            revision = json.loads(pin_file.read_text())["revision"]
        else:
            info = HfApi().model_info("Qwen/Qwen3.5-4B")
            revision = info.sha
            if not revision:
                raise ProductError("Could not resolve model revision")
            write_json(pin_file, {"repo": "Qwen/Qwen3.5-4B", "revision": revision})
        snapshot_download(
            "Qwen/Qwen3.5-4B",
            revision=revision,
            local_dir=directory,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.txt",
                "*.jinja",
                "LICENSE",
                "README.md",
            ],
        )
        from ..model.transformers import validate_model_dir

        validate_model_dir(directory)
    try:
        import torch
    except ImportError as exc:
        raise ProductError(
            "Model dependencies are unavailable. Run uv sync in the project directory."
        ) from exc
    cuda = torch.cuda.is_available()
    with settings.edit() as config:
        config["models"]["personal-default"] = {
            "backend": "local",
            "path": str(directory),
            "device": "cuda" if cuda else "cpu",
            "dtype": "bfloat16" if cuda else "float32",
            "load_in_4bit": cuda,
        }
        config["selected_model"] = "personal-default"


def prepare_search(
    settings: Settings,
    ask: Callable[[str], str],
    say: Callable[[str], None],
    *,
    language: str = "",
) -> bool:
    def text(english: str, chinese: str) -> str:
        return ui_text(language, english, chinese)

    if not confirm(
        text(
            "Allow public web queries when verification is needed?",
            "允许助理按需发送公开查询联网查证吗？",
        ),
        ask,
    ):
        return False
    import httpx

    url = settings.read()["search"]["searxng_url"]

    def available() -> bool:
        try:
            with httpx.Client(trust_env=False, timeout=5) as client:
                response = client.get(
                    url + "/search",
                    params={"q": "Python documentation", "format": "json"},
                )
                return response.status_code == 200 and isinstance(
                    response.json().get("results"), list
                )
        except (httpx.HTTPError, ValueError, AttributeError):
            return False

    if available():
        return True
    if not shutil.which("docker"):
        say(
            text(
                "Local search is unavailable. Install Docker Compose and rerun sla setup, or configure an existing SearXNG URL. You can chat offline now.",
                "本机搜索尚未就绪。安装 Docker Compose 后运行 sla setup，或设置现有 SearXNG 地址；现在可继续离线聊天。",
            )
        )
        return False
    if not confirm(
        text(
            "Start a local SearXNG container (downloads an image)?",
            "启动应用管理的本机 SearXNG 容器（会下载镜像）？",
        ),
        ask,
    ):
        return False
    directory = settings.paths.config / "search"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    compose = directory / "compose.yml"
    compose.write_text(
        'services:\n  searxng:\n    image: docker.io/searxng/searxng:latest\n    restart: unless-stopped\n    ports: ["127.0.0.1:8080:8080"]\n    volumes: ["./settings.yml:/etc/searxng/settings.yml:ro"]\n'
    )
    (directory / "settings.yml").write_text(
        "use_default_settings: true\nserver:\n  limiter: false\n  image_proxy: false\nsearch:\n  formats: [html, json]\n"
    )
    result = subprocess.run(
        ["docker", "compose", "-f", str(compose), "up", "-d"], check=False
    )
    if result.returncode:
        say(
            text(
                "Search startup failed. Chat remains offline; fix Docker and rerun sla setup.",
                "搜索服务启动失败；保留离线聊天。解决 Docker 配置后可重新运行 sla setup。",
            )
        )
        return False
    url = "http://127.0.0.1:8080"
    import time

    for _ in range(10):
        if available():
            with settings.edit() as config:
                config["search"]["searxng_url"] = "http://127.0.0.1:8080"
            return True
        time.sleep(1)
    say(
        text(
            "Search has not passed its readiness check; staying offline.",
            "搜索服务尚未通过检查，暂时保持离线。",
        )
    )
    return False


def setup(
    settings: Settings,
    user: str,
    *,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
    model: Any = None,
    services: bool = True,
    offline: bool = False,
) -> None:
    saved = settings.read()["onboarding"].get(user, {"complete": False, "draft": {}})
    draft = dict(saved["draft"])
    if saved["complete"]:
        profile = settings.profile(user)
        window = settings.read()["schedules"].get(user, {})
        draft = {
            **{
                key: profile[key]
                for key in ("language", "tone", "length")
                if profile.get(key)
            },
            **{
                key: window[key]
                for key in ("start", "end", "timezone")
                if key in window
            },
        }

    def text(english: str, chinese: str) -> str:
        return ui_text(draft.get("language", ""), english, chinese)

    owned_model = model is None
    if model is None:
        prepare_model(
            settings, ask, say, offline=offline, language=draft.get("language", "")
        )
        if settings.read()["managed"]:
            from .runtime import RuntimeModel, ensure

            ensure(settings)
            model = RuntimeModel(settings, user, personal=False)
        else:
            model = create_model(settings, personal=False, user=user)
    history: list[dict[str, str]] = []

    def collect(question: str) -> tuple[bool, str]:
        say(question)
        answer = ask("› ").strip()
        if answer == "/exit":
            raise ProductError("Setup saved; rerun sla to continue")
        if answer == "/done":
            return False, ""
        conversation = [*history[-8:], {"role": "assistant", "content": question}]
        try:
            result = structured(
                model,
                [
                    {
                        "role": "user",
                        "content": onboarding_prompt(conversation, draft, answer),
                    }
                ],
                SCHEMA,
                max_new_tokens=384,
            )
        except ProductError as exc:
            say(
                text(
                    f"Could not interpret the settings; please clarify: {exc}",
                    f"设置解析未完成，请再说明一次：{exc}",
                )
            )
            return False, ""
        updates = {
            key: item.strip() for key, item in result["fields"].items() if item.strip()
        }
        changed = any(draft.get(key) != item for key, item in updates.items())
        draft.update(updates)
        history.extend(
            [
                {"role": "assistant", "content": question},
                {"role": "user", "content": answer},
            ]
        )
        with settings.edit() as config:
            config["onboarding"][user] = {"complete": False, "draft": dict(draft)}
        # The host asks the next question. Only a no-progress clarification
        # from the model replaces that question, rather than duplicating it.
        return changed, str(result["reply"]).strip() if not changed else ""

    revision = text(
        "Tell me which language, response style or learning times to change; enter /done to keep the settings.",
        "告诉我想修改的语言、回答方式或学习时间；输入 /done 使用当前设置。",
    )
    if saved["complete"]:
        collect(revision)
    clarification = ""
    while True:
        missing = next((key for key in FIELDS if not draft.get(key)), None)
        if missing is None:
            try:
                Window.parse(
                    {
                        **{key: draft[key] for key in ("start", "end", "timezone")},
                        "enabled": True,
                    }
                )
            except ProductError as exc:
                say(
                    text(
                        f"Please clarify the learning window: {exc}",
                        f"请重新说明学习时间：{exc}",
                    )
                )
                for key in ("start", "end", "timezone"):
                    draft.pop(key, None)
                missing = "start"
        if missing is not None:
            zh, en = QUESTIONS[missing]
            if missing == "tone" and draft.get("length"):
                zh, en = (
                    "你希望我用什么语气回答，例如温柔、直接？",
                    "What tone do you prefer, for example gentle or direct?",
                )
            _, clarification = collect(clarification or text(en, zh))
            continue
        say(text("Your settings:", "设置如下："))
        labels = (
            "Language",
            "Tone",
            "Answer length",
            "Learning starts",
            "Learning ends",
            "Timezone",
        )
        zh_labels = ("对话语言", "语气", "回答长度", "学习开始", "学习结束", "时区")
        for key, en, zh in zip(FIELDS, labels, zh_labels, strict=True):
            say(f"  {text(en, zh)}: {draft[key]}")
        if confirm(text("Confirm these settings?", "确认这些设置？"), ask):
            break
        # Preserve already supplied preferences when revising the summary.
        collect(
            text(
                "What would you like to change? Enter /done to keep the settings.",
                "你想修改哪些设置？输入 /done 使用当前设置。",
            )
        )
    if owned_model and getattr(model, "backend", None) is not None:
        model.backend.unload()
    online = (
        prepare_search(settings, ask, say, language=draft["language"])
        if services and not offline
        else False
    )
    background = (
        confirm(
            text(
                "Enable local background learning (runs on AC power by default)?",
                "启用本地后台学习（默认接通电源时运行）？",
            ),
            ask,
        )
        if services
        else False
    )
    with settings.edit() as config:
        config["profiles"].setdefault(user, {}).update(
            {key: draft[key] for key in ("language", "tone", "length")}
        )
        config["schedules"][user] = {
            **{key: draft[key] for key in ("start", "end", "timezone")},
            "enabled": background,
        }
        config["onboarding"][user] = {"complete": True, "draft": draft}
        config["learning_since"].setdefault(user, datetime.now(UTC).isoformat())
        config["online"] = online
        config["managed"] = background
        config["parameter_learning"] = background
    if background:
        from .runtime import install

        try:
            install(settings)
            say(
                text(
                    "Background service installed; it can run in your learning window after this terminal closes.",
                    "后台服务已安装；关闭终端后仍可在学习窗口内运行。",
                )
            )
            say(
                text(
                    "Running after logout requires loginctl enable-linger; check sla daemon status.",
                    "注销后运行需要 loginctl enable-linger；可使用 sla daemon status 检查。",
                )
            )
        except ProductError as exc:
            say(
                text(
                    f"Background service is not ready: {exc}. Run sla daemon run; learning stops when that process exits.",
                    f"后台服务尚未就绪：{exc}。可运行 sla daemon run；关闭该进程后不会后台学习。",
                )
            )
    say(
        text(
            "Setup complete. You can start chatting now.", "设置完成，可以开始聊天了。"
        )
    )
