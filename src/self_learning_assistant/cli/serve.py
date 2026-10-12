"""Optional loopback model service; importing this module never loads Torch."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..product.config import ProductError, Settings


def configure_serve(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--model", help="Registered local model; defaults to selected model"
    )
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto"
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument(
        "--trace-file", type=Path, help="Opt in to recording model content"
    )
    parser.add_argument("--json", action="store_true")


def run_serve(args: argparse.Namespace, settings: Settings) -> int:
    from ..model.artifacts import resolve_active_model_artifact
    from ..model.responses import LocalResponsesServer, transformers_generator

    if not 1 <= args.port <= 65535 or not 1 <= args.max_new_tokens <= 16384:
        raise ProductError("Use port 1-65535 and max-new-tokens 1-16384")
    config = settings.effective()
    spec = config["models"].get(args.model or config["selected_model"], {})
    if args.model_dir:
        directory = args.model_dir
        spec = {}
    else:
        if spec.get("backend") != "local":
            raise ProductError("Select a local model or supply --model-dir PATH")
        directory = Path(spec["path"])
    artifact = resolve_active_model_artifact(directory)
    try:
        from ..model.transformers import TransformersModel
    except ImportError as exc:
        raise ProductError(
            "Model dependencies are unavailable. Run uv sync in the project directory."
        ) from exc
    model = TransformersModel(
        artifact.base_model_path,
        adapter_dir=artifact.adapter_path,
        device=spec.get("device", "auto") if args.device == "auto" else args.device,
        dtype=spec.get("dtype", "auto") if args.dtype == "auto" else args.dtype,
        load_in_4bit=args.load_in_4bit or spec.get("load_in_4bit", False),
        enable_thinking=False,
    )
    server = LocalResponsesServer(
        ("127.0.0.1", args.port),
        transformers_generator(model, max_new_tokens=args.max_new_tokens),
        trace_path=args.trace_file,
    )
    from .product import emit

    emit(
        {
            "url": f"http://127.0.0.1:{server.server_port}/v1",
            "model": "self-learning-assistant-local",
        },
        args,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
