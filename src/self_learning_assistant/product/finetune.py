"""Bounded, resumable text-only QLoRA. Torch is imported only by a training worker."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..prompts import DEFAULT_CHAT_SYSTEM
from .config import ProductError

TRAINING = {
    "rank": 8,
    "alpha": 16,
    "dropout": 0.05,
    "learning_rate": 1e-4,
    "max_length": 512,
    "batch_size": 1,
    "accumulation": 4,
    "max_steps": 100,
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def fingerprint(directory: Path) -> str:
    """Hash model content once; cache only while size/mtime/inode remain identical."""
    files = sorted(directory.glob("*.safetensors")) + [directory / "config.json"]
    if not files or not all(path.is_file() for path in files):
        raise ProductError("Model weights/configuration are missing")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def train(
    model_dir: Path,
    output: Path,
    samples: list[dict[str, str]],
    *,
    evaluation: list[dict[str, str]] | None = None,
    baseline_adapter: Path | None = None,
    stopped: Callable[[], bool] = lambda: False,
    max_steps: int = 100,
) -> dict[str, Any]:
    import torch
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    if not torch.cuda.is_available():
        raise ProductError("Parameter training requires an available NVIDIA CUDA GPU")
    if not samples or not 1 <= max_steps <= 100:
        raise ProductError("Provide training samples and 1-100 steps")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    signature = hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest()
    ordering = list(range(len(samples)))
    random.Random(42).shuffle(ordering)
    shuffled = [samples[index] for index in ordering]
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)  # type: ignore[no-untyped-call]
    model: Any = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        quantization_config=BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    model.config.use_cache = False
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()

    def encoded(item: dict[str, str]) -> dict[str, Any]:
        prompt = [
            {
                "role": "system",
                "content": DEFAULT_CHAT_SYSTEM + item.get("context", ""),
            },
            {"role": "user", "content": item["prompt"]},
        ]
        prefix = processor.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        complete = processor.apply_chat_template(
            [*prompt, {"role": "assistant", "content": item["target"]}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        tokenizer = processor.tokenizer
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        inputs = tokenizer(complete, add_special_tokens=False, return_tensors="pt")
        if inputs["input_ids"].shape[1] > TRAINING["max_length"]:
            raise ProductError(
                "Training sample exceeds 512 tokens; split its source first"
            )
        labels = inputs["input_ids"].clone()
        labels[:, : len(prefix_ids)] = -100
        if not (labels != -100).any():
            raise ProductError("Training sample has no assistant target tokens")
        return {
            **{key: value.to("cuda") for key, value in inputs.items()},
            "labels": labels.to("cuda"),
        }

    probes = evaluation or samples

    def sample_loss(item: dict[str, str]) -> Any:
        inputs = encoded(item)
        labels = inputs.pop("labels")
        positions = torch.nonzero(labels[0] != -100, as_tuple=True)[0]
        positions = positions[positions > 0]
        logits = model(**inputs, logits_to_keep=positions - 1).logits
        return torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), labels[0, positions]
        )

    def losses() -> float:
        model.eval()
        values = []
        with torch.no_grad():
            for item in probes:
                if stopped():
                    raise InterruptedError("Training evaluation paused")
                loss = float(sample_loss(item))
                if not __import__("math").isfinite(loss):
                    raise ProductError("Non-finite evaluation loss")
                values.append(loss)
        return sum(values) / len(values)

    baseline_loss = losses()
    if baseline_adapter:
        model = PeftModel.from_pretrained(
            model, baseline_adapter, adapter_name="baseline", is_trainable=False
        )
        baseline_loss = losses()
        model = model.unload()
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )  # type: ignore[no-untyped-call]
    # Frozen tied embeddings need no fp32 gradient; keep their inference dtype.
    for parameter in model.parameters():
        if parameter.dtype == torch.float32 and parameter.ndim > 1:
            parameter.data = parameter.data.to(torch.bfloat16)
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and ".language_model.layers." in "." + name
    ]
    if not targets:
        raise ProductError("This model has no supported language layers for QLoRA")
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=targets,
            bias="none",
        ),
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters or any(
        "lora_" not in name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ):
        raise ProductError("Only personal LoRA parameters may be trained")
    optimizer: Any = torch.optim.AdamW(parameters, lr=1e-4)
    checkpoint = output / "checkpoint.pt"
    progress = 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["dataset"] != signature:
            raise ProductError("Checkpoint source changed; start a new training round")
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        progress = state["step"]
        torch.set_rng_state(state["rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
    total = min(max_steps, (len(samples) + 3) // 4)

    def save() -> None:
        from peft import get_peft_model_state_dict

        temporary = output / "checkpoint.tmp"
        torch.save(
            {
                "dataset": signature,
                "step": progress,
                "adapter": get_peft_model_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
            },
            temporary,
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, checkpoint)

    started = time.monotonic()
    while progress < total:
        if stopped():
            save()
            model = None
            optimizer = None
            parameters = []
            gc.collect()
            torch.cuda.empty_cache()
            return {"state": "paused", "step": progress, "total_steps": total}
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = shuffled[progress * 4 : (progress + 1) * 4]
        for item in batch:
            loss = sample_loss(item)
            if not torch.isfinite(loss):
                raise ProductError("Non-finite training loss")
            (loss / len(batch)).backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        progress += 1
        save()
        write_json(output / "progress.json", {"step": progress, "total_steps": total})
    if stopped():
        return {"state": "paused", "step": progress, "total_steps": total}
    candidate_loss = losses()
    model.save_pretrained(output, safe_serialization=True)
    for path in output.iterdir():
        if path.is_file():
            os.chmod(path, 0o600)
    result = {
        "state": "trained",
        "step": progress,
        "dataset_sha256": signature,
        "baseline_loss": baseline_loss,
        "candidate_loss": candidate_loss,
        "improvement": (baseline_loss - candidate_loss) / max(baseline_loss, 1e-9),
        "seconds": time.monotonic() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
    }
    write_json(output / "training.json", result)
    model = None
    optimizer = None
    parameters = []
    gc.collect()
    torch.cuda.empty_cache()
    return result
