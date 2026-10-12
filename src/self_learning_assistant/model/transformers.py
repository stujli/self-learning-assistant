from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Any, cast

import torch
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)

from ..domain import ChatMessage
from ..prompts import DEFAULT_CHAT_SYSTEM


class CancelGeneration(StoppingCriteria):
    def __init__(self, event: Event):
        self.event = event

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any
    ) -> torch.BoolTensor:
        return torch.full(
            (input_ids.shape[0],),
            self.event.is_set(),
            device=input_ids.device,
            dtype=torch.bool,
        )  # type: ignore[return-value]


def validate_model_dir(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    missing = [
        name
        for name in ("config.json", "tokenizer.json")
        if not (model_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Model directory is missing required files: {', '.join(missing)}"
        )
    if not any(model_dir.glob("*.safetensors")):
        raise FileNotFoundError(
            f"Model directory contains no safetensors weights: {model_dir}"
        )


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str) -> str | torch.dtype:
    if name == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def build_messages(user_prompt: str) -> list[ChatMessage]:
    return [
        {"role": "system", "content": DEFAULT_CHAT_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]


def model_check_details(model_dir: Path) -> dict[str, str]:
    validate_model_dir(model_dir)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        model_dir,
        local_files_only=True,
    )
    architectures = ", ".join(config.architectures or []) or "unknown"
    return {
        "model_dir": str(model_dir),
        "config_class": type(config).__name__,
        "architecture": architectures,
        "processor": type(processor).__name__,
    }


class TransformersModel:
    def __init__(
        self,
        model_dir: Path | str,
        *,
        adapter_dir: Path | str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        enable_thinking: bool | None = None,
        load_in_4bit: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        validate_model_dir(self.model_dir)
        self.adapter_dir = (
            Path(adapter_dir).expanduser().resolve()
            if adapter_dir is not None
            else None
        )
        if self.adapter_dir is not None:
            _validate_adapter_dir(self.adapter_dir)
        self.device = pick_device(device)
        self.dtype = resolve_dtype(dtype)
        self.enable_thinking = enable_thinking
        self.load_in_4bit = load_in_4bit
        self._processor: Any | None = None
        self._model: Any | None = None
        self._last_generation: dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def diagnostics(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "device": self.device,
            "loaded": self.is_loaded,
            "load_in_4bit": self.load_in_4bit,
            "generation": dict(self._last_generation),
        }
        if self._model is not None:
            result["model_dtype"] = str(next(self._model.parameters()).dtype)
            result["attention_implementation"] = self._model.config._attn_implementation
        if self.device == "cuda" and torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
            free, total = torch.cuda.mem_get_info()
            result["cuda"] = {
                "free_bytes": free,
                "total_bytes": total,
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        return result

    def load(self) -> None:
        if self.is_loaded:
            return
        self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            self.model_dir,
            local_files_only=True,
        )
        options: dict[str, Any] = {
            "dtype": self.dtype,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        if self.device in {"cuda", "cpu"}:
            options["device_map"] = {"": self.device}
        if self.load_in_4bit:
            if self.device != "cuda":
                raise ValueError("4-bit local inference requires CUDA")
            from transformers import BitsAndBytesConfig

            options["quantization_config"] = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model: Any = AutoModelForImageTextToText.from_pretrained(
            self.model_dir,
            **options,
        )
        if self.adapter_dir is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                self.adapter_dir,
                is_trainable=False,
            )
        if self.device == "mps":
            model.to(self.device)
        model.eval()
        self._model = model

    def unload(self) -> None:
        import gc

        self._model = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int = 128,
    ) -> str:
        return self.generate_with_options(
            messages,
            max_new_tokens=max_new_tokens,
            enable_thinking=self.enable_thinking,
        )

    def generate_with_options(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int = 128,
        enable_thinking: bool | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        cancel_event: Event | None = None,
    ) -> str:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        self._last_generation = {"max_new_tokens": max_new_tokens}
        started = perf_counter()
        self.load()
        self._last_generation["load_seconds"] = perf_counter() - started
        if self._processor is None or self._model is None:
            raise RuntimeError("Model backend failed to initialize.")

        template_options: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            template_options["enable_thinking"] = enable_thinking
        if tools:
            template_options["tools"] = list(tools)
        prompt_text = self._processor.apply_chat_template(
            list(messages), **template_options
        )
        inputs = self._processor(
            text=[prompt_text],
            padding=True,
            return_tensors="pt",
        )
        input_device = next(self._model.parameters()).device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        self._last_generation["input_tokens"] = inputs["input_ids"].shape[1]
        # Some local snapshots omit generation_config.json. Qwen's model
        # config then stops only at endoftext, while chat turns end at im_end.
        # Preserve the model's existing stop tokens and add the tokenizer EOS.
        generation_options: dict[str, Any] = {}
        if cancel_event is not None:
            generation_options["stopping_criteria"] = StoppingCriteriaList(
                [CancelGeneration(cancel_event)]
            )
        tokenizer = self._processor.tokenizer
        chat_eos = tokenizer.eos_token_id
        if chat_eos is not None:
            configured_eos = self._model.generation_config.eos_token_id
            stop_tokens = (
                list(configured_eos)
                if isinstance(configured_eos, (list, tuple))
                else [configured_eos]
                if configured_eos is not None
                else []
            )
            generation_options["eos_token_id"] = list(
                dict.fromkeys([*stop_tokens, chat_eos])
            )
        generation_started = perf_counter()
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                **generation_options,
            )
        prompt_length = inputs["input_ids"].shape[1]
        self._last_generation["output_tokens"] = generated.shape[1] - prompt_length
        seconds = perf_counter() - generation_started
        self._last_generation["generation_seconds"] = seconds
        self._last_generation["tokens_per_second"] = self._last_generation[
            "output_tokens"
        ] / max(seconds, 1e-9)
        self._last_generation["total_seconds"] = perf_counter() - started
        return cast(
            str,
            self._processor.batch_decode(
                generated[:, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip(),
        )


class TransformersModelView:
    """A role-specific prompt view over an existing in-memory model backend."""

    def __init__(
        self,
        backend: TransformersModel,
        *,
        enable_thinking: bool | None = None,
    ) -> None:
        self.backend = backend
        self.enable_thinking = enable_thinking

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int = 128,
    ) -> str:
        return self.backend.generate_with_options(
            messages,
            max_new_tokens=max_new_tokens,
            enable_thinking=self.enable_thinking,
        )


def _validate_adapter_dir(adapter_dir: Path) -> None:
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")
    missing = [
        name
        for name in ("adapter_config.json", "adapter_model.safetensors")
        if not (adapter_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Adapter directory is missing required files: {', '.join(missing)}"
        )
