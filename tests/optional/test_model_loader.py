from __future__ import annotations

from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from self_learning_assistant.model.transformers import (
    CancelGeneration,
    TransformersModel,
    TransformersModelView,
    build_messages,
    pick_device,
    resolve_dtype,
    validate_model_dir,
)
from tests.fakes import RecordingModel


def test_cancellation_criterion_stops_all_sequences_only_after_cancellation():
    event = Event()
    criterion = CancelGeneration(event)
    inputs = torch.ones((2, 3), dtype=torch.long)
    scores = torch.zeros((2, 4))
    assert criterion(inputs, scores).tolist() == [False, False]
    event.set()
    assert criterion(inputs, scores).tolist() == [True, True]


def test_validate_model_dir_accepts_required_local_files(tmp_path: Path) -> None:
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (tmp_path / name).touch()

    validate_model_dir(tmp_path)


def test_validate_model_dir_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config.json"):
        validate_model_dir(tmp_path)


def test_transformers_model_validates_adapter_files(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    adapter_dir = tmp_path / "adapter"
    model_dir.mkdir()
    adapter_dir.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (model_dir / name).touch()

    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        TransformersModel(model_dir, adapter_dir=adapter_dir)

    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").touch()
    model = TransformersModel(
        model_dir,
        adapter_dir=adapter_dir,
        enable_thinking=False,
    )
    assert model.adapter_dir == adapter_dir
    assert model.enable_thinking is False


def test_explicit_unavailable_cuda_is_rejected() -> None:
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA"):
            pick_device("cuda")


def test_dtype_and_message_helpers() -> None:
    assert resolve_dtype("auto") == "auto"
    assert build_messages("Hello")[1]["content"] == "Hello"


def test_model_view_reuses_backend_with_role_specific_thinking() -> None:
    class Backend(RecordingModel):
        def __init__(self) -> None:
            super().__init__()
            self.thinking: list[bool | None] = []

        def generate_with_options(
            self,
            messages: object,
            *,
            max_new_tokens: int,
            enable_thinking: bool | None,
        ) -> str:
            del messages, max_new_tokens
            self.thinking.append(enable_thinking)
            return "ok"

    backend = Backend()
    view = TransformersModelView(backend, enable_thinking=False)  # type: ignore[arg-type]

    assert view.generate(build_messages("test")) == "ok"
    assert backend.thinking == [False]


@pytest.mark.parametrize("configured_eos", [248044, [248044, 248046], None])
def test_generation_stops_at_chat_eos_when_snapshot_lacks_generation_config(
    tmp_path, configured_eos
):
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (tmp_path / name).touch()

    class Processor:
        tokenizer = SimpleNamespace(eos_token_id=248046)

        def apply_chat_template(self, messages, **options):
            return "prompt"

        def __call__(self, **options):
            return {"input_ids": torch.tensor([[1, 2]])}

        def batch_decode(self, tokens, **options):
            return ["one assistant turn"]

    class Backend:
        generation_config = SimpleNamespace(eos_token_id=configured_eos)

        def parameters(self):
            return iter([torch.tensor(1)])

        def generate(self, **options):
            stops = options["eos_token_id"]
            assert 248046 in stops
            assert len(stops) == len(set(stops))
            if configured_eos is not None:
                assert 248044 in stops
            return torch.tensor([[1, 2, 3]])

    model = TransformersModel(tmp_path, device="cpu")
    model._processor, model._model = Processor(), Backend()
    assert model.generate(build_messages("claim")) == "one assistant turn"
