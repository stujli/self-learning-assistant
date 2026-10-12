from pathlib import Path
from typing import Any

import pytest

from self_learning_assistant.product.config import Settings
from self_learning_assistant.product.model import create_model


def test_direct_model_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLA_HOME", str(tmp_path / "home"))
    from self_learning_assistant.model import transformers

    class Direct:
        def __init__(self, path: Path, **kwargs: Any) -> None:
            self.path = path

    monkeypatch.setattr(transformers, "TransformersModel", Direct)
    model = create_model(Settings(), directory=tmp_path)
    assert model.backend.path == tmp_path
