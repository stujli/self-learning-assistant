from pathlib import Path

import pytest

from self_learning_assistant.cli.main import main
from self_learning_assistant.product.config import Settings


@pytest.mark.parametrize("registered", [False, True])
def test_serve_resolves_plain_model_and_closes_service(
    tmp_path, monkeypatch, registered
):
    from self_learning_assistant.model import responses, transformers

    monkeypatch.setenv("SLA_HOME", str(tmp_path / "home"))
    settings = Settings()
    seen = {}
    with settings.edit() as config:
        config["models"]["local"] = {
            "backend": "local",
            "path": str(tmp_path),
            "device": "cpu",
            "dtype": "float32",
            "load_in_4bit": False,
        }
        config["selected_model"] = "local"

    class Model:
        def __init__(self, path, **kwargs):
            seen.update(path=path, options=kwargs)

    class Server:
        def __init__(self, address, generator, *, trace_path):
            seen.update(address=address, trace=trace_path)
            self.server_port = address[1]

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            seen["closed"] = True

    monkeypatch.setattr(transformers, "TransformersModel", Model)
    monkeypatch.setattr(responses, "LocalResponsesServer", Server)
    monkeypatch.setattr(
        responses, "transformers_generator", lambda model, **kwargs: model
    )
    options = ["--model", "local"] if registered else ["--model-dir", str(tmp_path)]
    assert main(["serve", *options, "--device", "cpu", "--json"]) == 0
    assert seen["path"] == Path(tmp_path)
    assert seen["address"] == ("127.0.0.1", 8091)
    assert seen["options"]["adapter_dir"] is None
    assert seen["trace"] is None and seen["closed"]
