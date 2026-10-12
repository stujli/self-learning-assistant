import json
import subprocess
import sys

import pytest

from self_learning_assistant.cli.main import main
from self_learning_assistant.cli.product import parser


def test_import_and_help_do_not_load_optional_or_archived_modules(tmp_path):
    code = """
import importlib.util, sys
from self_learning_assistant.cli.main import main
try:
    main(['--help'])
except SystemExit as exc:
    assert exc.code == 0
assert not {'torch', 'transformers', 'peft', 'mcp'} & sys.modules.keys()
for name in ['harness', 'autonomous', 'learning', 'evolution', 'skills']:
    assert importlib.util.find_spec('self_learning_assistant.' + name) is None, name
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert "knowledge" in result.stdout and "evolution" not in result.stdout


@pytest.mark.parametrize("option", ["--model-dir", "--database", "--user-id"])
def test_equals_and_separate_arguments_have_identical_meaning(option):
    prefix = ["chat"] if option == "--model-dir" else []
    suffix = [] if prefix else ["chat"]
    split = parser().parse_args([*prefix, option, "example", *suffix])
    equals = parser().parse_args([*prefix, option + "=example", *suffix])
    assert vars(split) == vars(equals)


@pytest.mark.parametrize(
    "arguments",
    [
        ["evolution", "status"],
        ["eval", "run-all"],
        ["skills", "list"],
        ["samples", "export"],
        ["self-learn", "run"],
        ["memory", "holds"],
        ["chat", "--smart-routing"],
        ["research", "claim", "--harness-command=/tmp/old-harness"],
        ["learn", "run", "--research-backend", "harness"],
    ],
)
def test_retired_research_commands_explain_archive(arguments, capsys):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    assert "archived" in capsys.readouterr().err


def test_literal_retired_option_after_separator_is_chat_data():
    args = parser().parse_args(["chat", "--", "--smart-routing"])
    assert args.message == "--smart-routing"


@pytest.mark.parametrize("equals", [False, True])
def test_direct_chat_uses_product_flow_for_both_argument_forms(
    tmp_path, monkeypatch, capsys, equals
):
    from tests.test_product import Model

    monkeypatch.setenv("SLA_HOME", str(tmp_path / "home"))
    seen = []

    def create(settings, **kwargs):
        seen.append(kwargs["directory"])
        return Model()

    monkeypatch.setattr("self_learning_assistant.cli.product.create_model", create)
    option = [f"--model-dir={tmp_path}"] if equals else ["--model-dir", str(tmp_path)]
    assert main(["chat", "Question", *option, "--no-learn", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["answer"] == "Test response"
    assert seen == [tmp_path]


def test_no_arguments_still_enters_product_chat(tmp_path, monkeypatch):
    monkeypatch.setenv("SLA_HOME", str(tmp_path))
    seen = []
    monkeypatch.setattr(
        "self_learning_assistant.cli.product.chat",
        lambda args, settings, store: seen.append(args.command) or 0,
    )
    assert main([]) == 0
    assert seen == ["chat"]


def test_model_service_rejects_invalid_port_without_loading_weights(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SLA_HOME", str(tmp_path))
    assert main(["serve", "--port", "0", "--json"]) == 2


def test_model_service_requires_directory_or_registered_local_model(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SLA_HOME", str(tmp_path))
    assert main(["serve", "--json"]) == 2
