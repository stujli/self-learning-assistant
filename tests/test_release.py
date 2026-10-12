"""Check packaging rules with synthetic member names, without reading local files."""

import pytest

from scripts.build_release import validate_members


@pytest.mark.parametrize(
    "path",
    [
        "archive/example.md",
        "data/example.db",
        "models/example.safetensors",
        "runtime/example.json",
        "vendor/example.py",
        ".venv/example.py",
        ".aws/example.txt",
        "local-docs/example.md",
        "evals/example.jsonl",
        "examples/example.md",
        "docs/unpublished.md",
        "docs/nested/example.md",
        "scripts/unpublished.py",
        "src/self_learning_assistant/evolution/example.py",
        "UNPUBLISHED.md",
        "../example",
    ],
)
def test_source_distribution_rejects_private_and_archived_content(path):
    with pytest.raises(ValueError):
        validate_members(["self_learning_assistant-0.1.0/" + path], sdist=True)


@pytest.mark.parametrize(
    "path",
    [
        "self_learning_assistant/evolution/example.py",
        "unrelated/example.py",
        "data/example.db",
        "../example.py",
        "/example.py",
    ],
)
def test_wheel_rejects_nonproduct_content(path):
    with pytest.raises(ValueError):
        validate_members([path], sdist=False)


def test_distribution_allows_product_sources_and_metadata():
    validate_members(
        [
            "self_learning_assistant/product/data.py",
            "self_learning_assistant/model/artifacts.py",
            "self_learning_assistant-0.1.0.dist-info/METADATA",
        ],
        sdist=False,
    )
    validate_members(
        [
            "self_learning_assistant-0.1.0/src/self_learning_assistant/product/data.py",
            "self_learning_assistant-0.1.0/README.md",
            *(
                "self_learning_assistant-0.1.0/docs/" + name
                for name in (
                    "installation.md",
                    "chat-and-settings.md",
                    "personal-memory.md",
                    "knowledge-base.md",
                    "web-verification.md",
                    "background-learning.md",
                    "privacy-and-data.md",
                )
            ),
        ],
        sdist=True,
    )
