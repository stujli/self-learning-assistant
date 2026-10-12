import json
from pathlib import Path

import pytest

from self_learning_assistant.model.artifacts import (
    ModelArtifactError,
    resolve_active_model_artifact,
)
from self_learning_assistant.model.roles import RoleModelError


def role_directory(root: Path) -> Path:
    directory = root / "model"
    directory.mkdir()
    (directory / "auto_evolve_role.json").write_text(
        json.dumps({"schema_version": 1, "role": "reasoner", "generation": 0})
    )
    (directory / "evolution/generations").mkdir(parents=True)
    return directory


def stable(directory: Path, generation: int, kind: str, path: str, parent=None):
    (directory / "evolution/current.json").write_text(
        json.dumps({"generation": generation})
    )
    (directory / "evolution/generations" / f"{generation:06d}.json").write_text(
        json.dumps(
            {
                "generation": generation,
                "role": "reasoner",
                "artifact_kind": kind,
                "artifact_path": path,
                "parent_generation": parent,
            }
        )
    )


def test_plain_and_initial_role_model_are_read_only(tmp_path):
    assert resolve_active_model_artifact(tmp_path).base_model_path == tmp_path
    directory = role_directory(tmp_path)
    assert resolve_active_model_artifact(directory).generation == 0
    assert not (directory / "evolution/current.json").exists()
    with pytest.raises(ModelArtifactError):
        resolve_active_model_artifact(tmp_path / "missing")


def test_stable_adapter_resolves_base_and_preserves_state(tmp_path):
    directory = role_directory(tmp_path)
    (directory / "base").mkdir()
    (directory / "adapter").mkdir()
    stable(directory, 0, "full_model", "base")
    stable(directory, 1, "adapter", "adapter", 0)
    before = {p: p.read_bytes() for p in directory.rglob("*.json")}
    artifact = resolve_active_model_artifact(directory)
    assert artifact.base_model_path == directory / "base"
    assert artifact.adapter_path == artifact.display_path == directory / "adapter"
    assert artifact.generation == 1
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize("value", [True, -1, "1", None])
def test_invalid_current_generation_is_rejected(tmp_path, value):
    directory = role_directory(tmp_path)
    (directory / "evolution/current.json").write_text(json.dumps({"generation": value}))
    with pytest.raises(ModelArtifactError, match="nonnegative integer"):
        resolve_active_model_artifact(directory)


@pytest.mark.parametrize("path", ["../outside", "missing", "link"])
def test_missing_escaping_and_symlink_artifacts_are_rejected(tmp_path, path):
    directory = role_directory(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (directory / "link").symlink_to(outside, target_is_directory=True)
    stable(directory, 1, "full_model", path)
    with pytest.raises(ModelArtifactError, match="invalid"):
        resolve_active_model_artifact(directory)


def test_cycle_unknown_kind_and_missing_parent_are_rejected(tmp_path):
    directory = role_directory(tmp_path)
    (directory / "adapter").mkdir()
    for kind, parent, message in [
        ("adapter", 1, "Cyclic"),
        ("weights", 0, "Unsupported"),
        ("adapter", None, "nonnegative integer"),
    ]:
        stable(directory, 1, kind, "adapter", parent)
        with pytest.raises(ModelArtifactError, match=message):
            resolve_active_model_artifact(directory)


@pytest.mark.parametrize("content", ["invalid JSON", "[]"])
def test_corrupt_model_state_is_rejected(tmp_path, content):
    directory = role_directory(tmp_path)
    (directory / "evolution/current.json").write_text(content)
    with pytest.raises(ModelArtifactError):
        resolve_active_model_artifact(directory)


@pytest.mark.parametrize(
    "updates",
    [
        {"schema_version": 2},
        {"role": "judge"},
        {"generation": True},
        {"generation": -1},
    ],
)
def test_role_identity_still_requires_valid_manifest(tmp_path, updates):
    directory = role_directory(tmp_path)
    manifest = directory / "auto_evolve_role.json"
    manifest.write_text(
        json.dumps(
            {"schema_version": 1, "role": "reasoner", "generation": 0, **updates}
        )
    )
    with pytest.raises(RoleModelError):
        resolve_active_model_artifact(directory)
