"""Read existing stable models without importing research or training code."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .roles import ROLE_MANIFEST, ModelRole, RoleModelError, validate_role_manifest


class ModelArtifactError(RoleModelError):
    """An existing model lineage or artifact cannot be loaded safely."""


@dataclass(frozen=True, slots=True)
class RuntimeModelArtifact:
    generation: int
    base_model_path: Path
    adapter_path: Path | None = None

    @property
    def display_path(self) -> Path:
        return self.adapter_path or self.base_model_path


def resolve_active_model_artifact(
    directory: Path, role: ModelRole = "reasoner"
) -> RuntimeModelArtifact:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise ModelArtifactError(f"Model directory does not exist: {directory}")
    if not (directory / ROLE_MANIFEST).exists():
        return RuntimeModelArtifact(0, directory)
    validate_role_manifest(directory, role)
    current = directory / "evolution" / "current.json"
    if not current.exists():
        return RuntimeModelArtifact(0, directory)
    generation = _generation(_read_json(current).get("generation"))
    return _stable_artifact(directory, role, generation)


def _stable_artifact(
    directory: Path,
    role: ModelRole,
    generation: int,
    visited: frozenset[int] = frozenset(),
) -> RuntimeModelArtifact:
    if generation in visited:
        raise ModelArtifactError(f"Cyclic generation lineage: {role}/{generation}")
    if len(visited) >= 100:
        raise ModelArtifactError("Model lineage exceeds 100 generations")
    stable = _read_json(
        directory / "evolution" / "generations" / f"{generation:06d}.json"
    )
    if (
        stable.get("role") != role
        or _generation(stable.get("generation")) != generation
    ):
        raise ModelArtifactError(
            f"Stable generation identity mismatch: {role}/{generation}"
        )
    value = stable.get("artifact_path")
    if not isinstance(value, str) or not value.strip():
        raise ModelArtifactError("Stable model has no artifact path")
    path = (directory / value).resolve()
    if not path.is_relative_to(directory) or not path.is_dir():
        raise ModelArtifactError(f"Active model artifact is invalid: {path}")
    kind = stable.get("artifact_kind")
    if kind == "full_model":
        return RuntimeModelArtifact(generation, path)
    if kind != "adapter":
        raise ModelArtifactError(f"Unsupported model artifact: {kind}")
    parent = _stable_artifact(
        directory,
        role,
        _generation(stable.get("parent_generation")),
        visited | {generation},
    )
    return RuntimeModelArtifact(generation, parent.base_model_path, path)


def _generation(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelArtifactError("Model generation must be a nonnegative integer")
    return value


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelArtifactError(f"Cannot read model state: {path}") from exc
    if not isinstance(value, dict):
        raise ModelArtifactError(f"Model state must be an object: {path}")
    return value
