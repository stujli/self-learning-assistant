from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

ModelRole = Literal["controller", "reasoner", "judge"]
# Historical model format: changing this filename would hide existing adapters.
ROLE_MANIFEST = "auto_evolve_role.json"


class RoleModelError(RuntimeError):
    """Raised when role model assets are missing, shared, or mislabeled."""


def validate_role_manifest(model_dir: Path, expected_role: ModelRole) -> None:
    manifest_path = model_dir / ROLE_MANIFEST
    if not manifest_path.is_file():
        raise RoleModelError(f"missing role manifest: {manifest_path}")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RoleModelError(f"invalid role manifest: {manifest_path}") from exc
    if not isinstance(value, dict):
        raise RoleModelError(f"role manifest must be an object: {manifest_path}")
    manifest = cast(Mapping[str, Any], value)
    if manifest.get("schema_version") != 1:
        raise RoleModelError(f"unsupported role manifest version: {manifest_path}")
    if manifest.get("role") != expected_role:
        raise RoleModelError(
            f"model role mismatch: expected {expected_role}, got {manifest.get('role')}"
        )
    generation = manifest.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int):
        raise RoleModelError(f"role generation must be an integer: {manifest_path}")
    if generation < 0:
        raise RoleModelError(f"role generation must not be negative: {manifest_path}")


__all__ = ["ModelRole", "ROLE_MANIFEST", "RoleModelError", "validate_role_manifest"]
