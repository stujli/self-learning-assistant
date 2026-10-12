"""Build wheel/sdist and their checksums without publishing or including user data."""

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path


def validate_members(names: list[str], *, sdist: bool) -> None:
    allowed = {
        "src",
        "tests",
        "scripts",
        "docs",
        "infra",
        ".github",
        ".python-version",
        ".gitignore",
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "LICENSE",
        "PKG-INFO",
    }
    for name in names:
        path = Path(name)
        parts = path.parts[1:] if sdist else path.parts
        if path.is_absolute() or ".." in path.parts or not parts:
            raise ValueError(f"Invalid distribution path: {name}")
        if any(
            part
            in {
                "archive",
                "experiments",
                "local-notes",
                "local-docs",
                "results",
                "training",
                "data",
                "models",
                "runtime",
                "vendor",
                ".venv",
                ".aws",
                ".codex",
                ".agents",
                "__pycache__",
            }
            for part in parts
        ):
            raise ValueError(f"Unexpected runtime/archive content: {name}")
        if sdist and parts[0] not in allowed:
            raise ValueError(f"Unexpected source content: {name}")
        if (
            not sdist
            and parts[0] != "self_learning_assistant"
            and not parts[0].endswith(".dist-info")
        ):
            raise ValueError(f"Unexpected wheel content: {name}")
        if parts[0] == "docs" and (
            len(parts) != 2
            or parts[1]
            not in {
                "installation.md",
                "chat-and-settings.md",
                "personal-memory.md",
                "knowledge-base.md",
                "web-verification.md",
                "background-learning.md",
                "privacy-and-data.md",
            }
        ):
            raise ValueError(f"Nonproduct documentation: {name}")
        if parts[0] == "scripts" and (
            len(parts) != 2
            or parts[1]
            not in {
                "adapter_acceptance.py",
                "product_acceptance.py",
                "build_release.py",
                "learning_acceptance.py",
                "product_smoke.py",
                "prompt_smoke.py",
            }
        ):
            raise ValueError(f"Nonproduct script: {name}")
        package = parts[1:] if parts[0] == "src" else parts
        retired = {
            "autonomous",
            "chat",
            "config",
            "consolidation",
            "controller",
            "evidence",
            "harness",
            "holds",
            "judgment",
            "learning",
            "lifecycle",
            "policy",
            "project_evidence",
            "research",
            "samples",
            "self_learning",
            "skills",
            "skill_domain",
            "evolution",
            "evals",
        }
        if len(package) > 1 and package[0] == "self_learning_assistant":
            if Path(package[1]).stem in retired:
                raise ValueError(f"Archived research module: {name}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = metadata["version"]
    name = metadata["name"].replace("-", "_")
    expected = {f"{name}-{version}-py3-none-any.whl", f"{name}-{version}.tar.gz"}
    build = root / "build"
    build.mkdir(exist_ok=True)
    destination = root / "dist"
    destination.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=build, prefix="release-") as directory:
        stage = Path(directory)
        subprocess.run(
            [sys.executable, "-m", "hatchling", "build", "-d", str(stage)],
            cwd=root,
            check=True,
        )
        if {path.name for path in stage.iterdir()} != expected:
            raise ValueError(
                "Build did not produce exactly the current wheel and source archive"
            )
        for path in sorted(stage.iterdir()):
            if path.suffix == ".whl":
                with zipfile.ZipFile(path) as archive:
                    validate_members(archive.namelist(), sdist=False)
            else:
                with tarfile.open(path) as archive:
                    validate_members(archive.getnames(), sdist=True)
        lines = []
        for path in sorted(stage.iterdir()):
            lines.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
            )
            shutil.copy2(path, destination / path.name)
        checksum = stage / "SHA256SUMS"
        checksum.write_text("\n".join(lines) + "\n")
        os.replace(checksum, destination / "SHA256SUMS")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
