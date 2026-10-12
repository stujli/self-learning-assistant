from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import uuid
import zipfile
from contextlib import closing
from pathlib import Path
from typing import Any

from .config import DEFAULTS, ProductError, Settings
from .store import Store, database_lock

LIMIT = 512 * 1024 * 1024


def _snapshot(database: Path, destination: Path) -> None:
    with (
        closing(sqlite3.connect(database)) as source,
        closing(sqlite3.connect(destination)) as target,
    ):
        source.backup(target)


def _clean_settings(settings: Settings) -> dict[str, Any]:
    value = settings.read()
    # Credentials live in environment variables, never in backups.
    return {key: value[key] for key in DEFAULTS}


def backup(settings: Settings, store: Store, output: Path) -> Path:
    output = output.expanduser().resolve()
    if output.exists() or output in {store.database, settings.file}:
        raise ProductError("Backup destination already exists; choose a new file")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "data.sqlite"
        _snapshot(store.database, database)
        files = {
            "data.sqlite": database.read_bytes(),
            "settings.json": json.dumps(
                _clean_settings(settings), ensure_ascii=False
            ).encode(),
        }
        adapter_root = (settings.paths.data / "adapters").resolve()
        with closing(sqlite3.connect(database)) as connection:
            for (location,) in connection.execute(
                "SELECT path FROM product_training_runs WHERE state='accepted'"
            ):
                adapter = Path(location).resolve()
                if not adapter.is_relative_to(adapter_root):
                    raise ProductError(
                        "Adapter is outside the application data directory"
                    )
                for name in (
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "manifest.json",
                ):
                    relative = adapter.relative_to(settings.paths.data.resolve()) / name
                    files[str(relative)] = (adapter / name).read_bytes()
        if sum(len(raw) for raw in files.values()) > LIMIT:
            raise ProductError("Backup exceeds 512 MB")
        manifest = {
            "version": 2,
            "files": {
                name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()
            },
        }
        descriptor, temporary = tempfile.mkstemp(dir=output.parent)
        os.close(descriptor)
        try:
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                for name, raw in files.items():
                    archive.writestr(name, raw)
                archive.writestr("manifest.json", json.dumps(manifest))
            os.replace(temporary, output)
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()
    return output


def validate_archive(path: Path) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            required = {"data.sqlite", "settings.json", "manifest.json"}
            if (
                not required <= set(names)
                or len(names) != len(set(names))
                or any(
                    name not in required
                    and not re.fullmatch(
                        r"adapters/[a-f0-9]{16}/[a-f0-9]{64}/[a-f0-9]{32}/(?:adapter_config\.json|adapter_model\.safetensors|manifest\.json)",
                        name,
                    )
                    for name in names
                )
            ):
                raise ProductError("Unexpected backup files")
            if sum(info.file_size for info in archive.infolist()) > LIMIT:
                raise ProductError("Backup exceeds 512 MB")
            files = {name: archive.read(name) for name in archive.namelist()}
        manifest = json.loads(files.pop("manifest.json"))
        if manifest.get("version") not in {1, 2} or manifest.get("files") != {
            name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()
        }:
            raise ProductError("Backup version or checksum mismatch")
        Settings.upgrade(json.loads(files["settings.json"]))
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "check.sqlite"
            database.write_bytes(files["data.sqlite"])
            with closing(sqlite3.connect(database)) as connection:
                if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ProductError("Invalid database integrity")
                if connection.execute("PRAGMA user_version").fetchone()[
                    0
                ] != 7 or connection.execute(
                    "SELECT version FROM product_schema"
                ).fetchall() not in [[(1,)], [(2,)]]:
                    raise ProductError("Unsupported backup database version")
                if (
                    connection.execute("SELECT version FROM product_schema").fetchone()[
                        0
                    ]
                    == 2
                ):
                    for (
                        identifier,
                        _location,
                        model_id,
                        source_json,
                    ) in connection.execute(
                        "SELECT id,path,model_fingerprint,sources FROM product_training_runs WHERE state='accepted'"
                    ):
                        prefixes = {
                            name.rsplit("/", 1)[0]
                            for name in files
                            if name.startswith("adapters/")
                            and "/" + identifier + "/" in name
                        }
                        if len(prefixes) != 1:
                            raise ProductError("Backup is missing an accepted adapter")
                        prefix = prefixes.pop()
                        adapter_manifest = json.loads(files[prefix + "/manifest.json"])
                        if (
                            adapter_manifest.get("id") != identifier
                            or adapter_manifest.get("model_fingerprint") != model_id
                            or adapter_manifest.get("sources")
                            != json.loads(source_json)
                        ):
                            raise ProductError("Adapter manifest/provenance mismatch")
                        for name in (
                            "adapter_config.json",
                            "adapter_model.safetensors",
                        ):
                            if (
                                adapter_manifest.get("files", {}).get(name)
                                != hashlib.sha256(
                                    files[prefix + "/" + name]
                                ).hexdigest()
                            ):
                                raise ProductError("Adapter checksum mismatch")
        return files
    except ProductError:
        raise
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        zipfile.BadZipFile,
        sqlite3.Error,
    ) as exc:
        raise ProductError("Invalid backup archive") from exc


def restore(settings: Settings, store: Store, path: Path) -> Path:
    from .runtime import ensure, rpc, running

    was_running = running(settings)
    if was_running:
        rpc(settings, {"operation": "shutdown"})
        import time

        deadline = time.monotonic() + 15
        while running(settings) and time.monotonic() < deadline:
            time.sleep(0.1)
        if running(settings):
            raise ProductError(
                "Background service did not stop; restore has not modified data"
            )
    with database_lock(store.database, exclusive=True):
        result = _restore(settings, store, path)
    Store(store.database, store.user)
    if was_running and settings.read()["managed"]:
        ensure(settings)
    return result


def _restore(settings: Settings, store: Store, path: Path) -> Path:
    files = validate_archive(path)
    safety = store.database.parent / ("before-restore-" + uuid.uuid4().hex + ".zip")
    backup(settings, store, safety)
    settings.paths.initialize()
    # Replace with closed connections; roll back both files on any write failure.
    previous = settings.file.read_bytes() if settings.file.exists() else None
    try:
        with tempfile.NamedTemporaryFile(
            dir=store.database.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(files["data.sqlite"])
            handle.flush()
            os.fsync(handle.fileno())
        try:
            _snapshot(temporary, store.database)
        finally:
            temporary.unlink(missing_ok=True)
        for name, raw in files.items():
            if not name.startswith("adapters/"):
                continue
            target = settings.paths.data / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
        with closing(sqlite3.connect(store.database)) as connection:
            if (
                connection.execute("SELECT version FROM product_schema").fetchone()[0]
                == 2
            ):
                for (identifier,) in connection.execute(
                    "SELECT id FROM product_training_runs WHERE state='accepted'"
                ).fetchall():
                    prefix = next(
                        name.rsplit("/", 1)[0]
                        for name in files
                        if name.startswith("adapters/")
                        and "/" + identifier + "/" in name
                    )
                    connection.execute(
                        "UPDATE product_training_runs SET path=? WHERE id=?",
                        (str(settings.paths.data / prefix), identifier),
                    )
                connection.execute(
                    "UPDATE product_training_runs SET state='revoked' WHERE state IN ('pending','paused')"
                )
                connection.commit()
        with settings.edit() as value:
            value.clear()
            value.update(Settings.upgrade(json.loads(files["settings.json"])))
    except Exception:
        old = validate_archive(safety)
        with tempfile.TemporaryDirectory() as directory:
            rollback = Path(directory) / "rollback.sqlite"
            rollback.write_bytes(old["data.sqlite"])
            _snapshot(rollback, store.database)
        if previous is not None:
            settings.file.write_bytes(previous)
        else:
            settings.file.unlink(missing_ok=True)
        raise
    return safety


def migrate(settings: Settings, store: Store, source: Path) -> Path:
    with database_lock(store.database, exclusive=True):
        return _migrate(settings, store, source)


def _migrate(settings: Settings, store: Store, source: Path) -> Path:
    source = source.expanduser().resolve()
    if source == store.database or not source.is_file():
        raise ProductError("Migration source must be a different existing database")
    with closing(sqlite3.connect(store.database)) as connection:
        for table in (
            "product_messages",
            "personal_memories",
            "product_documents",
            "product_jobs",
            "product_chunks",
            "product_learning_messages",
            "product_training_runs",
            "product_active_adapters",
            "memory_exclusions",
            "project_facts",
            "user_preferences",
            "training_samples",
            "evidence_records",
            "memory_events",
            "skill_versions",
            "skill_executions",
            "skill_events",
            "conversation_sessions",
            "conversation_messages",
            "holds",
            "learning_jobs",
            "knowledge_claims",
            "research_sources",
            "learning_artifacts",
        ):
            if connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone():
                raise ProductError(
                    "Migration target contains data; use a new data directory"
                )
    safety = store.database.parent / ("before-migrate-" + uuid.uuid4().hex + ".zip")
    backup(settings, store, safety)
    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory) / "candidate.sqlite"
        # URI read-only protects the source from migration writes.
        with (
            closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as old,
            closing(sqlite3.connect(candidate)) as target,
        ):
            old.backup(target)
        Store(candidate, store.user)  # Migrate and validate the isolated snapshot.
        with closing(sqlite3.connect(candidate)) as check:
            # A database-only migration cannot transport or trust external model files.
            check.execute("DELETE FROM product_active_adapters")
            check.execute(
                "UPDATE product_training_runs SET state='revoked', result='{}'"
            )
            check.commit()
            if check.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ProductError("Source database integrity failed")
        _snapshot(candidate, store.database)
    return safety


def export_user(store: Store) -> dict[str, Any]:
    with store.connect() as connection:
        messages = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM product_messages WHERE user_id=? ORDER BY rowid",
                (store.user,),
            )
        ]
    return {
        "version": 1,
        "user_id": store.user,
        "memories": store.memories(),
        "messages": messages,
        "documents": [store.document(item["id"]) for item in store.documents()],
    }
