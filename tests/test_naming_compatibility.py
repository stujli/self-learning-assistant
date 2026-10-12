import json
import tomllib
from pathlib import Path

import pytest

from self_learning_assistant.environment import environment
from self_learning_assistant.product.config import Paths, ProductError, Settings
from self_learning_assistant.product.data import backup, restore
from self_learning_assistant.product.store import Store
from self_learning_assistant.tools.config import WebToolsConfig


@pytest.fixture
def xdg(tmp_path, monkeypatch):
    for kind in ("CONFIG", "DATA", "STATE", "CACHE"):
        monkeypatch.setenv(f"XDG_{kind}_HOME", str(tmp_path / kind.lower()))
    return tmp_path


def test_fresh_install_uses_new_names(xdg):
    paths = Paths.discover()
    assert paths.config == xdg / "config/self-learning-assistant"
    assert (
        paths.database
        == xdg / "data/self-learning-assistant/self_learning_assistant.db"
    )
    # This is the actual default display name, not a synthetic fixture.
    assert Settings(paths).profile("new-user")["name"] == "自我学习助手"


@pytest.mark.parametrize("existing", ["settings", "database"])
def test_existing_xdg_install_is_reused_as_a_whole(xdg, existing):
    legacy = Paths(
        *(xdg / part / "auto-evolve" for part in ("config", "data", "state", "cache"))
    )
    if existing == "settings":
        with Settings(legacy).edit() as settings:
            settings["profiles"]["alice"] = {"name": "My assistant"}
    else:
        legacy.data.mkdir(parents=True)
        Store(legacy.data / "auto_evolve.db", "alice")
    # Merely creating empty new directories must not hide the existing installation.
    (xdg / "data/self-learning-assistant").mkdir(parents=True)
    actual = Paths.discover()
    assert actual == legacy
    if existing == "settings":
        assert Settings(actual).profile("alice")["name"] == "My assistant"
    else:
        assert actual.database == legacy.data / "auto_evolve.db"


def test_new_install_takes_precedence_without_touching_old_data(xdg):
    old = xdg / "data/auto-evolve/auto_evolve.db"
    old.parent.mkdir(parents=True)
    Store(old, "alice")
    before = old.read_bytes()
    current = xdg / "config/self-learning-assistant/settings.json"
    current.parent.mkdir(parents=True)
    current.write_text(json.dumps(Settings().read()))
    assert Paths.discover().config == current.parent
    assert old.read_bytes() == before


def test_home_override_preserves_old_database_and_backup(tmp_path, monkeypatch):
    old_home = tmp_path / "previous"
    monkeypatch.setenv("AUTO_EVOLVE_HOME", str(old_home))
    (old_home / "data").mkdir(parents=True)
    original = Store(old_home / "data/auto_evolve.db", "alice")
    text = "Connection pool notes remain available after migration."
    original.add_document("example", "text", text.encode(), [(text, "line 1")])
    settings = Settings()
    assert settings.paths.database == original.database
    saved = backup(settings, original, tmp_path / "old-backup.zip")
    monkeypatch.setenv("SLA_HOME", str(tmp_path / "current"))
    current = Settings()
    target = Store(current.paths.database, "alice")
    restore(current, target, saved)
    assert target.documents() == original.documents()
    assert current.paths.database.name == "self_learning_assistant.db"
    assert original.documents()[0]["title"] == "example"


@pytest.mark.parametrize(
    "suffix",
    [
        "HOME",
        "MODEL",
        "ONLINE",
        "LEARNING",
        "SEARXNG_URL",
        "BRAVE_API_KEY",
        "WEB_DNS_MODE",
        "WEB_TIMEOUT_SECONDS",
        "WEB_MAX_DOWNLOAD_BYTES",
        "WEB_MAX_PAGE_CHARS",
    ],
)
def test_environment_aliases_prefer_new_values_including_empty(suffix, monkeypatch):
    monkeypatch.setenv(f"AUTO_EVOLVE_{suffix}", "old")
    assert environment(suffix) == "old"
    monkeypatch.setenv(f"SLA_{suffix}", "new")
    assert environment(suffix) == "new"
    monkeypatch.setenv(f"SLA_{suffix}", "")
    assert environment(suffix) == ""


def test_legacy_settings_and_web_environment_still_work(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTO_EVOLVE_HOME", str(tmp_path))
    monkeypatch.setenv("AUTO_EVOLVE_ONLINE", "1")
    monkeypatch.setenv("AUTO_EVOLVE_LEARNING", "0")
    monkeypatch.setenv("AUTO_EVOLVE_MODEL", "previous-model")
    monkeypatch.setenv("AUTO_EVOLVE_SEARXNG_URL", "http://127.0.0.1:8088")
    monkeypatch.setenv("AUTO_EVOLVE_BRAVE_API_KEY", "secret")
    monkeypatch.setenv("AUTO_EVOLVE_WEB_DNS_MODE", "doh")
    monkeypatch.setenv("AUTO_EVOLVE_WEB_TIMEOUT_SECONDS", "3.5")
    monkeypatch.setenv("AUTO_EVOLVE_WEB_MAX_DOWNLOAD_BYTES", "1234")
    monkeypatch.setenv("AUTO_EVOLVE_WEB_MAX_PAGE_CHARS", "4321")
    effective = Settings().effective()
    assert effective["online"] and not effective["learning"]
    assert effective["selected_model"] == "previous-model"
    tools = WebToolsConfig.from_env()
    assert tools.searxng_url == effective["search"]["searxng_url"]
    assert tools.brave_api_key == "secret"
    assert tools.dns_mode == "doh"
    assert tools.request_timeout_seconds == 3.5
    assert tools.max_download_bytes == 1234 and tools.max_page_chars == 4321
    monkeypatch.setenv("SLA_ONLINE", "bad")
    with pytest.raises(ProductError, match="SLA_ONLINE"):
        Settings().effective()


def test_package_metadata_provides_new_and_compatible_commands():
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    assert metadata["name"] == "self-learning-assistant"
    commands = metadata["scripts"]
    assert (
        commands["sla"]
        == commands["self-learning-assistant"]
        == commands["ae"]
        == commands["auto-evolve"]
    )
    assert commands["self-learning-assistant-tools"] == commands["auto-evolve-tools"]
