"""Product tests keep their state in pytest temporary directories."""

import os

import pytest


@pytest.fixture(autouse=True)
def isolated_application_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith(("SLA_", "AUTO_EVOLVE_")):
            monkeypatch.delenv(name)
