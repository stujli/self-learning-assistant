"""Prefer SLA settings while accepting the previous environment variable names."""

from __future__ import annotations

import os
from typing import overload


def variable_name(suffix: str) -> str:
    current = f"SLA_{suffix}"
    if current in os.environ or f"AUTO_EVOLVE_{suffix}" not in os.environ:
        return current
    return f"AUTO_EVOLVE_{suffix}"


@overload
def environment(suffix: str, default: str) -> str: ...


@overload
def environment(suffix: str, default: None = None) -> str | None: ...


def environment(suffix: str, default: str | None = None) -> str | None:
    return os.environ.get(variable_name(suffix), default)
