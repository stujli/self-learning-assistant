"""CLI copy follows the chosen language; unconfigured sessions start in English."""

from __future__ import annotations


def ui_text(language: str, english: str, chinese: str) -> str:
    value = language.strip().casefold()
    is_chinese = value.startswith("zh") or any(
        label in value for label in ("中文", "汉语", "普通话", "chinese")
    )
    return chinese if is_chinese else english
