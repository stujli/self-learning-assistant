from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from .product import execute

    return execute(list(sys.argv[1:] if argv is None else argv))
