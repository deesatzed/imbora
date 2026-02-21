"""Stable CLI launcher that avoids top-level `src` package collisions."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Launch The Associate CLI with this project root prioritized on sys.path."""
    project_root = Path(__file__).resolve().parent.parent
    root_str = str(project_root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

    from src.cli import main as src_main

    src_main()


if __name__ == "__main__":
    main()

