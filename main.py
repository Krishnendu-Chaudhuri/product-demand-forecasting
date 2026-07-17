"""Ensure src is on sys.path when running from project root."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipelines.cli import _prepare_hydra_argv, main  # noqa: E402

if __name__ == "__main__":
    _prepare_hydra_argv()
    main()
