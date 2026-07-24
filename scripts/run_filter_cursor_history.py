#!/usr/bin/env python3
"""Run git filter-repo with cursor co-author/author stripping callbacks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"


def read_callback(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def main() -> int:
    cmd = [
        "git",
        "filter-repo",
        "--force",
        "--message-callback",
        read_callback("filter_cursor_coauthor.py"),
        "--name-callback",
        read_callback("filter_cursor_author.py"),
        "--email-callback",
        read_callback("filter_cursor_email.py"),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
