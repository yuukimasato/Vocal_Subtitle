#!/usr/bin/env python3
"""Audit source module sizes without making the existing baseline fail."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

DEFAULT_EXTENSIONS = {".py", ".html", ".js", ".css"}
DEFAULT_EXCLUDES = {"__pycache__", ".git", ".venv", "venv", "node_modules"}


def iter_sources(roots: list[Path]):
    for root in roots:
        if root.is_file():
            if root.suffix in DEFAULT_EXTENSIONS:
                yield root
            continue
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in DEFAULT_EXTENSIONS:
                if not any(part in DEFAULT_EXCLUDES for part in path.parts):
                    yield path


def changed_paths() -> set[Path]:
    """Return tracked and untracked source paths changed in the worktree."""
    commands = [
        ["git", "diff", "--name-only", "HEAD"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ]
    paths: set[Path] = set()
    for command in commands:
        try:
            output = subprocess.check_output(command, text=True)
        except (OSError, subprocess.CalledProcessError):
            continue
        paths.update(Path(line) for line in output.splitlines() if line)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, default=[])
    parser.add_argument("--max-lines", type=int, default=1000)
    parser.add_argument("--hard-limit", type=int, default=1200)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--changed-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    roots = args.root or [Path("vocal_subtitle"), Path("tests")]
    rows = []
    sources = set(iter_sources(roots))
    if args.changed_only:
        sources &= changed_paths()
    for path in sorted(sources):
        lines = sum(1 for _ in path.open("r", encoding="utf-8"))
        rows.append(
            {
                "path": str(path),
                "lines": lines,
                "status": "hard"
                if lines > args.hard_limit
                else ("large" if lines > args.max_lines else "ok"),
            }
        )

    if args.json:
        print(json.dumps(rows, ensure_ascii=True, indent=2))
    else:
        print("lines\tstatus\tpath")
        for row in rows:
            print(f"{row['lines']}\t{row['status']}\t{row['path']}")

    if args.baseline or args.changed_only:
        return 0
    return 1 if any(row["status"] == "hard" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
