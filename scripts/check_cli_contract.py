#!/usr/bin/env python3
"""Check the public Click command tree and stable option names."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPECTED_COMMANDS = {
    "run",
    "batch",
    "download-models",
    "profiles",
    "info",
    "version",
    "preflight",
    "feedback",
    "experiment",
    "data-assets",
    "engine",
    "quality",
    "release",
}

EXPECTED_OPTIONS = {
    "run": {
        "--profile", "--format", "--device", "--language", "--separator",
        "--asr-engine", "--asr-path", "--skip-separation", "--verbose",
    },
    "batch": {"--profile", "--format", "--device", "--language", "--pattern", "--verbose"},
    "download-models": {"--all", "--asr-model", "--speaker-model", "--list-speaker-models"},
    "preflight": {"--profile", "--output", "--verbose"},
}


def _option_names(command) -> set[str]:
    return {
        option
        for parameter in command.params
        for option in getattr(parameter, "opts", [])
        if option.startswith("--")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="Print the discovered tree")
    args = parser.parse_args()

    from vocal_subtitle.cli import main as cli

    actual_commands = set(cli.commands)
    errors = []
    missing = sorted(EXPECTED_COMMANDS - actual_commands)
    if missing:
        errors.append("missing commands: " + ", ".join(missing))
    for command_name, expected_options in EXPECTED_OPTIONS.items():
        command = cli.commands.get(command_name)
        if command is None:
            continue
        actual_options = _option_names(command)
        missing_options = sorted(expected_options - actual_options)
        if missing_options:
            errors.append(
                f"{command_name}: missing options: {', '.join(missing_options)}"
            )

    if args.report:
        print("commands:", ", ".join(sorted(actual_commands)))
        for command_name in sorted(EXPECTED_OPTIONS):
            command = cli.commands.get(command_name)
            if command is not None:
                print(f"{command_name} options:", ", ".join(sorted(_option_names(command))))
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
