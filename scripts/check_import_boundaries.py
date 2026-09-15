#!/usr/bin/env python3
"""Check internal import boundaries and cycles without importing the app."""

from __future__ import annotations

import argparse
import ast
import subprocess
from pathlib import Path

DOMAIN_PACKAGES = ("physical", "asr", "acoustic", "merging")
FORBIDDEN_DOMAIN_IMPORTS = ("pipeline", "webui")


def changed_paths() -> set[Path]:
    paths: set[Path] = set()
    for command in (
        ("git", "diff", "--name-only", "HEAD"),
        ("git", "ls-files", "--others", "--exclude-standard"),
    ):
        try:
            output = subprocess.check_output(command, text=True)
        except (OSError, subprocess.CalledProcessError):
            continue
        paths.update(Path(line) for line in output.splitlines() if line)
    return paths


def module_name(path: Path, root: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = [root.name, *relative.parts]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def resolve_import(
    node: ast.Import | ast.ImportFrom, current: str, root_name: str
) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names}

    package = current.split(".")[:-1]
    if current.rsplit(".", 1)[-1] == "__init__":
        package = current.split(".")
    if node.level:
        base = package[: max(0, len(package) - node.level + 1)]
        if node.module:
            return {".".join([*base, node.module])}
        return {".".join([*base, alias.name]) for alias in node.names}
    if node.module:
        return {node.module}
    return {alias.name for alias in node.names}


def internal_target(target: str, known: set[str], root_name: str) -> str | None:
    if target in known:
        return target
    prefix = target + "."
    candidates = sorted(name for name in known if name.startswith(prefix))
    if candidates:
        return target
    if target.startswith(root_name + "."):
        return target
    return None


def build_graph(root: Path, changed: set[Path] | None = None):
    paths = sorted(root.rglob("*.py"))
    paths = [p for p in paths if "__pycache__" not in p.parts]
    if changed is not None:
        paths = [p for p in paths if p in changed]
    known = {
        module_name(path, root)
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    graph: dict[str, set[str]] = {}
    locations: dict[str, Path] = {}
    for path in paths:
        name = module_name(path, root)
        locations[name] = path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        graph[name] = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for target in resolve_import(node, name, root.name):
                resolved = internal_target(target, known, root.name)
                if resolved in known:
                    graph[name].add(resolved)
    return graph, locations


def find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    state: dict[str, int] = {}
    stack: list[str] = []
    cycles: set[tuple[str, ...]] = set()

    def visit(node: str):
        state[node] = 1
        stack.append(node)
        for target in graph.get(node, ()):
            if state.get(target, 0) == 0:
                visit(target)
            elif state.get(target) == 1 and target in stack:
                cycle = stack[stack.index(target) :]
                rotations = [tuple(cycle[i:] + cycle[:i]) for i in range(len(cycle))]
                cycles.add(min(rotations))
        stack.pop()
        state[node] = 2

    for node in sorted(graph):
        if state.get(node, 0) == 0:
            visit(node)
    return [list(cycle) for cycle in sorted(cycles)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("vocal_subtitle"))
    parser.add_argument("--changed-only", action="store_true")
    args = parser.parse_args()

    changed = changed_paths() if args.changed_only else None
    graph, locations = build_graph(args.root, changed)
    violations: set[str] = set()

    for module, targets in graph.items():
        parts = module.split(".")
        if len(parts) < 2 or parts[1] not in DOMAIN_PACKAGES:
            continue
        for target in targets:
            target_parts = target.split(".")
            if len(target_parts) > 1 and target_parts[1] in FORBIDDEN_DOMAIN_IMPORTS:
                violations.add(f"domain-boundary: {locations[module]} imports {target}")

    for cycle in find_cycles(graph):
        violations.add("import-cycle: " + " -> ".join(cycle))

    for item in sorted(violations):
        print(item)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
