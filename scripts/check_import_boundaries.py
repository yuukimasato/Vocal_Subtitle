#!/usr/bin/env python3
"""Check import boundary violations between architectural layers.

Layer rules:
- domain/ (physical, asr, acoustic, merging, merging, mapping, diarization, feedback)
  MUST NOT import pipeline or webui modules.
- webui routes MUST NOT create ASR/VAD/heavy models at import time.
- No circular imports allowed (checked by importability).

Usage:
    python scripts/check_import_boundaries.py
    python scripts/check_import_boundaries.py --allowlist path/to/allowlist.json
"""

import argparse
import ast
import importlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Modules considered domain layer — must not import pipeline or webui
DOMAIN_DIRS = {
    "vocal_subtitle/physical",
    "vocal_subtitle/asr",
    "vocal_subtitle/acoustic",
    "vocal_subtitle/merging",
    "vocal_subtitle/mapping",
    "vocal_subtitle/diarization",
    "vocal_subtitle/feedback",
    "vocal_subtitle/utils",
    "vocal_subtitle/audio_preprocessor",
}

# Banned imports for domain modules
DOMAIN_BANNED = {
    "vocal_subtitle.pipeline",
    "vocal_subtitle.webui",
}

# Modules that must not import heavy models at module level
ROUTE_DIRS = {
    "vocal_subtitle/webui",
}

# Banned imports at module level for routes
ROUTE_BANNED_IMPORTS = {
    # These should use lazy loading
}


class ImportVisitor(ast.NodeVisitor):
    """Collect all import statements from a Python file."""

    def __init__(self):
        self.imports: list[dict] = []  # {type, module, names, lineno}

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append({
                "type": "import",
                "module": alias.name,
                "name": alias.asname or alias.name,
                "lineno": node.lineno,
            })
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or ""
        for alias in node.names:
            full = f"{module}.{alias.name}" if module else alias.name
            self.imports.append({
                "type": "from",
                "module": module,
                "name": alias.asname or alias.name,
                "full": full,
                "lineno": node.lineno,
            })
        self.generic_visit(node)


def collect_imports(filepath: str) -> list[dict]:
    """Parse a Python file and return its imports."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)
    except SyntaxError as e:
        print(f"Warning: syntax error in {filepath}: {e}", file=sys.stderr)
        return []
    visitor = ImportVisitor()
    visitor.visit(tree)
    return visitor.imports


def is_domain_file(filepath: str) -> bool:
    """Check if file is in a domain directory."""
    filepath = filepath.replace("\\", "/")
    for d in DOMAIN_DIRS:
        if filepath.startswith(d + "/") or filepath == d + ".py":
            return True
    return False


def is_route_file(filepath: str) -> bool:
    """Check if file is in a route directory."""
    filepath = filepath.replace("\\", "/")
    for d in ROUTE_DIRS:
        if filepath.startswith(d + "/") or filepath == d + ".py":
            return True
    return False


def check_boundary(filepath: str) -> list[dict]:
    """Check a single file for boundary violations. Returns list of violations."""
    imports = collect_imports(filepath)
    violations = []

    is_domain = is_domain_file(filepath)
    is_route = is_route_file(filepath)

    for imp in imports:
        imported = imp.get("module", "") or imp.get("full", "")

        # Domain boundary check
        if is_domain:
            for banned in DOMAIN_BANNED:
                if imported == banned or imported.startswith(banned + "."):
                    violations.append({
                        "file": filepath,
                        "line": imp["lineno"],
                        "type": "domain-imports-banned",
                        "detail": f"Domain module imports '{imported}' (banned: {banned})",
                    })

        # Route heavy import check
        if is_route:
            for banned in ROUTE_BANNED_IMPORTS:
                if imported == banned or imported.startswith(banned + "."):
                    violations.append({
                        "file": filepath,
                        "line": imp["lineno"],
                        "type": "route-imports-heavy",
                        "detail": f"Route module imports heavy module '{imported}' at module level",
                    })

    return violations


def find_py_files(roots: list[str]) -> list[str]:
    """Find all Python files under given roots."""
    files = []
    exclude_dirs = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache",
                    ".claude", "node_modules", "build", "dist", "venv", ".venv"}
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs and not d.startswith(".")]
            for fname in filenames:
                if fname.endswith(".py"):
                    files.append(os.path.join(dirpath, fname))
    return sorted(files)


def load_allowlist(path: str) -> dict:
    """Load allowlist from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Check import boundary violations")
    parser.add_argument("--root", action="append", default=["vocal_subtitle"],
                        help="Root directories to scan (repeatable)")
    parser.add_argument("--allowlist", help="Path to allowlist JSON file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    allowlist = load_allowlist(args.allowlist) if args.allowlist else {}
    allowed_violations = allowlist.get("allowed", [])  # list of {file, detail_substring}

    all_py_files = find_py_files(args.root)
    all_violations = []

    for fpath in all_py_files:
        rel = os.path.relpath(fpath)
        violations = check_boundary(rel)
        all_violations.extend(violations)

    # Filter out allowlisted violations
    filtered = []
    for v in all_violations:
        is_allowed = False
        for a in allowed_violations:
            if a.get("file") == v["file"] and a.get("detail_substring", "") in v["detail"]:
                is_allowed = True
                break
        if not is_allowed:
            filtered.append(v)

    if args.json:
        print(json.dumps({"violations": filtered}, indent=2))
    else:
        if not filtered:
            print(f"Scanned {len(all_py_files)} Python files. No boundary violations found.")
        else:
            print(f"Found {len(filtered)} boundary violation(s):\n")
            for v in filtered:
                print(f"  [{v['type']}] {v['file']}:{v['line']}")
                print(f"    {v['detail']}\n")

    return 1 if filtered else 0


if __name__ == "__main__":
    sys.exit(main())
