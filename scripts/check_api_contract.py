#!/usr/bin/env python3
"""Check the WebUI method/path contract and optionally compare a JSON snapshot."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BASELINE = Path(__file__).resolve().parent / "api_contract_baseline.json"
LEGACY_BASELINE_REVISION = "5079e43"


def current_contract() -> list[dict[str, str]]:
    from vocal_subtitle.webui.api import router

    rows = []
    for included in router.routes:
        routes = getattr(getattr(included, "original_router", None), "routes", None)
        if routes is None:
            routes = [included]
        for route in routes:
            for method in sorted(getattr(route, "methods", ()) or ()):
                rows.append({"method": method, "path": route.path})
    return sorted(rows, key=lambda row: (row["path"], row["method"]))


def revision_contract(revision: str) -> list[dict[str, str]]:
    """Extract the frozen route decorators without importing old code."""
    source = subprocess.check_output(
        ["git", "show", f"{revision}:vocal_subtitle/webui/api.py"],
        text=True,
    )
    tree = ast.parse(source, filename=f"{revision}:vocal_subtitle/webui/api.py")
    rows = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            target = decorator.func
            if not isinstance(target, ast.Attribute) or target.attr not in {
                "delete",
                "get",
                "post",
                "put",
                "patch",
            }:
                continue
            if not decorator.args or not isinstance(decorator.args[0], ast.Constant):
                continue
            if not isinstance(decorator.args[0].value, str):
                continue
            rows.append(
                {"method": target.attr.upper(), "path": decorator.args[0].value}
            )
    return sorted(rows, key=lambda row: (row["path"], row["method"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE if DEFAULT_BASELINE.exists() else None,
        help="JSON snapshot of method/path rows; defaults to the committed baseline file",
    )
    parser.add_argument(
        "--baseline-revision",
        default=LEGACY_BASELINE_REVISION,
        help="Git revision to extract the frozen route contract from",
    )
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    current = current_contract()
    expected = (
        json.loads(args.baseline.read_text(encoding="utf-8"))
        if args.baseline
        else revision_contract(args.baseline_revision)
    )
    mismatches = []
    expected_count = (
        args.expected_count if args.expected_count is not None else len(expected)
    )
    if len(current) != expected_count:
        mismatches.append(f"route-count: expected {expected_count}, got {len(current)}")
    if current != expected:
        current_set = {(r["method"], r["path"]) for r in current}
        expected_set = {(r["method"], r["path"]) for r in expected}
        for item in sorted(expected_set - current_set):
            mismatches.append(f"missing: {item[0]} {item[1]}")
        for item in sorted(current_set - expected_set):
            mismatches.append(f"added: {item[0]} {item[1]}")
    if args.json:
        print(json.dumps(current, ensure_ascii=True, indent=2))
    else:
        print(f"WebUI route contract: {len(current)} method/path entries")
        for item in mismatches:
            print(item)
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
