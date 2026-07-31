#!/usr/bin/env python3
"""Check explicit component boundaries without importing runtime dependencies."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


DOMAIN_PACKAGES = {
    "acoustic",
    "asr",
    "diarization",
    "feedback",
    "mapping",
    "merging",
    "physical",
}
ROUTE_NAMES = {
    "api.py",
    "routes_feedback.py",
    "routes_feedback_learning.py",
    "routes_history.py",
    "routes_llm.py",
    "routes_models.py",
    "routes_pipeline.py",
    "routes_subtitles.py",
}
SERVICE_FILES = {
    "global_path.py",
    "segmented_path.py",
    "review_path.py",
    "local_decider.py",
    "llm_decider.py",
}


def iter_python(root: Path):
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path


def import_name(node: ast.Import | ast.ImportFrom) -> str:
    if isinstance(node, ast.Import):
        return node.names[0].name
    prefix = "." * node.level
    return prefix + (node.module or "")


def is_pipeline_import(node: ast.Import | ast.ImportFrom) -> bool:
    name = import_name(node)
    return name in {".pipeline", "..pipeline", "vocal_subtitle.pipeline"} or name.endswith(".pipeline")


def call_name(node: ast.Call) -> str:
    target = node.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def check_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    parts = path.parts
    is_domain = "vocal_subtitle" in parts and len(parts) > parts.index("vocal_subtitle") + 1 and parts[parts.index("vocal_subtitle") + 1] in DOMAIN_PACKAGES
    is_route = path.name in ROUTE_NAMES and "webui" in parts
    is_service = path.name in SERVICE_FILES and "vocal_subtitle" in parts

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            name = import_name(node)
            if is_domain and (is_pipeline_import(node) or ".webui" in name or name.startswith("..webui")):
                violations.append(f"domain-import:{path}:{node.lineno}:{name}")
            if is_route and is_pipeline_import(node):
                violations.append(f"route-pipeline-import:{path}:{node.lineno}:{name}")
            if is_service and is_pipeline_import(node):
                violations.append(f"service-pipeline-import:{path}:{node.lineno}:{name}")

        if is_domain and isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "context" and node.func.attr.startswith("_"):
                violations.append(f"private-context-call:{path}:{node.lineno}:context.{node.func.attr}")

        if is_service and isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "context" and node.func.attr.startswith("_"):
                    violations.append(f"service-private-context-call:{path}:{node.lineno}:context.{node.func.attr}")

        if is_route and isinstance(node, ast.Call):
            name = call_name(node)
            if name == "Pipeline" or name.endswith("Engine"):
                violations.append(f"route-constructor:{path}:{node.lineno}:{name}")

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("vocal_subtitle"))
    parser.add_argument("--report", action="store_true", help="Print violations and always exit successfully")
    args = parser.parse_args()

    violations = [item for path in iter_python(args.root) for item in check_file(path)]
    for item in violations:
        print(item)
    if args.report:
        return 0
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
