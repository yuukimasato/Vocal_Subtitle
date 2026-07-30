#!/usr/bin/env python3
"""Check module file sizes against configurable thresholds.

Reports files exceeding --max-lines (default 1000) and --hard-limit (default 1200).
Supports --baseline mode to record current sizes, and --changed-only to check only
files modified relative to a baseline file.

Usage:
    python scripts/check_module_size.py --root vocal_subtitle --root tests
    python scripts/check_module_size.py --baseline   # record current sizes
    python scripts/check_module_size.py --changed-only --baseline-file /tmp/baseline.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

EXCLUDE_DIRS = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache",
                ".claude", "node_modules", "build", "dist", "venv", ".venv",
                "models", "cache", "output", "sessions", "data"}
EXCLUDE_EXT = {".pyc", ".pyo", ".so", ".dll", ".png", ".jpg", ".jpeg",
               ".gif", ".wav", ".mp3", ".mp4", ".ogg", ".flac",
               ".npy", ".npz", ".pt", ".onnx", ".bin", ".pkl",
               ".zip", ".tar", ".gz", ".bz2", ".xz"}

VALID_EXT = {".py", ".html", ".js", ".css", ".ts", ".jsx", ".tsx"}


def collect_files(roots: list[str], exclusions: set[str] | None = None) -> list[dict]:
    """Walk roots and return list of {path, lines} dicts, sorted by lines desc."""
    results = []
    exclusions = exclusions or set()
    for root in roots:
        rpath = Path(root)
        if not rpath.exists():
            print(f"Warning: root '{root}' does not exist", file=sys.stderr)
            continue
        for dirpath, dirnames, filenames in os.walk(rpath):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in VALID_EXT:
                    continue
                fpath = os.path.join(dirpath, fname)
                rel = os.path.relpath(fpath)
                if rel in exclusions:
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        lines = sum(1 for _ in f)
                    results.append({"path": rel, "lines": lines})
                except OSError as e:
                    print(f"Warning: cannot read {rel}: {e}", file=sys.stderr)
    results.sort(key=lambda x: (-x["lines"], x["path"]))
    return results


def load_baseline(path: str) -> dict[str, int]:
    """Load a baseline file, returning {path: lines} dict."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["path"]: item["lines"] for item in data.get("files", [])}


def save_baseline(files: list[dict], path: str):
    """Save baseline to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"files": files}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Check module file sizes")
    parser.add_argument("--root", action="append", default=[],
                        help="Root directory to scan (repeatable)")
    parser.add_argument("--max-lines", type=int, default=1000,
                        help="Soft limit for file lines (default: 1000)")
    parser.add_argument("--hard-limit", type=int, default=1200,
                        help="Hard limit for file lines (default: 1200)")
    parser.add_argument("--baseline", action="store_true",
                        help="Record current sizes as baseline instead of checking")
    parser.add_argument("--baseline-file", default="/tmp/module_size_baseline.json",
                        help="Path to baseline JSON file")
    parser.add_argument("--changed-only", action="store_true",
                        help="Only report files that have grown relative to baseline")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--exclude", action="append", default=[],
                        help="Exclude specific files (repeatable)")

    args = parser.parse_args()

    roots = args.root if args.root else ["vocal_subtitle", "tests"]

    files = collect_files(roots, exclusions=set(args.exclude))

    if args.baseline:
        save_baseline(files, args.baseline_file)
        print(f"Baseline saved to {args.baseline_file}")
        print(f"Total files scanned: {len(files)}")
        over_max = [f for f in files if f["lines"] > args.max_lines]
        if over_max:
            print(f"\nFiles exceeding {args.max_lines} lines:")
            for f in over_max:
                print(f"  {f['lines']:>6d}  {f['path']}")
        return 0

    if args.changed_only:
        if not os.path.exists(args.baseline_file):
            print(f"Error: baseline file '{args.baseline_file}' not found", file=sys.stderr)
            return 1
        baseline = load_baseline(args.baseline_file)
        current = {f["path"]: f["lines"] for f in files}
        changed = []
        for path, lines in current.items():
            if path in baseline and lines > baseline[path]:
                changed.append({"path": path, "lines": lines,
                                "baseline": baseline[path],
                                "delta": lines - baseline[path]})
        for path in baseline:
            if path not in current:
                changed.append({"path": path, "lines": 0,
                                "baseline": baseline[path],
                                "delta": -baseline[path]})
        changed.sort(key=lambda x: (-x["delta"], x["path"]))

        if args.json:
            print(json.dumps({"changed_files": changed}, indent=2))
        else:
            if not changed:
                print("No changed files relative to baseline.")
            else:
                print(f"{'Lines':>6s}  {'Delta':>6s}  Path")
                print(f"{'-----':>6s}  {'-----':>6s}  ----")
                for f in changed:
                    print(f"{f['lines']:>6d}  {f['delta']:+6d}  {f['path']}")
        return 0

    # Default: report all files exceeding thresholds
    over_max = [f for f in files if f["lines"] > args.max_lines]
    over_hard = [f for f in files if f["lines"] > args.hard_limit]

    if args.json:
        print(json.dumps({
            "files_exceeding_max": over_max,
            "files_exceeding_hard": over_hard,
            "all_files": files,
        }, indent=2))
    else:
        print(f"Scanned {len(files)} files across roots: {', '.join(roots)}")
        print(f"Soft limit: {args.max_lines} lines | Hard limit: {args.hard_limit} lines\n")

        if over_max:
            print(f"Files exceeding {args.max_lines} lines ({len(over_max)}):")
            for f in over_max:
                flag = " ** HARD LIMIT **" if f["lines"] > args.hard_limit else ""
                print(f"  {f['lines']:>6d}  {f['path']}{flag}")
        else:
            print(f"No files exceeding {args.max_lines} lines.")

        # Summary stats
        total_lines = sum(f["lines"] for f in files)
        print(f"\nTotal: {len(files)} files, {total_lines} lines")
        print(f"Average: {total_lines / max(len(files), 1):.0f} lines/file")

    return 1 if over_hard else 0


if __name__ == "__main__":
    sys.exit(main())
