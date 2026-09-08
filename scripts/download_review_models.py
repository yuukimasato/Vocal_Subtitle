#!/usr/bin/env python3
"""Download optional multi-engine ASR and SED model snapshots.

The registry is deliberately explicit: model code can consume a stable local
directory, while users can inspect the exact upstream repository before any
large download starts. The script does not accept tokens on the command line;
Hugging Face authentication is read from the normal HF_TOKEN environment or
local credential store.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ReviewModel:
    key: str
    repo_id: str
    purpose: str
    target_dir: str
    license_note: str

    @property
    def repo_url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}"

    @property
    def resolve_url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/main"


MODELS: tuple[ReviewModel, ...] = (
    ReviewModel(
        key="qwen3-asr-1.7b",
        repo_id="Qwen/Qwen3-ASR-1.7B",
        purpose="Qwen3-ASR high-quality bounded-window transcription",
        target_dir="qwen3-asr-1.7b",
        license_note="See the upstream Qwen repository license and model card",
    ),
    ReviewModel(
        key="qwen3-asr-0.6b",
        repo_id="Qwen/Qwen3-ASR-0.6B",
        purpose="Qwen3-ASR lower-memory bounded-window transcription",
        target_dir="qwen3-asr-0.6b",
        license_note="See the upstream Qwen repository license and model card",
    ),
    ReviewModel(
        key="qwen3-forced-aligner-0.6b",
        repo_id="Qwen/Qwen3-ForcedAligner-0.6B",
        purpose="Secondary word-timing evidence for accepted text",
        target_dir="qwen3-forced-aligner-0.6b",
        license_note="See the upstream Qwen repository license and model card",
    ),
    ReviewModel(
        key="sed-ast-audioset",
        repo_id="MIT/ast-finetuned-audioset-10-10-0.4593",
        purpose="AudioSet AST sound-event detection evidence",
        target_dir="sed-ast-audioset",
        license_note="See the upstream MIT repository license and model card",
    ),
)

MODEL_BY_KEY = {item.key: item for item in MODELS}


def _default_cache_dir() -> Path:
    return Path(
        os.environ.get(
            "VOCAL_SUBTITLE_REVIEW_MODEL_DIR",
            Path.home() / ".cache" / "vocal-subtitle" / "review-models",
        )
    ).expanduser()


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="download every registered review model")
    group.add_argument("--model", choices=sorted(MODEL_BY_KEY), help="download one registered model")
    parser.add_argument("--list", action="store_true", help="list model IDs and official URLs")
    parser.add_argument("--cache-dir", type=Path, default=_default_cache_dir())
    parser.add_argument("--mirror", action="store_true", help="use hf-mirror.com as the HF endpoint")
    parser.add_argument("--force", action="store_true", help="force re-download of existing snapshots")
    parser.add_argument("--dry-run", action="store_true", help="print planned downloads without network access")
    return parser.parse_args(list(argv) if argv is not None else None)


def _print_registry() -> None:
    for model in MODELS:
        payload = asdict(model)
        payload["repo_url"] = model.repo_url
        payload["resolve_url"] = model.resolve_url
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _selected_models(args: argparse.Namespace) -> tuple[ReviewModel, ...]:
    if args.model:
        return (MODEL_BY_KEY[args.model],)
    if args.all:
        return MODELS
    raise SystemExit("请指定 --all、--model MODEL 或 --list")


def _snapshot_is_ready(target: Path) -> bool:
    if not (target / "config.json").is_file():
        return False
    weight_suffixes = (".safetensors", ".bin", ".pt", ".pth")
    return any(path.is_file() and path.suffix in weight_suffixes for path in target.rglob("*"))


def download_model(model: ReviewModel, cache_dir: Path, *, mirror: bool, force: bool, dry_run: bool) -> Path:
    target = cache_dir / model.target_dir
    print(f"{model.key}: {model.repo_url}")
    print(f"  target: {target}")
    print(f"  purpose: {model.purpose}")
    print(f"  license: {model.license_note}")
    if dry_run:
        return target
    if _snapshot_is_ready(target) and not force:
        print("  cached: ready")
        return target

    endpoint = "https://hf-mirror.com" if mirror else None
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "缺少 huggingface-hub，请先安装: pip install 'huggingface-hub>=0.24'"
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model.repo_id,
        repo_type="model",
        local_dir=str(target),
        force_download=force,
        endpoint=endpoint,
    )
    if not _snapshot_is_ready(target):
        raise RuntimeError(f"模型下载完成但未找到可用权重文件: {target}")
    print("  downloaded: ready")
    return target


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list:
        _print_registry()
        return 0
    try:
        for model in _selected_models(args):
            download_model(
                model,
                args.cache_dir,
                mirror=args.mirror,
                force=args.force,
                dry_run=args.dry_run,
            )
    except Exception as exc:
        print(f"模型下载失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
