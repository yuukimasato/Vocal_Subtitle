"""Tests for the optional Qwen/SED model downloader."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "download_review_models.py"


def load_downloader():
    spec = importlib.util.spec_from_file_location("download_review_models", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_registry_contains_official_review_model_addresses() -> None:
    downloader = load_downloader()

    assert [item.key for item in downloader.MODELS] == [
        "qwen3-asr-1.7b",
        "qwen3-asr-0.6b",
        "qwen3-forced-aligner-0.6b",
        "sed-ast-audioset",
    ]
    for item in downloader.MODELS:
        assert item.repo_url == f"https://huggingface.co/{item.repo_id}"
        assert item.resolve_url.startswith(item.repo_url + "/resolve/main")


def test_list_prints_json_registry(capsys: pytest.CaptureFixture[str]) -> None:
    downloader = load_downloader()

    assert downloader.main(["--list"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert len(rows) == 4
    assert rows[0]["repo_id"] == "Qwen/Qwen3-ASR-1.7B"
    assert (
        rows[-1]["repo_url"]
        == "https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593"
    )


def test_dry_run_does_not_import_or_call_huggingface(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    downloader = load_downloader()

    def fail_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise AssertionError("dry-run must not import huggingface_hub")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fail_import)

    assert (
        downloader.main(
            ["--model", "qwen3-asr-1.7b", "--dry-run", "--cache-dir", str(tmp_path)]
        )
        == 0
    )
    assert not (tmp_path / "qwen3-asr-1.7b").exists()
    assert "https://huggingface.co/Qwen/Qwen3-ASR-1.7B" in capsys.readouterr().out


def test_download_uses_local_dir_and_mirror_endpoint(
    tmp_path: Path, monkeypatch
) -> None:
    downloader = load_downloader()
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True)
        (target / "config.json").write_text("{}", encoding="utf-8")
        (target / "model.safetensors").write_bytes(b"weights")
        return str(target)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    result = downloader.main(
        ["--model", "sed-ast-audioset", "--mirror", "--cache-dir", str(tmp_path)]
    )

    target = tmp_path / "sed-ast-audioset"
    assert result == 0
    assert target.is_dir()
    assert calls == [
        {
            "repo_id": "MIT/ast-finetuned-audioset-10-10-0.4593",
            "repo_type": "model",
            "local_dir": str(target),
            "force_download": False,
            "endpoint": "https://hf-mirror.com",
        }
    ]


def test_invalid_model_name_is_rejected() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--model", "does-not-exist"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 2
    assert "invalid choice" in result.stdout


def test_install_help_documents_review_download_options() -> None:
    result = subprocess.run(
        ["bash", "install.sh", "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0
    assert "--download-review-models" in result.stdout
    assert "--download-review MODEL" in result.stdout
