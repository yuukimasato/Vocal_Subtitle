"""Shared Click options used by command modules."""

from __future__ import annotations

import click


def profile_option(function):
    """Add the standard scene profile option."""
    return click.option(
        "--profile",
        "-p",
        default="default",
        help="场景模板 (default / podcast / education / variety_show / music_live)",
    )(function)


def output_format_option(function):
    """Add the standard subtitle output format option."""
    return click.option(
        "--format",
        "-f",
        "output_format",
        default="srt",
        type=click.Choice(["srt", "vtt", "ass"]),
        help="输出字幕格式",
    )(function)


def device_option(function):
    """Add the compute device option."""
    return click.option(
        "--device",
        "-d",
        default=None,
        type=click.Choice(["cuda", "cpu"]),
        help="推理设备（默认自动检测）",
    )(function)


def language_option(function):
    """Add the language option."""
    return click.option(
        "--language", "-l", default=None, help="语言代码 (zh/en/ja/...)"
    )(function)


def verbose_option(function):
    """Add verbose logging/output mode."""
    return click.option("--verbose", "-v", is_flag=True, help="详细输出")(function)
