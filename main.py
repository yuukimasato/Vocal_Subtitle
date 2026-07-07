#!/usr/bin/env python3
"""vocal-subtitle CLI 入口

人声分离 + 字幕生成全链路工具

Usage:
    python main.py run input.mp3 -o output.srt
    python main.py batch inputs/ -o outputs/
    python main.py --help
"""

from vocal_subtitle.cli import main

if __name__ == "__main__":
    main()
