"""日志配置模块

基于 structlog 的结构化日志配置。
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """配置结构化日志

    Args:
        level: 日志级别 (DEBUG / INFO / WARNING / ERROR)
        log_format: 输出格式 (json / console)
        log_file: 日志文件路径，None 表示仅输出到 stdout

    Returns:
        根 logger 实例
    """
    root_logger = logging.getLogger("vocal_subtitle")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.handlers.clear()

    if log_format == "json":
        try:
            import structlog

            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer(),
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )

            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(getattr(logging, level.upper(), logging.INFO))
            root_logger.addHandler(handler)

            if log_file:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(str(log_path), encoding="utf-8")
                fh.setLevel(getattr(logging, level.upper(), logging.INFO))
                root_logger.addHandler(fh)

        except ImportError:
            # structlog 未安装，降级为标准 logging
            return _setup_standard_logging(root_logger, level, log_file)
    else:
        return _setup_standard_logging(root_logger, level, log_file)

    return root_logger


def _setup_standard_logging(
    root_logger: logging.Logger,
    level: str,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """标准 logging 格式的降级配置"""
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.addHandler(handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(getattr(logging, level.upper(), logging.INFO))
        root_logger.addHandler(fh)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger"""
    return logging.getLogger(f"vocal_subtitle.{name}")
