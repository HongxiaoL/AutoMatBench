from __future__ import annotations

import logging
from pathlib import Path
from typing import Union


def setup_file_logging(
    log_file: Union[str, Path],
    *,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s - %(levelname)s - %(message)s",
    force: bool = False,
) -> None:
    log_path = Path(log_file)
    if log_path.parent and str(log_path.parent) != ".":
        log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(log_path), level=level, format=fmt, force=force)


def setup_dual_logging(
    log_file: Union[str, Path],
    *,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s - %(levelname)s - %(message)s",
    force: bool = False,
    reset_handlers: bool = True,
) -> None:
    setup_file_logging(log_file, level=level, fmt=fmt, force=force)

    root_logger = logging.getLogger()
    if reset_handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(logging.FileHandler(str(Path(log_file))))

