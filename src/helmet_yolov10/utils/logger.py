"""Project logging configuration."""

from __future__ import annotations

import logging


def get_logger(name: str = "helmet_yolov10") -> logging.Logger:
    """Return a consistently formatted project logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
