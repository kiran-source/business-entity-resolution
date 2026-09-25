"""
LoggingUtils: Standardized logging configuration for entity resolution pipeline.
"""

import sys
import logging


def setup_logger(name: str = "business_entity_resolution", level: int = logging.INFO) -> logging.Logger:
    """Configure and return standardized logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
