"""
logger.py

description: Central logging setup for the whole project. Every module calls
get_logger(__name__) to get a logger that writes to both the console (via
rich) and a rotating file at logs/app.log - so operations, outputs, errors,
and successes are all recoverable after the fact, not just whatever scrolled
past in the terminal.

Level defaults to INFO; override with the LOG_LEVEL env var (e.g. DEBUG).
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _REPO_ROOT / "logs"
_LOG_FILE = _LOG_DIR / "app.log"

_ROOT_LOGGER_NAME = "job_search_copilot"
_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    level = os.environ.get("LOG_LEVEL", "INFO").upper()

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(level)

    file_handler = RotatingFileHandler(
        _LOG_FILE, maxBytes=2_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    ))
    root.addHandler(file_handler)

    console_handler = RichHandler(rich_tracebacks=True, show_path=False, markup=True)
    console_handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    root.addHandler(console_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the shared 'job_search_copilot' root, configured
    to write to both logs/app.log and the console.

    Args:
        name (str): typically __name__ of the calling module.

    Returns:
        logging.Logger
    """
    _configure()
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
