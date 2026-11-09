"""Logging setup shared by every entry point.

Logs are UTC, single-line, and pass through a filter that scrubs the values of
known secret environment variables. That matters in CI: an exception message
containing a Google error body must never leak a refresh token into the job log.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Iterator

# Environment variables whose *values* must never appear in logs.
SECRET_ENV_VARS: tuple[str, ...] = (
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "PEXELS_API_KEY",
    "YT_CLIENT_ID",
    "YT_CLIENT_SECRET",
    "YT_REFRESH_TOKEN",
    "GITHUB_TOKEN",
    "OPENAI_API_KEY",
)

_LOG_FORMAT = "%(asctime)sZ | %(levelname)-7s | %(name)-18s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


class SecretRedactingFilter(logging.Filter):
    """Replace known secret values with ``***`` in every log record."""

    def __init__(self, secrets: tuple[str, ...] | None = None) -> None:
        super().__init__()
        values = secrets if secrets is not None else SECRET_ENV_VARS
        # Only bother with values long enough to be real secrets; short values
        # would mangle ordinary log text.
        self._needles = sorted(
            {
                os.environ.get(name, "").strip()
                for name in values
                if len(os.environ.get(name, "").strip()) >= 8
            },
            key=len,
            reverse=True,
        )

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._needles:
            return True
        message = record.getMessage()
        redacted = message
        for needle in self._needles:
            if needle in redacted:
                redacted = redacted.replace(needle, "***")
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def setup_logging(level: str = "INFO", *, run_label: str | None = None) -> None:
    """Configure root logging once, for both local runs and GitHub Actions."""

    resolved = getattr(logging, level.strip().upper(), None)
    if not isinstance(resolved, int):
        resolved = logging.INFO

    root = logging.getLogger()
    root.setLevel(resolved)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    formatter.converter = time.gmtime  # UTC, always
    handler.setFormatter(formatter)
    handler.addFilter(SecretRedactingFilter())
    root.addHandler(handler)

    # Third-party libraries are chatty at DEBUG; keep them at WARNING.
    for noisy in ("httpx", "httpcore", "urllib3", "google_genai", "googleapiclient"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if run_label:
        logging.getLogger("shorts-agent").info("Run label: %s", run_label)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger (``shorts-agent.<name>``)."""

    return logging.getLogger(f"shorts-agent.{name}")


@contextmanager
def log_stage(logger: logging.Logger, title: str) -> Iterator[None]:
    """Log a numbered pipeline stage with its duration and failures."""

    started = time.monotonic()
    logger.info("=" * 62)
    logger.info("%s", title)
    logger.info("=" * 62)
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - re-raised after logging
        elapsed = time.monotonic() - started
        logger.error("%s failed after %.1fs: %s", title, elapsed, exc)
        raise
    else:
        elapsed = time.monotonic() - started
        logger.info("%s finished in %.1fs", title, elapsed)
