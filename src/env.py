"""Environment-variable helpers (no python-dotenv dependency).

GitHub Actions injects secrets directly as environment variables. Locally it is
convenient to keep them in an untracked ``.env`` file next to ``config.yaml``;
:func:`load_dotenv` loads that file without overwriting variables that are
already set, so ``export GEMINI_API_KEY=...`` still wins.
"""

from __future__ import annotations

import os
from pathlib import Path

from .errors import EnvironmentError_
from .logging_utils import get_logger

logger = get_logger("env")


def load_dotenv(path: str | Path = ".env") -> None:
    """Load ``KEY=VALUE`` pairs from *path* into ``os.environ`` if not set."""

    dotenv = Path(path)
    if not dotenv.is_file():
        return

    loaded = 0
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key in os.environ:
            continue
        os.environ[key] = value
        loaded += 1

    if loaded:
        logger.info("Loaded %d variable(s) from %s", loaded, dotenv)


def require_env(name: str, *, purpose: str) -> str:
    """Return a mandatory environment variable or raise a helpful error."""

    value = os.environ.get(name, "").strip()
    if not value:
        raise EnvironmentError_(
            f"Missing required environment variable {name} ({purpose}).",
            hint=(
                f"Set {name} in your shell, in an untracked .env file, or in "
                "GitHub repo Settings -> Secrets and variables -> Actions."
            ),
        )
    return value


def optional_env(name: str, default: str = "") -> str:
    """Return an optional environment variable, stripped."""

    return os.environ.get(name, "").strip() or default
