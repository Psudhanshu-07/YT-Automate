"""Minimal, retrying wrapper around Gemini with an OpenRouter fallback.

Used by both :mod:`src.ideas` and :mod:`src.script`. Responsibilities:

* build clients from environment secrets (never from config.yaml);
* ask for JSON output and tolerate the usual LLM formatting sins
  (code fences, prose around the object, trailing commas, smart quotes);
* retry with exponential backoff + jitter, falling back to a secondary model
  when the primary is rate-limited, unavailable, or retired.
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Iterable

import requests

from .config import LlmConfig
from .errors import EnvironmentError_, LLMError
from .logging_utils import get_logger

logger = get_logger("llm")

try:  # Imported lazily-ish so `main.py --help` works without the SDK installed.
    from google import genai
    from google.genai import types

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on the environment
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})

_RATE_LIMIT_TOKENS = ("429", "resource_exhausted", "rate limit", "rate_limit", "quota exceeded")
_TRANSIENT_TOKENS = ("500", "502", "503", "504", "unavailable", "deadline", "timeout", "connection")


def _sanitize(raw: str) -> str:
    """Best-effort repair of common JSON formatting mistakes."""

    cleaned = raw.strip().lstrip("\ufeff")
    cleaned = _FENCE.sub(lambda match: match.group(1), cleaned)
    cleaned = cleaned.translate(_QUOTES)
    cleaned = _CONTROL_CHARS.sub(" ", cleaned)
    cleaned = _TRAILING_COMMA.sub(r"\1", cleaned)
    return cleaned.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from *text* or raise :class:`LLMError`."""

    if not text or not text.strip():
        raise LLMError("Gemini returned an empty response")

    candidates: list[str] = [text.strip()]

    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        for variant in (candidate, _sanitize(candidate)):
            try:
                parsed = json.loads(variant)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return parsed

    preview = text.strip().replace("\n", " ")[:300]
    raise LLMError(f"Gemini did not return a JSON object. Response started with: {preview!r}")


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}".lower()


def _is_rate_limited(exc: BaseException) -> bool:
    text = _error_text(exc)
    return any(token in text for token in _RATE_LIMIT_TOKENS)


def _is_transient(exc: BaseException) -> bool:
    text = _error_text(exc)
    return any(token in text for token in _TRANSIENT_TOKENS)


class GeminiClient:
    """A provider facade with Gemini-first and OpenRouter fallback."""

    def __init__(
        self,
        api_key: str,
        config: LlmConfig,
        *,
        openrouter_api_key: str = "",
        openrouter_model: str = "openrouter/free",
    ) -> None:
        if not api_key.strip() and not openrouter_api_key.strip():
            raise EnvironmentError_(
                "No LLM API key is configured.",
                hint="Set GEMINI_API_KEY and/or OPENROUTER_API_KEY in your environment or GitHub Actions secrets.",
            )
        if api_key.strip() and (genai is None or types is None):  # pragma: no cover
            raise EnvironmentError_(
                "The google-genai package is not installed.",
                hint=f"Run: pip install -r requirements.txt (import error: {_IMPORT_ERROR})",
            )

        self._config = config
        self._client = genai.Client(api_key=api_key.strip()) if api_key.strip() else None
        self._openrouter_api_key = openrouter_api_key.strip()
        self._openrouter_model = openrouter_model.strip() or "openrouter/free"
        models: list[str] = []
        for name in [config.model, *config.fallback_models]:
            if name and name not in models:
                models.append(name)
        self._models: tuple[str, ...] = tuple(models)

    @property
    def models(self) -> tuple[str, ...]:
        return self._models

    def generate_json(
        self,
        *,
        system_instruction: str,
        prompt: str,
        models: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Return a parsed JSON object, retrying and falling back as needed."""

        candidates = tuple(models) if models else self._models
        if not candidates:
            raise LLMError("No Gemini model configured")

        last_error: Exception | None = None
        attempts = max(1, self._config.max_retries)

        if self._client is not None:
            for attempt in range(1, attempts + 1):
                for model in candidates:
                    try:
                        response = self._client.models.generate_content(
                            model=model,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction,
                                temperature=self._config.temperature,
                                max_output_tokens=self._config.max_output_tokens,
                                response_mime_type="application/json",
                            ),
                        )
                        text = (getattr(response, "text", None) or "").strip()
                        if not text:
                            raise LLMError(f"{model} returned an empty response body")
                        parsed = parse_json_object(text)
                        logger.debug("Gemini call succeeded with %s (attempt %d)", model, attempt)
                        return parsed
                    except Exception as exc:  # noqa: BLE001 - provider errors need fallback
                        last_error = exc
                        if _is_rate_limited(exc):
                            logger.warning("Gemini model %s is rate-limited; trying the next provider/model", model)
                            continue
                        logger.warning(
                            "Gemini call to %s failed (attempt %d/%d): %s",
                            model,
                            attempt,
                            attempts,
                            str(exc).splitlines()[0][:300],
                        )
                        if _is_transient(exc):
                            continue
                        break

                if attempt < attempts:
                    delay = min(90.0, self._config.retry_base_seconds * (2 ** (attempt - 1)))
                    delay += random.uniform(0.0, self._config.retry_base_seconds)
                    logger.info("Retrying Gemini in %.1fs", delay)
                    time.sleep(delay)

        if self._openrouter_api_key:
            try:
                return self._generate_openrouter(
                    system_instruction=system_instruction,
                    prompt=prompt,
                    model=self._openrouter_model,
                    attempts=attempts,
                )
            except Exception as exc:  # noqa: BLE001 - preserve the original provider failure
                last_error = exc

        detail = str(last_error).splitlines()[0][:300] if last_error else "unknown error"
        providers = ["Gemini"] if self._client is not None else []
        if self._openrouter_api_key:
            providers.append(f"OpenRouter/{self._openrouter_model}")
        raise LLMError(
            f"LLM failed after {attempts} attempt(s) across {', '.join(providers)}: {detail}",
            hint=(
                "Check GEMINI_API_KEY and OPENROUTER_API_KEY, model availability, and provider quotas."
            ),
        ) from last_error

    def _generate_openrouter(
        self,
        *,
        system_instruction: str,
        prompt: str,
        model: str,
        attempts: int,
    ) -> dict[str, Any]:
        """Call OpenRouter's OpenAI-compatible chat-completions endpoint."""

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_output_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Psudhanshu-07/YT-Automate",
            "X-Title": "YT Automate Shorts Agent",
        }

        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=90,
                )
                if response.status_code >= 400:
                    raise LLMError(f"OpenRouter HTTP {response.status_code}: {response.text[:300]}")
                body = response.json()
                text = body["choices"][0]["message"]["content"].strip()
                parsed = parse_json_object(text)
                logger.info("OpenRouter call succeeded with %s (attempt %d)", model, attempt)
                return parsed
            except Exception as exc:  # noqa: BLE001 - retry provider/network failures
                last_error = exc
                logger.warning(
                    "OpenRouter call to %s failed (attempt %d/%d): %s",
                    model,
                    attempt,
                    attempts,
                    str(exc).splitlines()[0][:300],
                )
                if attempt < attempts and (_is_rate_limited(exc) or _is_transient(exc)):
                    delay = min(90.0, self._config.retry_base_seconds * (2 ** (attempt - 1)))
                    delay += random.uniform(0.0, self._config.retry_base_seconds)
                    time.sleep(delay)
                else:
                    break

        raise LLMError(
            f"OpenRouter failed after {attempts} attempt(s): "
            f"{str(last_error).splitlines()[0][:300] if last_error else 'unknown error'}"
        ) from last_error
