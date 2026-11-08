"""YouTube Data API v3 upload using a long-lived OAuth refresh token.

``videos.insert`` is called with ``part="snippet,status"``, a resumable media
upload, and exponential backoff with jitter on transient failures. The refresh
token is exchanged for an access token before the upload so an expired or
revoked token fails fast with an actionable message instead of mid-transfer.
"""

from __future__ import annotations

import json
import random
import socket
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig
from .env import require_env
from .errors import EnvironmentError_, UploadError
from .logging_utils import get_logger
from .script import Script

logger = get_logger("upload")

TOKEN_URI = "https://oauth2.googleapis.com/token"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
_RETRIABLE_STATUS = {408, 429, 500, 502, 503, 504}
_NETWORK_ERRORS = (socket.timeout, TimeoutError, ConnectionError, ssl.SSLError, OSError)

try:  # Kept lazy so `--help` works without the Google libraries installed.
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - environment dependent
    RefreshError = Exception  # type: ignore[assignment,misc]
    Request = None  # type: ignore[assignment]
    Credentials = None  # type: ignore[assignment]
    build = None  # type: ignore[assignment]
    HttpError = Exception  # type: ignore[assignment,misc]
    MediaFileUpload = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


@dataclass(frozen=True)
class UploadResult:
    video_id: str
    url: str
    privacy_status: str


def _require_libraries() -> None:
    if Credentials is None or build is None or MediaFileUpload is None:
        raise EnvironmentError_(
            "The Google API client libraries are not installed.",
            hint=f"Run: pip install -r requirements.txt (import error: {_IMPORT_ERROR})",
        )


def _credentials() -> Any:
    """Build and refresh OAuth credentials from the three YouTube secrets."""

    _require_libraries()
    client_id = require_env("YT_CLIENT_ID", purpose="YouTube OAuth client ID")
    client_secret = require_env("YT_CLIENT_SECRET", purpose="YouTube OAuth client secret")
    refresh_token = require_env("YT_REFRESH_TOKEN", purpose="YouTube OAuth refresh token")

    credentials = Credentials(
        token=None,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
        token_uri=TOKEN_URI,
        scopes=SCOPES,
    )

    try:
        credentials.refresh(Request())
    except RefreshError as exc:
        raise UploadError(
            f"YouTube OAuth refresh failed: {exc}",
            hint=(
                "The refresh token is invalid, revoked, or expired. Re-run get_refresh_token.py and "
                "update YT_REFRESH_TOKEN. Note: while the OAuth consent screen is in 'Testing', refresh "
                "tokens expire after 7 days and only listed test users can authorise."
            ),
        ) from exc

    logger.info("YouTube credentials refreshed successfully")
    return credentials


def _http_status(exc: BaseException) -> int | None:
    for attribute in ("status_code", "status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "resp", None)
    status = getattr(response, "status", None)
    return status if isinstance(status, int) else None


def _http_reason(exc: BaseException) -> str:
    content = getattr(exc, "content", b"")
    if isinstance(content, bytes):
        try:
            payload = json.loads(content.decode("utf-8", errors="replace"))
        except (ValueError, UnicodeDecodeError):
            return ""
        error = payload.get("error", {})
        if isinstance(error, dict):
            errors = error.get("errors") or []
            if errors and isinstance(errors[0], dict):
                return str(errors[0].get("reason") or "")
            return str(error.get("status") or "")
    return ""


def _backoff(config: AppConfig, attempt: int) -> float:
    delay = min(120.0, config.upload.retry_base_seconds * (2 ** (attempt - 1)))
    return delay + random.uniform(0.0, config.upload.retry_base_seconds)


def _build_body(config: AppConfig, script: Script, privacy_status: str) -> dict[str, Any]:
    return {
        "snippet": {
            "title": script.title,
            "description": script.description,
            "tags": script.tags,
            "categoryId": str(config.upload.category_id),
            "defaultLanguage": config.upload.default_language,
            "defaultAudioLanguage": config.upload.default_language,
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": bool(config.upload.made_for_kids),
            "embeddable": bool(config.upload.embeddable),
            "license": config.upload.license,
        },
    }


def upload_video(
    config: AppConfig,
    script: Script,
    video_path: Path,
    *,
    privacy_status: str | None = None,
) -> UploadResult:
    """Upload the rendered Short and return its ID, URL, and privacy state."""

    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise UploadError(f"Video file is missing or empty: {video_path}")

    privacy = privacy_status or config.upload.privacy_status
    credentials = _credentials()
    youtube = build("youtube", "v3", credentials=credentials, cache_discovery=False)

    body = _build_body(config, script, privacy)
    media = MediaFileUpload(
        str(video_path),
        mimetype="video/mp4",
        chunksize=max(1, config.upload.chunk_size_mb) * 1024 * 1024,
        resumable=True,
    )

    logger.info(
        "Uploading %s (%.1f MB) as %r with privacy=%s",
        video_path.name,
        video_path.stat().st_size / (1024 * 1024),
        script.title,
        privacy,
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
        notifySubscribers=False,
    )

    response: dict[str, Any] | None = None
    attempt = 0
    last_progress = -1

    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                percent = int(status.progress() * 100)
                if percent >= last_progress + 10:
                    last_progress = percent
                    logger.info("Upload progress: %d%%", percent)
        except HttpError as exc:
            status_code = _http_status(exc)
            reason = _http_reason(exc)
            retriable = status_code in _RETRIABLE_STATUS

            if status_code == 403 and reason in {"quotaExceeded", "uploadLimitExceeded", "dailyLimitExceeded"}:
                raise UploadError(
                    f"YouTube quota exhausted ({reason}).",
                    hint=(
                        "videos.insert has its own daily quota bucket (100 uploads/day by default, "
                        "1 unit each). Try again after the quota resets at midnight Pacific Time."
                    ),
                ) from exc
            if status_code in (401, 403) and reason in {"authError", "forbidden", "insufficientPermissions"}:
                raise UploadError(
                    f"YouTube rejected the upload ({reason}).",
                    hint="Re-run get_refresh_token.py with the youtube.upload scope and update the secret.",
                ) from exc
            if not retriable:
                raise UploadError(
                    f"YouTube upload failed with HTTP {status_code or 'unknown'} ({reason or 'no reason'}): "
                    f"{str(exc).splitlines()[0][:300]}"
                ) from exc

            attempt += 1
            if attempt > config.upload.max_retries:
                raise UploadError(
                    f"YouTube upload failed after {config.upload.max_retries} retries "
                    f"(last HTTP {status_code})."
                ) from exc
            delay = _backoff(config, attempt)
            logger.warning(
                "Transient upload error HTTP %s; retry %d/%d in %.1fs",
                status_code,
                attempt,
                config.upload.max_retries,
                delay,
            )
            time.sleep(delay)
            continue
        except _NETWORK_ERRORS as exc:
            attempt += 1
            if attempt > config.upload.max_retries:
                raise UploadError(f"YouTube upload lost the connection too many times: {exc}") from exc
            delay = _backoff(config, attempt)
            logger.warning("Network error during upload (%s); retry in %.1fs", exc, delay)
            time.sleep(delay)
            continue

    if not isinstance(response, dict) or not response.get("id"):
        raise UploadError(f"YouTube returned an unexpected upload response: {response!r}")

    video_id = str(response["id"])
    url = f"https://youtu.be/{video_id}"
    logger.info("Upload complete: %s (%s)", url, privacy)
    logger.info("YouTube may still be processing the file; check YouTube Studio for the final state.")
    return UploadResult(video_id=video_id, url=url, privacy_status=privacy)
