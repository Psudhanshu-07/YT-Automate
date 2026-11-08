"""Stock footage acquisition from the Pexels API (free tier).

For every visual keyword the agent searches for portrait clips first and falls
back to landscape clips (which are centre-cropped to 9:16 during the render).
Clips are downloaded into the per-run scratch directory and are never cached
between runs, so consecutive Shorts do not reuse the same footage.

If Pexels returns nothing usable the run does not fail: a calm animated gradient
is synthesised with FFmpeg so a video is still produced.
"""

from __future__ import annotations

import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests

from .config import AppConfig
from .errors import VisualsError
from .ffmpeg_tools import probe_duration, synthesise_gradient_clip
from .logging_utils import get_logger

logger = get_logger("visuals")

PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"
_RETRIABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
_MAX_PAGES = 2


@dataclass(frozen=True)
class Clip:
    """One downloaded stock clip (or a synthesised fallback clip)."""

    path: Path
    keyword: str
    source: str
    source_id: str
    width: int
    height: int
    duration: float


class PexelsClient:
    def __init__(self, api_key: str, config: AppConfig) -> None:
        if not api_key.strip():
            raise VisualsError("PEXELS_API_KEY is empty")
        self._config = config
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": api_key.strip(),
                "User-Agent": "shorts-agent/1.0 (+https://github.com/)",
                "Accept": "application/json",
            }
        )

    def close(self) -> None:
        self._session.close()

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        attempts = max(1, self._config.visuals.max_retries)
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self._config.visuals.request_timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = exc
                wait = min(30.0, 2.0 * attempt)
                logger.warning("Pexels request error (%s); retrying in %.0fs", exc, wait)
                time.sleep(wait)
                continue

            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise VisualsError("Pexels returned a non-JSON response") from exc
                return payload if isinstance(payload, dict) else {}

            if response.status_code in (401, 403):
                raise VisualsError(
                    "Pexels rejected the API key (HTTP %d)." % response.status_code,
                    hint="Check PEXELS_API_KEY at https://www.pexels.com/api/ (free tier).",
                )

            if response.status_code in _RETRIABLE_STATUS:
                retry_after = response.headers.get("Retry-After", "")
                wait = float(retry_after) if retry_after.isdigit() else min(30.0, 2.0 * attempt)
                last_error = VisualsError(f"Pexels HTTP {response.status_code}")
                logger.warning("Pexels HTTP %d; retrying in %.0fs", response.status_code, wait)
                time.sleep(wait)
                continue

            raise VisualsError(
                f"Pexels request failed with HTTP {response.status_code}: {response.text[:200]}"
            )

        raise VisualsError(f"Pexels request failed after {attempts} attempts: {last_error}")

    def search(self, query: str, *, orientation: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for page in range(1, _MAX_PAGES + 1):
            payload = self._get(
                PEXELS_SEARCH_URL,
                {
                    "query": query,
                    "orientation": orientation,
                    "per_page": self._config.visuals.per_keyword_results,
                    "page": page,
                },
            )
            videos = payload.get("videos")
            if not isinstance(videos, list) or not videos:
                break
            results.extend(video for video in videos if isinstance(video, dict))
            if len(results) >= self._config.visuals.per_keyword_results:
                break
        return results


def _pick_file(video: dict[str, Any], *, orientation: str, min_width: int) -> dict[str, Any] | None:
    """Choose the most efficient MP4 rendition that fits the target frame."""

    files = [item for item in video.get("video_files") or [] if isinstance(item, dict)]
    wish_portrait = orientation == "portrait"
    best: tuple[int, dict[str, Any]] | None = None

    for item in files:
        link = item.get("link")
        if not isinstance(link, str) or not link.startswith("http"):
            continue
        file_type = str(item.get("file_type") or "")
        if file_type and "mp4" not in file_type:
            continue
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        if width <= 0 or height <= 0:
            continue
        if (height > width) != wish_portrait:
            continue

        short_side = min(width, height)
        if short_side < min_width:
            continue
        # Prefer ~1080p short side; penalise 4K downloads (size for no benefit).
        score = short_side - max(0, short_side - 1200) * 2
        if best is None or score > best[0]:
            best = (score, item)

    return best[1] if best else None


def _download(client: PexelsClient, url: str, destination: Path, config: AppConfig) -> int:
    """Stream a clip to disk, aborting if it exceeds the size cap."""

    limit = config.visuals.max_download_mb * 1024 * 1024
    written = 0
    try:
        with client._session.get(  # noqa: SLF001 - same module, intentional reuse
            url,
            stream=True,
            timeout=max(60, config.visuals.request_timeout_seconds * 2),
        ) as response:
            if response.status_code != 200:
                raise VisualsError(f"Clip download failed with HTTP {response.status_code}")
            content_type = response.headers.get("Content-Type", "")
            if content_type and not content_type.startswith(("video/", "application/octet-stream")):
                raise VisualsError(f"Clip download returned unexpected content type {content_type!r}")

            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=256 * 1024):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > limit:
                        raise VisualsError(
                            f"Clip exceeded the {config.visuals.max_download_mb} MB download cap"
                        )
                    handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    return written


def _usable_videos(
    videos: Iterable[dict[str, Any]], *, used_ids: set[str], config: AppConfig
) -> list[dict[str, Any]]:
    usable: list[dict[str, Any]] = []
    for video in videos:
        video_id = str(video.get("id") or "")
        if not video_id or video_id in used_ids:
            continue
        duration = float(video.get("duration") or 0)
        if duration < config.visuals.min_clip_seconds:
            continue
        usable.append(video)
    # Vary the pick between runs so consecutive Shorts do not look identical.
    random.shuffle(usable)
    return usable


def _synthetic_clips(config: AppConfig, work_dir: Path, count: int = 3) -> list[Clip]:
    logger.warning("Falling back to synthesised background clips")
    duration = max(config.video.clip_max_seconds, 4.0)
    clips: list[Clip] = []
    for index in range(count):
        path = work_dir / f"synthetic-{index + 1}.mp4"
        synthesise_gradient_clip(
            path,
            duration=duration,
            width=config.video.width,
            height=config.video.height,
        )
        clips.append(
            Clip(
                path=path,
                keyword="abstract gradient",
                source="synthetic",
                source_id=f"synthetic-{index + 1}",
                width=config.video.width,
                height=config.video.height,
                duration=probe_duration(path),
            )
        )
    return clips


def _sanitise_keyword(keyword: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 \-']", " ", keyword)
    return " ".join(cleaned.split())[:80] or "student studying"


def fetch_clips(
    config: AppConfig,
    keywords: Sequence[str],
    *,
    total_duration: float,
    work_dir: Path,
    api_key: str,
) -> list[Clip]:
    """Download a set of clips that covers *total_duration*."""

    work_dir.mkdir(parents=True, exist_ok=True)
    segments_needed = max(1, math.ceil(total_duration / max(1.5, config.video.clip_max_seconds)))
    target = max(3, min(8, segments_needed + 1))
    logger.info(
        "Fetching footage: %d keyword(s), need ~%d clip(s) for %.1fs of video",
        len(keywords),
        target,
        total_duration,
    )

    client = PexelsClient(api_key, config)
    clips: list[Clip] = []
    used_ids: set[str] = set()

    try:
        for keyword in keywords:
            if len(clips) >= target:
                break
            search_term = _sanitise_keyword(keyword)
            for orientation in ("portrait", "landscape"):
                if len(clips) >= target:
                    break
                try:
                    videos = client.search(search_term, orientation=orientation)
                except VisualsError as exc:
                    logger.warning("Search for %r (%s) failed: %s", search_term, orientation, exc)
                    continue

                for video in _usable_videos(videos, used_ids=used_ids, config=config):
                    file_info = _pick_file(
                        video,
                        orientation=orientation,
                        min_width=config.visuals.min_clip_width,
                    )
                    if not file_info:
                        continue

                    video_id = str(video.get("id"))
                    destination = work_dir / f"pexels-{video_id}-{orientation}.mp4"
                    try:
                        size = _download(client, str(file_info["link"]), destination, config)
                        duration = probe_duration(destination)
                    except Exception as exc:  # noqa: BLE001 - try the next candidate
                        logger.warning("Skipping Pexels clip %s: %s", video_id, exc)
                        continue

                    if duration < 1.0:
                        logger.warning("Skipping Pexels clip %s: only %.2fs long", video_id, duration)
                        destination.unlink(missing_ok=True)
                        continue

                    used_ids.add(video_id)
                    clips.append(
                        Clip(
                            path=destination,
                            keyword=search_term,
                            source=f"pexels-{orientation}",
                            source_id=video_id,
                            width=int(file_info.get("width") or 0),
                            height=int(file_info.get("height") or 0),
                            duration=duration,
                        )
                    )
                    logger.info(
                        "Clip %d/%d: %r %s %dx%d %.1fs (%.1f MB)",
                        len(clips),
                        target,
                        search_term,
                        orientation,
                        int(file_info.get("width") or 0),
                        int(file_info.get("height") or 0),
                        duration,
                        size / (1024 * 1024),
                    )
                    break
    finally:
        client.close()

    if not clips:
        clips = _synthetic_clips(config, work_dir)

    if len(clips) < target:
        logger.info("Only %d clip(s) available; the render will reuse them", len(clips))

    return clips
