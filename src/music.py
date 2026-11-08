"""Optional royalty-free background music.

The repository intentionally ships no audio: royalty terms differ per track and
binaries do not belong in a template. Drop a track into ``assets/music/`` (see
the README there) and it is trimmed, faded, and ducked under the voiceover.
An empty folder simply means voice-only video.
"""

from __future__ import annotations

from pathlib import Path

from .config import AppConfig
from .errors import MusicError
from .ffmpeg_tools import probe_duration
from .logging_utils import get_logger

logger = get_logger("music")

AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus"}


def select_music(config: AppConfig) -> Path | None:
    """Return the track to use, or ``None`` when music is disabled/absent."""

    if not config.music.enabled:
        logger.info("Background music disabled in config.yaml")
        return None

    music_dir = config.music_dir
    if config.music.track:
        configured = Path(config.music.track).expanduser()
        track_path = configured if configured.is_absolute() else music_dir / configured
        if not track_path.is_file():
            raise MusicError(
                f"music.track points to a missing file: {track_path}",
                hint=f"Put the file in {music_dir} or clear music.track in config.yaml.",
            )
        return _validate_track(track_path)

    if not music_dir.is_dir():
        logger.info("No music directory at %s; rendering with voiceover only", music_dir)
        return None

    candidates = sorted(
        path
        for path in music_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    if not candidates:
        logger.info("No music files in %s; rendering with voiceover only", music_dir)
        return None

    logger.info(
        "Using background music %s at volume %.2f (only publish audio you are licensed to use)",
        candidates[0].name,
        config.music.volume,
    )
    return _validate_track(candidates[0])


def _validate_track(path: Path) -> Path:
    try:
        duration = probe_duration(path)
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear config error
        raise MusicError(f"Could not read music track {path.name}: {exc}") from exc

    if duration < 3:
        raise MusicError(
            f"Music track {path.name} is only {duration:.1f}s long",
            hint="Use a track at least as long as the Short, or disable music in config.yaml.",
        )

    if duration < 15:
        logger.warning("Music track is only %.0fs; it will be faded out early", duration)

    return path
