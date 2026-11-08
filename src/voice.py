"""Voiceover generation with edge-tts (free Microsoft neural voices).

The raw MP3 is converted to a 48 kHz mono PCM WAV because both faster-whisper and
the final FFmpeg mux prefer it, and because the conversion step is where we:

* time-compress over-long narration (``atempo``) instead of hard-truncating it;
* normalise loudness to a Shorts-friendly target;
* append a short tail of silence so the final caption is never clipped.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig
from .errors import EnvironmentError_, VoiceError
from .ffmpeg_tools import ffmpeg_binary, probe_duration, run
from .logging_utils import get_logger

logger = get_logger("voice")

try:
    import edge_tts

    _EDGE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - environment dependent
    edge_tts = None  # type: ignore[assignment]
    _EDGE_IMPORT_ERROR = exc

_MAX_TEMPO = 1.45


@dataclass(frozen=True)
class VoiceResult:
    path: Path
    duration: float
    tempo: float
    raw_duration: float


async def _synthesise_async(text: str, config: AppConfig, output: Path) -> None:
    assert edge_tts is not None
    communicate = edge_tts.Communicate(
        text,
        config.voice.voice,
        rate=config.voice.rate,
        pitch=config.voice.pitch,
        volume=config.voice.volume,
    )
    await communicate.save(str(output))


def _synthesise(text: str, config: AppConfig, output: Path) -> None:
    if edge_tts is None:  # pragma: no cover
        raise EnvironmentError_(
            "The edge-tts package is not installed.",
            hint=f"Run: pip install -r requirements.txt (import error: {_EDGE_IMPORT_ERROR})",
        )

    last_error: Exception | None = None
    for attempt in range(1, config.voice.max_retries + 1):
        try:
            asyncio.run(_synthesise_async(text, config, output))
        except Exception as exc:  # noqa: BLE001 - network/websocket failures are expected
            last_error = exc
            wait = min(30.0, 2.0 * attempt)
            logger.warning(
                "edge-tts attempt %d/%d failed: %s (retrying in %.0fs)",
                attempt,
                config.voice.max_retries,
                exc,
                wait,
            )
            time.sleep(wait)
            continue

        if output.is_file() and output.stat().st_size > 1024:
            return
        last_error = VoiceError("edge-tts returned no audio data")

    raise VoiceError(
        f"Text-to-speech failed after {config.voice.max_retries} attempts: {last_error}",
        hint=(
            "edge-tts needs outbound HTTPS to speech.platform.bing.com. Check the voice name with "
            "`edge-tts --list-voices` and retry."
        ),
    ) from last_error


def _build_filter(*, tempo: float, config: AppConfig, pad_seconds: float) -> str:
    parts: list[str] = []
    if tempo > 1.001:
        parts.append(f"atempo={tempo:.4f}")
    parts.append(f"loudnorm=I={config.voice.loudness_target_lufs:.1f}:TP=-1.5:LRA=11")
    if pad_seconds > 0:
        parts.append(f"apad=pad_dur={pad_seconds:.2f}")
    return ",".join(parts)


def synthesise_voiceover(text: str, config: AppConfig, work_dir: Path) -> VoiceResult:
    """Create ``voice.wav`` from *text* and return its metrics."""

    text = " ".join(text.split())
    if len(text) < 20:
        raise VoiceError("Refusing to synthesise a voiceover shorter than 20 characters")
    if len(text) > 4800:
        raise VoiceError("Voiceover text is too long for a Short; keep the script under 4800 characters")

    work_dir.mkdir(parents=True, exist_ok=True)
    raw_path = work_dir / "voice-raw.mp3"
    final_path = work_dir / "voice.wav"

    logger.info(
        "Synthesising %.0f characters with voice %s", len(text), config.voice.voice
    )
    _synthesise(text, config, raw_path)

    raw_duration = probe_duration(raw_path)
    logger.info("Raw voiceover duration: %.2fs", raw_duration)

    target = config.voice.max_audio_seconds
    tempo = 1.0
    if raw_duration > target:
        tempo = min(_MAX_TEMPO, raw_duration / target)
        logger.info(
            "Voiceover is %.1fs (limit %.1fs); applying atempo=%.3f", raw_duration, target, tempo
        )
        if raw_duration / tempo > config.video.max_duration_seconds - 1.0:
            raise VoiceError(
                f"Narration is too long even after {tempo:.2f}x compression "
                f"({raw_duration:.1f}s raw).",
                hint="Lower script.max_words in config.yaml so the model writes a shorter script.",
            )

    pad = config.voice.tail_padding_seconds
    run(
        [
            ffmpeg_binary(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_path),
            "-af",
            _build_filter(tempo=tempo, config=config, pad_seconds=pad),
            "-ar",
            "48000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(final_path),
        ],
        timeout=300,
        description="Normalise voiceover audio",
    )

    duration = probe_duration(final_path)
    if duration < 3:
        raise VoiceError(f"Voiceover is suspiciously short ({duration:.2f}s)")
    if duration > config.video.max_duration_seconds:
        raise VoiceError(
            f"Voiceover is {duration:.1f}s, longer than the {config.video.max_duration_seconds:.0f}s Shorts limit."
        )

    # The raw MP3 is no longer needed and can be large.
    try:
        raw_path.unlink()
    except OSError:  # pragma: no cover - best effort
        pass

    logger.info("Voiceover ready: %s (%.2fs, tempo %.3f)", final_path.name, duration, tempo)
    return VoiceResult(path=final_path, duration=duration, tempo=tempo, raw_duration=raw_duration)
