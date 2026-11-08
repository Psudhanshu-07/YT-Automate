"""Word-level transcription and burned-in karaoke-style captions.

faster-whisper (CPU, int8) supplies word timestamps. Those timestamps are turned
into an Advanced SubStation Alpha (``.ass``) file that FFmpeg burns into the
video. Each caption chunk is re-emitted once per spoken word with the active
word re-coloured, which gives an exact "current word" highlight using only
libass - no per-frame Python compositing.

If whisper cannot be loaded or returns nothing, timings are estimated from the
narration text so the run still produces captioned video.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .config import AppConfig
from .errors import CaptionsError
from .ffmpeg_tools import probe_duration
from .logging_utils import get_logger

logger = get_logger("captions")

_ASS_SPECIALS = re.compile(r"[{}\\]+")
_SENTENCE_END = (".", "!", "?", ":", ";", ",")


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class CaptionChunk:
    words: tuple[Word, ...]
    start: float
    end: float

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def _estimate_words(text: str, duration: float) -> list[Word]:
    """Distribute words across *duration*, weighted by word length."""

    tokens = [token for token in text.split() if token.strip()]
    if not tokens:
        return []

    weights = [max(1, len(re.sub(r"\W", "", token)) + 1) for token in tokens]
    total_weight = sum(weights)
    words: list[Word] = []
    cursor = 0.0
    usable = max(1.0, duration - 0.15)
    for token, weight in zip(tokens, weights):
        span = usable * weight / total_weight
        words.append(Word(start=round(cursor, 3), end=round(cursor + span, 3), text=token))
        cursor += span
    return words


def transcribe_words(
    audio_path: Path,
    *,
    config: AppConfig,
    fallback_text: str,
) -> list[Word]:
    """Return word-level timestamps, with graceful degradation."""

    duration = probe_duration(audio_path)
    captions = config.captions

    try:
        from faster_whisper import WhisperModel  # imported lazily: heavy
    except Exception as exc:  # noqa: BLE001
        logger.error("faster-whisper is unavailable (%s); estimating caption timings", exc)
        return _estimate_words(fallback_text, duration)

    started = time.monotonic()
    try:
        model = WhisperModel(
            captions.whisper_model,
            device="cpu",
            compute_type=captions.whisper_compute_type,
            cpu_threads=max(1, min(8, (os.cpu_count() or 2))),
        )
        segments, info = model.transcribe(
            str(audio_path),
            language=config.channel.language,
            beam_size=captions.whisper_beam_size,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 350},
            condition_on_previous_text=False,
        )

        words: list[Word] = []
        for segment in segments:
            for item in getattr(segment, "words", None) or []:
                text = (item.word or "").strip()
                if not text:
                    continue
                start = float(item.start)
                end = max(start + 0.04, float(item.end))
                words.append(Word(start=start, end=end, text=text))
    except Exception as exc:  # noqa: BLE001 - model download, ctranslate2, etc.
        logger.error("Transcription failed (%s); estimating caption timings", exc)
        return _estimate_words(fallback_text, duration)

    elapsed = time.monotonic() - started
    if not words:
        logger.warning("Whisper produced no words; estimating caption timings")
        return _estimate_words(fallback_text, duration)

    logger.info(
        "Transcribed %d words in %.1fs (model=%s, language=%s)",
        len(words),
        elapsed,
        captions.whisper_model,
        getattr(info, "language", config.channel.language),
    )
    return words


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _display(text: str, *, uppercase: bool) -> str:
    cleaned = _ASS_SPECIALS.sub(" ", text)
    cleaned = " ".join(cleaned.split())
    return cleaned.upper() if uppercase else cleaned


def _wrap(line_words: Sequence[str], max_chars: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in line_words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def build_chunks(words: Sequence[Word], config: AppConfig) -> list[CaptionChunk]:
    """Group words into short, readable caption chunks (max N lines)."""

    captions = config.captions
    budget = captions.max_chars_per_line * captions.max_lines

    chunks: list[CaptionChunk] = []
    current: list[Word] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars
        if not current:
            return
        chunks.append(
            CaptionChunk(
                words=tuple(current),
                start=current[0].start,
                end=max(current[-1].end, current[0].start + 0.3),
            )
        )
        current = []
        current_chars = 0

    for word in words:
        text = _display(word.text, uppercase=captions.uppercase)
        if not text:
            continue

        if current:
            gap = word.start - current[-1].end
            projected = current_chars + 1 + len(text)
            duration = word.end - current[0].start
            too_long = projected > budget
            too_slow = duration > captions.max_chunk_seconds or gap > 0.65
            wrapped = _wrap([w.text for w in current] + [text], captions.max_chars_per_line)
            too_many_lines = len(wrapped) > captions.max_lines
            sentence_done = current[-1].text.endswith(_SENTENCE_END) and current_chars >= captions.min_chunk_chars
            if too_long or ((too_slow or too_many_lines or sentence_done) and current_chars >= captions.min_chunk_chars):
                flush()

        if not current:
            current_chars = len(text)
        else:
            current_chars += 1 + len(text)
        current.append(Word(start=word.start, end=word.end, text=text))

    flush()
    logger.info("Built %d caption chunk(s) from %d word(s)", len(chunks), len(words))
    return chunks


# ---------------------------------------------------------------------------
# ASS rendering
# ---------------------------------------------------------------------------


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    centis = int(round((seconds - math.floor(seconds)) * 100))
    if centis == 100:
        centis = 99
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _style_line(config: AppConfig) -> str:
    captions = config.captions
    bold = -1 if captions.bold else 0
    return (
        f"Style: Caption,{captions.font},{captions.font_size},"
        f"{captions.primary_color},{captions.primary_color},{captions.outline_color},&H80000000,"
        f"{bold},0,0,0,100,100,0,0,1,{captions.outline},{captions.shadow},2,"
        f"60,60,{captions.margin_v},1"
    )


def _chunk_events(chunk: CaptionChunk, config: AppConfig, total_duration: float) -> list[str]:
    captions = config.captions
    events: list[str] = []
    line_words = [word.text for word in chunk.words]
    wrapped = _wrap(line_words, captions.max_chars_per_line)
    if len(wrapped) > captions.max_lines:
        # Very long chunk: keep the first lines only rather than overflowing.
        wrapped = wrapped[: captions.max_lines]

    # Rebuild per-line word ownership so highlighting follows the wrapped text.
    remaining = list(chunk.words)
    lines_with_words: list[list[Word]] = []
    for line in wrapped:
        line_count = len(line.split())
        lines_with_words.append(remaining[:line_count])
        remaining = remaining[line_count:]

    for index, word in enumerate(chunk.words):
        start = max(chunk.start, word.start)
        if index + 1 < len(chunk.words):
            end = max(start + 0.05, chunk.words[index + 1].start)
        else:
            end = max(start + 0.15, chunk.end + 0.12)
        end = min(end, total_duration - 0.02)
        if end <= start:
            continue

        rendered_lines: list[str] = []
        for line_words, line in zip(lines_with_words, wrapped):
            pieces: list[str] = []
            for line_word in line_words:
                if line_word is word:
                    pieces.append(
                        f"{{\\c{captions.highlight_color}&}}{line_word.text}"
                        f"{{\\c{captions.primary_color}&}}"
                    )
                else:
                    pieces.append(line_word.text)
            if pieces:
                rendered_lines.append(" ".join(pieces))

        text = "\\N".join(rendered_lines) or chunk.text
        prefix = "{\\fad(90,60)}" if index == 0 else ""

        events.append(
            "Dialogue: 0,{start},{end},Caption,,0,0,0,,{text}".format(
                start=_ass_time(start),
                end=_ass_time(end),
                text=f"{prefix}{text}",
            )
        )
    return events


def render_ass(
    chunks: Sequence[CaptionChunk],
    *,
    config: AppConfig,
    output_path: Path,
    total_duration: float,
) -> Path:
    """Write the .ass subtitle file burned into the final render."""

    if not chunks:
        raise CaptionsError("No caption chunks to render")

    captions = config.captions
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {config.video.width}",
        f"PlayResY: {config.video.height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        _style_line(config),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    events: list[str] = []
    for chunk in chunks:
        events.extend(_chunk_events(chunk, config, total_duration))

    if not events:
        raise CaptionsError("Caption rendering produced no subtitle events")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join([*header, *events]) + "\n", encoding="utf-8")
    logger.info("Wrote %d caption event(s) to %s", len(events), output_path.name)
    return output_path


def generate_captions(
    audio_path: Path,
    *,
    config: AppConfig,
    narration: str,
    work_dir: Path,
    total_duration: float,
) -> tuple[Path, list[CaptionChunk]]:
    """Transcribe, chunk, and write captions in one call."""

    words = transcribe_words(audio_path, config=config, fallback_text=narration)
    words = sorted(words, key=lambda word: word.start)
    chunks = build_chunks(words, config)
    if not chunks:
        raise CaptionsError("No captions could be built from the transcription")
    path = render_ass(
        chunks,
        config=config,
        output_path=work_dir / "captions.ass",
        total_duration=total_duration,
    )
    return path, chunks


def chunk_texts(chunks: Iterable[CaptionChunk]) -> list[str]:
    """Small helper used by metadata/debug output."""

    return [chunk.text for chunk in chunks]
