"""Video assembly with FFmpeg (1080x1920, 30 fps, H.264/AAC).

The render is a single FFmpeg invocation with an inline filter graph (which is
also written to ``filter.txt`` in the scratch directory for inspection):

* every stock clip is scaled into a framed portrait card;
* the card is composited over a textured, branded background;
* clips are trimmed to the measured narration length and faded in/out;
* the burned-in ASS captions are overlaid on the concatenated base;
* the music bed is sidechain-compressed under the voice (optional);
* output is capped below the 60 s Shorts limit.

The graph is passed inline rather than through ``-filter_complex_script`` because
FFmpeg 8 removed that option; a 10-segment graph is far below every platform's
argument length limit.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import AppConfig
from .errors import FFmpegError, VideoError
from .ffmpeg_tools import extract_frame, ffmpeg_binary, probe_duration, probe_media, progress_run
from .logging_utils import get_logger
from .visuals import Clip
from .voice import VoiceResult

logger = get_logger("video")

HARD_MAX_SECONDS = 58.0


@dataclass(frozen=True)
class ClipSegment:
    clip: Clip
    duration: float
    index: int
    flip: bool


@dataclass(frozen=True)
class RenderedVideo:
    path: Path
    preview: Path
    duration: float
    segments: tuple[ClipSegment, ...]


def _escape_filter_path(path: str) -> str:
    """Escape a path for use inside an FFmpeg filter argument."""

    return path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def build_timeline(clips: Sequence[Clip], total_duration: float, config: AppConfig) -> list[ClipSegment]:
    """Split *total_duration* into equal segments cycled across the clips."""

    if not clips:
        raise VideoError("No clips available for the timeline")
    if total_duration <= 1.0:
        raise VideoError(f"Video duration {total_duration:.2f}s is too short to render")

    step_size = max(1.5, config.video.clip_max_seconds)
    count = max(1, math.ceil(total_duration / step_size))
    step = total_duration / count

    segments: list[ClipSegment] = []
    cursor = 0.0
    for index in range(count):
        duration = total_duration - cursor if index == count - 1 else step
        if duration <= 0.2:
            continue
        clip = clips[index % len(clips)]
        # Mirror reused clips so repetition is less obvious.
        flip = len(clips) > 0 and (index // len(clips)) % 2 == 1
        segments.append(ClipSegment(clip=clip, duration=duration, index=index, flip=flip))
        cursor += duration

    return segments


def _video_chain(
    segment: ClipSegment,
    config: AppConfig,
    *,
    frame_width: int,
    frame_height: int,
    first: bool = False,
    last: bool = False,
) -> str:
    video = config.video
    width, height = frame_width, frame_height
    fade = min(video.fade_seconds, segment.duration / 4)

    # Overscan the source and animate the crop window so even a static stock
    # shot has the subtle push/pan treatment expected in a creator edit.
    overscan_width = int(math.ceil(width * 1.12 / 2) * 2)
    overscan_height = int(math.ceil(height * 1.12 / 2) * 2)
    motion = 0.12 + (segment.index % 3) * 0.04
    direction = 1 if segment.index % 2 == 0 else -1
    x_motion = f"(iw-ow)/2+{direction * motion:.3f}*(iw-ow)*sin(n/180)"
    y_motion = f"(ih-oh)/2+{direction * motion * 0.7:.3f}*(ih-oh)*cos(n/220)"
    chain = [
        f"fps={video.fps}",
        f"scale={overscan_width}:{overscan_height}:force_original_aspect_ratio=increase",
    ]
    if segment.flip:
        chain.append("hflip")
    chain.extend(
        [
            f"crop={width}:{height}:{x_motion}:{y_motion}",
            "setsar=1",
            "format=yuv420p",
            "eq=contrast=1.06:saturation=1.12:brightness=0.01",
            "unsharp=5:5:0.35:5:5:0",
            "vignette=PI/5",
            f"drawbox=x=0:y=0:w=iw-1:h=ih-1:color={video.accent_color}@0.82:t=6",
        ]
    )
    if fade > 0 and first:
        chain.append(f"fade=t=in:st=0:d={fade:.3f}")
    if fade > 0 and last:
        chain.append(f"fade=t=out:st={max(0.0, segment.duration - fade):.3f}:d={fade:.3f}")
    chain.append("setpts=PTS-STARTPTS")
    return f"[{segment.index}:v]" + ",".join(chain) + f"[v{segment.index}]"


def _build_filters(
    segments: Sequence[ClipSegment],
    *,
    config: AppConfig,
    captions_relative: str,
    total_duration: float,
    has_music: bool,
    duck_music: bool,
    music_volume: float,
) -> list[str]:
    """Return filter-graph chains (joined with ';' by the caller)."""

    video = config.video
    card_width = int(video.width * video.card_width_ratio) // 2 * 2
    card_height = int(video.height * video.card_height_ratio) // 2 * 2
    card_x = (video.width - card_width) // 2
    card_y = int(video.height * video.card_top_ratio) // 2 * 2
    lines: list[str] = [
        _video_chain(
            segment,
            config,
            frame_width=card_width,
            frame_height=card_height,
            first=index == 0,
            last=index == len(segments) - 1,
        )
        for index, segment in enumerate(segments)
    ]

    concat_inputs = "".join(f"[v{segment.index}]" for segment in segments)
    lines.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=0[vbase]")
    lines.append(
        f"color=c={video.background_color}:s={video.width}x{video.height}:r={video.fps}:"
        f"d={total_duration:.3f},noise=alls=10:allf=t+u,"
        f"drawgrid=w=180:h=180:t=1:c={video.accent_color}@0.08,"
        "vignette=PI/3,"
        f"drawbox=x={card_x - 14}:y={card_y - 14}:w={card_width + 28}:h={card_height + 28}:"
        f"color={video.accent_color}@0.14:t=18[bg]"
    )
    lines.append(
        f"[bg][vbase]overlay=x={card_x}:y={card_y}:eof_action=pass:shortest=1,"
        "format=yuv420p[vframed]"
    )

    ass_options = _escape_filter_path(captions_relative)
    if config.captions.fonts_dir:
        ass_options += f":fontsdir={_escape_filter_path(config.captions.fonts_dir)}"
    lines.append(f"[vframed]ass={ass_options}[vout]")

    voice_index = len(segments)
    duration = f"{total_duration:.3f}"

    if not has_music:
        lines.append(
            f"[{voice_index}:a]aresample=48000,apad=whole_dur={duration},"
            f"atrim=0:{duration},alimiter=limit=0.95[aout]"
        )
        return lines

    music_index = voice_index + 1
    music = config.music
    fade_out_start = max(0.0, total_duration - music.fade_out_seconds)

    lines.append(f"[{voice_index}:a]aresample=48000,asplit=2[voc1][voc2]")
    lines.append(
        f"[{music_index}:a]aresample=48000,atrim=0:{duration},asetpts=PTS-STARTPTS,"
        f"volume={music_volume:.3f},"
        f"afade=t=in:st=0:d={music.fade_in_seconds:.2f},"
        f"afade=t=out:st={fade_out_start:.3f}:d={music.fade_out_seconds:.2f}[music]"
    )

    if duck_music:
        lines.append(
            "[music][voc2]sidechaincompress=threshold=0.030:ratio=12:attack=20:release=400[musicduck]"
        )
        mix_input = "[musicduck]"
    else:
        mix_input = "[music]"

    lines.append(
        f"[voc1]{mix_input}amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
        f"apad=whole_dur={duration},atrim=0:{duration},alimiter=limit=0.95[aout]"
    )
    return lines


def _render(
    *,
    config: AppConfig,
    segments: Sequence[ClipSegment],
    voice: VoiceResult,
    music_path: Path | None,
    output: Path,
    work_dir: Path,
    total_duration: float,
    duck_music: bool,
    music_volume: float,
) -> None:
    video = config.video
    filter_lines = _build_filters(
        segments,
        config=config,
        captions_relative="captions.ass",
        total_duration=total_duration,
        has_music=music_path is not None,
        duck_music=duck_music,
        music_volume=music_volume,
    )
    # Inline filter graphs separate chains with ';' (a filter *script* file would
    # allow newlines, but that option no longer exists in FFmpeg 8+).
    filters = ";".join(filter_lines)
    (work_dir / "filter.txt").write_text(";\n".join(filter_lines) + "\n", encoding="utf-8")

    command: list[str] = [
        ffmpeg_binary(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-progress",
        "pipe:1",
        "-nostats",
        # Keep the GitHub-hosted runner within its memory budget. The filter
        # graph has several 1080x1920 inputs and unrestricted worker threads can
        # be killed by the runner before FFmpeg writes the MP4 trailer.
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-filter_complex_threads",
        "1",
    ]
    for segment in segments:
        command.extend(
            [
                "-stream_loop",
                "-1",
                "-t",
                f"{segment.duration:.3f}",
                "-i",
                str(segment.clip.path),
            ]
        )
    command.extend(["-i", str(voice.path)])
    if music_path is not None:
        command.extend(["-stream_loop", "-1", "-i", str(music_path)])

    command.extend(
        [
            "-filter_complex",
            filters,
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-t",
            f"{total_duration:.3f}",
            "-r",
            str(video.fps),
            "-c:v",
            "libx264",
            "-preset",
            video.preset,
            "-crf",
            str(video.crf),
            "-profile:v",
            "high",
            "-level",
            "4.1",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(video.fps * 2),
            "-c:a",
            "aac",
            "-b:a",
            video.audio_bitrate,
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )

    progress_run(
        command,
        timeout=video.encode_timeout_seconds,
        description="Render Short",
        cwd=work_dir,
    )


def _validate_output(path: Path, config: AppConfig) -> float:
    if not path.is_file() or path.stat().st_size < 50_000:
        raise VideoError(f"Render produced no usable output at {path}")

    media = probe_media(path)
    duration = 0.0
    if media:
        try:
            duration = float(media.get("format", {}).get("duration", 0.0) or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        streams = media.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        if not video_stream:
            raise VideoError("Rendered file has no video stream")
        if str(video_stream.get("codec_name")) != "h264":
            raise VideoError(f"Expected H.264 video, got {video_stream.get('codec_name')!r}")
        if not audio_stream:
            raise VideoError("Rendered file has no audio stream")
        if str(audio_stream.get("codec_name")) != "aac":
            raise VideoError(f"Expected AAC audio, got {audio_stream.get('codec_name')!r}")

        width, height = int(video_stream.get("width") or 0), int(video_stream.get("height") or 0)
        if (width, height) != (config.video.width, config.video.height):
            raise VideoError(
                f"Rendered frame is {width}x{height}, expected {config.video.width}x{config.video.height}"
            )

    if duration <= 0:
        duration = probe_duration(path)

    if duration >= HARD_MAX_SECONDS:
        raise VideoError(
            f"Rendered video is {duration:.1f}s; Shorts must stay under {HARD_MAX_SECONDS:.0f}s"
        )

    logger.info("Render validated: %.2fs, %.1f MB", duration, path.stat().st_size / (1024 * 1024))
    return duration


def build_video(
    *,
    config: AppConfig,
    clips: Sequence[Clip],
    voice: VoiceResult,
    captions_path: Path,
    music_path: Path | None,
    output: Path,
    work_dir: Path,
) -> RenderedVideo:
    """Render the Short and return the validated artefact."""

    if not captions_path.is_file():
        raise VideoError(f"Captions file missing: {captions_path}")

    total_duration = voice.duration
    if total_duration >= HARD_MAX_SECONDS:
        raise VideoError(
            f"Narration is {total_duration:.1f}s; cannot fit a Short under {HARD_MAX_SECONDS:.0f}s"
        )

    segments = build_timeline(clips, total_duration, config)
    logger.info(
        "Timeline: %d segment(s) over %.2fs, %d clip(s) in rotation",
        len(segments),
        total_duration,
        len(clips),
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    # Keep the final media extension so FFmpeg can infer the MP4 muxer.
    temporary_output = output.with_name(f".{output.stem}.part{output.suffix}")
    temporary_output.unlink(missing_ok=True)

    try:
        _render(
            config=config,
            segments=segments,
            voice=voice,
            music_path=music_path,
            output=temporary_output,
            work_dir=work_dir,
            total_duration=total_duration,
            duck_music=bool(music_path) and config.music.duck,
            music_volume=config.music.volume,
        )
    except FFmpegError as exc:
        if music_path is None or not config.music.duck:
            raise
        logger.warning(
            "Render with music ducking failed (%s); retrying with a static low music volume",
            str(exc).splitlines()[0][:200],
        )
        _render(
            config=config,
            segments=segments,
            voice=voice,
            music_path=music_path,
            output=temporary_output,
            work_dir=work_dir,
            total_duration=total_duration,
            duck_music=False,
            music_volume=config.music.volume * 0.6,
        )

    # Validate the completed file before exposing it under the final name.
    duration = _validate_output(temporary_output, config)
    os.replace(temporary_output, output)

    preview = output.parent / "preview.jpg"
    extract_frame(output, preview, at_seconds=duration * config.video.preview_position)

    logger.info("Video ready: %s (%.2fs)", output, duration)
    return RenderedVideo(
        path=output,
        preview=preview,
        duration=duration,
        segments=tuple(segments),
    )
