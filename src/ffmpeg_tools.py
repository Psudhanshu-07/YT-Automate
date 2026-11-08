"""Small, defensive wrappers around the ffmpeg/ffprobe binaries.

Everything in the pipeline shells out to FFmpeg rather than pulling in MoviePy,
which keeps the dependency surface small and the render deterministic. This
module finds the binaries, runs commands with timeouts, probes durations, and
can synthesise a gradient clip as a last-resort visual.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

from .errors import EnvironmentError_, FFmpegError
from .logging_utils import get_logger

logger = get_logger("ffmpeg")

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def _candidate(name: str, env_var: str) -> str | None:
    override = os.environ.get(env_var, "").strip()
    if override:
        return override
    found = shutil.which(name)
    if found:
        return found
    if name == "ffmpeg":
        try:  # Optional convenience for local Windows runs.
            import imageio_ffmpeg  # type: ignore

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001 - optional dependency
            return None
    return None


def ffmpeg_binary() -> str:
    binary = _candidate("ffmpeg", "FFMPEG_BINARY")
    if not binary:
        raise EnvironmentError_(
            "FFmpeg was not found on PATH.",
            hint=(
                "Install it (Ubuntu: sudo apt-get install -y ffmpeg; macOS: brew install ffmpeg; "
                "Windows: winget install Gyan.FFmpeg) or set FFMPEG_BINARY."
            ),
        )
    return binary


def ffprobe_binary() -> str | None:
    return _candidate("ffprobe", "FFPROBE_BINARY")


def ffmpeg_available() -> bool:
    return _candidate("ffmpeg", "FFMPEG_BINARY") is not None


def run(
    command: Sequence[str],
    *,
    timeout: float,
    description: str,
    cwd: Path | str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command, raising :class:`FFmpegError` with stderr on failure."""

    logger.debug("%s: %s", description, " ".join(command))
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(
            f"{description} timed out after {timeout:.0f}s",
            hint="Increase the relevant timeout in config.yaml or reduce the video length.",
        ) from exc
    except FileNotFoundError as exc:
        raise EnvironmentError_(
            f"{description} could not start because a required binary is missing: {command[0]}",
            hint="Install FFmpeg and make sure it is on PATH.",
        ) from exc

    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or "").strip().splitlines()[-25:])
        raise FFmpegError(f"{description} failed with exit code {completed.returncode}:\n{tail}")
    return completed


def probe_duration(path: Path) -> float:
    """Return media duration in seconds, using ffprobe when available."""

    if not path.is_file() or path.stat().st_size == 0:
        raise FFmpegError(f"Cannot probe an empty or missing file: {path}")

    probe = ffprobe_binary()
    if probe:
        completed = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
            check=False,
        )
        if completed.returncode == 0:
            try:
                payload = json.loads(completed.stdout or "{}")
                duration = float(payload.get("format", {}).get("duration", 0.0))
            except (ValueError, TypeError, json.JSONDecodeError):
                duration = 0.0
            if duration > 0:
                return duration

    # Fallback: parse the banner ffmpeg prints for the first input.
    completed = subprocess.run(
        [ffmpeg_binary(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
        check=False,
    )
    match = _DURATION_RE.search(completed.stderr or "")
    if not match:
        raise FFmpegError(f"Could not determine the duration of {path.name}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def probe_media(path: Path) -> dict[str, Any]:
    """Return ffprobe's full JSON for *path* (empty dict if unavailable)."""

    probe = ffprobe_binary()
    if not probe:
        return {}
    completed = subprocess.run(
        [probe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        return {}
    try:
        payload = json.loads(completed.stdout or "{}")
    except (json.JSONDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def progress_run(
    command: Sequence[str],
    *,
    timeout: float,
    description: str,
    cwd: Path,
) -> None:
    """Run an FFmpeg encode, logging progress values after it exits.

    stderr goes to a file rather than a pipe: a long encode that emits warnings
    can otherwise fill the OS pipe buffer while we read stdout and deadlock.
    ``communicate`` is used so the timeout also covers a process that stops
    producing progress output.
    """

    logger.info("%s: starting", description)
    stderr_path = Path(cwd) / "ffmpeg-stderr.log"
    last_logged = -1

    with stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_handle:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
            text=True,
            errors="replace",
        )
        try:
            stdout, _ = process.communicate(timeout=timeout)
            for line in stdout.splitlines():
                line = line.strip()
                if not line.startswith("out_time_ms="):
                    continue
                value = line.split("=", 1)[1].strip()
                if not value.isdigit():
                    continue
                seconds = int(value) / 1_000_000
                bucket = int(seconds // 10) * 10
                if bucket > last_logged:
                    last_logged = bucket
                    logger.info("%s: %ds encoded", description, int(seconds))
            return_code = process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=30)
            raise FFmpegError(f"{description} timed out after {timeout:.0f}s")
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=30)

    if return_code != 0:
        try:
            stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        except OSError:  # pragma: no cover - defensive
            stderr = ""
        tail = "\n".join(stderr.strip().splitlines()[-30:])
        raise FFmpegError(
            f"{description} failed with exit code {return_code}:\n{tail}",
            hint=(
                "Check src/video.py's filter graph and that your FFmpeg build includes libx264, "
                "libass and aac."
            ),
        )


def extract_frame(video: Path, output: Path, *, at_seconds: float) -> Path:
    """Save a single JPEG frame, used for ``preview.jpg``."""

    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg_binary(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, at_seconds):.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output),
        ],
        timeout=120,
        description="Extract preview frame",
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise FFmpegError("Preview frame extraction produced no image")
    return output


def synthesise_gradient_clip(output: Path, *, duration: float, width: int, height: int) -> Path:
    """Create a calm gradient clip when no stock footage is available."""

    output.parent.mkdir(parents=True, exist_ok=True)
    base = [
        ffmpeg_binary(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    encode = [
        "-t",
        f"{duration:.3f}",
        "-r",
        "30",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output),
    ]

    gradient_source = (
        f"gradients=s={width}x{height}:r=30:c0=0x0B1020:c1=0x1B4B6B:c2=0x0F766E:n=2:"
        f"speed=0.06:d={max(1.0, duration):.2f}"
    )
    try:
        run(
            [*base, "-f", "lavfi", "-i", gradient_source, *encode],
            timeout=180,
            description="Synthesise gradient clip",
        )
    except FFmpegError as exc:
        logger.warning("Gradient synthesis failed (%s); using a solid colour", exc)
        run(
            [*base, "-f", "lavfi", "-i", f"color=c=0x101C2C:s={width}x{height}:r=30", *encode],
            timeout=180,
            description="Synthesise solid-colour clip",
        )

    return output


def concat_music(parts: Iterable[str]) -> str:
    """Join ffmpeg filter fragments with commas, skipping empty entries."""

    return ",".join(part for part in parts if part)
