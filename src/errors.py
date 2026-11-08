"""Exception hierarchy for the pipeline.

Every failure the pipeline can anticipate is raised as a subclass of
:class:`ShortsAgentError` so ``main.py`` can log a readable message instead of a
traceback and pick a meaningful process exit code.

Exit codes
----------
1  generic pipeline failure (ideas, script, voice, captions, visuals, render)
2  video produced but the YouTube upload failed (artifact still preserved)
3  configuration / environment problem (bad config.yaml, missing secret, no ffmpeg)
130 interrupted by the user (Ctrl+C)
"""

from __future__ import annotations


class ShortsAgentError(Exception):
    """Base class for all anticipated pipeline failures."""

    exit_code: int = 1

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.hint:
            return f"{self.message}\nHint: {self.hint}"
        return self.message


class ConfigError(ShortsAgentError):
    """config.yaml is missing, malformed, or contains an invalid value."""

    exit_code = 3


class EnvironmentError_(ShortsAgentError):
    """A required binary or environment variable is missing."""

    exit_code = 3


class FFmpegError(ShortsAgentError):
    """An ffmpeg/ffprobe command failed or timed out."""


class LLMError(ShortsAgentError):
    """Gemini could not be reached or returned nothing usable."""


class HistoryError(ShortsAgentError):
    """history.json is corrupt or the topic space is exhausted."""


class ScriptValidationError(ShortsAgentError):
    """A generated script failed schema validation or the content guard."""


class VoiceError(ShortsAgentError):
    """Text-to-speech synthesis or audio post-processing failed."""


class CaptionsError(ShortsAgentError):
    """Transcription or subtitle rendering failed."""


class VisualsError(ShortsAgentError):
    """No usable stock footage could be obtained."""


class MusicError(ShortsAgentError):
    """The configured music track could not be prepared."""


class VideoError(ShortsAgentError):
    """FFmpeg rendering failed or produced an invalid video."""


class UploadError(ShortsAgentError):
    """The YouTube upload failed after all retries."""

    exit_code = 2
