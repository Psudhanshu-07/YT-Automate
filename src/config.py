"""Typed configuration for the Shorts agent.

Every tunable lives in ``config.yaml`` at the repository root. This module turns
that file into immutable dataclasses, rejects typos and out-of-range values with
a :class:`~src.errors.ConfigError`, and resolves relative paths against the repo
root so the pipeline behaves the same on a laptop and on a GitHub runner.
"""

from __future__ import annotations

import dataclasses
import re
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

_HEX_COLOR = re.compile(r"^&H[0-9A-Fa-f]{8}$")
_PRIVACY_VALUES = ("private", "unlisted", "public")
_WHISPER_MODELS = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v3",
    "large-v3-turbo",
)
_LICENSE_VALUES = ("youtube", "creativeCommon")
_LANGUAGE_VALUES = ("en", "hi")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelConfig:
    name: str = "Shorts Agent"
    language: str = "en"


@dataclass(frozen=True)
class LlmConfig:
    model: str = "gemini-2.5-flash"
    fallback_models: list[str] = field(default_factory=list)
    temperature: float = 0.95
    max_output_tokens: int = 2048
    max_retries: int = 3
    retry_base_seconds: float = 4.0


@dataclass(frozen=True)
class IdeasConfig:
    history_file: str = "history.json"
    max_attempts: int = 4
    max_topic_similarity: float = 0.75
    history_prompt_limit: int = 60


@dataclass(frozen=True)
class ScriptConfig:
    min_words: int = 78
    max_words: int = 120
    min_keywords: int = 5
    max_keywords: int = 8
    title_max_chars: int = 70
    tag_count: int = 8
    max_attempts: int = 3


@dataclass(frozen=True)
class QualityGuardConfig:
    max_similarity_to_recent_scripts: float = 0.82
    recent_scripts_window: int = 15
    require_number_or_tool: bool = True
    banned_phrases: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VoiceConfig:
    voice: str = "en-US-AndrewNeural"
    rate: str = "+6%"
    pitch: str = "+0Hz"
    volume: str = "+0%"
    max_audio_seconds: float = 52.0
    tail_padding_seconds: float = 0.45
    loudness_target_lufs: float = -14.0
    max_retries: int = 3


@dataclass(frozen=True)
class CaptionsConfig:
    whisper_model: str = "base"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 5
    font: str = "Montserrat"
    fonts_dir: str = ""
    font_size: int = 92
    bold: bool = True
    primary_color: str = "&H00FFFFFF"
    highlight_color: str = "&H0000E5FF"
    outline_color: str = "&H00000000"
    outline: int = 5
    shadow: int = 2
    margin_v: int = 520
    uppercase: bool = True
    max_chars_per_line: int = 20
    max_lines: int = 2
    max_chunk_seconds: float = 3.2
    min_chunk_chars: int = 8


@dataclass(frozen=True)
class VisualsConfig:
    per_keyword_results: int = 15
    max_download_mb: int = 40
    min_clip_seconds: float = 3.0
    min_clip_width: int = 720
    request_timeout_seconds: int = 30
    max_retries: int = 3


@dataclass(frozen=True)
class MusicConfig:
    enabled: bool = True
    track: str = ""
    volume: float = 0.07
    duck: bool = True
    fade_in_seconds: float = 1.5
    fade_out_seconds: float = 2.0


@dataclass(frozen=True)
class VideoConfig:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    max_duration_seconds: float = 57.0
    clip_max_seconds: float = 5.5
    fade_seconds: float = 0.3
    crf: int = 21
    preset: str = "veryfast"
    audio_bitrate: str = "192k"
    preview_position: float = 0.35
    encode_timeout_seconds: int = 1800
    background_color: str = "0x09111F"
    accent_color: str = "0x00E5FF"
    card_width_ratio: float = 0.88
    card_height_ratio: float = 0.62
    card_top_ratio: float = 0.07


@dataclass(frozen=True)
class UploadConfig:
    privacy_status: str = "private"
    category_id: str = "28"
    default_language: str = "en"
    made_for_kids: bool = False
    embeddable: bool = True
    license: str = "youtube"
    max_retries: int = 6
    retry_base_seconds: float = 5.0
    chunk_size_mb: int = 8


@dataclass(frozen=True)
class PathsConfig:
    output_dir: str = "output"
    work_dir: str = "build"
    music_dir: str = "assets/music"


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    niche: str
    source_path: Path
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    ideas: IdeasConfig = field(default_factory=IdeasConfig)
    script: ScriptConfig = field(default_factory=ScriptConfig)
    quality_guard: QualityGuardConfig = field(default_factory=QualityGuardConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    captions: CaptionsConfig = field(default_factory=CaptionsConfig)
    visuals: VisualsConfig = field(default_factory=VisualsConfig)
    music: MusicConfig = field(default_factory=MusicConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def history_path(self) -> Path:
        return _resolve(self.ideas.history_file)

    @property
    def output_dir(self) -> Path:
        return _resolve(self.paths.output_dir)

    @property
    def work_dir(self) -> Path:
        return _resolve(self.paths.work_dir)

    @property
    def music_dir(self) -> Path:
        return _resolve(self.paths.music_dir)


# ---------------------------------------------------------------------------
# Loading / coercion
# ---------------------------------------------------------------------------


def _resolve(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _coerce(value: Any, expected: Any, path: str) -> Any:
    """Validate *value* against the annotated type *expected*."""

    origin = typing.get_origin(expected)
    if origin in (list, tuple):
        args = typing.get_args(expected) or (str,)
        item_type = args[0]
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path} must be a list, got {type(value).__name__}")
        return [_coerce(item, item_type, f"{path}[{index}]") for index, item in enumerate(value)]

    if expected is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path} must be true or false, got {value!r}")
        return value

    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path} must be an integer, got {value!r}")
        return value

    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path} must be a number, got {value!r}")
        return float(value)

    if expected is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path} must be a string, got {value!r}")
        return value

    raise ConfigError(f"{path} has an unsupported configuration type ({expected!r})")


def _build_section(cls: type, raw: Any, path: str) -> Any:
    """Instantiate a config section from a YAML mapping, rejecting unknown keys."""

    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path} must be a mapping of keys to values, got {type(raw).__name__}")

    hints = typing.get_type_hints(cls)
    fields = {item.name: item for item in dataclasses.fields(cls)}

    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ConfigError(
            f"unknown key(s) in {path}: {', '.join(unknown)}",
            hint=f"Valid keys for {path}: {', '.join(sorted(fields))}",
        )

    kwargs: dict[str, Any] = {}
    for name, item in fields.items():
        value = raw.get(name, item.default)
        if value is dataclasses.MISSING:
            raise ConfigError(f"{path}.{name} is required but missing")
        kwargs[name] = _coerce(value, hints[name], f"{path}.{name}")
    return cls(**kwargs)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Read, validate and return the application configuration."""

    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.is_file():
        raise ConfigError(
            f"Configuration file not found: {config_path}",
            hint="Run the pipeline from the repository root or pass --config PATH.",
        )

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")

    section_types: dict[str, type] = {
        "channel": ChannelConfig,
        "llm": LlmConfig,
        "ideas": IdeasConfig,
        "script": ScriptConfig,
        "quality_guard": QualityGuardConfig,
        "voice": VoiceConfig,
        "captions": CaptionsConfig,
        "visuals": VisualsConfig,
        "music": MusicConfig,
        "video": VideoConfig,
        "upload": UploadConfig,
        "paths": PathsConfig,
        "logging": LoggingConfig,
    }

    unknown = sorted(set(raw) - {"niche", *section_types})
    if unknown:
        raise ConfigError(
            f"unknown top-level key(s) in {config_path.name}: {', '.join(unknown)}",
            hint=f"Valid keys: niche, {', '.join(section_types)}",
        )

    niche = raw.get("niche")
    if not isinstance(niche, str) or not niche.strip():
        raise ConfigError("niche must be a non-empty string", hint='Example: niche: "tech and AI tips for students"')

    sections = {name: _build_section(cls, raw.get(name), name) for name, cls in section_types.items()}

    config = AppConfig(niche=niche.strip(), source_path=config_path, **sections)
    _validate(config)
    return config


# ---------------------------------------------------------------------------
# Cross-field validation
# ---------------------------------------------------------------------------


def _validate(config: AppConfig) -> None:
    problems: list[str] = []

    if len(config.niche) > 300:
        problems.append("niche must be 300 characters or fewer")
    if config.channel.language not in _LANGUAGE_VALUES:
        problems.append(f"channel.language must be one of: {', '.join(_LANGUAGE_VALUES)}")
    if config.upload.default_language not in _LANGUAGE_VALUES:
        problems.append(
            f"upload.default_language must be one of: {', '.join(_LANGUAGE_VALUES)}"
        )

    video = config.video
    if video.width % 2 or video.height % 2:
        problems.append("video.width and video.height must be even numbers for yuv420p")
    if video.width < 240 or video.height < 240:
        problems.append("video.width/video.height must be at least 240")
    if video.height <= video.width:
        problems.append("video.height must be greater than video.width for a Shorts (9:16) render")
    if not 24 <= video.fps <= 60:
        problems.append("video.fps must be between 24 and 60")
    if not 5 < video.max_duration_seconds < 60:
        problems.append("video.max_duration_seconds must be greater than 5 and less than 60")
    if video.clip_max_seconds < 1.5:
        problems.append("video.clip_max_seconds must be at least 1.5")
    if video.fade_seconds < 0 or video.fade_seconds * 2 >= video.clip_max_seconds:
        problems.append("video.fade_seconds must be non-negative and less than half of clip_max_seconds")
    if not 16 <= video.crf <= 32:
        problems.append("video.crf must be between 16 and 32")
    if video.preset not in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"}:
        problems.append("video.preset must be a valid libx264 preset")
    if not 0.0 <= video.preview_position <= 0.95:
        problems.append("video.preview_position must be between 0.0 and 0.95")
    if video.encode_timeout_seconds < 60:
        problems.append("video.encode_timeout_seconds must be at least 60")
    for name in ("background_color", "accent_color"):
        if not re.fullmatch(r"0x[0-9A-Fa-f]{6}", getattr(video, name)):
            problems.append(f"video.{name} must look like 0xRRGGBB")
    if not 0.5 <= video.card_width_ratio <= 1.0:
        problems.append("video.card_width_ratio must be between 0.5 and 1.0")
    if not 0.4 <= video.card_height_ratio <= 0.85:
        problems.append("video.card_height_ratio must be between 0.4 and 0.85")
    if not 0.0 <= video.card_top_ratio <= 0.3:
        problems.append("video.card_top_ratio must be between 0.0 and 0.3")

    script = config.script
    if script.min_words < 30:
        problems.append("script.min_words must be at least 30")
    if script.max_words <= script.min_words:
        problems.append("script.max_words must be greater than script.min_words")
    if not 3 <= script.min_keywords <= script.max_keywords <= 10:
        problems.append("script keywords must satisfy 3 <= min_keywords <= max_keywords <= 10")
    if not 30 <= script.title_max_chars <= 100:
        problems.append("script.title_max_chars must be between 30 and 100")
    if not 1 <= script.tag_count <= 30:
        problems.append("script.tag_count must be between 1 and 30")
    if script.max_attempts < 1:
        problems.append("script.max_attempts must be at least 1")

    voice = config.voice
    if not 0 < voice.max_audio_seconds <= video.max_duration_seconds:
        problems.append("voice.max_audio_seconds must be > 0 and <= video.max_duration_seconds")
    if not 0 <= voice.tail_padding_seconds <= 2:
        problems.append("voice.tail_padding_seconds must be between 0 and 2")
    if not -30 <= voice.loudness_target_lufs <= -5:
        problems.append("voice.loudness_target_lufs must be between -30 and -5")
    if voice.max_retries < 1:
        problems.append("voice.max_retries must be at least 1")

    captions = config.captions
    if captions.whisper_model not in _WHISPER_MODELS:
        problems.append(f"captions.whisper_model must be one of: {', '.join(_WHISPER_MODELS)}")
    if captions.whisper_compute_type not in {"int8", "int8_float32", "float32", "float16"}:
        problems.append("captions.whisper_compute_type must be int8, int8_float32, float32 or float16")
    if not 20 <= captions.font_size <= 200:
        problems.append("captions.font_size must be between 20 and 200")
    for key in ("primary_color", "highlight_color", "outline_color"):
        if not _HEX_COLOR.match(getattr(captions, key)):
            problems.append(f"captions.{key} must look like &HAABBGGRR (for example &H00FFFFFF)")
    if captions.max_lines < 1:
        problems.append("captions.max_lines must be at least 1")
    if captions.max_chars_per_line < 8:
        problems.append("captions.max_chars_per_line must be at least 8")
    if captions.margin_v < 0:
        problems.append("captions.margin_v must be non-negative")

    visuals = config.visuals
    if visuals.per_keyword_results < 1:
        problems.append("visuals.per_keyword_results must be at least 1")
    if visuals.max_download_mb < 5:
        problems.append("visuals.max_download_mb must be at least 5")
    if visuals.min_clip_seconds < 1:
        problems.append("visuals.min_clip_seconds must be at least 1")

    music = config.music
    if not 0.0 <= music.volume <= 1.0:
        problems.append("music.volume must be between 0.0 and 1.0")
    if music.fade_in_seconds < 0 or music.fade_out_seconds < 0:
        problems.append("music fade values must be non-negative")

    upload = config.upload
    if upload.privacy_status not in _PRIVACY_VALUES:
        problems.append(f"upload.privacy_status must be one of: {', '.join(_PRIVACY_VALUES)}")
    if not upload.category_id.isdigit() or upload.category_id == "0":
        problems.append("upload.category_id must be a YouTube category ID such as 28")
    if upload.license not in _LICENSE_VALUES:
        problems.append(f"upload.license must be one of: {', '.join(_LICENSE_VALUES)}")
    if upload.max_retries < 1:
        problems.append("upload.max_retries must be at least 1")
    if not 1 <= upload.chunk_size_mb <= 64:
        problems.append("upload.chunk_size_mb must be between 1 and 64")

    ideas = config.ideas
    if not 0.0 <= ideas.max_topic_similarity <= 1.0:
        problems.append("ideas.max_topic_similarity must be between 0.0 and 1.0")
    if ideas.max_attempts < 1:
        problems.append("ideas.max_attempts must be at least 1")

    guard = config.quality_guard
    if not 0.0 <= guard.max_similarity_to_recent_scripts <= 1.0:
        problems.append("quality_guard.max_similarity_to_recent_scripts must be between 0.0 and 1.0")

    if config.llm.max_retries < 1:
        problems.append("llm.max_retries must be at least 1")
    if not 0.0 <= config.llm.temperature <= 2.0:
        problems.append("llm.temperature must be between 0.0 and 2.0")

    if problems:
        raise ConfigError(
            "Invalid configuration:\n  - " + "\n  - ".join(problems),
            hint=f"Fix these values in {config.source_path}",
        )
