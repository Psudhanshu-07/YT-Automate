#!/usr/bin/env python3
"""shorts-agent entry point.

Produces one complete YouTube Short: topic -> script -> voiceover -> captions ->
stock footage -> render -> optional upload. State (the topic/script history) is
persisted to ``history.json`` so the scheduled workflow can commit it back and
never repeat a topic.

Usage:
    python main.py                    # generate and upload (privacy from config.yaml)
    python main.py --dry-run          # generate only, skip the upload
    python main.py --privacy public   # one-off override, full auto

Exit codes: 0 success, 1 pipeline failure, 2 upload failure (artefacts kept),
3 configuration/environment problem, 130 interrupted.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from src.captions import generate_captions
from src.config import AppConfig, load_config
from src.env import load_dotenv, optional_env, require_env
from src.errors import EnvironmentError_, ShortsAgentError, UploadError
from src.ffmpeg_tools import ffmpeg_available
from src.ideas import (
    TopicIdea,
    generate_topic,
    load_history,
    record_script,
    record_topic,
    record_visuals,
    save_history,
    visual_source_ids,
)
from src.llm import GeminiClient
from src.logging_utils import get_logger, log_stage, setup_logging
from src.music import select_music
from src.script import Script, generate_script
from src.upload import UploadResult, upload_video
from src.video import RenderedVideo, build_video
from src.visuals import Clip, fetch_clips
from src.voice import VoiceResult, synthesise_voiceover

logger = get_logger("main")

PRIVACY_CHOICES = ("private", "unlisted", "public")
LANGUAGE_CHOICES = ("en", "hi")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="shorts-agent",
        description="Generate (and optionally publish) one YouTube Short per run.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml at the repo root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate the video but skip the YouTube upload. history.json is not modified.",
    )
    parser.add_argument(
        "--privacy",
        choices=PRIVACY_CHOICES,
        default=None,
        help="Override upload.privacy_status from config.yaml for this run.",
    )
    parser.add_argument(
        "--language",
        choices=LANGUAGE_CHOICES,
        default=None,
        help="Override channel.language for this run (en or hi).",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default=None,
        help="Override the logging level from config.yaml.",
    )
    parser.add_argument(
        "--keep-build",
        action="store_true",
        help="Keep the scratch directory (intermediate audio, filter graph, clips).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _warn_if_public(config: AppConfig) -> None:
    if config.upload.privacy_status == "public":
        logger.warning(
            "privacy_status=public: this run will publish the Short publicly as soon as the upload completes."
        )


def _build_metadata(
    *,
    run_id: str,
    config: AppConfig,
    idea: TopicIdea,
    script: Script,
    voice: VoiceResult,
    render: RenderedVideo,
    clips: Sequence[Clip],
    music_path: Path | None,
    upload: UploadResult | None,
    dry_run: bool,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "niche": config.niche,
        "topic": {
            "topic": idea.topic,
            "angle": idea.angle,
            "why_it_matters": idea.why_it_matters,
            "concrete_detail": idea.concrete_detail,
        },
        "script": script.to_dict(),
        "voice": {
            "voice": config.voice.voice,
            "rate": config.voice.rate,
            "duration_seconds": round(voice.duration, 2),
            "raw_duration_seconds": round(voice.raw_duration, 2),
            "tempo": round(voice.tempo, 3),
        },
        "captions": {"file": "captions.ass", "style": config.captions.font},
        "clips": [
            {
                "keyword": clip.keyword,
                "source": clip.source,
                "source_id": clip.source_id,
                "width": clip.width,
                "height": clip.height,
                "duration_seconds": round(clip.duration, 2),
            }
            for clip in clips
        ],
        "music": music_path.name if music_path else None,
        "video": {
            "file": render.path.name,
            "preview": render.preview.name,
            "duration_seconds": round(render.duration, 2),
            "width": config.video.width,
            "height": config.video.height,
            "fps": config.video.fps,
            "segments": len(render.segments),
        },
        "upload": {
            "status": "dry_run" if dry_run else ("uploaded" if upload else "failed"),
            "video_id": upload.video_id if upload else None,
            "url": upload.url if upload else None,
            "privacy_status": upload.privacy_status if upload else None,
        },
        "dry_run": dry_run,
        "elapsed_seconds": round(elapsed, 1),
    }


def _write_step_summary(*, metadata: dict[str, Any], render: RenderedVideo) -> None:
    """Append a short Markdown summary to the GitHub Actions run page."""

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
    if not summary_path:
        return

    script = metadata["script"]
    upload = metadata["upload"]
    lines = [
        "## YouTube Short produced",
        "",
        f"**Topic:** {metadata['topic']['topic']}",
        "",
        f"**Title:** {script['title']}",
        "",
        f"**Angle:** {metadata['topic']['angle']}",
        "",
        f"**Video:** `{render.path.name}` - {metadata['video']['duration_seconds']}s, "
        f"{metadata['video']['width']}x{metadata['video']['height']} @ {metadata['video']['fps']}fps",
        "",
        f"**Upload:** {upload['status']}"
        + (f" - {upload['url']}" if upload.get("url") else ""),
        "",
        f"**Privacy:** {upload.get('privacy_status') or 'n/a'}",
        "",
        f"**Keywords:** {', '.join(script['visual_keywords'])}",
        "",
    ]
    try:
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    except OSError as exc:  # pragma: no cover - CI only
        logger.debug("Could not write the step summary: %s", exc)


def _prepare_directories(config: AppConfig) -> tuple[str, Path, Path]:
    run_id = _run_id()
    run_dir = config.output_dir / run_id
    build_dir = config.work_dir / run_id

    if build_dir.exists():
        shutil.rmtree(build_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir, build_dir


def _validate_environment(config: AppConfig, *, dry_run: bool) -> tuple[str, str]:
    """Fail fast on missing binaries/secrets before doing expensive work."""

    if not ffmpeg_available():
        raise EnvironmentError_(
            "FFmpeg is not installed or not on PATH.",
            hint=(
                "Ubuntu: sudo apt-get install -y ffmpeg. macOS: brew install ffmpeg. "
                "Windows: winget install Gyan.FFmpeg, then reopen the terminal."
            ),
        )

    gemini_key = require_env("GEMINI_API_KEY", purpose="Gemini topic and script generation")
    pexels_key = optional_env("PEXELS_API_KEY")

    if not dry_run:
        require_env("YT_CLIENT_ID", purpose="YouTube OAuth client ID")
        require_env("YT_CLIENT_SECRET", purpose="YouTube OAuth client secret")
        require_env("YT_REFRESH_TOKEN", purpose="YouTube OAuth refresh token")

    return gemini_key, pexels_key


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _run(args: argparse.Namespace, config: AppConfig) -> int:
    started = time.monotonic()
    run_id, run_dir, build_dir = _prepare_directories(config)

    logger.info("shorts-agent run %s", run_id)
    logger.info("Niche: %s", config.niche)
    logger.info("Output: %s", run_dir)
    logger.info("Privacy status: %s%s", config.upload.privacy_status, " (dry run)" if args.dry_run else "")
    _warn_if_public(config)

    gemini_key, pexels_key = _validate_environment(config, dry_run=args.dry_run)

    client = GeminiClient(gemini_key, config.llm)
    history = load_history(config.history_path, config.niche)

    with log_stage(logger, "1/8 Choose a fresh topic"):
        idea = generate_topic(config, client, history)

    with log_stage(logger, "2/8 Write and validate the script"):
        script = generate_script(config, client, idea, history)

    with log_stage(logger, "3/8 Synthesise the voiceover"):
        voice = synthesise_voiceover(script.voiceover_text, config, build_dir)

    with log_stage(logger, "4/8 Transcribe and burn captions"):
        captions_path, chunks = generate_captions(
            voice.path,
            config=config,
            narration=script.voiceover_text,
            work_dir=build_dir,
            total_duration=voice.duration,
        )

    with log_stage(logger, "5/8 Fetch stock footage"):
        clips = fetch_clips(
            config,
            script.visual_keywords,
            total_duration=voice.duration,
            work_dir=build_dir,
            api_key=pexels_key,
            excluded_source_ids=visual_source_ids(history, niche=config.niche),
            variant=run_id,
        )

    with log_stage(logger, "6/8 Select background music"):
        music_path = select_music(config)

    with log_stage(logger, "7/8 Render the Short"):
        render = build_video(
            config=config,
            clips=clips,
            voice=voice,
            captions_path=captions_path,
            music_path=music_path,
            output=run_dir / "short.mp4",
            work_dir=build_dir,
        )

    shutil.copy2(captions_path, run_dir / "captions.ass")

    upload_result: UploadResult | None = None
    upload_failure: UploadError | None = None

    if args.dry_run:
        logger.info("8/8 Upload skipped (--dry-run)")
    else:
        try:
            with log_stage(logger, "8/8 Upload to YouTube"):
                upload_result = upload_video(
                    config,
                    script,
                    render.path,
                    privacy_status=config.upload.privacy_status,
                )
        except UploadError as exc:
            upload_failure = exc
            logger.error("Upload failed: %s", exc)
            if exc.hint:
                logger.error("Hint: %s", exc.hint)

    elapsed = time.monotonic() - started
    metadata = _build_metadata(
        run_id=run_id,
        config=config,
        idea=idea,
        script=script,
        voice=voice,
        render=render,
        clips=clips,
        music_path=music_path,
        upload=upload_result,
        dry_run=args.dry_run,
        elapsed=elapsed,
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_step_summary(metadata=metadata, render=render)

    # Persist history unless this was a dry run. Upload failures still consume the
    # topic: the video exists and can be uploaded manually from the artifact.
    if args.dry_run:
        logger.info("Dry run: history.json left untouched")
    else:
        status = "uploaded" if upload_result else "upload_failed"
        record_topic(
            history,
            idea,
            niche=config.niche,
            video_id=upload_result.video_id if upload_result else None,
            privacy_status=upload_result.privacy_status if upload_result else config.upload.privacy_status,
            status=status,
        )
        record_script(
            history,
            niche=config.niche,
            topic=idea.topic,
            text=script.voiceover_text,
            keep=config.quality_guard.recent_scripts_window * 2,
        )
        record_visuals(
            history,
            niche=config.niche,
            source_ids=(clip.source_id for clip in clips if clip.source != "synthetic"),
        )
        save_history(config.history_path, history)

    if not args.keep_build:
        shutil.rmtree(build_dir, ignore_errors=True)
    else:
        logger.info("Scratch directory kept at %s", build_dir)

    logger.info("=" * 62)
    logger.info("Finished in %.1fs", elapsed)
    logger.info("Video:   %s", render.path)
    logger.info("Preview: %s", render.preview)
    logger.info("Metadata: %s", run_dir / "metadata.json")
    if upload_result:
        logger.info("YouTube: %s (%s)", upload_result.url, upload_result.privacy_status)
    elif args.dry_run:
        logger.info("YouTube: skipped (dry run) - upload the file from %s", run_dir)
    else:
        logger.warning("YouTube: upload failed - the video is kept at %s", render.path)

    if upload_failure is not None:
        raise upload_failure

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    load_dotenv()

    try:
        config = load_config(args.config)
    except ShortsAgentError as exc:
        setup_logging(args.log_level or "INFO")
        logger.error("%s", exc)
        if exc.hint:
            logger.error("Hint: %s", exc.hint)
        return exc.exit_code

    setup_logging(args.log_level or config.logging.level)

    if args.privacy:
        config = dataclasses.replace(
            config,
            upload=dataclasses.replace(config.upload, privacy_status=args.privacy),
        )
    if args.language:
        voice_by_language = {
            "en": "en-IN-NeerjaNeural",
            "hi": "hi-IN-SwaraNeural",
        }
        config = dataclasses.replace(
            config,
            channel=dataclasses.replace(config.channel, language=args.language),
            upload=dataclasses.replace(config.upload, default_language=args.language),
            voice=dataclasses.replace(config.voice, voice=voice_by_language[args.language]),
        )

    try:
        return _run(args, config)
    except ShortsAgentError as exc:
        logger.error("%s", exc)
        if exc.hint:
            logger.error("Hint: %s", exc.hint)
        return exc.exit_code
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception:  # noqa: BLE001 - last-resort guard, log the traceback
        logger.exception("Unexpected failure - please report this with the log above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
