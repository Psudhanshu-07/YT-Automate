"""Script writing, strict JSON validation, and the content-quality guard.

Gemini returns a single JSON object containing the hook, the narration, visual
search keywords, and the YouTube metadata. Everything is validated here so the
rest of the pipeline can trust its inputs.

The guard exists because YouTube penalises "mass-produced" content. It enforces:

1. a distinct angle per video (chosen by :mod:`src.ideas`, echoed in the script);
2. banned generic filler phrases;
3. at least one concrete detail (a number, a worked example, or a named tool);
4. low textual similarity against the last N scripts in ``history.json``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .config import AppConfig
from .errors import ScriptValidationError
from .ideas import TopicIdea, recent_scripts, similarity
from .llm import GeminiClient
from .logging_utils import get_logger

logger = get_logger("script")

SHORTS_SUFFIX = "#Shorts"

_CONCRETE_MARKERS = (
    "for example",
    "for instance",
    "e.g.",
    "like when",
    "step 1",
    "step one",
    "try this",
)
_TOOL_WORDS = (
    "chatgpt", "gemini", "claude", "copilot", "notebooklm", "perplexity", "notion",
    "obsidian", "anki", "quizlet", "excel", "sheets", "google docs", "word",
    "chrome", "firefox", "edge", "vscode", "vs code", "github", "colab", "python",
    "javascript", "zotero", "canva", "figma", "wolfram", "khan academy", "coursera",
    "youtube", "pdf", "ocr", "spreadsheet", "browser extension", "bookmark",
)
_STAGE_DIRECTION = re.compile(r"\[[^\]]{0,40}\]|\([^)]{0,20}(?:pause|music|laughs)[^)]{0,20}\)", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_HASHTAG = re.compile(r"#[A-Za-z0-9_]+")


@dataclass(frozen=True)
class Script:
    """A validated, publish-ready script."""

    hook: str
    narration: str
    visual_keywords: list[str]
    title: str
    description: str
    tags: list[str]
    angle: str
    quality_notes: list[str] = field(default_factory=list)

    @property
    def voiceover_text(self) -> str:
        return f"{self.hook} {self.narration}".strip()

    @property
    def word_count(self) -> int:
        return len(self.voiceover_text.split())

    def to_dict(self) -> dict[str, Any]:
        return {
            "hook": self.hook,
            "narration": self.narration,
            "visual_keywords": list(self.visual_keywords),
            "title": self.title,
            "description": self.description,
            "tags": list(self.tags),
            "angle": self.angle,
            "word_count": self.word_count,
            "quality_notes": list(self.quality_notes),
        }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


_SYSTEM = """You write scripts for a YouTube Shorts channel.
Niche: {niche}

You are not a generic content generator. Every script must be built around the
provided unique angle and must contain a specific, verifiable detail. Never reuse
a sentence pattern or opening from a past video.

House style:
- Spoken English, short sentences, second person ("you").
- The first sentence is the hook: a concrete claim or tension, no greeting,
  no "in this video", no "hey guys".
- No filler, no hype adjectives, no promises about income, health, or grades.
- No stage directions, no emoji, no markdown.
- Facts must be ordinary and safe: describe widely documented features of tools
  and study techniques. Do not invent statistics or cite studies.
- End with one action the viewer can take in the next minute.

Return STRICT JSON with exactly these keys:
{{
  "hook": "one spoken sentence, 4-18 words, pays off immediately",
  "narration": "{min_words}-{max_words} words of spoken narration that continues after the hook",
  "visual_keywords": [{keyword_examples}],
  "title": "<= {title_max} characters, ends with {suffix}",
  "description": "2-4 sentences plus 2-3 hashtags, no links",
  "tags": ["{tag_count} lowercase tags, no # symbol"]
}}"""

_PROMPT = """Write the script for this short.

Topic: {topic}
Unique angle (must be the spine of the script): {angle}
Why it matters: {why}
Specific detail to include verbatim or as an equivalent: {detail}

Structure:
1. Hook (already counted separately).
2. One sentence of context.
3. The core explanation, with the concrete detail in the middle.
4. One common mistake or counter-intuitive point.
5. One action the viewer can take today.

Visual keywords must be short, shootable stock-footage searches in English
(e.g. "student typing laptop", "server room lights"), not abstract concepts.
{feedback}"""

_KEYWORD_EXAMPLES = '"student typing on laptop", "library study desk", "code on screen"'


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _clean_text(value: str) -> str:
    text = _WHITESPACE.sub(" ", value).strip()
    text = _STAGE_DIRECTION.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def _words(text: str) -> list[str]:
    return [word for word in text.split() if word.strip()]


def _finalise_title(raw: str, *, fallback: str, max_chars: int) -> str:
    base = _clean_text(raw).strip(" \"'")
    base = re.sub(r"#shorts\s*$", "", base, flags=re.IGNORECASE).strip()
    base = base.rstrip("#.,;:- ").strip()
    if not base:
        base = _clean_text(fallback)
    if not base:
        base = "A quick tech tip for students"

    budget = max_chars - len(SHORTS_SUFFIX) - 1
    if len(base) > budget:
        cut = base[:budget].rsplit(" ", 1)[0].strip()
        base = cut or base[:budget].strip()
    title = f"{base} {SHORTS_SUFFIX}"
    if len(title) > max_chars:  # pathological single long word
        title = f"{base[: max(1, budget)].strip()} {SHORTS_SUFFIX}"[:max_chars].strip()
    return title


def _finalise_tags(raw: Any, *, config: AppConfig, idea: TopicIdea) -> list[str]:
    tags: list[str] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if not isinstance(item, str):
                continue
            tag = _clean_text(item).strip("#").lower()
            tag = re.sub(r"[^a-z0-9 +\-]", "", tag).strip()
            if 2 <= len(tag) <= 40 and tag not in tags:
                tags.append(tag)

    niche_tags = [
        config.niche.lower(),
        "students",
        "study tips",
        "tech tips",
        "ai tools",
        "productivity",
        "learn faster",
        "free tools",
    ]
    for tag in niche_tags:
        if len(tags) >= config.script.tag_count:
            break
        cleaned = re.sub(r"[^a-z0-9 +\-]", "", _clean_text(tag).lower()).strip()
        if cleaned and cleaned not in tags:
            tags.append(cleaned)

    for token in re.findall(r"[a-z0-9]{4,}", idea.topic.lower()):
        if len(tags) >= config.script.tag_count:
            break
        if token not in tags:
            tags.append(token)

    return tags[: config.script.tag_count]


def _finalise_description(raw: str, *, tags: list[str], idea: TopicIdea) -> str:
    description = _clean_text(raw)
    if len(description) < 40:
        description = _clean_text(
            f"{idea.why_it_matters} A practical, no-hype tip for students in under a minute."
        )
    existing = {match.lower() for match in _HASHTAG.findall(description)}
    required = ["#Shorts", "#AITips", "#StudyTips"]
    hashtags = [tag for tag in required if tag.lower() not in existing]
    hashtags.extend(
        f"#{tag.replace(' ', '')}"
        for tag in tags[:3]
        if f"#{tag.replace(' ', '')}".lower() not in existing
    )
    if hashtags:
        description = f"{description}\n\n{' '.join(hashtags[:5])}".strip()
    return description[:4900]


def _hook_word_range(config: AppConfig) -> tuple[int, int]:
    return 4, 20


# ---------------------------------------------------------------------------
# Content-quality guard
# ---------------------------------------------------------------------------


def _has_concrete_detail(text: str) -> bool:
    lowered = text.lower()
    if re.search(r"\d", lowered):
        return True
    if any(marker in lowered for marker in _CONCRETE_MARKERS):
        return True
    return any(tool in lowered for tool in _TOOL_WORDS)


def _angle_is_reflected(narration: str, angle: str) -> bool:
    angle_tokens = {tok for tok in re.findall(r"[a-z]{5,}", angle.lower())}
    if not angle_tokens:
        return True
    narration_tokens = set(re.findall(r"[a-z]{5,}", narration.lower()))
    return bool(angle_tokens & narration_tokens)


def check_quality(
    *,
    narration: str,
    idea: TopicIdea,
    config: AppConfig,
    history: Mapping[str, Any],
) -> list[str]:
    """Return hard failures; soft quality notes are logged by the caller."""

    issues: list[str] = []
    lowered = narration.lower()

    for phrase in config.quality_guard.banned_phrases:
        if phrase and phrase.lower() in lowered:
            issues.append(f"narration contains the generic filler phrase {phrase!r}")

    if not _angle_is_reflected(narration, idea.angle):
        issues.append("narration does not reflect the chosen unique angle")

    if config.quality_guard.require_number_or_tool and not _has_concrete_detail(narration):
        issues.append(
            "narration has no concrete detail (add a specific number, tool name, setting, or worked example)"
        )

    recent = recent_scripts(history, config.niche, config.quality_guard.recent_scripts_window)
    for entry in recent:
        score = similarity(narration, str(entry.get("text", "")))
        if score >= config.quality_guard.max_similarity_to_recent_scripts:
            issues.append(
                f"narration is {score:.0%} similar to a recent script ({entry.get('topic', 'unknown')!r}); "
                "write it from a different angle"
            )
            break

    return issues


def _soft_notes(*, narration: str, config: AppConfig) -> list[str]:
    notes: list[str] = []
    count = len(_words(narration))
    if count < config.script.min_words:
        notes.append(f"narration is shorter than the ideal ({count} words)")
    if count > config.script.max_words:
        notes.append(f"narration is longer than the ideal ({count} words)")
    return notes


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_payload(
    payload: Mapping[str, Any],
    *,
    idea: TopicIdea,
    config: AppConfig,
    history: Mapping[str, Any],
) -> Script:
    problems: list[str] = []

    hook_raw = payload.get("hook")
    hook = _clean_text(hook_raw) if isinstance(hook_raw, str) else ""
    hook_min, hook_max = _hook_word_range(config)
    if not hook:
        problems.append("hook is missing")
    else:
        hook_words = len(_words(hook))
        if not hook_min <= hook_words <= hook_max:
            problems.append(f"hook must be {hook_min}-{hook_max} words, got {hook_words}")
        if len(hook) > 140:
            problems.append(f"hook is {len(hook)} characters; keep it under 140")

    narration_raw = payload.get("narration")
    narration = _clean_text(narration_raw) if isinstance(narration_raw, str) else ""
    if not narration:
        problems.append("narration is missing")
    else:
        count = len(_words(narration))
        lower_bound = max(30, int(config.script.min_words * 0.95))
        upper_bound = int(config.script.max_words * 1.05) + 1
        if not lower_bound <= count <= upper_bound:
            problems.append(
                f"narration must be {config.script.min_words}-{config.script.max_words} words "
                f"(30-45 seconds); got {count}"
            )
        if not narration.rstrip().endswith((".", "!", "?")):
            narration = f"{narration.rstrip('.!? ')}."
        if any(token in narration for token in ("http://", "https://")):
            problems.append("narration must not contain URLs")

    keywords_raw = payload.get("visual_keywords")
    keywords: list[str] = []
    if isinstance(keywords_raw, Sequence) and not isinstance(keywords_raw, (str, bytes)):
        for item in keywords_raw:
            if not isinstance(item, str):
                continue
            keyword = _clean_text(item).strip(" .,;:\"'")
            keyword = re.sub(r"[^A-Za-z0-9 \-]", "", keyword).strip()
            if not keyword or len(keyword.split()) > 6:
                continue
            if keyword.lower() not in {existing.lower() for existing in keywords}:
                keywords.append(keyword)
    if not config.script.min_keywords <= len(keywords) <= config.script.max_keywords:
        problems.append(
            f"visual_keywords must contain {config.script.min_keywords}-{config.script.max_keywords} "
            f"usable English search phrases; got {len(keywords)}"
        )

    raw_tags = payload.get("tags")
    tags = _finalise_tags(raw_tags, config=config, idea=idea)
    if len(tags) < min(3, config.script.tag_count):
        problems.append("tags could not be normalised into a usable list")

    title_raw = payload.get("title")
    title = _finalise_title(
        title_raw if isinstance(title_raw, str) else "",
        fallback=idea.topic,
        max_chars=config.script.title_max_chars,
    )
    if not title.endswith(SHORTS_SUFFIX):
        problems.append("title could not be normalised to end with #Shorts")

    description_raw = payload.get("description")
    description = _finalise_description(
        description_raw if isinstance(description_raw, str) else "",
        tags=tags,
        idea=idea,
    )
    if len(description) < 40:
        problems.append("description is too short")

    if problems:
        raise ScriptValidationError("; ".join(problems))

    guard_issues = check_quality(narration=narration, idea=idea, config=config, history=history)
    if guard_issues:
        raise ScriptValidationError("; ".join(guard_issues))

    notes = _soft_notes(narration=narration, config=config)
    for note in notes:
        logger.info("Quality note: %s", note)

    return Script(
        hook=hook,
        narration=narration,
        visual_keywords=keywords,
        title=title,
        description=description,
        tags=tags,
        angle=idea.angle,
        quality_notes=notes,
    )


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------


def _fallback_script(idea: TopicIdea, config: AppConfig) -> Script:
    """Build a safe script from the idea when the LLM output keeps failing.

    This is intentionally plain rather than clever: it still carries the unique
    angle and a concrete action, so the video is publishable, not a placeholder.
    """

    logger.warning("Falling back to a deterministic script built from the topic idea")

    hook = _clean_text(idea.angle)
    if len(_words(hook)) > 18:
        hook = " ".join(_words(hook)[:18]).rstrip(",;:") + "."

    narration = _clean_text(
        f"{idea.why_it_matters} "
        f"The trick is simple: {idea.concrete_detail} "
        f"Most students skip this because it feels too small to matter, but the small step is "
        f"what makes the rest stick. Try it once, give it 20 minutes, then compare what you "
        f"remember with what you would have remembered without it. "
        f"If the result surprises you, keep the habit for one week and note what changed."
    )

    # Keep the narration inside the configured word budget by trimming full
    # sentences rather than cutting mid-sentence.
    sentences = re.split(r"(?<=[.!?])\s+", narration)
    trimmed: list[str] = []
    total = 0
    for sentence in sentences:
        count = len(_words(sentence))
        if total + count > config.script.max_words and len(trimmed) >= 3:
            break
        trimmed.append(sentence)
        total += count
    narration = " ".join(trimmed)

    if len(_words(narration)) < config.script.min_words:
        narration = (
            f"{narration} Write the step down where you will see it tomorrow, "
            f"run it for one study session, and keep it only if it saved you time."
        )

    keywords = [
        "student studying laptop",
        "typing on laptop close up",
        "notebook and pen desk",
        "library studying",
        "code on computer screen",
        "smartphone notes app",
    ]

    script = Script(
        hook=hook,
        narration=narration,
        visual_keywords=keywords,
        title=_finalise_title(idea.topic, fallback=idea.topic, max_chars=config.script.title_max_chars),
        description=_finalise_description(idea.why_it_matters, tags=["students", "study tips"], idea=idea),
        tags=_finalise_tags([], config=config, idea=idea),
        angle=idea.angle,
        quality_notes=["deterministic fallback script"],
    )

    guard_issues = check_quality(
        narration=script.narration,
        idea=idea,
        config=config,
        history={"scripts": []},
    )
    if "does not reflect" in " ".join(guard_issues) or any("concrete detail" in issue for issue in guard_issues):
        logger.error("Fallback script quality issues: %s", guard_issues)
    return script


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate_script(
    config: AppConfig,
    client: GeminiClient,
    idea: TopicIdea,
    history: Mapping[str, Any],
) -> Script:
    """Generate and validate a script, retrying with validator feedback."""

    system = _SYSTEM.format(
        niche=config.niche,
        min_words=config.script.min_words,
        max_words=config.script.max_words,
        keyword_examples=_KEYWORD_EXAMPLES,
        title_max=config.script.title_max_chars,
        suffix=SHORTS_SUFFIX,
        tag_count=config.script.tag_count,
    )

    feedback = ""
    last_error: Exception | None = None

    for attempt in range(1, config.script.max_attempts + 1):
        prompt = _PROMPT.format(
            topic=idea.topic,
            angle=idea.angle,
            why=idea.why_it_matters,
            detail=idea.concrete_detail,
            feedback=feedback,
        )
        try:
            payload = client.generate_json(system_instruction=system, prompt=prompt)
            script = _validate_payload(payload, idea=idea, config=config, history=history)
            logger.info(
                "Script ready: %d words, %d keywords, title %r",
                script.word_count,
                len(script.visual_keywords),
                script.title,
            )
            return script
        except (ScriptValidationError, Exception) as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("Script attempt %d/%d failed: %s", attempt, config.script.max_attempts, exc)
            feedback = (
                f"\nYour previous answer was rejected by validation: {exc}\n"
                "Rewrite the whole JSON object and fix every problem listed."
            )

    logger.error("All script attempts failed (%s)", last_error)
    return _fallback_script(idea, config)
