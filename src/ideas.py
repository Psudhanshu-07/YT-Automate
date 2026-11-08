"""Topic selection with permanent de-duplication.

``history.json`` is the single source of truth for what has already been
published (or at least produced). It is committed back to the repository by the
GitHub Actions workflow, so scheduled runs keep accumulating memory even though
each runner starts from a clean checkout.

A topic is rejected when it is textually too close to *any* past topic for the
same niche. That is the first half of the content-quality guard: no recycled
subject matter.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import AppConfig
from .errors import HistoryError, LLMError
from .llm import GeminiClient
from .logging_utils import get_logger

logger = get_logger("ideas")

HISTORY_VERSION = 1
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "get", "got", "has", "have", "how", "i", "if", "in", "into",
    "is", "it", "its", "just", "like", "me", "more", "most", "my", "no", "not",
    "of", "on", "one", "or", "our", "out", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "up", "use", "using", "was",
    "we", "what", "when", "which", "who", "why", "will", "with", "you", "your",
}
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class TopicIdea:
    """One fresh video concept, produced by Gemini and recorded in history."""

    topic: str
    angle: str
    why_it_matters: str
    concrete_detail: str
    created_at: str

    def as_history_entry(self, *, niche: str) -> dict[str, Any]:
        entry = asdict(self)
        entry["niche"] = niche
        entry["status"] = "planned"
        entry["video_id"] = None
        entry["privacy_status"] = None
        return entry


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""

    return " ".join(_WORD.findall(text.lower()))


def _tokens(text: str) -> set[str]:
    return {token for token in _WORD.findall(text.lower()) if token not in _STOPWORDS and len(token) > 2}


def similarity(left: str, right: str) -> float:
    """Return a 0-1 similarity score combining sequence and vocabulary overlap."""

    a, b = normalise(left), normalise(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    sequence = SequenceMatcher(None, a, b).ratio()
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if left_tokens and right_tokens:
        overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    else:
        overlap = 0.0
    return max(sequence, overlap)


def _short_niche(niche: str) -> str:
    """Turn "tech and AI tips for students" into "tech and AI" for prose."""

    label = re.split(r"\s+for\s+", niche.strip(), maxsplit=1, flags=re.IGNORECASE)[0]
    label = label or niche.strip()
    words = label.split()
    return " ".join(words[:4]).strip(" ,.;:")


# ---------------------------------------------------------------------------
# history.json
# ---------------------------------------------------------------------------


def _empty_history(niche: str) -> dict[str, Any]:
    return {
        "version": HISTORY_VERSION,
        "niche": niche,
        "updated_at": None,
        "topics": [],
        "scripts": [],
    }


def load_history(path: Path, niche: str) -> dict[str, Any]:
    """Load history, tolerating missing files and hand-edited entries."""

    if not path.is_file():
        logger.warning("No history file at %s; starting a new one", path)
        return _empty_history(niche)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HistoryError(
            f"{path} is not valid JSON: {exc}",
            hint=(
                "Fix or restore history.json from git. The agent refuses to continue so it "
                "cannot accidentally repeat previously used topics."
            ),
        ) from exc

    if not isinstance(raw, Mapping):
        raise HistoryError(f"{path} must contain a JSON object")

    topics = raw.get("topics")
    if not isinstance(topics, list):
        topics = []

    clean_topics: list[dict[str, Any]] = []
    for entry in topics:
        if isinstance(entry, Mapping) and isinstance(entry.get("topic"), str) and entry["topic"].strip():
            clean_topics.append(dict(entry))
        else:
            logger.warning("Ignoring malformed history topic entry: %r", entry)

    scripts = raw.get("scripts")
    clean_scripts = [
        dict(entry)
        for entry in scripts
        if isinstance(entry, Mapping) and isinstance(entry.get("text"), str)
    ] if isinstance(scripts, list) else []

    return {
        "version": int(raw.get("version", HISTORY_VERSION)),
        "niche": str(raw.get("niche") or niche),
        "updated_at": raw.get("updated_at"),
        "topics": clean_topics,
        "scripts": clean_scripts,
    }


def save_history(path: Path, history: Mapping[str, Any]) -> None:
    """Atomically persist history so a crash cannot truncate the file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(history)
    payload["version"] = HISTORY_VERSION
    payload["updated_at"] = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    logger.info(
        "Saved history: %d topic(s), %d script sample(s)",
        len(payload.get("topics", [])),
        len(payload.get("scripts", [])),
    )


def topics_for_niche(history: Mapping[str, Any], niche: str) -> list[dict[str, Any]]:
    wanted = niche.strip().lower()
    return [
        entry
        for entry in history.get("topics", [])
        if str(entry.get("niche", wanted)).strip().lower() == wanted
    ]


def recent_scripts(history: Mapping[str, Any], niche: str, limit: int) -> list[dict[str, Any]]:
    wanted = niche.strip().lower()
    matches = [
        entry
        for entry in history.get("scripts", [])
        if str(entry.get("niche", wanted)).strip().lower() == wanted
    ]
    return matches[-limit:] if limit > 0 else []


def record_topic(
    history: dict[str, Any],
    idea: TopicIdea,
    *,
    niche: str,
    video_id: str | None,
    privacy_status: str | None,
    status: str,
) -> None:
    entry = idea.as_history_entry(niche=niche)
    entry["video_id"] = video_id
    entry["privacy_status"] = privacy_status
    entry["status"] = status
    history.setdefault("topics", []).append(entry)


def record_script(history: dict[str, Any], *, niche: str, topic: str, text: str, keep: int) -> None:
    entry = {
        "topic": topic,
        "text": text[:4000],
        "niche": niche,
        "created_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    }
    scripts = history.setdefault("scripts", [])
    scripts.append(entry)
    limit = max(5, keep)
    if len(scripts) > limit:
        del scripts[: len(scripts) - limit]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


_SYSTEM = """You are the research lead for a YouTube Shorts channel in the niche below.
Your job is to propose ONE specific, original, genuinely useful video topic.

Niche: {niche}

Requirements for the topic:
- Specific, not a broad category. Name the tool, technique, mistake, or situation.
- Immediately useful to a student within 60 seconds of watching.
- Evergreen: still true and useful in a year. No breaking news, no rumours.
- Never repeat, paraphrase, or lightly reword a past topic.
- No clickbait promises the video cannot keep, no medical/legal/financial advice.
- Each proposal needs a distinctive angle that most videos on this subject miss.

Return STRICT JSON with exactly these keys and string values:
{{
  "topic": "<= 90 characters, specific and concrete",
  "angle": "one sentence: the non-obvious take that makes this video different",
  "why_it_matters": "one sentence on the concrete benefit for a student",
  "concrete_detail": "one specific step, tool name, setting or number the script must mention"
}}
Do not add markdown, commentary, or extra keys."""

_PROMPT = """Propose one fresh topic for a new Short.

Past topics for this niche (do NOT reuse or paraphrase any of these):
{past}

Write for students who are busy, sceptical of hype, and want something they can
apply today. Prefer a concrete, verifiable technique over motivation."""

_FALLBACK_BLUEPRINTS: tuple[dict[str, str], ...] = (
    {
        "topic": "The 10-minute {short} reset that fixes a messy week",
        "angle": "Spending ten minutes organising notes and files on Sunday beats an hour of re-reading later.",
        "why_it_matters": "A small weekly reset removes the daily friction that eats study time.",
        "concrete_detail": "Block 10 minutes, name one folder and one note file, and archive everything older than a week.",
    },
    {
        "topic": "One free tool that replaces three {short} apps",
        "angle": "Consolidating tools beats adding another app, because your notes stay in one searchable place.",
        "why_it_matters": "Every extra app adds a place to lose notes and a subscription to forget.",
        "concrete_detail": "Pick a single free notes app, import two old documents, and delete the duplicate shortcuts.",
    },
    {
        "topic": "Stop revising {short} the hard way: try one-pass recall",
        "angle": "Writing what you remember before opening the notes exposes gaps in five minutes.",
        "why_it_matters": "Recall practice finds the two facts you actually failed to learn.",
        "concrete_detail": "Set a 5-minute timer, write everything you remember on one page, then check only what you missed.",
    },
    {
        "topic": "The {short} mistake that quietly costs two hours a day",
        "angle": "Context switching between tabs and notes costs more time than any individual task.",
        "why_it_matters": "Removing one daily switch can return two hours a week.",
        "concrete_detail": "Close every tab you are not reading now, then keep one workspace per subject.",
    },
    {
        "topic": "Turn one lecture into a {short} cheat sheet in 15 minutes",
        "angle": "Compressing a lecture into ten lines forces the decisions that plain re-reading skips.",
        "why_it_matters": "A one-page summary is faster to review before an exam than a full transcript.",
        "concrete_detail": "Write 10 lines: five facts, three definitions, and two open questions from today's lecture.",
    },
    {
        "topic": "One keyboard-level {short} habit you can learn today",
        "angle": "Learning three shortcuts beats installing another productivity extension.",
        "why_it_matters": "Three shortcuts save a few seconds hundreds of times per week.",
        "concrete_detail": "Learn Ctrl/Cmd+L, Ctrl/Cmd+Shift+T and Ctrl/Cmd+F, and use them for one full study session.",
    },
)


def _fallback_idea(niche: str, past: list[dict[str, Any]]) -> TopicIdea:
    """Deterministic last resort so a Gemini outage cannot block publication."""

    short = _short_niche(niche)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()

    for offset in range(len(_FALLBACK_BLUEPRINTS)):
        blueprint = _FALLBACK_BLUEPRINTS[(len(past) + offset) % len(_FALLBACK_BLUEPRINTS)]
        topic = blueprint["topic"].format(short=short)
        if all(similarity(topic, str(entry.get("topic", ""))) < 0.7 for entry in past):
            logger.warning("Using deterministic fallback topic after LLM failures")
            return TopicIdea(
                topic=topic,
                angle=blueprint["angle"],
                why_it_matters=blueprint["why_it_matters"],
                concrete_detail=blueprint["concrete_detail"],
                created_at=now,
            )

    raise HistoryError(
        "Could not find an unused topic: the LLM failed and every fallback topic is already in history.",
        hint="Add more ideas to _FALLBACK_BLUEPRINTS in src/ideas.py or reset history.json intentionally.",
    )


def _prompt_history(past: Iterable[Mapping[str, Any]], limit: int) -> str:
    topics = [str(entry.get("topic", "")).strip() for entry in past]
    topics = [topic for topic in topics if topic]
    if not topics:
        return "(none yet - this is the first video)"
    return "\n".join(f"- {topic}" for topic in topics[-limit:])


def _clean_field(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _validate_idea(payload: Mapping[str, Any], *, niche: str, past: list[dict[str, Any]], threshold: float) -> TopicIdea:
    topic = _clean_field(payload, "topic").strip(" \"'")
    angle = _clean_field(payload, "angle")
    why = _clean_field(payload, "why_it_matters")
    detail = _clean_field(payload, "concrete_detail")

    if not 10 <= len(topic) <= 120:
        raise LLMError(f"Topic must be 10-120 characters, got {len(topic)}: {topic!r}")
    if len(angle) < 15:
        raise LLMError(f"Angle is too short: {angle!r}")
    if not why:
        raise LLMError("why_it_matters is missing")
    if not detail:
        raise LLMError("concrete_detail is missing")

    for entry in past:
        previous = str(entry.get("topic", ""))
        score = similarity(topic, previous)
        if score >= threshold:
            raise LLMError(
                f"Topic too similar to a past topic ({score:.2f} >= {threshold:.2f}): "
                f"{topic!r} vs {previous!r}"
            )

    return TopicIdea(
        topic=topic,
        angle=angle,
        why_it_matters=why,
        concrete_detail=detail,
        created_at=dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    )


def generate_topic(config: AppConfig, client: GeminiClient, history: Mapping[str, Any]) -> TopicIdea:
    """Ask Gemini for a topic that is not in history, with retries and a fallback."""

    past = topics_for_niche(history, config.niche)
    logger.info("Selecting a topic for niche %r (%d past topic(s))", config.niche, len(past))

    system = _SYSTEM.format(niche=config.niche)
    prompt = _PROMPT.format(past=_prompt_history(past, config.ideas.history_prompt_limit))

    last_error: Exception | None = None
    for attempt in range(1, config.ideas.max_attempts + 1):
        try:
            payload = client.generate_json(system_instruction=system, prompt=prompt)
            idea = _validate_idea(
                payload,
                niche=config.niche,
                past=past,
                threshold=config.ideas.max_topic_similarity,
            )
            logger.info("Topic: %s", idea.topic)
            logger.info("Angle: %s", idea.angle)
            return idea
        except (LLMError, HistoryError) as exc:
            last_error = exc
            logger.warning("Topic attempt %d/%d rejected: %s", attempt, config.ideas.max_attempts, exc)
            prompt = (
                f"{_PROMPT.format(past=_prompt_history(past, config.ideas.history_prompt_limit))}\n\n"
                f"Your previous answer was rejected: {exc}\n"
                "Return a DIFFERENT topic that shares almost no wording with the past topics."
            )

    logger.error("All topic attempts failed; falling back to a deterministic topic (%s)", last_error)
    return _fallback_idea(config.niche, past)
