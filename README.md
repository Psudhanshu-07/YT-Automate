# shorts-agent

An automated YouTube Shorts studio that runs entirely on free tiers.

One scheduled GitHub Actions run picks a fresh topic in your niche, writes a
30-45 second script, records a neural voiceover, burns in word-by-word captions,
edits portrait stock footage to the voice track, and uploads the result to
YouTube with the metadata you configured. `history.json` is committed back after
every run so the agent never repeats a topic.

Default niche: **tech and AI tips for students** (change one line in
`config.yaml`).

---

## Table of contents

1. [What one run does](#what-one-run-does)
2. [Cost](#cost)
3. [Requirements](#requirements)
4. [Repository layout](#repository-layout)
5. [Quick start (local)](#quick-start-local)
6. [Full setup on GitHub, step by step](#full-setup-on-github-step-by-step)
7. [Review mode vs. full auto](#review-mode-vs-full-auto)
8. [Configuration reference](#configuration-reference)
9. [How the pipeline works](#how-the-pipeline-works)
10. [The content-quality guard](#the-content-quality-guard)
11. [Scheduling](#scheduling)
12. [Local dry runs](#local-dry-runs)
13. [Troubleshooting](#troubleshooting)
14. [Manual checklist](#manual-checklist)
15. [Policies, licences, credits](#policies-licences-credits)

---

## What one run does

| Stage | Module | Output |
| --- | --- | --- |
| 1. Pick a topic | `src/ideas.py` | A specific topic + unique angle, checked against `history.json` |
| 2. Write the script | `src/script.py` | Strict JSON: hook, narration, keywords, title, description, tags |
| 3. Voiceover | `src/voice.py` | 48 kHz mono WAV from edge-tts, loudness-normalised |
| 4. Captions | `src/captions.py` | Word-level timestamps from faster-whisper, burned-in `.ass` |
| 5. Footage | `src/visuals.py` | Portrait Pexels clips (landscape centre-cropped as fallback) |
| 6. Music | `src/music.py` | Optional royalty-free track, ducked under the voice |
| 7. Render | `src/video.py` | 1080x1920, 30 fps, H.264/AAC, < 58 s, fades, `preview.jpg` |
| 8. Upload | `src/upload.py` | YouTube `videos.insert`, resumable, retries with backoff |

Every run writes to `output/<UTC-timestamp>/`:

```
output/20261108T131500Z/
├── short.mp4        # the Short (1080x1920, H.264 + AAC)
├── preview.jpg      # a frame from the finished video
├── captions.ass     # the exact subtitle file that was burned in
└── metadata.json    # topic, script, clips used, duration, upload result
```

Nothing in `output/` is committed: CI uploads it as a workflow artifact only when
a run fails, so you can inspect or publish the video manually.

---

## Cost

| Service | Free tier | Used for |
| --- | --- | --- |
| Gemini API | Free tier, per-model daily request limits | Topic + script |
| Pexels API | Free, no card required | Stock footage |
| edge-tts | Free (Microsoft Edge read-aloud endpoint, no key) | Voiceover |
| faster-whisper | Free, open source (runs locally on CPU) | Captions |
| YouTube Data API v3 | Free quota: 100 `videos.insert`/day in its own bucket | Upload |
| GitHub Actions | Free minutes for public repos | Automation |

No paid API, no cloud storage, no GPU. A run takes roughly 5-12 minutes on a
GitHub-hosted runner.

---

## Requirements

* **Python 3.10+** (CI uses 3.11)
* **FFmpeg 5+** built with `libx264`, `libass`, and `aac`
* A YouTube channel + a Google account
* Free Gemini and Pexels API keys

---

## Repository layout

```
shorts-agent/
├── main.py                    # CLI orchestrator (--dry-run, --privacy, ...)
├── get_refresh_token.py       # one-time OAuth helper (run locally)
├── config.yaml                # niche + every tunable
├── history.json               # committed topic/script memory
├── requirements.txt           # pinned dependencies
├── .env.example               # local secret template
├── .github/workflows/shorts.yml
├── assets/music/              # optional royalty-free track (see its README)
└── src/
    ├── config.py              # typed config loading + validation
    ├── errors.py              # exception hierarchy with exit codes
    ├── logging_utils.py       # UTC logging with secret redaction
    ├── env.py                 # .env loader + require_env
    ├── llm.py                 # Gemini client, JSON repair, retries
    ├── ideas.py               # topic selection, de-duplication, history
    ├── script.py              # script JSON validation + quality guard
    ├── voice.py               # edge-tts + loudnorm/atempo
    ├── captions.py            # whisper word timings + ASS renderer
    ├── visuals.py             # Pexels search/download + synthetic fallback
    ├── music.py               # music selection
    ├── ffmpeg_tools.py        # ffmpeg/ffprobe wrappers
    ├── video.py               # timeline + filter graph + render
    └── upload.py              # resumable YouTube upload
```

---

## Quick start (local)

```bash
git clone https://github.com/<you>/shorts-agent.git
cd shorts-agent
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# FFmpeg
#   Ubuntu/Debian: sudo apt-get install -y ffmpeg fonts-dejavu-core
#   macOS:         brew install ffmpeg
#   Windows:       winget install Gyan.FFmpeg   (then reopen the terminal)

cp .env.example .env               # fill in the keys
python main.py --dry-run           # builds the video, skips the upload
```

On Windows PowerShell, set the variables for the session instead of using `cp`:

```powershell
$env:GEMINI_API_KEY="..."
$env:PEXELS_API_KEY="..."
python main.py --dry-run
```

`--dry-run` never touches YouTube and never modifies `history.json`, so you can
iterate on the look and feel without burning topics.

CLI options:

| Flag | Meaning |
| --- | --- |
| `--dry-run` | Generate the video, skip the upload, leave history unchanged |
| `--privacy private\|unlisted\|public` | One-off override of `upload.privacy_status` |
| `--language en\|hi` | One-off narration/voice language override; captions remain English |
| `--config PATH` | Use a different config file |
| `--log-level DEBUG\|INFO\|WARNING\|ERROR` | Verbosity |
| `--keep-build` | Keep intermediate clips/audio/filter graph in `build/<run>/` |

Exit codes: `0` success, `1` generation failure, `2` upload failure (video kept),
`3` configuration/environment problem, `130` interrupted.

---

## Full setup on GitHub, step by step

You only do this once. Budget 20-30 minutes, mostly waiting for Google's UI.

### 1. Get the repository onto GitHub

Fork or create a repository from this code and make sure Actions are enabled
(**Settings -> Actions -> General -> Allow all actions**). The workflow needs
**Settings -> Actions -> General -> Workflow permissions -> Read and write
permissions** (or the `permissions: contents: write` already declared in the
workflow, which is sufficient on its own).

### 2. Get a Gemini API key (free)

1. Go to <https://aistudio.google.com/apikey>.
2. Sign in with your Google account.
3. **Create API key** -> choose/create a Google Cloud project -> copy the key.
4. Keep it private; you will paste it into GitHub Secrets in step 8.

The default model is `gemini-3.8-flash`. Free-tier quotas and model names
change over time; if your account does not have this model, change `llm.model`
in `config.yaml` and update its fallback list.

### 3. Get stock footage (Pexels optional)

1. Create an account at <https://www.pexels.com/>.
2. Open <https://www.pexels.com/api/> and request a key (instant, no card).
3. Copy the key. Respect the Pexels licence: do not resell the footage itself.

Pexels is optional. If `PEXELS_API_KEY` is missing, the pipeline automatically
uses video files from [Wikimedia Commons](https://commons.wikimedia.org/)
through its public API; no key or account is required. Commons results are
less predictable than Pexels, so the pipeline may use its generated gradient
fallback when no suitable video is available. Source IDs are recorded in
`history.json`, so future runs avoid reusing footage already used by this
channel; fallback gradients are varied per run.

### 4. Create a Google Cloud project and enable the YouTube Data API v3

1. Open <https://console.cloud.google.com/>.
2. Top bar -> project selector -> **New project** -> name it (e.g.
   `shorts-agent`) -> **Create**, then select it.
3. **APIs & Services -> Library** -> search **YouTube Data API v3** -> **Enable**.

> If your Google account manages more than one channel, make sure you later sign
> in with the account that owns the intended channel.

### 5. Configure the OAuth consent screen

Google now calls this **Google Auth Platform** in some accounts; older accounts
show **OAuth consent screen**.

1. **APIs & Services -> OAuth consent screen** (or **Google Auth Platform**).
2. **Get started** / **Branding**: app name `shorts-agent`, your email as user
   support email, your email again as developer contact. Save.
3. **Audience**: choose **External** (required for a normal Gmail account), save.
4. **Test users -> Add users**: add the exact Google account that owns the
   YouTube channel. While the app is in **Testing**, only these accounts can
   authorise it.
5. **Data access -> Add or remove scopes**: add
   `https://www.googleapis.com/auth/youtube.upload` (shown as "Manage your
   YouTube videos"). Save.

> **Testing mode expires refresh tokens after 7 days.** For a long-lived token,
> publish the app: **OAuth consent screen -> Publishing status -> Publish app**.
> For personal use you can stay "unverified"; Google shows a warning screen and
> caps the app at 100 users, which is fine here.

### 6. Create OAuth Desktop credentials

1. **APIs & Services -> Credentials -> Create credentials -> OAuth client ID**.
2. Application type: **Desktop app** (not Web application - the refresh-token
   flow in `get_refresh_token.py` depends on it).
3. **Create**, then **Download JSON**. Rename it to `client_secret.json` and put
   it in the repository root. It is git-ignored.

### 7. Generate the refresh token locally

```bash
pip install -r requirements.txt
python get_refresh_token.py --client-secrets client_secret.json
```

* A browser opens. Sign in with the channel's Google account.
* If you see "Google hasn't verified this app": **Advanced** ->
  **Go to shorts-agent (unsafe)**. This is expected for your own unverified app.
* Approve **Manage your YouTube videos**.
* The terminal prints the refresh token. Keep it private.

You can add `--write-env` to store it in `.env` locally instead of printing only:

```bash
python get_refresh_token.py --client-secrets client_secret.json --write-env
```

Delete `client_secret.json` afterwards if you like; only the refresh token is
needed by CI.

### 8. Add the required GitHub secrets

**Repository -> Settings -> Secrets and variables -> Actions -> New repository
secret**. Add exactly these names:

| Secret | Value |
| --- | --- |
| `GEMINI_API_KEY` | From step 2 |
| `OPENROUTER_API_KEY` | Optional fallback key from <https://openrouter.ai/keys> |
| `YT_CLIENT_ID` | From the downloaded client JSON (`installed.client_id`) |
| `YT_CLIENT_SECRET` | From the client JSON (`installed.client_secret`) |
| `YT_REFRESH_TOKEN` | From step 7 |

`PEXELS_API_KEY` is optional. If omitted, Wikimedia Commons is used for
footage. `OPENROUTER_API_KEY` is optional; when set, it is used automatically
if Gemini is unavailable or its quota is exhausted. Nothing else is required;
`GITHUB_TOKEN` is provided automatically.

### 9. Run the workflow manually

1. **Actions -> Produce YouTube Short -> Run workflow**.
2. First run: set **dry_run** to `true`. This produces the video but publishes
   nothing. Check the logs and the "YouTube Short produced" summary.
3. For a real upload, set **dry_run** to `false`. Action-triggered uploads are
   forced to **public**; there is no private/unlisted override in the workflow.
4. Open **YouTube Studio** to confirm the public video. If the YouTube app is in
   "Testing" mode and your refresh token is older than 7 days, redo step 7.

> A failed run still uploads the produced video as an artifact
> (`Actions -> the run -> Artifacts -> short-<run id>`), retained for 7 days.

### 10. Public automation

The workflow is configured for four public Shorts per day at 07:17, 11:17,
15:17, and 19:17 UTC. `config.yaml` also uses `upload.privacy_status: "public"` for
local runs. Change the cron expressions in
[`.github/workflows/shorts.yml`](./.github/workflows/shorts.yml) if you need
different times.

> Uploads from unverified API projects are **no longer locked to private** as of
> the July 2026 policy change; `privacyStatus` is honoured. If a video still
> appears private, check the account/channel settings and YouTube Studio's
> "locked as private" notice.

---

## Review mode vs. full auto

`config.yaml -> upload.privacy_status` accepts:

* `private`: every video lands in YouTube Studio for review.
* `unlisted`: publishable by link only, useful for sharing drafts.
* `public` (current setting): fully automatic publishing.

The generated metadata includes a search-friendly title, description, hashtags,
tags, Science & Technology category, English language metadata, and Shorts
portrait dimensions. These improve discoverability but cannot guarantee
views. Monitor YouTube Studio and the channel's daily upload limits.

---

## Configuration reference

Everything is validated at startup; typos in section or key names fail the run
with a clear message rather than being silently ignored.

### Niche and channel

```yaml
niche: "tech and AI tips for students"    # change this to re-target the agent
channel:
  name: "Shorts Agent"
  language: "en"                          # used by whisper and YouTube metadata
```

`history.json` stores the niche on each entry, so changing the niche starts a
fresh topic space while keeping old entries for reference.

### LLM

```yaml
llm:
  model: "gemini-2.5-flash"
  fallback_models: ["gemini-2.0-flash", "gemini-flash-latest"]
  temperature: 0.95
  max_output_tokens: 2048
  max_retries: 3
  retry_base_seconds: 4.0
```

### Script shape and quality guard

```yaml
script:
  min_words: 78          # ~31 s at 150 wpm
  max_words: 110         # ~44 s at 150 wpm
  min_keywords: 5
  max_keywords: 8
  title_max_chars: 70    # includes " #Shorts"
  tag_count: 8

quality_guard:
  max_similarity_to_recent_scripts: 0.82
  recent_scripts_window: 15
  require_number_or_tool: true
  banned_phrases: ["in today's fast-paced world", ...]
```

### Voice (`edge-tts`)

```yaml
voice:
  voice: "en-US-AndrewNeural"
  rate: "+6%"
  pitch: "+0Hz"
  max_audio_seconds: 50       # longer narration is atempo-compressed (max 1.45x)
  tail_padding_seconds: 0.45  # silence so the last caption is not clipped
  loudness_target_lufs: -14.0
```

List voices with `edge-tts --list-voices`. Good alternatives:
`en-US-AriaNeural`, `en-GB-RyanNeural`, `en-IN-PrabhatNeural`.

### Captions

```yaml
captions:
  whisper_model: "base"        # tiny (fastest) .. large-v3 (most accurate)
  whisper_compute_type: "int8" # CPU-friendly
  font: "Montserrat"           # installed by the workflow
  font_size: 92
  primary_color: "&H00FFFFFF"  # ASS is &HAABBGGRR
  highlight_color: "&H0000E5FF"
  outline: 5
  shadow: 2
  margin_v: 520                # distance from the bottom edge, in pixels
  uppercase: true
  max_chars_per_line: 20
  max_lines: 2
```

### Video

```yaml
video:
  width: 1080
  height: 1920
  fps: 30
  max_duration_seconds: 57     # hard ceiling is 58 in code
  clip_max_seconds: 4.2        # faster visual rhythm for creator-style edits
  fade_seconds: 0.18           # applied only at the start and end
  crf: 21
  preset: "veryfast"
```

### Upload

```yaml
upload:
  privacy_status: "public"
  category_id: "28"            # 28 = Science & Technology
  made_for_kids: false
  embeddable: true
  license: "youtube"
  chunk_size_mb: 8
  max_retries: 6
```

### Music

```yaml
music:
  enabled: true
  track: ""          # empty = first audio file in assets/music/
  volume: 0.07
  duck: true         # sidechain-compress under the voice
```

### Schedule

Edit the cron line in `.github/workflows/shorts.yml`:

```yaml
on:
  schedule:
    - cron: "17 7 * * *"   # Hindi, 12:47 IST
    - cron: "17 11 * * *"  # English, 16:47 IST
    - cron: "17 15 * * *"  # Hindi, 20:47 IST
    - cron: "17 19 * * *"  # English, 00:47 IST
```

---

## How the pipeline works

**Topic de-duplication** (`src/ideas.py`)
Topics are normalised (lowercased, punctuation stripped) and compared with every
past topic for the niche using a combination of `difflib` sequence ratio and
Jaccard vocabulary overlap. Anything above `ideas.max_topic_similarity` (0.75 by
default) is rejected and regenerated. The prompt also receives the last 60 topics
explicitly, and malformed history entries are skipped rather than crashing the
run.

**Strict script JSON** (`src/script.py`)
The model must return exactly `hook`, `narration`, `visual_keywords`, `title`,
`description`, `tags`. Every field is validated (hook length, narration word
budget, 5-8 usable English search phrases, title <= 70 chars ending in
`#Shorts`, 8 tags, description length). Validation errors are fed back into the
next attempt, and a deterministic script built from the topic idea is used if all
attempts fail - so a bad model day never blocks the channel.

**Voice** (`src/voice.py`)
edge-tts synthesises an MP3; FFmpeg then applies `atempo` (only if the narration
exceeds `voice.max_audio_seconds`), `loudnorm` to -14 LUFS, and a short silence
tail, producing 48 kHz mono PCM WAV.

**Captions** (`src/captions.py`)
faster-whisper runs on CPU with `word_timestamps=True` and VAD. Words are grouped
into 1-2 line chunks (20 characters per line, max 3.2 s). Each chunk is emitted
once per spoken word with the active word recoloured amber, which gives an exact
karaoke highlight with plain libass. If whisper cannot run, timings are estimated
from the narration text so captions still appear.

**Footage** (`src/visuals.py`)
Portrait search first, landscape second; the best efficient MP4 rendition is
chosen (favour ~1080p, avoid huge 4K downloads), streamed to disk with a size
cap, duration-checked with ffprobe, and shuffled so runs differ. Pexels is used
when `PEXELS_API_KEY` is set; otherwise Wikimedia Commons is queried without a
key. Nothing is cached between runs. If no source yields a usable clip, a
gradient clip is synthesised with animated texture and a grid rather than a
flat background.

**Render** (`src/video.py`)
A single FFmpeg invocation with a filter-graph script file:
`fps -> overscan -> animated crop/pan -> optional hflip -> colour grade ->
sharpen -> concat -> ass burn-in`, plus an audio graph that splits the voice, sidechain-
compresses the music under it, then pads and limits the mix. Output is validated
for H.264, AAC, exact frame size, and duration < 58 s.

**Upload** (`src/upload.py`)
The refresh token is exchanged for an access token up front (so expired tokens
fail fast), then `videos.insert` streams the file resumably (8 MB chunks) with
exponential backoff + jitter on 429/5xx/network errors. `selfDeclaredMadeForKids`
and privacy come from config.

**History commit-back**
Only after a successful run does the workflow commit `history.json` with
`GITHUB_TOKEN`, retrying with `git pull --rebase` if the branch moved. Dry runs
do not modify it. Failed uploads *do* record the topic, so the next run cannot
produce a duplicate of a video that is sitting in the artifact.

---

## The content-quality guard

YouTube penalises channels that mass-produce near-identical content. Four
independent checks run before a script is accepted:

1. **Unique angle per topic.** Every idea carries an `angle` (the non-obvious
   take) and a `concrete_detail`. The script must reflect the angle - checked by
   token overlap - or it is rejected and rewritten.
2. **No repeated subjects.** Topic similarity against the entire niche history
   (see above), plus a hard block on identical/lightly reworded topics.
3. **No template filler.** Configurable `banned_phrases` ("in today's fast-paced
   world", "game-changer", "let's dive in", ...) are rejected outright.
4. **Substance required.** `require_number_or_tool` demands a specific number, a
   worked example marker, or a named tool in the narration.
5. **Script-level novelty.** The narration is compared against the last
   `recent_scripts_window` scripts in `history.json`; anything above
   `max_similarity_to_recent_scripts` is rejected and regenerated.

The prompt also forbids inventing statistics, cites no sources, and asks for a
single actionable step - which keeps claims safe and content genuinely useful.

This reduces risk; it cannot guarantee monetisation. Review the videos yourself,
keep the niche focused, and do not publish content you would not defend.

---

## Scheduling

* `.github/workflows/shorts.yml` runs four times daily and can be started
  manually (`workflow_dispatch`) with the `dry_run` input. Non-dry-run Action
  uploads are forced to public.
* `concurrency` prevents two runs from writing `history.json` at once.
* Cron is UTC and GitHub may delay scheduled jobs by several minutes under load.
* Scheduled workflows run only on the **default branch**.
* GitHub disables scheduled workflows after **60 days of repository inactivity**.
  Any push, or manually running the workflow, resets that timer. If your channel
  can afford a gap, ignore it; otherwise add a monthly reminder to run it once.
* Each scheduled run consumes a new topic and one
  `videos.insert` call from the daily bucket. Four runs are already configured;
  add more cron lines only if you need additional uploads.
* LLM calls try Gemini first, then OpenRouter (`openrouter/free` by default) if
  Gemini fails or reaches a rate/quota limit. Store both keys as Actions secrets;
  never put either key in `config.yaml` or source code.

---

## Local dry runs

```bash
python main.py --dry-run --log-level DEBUG --keep-build
```

What to inspect:

* `output/<run>/short.mp4` - the deliverable
* `output/<run>/preview.jpg` - a representative frame
* `output/<run>/captions.ass` - open in VLC/HandBrake to inspect styling
* `build/<run>/filter.txt` - the exact FFmpeg filter graph
* `build/<run>/voice.wav` - the normalised voice track

Common tweaks while iterating:

| Goal | Change |
| --- | --- |
| Bigger/smaller captions | `captions.font_size` |
| Captions higher up | decrease `captions.margin_v` |
| Different caption colour | `captions.highlight_color` (`&HAABBGGRR`) |
| Faster/slower speech | `voice.rate` (`+10%`, `-5%`) |
| Longer clips on screen | `video.clip_max_seconds` |
| Softer music | `music.volume` (0.03-0.10) |
| No music | `music.enabled: false` |
| Faster renders | `video.preset: "ultrafast"`, `video.crf: 24` |
| Faster captions | `captions.whisper_model: "tiny"` |

---

## Troubleshooting

### "Missing required environment variable ..." (exit code 3)

Locally: export the required variables or copy `.env.example` to `.env`.
In CI: check **Settings -> Secrets and variables -> Actions** for the exact names
`GEMINI_API_KEY`, `YT_CLIENT_ID`, `YT_CLIENT_SECRET`, and `YT_REFRESH_TOKEN`.
`PEXELS_API_KEY` is optional; without it, Wikimedia Commons is used. Secret
names are case-sensitive; values are never printed in logs.

### "FFmpeg was not found on PATH"

Ubuntu: `sudo apt-get install -y ffmpeg`. macOS: `brew install ffmpeg`. Windows:
`winget install Gyan.FFmpeg` then reopen the terminal. Verify with
`ffmpeg -version` and `ffprobe -version`. If FFmpeg lives somewhere odd, set
`FFMPEG_BINARY` / `FFPROBE_BINARY`.

### "No such filter: 'ass'" or captions are missing

Your FFmpeg build lacks libass. Ubuntu's `ffmpeg` package has it; some minimal
Docker images and Windows builds do not. Install a full build (gyan.dev or BtbN
for Windows) and re-render.

### Gemini 404 "model not found" or 429 "quota exceeded"

* Model IDs change. Set `llm.model` to a model your key can use (check
  <https://ai.google.dev/gemini-api/docs/models>) and keep alternatives in
  `llm.fallback_models`.
* 429 means the free-tier daily/ per-minute limit is exhausted. Wait for the
  reset or switch the project/key. The client retries with backoff and then tries
  the next model before failing.
* The topic generator falls back to a deterministic idea list if Gemini is
  completely unavailable, so a quota day still produces a video.

### edge-tts fails ("NoAudioReceived", websocket, timeout)

edge-tts needs outbound HTTPS to Microsoft's speech endpoint. Check your network
or proxy, and confirm the voice name with `edge-tts --list-voices`. Transient
failures are retried; persistent ones usually mean a blocked network or a typo in
`voice.voice`.

### Pexels 401 / 403

The key is wrong or rate-limited. Keys are instant from
<https://www.pexels.com/api/>. The pipeline logs a warning and keeps going with
other keywords, then falls back to a synthesised gradient.

### Transcription is slow

The first run downloads the whisper model (~75 MB for `base`) from Hugging Face;
CI caches it. On a slow connection use `captions.whisper_model: "tiny"`. The
workflow caches it under `HF_HOME` and the cache key includes `config.yaml`, so
changing the model re-downloads once.

### Rendering is slow or the job times out

The job has a 45-minute timeout. Lower `video.preset` to `ultrafast`, raise
`video.crf` to 23-24, or reduce `captions.whisper_model`. A 57-second 1080x1920
render normally takes 1-4 minutes with `veryfast`.

### "invalid_grant" / token expired after a week

Your OAuth consent screen is in **Testing**: refresh tokens expire after 7 days.
Either re-run `get_refresh_token.py` weekly, or publish the app
(**OAuth consent screen -> Publish app**) to get a non-expiring token.

### "Access blocked: ... has not completed the Google verification process"

Add your Google account under **OAuth consent screen -> Test users**, or publish
the app.

### "redirect_uri_mismatch" / "Error 400: invalid_request"

You created a **Web application** client instead of a **Desktop app** client, or
you are running `get_refresh_token.py` with a browser that cannot reach
`localhost`. Create a Desktop client and retry; the script uses a random local
port.

### YouTube upload 403 `quotaExceeded` / `uploadLimitExceeded`

`videos.insert` has its own daily bucket (100 uploads/day by default, 1 unit
each) and resets at midnight Pacific Time. Also check the channel's own daily
upload limit (new channels are capped). Nothing else consumes this bucket.

### Upload fails with 400 and a "tags" or "title" error

YouTube limits titles to 100 characters and tags to 500 characters total. The
validator keeps titles <= 70 and 8 tags, so this usually means a title contains
disallowed characters (`<`, `>`) - adjust the model output or your config.

### The workflow does not run on schedule

* Schedules only run on the default branch.
* GitHub disables cron after 60 days without repository activity - push a commit
  or run it manually to re-enable.
* Cron can be delayed; it is not real-time.

### "Could not push history.json"

The job needs **contents: write**. If branch protection blocks bots, allow
`github-actions[bot]` to push, or run the workflow from a branch without
protection. The pipeline's video still uploaded successfully; only the memory
commit failed.

### Video appears vertically stretched or cropped oddly

`video.width`/`video.height` must be 9:16 (1080x1920 by default). Clips are scaled
to *cover* and centre-cropped, so a landscape 16:9 source loses its sides - that
is expected. Prefer portrait footage; Pexels portrait results are tried first.

### "history.json is not valid JSON"

The agent refuses to run so it cannot accidentally repeat topics. Restore the file
from git (`git checkout -- history.json`) or fix the syntax.

### ModuleNotFoundError: No module named 'src'

Run the pipeline from the repository root: `python main.py`, not
`python src/main.py`.

### Dry run did not add the topic to history

By design: dry runs leave `history.json` untouched so you can iterate without
consuming topics. The topic is recorded only on real runs.

---

## Manual checklist

Things that cannot be automated away (all one-time):

- [ ] Create the GitHub repository (or fork) and enable Actions.
- [ ] Create a free Gemini API key and a free Pexels API key.
- [ ] Create a Google Cloud project, enable **YouTube Data API v3**.
- [ ] Configure the OAuth consent screen (External), add your Google account as a
      **test user**, add the `youtube.upload` scope.
- [ ] Create **Desktop app** OAuth credentials and download the JSON.
- [ ] Run `python get_refresh_token.py --client-secrets client_secret.json`.
- [ ] Add the required repository secrets (`PEXELS_API_KEY` is optional).
- [ ] Run the workflow manually with `dry_run=true`, then use
      `dry_run=false` for a public upload.
- [ ] Decide whether to publish the OAuth app (to stop 7-day token expiry).
- [ ] Confirm the four daily public cron runs are enabled.
- [ ] Optional: drop a royalty-free track into `assets/music/`.
- [ ] Optional: tune `config.yaml` (niche, voice, caption style, schedule).

---

## Policies, licences, credits

* **Footage:** Pexels when configured, otherwise Wikimedia Commons. Check the
  individual Commons file licence before publishing and provide attribution
  when that licence requires it. Do not resell the footage itself or imply
  endorsement. Landscape clips are centre-cropped.
* **Music:** you must supply a track you are licensed to publish. See
  `assets/music/README.md`.
* **Voice:** edge-tts uses Microsoft's read-aloud service. Free for personal use;
  do not build a paid TTS product on it.
* **Font:** Montserrat is licensed under the SIL Open Font License and is
  downloaded by CI from the official Montserrat repository.
* **AI disclosure:** YouTube requires disclosure when content is realistic and
  AI-generated in a way that could mislead. These Shorts use stock footage, TTS
  and captions rather than synthetic people or events, so disclosure is normally
  not required - but you are responsible for what your channel publishes.
* **Originality:** the guard above reduces mass-produced-content risk. Review the
  output and keep each video genuinely useful.
* **Your account:** the agent uploads with your credentials, under the YouTube
  API Services Terms of Service. You are responsible for the channel's compliance.

Everything in this repository is free to use and modify. No paid services, no
trackers, no telemetry.
