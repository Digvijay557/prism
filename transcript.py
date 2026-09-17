"""
transcript.py
Handles: extracting a YouTube video ID from any URL format,
and fetching the transcript/caption text (timed segments) for that video.

Source order:
  1. yt-dlp directly (needs YT_COOKIES to get past YouTube's bot check
     from most server IPs)
  2. youtube-transcript.ai (free, no key, occasionally flaky/unparseable)
  3. Supadata (paid/free-tier API, proxies on their end -- not IP blocked)
     -- used as a fallback when the above fail
"""

import re
import os
import json
import time
import urllib.request
import urllib.parse
import urllib.error
import yt_dlp

COOKIE_PATH = "/tmp/yt_cookies.txt"

# Routes yt-dlp's YouTube requests through your phone's ngrok tunnel,
# since that's the only source here that talks to YouTube directly
# and gets blocked on Render's datacenter IP.
PHONE_PROXY = os.environ.get("PHONE_PROXY", "").strip()

# ---------------------------------------------------------------------------
# Supadata config
# ---------------------------------------------------------------------------
SUPADATA_API_KEY = os.environ.get("SUPADATA_API_KEY", "").strip()
SUPADATA_BASE = "https://api.supadata.ai/v1"
SUPADATA_TIMEOUT = float(os.environ.get("SUPADATA_TIMEOUT", "30"))

# Override in the environment to point at a different transcript provider
# without touching this file, e.g. TRANSCRIPT_API_BASE=https://other-host/api
TRANSCRIPT_API_BASE = os.environ.get(
    "TRANSCRIPT_API_BASE", "https://youtube-transcript.ai"
).rstrip("/")

# The endpoint generates transcripts on demand, so a cold request for an
# uncached video can take a while. Generous by default; tune via env if needed.
TRANSCRIPT_TIMEOUT = float(os.environ.get("TRANSCRIPT_TIMEOUT", "45"))
TRANSCRIPT_RETRIES = int(os.environ.get("TRANSCRIPT_RETRIES", "3"))

# Status codes worth trying again. 404 is deliberately absent: it means
# there's no transcript for this video, and no amount of retrying helps.
_RETRYABLE_STATUSES = {408, 425, 429, 500, 502, 503, 504}


def extract_video_id(url: str) -> str | None:
    """
    Pulls the 11-character YouTube video ID out of any common URL format:
    - https://www.youtube.com/watch?v=VIDEOID
    - https://youtu.be/VIDEOID
    - https://www.youtube.com/watch?v=VIDEOID&t=30s
    - https://m.youtube.com/watch?v=VIDEOID
    - https://www.youtube.com/shorts/VIDEOID
    Returns None if no valid ID pattern is found.
    """
    if not url:
        return None

    patterns = [
        r"(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/live/)([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


# ---------------------------------------------------------------------------
# Fallback source: Supadata
# ---------------------------------------------------------------------------
def _fetch_from_supadata(video_id: str) -> dict:
    """
    Fetches timestamped transcript segments from Supadata.

    Returns {"success": True, "segments": [...], "title": ""}
    (Supadata's transcript endpoint doesn't return a title -- callers
    should fall back to scraper.py's oEmbed/scrape path for that.)
    or {"success": False, "error": "..."}
    """
    if not SUPADATA_API_KEY:
        return {"success": False, "error": "SUPADATA_API_KEY not set."}

    video_url = f"https://www.youtube.com/watch?v={video_id}"
    params = urllib.parse.urlencode(
        {"url": video_url, "text": "false", "lang": "en"}
    )
    url = f"{SUPADATA_BASE}/transcript?{params}"

    req = urllib.request.Request(
        url,
        headers={"x-api-key": SUPADATA_API_KEY, "Accept": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=SUPADATA_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        if e.code == 402:
            return {"success": False, "error": "Supadata: credits exhausted (402)."}
        if e.code == 401:
            return {"success": False, "error": "Supadata: invalid API key (401)."}
        return {"success": False, "error": f"Supadata HTTP {e.code}: {body[:200]}"}
    except Exception as e:
        return {"success": False, "error": f"Supadata request failed: {type(e).__name__}: {e}"}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"success": False, "error": f"Supadata returned unparseable JSON: {raw[:200]}"}

    if "error" in data:
        return {"success": False, "error": f"Supadata: {data.get('message', data['error'])}"}

    content = data.get("content", [])
    if not content:
        return {"success": False, "error": "Supadata returned no transcript content."}

    segments = []
    for chunk in content:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        segments.append(
            {
                "text": text,
                "start": chunk.get("offset", 0) / 1000.0,   # ms -> s
                "duration": chunk.get("duration", 0) / 1000.0,  # ms -> s
            }
        )

    if not segments:
        return {"success": False, "error": "Supadata content had no usable segments."}

    return {"success": True, "segments": segments, "title": ""}


# ---------------------------------------------------------------------------
# Secondary source: youtube-transcript.ai (or whatever TRANSCRIPT_API_BASE is)
# ---------------------------------------------------------------------------

_TIMESTAMP_LINE_RE = re.compile(r"^\[(\d+):(\d{2})(?::(\d{2}))?\]\s*(.*)$")
_TITLE_HEADER_RE = re.compile(r"^#\s*Transcript:\s*(.+)$")


def _timestamp_to_seconds(mm_or_hh: str, ss: str, extra_ss: str | None) -> float:
    """
    Converts a [M:SS] or [H:MM:SS] style timestamp into total seconds.
    """
    if extra_ss is not None:
        # Format was [H:MM:SS]
        hours = int(mm_or_hh)
        minutes = int(ss)
        seconds = int(extra_ss)
        return hours * 3600 + minutes * 60 + seconds
    else:
        # Format was [M:SS] or [MM:SS]
        minutes = int(mm_or_hh)
        seconds = int(ss)
        return minutes * 60 + seconds


def _extract_title_from_header(raw_text: str) -> str:
    """
    youtube-transcript.ai's response starts with a line like:
        # Transcript: Everyone Is Wrong About AI Data Centers Water Use
    Pulls the title out of that line. Returns "" if not found.
    """
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _TITLE_HEADER_RE.match(line)
        if match:
            return match.group(1).strip()
        # The title line is always the first non-blank line -- if we hit
        # something else first, there's no title header to find.
        break
    return ""


def _parse_transcript_ai_text(raw_text: str) -> list:
    """
    Parses the markdown-ish response from youtube-transcript.ai into our
    segments shape: [{"text": ..., "start": ..., "duration": ...}, ...]

    Expected format includes lines like:
        [0:01] Some spoken text here...
        [0:31] More text...
    with a header before the "## Transcript" line and a footer after
    the final "---" that we ignore.
    """
    lines = raw_text.splitlines()

    # Only look at content after the "## Transcript" marker, if present.
    if "## Transcript" in raw_text:
        start_idx = next(
            (i for i, line in enumerate(lines) if line.strip() == "## Transcript"),
            0,
        )
        lines = lines[start_idx + 1:]

    raw_segments = []
    for line in lines:
        line = line.strip()
        if not line or line == "---":
            continue
        if line.startswith("Generated by"):
            continue

        match = _TIMESTAMP_LINE_RE.match(line)
        if not match:
            continue

        mm_or_hh, ss, extra_ss, text = match.groups()
        start_seconds = _timestamp_to_seconds(mm_or_hh, ss, extra_ss)
        text = text.strip()
        if text:
            raw_segments.append({"start": start_seconds, "text": text})

    if not raw_segments:
        return []

    # Compute a rough duration for each segment as the gap to the next one.
    segments = []
    for i, seg in enumerate(raw_segments):
        if i + 1 < len(raw_segments):
            duration = max(raw_segments[i + 1]["start"] - seg["start"], 1.0)
        else:
            duration = 30.0  # best guess for the final segment
        segments.append(
            {"text": seg["text"], "start": seg["start"], "duration": duration}
        )

    return segments


def _transcript_url(video_id: str) -> str:
    return f"{TRANSCRIPT_API_BASE}/transcript/{video_id}.txt"


def _fetch_from_transcript_ai(video_id: str) -> dict:
    """
    Fetches the transcript (and title) from the transcript API, retrying
    on timeouts and transient server errors.

    Returns {"success": True, "segments": [...], "title": "..."}
    or {"success": False, "error": ...}
    """
    url = _transcript_url(video_id)
    last_error = "no attempt made"

    for attempt in range(1, TRANSCRIPT_RETRIES + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/plain, */*",
                },
            )
            with urllib.request.urlopen(req, timeout=TRANSCRIPT_TIMEOUT) as resp:
                raw_text = resp.read().decode("utf-8", errors="replace")

        except urllib.error.HTTPError as e:
            last_error = f"HTTP {e.code} from {TRANSCRIPT_API_BASE}"
            if e.code == 404:
                return {
                    "success": False,
                    "error": (
                        "Transcript source has no captions for this video "
                        "(HTTP 404) -- the video may have captions disabled."
                    ),
                }
            if e.code not in _RETRYABLE_STATUSES:
                return {"success": False, "error": last_error}

        except Exception as e:
            # Covers socket.timeout, URLError, connection resets, etc.
            last_error = f"{type(e).__name__}: {e}"

        else:
            segments = _parse_transcript_ai_text(raw_text)
            if segments:
                return {
                    "success": True,
                    "segments": segments,
                    "title": _extract_title_from_header(raw_text),
                }

            # A 200 with nothing parseable sometimes means "still generating,
            # come back shortly", so this is worth one more go.
            last_error = (
                f"returned {len(raw_text)} chars but no parseable "
                f"timestamped segments"
            )

        if attempt < TRANSCRIPT_RETRIES:
            backoff = 2 ** (attempt - 1)  # 1s, 2s, 4s...
            print(
                f"[transcript] attempt {attempt}/{TRANSCRIPT_RETRIES} failed "
                f"({last_error}); retrying in {backoff}s",
                flush=True,
            )
            time.sleep(backoff)

    return {
        "success": False,
        "error": (
            f"Transcript source failed after {TRANSCRIPT_RETRIES} attempts. "
            f"Last error: {last_error}"
        ),
    }


# ---------------------------------------------------------------------------
# Primary source: yt-dlp
# ---------------------------------------------------------------------------


def _prepare_cookie_file() -> str | None:
    """
    If the YT_COOKIES env var is set (full Netscape cookies.txt contents),
    write it to a temp file and return the path. Otherwise return None.
    """
    raw = os.environ.get("YT_COOKIES", "").strip()
    if not raw:
        return None

    try:
        if "\\n" in raw and "\n" not in raw:
            raw = raw.replace("\\n", "\n")
        with open(COOKIE_PATH, "w", encoding="utf-8") as f:
            f.write(raw + "\n")
        return COOKIE_PATH
    except Exception:
        return None


def _parse_json3_captions(raw_bytes: bytes) -> list:
    """
    Parses YouTube's json3 caption format into our segments shape.
    """
    data = json.loads(raw_bytes.decode("utf-8"))
    segments = []

    for event in data.get("events", []):
        if "segs" not in event:
            continue
        text = "".join(seg.get("utf8", "") for seg in event["segs"]).strip()
        if not text:
            continue
        start_ms = event.get("tStartMs", 0)
        duration_ms = event.get("dDurationMs", 0)
        segments.append(
            {
                "text": text,
                "start": start_ms / 1000.0,
                "duration": duration_ms / 1000.0,
            }
        )

    return segments


def _build_ydl_opts(client: str, cookie_file: str | None) -> dict:
    opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {"youtube": {"player_client": [client]}},
    }
    if cookie_file:
        opts["cookiefile"] = cookie_file
    if PHONE_PROXY:
        opts["proxy"] = PHONE_PROXY
    return opts


def _fetch_from_yt_dlp(video_id: str) -> dict:
    """
    Primary transcript fetch using yt-dlp directly. Also grabs the title
    from yt-dlp's info dict, since we have it right there.

    Note: from most server/datacenter IPs, YouTube will show a "Sign in to
    confirm you're not a bot" error here regardless of client spoofing --
    that's not something this code can work around. Set YT_COOKIES (a full
    Netscape cookies.txt exported from a logged-in browser session, ideally
    on a throwaway account) to get past it.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    cookie_file = _prepare_cookie_file()
    clients = ["android", "ios", "mweb", "web"]

    info = None
    ydl_used = None
    last_error = "Unknown error."

    for client in clients:
        try:
            ydl = yt_dlp.YoutubeDL(_build_ydl_opts(client, cookie_file))
            info = ydl.extract_info(url, download=False)
            ydl_used = ydl
            break
        except Exception as e:
            last_error = str(e)
            info = None
            continue

    if not info:
        hint = ""
        if "not a bot" in last_error or "Sign in" in last_error:
            if cookie_file:
                hint = (
                    " YT_COOKIES is set but YouTube still rejected the "
                    "request -- the cookies may be expired or the account "
                    "flagged. Re-export fresh cookies from a logged-in "
                    "browser session."
                )
            else:
                hint = (
                    " YouTube is blocking this server's IP. "
                    "Set the YT_COOKIES environment variable to fix this."
                )
        return {"success": False, "error": f"yt-dlp fallback failed: {last_error}{hint}"}

    title = (info.get("title") or "").strip()

    subs = info.get("subtitles", {}).get("en") or info.get("automatic_captions", {}).get("en")
    if not subs:
        return {"success": False, "error": "No transcript/caption available for this video."}

    track = next((s for s in subs if s.get("ext") == "json3"), None)
    if not track:
        return {"success": False, "error": "No json3 caption track available for this video."}

    caption_url = track.get("url")
    if not caption_url:
        return {"success": False, "error": "Caption track had no URL."}

    try:
        raw = ydl_used.urlopen(caption_url).read()
        segments = _parse_json3_captions(raw)
    except Exception as e:
        return {"success": False, "error": f"Failed to download/parse captions: {str(e)}"}

    if not segments:
        return {"success": False, "error": "Transcript came back empty."}

    return {"success": True, "segments": segments, "title": title}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fetch_transcript(video_id: str) -> dict:
    """
    Fetches the transcript (and title, when available) for a video ID.

    Tries yt-dlp first (with cookies if YT_COOKIES is set), then falls back
    to youtube-transcript.ai (free, no key, occasionally unreliable), then
    Supadata as the last resort (not IP-blocked, has a free tier).

    Returns:
        {"success": True, "segments": [{"text": ..., "start": ..., "duration": ...}, ...], "title": "..."}
    ("title" may be "" if none of the sources could supply one -- callers
    should fall back to scraper.py's oEmbed/scrape path in that case.)
    or:
        {"success": False, "error": "<reason>"}
    """
    if not video_id:
        return {"success": False, "error": "Invalid or missing video ID."}

    errors = []

    primary_result = _fetch_from_yt_dlp(video_id)
    if primary_result["success"]:
        return primary_result
    errors.append(f"yt-dlp error: {primary_result['error']}")

    secondary_result = _fetch_from_transcript_ai(video_id)
    if secondary_result["success"]:
        return secondary_result
    errors.append(f"youtube-transcript.ai error: {secondary_result['error']}")

    supadata_result = _fetch_from_supadata(video_id)
    if supadata_result["success"]:
        return supadata_result
    errors.append(f"Supadata error: {supadata_result['error']}")

    return {
        "success": False,
        "error": "Transcript fetch failed. " + " | ".join(errors),
    }


def format_transcript_for_prompt(segments: list, chunk_seconds: int = 30) -> str:
    """
    Converts raw transcript segments into readable chunks with timestamp
    markers, instead of dumping hundreds of tiny fragments into the prompt.

    Groups consecutive segments into ~chunk_seconds-second blocks, each
    prefixed with [start_seconds], so Gemini/Groq has real time-anchors
    to build the valuable_timeline output accurately.
    """
    if not segments:
        return ""

    chunks = []
    current_chunk_text = []
    current_chunk_start = segments[0]["start"]

    for seg in segments:
        if seg["start"] - current_chunk_start >= chunk_seconds and current_chunk_text:
            chunks.append(
                f"[{int(current_chunk_start)}s] " + " ".join(current_chunk_text)
            )
            current_chunk_text = []
            current_chunk_start = seg["start"]

        current_chunk_text.append(seg["text"].strip())

    if current_chunk_text:
        chunks.append(f"[{int(current_chunk_start)}s] " + " ".join(current_chunk_text))

    return "\n".join(chunks)


if __name__ == "__main__":
    test_url = "https://www.youtube.com/watch?v=mvod11IL-Zc"
    vid = extract_video_id(test_url)
    print("Extracted video ID:", vid)

    result = fetch_transcript(vid)
    if result["success"]:
        print("Title:", result.get("title"))
        print(f"Got {len(result['segments'])} segments.")
        print(format_transcript_for_prompt(result["segments"])[:500])
    else:
        print("Error:", result["error"])