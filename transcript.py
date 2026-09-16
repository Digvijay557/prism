"""
transcript.py
Handles: extracting a YouTube video ID from any URL format,
and fetching the transcript/caption text (timed segments) for that video.

Uses yt-dlp. On servers (Render etc.) YouTube often demands sign-in, so this
supports an optional cookies file supplied via the YT_COOKIES env var.
"""

import re
import os
import json
import yt_dlp

COOKIE_PATH = "/tmp/yt_cookies.txt"


def _prepare_cookie_file() -> str | None:
    """
    If the YT_COOKIES env var is set (full Netscape cookies.txt contents),
    write it to a temp file and return the path. Otherwise return None.
    """
    raw = os.environ.get("YT_COOKIES", "").strip()
    if not raw:
        return None

    try:
        # Render's env vars sometimes collapse newlines; repair the common case.
        if "\\n" in raw and "\n" not in raw:
            raw = raw.replace("\\n", "\n")
        with open(COOKIE_PATH, "w", encoding="utf-8") as f:
            f.write(raw + "\n")
        return COOKIE_PATH
    except Exception:
        return None


def extract_video_id(url: str) -> str | None:
    """
    Pulls the 11-character YouTube video ID out of any common URL format.
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


def _build_opts(client: str, cookie_file: str | None) -> dict:
    """Builds yt-dlp options for one attempt with a specific player client."""
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
    return opts


def fetch_transcript(video_id: str) -> dict:
    """
    Fetches the transcript for a video ID using yt-dlp.

    Tries several YouTube "player clients" in order, because some are checked
    for bot-detection far more aggressively than others. Uses cookies if
    available. Returns:
        {"success": True, "segments": [...]}
    or:
        {"success": False, "error": "<reason>"}
    """
    if not video_id:
        return {"success": False, "error": "Invalid or missing video ID."}

    url = f"https://www.youtube.com/watch?v={video_id}"
    cookie_file = _prepare_cookie_file()

    # Order matters: these tend to be least-to-most likely to trigger a bot check.
    clients = ["android", "ios", "mweb", "web"]

    info = None
    ydl_used = None
    last_error = "Unknown error."

    for client in clients:
        try:
            ydl = yt_dlp.YoutubeDL(_build_opts(client, cookie_file))
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
            hint = (
                " YouTube is blocking this server's IP. "
                "Set the YT_COOKIES environment variable to fix this."
            )
        return {"success": False, "error": f"Transcript fetch failed: {last_error}{hint}"}

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
        # Use yt-dlp's own opener so cookies/headers carry over.
        raw = ydl_used.urlopen(caption_url).read()
        segments = _parse_json3_captions(raw)
    except Exception as e:
        return {"success": False, "error": f"Failed to download/parse captions: {str(e)}"}

    if not segments:
        return {"success": False, "error": "Transcript came back empty."}

    return {"success": True, "segments": segments}


def format_transcript_for_prompt(segments: list, chunk_seconds: int = 30) -> str:
    """
    Converts raw transcript segments into readable ~chunk_seconds blocks,
    each prefixed with [start_seconds], so the model has real time anchors.
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
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    vid = extract_video_id(test_url)
    print("Extracted video ID:", vid)

    result = fetch_transcript(vid)
    if result["success"]:
        print(f"Got {len(result['segments'])} segments.")
        print(format_transcript_for_prompt(result["segments"])[:500])
    else:
        print("Error:", result["error"])