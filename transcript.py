"""
transcript.py
Handles: extracting a YouTube video ID from any URL format,
and fetching the transcript/caption text (timed segments) for that video.

Uses yt-dlp to pull caption tracks. This hits a different YouTube endpoint
than youtube_transcript_api, so it's less likely to be IP-blocked the same way.
"""

import re
import os
import json
import urllib.request
import yt_dlp


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
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def _parse_json3_captions(raw_bytes: bytes) -> list:
    """
    Parses YouTube's json3 caption format into our segments shape.
    json3 events look like: {"tStartMs": 1000, "dDurationMs": 2000, "segs": [{"utf8": "hello"}]}
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


def fetch_transcript(video_id: str) -> dict:
    """
    Fetches the transcript (caption track) for a given video ID using yt-dlp.
    Tries manually-uploaded subtitles first, falls back to auto-generated captions.

    Returns a dict:
        {"success": True, "segments": [{"text": ..., "start": ..., "duration": ...}, ...]}
    or on failure:
        {"success": False, "error": "<reason>"}
    """
    if not video_id:
        return {"success": False, "error": "Invalid or missing video ID."}

    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        return {"success": False, "error": f"Transcript fetch failed: {str(e)}"}

    # Prefer manual subtitles, fall back to auto-generated ("automatic_captions")
    subs = info.get("subtitles", {}).get("en") or info.get("automatic_captions", {}).get("en")

    if not subs:
        return {"success": False, "error": "No transcript/caption available for this video."}

    # Find a json3 format track if available (easiest to parse); else take the first one
    track = next((s for s in subs if s.get("ext") == "json3"), subs[0])
    caption_url = track.get("url")

    if not caption_url:
        return {"success": False, "error": "Caption track had no URL."}

    try:
        with urllib.request.urlopen(caption_url, timeout=15) as resp:
            raw = resp.read()

        if track.get("ext") == "json3":
            segments = _parse_json3_captions(raw)
        else:
            # Unknown format fallback - shouldn't normally hit this since we prefer json3
            return {"success": False, "error": f"Unsupported caption format: {track.get('ext')}"}

    except Exception as e:
        return {"success": False, "error": f"Failed to download/parse captions: {str(e)}"}

    if not segments:
        return {"success": False, "error": "Transcript came back empty."}

    return {"success": True, "segments": segments}


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
        # If this segment starts a new chunk_seconds window, flush the current one.
        if seg["start"] - current_chunk_start >= chunk_seconds and current_chunk_text:
            chunks.append(
                f"[{int(current_chunk_start)}s] " + " ".join(current_chunk_text)
            )
            current_chunk_text = []
            current_chunk_start = seg["start"]

        current_chunk_text.append(seg["text"].strip())

    # Flush whatever's left.
    if current_chunk_text:
        chunks.append(f"[{int(current_chunk_start)}s] " + " ".join(current_chunk_text))

    return "\n".join(chunks)


# Quick manual test when running this file directly.
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