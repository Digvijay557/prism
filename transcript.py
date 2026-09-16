"""
transcript.py
Handles: extracting a YouTube video ID from any URL format,
and fetching the transcript/caption text (timed segments) for that video.
"""

import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig
import os
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)


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

def fetch_transcript(video_id: str) -> dict:
    """
    Fetches the transcript (caption track) for a given video ID.
    Works identically whether the caption is auto-generated or
    manually uploaded by the creator -- the API returns the same shape.

    Returns a dict:
        {"success": True, "segments": [{"text": ..., "start": ..., "duration": ...}, ...]}
    or on failure:
        {"success": False, "error": "<reason>"}
    """
    if not video_id:
        return {"success": False, "error": "Invalid or missing video ID."}

    try:
        proxy_username = os.environ["WEBSHARE_USERNAME"]
        proxy_password = os.environ["WEBSHARE_PASSWORD"]
        proxy_url = f"http://{proxy_username}:{proxy_password}@31.59.20.176:6754"

        ytt_api = YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(
                http_url=proxy_url,
                https_url=proxy_url,
            )
        )
        fetched = ytt_api.fetch(video_id)

        segments = [
            {
                "text": snippet.text,
                "start": snippet.start,
                "duration": snippet.duration,
            }
            for snippet in fetched
        ]

        if not segments:
            return {"success": False, "error": "Transcript came back empty."}

        return {"success": True, "segments": segments}

    except TranscriptsDisabled:
        return {"success": False, "error": "Captions are disabled for this video."}
    except NoTranscriptFound:
        return {"success": False, "error": "No transcript/caption available for this video."}
    except VideoUnavailable:
        return {"success": False, "error": "This video is unavailable or private."}
    except Exception as e:
        # Catch-all so a weird library error never crashes the whole request.
        return {"success": False, "error": f"Transcript fetch failed: {str(e)}"}
    """
    Fetches the transcript (caption track) for a given video ID.
    Works identically whether the caption is auto-generated or
    manually uploaded by the creator -- the API returns the same shape.

    Returns a dict:
        {"success": True, "segments": [{"text": ..., "start": ..., "duration": ...}, ...]}
    or on failure:
        {"success": False, "error": "<reason>"}
    """
    if not video_id:
        return {"success": False, "error": "Invalid or missing video ID."}

    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id)

        segments = [
            {
                "text": snippet.text,
                "start": snippet.start,
                "duration": snippet.duration,
            }
            for snippet in fetched
        ]

        if not segments:
            return {"success": False, "error": "Transcript came back empty."}

        return {"success": True, "segments": segments}

    except TranscriptsDisabled:
        return {"success": False, "error": "Captions are disabled for this video."}
    except NoTranscriptFound:
        return {"success": False, "error": "No transcript/caption available for this video."}
    except VideoUnavailable:
        return {"success": False, "error": "This video is unavailable or private."}
    except Exception as e:
        # Catch-all so a weird library error never crashes the whole request.
        return {"success": False, "error": f"Transcript fetch failed: {str(e)}"}


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
