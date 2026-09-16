"""
scraper.py
Fetches the video description for a YouTube video.

Title no longer comes from here -- transcript.py already fetches
youtube-transcript.ai for the transcript, and that response's header
line ("# Transcript: <title>") gives us a reliable title for free,
from a source we know isn't blocked. See transcript.py's
fetch_transcript(), which now returns "title" alongside "segments".

This module is now just a best-effort description source: scrape the
watch page's og:description tag. oEmbed is kept as a title fallback
ONLY for the rare case where transcript.py couldn't get one either
(e.g. the yt-dlp fallback path ran and its info dict had no title).
"""

import json
import urllib.request
import urllib.error
import requests
from bs4 import BeautifulSoup

HEADERS = {
    # A normal browser User-Agent avoids some basic bot-blocking.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Primary source: YouTube oEmbed
# ---------------------------------------------------------------------------

def _fetch_from_oembed(video_id: str) -> dict:
    """
    Tries to fetch just the title via YouTube's oEmbed endpoint.
    Returns {"success": True, "title": "..."} or {"success": False, "error": ...}
    """
    url = (
        "https://www.youtube.com/oembed"
        f"?url=https://www.youtube.com/watch?v={video_id}&format=json"
    )

    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"success": False, "error": f"oEmbed returned HTTP {e.code}"}
    except Exception as e:
        return {"success": False, "error": f"oEmbed request failed: {str(e)}"}

    title = (data.get("title") or "").strip()
    if not title:
        return {"success": False, "error": "oEmbed returned no title."}

    return {"success": True, "title": title}


# ---------------------------------------------------------------------------
# Fallback / enrichment source: scrape the watch page
# ---------------------------------------------------------------------------

def _scrape_watch_page(video_id: str) -> dict:
    """
    Fetches https://www.youtube.com/watch?v=<video_id> and extracts
    og:title / og:description meta tags.

    Returns {"success": True, "title": "...", "description": "..."}
    or {"success": False, "error": "<reason>"}
    """
    url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return {"success": False, "error": f"Failed to fetch video page: {str(e)}"}

    soup = BeautifulSoup(response.text, "html.parser")

    title_tag = soup.find("meta", property="og:title")
    description_tag = soup.find("meta", property="og:description")

    title = title_tag["content"].strip() if title_tag and title_tag.get("content") else ""
    description = (
        description_tag["content"].strip()
        if description_tag and description_tag.get("content")
        else ""
    )

    if not description:
        fallback_desc = soup.find("meta", attrs={"name": "description"})
        if fallback_desc and fallback_desc.get("content"):
            description = fallback_desc["content"].strip()

    if not title and not description:
        return {"success": False, "error": "Watch page scrape returned no usable tags."}

    return {"success": True, "title": title, "description": description}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def scrape_description(video_id: str, fallback_title: str = "") -> dict:
    """
    Gets the video description via a best-effort watch-page scrape.
    Never hard-fails on a missing description -- just notes it.

    fallback_title: pass this in if the caller (app.py) doesn't already
    have a title from transcript.py's header parse. If the watch-page
    scrape doesn't turn one up either, we try oEmbed as a last resort.

    Returns:
        {"success": True, "description": "...", "title": "<best title found, may equal fallback_title>"}
    """
    if not video_id:
        return {"success": False, "error": "Invalid or missing video ID."}

    description = ""
    title = fallback_title

    scrape_result = _scrape_watch_page(video_id)
    if scrape_result["success"]:
        description = scrape_result.get("description", "")
        if not title:
            title = scrape_result.get("title", "")

    if not title:
        oembed_result = _fetch_from_oembed(video_id)
        if oembed_result["success"]:
            title = oembed_result["title"]

    if not description:
        description = "(No description provided by the creator.)"

    return {"success": True, "title": title, "description": description}


# Kept for backwards compatibility with any old callers -- prefer
# scrape_description() plus transcript.py's title going forward.
def scrape_title_and_description(video_id: str) -> dict:
    result = scrape_description(video_id)
    if not result.get("title"):
        return {
            "success": False,
            "error": (
                "Could not extract video title from any source (video may "
                "be private, deleted, or age-restricted)."
            ),
        }
    return result


if __name__ == "__main__":
    test_video_id = "dQw4w9WgXcQ"
    print(scrape_description(test_video_id))