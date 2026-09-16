"""
scraper.py
Fetches the YouTube watch page HTML and pulls the title/description
from meta tags -- no YouTube Data API key needed.
"""

import requests
from bs4 import BeautifulSoup

HEADERS = {
    # A normal browser User-Agent avoids some basic bot-blocking.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def scrape_title_and_description(video_id: str) -> dict:
    """
    Fetches https://www.youtube.com/watch?v=<video_id> and extracts
    og:title / og:description meta tags.

    Returns:
        {"success": True, "title": "...", "description": "..."}
    or on failure (page didn't load, or tags came back empty -- we fail
    LOUD here rather than silently sending Gemini blank fields):
        {"success": False, "error": "<reason>"}
    """
    if not video_id:
        return {"success": False, "error": "Invalid or missing video ID."}

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

    # Fallback: some pages don't have og:description but do have the
    # standard meta description tag.
    if not description:
        fallback_desc = soup.find("meta", attrs={"name": "description"})
        if fallback_desc and fallback_desc.get("content"):
            description = fallback_desc["content"].strip()

    # Hard check: never silently pass blank fields downstream to Gemini.
    if not title:
        return {
            "success": False,
            "error": "Could not extract video title (page may have changed structure or blocked the request).",
        }

    # Description can legitimately be empty for some videos -- that's fine,
    # just note it so the prompt doesn't imply info that doesn't exist.
    if not description:
        description = "(No description provided by the creator.)"

    return {"success": True, "title": title, "description": description}


# Quick manual test when running this file directly.
if __name__ == "__main__":
    test_video_id = "dQw4w9WgXcQ"
    result = scrape_title_and_description(test_video_id)
    print(result)
