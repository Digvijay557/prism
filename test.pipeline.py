"""
test_pipeline.py
Run this from the Prism project root, on a machine with network access:

    python test_pipeline.py

Checks each layer separately (transcript / title / description) so a
failure tells you WHICH part broke, not just that something did.
No new dependencies -- uses only what transcript.py and scraper.py
already import.
"""

import sys
import time

from transcript import (
    extract_video_id,
    fetch_transcript,
    format_transcript_for_prompt,
)
from scraper import scrape_description

# A few real videos with different shapes. Swap these for whatever you
# actually care about -- the point is variety, not these specific IDs.
TEST_URLS = [
    # long-form talk, definitely has captions + a real description
    "https://www.youtube.com/watch?v=mvod11IL-Zc",
    # short URL format, to exercise extract_video_id's other branch
    "https://youtu.be/dQw4w9WgXcQ",
    # shorts format
    "https://www.youtube.com/shorts/tPEE9ZwTmy0",
]

NO_DESC = "(No description provided by the creator.)"


def check(label, ok, detail=""):
    mark = "PASS" if ok else "FAIL"
    line = f"  [{mark}] {label}"
    if detail:
        line += f" -- {detail}"
    print(line)
    return ok


def test_one(url):
    print(f"\n=== {url} ===")
    results = []

    # --- Layer 1: video ID extraction (pure, no network) ---
    video_id = extract_video_id(url)
    results.append(check("extract_video_id", bool(video_id), video_id or "returned None"))
    if not video_id:
        return results

    # --- Layer 2: transcript fetch ---
    t0 = time.time()
    tr = fetch_transcript(video_id)
    elapsed = time.time() - t0

    if not tr["success"]:
        results.append(check("fetch_transcript", False, tr["error"]))
        return results

    segs = tr["segments"]
    results.append(
        check("fetch_transcript", True, f"{len(segs)} segments in {elapsed:.1f}s")
    )

    # Sanity-check the segment shape -- catches a parser regression where
    # timestamps come back as 0 or the shape drifts.
    shape_ok = all(
        isinstance(s.get("start"), (int, float))
        and isinstance(s.get("duration"), (int, float))
        and isinstance(s.get("text"), str)
        and s["text"].strip()
        for s in segs
    )
    results.append(check("segment shape", shape_ok))

    starts_ascend = all(
        segs[i]["start"] <= segs[i + 1]["start"] for i in range(len(segs) - 1)
    )
    results.append(check("timestamps ascend", starts_ascend))

    nonzero_starts = sum(1 for s in segs if s["start"] > 0)
    results.append(
        check(
            "timestamps parsed (not all zero)",
            nonzero_starts > 0,
            f"{nonzero_starts}/{len(segs)} nonzero",
        )
    )

    # --- Layer 3: title from the transcript header ---
    title_from_transcript = tr.get("title", "")
    results.append(
        check(
            "title from transcript header",
            bool(title_from_transcript),
            title_from_transcript or "EMPTY -- will fall back to scraper/oEmbed",
        )
    )

    # --- Layer 4: description scrape ---
    meta = scrape_description(video_id, fallback_title=title_from_transcript)
    results.append(check("scrape_description returned", meta["success"]))

    desc = meta.get("description", "")
    got_real_desc = bool(desc) and desc != NO_DESC
    results.append(
        check(
            "description is real (not the placeholder)",
            got_real_desc,
            f"{len(desc)} chars" if got_real_desc else "watch-page scrape likely blocked",
        )
    )

    # --- Layer 5: final title precedence, mirroring app.py ---
    final_title = meta.get("title") or title_from_transcript
    results.append(check("final title resolved", bool(final_title), final_title))

    if title_from_transcript and final_title != title_from_transcript:
        check(
            "title precedence (transcript should win)",
            False,
            f"transcript={title_from_transcript!r} final={final_title!r}",
        )
        results.append(False)

    # --- Layer 6: prompt formatting ---
    formatted = format_transcript_for_prompt(segs)
    results.append(
        check(
            "format_transcript_for_prompt",
            bool(formatted.strip()),
            f"{len(formatted)} chars, ~{len(formatted) // 4} tokens",
        )
    )
    if formatted:
        print("    first chunk:", formatted.splitlines()[0][:120])

    return results


def main():
    urls = sys.argv[1:] or TEST_URLS
    all_results = []

    for url in urls:
        try:
            all_results.extend(test_one(url))
        except Exception as e:
            print(f"  [FAIL] UNCAUGHT EXCEPTION -- {type(e).__name__}: {e}")
            all_results.append(False)

    passed = sum(1 for r in all_results if r)
    total = len(all_results)
    print(f"\n{'=' * 40}\n{passed}/{total} checks passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()