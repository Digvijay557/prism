"""
app.py
Single Flask endpoint: /analyze
Takes a YouTube URL, runs the full pipeline, returns the verdict JSON.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask import render_template
from transcript import extract_video_id, fetch_transcript, format_transcript_for_prompt
from scraper import scrape_description
from ai_client import get_verdict

app = Flask(__name__)
CORS(app)  # allow the frontend (served from anywhere) to call this API

@app.route("/")
def home():
    return render_template("index.html")


@app.route('/.well-known/assetlinks.json')
def assetlinks():
    return send_from_directory('static/.well-known', 'assetlinks.json')

@app.route("/analyze", methods=["POST"])
def analyze():
    body = request.get_json(silent=True) or {}
    url = body.get("url", "").strip()

    if not url:
        return jsonify({"success": False, "error": "No URL provided."}), 400

    # Step 1: extract video ID
    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({
            "success": False,
            "error": "Could not extract a valid YouTube video ID from that URL."
        }), 400

    # Step 2: fetch transcript (this also gives us the title, straight out
    # of youtube-transcript.ai's response header -- no extra request needed)
    transcript_result = fetch_transcript(video_id)
    if not transcript_result["success"]:
        return jsonify({"success": False, "error": transcript_result["error"]}), 422

    formatted_transcript = format_transcript_for_prompt(transcript_result["segments"])
    title_from_transcript = transcript_result.get("title", "")
    duration_seconds = max(
        (
            segment.get("start", 0) + segment.get("duration", 0)
            for segment in transcript_result["segments"]
        ),
        default=0,
    )

    # Step 3: get the description (best-effort watch-page scrape). Pass
    # along the title we already have so scraper.py only needs to find
    # one itself (via oEmbed) if transcript.py came up empty.
    meta_result = scrape_description(video_id, fallback_title=title_from_transcript)
    if not meta_result["success"]:
        return jsonify({"success": False, "error": meta_result["error"]}), 422

    title = meta_result.get("title") or title_from_transcript
    if not title:
        return jsonify({
            "success": False,
            "error": "Could not determine the video title from any source."
        }), 422

    # Step 4-9: build prompt, call Gemini -> Groq -> cache, validate
    verdict_result = get_verdict(
        title=title,
        description=meta_result["description"],
        transcript=formatted_transcript,
        video_id=video_id,
        duration_seconds=duration_seconds,
    )

    if not verdict_result["success"]:
        return jsonify({"success": False, "error": verdict_result["error"]}), 502

    # Step 10: return to frontend
    return jsonify({
        "success": True,
        "video_id": video_id,
        "title": title,
        "verdict": verdict_result["verdict"],
        "source": verdict_result["source"],  # "gemini" | "groq" | "cache" -- useful for your own debugging
    })


@app.route("/health", methods=["GET"])
def health():
    """Quick check that the server is up -- hit this first when testing."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # host="0.0.0.0" so it's reachable via ngrok/local network, not just localhost.
    app.run(host="0.0.0.0", port=5000, debug=True)