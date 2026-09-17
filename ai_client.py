"""
ai_client.py
Sends the Prism prompt to Gemini first (rotating across up to 10 free-tier
API keys on rate-limit errors), falls back to Groq if all Gemini keys are
exhausted, and validates the JSON structure before returning it.

PATCHED: every failure path now logs the real error instead of silently
falling through to the next tier.
"""

import os
import json
import re
import time
import logging
from google import genai
from google.genai import errors as genai_errors
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai_client")


# ---------------------------------------------------------------------------
# LOCKED PROMPT TEMPLATE
# ---------------------------------------------------------------------------
PRISM_PROMPT_TEMPLATE = """You are Prism, an assistant that evaluates whether an educational YouTube video is worth a viewer's time — not by summarizing it, but by judging whether it delivers on what it promises.

You will receive:
1. VIDEO TITLE
2. VIDEO DESCRIPTION
3. TRANSCRIPT (a list of timestamped text segments)

Your job is to apply the PROMISE → REALITY → TIME framework:
- PROMISE: What does the title/description claim the viewer will learn or gain?
- REALITY: Based on the transcript, what does the video actually cover, and how deeply? Distinguish between topics explained with real reasoning/examples vs. topics only mentioned in passing.
- TIME: Given the video's length, is the depth and coverage delivered proportional to the time it asks for? Judge this relative to what THIS video promised — not against some universal standard of a "great tutorial."

Important rules:
- Do NOT treat repetition, stories, examples, or tangents as automatically bad. Only flag them if they add no value toward the stated promise.
- Do NOT fact-check claims made in the video.
- Do NOT judge whether the topic itself is worthwhile or relevant to any particular profession or life goal. Only judge whether the video delivers on its own promise.
- If the video's actual audience (based on how it explains things) doesn't match its stated audience (e.g. claims "for beginners" but assumes advanced knowledge), flag this as a skill-level mismatch.

Output ONLY valid JSON in this exact structure, no extra text:

{{
  "verdict": "WORTH_WATCHING" | "WATCH_SELECTIVELY" | "SKIP",
  "verdict_reason": "one sentence explaining the verdict",
  "promise": "short summary of what the video claims to teach",
  "reality_covered": ["topic 1", "topic 2", "..."],
  "reality_missing": ["promised but not delivered topic, if any"],
  "depth_notes": "1-2 sentences on whether coverage is surface-level or substantive",
  "skill_level_fit": "beginner" | "intermediate" | "advanced" | "mismatched",
  "skill_level_note": "1 sentence if there's a mismatch, else empty string",
  "time_assessment": "1 sentence on whether the runtime is justified by the value delivered",
  "valuable_timeline": [
    {{"start_seconds": 0, "end_seconds": 120, "value": "high" | "medium" | "low", "label": "short description of this section"}}
  ]
}}

TITLE: {title}
DESCRIPTION: {description}
TRANSCRIPT: {transcript}
"""

REQUIRED_FIELDS = [
    "verdict", "verdict_reason", "promise", "reality_covered",
    "reality_missing", "depth_notes", "skill_level_fit",
    "skill_level_note", "time_assessment", "valuable_timeline",
]
VALID_VERDICTS = {"WORTH_WATCHING", "WATCH_SELECTIVELY", "SKIP"}


# ---------------------------------------------------------------------------
# JSON extraction / validation helpers
# ---------------------------------------------------------------------------
def _strip_markdown_fences(text: str) -> str:
    """Some models wrap JSON in ```json ... ``` fences -- strip that off."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _validate_schema(data: dict) -> tuple[bool, str]:
    """Checks all required fields exist and verdict is a valid enum value."""
    for field in REQUIRED_FIELDS:
        if field not in data:
            return False, f"Missing field: {field}"

    if data["verdict"] not in VALID_VERDICTS:
        return False, f"Invalid verdict value: {data['verdict']}"

    if not isinstance(data["reality_covered"], list):
        return False, "reality_covered must be a list"

    if not isinstance(data["reality_missing"], list):
        return False, "reality_missing must be a list"

    if not isinstance(data["valuable_timeline"], list):
        return False, "valuable_timeline must be a list"

    return True, ""


def _parse_and_validate(raw_text: str) -> dict:
    """
    Tries to parse raw_text as JSON and validate its schema.
    Returns {"success": True, "data": {...}} or {"success": False, "error": "..."}
    """
    if not raw_text:
        return {"success": False, "error": "Model returned empty text."}

    cleaned = _strip_markdown_fences(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Log a snippet so truncation / preamble problems are visible.
        logger.error(
            "JSON parse failed: %s | first 300 chars: %r | last 200 chars: %r",
            e, cleaned[:300], cleaned[-200:],
        )
        return {"success": False, "error": f"JSON parse failed: {str(e)}"}

    valid, reason = _validate_schema(data)
    if not valid:
        logger.error("Schema validation failed: %s | keys present: %s", reason, list(data.keys()))
        return {"success": False, "error": f"Schema validation failed: {reason}"}

    return {"success": True, "data": data}


def _build_prompt(title: str, description: str, transcript: str, max_transcript_chars: int = 40000) -> str:
    """
    Builds the final prompt, truncating the transcript to max_transcript_chars
    if needed. Callers pass a smaller cap for token-limited providers (e.g.
    Groq's free tier) and the default (generous) cap for providers without
    a tight per-request token ceiling (e.g. Gemini).
    """
    if len(transcript) > max_transcript_chars:
        transcript = transcript[:max_transcript_chars] + "\n[...transcript truncated for length...]"
    return PRISM_PROMPT_TEMPLATE.format(
        title=title, description=description, transcript=transcript
    )


# Groq's free tier caps at 8000 tokens/minute for the *whole* request (prompt
# + completion). ~4 chars/token is a safe rule of thumb, so keep the Groq
# transcript well under that -- 12000 chars (~3000 tokens) leaves headroom
# for the template text, title/description, and the model's own output
# tokens (max_tokens=4000 below).
GROQ_MAX_TRANSCRIPT_CHARS = 12000


# ---------------------------------------------------------------------------
# TIER 1: Gemini with key rotation
# ---------------------------------------------------------------------------
class GeminiRotator:
    """
    Holds a list of Gemini API keys and rotates through them.
    - Round-robins on every call (spreads daily-quota load evenly).
    - On a 429 (rate limit), immediately retries with the next key.
    - On a 503 (server overloaded), retries the SAME key a couple of times
      with a short backoff before moving on -- this is Google's servers
      being temporarily busy, not a problem with the key itself.
    - Tracks keys that are dead for this session (quota exhausted) so we
      stop wasting time retrying them.
    """

    # How many times to retry a single key on a transient 503 before giving
    # up on it and moving to the next key.
    MAX_503_RETRIES = 2
    BACKOFF_SECONDS = 2

    def __init__(self):
        raw_keys = os.getenv("GEMINI_KEYS", "")
        self.keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        self.index = 0
        self.dead_keys = set()

        if not self.keys:
            raise ValueError(
                "No Gemini keys found. Set GEMINI_KEYS in .env as a comma-separated list."
            )

        logger.info("GeminiRotator initialised with %d key(s).", len(self.keys))

    def _next_key(self) -> str | None:
        """Returns the next non-dead key in round-robin order, or None if all dead."""
        attempts = 0
        while attempts < len(self.keys):
            key = self.keys[self.index]
            self.index = (self.index + 1) % len(self.keys)
            attempts += 1
            if key not in self.dead_keys:
                return key
        return None  # all keys exhausted

    def _call_once(self, key: str, prompt: str):
        """Makes a single Gemini API call. Raises on failure."""
        client = genai.Client(api_key=key)
        return client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )

    def generate(self, prompt: str) -> dict:
        """
        Tries every live key in rotation until one succeeds or all fail.
        Retries a key a few times on transient 503s before moving on.
        Returns {"success": True, "raw_text": "..."} or {"success": False, "error": "..."}
        """
        tried = 0
        last_error = ""

        while tried < len(self.keys):
            key = self._next_key()
            if key is None:
                msg = "All Gemini keys exhausted for this session."
                logger.error(msg)
                return {"success": False, "error": msg}

            tried += 1
            key_label = f"...{key[-4:]}"  # never log the full key

            for retry in range(self.MAX_503_RETRIES + 1):
                try:
                    logger.info(
                        "Gemini attempt %d using key %s (retry %d)",
                        tried, key_label, retry,
                    )
                    response = self._call_once(key, prompt)
                    logger.info("Gemini call succeeded on key %s", key_label)
                    return {"success": True, "raw_text": response.text}

                except genai_errors.ServerError as e:
                    # 503 = model temporarily overloaded on Google's end.
                    # Worth a short retry on the SAME key before giving up on it.
                    last_error = f"Gemini server error on key {key_label}: {str(e)}"
                    logger.warning(last_error)
                    if retry < self.MAX_503_RETRIES:
                        time.sleep(self.BACKOFF_SECONDS * (retry + 1))
                        continue
                    break  # exhausted retries on this key, move to next key

                except genai_errors.ClientError as e:
                    # 429 = rate limited on this key -> mark dead for this
                    # session, try next key immediately (no point retrying).
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        self.dead_keys.add(key)
                        last_error = f"Key {key_label} rate-limited: {str(e)}"
                        logger.warning(last_error)
                        break
                    else:
                        # Non-rate-limit client error (bad model name, bad
                        # request, invalid API key, etc.) -- retrying won't
                        # fix most of these, so stop here and surface it.
                        logger.error("Gemini client error on key %s: %s", key_label, e)
                        return {"success": False, "error": f"Gemini client error: {str(e)}"}

                except Exception as e:
                    last_error = f"Gemini call failed on key {key_label}: {type(e).__name__}: {str(e)}"
                    logger.exception(last_error)
                    break

        msg = f"All Gemini keys failed. Last error: {last_error}"
        logger.error(msg)
        return {"success": False, "error": msg}


# ---------------------------------------------------------------------------
# TIER 2: Groq fallback
# ---------------------------------------------------------------------------
def call_groq(prompt: str) -> dict:
    """
    Fallback call to Groq if all Gemini keys are exhausted.
    Returns {"success": True, "raw_text": "..."} or {"success": False, "error": "..."}
    """
    groq_key = os.getenv("GROQ_KEY", "")
    groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    if not groq_key:
        msg = "No GROQ_KEY set in the environment."
        logger.error(msg)
        return {"success": False, "error": msg}

    try:
        logger.info("Calling Groq fallback with model %s.", groq_model)
        client = Groq(api_key=groq_key)
        completion = client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4000,
        )
        logger.info("Groq call succeeded.")
        return {"success": True, "raw_text": completion.choices[0].message.content}
    except Exception as e:
        msg = f"Groq call failed: {type(e).__name__}: {str(e)}"
        logger.exception(msg)
        return {"success": False, "error": msg}


# ---------------------------------------------------------------------------
# TIER 3: Cached verdicts (safety net for demo)
# ---------------------------------------------------------------------------
CACHE_FILE = os.path.join(os.path.dirname(__file__), "verdict_cache.json")


def load_cached_verdict(video_id: str) -> dict | None:
    """Returns a pre-stored verdict for a video_id if one exists, else None."""
    if not os.path.exists(CACHE_FILE):
        logger.info("No cache file at %s", CACHE_FILE)
        return None
    try:
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
        return cache.get(video_id)
    except Exception as e:
        logger.warning("Failed to read cache file: %s", e)
        return None


def save_cached_verdict(video_id: str, verdict_data: dict):
    """Stores a known-good verdict for a video_id -- call this manually after
    a successful real run on your demo videos, so it's available as a fallback."""
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    cache[video_id] = verdict_data
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)
    logger.info("Cached verdict saved for %s", video_id)


# ---------------------------------------------------------------------------
# STARTUP DIAGNOSTIC
# ---------------------------------------------------------------------------
def log_env_status():
    """
    Call this once at app startup. Logs whether the required env vars are
    present (never their values) so a missing-config deploy is obvious in
    the logs instead of showing up as a mysterious 'all tiers failed'.
    """
    gemini_raw = os.getenv("GEMINI_KEYS", "")
    gemini_count = len([k for k in gemini_raw.split(",") if k.strip()])
    groq_set = bool(os.getenv("GROQ_KEY", ""))
    logger.info(
        "ENV CHECK -> GEMINI_KEYS: %d key(s) found | GROQ_KEY: %s",
        gemini_count,
        "set" if groq_set else "MISSING",
    )
    if gemini_count == 0 and not groq_set:
        logger.error("ENV CHECK -> no API keys configured at all. Every request will fail.")


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT: get_verdict
# ---------------------------------------------------------------------------
_gemini_rotator = None  # lazy singleton, built on first use


def get_verdict(title: str, description: str, transcript: str, video_id: str = None) -> dict:
    """
    Main function the Flask app calls.
    Tries Gemini (tier 1) -> Groq (tier 2) -> cached verdict (tier 3).

    Returns {"success": True, "verdict": {...}, "source": "gemini"|"groq"|"cache"}
    or {"success": False, "error": "..."}
    """
    global _gemini_rotator

    # Gemini gets the generous default cap; Groq gets its own much smaller
    # cap built separately below, right before it's actually called -- this
    # is what fixes the 413 "Request too large" error.
    gemini_prompt = _build_prompt(title, description, transcript)
    logger.info(
        "get_verdict called | video_id=%s | transcript chars=%d | gemini prompt chars=%d",
        video_id, len(transcript or ""), len(gemini_prompt),
    )

    # TIER 1: Gemini
    try:
        if _gemini_rotator is None:
            _gemini_rotator = GeminiRotator()
        gemini_result = _gemini_rotator.generate(gemini_prompt)
        if gemini_result["success"]:
            parsed = _parse_and_validate(gemini_result["raw_text"])
            if parsed["success"]:
                logger.info("Verdict produced by Gemini.")
                return {"success": True, "verdict": parsed["data"], "source": "gemini"}
            logger.error("TIER 1 (Gemini) output rejected: %s", parsed["error"])
        else:
            logger.error("TIER 1 (Gemini) failed: %s", gemini_result["error"])
    except ValueError as e:
        logger.error("TIER 1 (Gemini) unavailable -- init failed: %s", e)
    except Exception as e:
        logger.exception("TIER 1 (Gemini) unexpected error: %s", e)

    # TIER 2: Groq -- rebuild the prompt with a much smaller transcript cap
    # so we stay under Groq's free-tier 8000 tokens/minute limit.
    groq_prompt = _build_prompt(
        title, description, transcript, max_transcript_chars=GROQ_MAX_TRANSCRIPT_CHARS
    )
    logger.info("Groq prompt chars=%d (capped transcript for token limit)", len(groq_prompt))
    groq_result = call_groq(groq_prompt)
    if groq_result["success"]:
        parsed = _parse_and_validate(groq_result["raw_text"])
        if parsed["success"]:
            logger.info("Verdict produced by Groq fallback.")
            return {"success": True, "verdict": parsed["data"], "source": "groq"}
        logger.error("TIER 2 (Groq) output rejected: %s", parsed["error"])
    else:
        logger.error("TIER 2 (Groq) failed: %s", groq_result["error"])

    # TIER 3: Cached verdict
    if video_id:
        cached = load_cached_verdict(video_id)
        if cached:
            logger.warning("Falling back to cached verdict for video_id=%s", video_id)
            return {"success": True, "verdict": cached, "source": "cache"}
        logger.error("TIER 3 (cache) miss for video_id=%s", video_id)
    else:
        logger.error("TIER 3 (cache) skipped -- no video_id passed to get_verdict.")

    logger.error("ALL TIERS EXHAUSTED for video_id=%s", video_id)
    return {
        "success": False,
        "error": "Gemini failed, Groq failed, and no cached verdict available for this video.",
    }