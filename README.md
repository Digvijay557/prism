# Prism — Setup Guide

## IMPORTANT: Run this on YOUR LAPTOP, not any cloud/online sandbox
YouTube blocks cloud/datacenter IPs (AWS, GCP, Render, etc.) from
fetching transcripts and even the page title. Your laptop's normal
home/college WiFi IP is NOT a datacenter IP, so it should work fine there.
This was tested and confirmed blocked from a cloud sandbox — untested
from your actual network yet. Test this FIRST before building anything
else on top.

---

## Step 1: Install Python (if you don't have it)
Check first:
```
python3 --version
```
If that fails, download Python 3.10+ from python.org and install it.

## Step 2: Install dependencies
Open a terminal in this folder and run:
```
pip install -r requirements.txt --break-system-packages
```
(Drop `--break-system-packages` if that flag errors on your system — it's
only needed on some Linux setups.)

## Step 3: Get your Gemini API keys (free, no card)
1. Go to https://aistudio.google.com/apikey
2. Sign in with a Google account
3. Click "Create API Key" — copy it
4. Repeat with up to 9 more Google accounts if you want the full 10-key
   rotation (each account gets its own free-tier quota)

## Step 4: Get a Groq API key (free, no card)
1. Go to https://console.groq.com/keys
2. Sign in, create an API key, copy it

## Step 5: Create your .env file
Copy `.env.example` to a new file named `.env` in this same folder, and
paste your real keys in:
```
GEMINI_KEYS=key1,key2,key3
GROQ_KEY=your_groq_key
```
(Comma-separated, no spaces, no quotes. Even just 1-2 Gemini keys is
fine to start testing — you don't need all 10 immediately.)

## Step 6: Test the transcript + scraper pieces FIRST
Before running the full server, confirm the YouTube-facing pieces work
from your network:
```
python3 transcript.py
python3 scraper.py
```
Both should print real data (a video ID, transcript segments, a title).
If you get a 403 error here too, see "If it's still blocked" below.

## Step 7: Run the backend server
```
python3 app.py
```
You should see it start on `http://0.0.0.0:5000`. Leave this running.

## Step 8: Test the /analyze endpoint
In a NEW terminal window (keep the server running in the first one):
```
curl -X POST http://localhost:5000/analyze -H "Content-Type: application/json" -d "{\"url\": \"https://www.youtube.com/watch?v=SOME_REAL_VIDEO_ID\"}"
```
Replace with a real educational video URL. You should get back JSON with
a verdict. This will take a few seconds (transcript fetch + AI call).

## Step 9: Open the frontend
Open `index.html` directly in your phone/laptop browser (double-click it,
or right-click → Open With → your browser). Paste a YouTube URL, hit
Check. It should call your locally running backend and show verdict cards.

**Note:** `index.html` currently points at `http://localhost:5000` as the
API address (see the `API_BASE` line near the top of the `<script>` tag).
This only works if you're opening index.html on the SAME machine that's
running app.py. Once you set up ngrok (next step), you'll update this to
the ngrok URL instead so it works from your phone too.

## Step 10: Expose your server publicly with ngrok (needed for PWABuilder + phone testing)
1. Go to https://ngrok.com, sign up free, download ngrok for your OS
2. Run: `ngrok http 5000`
3. It gives you a public HTTPS URL like `https://abcd1234.ngrok-free.app`
4. Open `index.html`, find the line `const API_BASE = "http://localhost:5000";`
   and change it to your ngrok URL, e.g. `const API_BASE = "https://abcd1234.ngrok-free.app";`
5. Now host index.html (and manifest.json, icon-192.png, icon-512.png) via
   ngrok too — simplest way: run a second local static server for the
   frontend folder (e.g. `python3 -m http.server 8000` in this folder,
   then `ngrok http 8000` in another terminal) so the frontend ALSO has
   a public HTTPS link. You now have two ngrok tunnels: one for the
   Flask backend, one for the static frontend.

## Step 11: Generate the APK
1. Go to https://www.pwabuilder.com
2. Paste your frontend's ngrok URL (the one serving index.html)
3. Let it scan — it should detect manifest.json and the icons automatically
4. Go to the Android package option, generate and download the APK
5. Transfer the APK to your phone (or download directly on the phone if
   you did this from the phone's browser) and install it (you'll need to
   allow "install from unknown sources" once)

## If it's still blocked (Step 6 fails even on your laptop)
This means your specific network is also flagged. Options, in order of
effort:
1. Try a different network (mobile hotspot vs WiFi, or a friend's WiFi)
2. As a last resort for demo day only: manually run steps 6-8 once on
   your 2-3 demo videos, and save each working verdict using
   `save_cached_verdict()` in `ai_client.py` (see the function at the
   bottom of that file) so your tier-3 cache has real answers ready,
   even if live calls fail during the actual demo.

## Files in this folder
- `transcript.py` — video ID extraction + transcript fetch
- `scraper.py` — title/description scrape
- `ai_client.py` — Gemini rotation + Groq fallback + validation + cache
- `app.py` — Flask server, single /analyze endpoint
- `index.html` — frontend (paste URL, see verdict cards)
- `manifest.json` + `icon-192.png` + `icon-512.png` — needed for PWABuilder
- `.env.example` — copy to `.env` and fill in your real keys
- `verdict_cache.json` — auto-created once you save cached verdicts
