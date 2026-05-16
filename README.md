# Voca

> A real-time voice relay pipeline — speech-to-text, operator text-to-speech, AI response, and scam detection — all in one call.

Voca establishes a live audio bridge between a caller and an operator (or AI). The caller speaks into their browser; their words are transcribed live. The operator can type a reply which is instantly spoken back to the caller as natural-sounding speech. On top of that, an AI persona can take over and respond automatically — and every sentence is silently scored by a fine-tuned scam-detection model that fires an alert the moment suspicious language is detected.

---

## Pipelines

### 1 · Speech-to-text (caller → operator)
```
Caller mic  ──►  Gradium STT  ──►  live transcript
                                        │
                               silence detector (1.5 s)
                                        │
                               finalised sentence bubble
```

### 2 · Text-to-speech (operator → caller)
```
Operator types
      │
      ▼ (word by word as they type)
Gradium TTS  ──►  PCM audio stream  ──►  Caller hears reply
```

### 3 · AI persona (automatic response)
```
Finalised sentence
      │
      ▼
OpenAI GPT-4o-mini  ──►  persona reply (as the logged-in user)
      │
Gradium TTS  ──►  audio back to caller
```

### 4 · Scam detection (parallel on every sentence)
```
Finalised sentence
      │
      ▼
Pioneer fine-tuned model  ──►  scam score
      │
      ├── score ≥ threshold  ──►  red alert on operator UI
      └── score < threshold  ──►  silent / clear alert
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python · FastAPI · uvicorn |
| Speech-to-text | Gradium STT (real-time WebSocket stream) |
| Text-to-speech | Gradium TTS (real-time WebSocket stream) |
| AI persona | OpenAI GPT-4o-mini |
| Scam detection | Pioneer AI (fine-tuned classifier) |
| Tunnelling | ngrok |
| Frontend | Vanilla JS · React (no build step) |
| iOS wrapper | Capacitor |
| Async HTTP | httpx |

---

## Prerequisites

- Python 3.11+
- Node.js 18+ (iOS wrapper only)
- Xcode (iOS build only)
- API keys for Gradium, OpenAI, and ngrok

---

## Setup

### 1. Clone

```bash
git clone <repo-url>
cd hackaton_paris
```

### 2. Python environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment variables

Create a `.env` file in the project root:

```env
GRADIUM_API_KEY=your_gradium_key
GRADIUM_VOICE_ID=YTpq7expH9539ERJ   # optional — defaults to this value
NGROK_AUTHTOKEN=your_ngrok_token
OPENAI_API_KEY=your_openai_key

# Optional overrides
AI_MODEL=gpt-4o-mini
SCAM_THRESHOLD=0.45                 # 0–1, lower = more sensitive
```

### 4. Run

```bash
uvicorn server:app --reload --port 8000
```

On startup the server prints public ngrok URLs:

```
  Local caller    → http://localhost:8000/call
  Local operator  → http://localhost:8000/iphone
  Caller          → https://xxxx.ngrok.io/call
  Operator        → https://xxxx.ngrok.io/iphone
```

---

## Usage

| URL | Who | Purpose |
|---|---|---|
| `/` or `/login` | Operator | Login — name, email, age, location |
| `/iphone` | Operator | Main UI — transcript, AI replies, scam alerts |
| `/call` | Caller | Mic page — tap to talk |
| `/type` | Operator (alt) | Classic text-reply UI |

### Demo flow

1. Open `http://localhost:8000` — pre-filled mock data, just click **Continue**
2. Share `/call` with the caller (or open in a second tab)
3. Caller taps the mic and speaks
4. Transcript appears word-by-word on the operator screen
5. Type a reply to speak it back to the caller via TTS, or toggle **AI** for automatic responses
6. If suspicious language is detected, a pulsing red banner fires instantly

---

## Features

- **Real-time STT** — Gradium streams words as they are spoken; a 1.5 s silence window locks each sentence into a bubble
- **Streaming TTS via typing** — words are piped to Gradium TTS as the operator types them, so audio starts before they finish writing
- **AI persona** — GPT-4o-mini responds in first person as the logged-in user, using their name, age, and location; tuned for active listening rather than eager responding
- **Scam detection** — Pioneer fine-tuned model scores every finalised sentence; alert fires at ≥ 45% confidence and auto-clears when the score drops
- **Voice cloning** — upload an audio sample to generate a custom TTS voice for replies
- **Login profile** — user data captured at login is injected into the AI system prompt for personalised responses
- **iOS app** — Capacitor wrapper for native iPhone deployment

---

## Project structure

```
├── server.py               # FastAPI backend — STT · TTS · AI persona · scam detection
├── requirements.txt
├── static/
│   ├── iphone.html         # Operator UI (React, no build step)
│   ├── index.html          # Classic operator UI
│   ├── login.html          # Login / profile capture
│   ├── caller.html         # Caller mic page
│   └── audio-processor.js  # AudioWorklet — mic PCM capture
├── ios/                    # Capacitor iOS project
└── node-server.js          # Node proxy for iOS dev
```

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `GRADIUM_API_KEY` | required | Gradium API key |
| `GRADIUM_VOICE_ID` | `YTpq7expH9539ERJ` | Default TTS voice |
| `NGROK_AUTHTOKEN` | optional | Exposes server publicly via ngrok |
| `OPENAI_API_KEY` | required | GPT-4o-mini access |
| `AI_MODEL` | `gpt-4o-mini` | OpenAI model override |
| `SCAM_THRESHOLD` | `0.45` | Scam alert sensitivity (0–1) |
