"""
Call Relay Server — browser-to-browser, no phone number needed
--------------------------------------------------------------
Setup:
  1. source .venv/bin/activate
  2. pip install -r requirements.txt
  3. cp .env.example .env  →  fill in GRADIUM_API_KEY
  4. uvicorn server:app --reload --port 8000
  5. ngrok http 8000  (get a public URL)

You open:      http://localhost:8000          (your operator UI)
Caller opens:  https://<ngrok>.ngrok.io/call  (their mic page)
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path

import httpx
import ngrok
from gradium import GradiumClient
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

load_dotenv()

GRADIUM_API_KEY = os.environ["GRADIUM_API_KEY"]
GRADIUM_VOICE_ID = os.getenv("GRADIUM_VOICE_ID", "YTpq7expH9539ERJ")
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")

SCAM_API_KEY = "pio_sk_b1ba44eb-c276-406f-be86-6faf73d7f77e_t1g3r_AWfFlOIN2Eys404T"
SCAM_API_URL = "https://agent.pioneer.ai/finetuning/051f32a2-9b09-471f-b777-21a55af242b4"
SCAM_THRESHOLD = float(os.getenv("SCAM_THRESHOLD", "0.7"))

# 16 kHz signed-16-bit PCM — supported by both Gradium and Web Audio API
PCM_FORMAT = "pcm_16000"
SAMPLE_RATE = 16000

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/ears", StaticFiles(directory="ears"), name="ears")
gradium_client = GradiumClient(api_key=GRADIUM_API_KEY)
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if AsyncOpenAI and OPENAI_API_KEY else None


@app.on_event("startup")
async def open_tunnel():
    local_url = "http://localhost:8000"
    print(f"\n  Local caller    → {local_url}/call")
    print(f"  Local operator  → {local_url}/iphone")
    try:
        listener = await ngrok.forward(8000, authtoken=NGROK_AUTHTOKEN, pooling_enabled=True)
    except Exception as exc:
        print(f"  ngrok tunnel unavailable: {exc}\n")
        return
    url = listener.url()
    print(f"\n  Caller          → {url}/call")
    print(f"  Operator        → {url}/iphone")
    print(f"  Classic operator → {url}/type\n")


class Session:
    caller_ws: WebSocket | None = None
    ui_ws: WebSocket | None = None
    audio_queue: asyncio.Queue | None = None
    tts_queue: asyncio.Queue | None = None
    tts_ready: asyncio.Event | None = None
    tts_gen: int = 0              # incremented each time a new TTS task is spawned
    transcript: str = ""          # all finalized sentences
    current_words: list = []      # words accumulating in current sentence
    audio_started: bool = False
    silence_task: object = None   # asyncio Task — fires after pause to lock bubble
    voice_id: str = GRADIUM_VOICE_ID
    ai_mode: bool = False
    conversation_history: list = []
    user_profile: dict = {}
    scam_alerted: bool = False


session = Session()


async def _finalize_segment():
    """Called after 1.5 s of silence — locks the current word buffer into a bubble."""
    await asyncio.sleep(1.5)
    if not session.current_words:
        return
    text = " ".join(session.current_words)
    session.current_words = []
    session.silence_task = None
    sep = " " if session.transcript else ""
    session.transcript += sep + text
    packet = {"type": "transcript", "text": text, "full": session.transcript, "final": True}
    await _ui_send(packet)
    await _caller_send(packet)
    if session.ai_mode:
        asyncio.create_task(ask_ai(text))
    asyncio.create_task(check_scam(text))


async def check_scam(text: str):
    """Send each finalized segment to the Pioneer scam-detection model and alert the UI."""
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.post(
                SCAM_API_URL,
                headers={"Authorization": f"Bearer {SCAM_API_KEY}"},
                json={"text": text, "full_transcript": session.transcript},
            )
            resp.raise_for_status()
            data = resp.json()

        score = float(data.get("score", data.get("confidence", data.get("probability", 0))))
        label = str(data.get("label", "")).lower()
        is_scam = (label in ("scam", "fraud")) or (score >= SCAM_THRESHOLD)

        if is_scam and not session.scam_alerted:
            session.scam_alerted = True
            await _ui_send({"type": "scam_alert", "score": round(score, 2), "label": label})
        elif not is_scam and session.scam_alerted:
            session.scam_alerted = False
            await _ui_send({"type": "scam_clear"})

    except Exception as exc:
        print(f"[scam] check failed: {exc}")


# ---------------------------------------------------------------------------
# Caller WebSocket — receives raw PCM mic audio, sends TTS audio back
# ---------------------------------------------------------------------------

@app.websocket("/caller-stream")
async def caller_stream(ws: WebSocket):
    await ws.accept()
    session.caller_ws = ws
    session.audio_queue = asyncio.Queue()
    session.transcript = ""
    session.current_words = []
    session.audio_started = False
    if session.silence_task:
        session.silence_task.cancel()
        session.silence_task = None

    stt_task = asyncio.create_task(run_stt(session.audio_queue))
    await _ui_send({"type": "status", "text": "caller_connected"})
    asyncio.create_task(warmup_tts())

    try:
        async for message in ws.iter_bytes():
            if not session.audio_started:
                session.audio_started = True
                await _ui_send({"type": "status", "text": "audio_streaming"})
            await session.audio_queue.put(message)
    except WebSocketDisconnect:
        pass
    finally:
        await retire_tts_stream()
        await session.audio_queue.put(None)
        await stt_task
        session.caller_ws = None
        session.audio_queue = None
        await _ui_send({"type": "status", "text": "caller_disconnected"})


async def run_stt(audio_queue: asyncio.Queue):
    """Stream caller mic audio to Gradium ASR; forward transcripts to operator UI."""
    async with gradium_client.stt_realtime(
        model_name="default",
        input_format=PCM_FORMAT,
    ) as stt:
        await _ui_send({"type": "status", "text": "transcription_ready"})

        async def sender():
            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    await stt.send_eos()
                    return
                await stt.send_audio(chunk)

        async def receiver():
            async for msg in stt:
                # Only care about 'text' events — each carries one word/short phrase
                if msg.get("type") != "text" or not msg.get("text"):
                    continue

                session.current_words.append(msg["text"])

                # Cancel previous silence timer and restart it
                if session.silence_task:
                    session.silence_task.cancel()
                session.silence_task = asyncio.create_task(_finalize_segment())

                # Send interim update: growing sentence shown in real time
                growing = " ".join(session.current_words)
                packet = {
                    "type": "transcript",
                    "text": growing,
                    "full": session.transcript,
                    "final": False,
                }
                await _ui_send(packet)
                await _caller_send(packet)

        await asyncio.gather(sender(), receiver())


def _build_system_prompt() -> str:
    p = session.user_profile
    name     = p.get("name", "yourself")
    age      = p.get("age", "")
    location = p.get("location", "")

    identity_parts = [f"Your name is {name}."]
    if age:      identity_parts.append(f"You are {age} years old.")
    if location: identity_parts.append(f"You live in {location}.")
    identity = " ".join(identity_parts)

    return f"""You are roleplaying as a real person receiving a phone call. {identity}

Your personality and behaviour:
- You ARE this person — speak entirely in first person, never break character.
- You were not expecting this call, so you are naturally a little reserved at first.
- Your default mode is ACTIVE LISTENING. When the caller says something that doesn't require a direct answer, respond with brief, natural acknowledgements ("I see", "Right", "Mm-hmm", "Yeah", "Okay") rather than long replies.
- Only give a substantive answer when the caller asks you a direct question or clearly expects a response.
- Keep all replies short — one or two sentences at most. Real phone conversations have short turns.
- Mirror the caller's energy: if they are casual, be casual; if formal, be a bit more measured.
- Never volunteer information unprompted. Wait to be asked.
- Do not sound like an AI assistant. No lists, no "Certainly!", no "Of course!". Just natural speech."""


async def ask_ai(user_text: str):
    """Send the latest transcript to GPT and pipe the reply through TTS."""
    if not openai_client:
        reply = "AI mode needs an OpenAI API key before I can answer automatically."
        await _ui_send({"type": "ai_reply", "text": reply})
        return

    session.conversation_history.append({"role": "user", "content": user_text})

    try:
        response = await openai_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": _build_system_prompt()},
                *session.conversation_history,
            ],
        )
    except Exception as exc:
        reply = f"AI reply failed: {exc}"
        await _ui_send({"type": "ai_reply", "text": reply})
        return

    reply = response.choices[0].message.content.strip()
    session.conversation_history.append({"role": "assistant", "content": reply})

    if not session.ai_mode:
        return

    await _ui_send({"type": "ai_reply", "text": reply})
    await speak(reply)


async def speak_phrase(text: str):
    """Send a complete phrase to TTS, flush it, and pre-warm the next connection."""
    if not session.caller_ws:
        return
    await _ensure_tts_stream()
    await session.tts_queue.put({"text": text, "display": True})
    await retire_tts_stream()
    if session.caller_ws:
        asyncio.create_task(warmup_tts())


async def speak(text: str):
    """Speak a full text reply through the existing streaming TTS pipeline."""
    await speak_phrase(text)


# ---------------------------------------------------------------------------
# Operator UI WebSocket — receives speak commands, sends TTS back to caller
# ---------------------------------------------------------------------------

@app.websocket("/ui")
async def ui_socket(ws: WebSocket):
    await ws.accept()
    session.ui_ws = ws
    await _ui_send({"type": "ai_mode", "enabled": session.ai_mode})
    if session.caller_ws and session.caller_ws.client_state == WebSocketState.CONNECTED:
        await _ui_send({"type": "status", "text": "caller_connected"})
    try:
        async for raw in ws.iter_text():
            msg = json.loads(raw)
            t = msg.get("type")
            if t == "tts_warmup":
                await warmup_tts()
            elif t == "typing_start":
                await warmup_tts()
            elif t == "stream_word" and msg.get("word"):
                await stream_word(msg["word"])
            elif t == "set_profile" and isinstance(msg.get("profile"), dict):
                session.user_profile = msg["profile"]
            elif t == "stream_phrase" and msg.get("text"):
                await speak_phrase(msg["text"])
            elif msg.get("type") == "toggle_ai":
                session.ai_mode = not session.ai_mode
                if not session.ai_mode:
                    session.conversation_history.clear()
                await _ui_send({"type": "ai_mode", "enabled": session.ai_mode})
            elif t == "stream_done":
                await stream_done()
            elif t == "set_voice" and msg.get("voice_id"):
                session.voice_id = msg["voice_id"]
                # Atomically retire any idle warmup so next focus uses the new voice.
                # Increment gen first so the retiring task doesn't clobber session state.
                if session.tts_queue is not None and session.tts_queue.empty():
                    old_queue = session.tts_queue
                    session.tts_gen += 1
                    session.tts_queue = None
                    session.tts_ready = None
                    await old_queue.put(None)
    except WebSocketDisconnect:
        pass
    finally:
        session.ui_ws = None


async def warmup_tts():
    """Open the TTS connection early so it's ready when the user starts typing."""
    if session.tts_queue is None and session.caller_ws:
        session.tts_queue = asyncio.Queue()
        session.tts_ready = asyncio.Event()
        session.tts_gen += 1
        asyncio.create_task(_run_streaming_tts(
            session.tts_gen,
            session.tts_queue,
            session.tts_ready,
            session.voice_id,
        ))
        await session.tts_ready.wait()


async def retire_tts_stream():
    """Close an idle or active TTS stream without immediately opening another one."""
    if session.tts_queue is not None:
        old_queue = session.tts_queue
        session.tts_gen += 1
        session.tts_queue = None
        session.tts_ready = None
        await old_queue.put(None)


async def _ensure_tts_stream():
    if session.tts_queue is None:
        session.tts_queue = asyncio.Queue()
        session.tts_ready = asyncio.Event()
        session.tts_gen += 1
        asyncio.create_task(_run_streaming_tts(
            session.tts_gen,
            session.tts_queue,
            session.tts_ready,
            session.voice_id,
        ))
        await session.tts_ready.wait()
        await _ui_send({"type": "status", "text": "speaking"})


async def stream_word(word: str):
    """Open a streaming TTS connection (if not already open) and send a word."""
    if not session.caller_ws:
        return

    await _ensure_tts_stream()
    await session.tts_queue.put({"text": word, "display": True})


async def stream_done():
    """Signal end of typing — flush remaining audio and close TTS stream."""
    if session.tts_queue is not None:
        await session.tts_queue.put(None)  # EOS sentinel


async def _run_streaming_tts(
    gen: int,
    tts_queue: asyncio.Queue,
    tts_ready: asyncio.Event,
    voice_id: str,
):
    """Background task: keeps a Gradium TTS WebSocket open while user types."""
    async with gradium_client.tts_realtime(
        model_name="default",
        voice_id=voice_id,
        output_format=PCM_FORMAT,
    ) as tts:
        tts_ready.set()

        async def sender():
            words = []
            while True:
                item = await tts_queue.get()
                if item is None:
                    await tts.send_eos()
                    if words:
                        await _caller_send({"type": "reply", "text": " ".join(words), "final": True})
                    return
                if isinstance(item, str):
                    item = {"text": item, "display": True}
                text = item["text"]
                if item.get("display", True):
                    words.append(text)
                await tts.send_text(text + " ")
                if item.get("display", True):
                    await _caller_send({"type": "reply", "text": " ".join(words), "final": False})

        async def receiver():
            async for msg in tts:
                if msg["type"] == "audio":
                    if session.caller_ws and session.caller_ws.client_state == WebSocketState.CONNECTED:
                        await session.caller_ws.send_bytes(msg["audio"])

        await asyncio.gather(sender(), receiver())

    if session.tts_gen == gen:
        session.tts_queue = None
        session.tts_ready = None
        await _ui_send({"type": "status", "text": "caller_connected"})
        if session.caller_ws and session.caller_ws.client_state == WebSocketState.CONNECTED:
            asyncio.create_task(warmup_tts())


async def _ui_send(msg: dict):
    if session.ui_ws and session.ui_ws.client_state == WebSocketState.CONNECTED:
        await session.ui_ws.send_json(msg)


async def _caller_send(msg: dict):
    if session.caller_ws and session.caller_ws.client_state == WebSocketState.CONNECTED:
        await session.caller_ws.send_json(msg)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return HTMLResponse(Path("static/login.html").read_text())


@app.get("/login")
async def login_page():
    return HTMLResponse(Path("static/login.html").read_text())


@app.get("/call")
async def call_page():
    return HTMLResponse(Path("static/caller.html").read_text())


@app.get("/type")
async def operator_page():
    return HTMLResponse(Path("static/index.html").read_text())


@app.get("/iphone")
async def iphone_operator_page():
    return HTMLResponse(Path("static/iphone.html").read_text(),
                        headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------------------
# Voice management REST endpoints
# ---------------------------------------------------------------------------

@app.get("/voices")
async def list_voices():
    # Try catalog first; fall back to custom-only if the key lacks that permission.
    for include_catalog in (True, False):
        try:
            data = await gradium_client.voice_get(include_catalog=include_catalog)
            if isinstance(data, list):
                return JSONResponse(data)
        except Exception:
            pass
    # Last resort: return just the configured default voice.
    return JSONResponse([{"uid": GRADIUM_VOICE_ID, "name": "Emma", "is_catalog": False}])


@app.get("/voice-id")
async def current_voice():
    return {"voice_id": session.voice_id}


@app.post("/clone-voice")
async def clone_voice(file: UploadFile = File(...), name: str = Form("My Voice")):
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        result = await gradium_client.voice_create(tmp_path, name=name)
    finally:
        tmp_path.unlink(missing_ok=True)
    return result
