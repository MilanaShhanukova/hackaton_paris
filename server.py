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
from pathlib import Path

import ngrok
from gradium import GradiumClient
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

load_dotenv()

GRADIUM_API_KEY = os.environ["GRADIUM_API_KEY"]
GRADIUM_VOICE_ID = os.getenv("GRADIUM_VOICE_ID", "YTpq7expH9539ERJ")
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")

# 16 kHz signed-16-bit PCM — supported by both Gradium and Web Audio API
PCM_FORMAT = "pcm_16000"
SAMPLE_RATE = 16000

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
gradium_client = GradiumClient(api_key=GRADIUM_API_KEY)


@app.on_event("startup")
async def open_tunnel():
    listener = await ngrok.forward(8000, authtoken=NGROK_AUTHTOKEN)
    print(f"\n  Caller link → {listener.url()}/call\n")


class Session:
    caller_ws: WebSocket | None = None
    ui_ws: WebSocket | None = None
    audio_queue: asyncio.Queue | None = None
    transcript: str = ""


session = Session()


# ---------------------------------------------------------------------------
# Caller WebSocket — receives raw PCM mic audio, sends TTS audio back
# ---------------------------------------------------------------------------

@app.websocket("/caller-stream")
async def caller_stream(ws: WebSocket):
    await ws.accept()
    session.caller_ws = ws
    session.audio_queue = asyncio.Queue()
    session.transcript = ""

    stt_task = asyncio.create_task(run_stt(session.audio_queue))
    await _ui_send({"type": "status", "text": "caller_connected"})

    try:
        async for chunk in ws.iter_bytes():
            await session.audio_queue.put(chunk)
    except WebSocketDisconnect:
        pass
    finally:
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

        async def sender():
            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    await stt.send_eos()
                    return
                await stt.send_audio(chunk)

        async def receiver():
            async for msg in stt:
                if msg.get("text"):
                    is_final = msg["type"] == "end_text"
                    if is_final:
                        sep = " " if session.transcript else ""
                        session.transcript += sep + msg["text"]
                    await _ui_send({
                        "type": "transcript",
                        "text": msg["text"],
                        "full": session.transcript,
                        "final": is_final,
                    })

        await asyncio.gather(sender(), receiver())


# ---------------------------------------------------------------------------
# Operator UI WebSocket — receives speak commands, sends TTS back to caller
# ---------------------------------------------------------------------------

@app.websocket("/ui")
async def ui_socket(ws: WebSocket):
    await ws.accept()
    session.ui_ws = ws
    try:
        async for raw in ws.iter_text():
            msg = json.loads(raw)
            if msg.get("type") == "speak" and msg.get("text"):
                await speak(msg["text"])
    except WebSocketDisconnect:
        pass
    finally:
        session.ui_ws = None


async def speak(text: str):
    """Convert text to speech and stream raw PCM audio to the caller's browser."""
    if not session.caller_ws:
        return

    await _ui_send({"type": "status", "text": "speaking"})

    async with gradium_client.tts_realtime(
        model_name="default",
        voice_id=GRADIUM_VOICE_ID,
        output_format=PCM_FORMAT,  # raw int16 PCM, ready for Web Audio API
    ) as tts:
        await tts.send_text(text)
        await tts.send_eos()

        async for msg in tts:
            if msg["type"] == "audio":
                audio: bytes = msg["audio"]  # already decoded bytes by the SDK
                if session.caller_ws and session.caller_ws.client_state == WebSocketState.CONNECTED:
                    await session.caller_ws.send_bytes(audio)

    await _ui_send({"type": "status", "text": "caller_connected"})


async def _ui_send(msg: dict):
    if session.ui_ws and session.ui_ws.client_state == WebSocketState.CONNECTED:
        await session.ui_ws.send_json(msg)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/")
async def operator_page():
    return HTMLResponse(Path("static/index.html").read_text())


@app.get("/call")
async def caller_page():
    return HTMLResponse(Path("static/caller.html").read_text())
