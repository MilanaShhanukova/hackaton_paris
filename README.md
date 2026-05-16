# Call Relay iPhone App

This repo now has two runnable versions:

- `server.py`: the original FastAPI/Gradium browser relay.
- `node-server.js`: a Node.js + Express + WebSocket version with an iPhone-style operator UI.

## FastAPI / Gradium iPhone Version

This is the version that uses Gradium STT/TTS for the nicer speech.

```bash
cp .env.example .env
# Fill in GRADIUM_API_KEY
uvicorn server:app --reload --port 8000
```

Open:

- Operator iPhone UI: `http://localhost:8000/iphone`
- Caller page: `http://localhost:8000/call`
- Classic operator UI: `http://localhost:8000/type`

The iPhone UI connects to `/ui` and sends `stream_word` / `stream_done`, so typed replies are spoken through the Gradium TTS stream to the caller page.

## Native iOS Shell

This project is pinned to Capacitor 7 so it works with Node 20.

```bash
npm install
npx cap sync ios
npx cap open ios
```

If `cap sync ios` says Xcode or CocoaPods is missing, install full Xcode from the App Store, open it once, then run:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
sudo gem install cocoapods
npx cap sync ios
npx cap open ios
```

For the iOS simulator, `capacitor.config.json` can point at `http://localhost:8000/iphone`. For a real iPhone, change `server.url` to your Mac LAN IP or an HTTPS ngrok URL, for example `https://your-url.ngrok.app/iphone`.

## Node iPhone Version

Install dependencies:

```bash
npm install
```

Run it:

```bash
npm run dev
```

Open:

- Operator iPhone UI: `http://localhost:3000/iphone`
- Caller page: `http://localhost:3000/call`

The caller page uses browser speech recognition for live transcript and browser speech synthesis for spoken replies. The backend protocol is intentionally relay-shaped so a future provider can handle binary audio, STT, TTS audio, and Twilio-style media streams.

For live speech transcription, open `/call` in Chrome or Safari and allow microphone access. If the caller is on a different phone, serve the app over HTTPS, because browser speech/microphone APIs are often blocked on plain HTTP except for localhost.

## Relay Modes

Set `RELAY_PROVIDER` in `.env`:

- `browser`: working local mode using browser speech APIs.
- `future`: placeholder mode that exposes the future audio/Twilio capabilities in the WebSocket protocol.
