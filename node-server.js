import "dotenv/config";

import express from "express";
import http from "http";
import path from "path";
import { fileURLToPath } from "url";
import { WebSocketServer } from "ws";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.NODE_PORT || process.env.PORT || 3000);
const HOST = process.env.NODE_HOST || "127.0.0.1";
const RELAY_PROVIDER = process.env.RELAY_PROVIDER || "browser";
const PROTOCOL_VERSION = "relay.v1";

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ noServer: true });
const sessions = new Map();

app.use("/static", express.static(path.join(__dirname, "static")));

app.get("/", (_req, res) => res.redirect("/iphone"));
app.get("/iphone", (_req, res) => {
  res.sendFile(path.join(__dirname, "static", "iphone.html"));
});
app.get("/call", (_req, res) => {
  res.sendFile(path.join(__dirname, "static", "call.html"));
});
app.get("/health", (_req, res) => {
  res.json({ ok: true, protocol: PROTOCOL_VERSION, relayProvider: RELAY_PROVIDER });
});
app.get("/debug/transcript", (req, res) => {
  const sessionId = req.query.session || "default";
  const text = cleanText(req.query.text || "debug transcript from server");
  const session = getSession(sessionId);
  const entry = appendTranscript(session, text);

  broadcast(session, {
    type: "transcript",
    text,
    full: fullTranscript(session),
    final: true,
    entry,
    source: "debug"
  });

  res.json({ ok: true, sessionId, text, operatorConnected: isOpen(session.sockets.operator) });
});

server.on("upgrade", (request, socket, head) => {
  const url = new URL(request.url, `http://${request.headers.host}`);
  if (url.pathname !== "/ws") {
    socket.destroy();
    return;
  }

  wss.handleUpgrade(request, socket, head, (ws) => {
    wss.emit("connection", ws, request, url);
  });
});

wss.on("connection", (ws, _request, url) => {
  const role = url.searchParams.get("role");
  const sessionId = url.searchParams.get("session") || "default";

  if (!["operator", "caller"].includes(role)) {
    send(ws, { type: "error", message: "role must be operator or caller" });
    ws.close(1008, "invalid role");
    return;
  }

  const session = getSession(sessionId);
  attachSocket(session, role, ws);

  send(ws, {
    type: "hello",
    protocol: PROTOCOL_VERSION,
    role,
    sessionId,
    relayProvider: session.relay.name,
    capabilities: session.relay.capabilities
  });

  if (role === "operator") sendSnapshot(session);
  broadcastPresence(session);

  ws.on("message", async (payload, isBinary) => {
    try {
      if (isBinary) {
        await session.relay.handleAudio(session, role, payload);
        return;
      }

      const message = JSON.parse(payload.toString());
      await handleMessage(session, role, message);
    } catch (error) {
      send(ws, { type: "error", message: error.message });
    }
  });

  ws.on("close", () => {
    if (session.sockets[role] === ws) session.sockets[role] = null;
    if (role === "caller") session.current = "";
    broadcastPresence(session);
  });
});

function getSession(id) {
  if (!sessions.has(id)) {
    sessions.set(id, {
      id,
      sockets: { operator: null, caller: null },
      transcript: [],
      replies: [],
      current: "",
      relay: createRelay(RELAY_PROVIDER)
    });
  }

  return sessions.get(id);
}

function attachSocket(session, role, ws) {
  const oldSocket = session.sockets[role];
  if (oldSocket && oldSocket.readyState === oldSocket.OPEN) {
    send(oldSocket, { type: "status", text: `${role}_replaced` });
    oldSocket.close(1012, "new client connected");
  }
  session.sockets[role] = ws;
}

async function handleMessage(session, role, message) {
  if (message.protocol && message.protocol !== PROTOCOL_VERSION) {
    sendRole(session, role, { type: "error", message: `unsupported protocol ${message.protocol}` });
    return;
  }

  if (role === "caller") {
    await handleCallerMessage(session, message);
    return;
  }

  await handleOperatorMessage(session, message);
}

async function handleCallerMessage(session, message) {
  if (message.type === "call.started") {
    sendRole(session, "operator", { type: "status", text: "caller_connected" });
    return;
  }

  if (message.type === "call.ended") {
    session.current = "";
    sendRole(session, "operator", { type: "status", text: "caller_disconnected" });
    return;
  }

  if (message.type === "transcript.delta") {
    const text = cleanText(message.text);
    if (!text) return;

    session.current = text;
    if (message.final) {
      const entry = appendTranscript(session, text);
      session.current = "";
      broadcast(session, { type: "transcript", text, full: fullTranscript(session), final: true, entry });
      return;
    }

    broadcast(session, {
      type: "transcript",
      text,
      full: fullTranscript(session),
      final: false
    });
  }

  if (message.type === "tts.played") {
    sendRole(session, "operator", { type: "status", text: "reply_played" });
  }
}

async function handleOperatorMessage(session, message) {
  if (message.type === "reply.submit") {
    const text = cleanText(message.text);
    if (!text) return;

    const entry = makeEntry("operator", text);
    session.replies.push(entry);
    sendRole(session, "operator", { type: "reply", text, final: true, entry });
    await session.relay.speak(session, text);
    return;
  }

  if (message.type === "relay.ping") {
    sendRole(session, "operator", {
      type: "relay.pong",
      provider: session.relay.name,
      capabilities: session.relay.capabilities
    });
  }
}

function createRelay(provider) {
  if (provider === "future") return new FutureRelay();
  return new BrowserSpeechRelay();
}

class BrowserSpeechRelay {
  name = "browser";
  capabilities = [
    "browser_speech_recognition",
    "browser_speech_synthesis",
    "operator_text_to_tts",
    "websocket_json_protocol"
  ];

  async speak(session, text) {
    sendRole(session, "operator", { type: "status", text: "speaking" });
    sendRole(session, "caller", { type: "tts.play", text });
  }

  async handleAudio(session, role, payload) {
    sendRole(session, "operator", {
      type: "relay.audio",
      role,
      bytes: payload.length,
      handled: false
    });
  }
}

class FutureRelay {
  name = "future";
  capabilities = [
    "twilio_media_streams_ready",
    "realtime_audio_in",
    "stt_transcript_out",
    "tts_audio_out",
    "websocket_binary_audio"
  ];

  async speak(session, text) {
    sendRole(session, "operator", { type: "status", text: "relay_pending" });
    sendRole(session, "caller", {
      type: "tts.request",
      text,
      provider: this.name
    });
  }

  async handleAudio(session, role, payload) {
    sendRole(session, "operator", {
      type: "relay.audio",
      role,
      bytes: payload.length,
      handled: false,
      next: "wire this to Twilio Media Streams or a realtime STT/TTS provider"
    });
  }
}

function sendSnapshot(session) {
  sendRole(session, "operator", {
    type: "snapshot",
    transcript: session.transcript,
    replies: session.replies,
    current: session.current,
    full: fullTranscript(session)
  });
}

function broadcastPresence(session) {
  const callerConnected = isOpen(session.sockets.caller);
  const operatorConnected = isOpen(session.sockets.operator);
  broadcast(session, {
    type: "presence",
    callerConnected,
    operatorConnected,
    sessionId: session.id
  });
}

function broadcast(session, message) {
  sendRole(session, "operator", message);
  sendRole(session, "caller", message);
}

function sendRole(session, role, message) {
  send(session.sockets[role], message);
}

function send(ws, message) {
  if (!isOpen(ws)) return;
  ws.send(JSON.stringify({ ...message, at: new Date().toISOString() }));
}

function isOpen(ws) {
  return ws && ws.readyState === ws.OPEN;
}

function makeEntry(speaker, text) {
  return {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    speaker,
    text,
    at: new Date().toISOString()
  };
}

function appendTranscript(session, text) {
  const entry = makeEntry("caller", text);
  session.transcript.push(entry);
  return entry;
}

function cleanText(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

function fullTranscript(session) {
  return session.transcript.map((entry) => entry.text).join(" ");
}

server.listen(PORT, HOST, () => {
  console.log(`iPhone relay app: http://${HOST}:${PORT}/iphone`);
  console.log(`Caller page:       http://${HOST}:${PORT}/call`);
  console.log(`Relay provider:    ${RELAY_PROVIDER}`);
});
