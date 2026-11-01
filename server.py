"""
Rachel → Warudo Bridge + TTS Server (FastMCP)

All voice synthesis lives here so Rachel can speak both:
  • as a response to user input (called via HTTP from stt-server.py)
  • proactively during heartbeats (called as an MCP tool by the LLM)

MCP endpoint : http://localhost:8050/mcp
Speak HTTP   : POST http://localhost:8051/speak  {"text":"...", "expression":"...", "animation":"..."}

Install:
    pip install fastmcp websocket-client httpx pillow chatterbox-tts sounddevice torch numpy
"""

import os
import site

# ── DLL PATH INJECTION (must precede any torch import) ────
_site_paths = site.getsitepackages()
for _p in _site_paths:
    for _lib in ["cudnn", "cublas"]:
        _dll = os.path.join(_p, "nvidia", _lib, "bin")
        if os.path.exists(_dll):
            os.add_dll_directory(_dll)
            os.environ["PATH"] = _dll + os.pathsep + os.environ.get("PATH", "")
            print(f"✅ Linked {_lib} DLLs from: {_dll}")

import base64
import io
import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import numpy as np
import requests
import torch
import wave
import websocket as ws_client
from fastmcp import FastMCP

from config import (
    WARUDO_WS_URL,
    VISION_URL, VISION_MODEL,
    VOICE_REF_PATH, TTS_DEVICE,
    TTS_SPEAK_PORT, OBS_BRIDGE_PORT,
    EMBED_URL, MEMORY_DB_PATH,
)
from memory import RachelMemory

EXPRESSION_EXAGGERATION = {
    "neutral":    0.3,
    "thinking":   0.4,
    "joy":        0.7,
    "anger":      0.5,
    "sadness":    0.2,
    "surprise":   0.7,
    "concern":    0.3,
    "disgused": 0.8,
}

# Valid Chatterbox Turbo vocal tags → Warudo facial expression
VOCAL_TAG_TO_EXPRESSION = {
    "clear throat": "neutral",
    "sigh":         "sadness",
    "shush":        "neutral",
    "cough":        "concern",
    "groan":        "anger",
    "sniff":        "sadness",
    "gasp":         "surprise",
    "chuckle":      "joy",
    "laugh":        "joy",
}

# Warudo expression names → nearest Chatterbox Turbo vocal tag
EXPRESSION_TO_VOCAL = {
    "joy":        "[chuckle]",
    "joy":        "[laugh]",
    "surprise":   "[gasp]",
    "sadness":    "[sigh]",
    "concern":    "[sigh]",
    "anger":      "[groan]",
    "thinking":   "[clear throat]",
    "neutral":    "",
}

# Warudo animation names the LLM may wrap in brackets
WARUDO_ANIM_NAMES = {"wave", "nod", "shrug", "head_tilt", "bounce", "dance_1", "dance_2", "clap"}

_ALL_TAG_RE      = re.compile(r'\[([^\]]+)\]', re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+')

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ─── Warudo WebSocket ──────────────────────────────────────

def send_warudo_action(action: str, data: str = None) -> dict:
    payload = {"action": action}
    if data is not None:
        payload["data"] = data
    try:
        conn = ws_client.create_connection(WARUDO_WS_URL, timeout=5)
        conn.send(json.dumps(payload))
        conn.close()
        return {"status": "ok", "sent": payload}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ─── TTS audio ship-to-bridge ─────────────────────────────
# Playback is owned by obs_bridge; server.py synthesises PCM and POSTs it.

_OBS_PLAY_URL = f"http://localhost:{OBS_BRIDGE_PORT}/play"


def _post_play_to_bridge(audio_np: np.ndarray, sample_rate: int):
    """Encode float32 mono PCM as WAV, POST to obs_bridge /play, block until done."""
    audio_i16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())
    wav_bytes = buf.getvalue()
    timeout = len(audio_np) / sample_rate + 15.0
    try:
        resp = requests.post(
            _OBS_PLAY_URL,
            data=wav_bytes,
            headers={"Content-Type": "audio/wav"},
            timeout=timeout,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"   🔴 _post_play_to_bridge: {e}")


# ─── Bracket Tag Processing ───────────────────────────────

def process_bracket_tags(text: str) -> tuple[str, str | None, str | None]:
    """
    Translate ALL bracket tags in LLM output for TTS and Warudo.

    - Valid Chatterbox vocal tag ([gasp], [laugh], …) → kept for TTS,
      mapped to Warudo expression.
    - Warudo expression name ([joy], [surprise], …) → replaced with the nearest
      Chatterbox vocal tag, used as Warudo expression.
    - Warudo animation name ([nod], [wave], …) → stripped, returned as animation.
    - Unknown tags → stripped.

    Returns (processed_text, warudo_expression | None, warudo_animation | None).
    """
    expression = None
    animation  = None

    def replace(m: re.Match) -> str:
        nonlocal expression, animation
        t = m.group(1).lower().strip()

        if t in VOCAL_TAG_TO_EXPRESSION:
            if expression is None:
                expression = VOCAL_TAG_TO_EXPRESSION[t]
            return m.group(0)                 # keep vocal tag for Chatterbox

        if t in EXPRESSION_TO_VOCAL:
            if expression is None:
                expression = t
            return EXPRESSION_TO_VOCAL[t]     # swap expression name → vocal tag

        if t in WARUDO_ANIM_NAMES:
            if animation is None:
                animation = t
            return ""                          # strip; animation fired at playback

        return ""                              # unknown tag — strip

    processed = _ALL_TAG_RE.sub(replace, text)
    processed = re.sub(r"  +", " ", processed).strip()
    return processed, expression, animation


# ─── Chatterbox Turbo TTS ─────────────────────────────────

class RachelTTS:
    """
    Wraps Chatterbox Turbo for Rachel's voice.
    Loaded once at startup; shared by all callers via the module-level _tts.
    """

    def __init__(self, device: str = TTS_DEVICE, voice_ref: str = VOICE_REF_PATH, skip_model: bool = False):
        self.voice_ref   = voice_ref
        self._gen_lock   = threading.Lock()   # GPU model is not re-entrant

        if skip_model:
            self.model       = None
            self.sample_rate = 24_000
            print("⚡ TTS offload mode — skipping model load")
        else:
            print("Loading Chatterbox Turbo TTS...")

            import torch.nn as nn
            import torchaudio
            import librosa

            # Force LSTMs to cast input to weight dtype (fixes RNN dtype mismatch)
            _orig_lstm = nn.LSTM.forward
            def _safe_lstm(self, input, *args, **kwargs):
                if isinstance(input, torch.Tensor):
                    input = input.to(self.weight_ih_l0.dtype)
                return _orig_lstm(self, input, *args, **kwargs)
            nn.LSTM.forward = _safe_lstm

            # Normalise STFT output to complex64
            _orig_stft = torch.stft
            def _float32_stft(*args, **kwargs):
                r = _orig_stft(*args, **kwargs)
                return r.to(torch.complex64) if r.is_complex() else r.float()
            torch.stft = _float32_stft

            # Force torchaudio.load to float32
            _orig_load = torchaudio.load
            def _float32_load(*args, **kwargs):
                wav, sr = _orig_load(*args, **kwargs)
                return wav.float(), sr
            torchaudio.load = _float32_load

            # Force librosa mel filters to float32
            _orig_mel = librosa.filters.mel
            def _float32_mel(*args, **kwargs):
                return _orig_mel(*args, **kwargs).astype(np.float32)
            librosa.filters.mel = _float32_mel

            torch.set_default_dtype(torch.float32)
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            self.model       = ChatterboxTurboTTS.from_pretrained(device=device)
            self.sample_rate = self.model.sr

            torch.set_default_dtype(torch.float32)
            print(f"✅ Chatterbox Turbo Ready (sr={self.sample_rate})")
            print("   Audio playback handled by obs_bridge /play")

    def generate_audio(self, text: str) -> np.ndarray:
        with self._gen_lock:
            wav = self.model.generate(text, audio_prompt_path=self.voice_ref)
            return wav.squeeze(0).cpu().float().numpy()


# ─── Clip API (orchestrator playback.py sequencer) ───────
# /synth synthesises ONE clip (a Speak chunk, or a vocal tag alone like "[sigh]")
# and caches it; /play-clip plays it and returns when playback has ended, so the
# sequencer measures real clip ends. Durations are samples / sample_rate.

_clips: "OrderedDict[str, tuple[np.ndarray, int]]" = OrderedDict()
_clips_lock = threading.Lock()
_MAX_CLIPS = 64


def synthesize_clip(text: str) -> tuple[np.ndarray, int]:
    if _offload_tts_url:
        r = requests.post(_offload_tts_url, json={"text": text}, timeout=60)
        r.raise_for_status()
        data = r.json()
        audio = np.frombuffer(base64.b64decode(data["audio_b64"]), dtype=np.float32)
        return audio, int(data.get("sample_rate", _tts.sample_rate))
    return _tts.generate_audio(text), _tts.sample_rate


def store_clip(audio: np.ndarray, sr: int) -> str:
    clip_id = uuid.uuid4().hex[:12]
    with _clips_lock:
        _clips[clip_id] = (audio, sr)
        while len(_clips) > _MAX_CLIPS:
            _clips.popitem(last=False)
    return clip_id


# ─── Speak Pipeline ──────────────────────────────────────

def speak_sync(text: str, expression: str | None = None, animation: str | None = None,
               reset: str | None = "neutral"):
    """
    Full TTS pipeline. Runs in a daemon thread (or inline for /speak wait=true).
    reset: expression to return to after playback; None keeps the current one
    (the frames path now uses /synth + /play-clip via playback.py instead).

    1. Split text into sentences.
    2. Submit each sentence to TTS generation (single GPU worker, concurrent
       with sentence splitting).
    3. Concatenate all audio after generation completes.
    4. Fire Warudo expression + animation + play audio simultaneously.
    """
    if _tts is None:
        print("   🔴 speak_sync: TTS not initialised")
        return

    text = text.strip()
    if not text:
        return

    if _offload_tts_url:
        # ── Offload path: process tags locally, generate audio remotely ──
        parts = []
        tag_expr = expression
        tag_anim = animation
        for part in (_SENTENCE_END_RE.split(text) or [text]):
            part = part.strip()
            if not part:
                continue
            processed, s_expr, s_anim = process_bracket_tags(part)
            processed = processed.strip()
            if not processed:
                continue
            if tag_expr is None and s_expr:
                tag_expr = s_expr
            if tag_anim is None and s_anim:
                tag_anim = s_anim
            parts.append(processed)
        full_text = " ".join(parts)
        if not full_text:
            return
        final_expr = tag_expr or "neutral"
        print(f"\n   🎭 {final_expr} | 🕺 {tag_anim or 'none'}  [offload]")
        threading.Thread(target=send_warudo_action, args=("set_expression", final_expr), daemon=True).start()
        if tag_anim:
            threading.Thread(target=send_warudo_action, args=("play_animation", tag_anim), daemon=True).start()
        try:
            r = requests.post(_offload_tts_url, json={"text": full_text}, timeout=60)
            r.raise_for_status()
            data = r.json()
            audio_np = np.frombuffer(base64.b64decode(data["audio_b64"]), dtype=np.float32)
            sr = int(data.get("sample_rate", _tts.sample_rate))
            _post_play_to_bridge(audio_np, sr)
        except Exception as e:
            print(f"   🔴 Offload TTS error: {e}")
        if reset:
            threading.Thread(target=send_warudo_action, args=("set_expression", reset), daemon=True).start()
        print("   😶 [done]")
        return

    executor = ThreadPoolExecutor(max_workers=1)
    pending  = []
    tag_expr = expression
    tag_anim = animation
    order    = 0

    for part in (_SENTENCE_END_RE.split(text) or [text]):
        part = part.strip()
        if not part:
            continue
        processed, s_expr, s_anim = process_bracket_tags(part)
        processed = processed.strip()
        if not processed:
            continue
        if tag_expr is None and s_expr:
            tag_expr = s_expr
        if tag_anim is None and s_anim:
            tag_anim = s_anim
        pending.append((order, executor.submit(_tts.generate_audio, processed)))
        print(f"\n   ⚡ [{order + 1}] {processed[:70]}")
        order += 1

    executor.shutdown(wait=True)

    if not pending:
        return

    chunks = [fut.result() for _, fut in sorted(pending, key=lambda x: x[0])]
    audio  = np.concatenate(chunks, axis=-1)

    final_expr = tag_expr or "neutral"
    print(f"\n   🎭 {final_expr} | 🕺 {tag_anim or 'none'}")

    threading.Thread(
        target=send_warudo_action, args=("set_expression", final_expr), daemon=True
    ).start()
    if tag_anim:
        threading.Thread(
            target=send_warudo_action, args=("play_animation", tag_anim), daemon=True
        ).start()
    _post_play_to_bridge(audio, _tts.sample_rate)

    if reset:
        threading.Thread(
            target=send_warudo_action, args=("set_expression", reset), daemon=True
        ).start()
    print("   😶 [done]")


# ─── HTTP Speak Server (called by stt-server.py) ─────────

def start_speak_server():
    """
    HTTP server on TTS_SPEAK_PORT.

    POST /speak                  {"text", "expression", "animation", "wait"?, "reset"?}
    POST /avatar                 {"expression"?, "animation"?}  (nothing spoken)
    POST /synth                  {"text"} → {"clip_id", "duration_s", "samples", "sample_rate"}
    POST /play-clip              {"clip_id"} → plays, returns when playback has ended
    POST /memory/recall          {"query", "emotion"?, "n_results"?}
    POST /memory/write           {"text", "emotion", "intensity"}
    POST /memory/dream           {}
    GET  /memory/stats
    """
    class _Handler(BaseHTTPRequestHandler):
        def _read_body(self):
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length)) if length else {}

        def _respond(self, data: dict, status: int = 200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                if self.path == "/memory/stats":
                    self._respond(_mem.stats() if _mem else {"error": "memory not ready"})
                else:
                    self._respond({"error": "not found"}, 404)
            except Exception as e:
                self._respond({"error": str(e)}, 500)

        def do_POST(self):
            try:
                body = self._read_body()

                if self.path == "/speak":
                    text       = body.get("text", "").strip()
                    expression = body.get("expression") or None
                    animation  = body.get("animation")  or None
                    # wait=true: synthesise + play before responding (orchestrator
                    # callers that need strict ordering; the frames path uses /synth + /play-clip).
                    wait       = bool(body.get("wait", False))
                    reset      = body["reset"] if "reset" in body else "neutral"
                    if wait:
                        t0 = time.time()
                        if text:
                            speak_sync(text, expression, animation, reset)
                        self._respond({"status": "ok", "seconds": round(time.time() - t0, 3)})
                    else:
                        self._respond({"status": "ok"})
                        if text:
                            threading.Thread(
                                target=speak_sync, args=(text, expression, animation, reset), daemon=True
                            ).start()

                elif self.path == "/synth":
                    text = body.get("text", "").strip()
                    if not text or _tts is None:
                        self._respond({"error": "empty text or TTS not ready"}, 400)
                        return
                    audio, sr = synthesize_clip(text)
                    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                    if audio.size == 0:
                        self._respond({"error": "empty clip"}, 500)
                        return
                    self._respond({"clip_id": store_clip(audio, sr), "samples": int(audio.size),
                                   "sample_rate": sr, "duration_s": round(audio.size / sr, 4)})

                elif self.path == "/play-clip":
                    with _clips_lock:
                        clip = _clips.pop(body.get("clip_id", ""), None)
                    if clip is None:
                        self._respond({"error": "unknown clip_id"}, 404)
                        return
                    audio, sr = clip
                    _post_play_to_bridge(audio, sr)   # blocks until obs_bridge finishes playback
                    self._respond({"status": "ok", "duration_s": round(audio.size / sr, 4)})

                elif self.path == "/avatar":
                    # Avatar-only action, nothing spoken: {"expression"?, "animation"?}
                    expression = body.get("expression") or None
                    animation  = body.get("animation")  or None
                    if expression:
                        send_warudo_action("set_expression", expression)
                    if animation:
                        send_warudo_action("play_animation", animation)
                    self._respond({"status": "ok"})

                elif self.path == "/memory/recall":
                    if not _mem:
                        self._respond({"error": "memory not ready"}, 503)
                        return
                    query    = body.get("query", "")
                    emotion  = body.get("emotion") or None
                    n        = int(body.get("n_results", 3))
                    results  = _mem.recall(query, n_results=n, current_emotion=emotion)
                    self._respond({"memories": [
                        {
                            "text":      r["text"],
                            "emotion":   r["emotion"],
                            "intensity": r["intensity"],
                            "age_days":  r["age_days"],
                            "score":     r["score"],
                        }
                        for r in results
                    ]})

                elif self.path == "/memory/write":
                    if not _mem:
                        self._respond({"error": "memory not ready"}, 503)
                        return
                    text      = body.get("text", "").strip()
                    emotion   = body.get("emotion",   "neutral")
                    intensity = body.get("intensity", "medium")
                    if not text:
                        self._respond({"error": "text required"}, 400)
                        return
                    mid = _mem.encode(text, emotion=emotion, intensity=intensity)
                    self._respond({"id": mid, "ok": True})

                elif self.path == "/memory/dream":
                    if not _mem:
                        self._respond({"error": "memory not ready"}, 503)
                        return
                    stats = _mem.dream(verbose=True)
                    self._respond(stats)

                else:
                    self._respond({"error": "not found"}, 404)

            except Exception as e:
                self._respond({"error": str(e)}, 500)

        def log_message(self, *_):
            pass

    # Threading server: /speak wait=true blocks for the whole utterance and must
    # not stall /avatar or the memory API meanwhile.
    srv = ThreadingHTTPServer(("localhost", TTS_SPEAK_PORT), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"✅ Speak server: http://localhost:{TTS_SPEAK_PORT}/speak")
    print(f"   Memory API : http://localhost:{TTS_SPEAK_PORT}/memory/{{recall,write,dream,stats}}")


# ─── FastMCP Server ───────────────────────────────────────

mcp = FastMCP(
    "Rachel Warudo Bridge",
    instructions="""
    You control Rachel's VTuber avatar in Warudo and her voice.

    speak_text     — make Rachel say something (use during heartbeats or any time)
    set_expression — set facial expression
    clear_expression — reset to neutral
    play_animation — one-shot body animation
    look_at_scene  — screenshot + vision analysis

    Available expressions: neutral, thinking, joy, anger, sadness, surprise,
    concern, excitement

    Available animations: wave, nod, shrug, head_tilt, bounce,
    dance_1, dance_2, clap
    """,
)


# ─── TTS Tool ─────────────────────────────────────────────

@mcp.tool
def speak_text(
    text: str,
    expression: str | None = None,
    animation:  str | None = None,
) -> dict:
    """
    Make Rachel speak text via Chatterbox Turbo TTS.

    Text may include vocal-cue tags such as [gasp], [chuckle], [sigh], [laugh]
    which Chatterbox vocalises directly, and expression tags like [joy] or
    [surprise] which are mapped to the nearest vocal cue and the corresponding
    Warudo facial expression automatically.

    Args:
        text:       Text to speak.
        expression: Optional Warudo expression override (neutral, joy, …).
        animation:  Optional Warudo animation to play alongside speech.
    """
    if _tts is None:
        return {"status": "error", "message": "TTS not initialised"}
    threading.Thread(
        target=speak_sync, args=(text, expression, animation), daemon=True
    ).start()
    return {"status": "ok", "text": text}


# ─── Expression Tools ─────────────────────────────────────

@mcp.tool
def set_expression(expression: str) -> dict:
    """
    Set Rachel's facial expression on the VTuber avatar.

    Args:
        expression: neutral, thinking, joy, anger, sadness, surprise,
                    concern, excitement
    """
    print(f"set expression = {expression}")
    valid = list(EXPRESSION_EXAGGERATION.keys())
    if expression.lower() not in valid:
        return {"error": f"Unknown expression '{expression}'. Valid: {valid}"}
    return send_warudo_action("set_expression", expression.lower())


@mcp.tool
def clear_expression() -> dict:
    """Reset Rachel's face to neutral expression."""
    return send_warudo_action("set_expression", "neutral")


# ─── Animation Tools ──────────────────────────────────────

@mcp.tool
def play_animation(animation: str) -> dict:
    """
    Play a one-shot body animation on Rachel's avatar.

    Args:
        animation: wave, nod, shrug, head_tilt, bounce, dance_1, dance_2, clap
    """
    print(f"set animation = {animation}")
    return send_warudo_action("play_animation", animation.lower())


@mcp.tool
def set_idle_animation(animation: str) -> dict:
    """
    Change Rachel's base idle animation.

    Args:
        animation: standing, sitting, leaning, relaxed
    """
    return send_warudo_action("set_idle", animation.lower())


# ─── Vision ────────────────────────────────────────────────

@mcp.tool
def look_at_scene(prompt: str = "Describe what you see on screen.") -> dict:
    """
    Take a screenshot and analyse it with the vision model.

    Args:
        prompt: What to ask the vision model about the screenshot.
    """
    try:
        from PIL import ImageGrab
        img = ImageGrab.grab()
        img.thumbnail((1280, 720))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        payload = {
            "model": VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }

        with httpx.Client(timeout=180) as client:
            resp = client.post(VISION_URL, json=payload)
            resp.raise_for_status()
            analysis = resp.json()["choices"][0]["message"]["content"]
        print(f"analysis = [{analysis}]")
        return {"analysis": analysis}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ─── Memory Tools ─────────────────────────────────────────

@mcp.tool
def recall_memory(
    query: str,
    emotion: str | None = None,
) -> dict:
    """
    Search Rachel's autobiographical memory for relevant past events.

    Use when someone references a past event, says 'remember when', or when
    the current context might connect to something that happened before.

    Args:
        query:   What to search for (natural language).
        emotion: Rachel's current emotion, for mood-congruent recall (joy, sadness, etc.)
    """
    if _mem is None:
        return {"memories": [], "error": "memory not initialised"}
    results = _mem.recall(query, n_results=3, current_emotion=emotion)
    return {"memories": [
        {
            "text":      r["text"],
            "emotion":   r["emotion"],
            "intensity": r["intensity"],
            "age_days":  r["age_days"],
        }
        for r in results
    ]}


@mcp.tool
def write_memory(
    text: str,
    emotion: str = "neutral",
    intensity: str = "medium",
) -> dict:
    """
    Explicitly save a memory in Rachel's first-person voice.

    Use when something genuinely notable just happened — a clutch play, a funny
    chat moment, something Ben mentioned, a stream milestone. Write the memory
    in past tense as Rachel would remember it.

    Args:
        text:      First-person memory. "Ben clutched a 1v4 and I screamed so loud."
        emotion:   Emotion associated with the memory (joy, surprise, anger, etc.)
        intensity: low, medium, or high
    """
    if _mem is None:
        return {"ok": False, "error": "memory not initialised"}
    try:
        mid = _mem.encode(text, emotion=emotion, intensity=intensity)
        return {"id": mid, "ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ─── Run ───────────────────────────────────────────────────

_tts: RachelTTS | None = None
_mem: RachelMemory | None = None
_offload_tts_url: str | None = None

if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Rachel TTS + Avatar server")
    _parser.add_argument(
        "--offload-tts",
        metavar="URL",
        default=None,
        help="Offload TTS to external endpoint. POST {text} → {audio_b64, sample_rate}. Skips loading Chatterbox locally.",
    )
    _args = _parser.parse_args()
    _offload_tts_url = _args.offload_tts
    _tts = RachelTTS(device=TTS_DEVICE, voice_ref=VOICE_REF_PATH, skip_model=bool(_offload_tts_url))
    _mem = RachelMemory(persist_dir=MEMORY_DB_PATH)
    start_speak_server()
    print(f"[OK] Warudo WebSocket : {WARUDO_WS_URL}")
    print(f"[OK] Vision Model     : {VISION_URL}")
    print(f"[OK] Memory DB        : {MEMORY_DB_PATH}  ({_mem.collection.count()} memories)")
    print(f"[OK] Starting MCP server on http://0.0.0.0:8050/mcp")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8050, path="/mcp", stateless_http=True)
