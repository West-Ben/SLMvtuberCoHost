"""
obs_bridge.py — OBS context aggregator + frame windows (+ legacy perceiver)  (port 8055)
──────────────────────────────────────────────────────────────────────
Background threads:
  screenshot_loop  — grabs current OBS scene every VISION_CAPTURE_INTERVAL_S
  status_loop      — polls stream status + CPU/FPS every 5 s
  event_listener   — OBS event callbacks (scene change, stream on/off)
  perceive_loop    — LEGACY: frames → HF call → scene description. Only started
                     when PERCEIVER_ENABLED=true (default false). The Adaptive
                     Loop reaction path reads raw frames via GET /window instead.

GET /window?seconds=10&max_frames=10&min_frames=4 snapshots the ring buffer
under _lock, releases it, then thins (motion-driven), downscales and saves the
frames to imagelog/DATE/w-<id>/ and returns their paths.

The perceiver (formerly vision_perceiver.py) runs inside this process because
obs_bridge owns the frame ring-buffer. Keeping perception here eliminates the
HTTP round-trip that vision_perceiver needed to fetch frames back.

At most ONE perception call is in-flight at a time (_perceive_in_flight lock).
_lock is NEVER held across the HF call — snapshot under _lock, release, then
call the LLM.

Inbound HTTP:
  POST /chat-message   {"user": str, "text": str}
  POST /twitch-event   {"type": str, "data": dict}
  POST /twitch-info    {"viewers": int, "game": str, ...}
  POST /ben-spoke      {"text": str, "timestamp"?: float}

Outbound:
  GET /context      → assembled JSON snapshot for orchestrator.py
  GET /window       → {"window_id", "frames": [{"id", "path", "t"}], "seconds", "now"}
  GET /description  → {"description": str, "timestamp": float, "age": float}  (legacy perceiver)
"""

import argparse
import base64
import io
import json
import logging
import re
import threading
import time
import wave
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
from PIL import Image

try:
    import pyaudiowpatch as _pyaudiowpatch
    _USE_WPATCH = True
except ImportError:
    _USE_WPATCH = False

import obsws_python as obs
from flask import Flask, request, jsonify

from config import (
    OBS_HOST, OBS_PORT, OBS_PASSWORD,
    OBS_BRIDGE_PORT, ORCHESTRATOR_URL,
    HF_URL, HF_TOKEN, HF_MODEL,
    VISION_CAPTURE_INTERVAL_S,
    VISION_BUFFER_SIZE,
    VISION_PERCEIVE_INTERVAL_S,
    VISION_FRAMES_MIN, VISION_FRAMES_MAX, VISION_MOTION_THRESHOLD,
    BEN_BUFFER_WINDOW_S,
    SPEAKER_DEVICE_NAME, AUDIO_CONTEXT_SECONDS,
    CABLE_DEVICE_NAME, TTS_VOLUME,
    STT_MUTE_LOCK,
    PERCEIVER_ENABLED,
    WINDOW_S, WINDOW_FRAMES_MIN, WINDOW_FRAMES_MAX, WINDOW_FRAME_WIDTH,
    IMAGELOG_DIR,
)
import frame_window

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [bridge] %(message)s")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [bridge] %(message)s")
log = logging.getLogger(__name__)

# ─── CLI ──────────────────────────────────────────────────

_parser = argparse.ArgumentParser(description="OBS Bridge + Scene Perceiver")
_parser.add_argument(
    "--test", action="store_true",
    help=(
        "Offline test mode: force the session active even though OBS is not "
        "streaming, so the orchestrator's silence timer fires without a live "
        "audience. Vision capture and context assembly work normally."
    ),
)
_parser.add_argument(
    "--imagelog-frames", type=int, default=3, metavar="N",
    help="Number of recent frames to save per imagelog group (default: 3)",
)
_parser.add_argument(
    "--imagelog-interval", type=float, default=10.0, metavar="S",
    help="Minimum seconds between imagelog group saves (default: 10.0)",
)
_parser.add_argument(
    "--perceive-interval", type=float, default=VISION_PERCEIVE_INTERVAL_S, metavar="S",
    help=f"Perception loop cadence in seconds (default: {VISION_PERCEIVE_INTERVAL_S})",
)
_parser.add_argument(
    "--frames-min", type=int, default=VISION_FRAMES_MIN, metavar="N",
    help=f"Minimum frames per perception call (default: {VISION_FRAMES_MIN})",
)
_parser.add_argument(
    "--frames-max", type=int, default=VISION_FRAMES_MAX, metavar="N",
    help=f"Maximum frames per perception call (default: {VISION_FRAMES_MAX})",
)
_parser.add_argument(
    "--loopback-device", metavar="NAME", default=None,
    help=(
        "Keyword to select the WASAPI output device for speaker loopback capture "
        f"(default: SPEAKER_DEVICE_NAME env var '{SPEAKER_DEVICE_NAME}'). "
        "The captured speaker audio feeds the audio perceiver."
    ),
)
_parser.add_argument(
    "--no-loopback", action="store_true",
    help="Disable speaker loopback capture; audio perceiver will be skipped each cycle.",
)
_args = _parser.parse_args()

TEST_MODE: bool             = _args.test
_IMAGELOG_FRAMES: int       = _args.imagelog_frames
_IMAGELOG_INTERVAL_S: float = _args.imagelog_interval
_PERCEIVE_INTERVAL: float   = _args.perceive_interval
_FRAMES_MIN: int            = _args.frames_min
_FRAMES_MAX: int            = _args.frames_max
_LOOPBACK_KW: str           = _args.loopback_device if _args.loopback_device is not None else SPEAKER_DEVICE_NAME
_NO_LOOPBACK: bool          = _args.no_loopback

# Suppress obsws_python's connection chatter
for _name in ("obsws_python", "obsws_python.baseclient", "obsws_python.reqs", "obsws_python.subs"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# ─── Shared state (all writes under _lock) ────────────────

_lock         = threading.Lock()
_screenshots  = deque(maxlen=VISION_BUFFER_SIZE)   # {"timestamp": float, "b64": str, "file": str | None}
_chat_lines   = deque(maxlen=100)
_chat_events  = deque(maxlen=200)   # {"t", "user", "text"} — timestamped copy for /window
_events       = deque(maxlen=20)
_stream_state = {
    "active":   TEST_MODE,
    "duration": "0:00:00",
    "scene":    "unknown",
    "fps":      0.0,
    "cpu":      0.0,
    "viewers":  0,
    "game":     "",
}

# ─── Speaker loopback ring buffer (audio perceiver input) ─
# Chunks are float32 16 kHz mono, captured from the speaker output device.
# Sized to hold AUDIO_CONTEXT_SECONDS of audio at 512-sample chunks.

_SPK_CHUNK = 512   # samples at 16 kHz per ring slot
_SPK_RATE  = 16000
_spk_ring: deque = deque(maxlen=int(AUDIO_CONTEXT_SECONDS * _SPK_RATE / _SPK_CHUNK) + 5)

# ─── AEC + Playback state ─────────────────────────────────

_AEC_CHUNK    = 160   # 10 ms at 16 kHz (WebRTC APM requirement)
_AEC_DELAY_MS = 100   # estimated hardware loopback round-trip ms

# Reference ring: 160-sample float32 chunks of what obs_bridge just played.
# Populated by /play endpoint; consumed (process_render) by aec_loop thread.
_aec_ref_deque: deque = deque()
_aec_ref_lock  = threading.Lock()

# AEC output: cleaned loopback frames → consumed by _fetch_audio_b64 / perceive_loop.
_spk_clean_ring: deque = deque(
    maxlen=int(AUDIO_CONTEXT_SECONDS * _SPK_RATE / _AEC_CHUNK) + 5
)

_play_lock   = threading.Lock()   # serialises audio output
_cable_idx:   int | None = None
_speaker_idx: int | None = None

# ─── Perceiver shared state ───────────────────────────────

_desc_lock             = threading.Lock()
_visual_desc:    str   = ""
_visual_desc_ts: float = 0.0
_audio_desc:     str   = ""
_audio_desc_ts:  float = 0.0

_perceive_in_flight = threading.Lock()   # caps perceiver at 1 HF slot

_ben_lock:   threading.Lock = threading.Lock()
_ben_buffer: list[dict]     = []   # [{"ts": float, "text": str}, ...]

# ─── OBS helpers ──────────────────────────────────────────

def _make_req_client() -> "obs.ReqClient | None":
    try:
        return obs.ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASSWORD, timeout=5)
    except Exception as e:
        log.warning(f"OBS ReqClient connect failed: {e}")
        return None


def _fmt_duration(ms: int) -> str:
    return str(timedelta(seconds=(ms or 0) // 1000))


def _is_connection_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(k in msg for k in ("connection", "timeout", "refused", "closed", "broken", "reset", "eof"))


# ─── Image logging (grouped, throttled) ──────────────────
# Each save: imagelog/YYYY-MM-DD/NNNN-HH-MM-SS/ with frame-1.jpg …

_imagelog_last_save:     float = 0.0
_imagelog_group_counter: int   = 0
_imagelog_group_date:    str   = ""


def _save_imagelog_group(frames: list[tuple[float, str]]) -> list[str]:
    """frames — [(timestamp, b64), ...] oldest-first. Returns relative paths."""
    global _imagelog_group_counter, _imagelog_group_date
    if not frames:
        return []
    try:
        now_dt   = datetime.now()
        date_str = now_dt.strftime("%Y-%m-%d")
        time_str = now_dt.strftime("%H-%M-%S")

        if date_str != _imagelog_group_date:
            _imagelog_group_counter = 0
            _imagelog_group_date    = date_str

        _imagelog_group_counter += 1
        group_name = f"{_imagelog_group_counter:04d}-{time_str}"
        group_dir  = Path("imagelog") / date_str / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        paths: list[str] = []
        for i, (ts, b64) in enumerate(frames, 1):
            fname = f"frame-{i}.jpg"
            (group_dir / fname).write_bytes(base64.b64decode(b64))
            paths.append(f"{date_str}/{group_name}/{fname}")
        return paths
    except Exception as e:
        log.debug(f"imagelog group save failed: {e}")
        return []


# ─── Perceiver helpers ────────────────────────────────────

_VISION_SYSTEM = (
    "You are a live-stream scene perception module feeding an AI co-host. "
    "Your output is the co-host's only window into what is happening on screen — make it rich and specific.\n\n"
    "HARD RULE — GAME TITLE: Only ever name a game using the exact 'Game:' field provided in the "
    "stream state. NEVER infer, guess, or recognize a title from the visuals, even if you are "
    "confident. If 'Game:' is 'unknown' or absent, describe the visual style only "
    "(e.g. 'a top-down strategy game', 'a 2D platformer') and give NO title.\n\n"
    "WHAT TO COVER (every cycle):\n"
    "1. SCREEN: The exact UI state — health bar, ammo, score, inventory, minimap, objective "
    "marker, menu, loading screen, cutscene. What specific thing is the streamer looking at?\n"
    "2. ACTION: What is the streamer actively doing — attempting, failing, succeeding, browsing, "
    "watching, idle? Going well or badly?\n"
    "3. NOTABLE (omit if nothing stands out): anything tense, funny, impressive, or chat-worthy.\n\n"
    "Write in plain prose, present tense. Be specific. Aim for 3-4 sentences. "
    "Do NOT describe audio or speech."
)

_AUDIO_SYSTEM = (
    "You are a live-stream audio transcription module feeding an AI co-host. "
    "Your job is to capture what was said or heard — not to describe visuals.\n\n"
    "WHAT TO COVER:\n"
    "- Transcribe or closely paraphrase any speech: streamer commentary, in-game dialogue, "
    "video narration, or other voices.\n"
    "- Note prominent identifiable audio only (e.g. 'victory fanfare', 'tense combat music') — "
    "skip generic ambient sound.\n"
    "- If nothing audible: say 'No speech detected.' in one sentence.\n\n"
    "Write in plain prose, present tense. Aim for 1-3 sentences. "
    "Do NOT describe anything visual."
)

def _frame_signature(b64: str):
    """Downsample frame to 32×32 grayscale array for motion detection."""
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L").resize((32, 32))
        return np.asarray(img, dtype=np.float32)
    except Exception:
        return None


def _decide_frame_count(frames_b64: list[str]) -> int:
    """
    Motion-driven frame selection. frames_b64 is newest-first.
    Returns N in [_FRAMES_MIN, _FRAMES_MAX].
    """
    if not frames_b64:
        return 0
    sigs = [s for s in (_frame_signature(b) for b in frames_b64[:_FRAMES_MAX]) if s is not None]
    if len(sigs) < 2:
        return min(_FRAMES_MIN, len(frames_b64))
    changed = sum(1 for a, b in zip(sigs, sigs[1:]) if np.abs(a - b).mean() > VISION_MOTION_THRESHOLD)
    return max(_FRAMES_MIN, min(_FRAMES_MIN + changed, _FRAMES_MAX, len(frames_b64)))


# ─── TTS Playback helpers (obs_bridge owns audio output) ─

def _resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return audio
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(from_sr), int(to_sr))
    return resample_poly(audio, int(to_sr) // g, int(from_sr) // g).astype(np.float32)


def _find_output_device(keyword: str, label: str) -> int | None:
    """WASAPI-preferred output device matching keyword."""
    hostapis = sd.query_hostapis()
    kw = keyword.lower()
    wasapi_match = any_match = None
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] <= 0:
            continue
        if kw not in dev["name"].lower():
            continue
        api_name = hostapis[dev["hostapi"]]["name"] if dev.get("hostapi") is not None else ""
        if "WASAPI" in api_name and wasapi_match is None:
            wasapi_match = (i, dev, api_name)
        if any_match is None:
            any_match = (i, dev, api_name)
    chosen = wasapi_match or any_match
    if chosen:
        i, dev, api_name = chosen
        log.info(f"  ✅ {label:12s} → [{i}] {dev['name']}  ({api_name})")
        return i
    return None


def _init_playback_devices():
    global _cable_idx, _speaker_idx
    _cable_idx   = _find_output_device(CABLE_DEVICE_NAME,   "VB-Cable") if CABLE_DEVICE_NAME   else None
    _speaker_idx = _find_output_device(SPEAKER_DEVICE_NAME, "Speaker")  if SPEAKER_DEVICE_NAME else None
    if _cable_idx is None:
        log.warning("[playback] VB-Cable not found — Warudo lip sync will not work")


def _find_loopback_device(keyword: str = "") -> tuple:
    """
    Find a WASAPI output device for loopback capture.
    Filters to WASAPI host API only (WasapiSettings is WASAPI-exclusive).
    Prefers keyword match → WASAPI default output → first WASAPI output found.
    Returns (device_index, sample_rate, channels) or (None, 48000, 2).
    """
    try:
        wasapi_idx = None
        for i, api in enumerate(sd.query_hostapis()):
            if "WASAPI" in api.get("name", ""):
                wasapi_idx = i
                break
        if wasapi_idx is None:
            log.warning("WASAPI host API not found — loopback capture unavailable")
            return None, 48000, 2

        wasapi_info  = sd.query_hostapis(wasapi_idx)
        default_out  = wasapi_info.get("default_output_device", -1)
        kw_lc        = keyword.strip().lower()
        kw_match = default_match = first_match = None

        log.info("Available WASAPI output devices (loopback candidates):")
        for i, dev in enumerate(sd.query_devices()):
            if dev["hostapi"] != wasapi_idx or dev["max_output_channels"] <= 0:
                continue
            rate = int(dev["default_samplerate"])
            ch   = min(int(dev["max_output_channels"]), 2)
            name = dev["name"]
            tags = []
            if i == default_out:
                tags.append("DEFAULT")
            if kw_lc and kw_lc in name.lower():
                tags.append("KEYWORD")
            log.info(f"  [{i:2d}] {name}  ch={ch}  sr={rate}  {' | '.join(tags)}")
            entry = (i, rate, ch)
            if kw_lc and kw_lc in name.lower() and kw_match is None:
                kw_match = entry
            if i == default_out and default_match is None:
                default_match = entry
            if first_match is None:
                first_match = entry

        return kw_match or default_match or first_match or (None, 48000, 2)
    except Exception as e:
        log.warning(f"Loopback device discovery failed: {e}")
        return None, 48000, 2


def _fetch_audio_b64() -> tuple[str | None, float]:
    """
    Read the local speaker ring buffer and return (base64 WAV string, duration_s).
    Returns (None, 0.0) if the ring is empty or Rachel is currently speaking.
    Clears the ring while TTS is active so Rachel's own voice never enters perception.
    """
    if STT_MUTE_LOCK.exists():
        _spk_clean_ring.clear()
        return None, 0.0
    chunks = list(_spk_clean_ring)
    if not chunks:
        return None, 0.0
    audio = np.concatenate(chunks)
    duration_s = len(audio) / _SPK_RATE
    audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SPK_RATE)
        wf.writeframes(audio_i16.tobytes())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode(), duration_s


def _build_vision_content(frames: list[dict]) -> list:
    """frames: oldest-first [{"b64": str, "seconds_ago": int}]"""
    n     = len(frames)
    parts = [{"type": "text", "text": f"[STREAM — {n} frame{'s' if n > 1 else ''}, chronological, oldest to newest]"}]
    for i, f in enumerate(frames):
        sec   = round(f.get("seconds_ago", 0))
        label = f"frame-{i+1}-now" if sec <= 1 else f"frame-{i+1}-{sec}s-ago"
        parts.append({"type": "text", "text": label})
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f['b64']}"}})
    return parts


def _build_vision_messages(
    frames: list[dict],
    stream: dict,
    ben_lines: list[dict],
    chat_lines: list[str] | None = None,
) -> list[dict]:
    messages = [{"role": "system", "content": _VISION_SYSTEM}]
    game_name = stream.get("game") or "unknown"
    state = [
        "[STREAM STATE]",
        f"Game: {game_name}",
        f"OBS Scene: {stream.get('scene', '')}",
        f"Viewers: {stream.get('viewers', 0)}",
    ]
    if ben_lines:
        state.append("\n[BEN SAID RECENTLY]")
        for b in ben_lines[-10:]:
            state.append(f"  {datetime.fromtimestamp(b['ts']).strftime('%H:%M:%S')} — {b['text']}")
    if chat_lines:
        state.append("\n[RECENT VIEWER CHAT]")
        for line in chat_lines[-8:]:
            state.append(f"  {line}")
    messages.append({"role": "user", "content": "\n".join(state)})
    if frames:
        messages.append({"role": "user", "content": _build_vision_content(frames)})
    messages.append({"role": "user", "content": (
        "Describe what you see on screen. Cover: exact UI state, what the streamer is doing, "
        "and anything notable. 3-4 sentences. No audio description."
    )})
    return messages


def _build_audio_messages(audio_b64: str, stream: dict) -> list[dict]:
    # No ben_lines/chat — prevents priming the audio model with Ben's expected speech.
    messages = [{"role": "system", "content": _AUDIO_SYSTEM}]
    game_name = stream.get("game") or "unknown"
    messages.append({"role": "user", "content": "\n".join([
        "[STREAM STATE]",
        f"Game: {game_name}",
        f"OBS Scene: {stream.get('scene', '')}",
    ])})
    messages.append({"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
    ]})
    messages.append({"role": "user", "content": (
        "Transcribe or paraphrase what was said or heard. "
        "1-3 sentences. If nothing audible, say 'No speech detected.'"
    )})
    return messages


def _normalise_hf_url() -> str:
    base = re.sub(r"/chat/completions$", "", HF_URL.rstrip("/"))
    base = re.sub(r"/v1$", "", base)
    return f"{base}/v1/chat/completions"


def _call_perceive_llm(messages: list[dict], max_tokens: int = 420) -> str | None:
    """Thinking OFF — we want a description, not Rachel's analytical voice."""
    url     = _normalise_hf_url()
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "model":    HF_MODEL,
        "messages": messages,
        "stream":   True,
        "max_tokens":  max_tokens,
        "temperature": 0.15,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=(15, 90))
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"perceive LLM failed: {e}")
        return None

    out = ""
    for raw in resp.iter_lines():
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if not line or not line.startswith("data: ") or line.strip() == "data: [DONE]":
            continue
        try:
            delta = json.loads(line[6:]).get("choices", [{}])[0].get("delta", {})
            out  += delta.get("content", "") or ""
        except Exception:
            continue
    return out.strip() or None


# ─── Background threads ───────────────────────────────────

def screenshot_loop():
    """Capture current OBS scene every VISION_CAPTURE_INTERVAL_S seconds."""
    global _imagelog_last_save
    cl = None
    logged_first = False
    while True:
        try:
            if cl is None:
                cl = _make_req_client()
                if cl:
                    log.info("Screenshot client connected")
            if cl:
                scene_name = cl.get_current_program_scene().current_program_scene_name
                raw = cl.get_source_screenshot(scene_name, "jpg", 1280, 720, 75).image_data
                if not raw:
                    log.warning(f"screenshot empty for scene '{scene_name}' — nothing cached")
                else:
                    b64 = raw.split(",", 1)[1] if "," in raw else raw
                    now = time.time()
                    with _lock:
                        _screenshots.append({"timestamp": now, "b64": b64, "file": None})
                        _stream_state["scene"] = scene_name

                    if now - _imagelog_last_save >= _IMAGELOG_INTERVAL_S:
                        _imagelog_last_save = now
                        with _lock:
                            recent = [
                                (s["timestamp"], s["b64"])
                                for s in list(_screenshots)[-_IMAGELOG_FRAMES:]
                            ]
                        paths = _save_imagelog_group(recent)
                        if paths:
                            ts_to_path = {ts: p for (ts, _), p in zip(recent, paths)}
                            with _lock:
                                for entry in _screenshots:
                                    if entry["timestamp"] in ts_to_path:
                                        entry["file"] = ts_to_path[entry["timestamp"]]
                            log.info(f"imagelog: {len(paths)} frame(s) → {paths[0].rsplit('/', 1)[0]}/")

                    if not logged_first:
                        log.info(f"First screenshot cached: scene='{scene_name}' {len(b64)} b64 chars")
                        logged_first = True
        except Exception as e:
            if _is_connection_error(e):
                log.warning(f"Screenshot client disconnected ({e}), reconnecting...")
                cl = None
            else:
                log.warning(f"screenshot request error: {e}")
        time.sleep(VISION_CAPTURE_INTERVAL_S)


def status_loop():
    """Poll stream status and OBS stats every 5 seconds."""
    cl = None
    while True:
        try:
            if cl is None:
                cl = _make_req_client()
                if cl:
                    log.info("Status client connected")
            if cl:
                st  = cl.get_stream_status()
                sta = cl.get_stats()
                with _lock:
                    if not TEST_MODE:
                        _stream_state["active"] = st.output_active
                    _stream_state["duration"] = _fmt_duration(getattr(st, "output_duration", 0))
                    _stream_state["fps"]      = round(getattr(sta, "active_fps", 0) or 0, 1)
                    _stream_state["cpu"]      = round(getattr(sta, "cpu_usage", 0) or 0, 1)
        except Exception as e:
            if _is_connection_error(e):
                log.warning(f"Status client disconnected ({e}), reconnecting...")
                cl = None
            else:
                log.debug(f"status request error: {e}")
        time.sleep(5)


def event_listener():
    """Subscribe to OBS events for scene changes and stream start/stop."""
    while True:
        try:
            ev = obs.EventClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASSWORD)

            def on_scene_changed(data):
                scene = getattr(data, "scene_name", "unknown")
                with _lock:
                    _stream_state["scene"] = scene
                log.info(f"Scene → {scene}")

            def on_stream_state_changed(data):
                active = getattr(data, "output_active", False)
                with _lock:
                    if not TEST_MODE:
                        _stream_state["active"] = active
                    _events.append({"type": "stream_start" if active else "stream_stop", "data": {}})
                log.info(f"Stream {'started' if active else 'stopped'}")

            ev.callback.register(on_scene_changed)
            ev.callback.register(on_stream_state_changed)
            log.info("OBS event listener connected.")
            while True:
                time.sleep(60)
        except Exception as e:
            log.warning(f"OBS event listener dropped ({e}), reconnecting in 10 s...")
            time.sleep(10)


def perceive_loop(interval: float):
    """
    Scene perceiver — motion-driven multimodal HF call → cached description.

    CRITICAL: snapshot frames under _lock, release _lock, THEN call the LLM.
    Holding _lock across the HF call would stall screenshot capture every cycle.

    At most 1 in-flight call at a time (_perceive_in_flight). If a cycle
    overruns the interval, the next tick is skipped.
    """
    global _visual_desc, _visual_desc_ts, _audio_desc, _audio_desc_ts
    log.info(f"perceive loop started  interval={interval}s  frames={_FRAMES_MIN}–{_FRAMES_MAX}")
    _ring_empty_cycles = 0
    while True:
        t0 = time.time()

        if not _perceive_in_flight.acquire(blocking=False):
            log.debug("perceive: previous cycle still running — skipping tick")
            time.sleep(interval)
            continue

        try:
            # ── snapshot under _lock, release immediately ──────────
            with _lock:
                snap      = [(s["timestamp"], s["b64"]) for s in reversed(_screenshots) if s.get("b64")]
                stream    = dict(_stream_state)
                chat_snap = list(_chat_lines)[-8:]

            if not snap:
                continue

            now = time.time()
            n   = _decide_frame_count([b for _, b in snap])
            if n == 0:
                continue

            # snap is newest-first; take N, reverse to oldest-first
            chosen = list(reversed(snap[:n]))
            frames = [{"b64": b, "seconds_ago": round(now - ts)} for ts, b in chosen]

            # ── fetch audio + ben outside _lock ───────────────────
            audio, audio_dur = _fetch_audio_b64()
            if audio:
                _ring_empty_cycles = 0
                log.debug(f"[audio] ring: {audio_dur:.1f}s captured → sending to perceiver")
            else:
                _ring_empty_cycles += 1
                if _ring_empty_cycles == 1:
                    log.warning("[audio] ring empty — speaker loopback not capturing; check startup logs for 'Speaker loopback error'")
                elif _ring_empty_cycles % 20 == 0:
                    log.warning(f"[audio] ring still empty after {_ring_empty_cycles} cycles — loopback thread may have crashed")

            with _ben_lock:
                ben = list(_ben_buffer)

            # ── Two parallel LLM calls (no lock held) ─────────────
            # Concurrency: 2 perceiver + 2 orchestrator = 4 — at HF endpoint limit.
            # _perceive_in_flight covers both threads as a unit: not released until
            # both join(), so the next cycle cannot start until both calls complete.
            vision_result: list[str | None] = [None]
            audio_result:  list[str | None] = [None]

            def _run_vision():
                vision_result[0] = _call_perceive_llm(
                    _build_vision_messages(frames, stream, ben, chat_lines=chat_snap),
                    max_tokens=300,
                )

            def _run_audio():
                if audio:
                    audio_result[0] = _call_perceive_llm(
                        _build_audio_messages(audio, stream),
                        max_tokens=150,
                    )

            t_vision = threading.Thread(target=_run_vision, name="perceive-vision", daemon=True)
            t_audio  = threading.Thread(target=_run_audio,  name="perceive-audio",  daemon=True)
            t_vision.start()
            t_audio.start()
            t_vision.join()
            t_audio.join()

            v = vision_result[0]
            a = audio_result[0]
            if v or a:
                now_ts = time.time()
                with _desc_lock:
                    if v is not None:
                        _visual_desc    = v
                        _visual_desc_ts = now_ts
                    if a is not None:
                        _audio_desc     = a
                        _audio_desc_ts  = now_ts
                log.info(f"[vision] ({n}f) {(v or '')[:100]!r}")
                if a:
                    log.info(f"[audio] → {a[:120]}")
                elif audio:
                    log.warning("[audio] LLM call returned empty — model may have timed out or refused")
            else:
                log.debug("perceive: both LLM calls returned empty — keeping previous description")

        except Exception as e:
            log.error(f"perceive cycle error: {e}", exc_info=True)
        finally:
            _perceive_in_flight.release()
            elapsed = time.time() - t0
            time.sleep(max(0.1, interval - elapsed))


def _spk_capture_wpatch():
    """
    Capture speaker loopback via pyaudiowpatch.
    Enumerates isLoopbackDevice entries, picks best match, streams into _spk_ring.
    Raises on fatal setup failure so the retry wrapper can sleep and retry.
    """
    p = _pyaudiowpatch.PyAudio()
    try:
        wasapi_info = p.get_host_api_info_by_type(_pyaudiowpatch.paWASAPI)
        default_out_idx  = wasapi_info["defaultOutputDevice"]
        default_out_name = p.get_device_info_by_index(default_out_idx)["name"]

        kw_lc = _LOOPBACK_KW.strip().lower()
        kw_match = default_match = first_match = None

        log.info(f"[audio] WASAPI default output: [{default_out_idx}] {default_out_name}")
        log.info("[audio] Available loopback devices (pyaudiowpatch):")
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if not dev.get("isLoopbackDevice"):
                continue
            sr   = int(dev["defaultSampleRate"])
            ch   = dev["maxInputChannels"]
            name = dev["name"]
            tags = []
            if default_out_name in name:
                tags.append("DEFAULT")
            if kw_lc and kw_lc in name.lower():
                tags.append("KEYWORD")
            log.info(f"  [{i:2d}] {name}  ch={ch}  sr={sr}  {' | '.join(tags)}")
            entry = (dev, sr, ch)
            if kw_lc and kw_lc in name.lower() and kw_match is None:
                kw_match = entry
            if default_out_name in name and default_match is None:
                default_match = entry
            if first_match is None:
                first_match = entry

        chosen = kw_match or default_match or first_match
        if chosen is None:
            log.error("[audio] pyaudiowpatch: no loopback device found")
            return

        ldev, lrate, lch = chosen
        chunk_native = int(round(_SPK_CHUNK * lrate / _SPK_RATE))

        def _callback(indata, _frame_count, _time_info, status):
            if status:
                log.warning(f"[audio] pyaudio status: {status}")
            audio = np.frombuffer(indata, dtype=np.float32)
            if lch > 1:
                audio = audio.reshape(-1, lch).mean(axis=1)
            if len(audio) != _SPK_CHUNK:
                audio = np.interp(
                    np.linspace(0, len(audio), _SPK_CHUNK, endpoint=False),
                    np.arange(len(audio)),
                    audio,
                )
            _spk_ring.append(audio.astype(np.float32))
            return (None, _pyaudiowpatch.paContinue)

        log.info(f"[audio] opening loopback: [{ldev['index']}] {ldev['name']} ch={lch} sr={lrate}")
        stream = p.open(
            format=_pyaudiowpatch.paFloat32,
            channels=lch,
            rate=lrate,
            frames_per_buffer=chunk_native,
            input=True,
            input_device_index=ldev["index"],
            stream_callback=_callback,
        )
        log.info(f"[audio] pyaudiowpatch loopback active → {_SPK_RATE} Hz mono")
        stream.start_stream()
        while stream.is_active():
            time.sleep(1)
        stream.stop_stream()
        stream.close()
        log.warning("[audio] pyaudiowpatch stream ended — will retry")
    finally:
        p.terminate()


def _spk_capture_sd(dev_idx: int, dev_rate: int, dev_ch: int):
    """
    Capture speaker loopback via sounddevice WASAPI.
    Raises on fatal setup failure so the retry wrapper can sleep and retry.
    """
    chunk_native = int(round(_SPK_CHUNK * dev_rate / _SPK_RATE))
    _cb_count = [0]

    def _callback(indata: np.ndarray, _frames, _time_info, _status):
        if _status:
            log.warning(f"[audio] loopback stream status: {_status}")
        _cb_count[0] += 1
        if _cb_count[0] == 1:
            log.info(f"[audio] loopback first callback: shape={indata.shape}")
        mono = indata.mean(axis=1) if indata.ndim > 1 and dev_ch > 1 else indata.flatten()
        if len(mono) != _SPK_CHUNK:
            mono = np.interp(
                np.linspace(0, len(mono), _SPK_CHUNK, endpoint=False),
                np.arange(len(mono)),
                mono,
            )
        _spk_ring.append(mono.astype(np.float32))

    try:
        extra = sd.WasapiSettings(loopback=True)
        log.info("sd.WasapiSettings(loopback=True) ready")
    except Exception as e:
        extra = None
        log.error(f"sd.WasapiSettings unavailable ({e})")

    log.info(f"[audio] opening InputStream: device={dev_idx} sr={dev_rate} ch={dev_ch} blocksize={chunk_native}")
    with sd.InputStream(
        device=dev_idx, samplerate=dev_rate, channels=dev_ch,
        dtype="float32", blocksize=chunk_native,
        extra_settings=extra, callback=_callback,
    ):
        log.info(f"[audio] sounddevice loopback active: [{dev_idx}] ch={dev_ch} sr={dev_rate} → {_SPK_RATE} Hz mono")
        while True:
            time.sleep(5)
            if _cb_count[0] == 0:
                log.warning("[audio] loopback stream open but 0 callbacks — device may be idle or wrong index")
            else:
                log.debug(f"[audio] loopback alive: {_cb_count[0]} callbacks, ring={len(_spk_ring)}")
                _cb_count[0] = 0


def speaker_capture_loop(dev_idx: int = 0, dev_rate: int = 48000, dev_ch: int = 2):
    """
    Outer retry wrapper for speaker loopback capture.
    Dispatches to pyaudiowpatch (preferred) or sounddevice WASAPI.
    Retries on any failure so transient driver errors don't permanently disable audio.
    """
    backend = "pyaudiowpatch" if _USE_WPATCH else "sounddevice"
    log.info(f"[audio] loopback backend: {backend}")
    while True:
        try:
            if _USE_WPATCH:
                _spk_capture_wpatch()
            else:
                _spk_capture_sd(dev_idx, dev_rate, dev_ch)
        except Exception as e:
            log.error(f"[audio] speaker capture error: {e} — retrying in 10s")
        time.sleep(10)


def aec_loop():
    """
    Acoustic Echo Cancellation — dedicated daemon thread.

    Reads raw loopback from _spk_ring (sole consumer — no other code pops it).
    Reads played-audio reference from _aec_ref_deque (populated by /play endpoint).
    Writes AEC-cleaned 160-sample frames to _spk_clean_ring for perceive_loop.

    Falls back to transparent pass-through if webrtc_audio_processing is absent.

    pip dependency: webrtc-audio-processing
    API note: if your installed version uses AudioProcessingModule / process_stream /
    process_reverse_stream instead of AudioProcessing / process_capture / process_render,
    adjust the class name and method names below accordingly.
    """
    try:
        from webrtc_audio_processing import AudioProcessing as _APM
        apm = _APM(
            num_capture_channels=1,
            num_render_channels=1,
            sample_rate_hz=_SPK_RATE,
        )
        apm.set_stream_delay_ms(_AEC_DELAY_MS)
        log.info("[aec] WebRTC APM ready — echo cancellation active")
        _aec_ok = True
    except Exception as e:
        log.warning(f"[aec] webrtc_audio_processing unavailable ({e}) — loopback passed through unfiltered")
        _aec_ok = False
        apm = None

    cap_buf: list = []

    while True:
        # Feed reference frames to APM render pipeline (far-end / played audio).
        # process_render must be called in real-time order, before matching capture.
        if _aec_ok:
            with _aec_ref_lock:
                while _aec_ref_deque:
                    ref_frame = _aec_ref_deque.popleft()
                    try:
                        apm.process_render(ref_frame)
                    except Exception:
                        pass

        # Drain raw loopback — AEC thread is the sole consumer of _spk_ring.
        try:
            while True:
                chunk = _spk_ring.popleft()
                if _aec_ok:
                    cap_buf.extend(chunk.tolist())
                else:
                    _spk_clean_ring.append(chunk)  # pass-through
        except IndexError:
            pass

        if _aec_ok:
            # Process in 10 ms (160-sample) frames required by WebRTC APM.
            while len(cap_buf) >= _AEC_CHUNK:
                frame = np.array(cap_buf[:_AEC_CHUNK], dtype=np.float32)
                del cap_buf[:_AEC_CHUNK]
                try:
                    clean = apm.process_capture(frame)
                    _spk_clean_ring.append(np.array(clean, dtype=np.float32))
                except Exception:
                    _spk_clean_ring.append(frame)  # fall back to uncleaned frame

        time.sleep(0.005)   # 5 ms poll — keeps latency under one AEC frame


# ─── Flask app ────────────────────────────────────────────

app = Flask(__name__)
app.logger.setLevel(logging.WARNING)


@app.post("/chat-message")
def recv_chat():
    d    = request.get_json(force=True, silent=True) or {}
    text = d.get("text", "").strip()
    if text:
        user = d.get("user", "viewer")
        with _lock:
            _chat_lines.append(f"{user}: {text}")
            _chat_events.append({"t": time.time(), "user": user, "text": text})
    return jsonify({"status": "ok"})


def _forward_event_to_orchestrator(ev: dict):
    try:
        requests.post(f"{ORCHESTRATOR_URL}/event", json=ev, timeout=2)
    except Exception:
        pass


@app.post("/twitch-event")
def recv_twitch_event():
    d = request.get_json(force=True, silent=True) or {}
    if d.get("type"):
        ev = {"type": d["type"], "data": d.get("data", {})}
        with _lock:
            _events.append(ev)
        # Direct trigger for the orchestrator's window flush (fire-and-forget, off the request thread).
        threading.Thread(target=_forward_event_to_orchestrator, args=(ev,), daemon=True).start()
    return jsonify({"status": "ok"})


@app.post("/twitch-info")
def recv_twitch_info():
    d = request.get_json(force=True, silent=True) or {}
    with _lock:
        if "viewers" in d:
            _stream_state["viewers"] = int(d["viewers"])
        if "game" in d:
            _stream_state["game"] = d["game"]
    return jsonify({"status": "ok"})


@app.get("/context")
def get_context():
    now = time.time()
    with _lock:
        stream = dict(_stream_state)
        vision = [
            {"seconds_ago": round(now - s["timestamp"]), "timestamp": s["timestamp"],
             "b64": s["b64"], "file": s.get("file")}
            for s in reversed(_screenshots)
        ]
        chat   = list(_chat_lines)
        events = list(_events)
    return jsonify({"stream": stream, "vision": vision, "chat": chat, "events": events})


@app.get("/status")
def get_status():
    """Cheap health/state probe for the control panel — /context without the frame b64."""
    now = time.time()
    with _lock:
        stream = dict(_stream_state)
        n_frames = len(_screenshots)
        last_frame = _screenshots[-1]["timestamp"] if _screenshots else None
        n_chat = len(_chat_lines)
    return jsonify({"stream": stream, "frames": n_frames, "chat_lines": n_chat,
                    "last_frame_age_s": round(now - last_frame, 1) if last_frame else None})


@app.get("/window")
def get_window():
    """
    Frames for one reaction window. _lock is held only while copying ring
    references; thinning, downscaling and disk writes happen after release so
    screenshot_loop never stalls on this request.
    """
    try:
        seconds    = float(request.args.get("seconds", WINDOW_S))
        max_frames = int(request.args.get("max_frames", WINDOW_FRAMES_MAX))
        min_frames = int(request.args.get("min_frames", WINDOW_FRAMES_MIN))
    except ValueError:
        return jsonify({"error": "bad query params"}), 400
    now      = time.time()
    entries  = frame_window.snapshot(_screenshots, _lock, seconds, now)
    with _lock:
        chat = [dict(c) for c in _chat_events if c["t"] >= now - seconds]
    selected = frame_window.select_frames(entries, min_frames, max_frames, VISION_MOTION_THRESHOLD)
    if not selected:
        return jsonify({"window_id": None, "frames": [], "chat": chat, "seconds": seconds, "now": now})
    try:
        saved = frame_window.save_frames(selected, IMAGELOG_DIR, WINDOW_FRAME_WIDTH, now=now)
    except Exception as e:
        log.warning(f"/window save failed: {e}")
        return jsonify({"error": f"save failed: {e}"}), 500
    return jsonify({**saved, "chat": chat, "seconds": seconds, "now": now})


@app.get("/description")
def get_description():
    with _desc_lock:
        v, v_ts = _visual_desc, _visual_desc_ts
        a, a_ts = _audio_desc,  _audio_desc_ts
    now   = time.time()
    v_age = round(now - v_ts, 1) if v_ts else None
    a_age = round(now - a_ts, 1) if a_ts else None
    parts = []
    if v:
        parts.append(f"[SCREEN] {v}")
    if a:
        parts.append(f"[AUDIO] {a}")
    best_ts  = max(v_ts, a_ts) if (v_ts or a_ts) else 0
    ages     = [x for x in [v_age, a_age] if x is not None]
    best_age = min(ages) if ages else None
    return jsonify({
        "screen":      v,
        "screen_age":  v_age,
        "audio":       a,
        "audio_age":   a_age,
        "description": "\n".join(parts),
        "timestamp":   best_ts,
        "age":         best_age,
    })


@app.post("/play")
def play_tts():
    """
    Receive float32 mono WAV from server.py TTS pipeline.
    Plays to VB-Cable + Speaker (same method as server.py did), pushes 16 kHz
    reference frames to AEC, and manages STT_MUTE_LOCK for stt-server.py.
    Blocks until playback completes so server.py can sequence Warudo actions.
    """
    wav_bytes = request.get_data()
    if not wav_bytes:
        return jsonify({"error": "empty body"}), 400

    try:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            sr  = wf.getframerate()
            sw  = wf.getsampwidth()
            nch = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
        if sw == 2:
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / (2 ** 31)
        else:
            return jsonify({"error": f"unsupported sample width {sw}"}), 400
        if nch > 1:
            pcm = pcm.reshape(-1, nch).mean(axis=1)
    except Exception as e:
        return jsonify({"error": f"WAV parse error: {e}"}), 400

    # Push 16 kHz reference to AEC (chunked at _AEC_CHUNK = 10 ms frames).
    ref_16k = _resample(pcm, sr, _SPK_RATE)
    with _aec_ref_lock:
        for i in range(0, len(ref_16k), _AEC_CHUNK):
            frame = ref_16k[i:i + _AEC_CHUNK]
            if len(frame) < _AEC_CHUNK:
                frame = np.pad(frame, (0, _AEC_CHUNK - len(frame)))
            _aec_ref_deque.append(frame.astype(np.float32))

    devices = []
    if _cable_idx   is not None: devices.append((_cable_idx,   "VB-Cable"))
    if _speaker_idx is not None: devices.append((_speaker_idx, "Speaker"))
    if not devices:
        log.warning("[playback] /play: no output devices configured")
        return jsonify({"status": "ok", "warn": "no output devices"}), 200

    def _play_dev(dev_idx: int, label: str):
        try:
            dev_sr = int(sd.query_devices(dev_idx)["default_samplerate"])
            out    = _resample(pcm, sr, dev_sr)
            out    = np.clip(out * TTS_VOLUME, -1.0, 1.0)
            stereo = np.stack([out, out], axis=1).astype(np.float32)
            dur    = len(stereo) / dev_sr
            peak   = float(np.max(np.abs(stereo)))
            log.info(f"   🔊 [{label}] sr={dev_sr}  dur={dur:.2f}s  peak={peak:.4f}")
            if peak < 0.0001:
                log.warning(f"   [{label}] audio is near-silent — check TTS gain")
            with sd.OutputStream(samplerate=dev_sr, device=dev_idx, channels=2, dtype="float32") as stream:
                # Blocking write lasts ~the clip; leaving the context calls stop(), which
                # waits for the remaining buffers to play out. (A previous sleep(dur + 0.25)
                # here doubled every clip — playback.py measures clip end from this call.)
                stream.write(stereo)
            log.info(f"   ✅ [{label}] done")
        except Exception as e:
            log.error(f"   [playback] error [{label}]: {e}")

    with _play_lock:
        STT_MUTE_LOCK.touch()
        try:
            threads = [
                threading.Thread(target=_play_dev, args=(idx, label), daemon=True)
                for idx, label in devices
            ]
            for t in threads: t.start()
            for t in threads: t.join()
        finally:
            STT_MUTE_LOCK.unlink(missing_ok=True)

    return jsonify({"status": "ok", "duration": round(len(pcm) / sr, 3)})


@app.post("/ben-spoke")
def ben_spoke():
    d    = request.get_json(force=True, silent=True) or {}
    text = d.get("text", "").strip()
    if text:
        cutoff = time.time() - BEN_BUFFER_WINDOW_S
        with _ben_lock:
            _ben_buffer.append({"ts": d.get("timestamp", time.time()), "text": text})
            while _ben_buffer and _ben_buffer[0]["ts"] < cutoff:
                _ben_buffer.pop(0)
    return jsonify({"status": "ok"})


# ─── Startup ──────────────────────────────────────────────

if __name__ == "__main__":
    _init_playback_devices()

    threads = [
        (screenshot_loop,                "screenshot", {}),
        (status_loop,                    "status",     {}),
        (event_listener,                 "events",     {}),
        (aec_loop,      "aec",      {}),
    ]
    # Legacy perceiver: kept in code, off by default. The reaction path uses /window.
    if PERCEIVER_ENABLED:
        threads.append((perceive_loop, "perceive", {"interval": _PERCEIVE_INTERVAL}))
    else:
        log.info("Perceiver disabled (PERCEIVER_ENABLED=false) — /description will stay empty")
    for target, name, kwargs in threads:
        threading.Thread(target=target, name=name, kwargs=kwargs, daemon=True).start()
        log.info(f"Started {name} thread.")

    if _NO_LOOPBACK:
        log.info("Speaker loopback disabled (--no-loopback) — audio perceiver will be skipped each cycle")
    else:
        _loopback_started = False
        if _USE_WPATCH:
            threading.Thread(target=speaker_capture_loop, name="spk-loopback", daemon=True).start()
            log.info("Started spk-loopback thread (pyaudiowpatch).")
            _loopback_started = True
        else:
            spk_idx, spk_rate, spk_ch = _find_loopback_device(_LOOPBACK_KW)
            if spk_idx is not None:
                threading.Thread(
                    target=speaker_capture_loop,
                    args=(spk_idx, spk_rate, spk_ch),
                    name="spk-loopback",
                    daemon=True,
                ).start()
                log.info("Started spk-loopback thread (sounddevice).")
                _loopback_started = True
            else:
                kw_hint = f" (keyword: '{_LOOPBACK_KW}')" if _LOOPBACK_KW else ""
                log.warning(f"No WASAPI output device found{kw_hint} — audio perceiver disabled")
                log.warning("Set SPEAKER_DEVICE_NAME in .env or pass --loopback-device NAME")

        if _loopback_started:
            def _validate_loopback():
                time.sleep(5)
                if _spk_ring:
                    log.info(f"[audio] loopback confirmed: {len(_spk_ring)} chunks in ring")
                else:
                    log.warning("[audio] ring still empty after 5s — loopback device enumeration will appear above; check backend errors")
            threading.Thread(target=_validate_loopback, name="loopback-probe", daemon=True).start()

    if TEST_MODE:
        log.info("*** TEST MODE — stream forced active, no live audience required ***")
    if PERCEIVER_ENABLED:
        log.info(f"perceiver: interval={_PERCEIVE_INTERVAL}s  frames={_FRAMES_MIN}–{_FRAMES_MAX}  motion_threshold={VISION_MOTION_THRESHOLD}")
    log.info(f"window: default {WINDOW_S}s  frames={WINDOW_FRAMES_MIN}–{WINDOW_FRAMES_MAX}  width={WINDOW_FRAME_WIDTH}")
    log.info(f"OBS bridge listening on port {OBS_BRIDGE_PORT}")
    app.run(host="0.0.0.0", port=OBS_BRIDGE_PORT, threaded=True)
