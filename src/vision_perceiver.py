"""
vision_perceiver.py — Background scene perception service  (port 8056)
────────────────────────────────────────────────────────────────────────
Decouples slow multimodal inference from Rachel's fast reaction path.

Runs an independent perception loop that:
  1. Fetches vision frames from obs_bridge /context
  2. Fetches mixed mic+speaker audio from stt-server /audio-context (optional)
  3. Includes accumulated [ben] utterances for conversational grounding
  4. Calls HF endpoint with thinking ON → produces a scene description (text)
  5. Stores the description in a thread-safe slot

Serves:
  GET  /description  → {"description": str, "timestamp": float, "age": float}
  POST /ben-spoke    → {"text": str, "timestamp": float?}

Run:
  python vision_perceiver.py [--thinking] [--interval 2.5] [--frames N] [--no-audio]
"""

import argparse
import json
import logging
import re
import threading
import time
from datetime import datetime

import requests
from flask import Flask, jsonify, request

from config import (
    OBS_BRIDGE_URL, STT_SERVER_URL,
    HF_URL, HF_TOKEN, HF_MODEL,
    VISION_PERCEIVER_PORT, VISION_PERCEIVE_INTERVAL_S,
    BEN_BUFFER_WINDOW_S,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [perc] %(message)s")
log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a live-stream scene perception module. "
    "Given video frames and any available audio, describe in 2-3 sentences: "
    "what is visible on screen, what the streamer is doing, and anything notable "
    "that was said or heard. Be specific — name the game, UI elements, player "
    "actions, and any relevant dialogue. Be factual and concise."
)

# ─── Shared description slot ─────────────────────────────

_desc_lock:   threading.Lock = threading.Lock()
_description: str   = ""
_desc_ts:     float = 0.0

# ─── [Ben] utterance buffer ──────────────────────────────

_ben_lock:   threading.Lock = threading.Lock()
_ben_buffer: list[dict]     = []   # [{"ts": float, "text": str}, ...]

# ─── Runtime flags (set in main) ─────────────────────────

_n_frames:        int  = 3
_no_audio:        bool = False
_enable_thinking: bool = True

# ─── Context helpers ──────────────────────────────────────

def _fetch_obs_context() -> dict:
    try:
        r = requests.get(f"{OBS_BRIDGE_URL}/context", timeout=3)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def _fetch_audio_b64() -> str | None:
    try:
        r = requests.get(f"{STT_SERVER_URL}/audio-context", timeout=3)
        if r.status_code == 200:
            return r.json().get("audio_b64")
    except Exception:
        pass
    return None


def _build_vision_content(frames: list[dict]) -> list:
    n     = len(frames)
    parts = [{"type": "text", "text":
              f"[STREAM — {n} frame{'s' if n > 1 else ''}, chronological, oldest to newest]"}]
    for i, frame in enumerate(frames):
        sec   = round(frame.get("seconds_ago", 0))
        label = f"frame-{i+1}-now" if sec <= 1 else f"frame-{i+1}-{sec}s-ago"
        parts.append({"type": "text", "text": label})
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:image/jpeg;base64,{frame['b64']}"}})
    return parts


def _build_perception_messages(
    frames:    list[dict],
    audio_b64: str | None,
    ben_lines: list[dict],
    stream:    dict,
) -> list[dict]:
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

    state_parts = [
        f"Game: {stream.get('game') or 'unknown'} | "
        f"Scene: {stream.get('scene', '')} | "
        f"Viewers: {stream.get('viewers', 0)}"
    ]
    if ben_lines:
        state_parts.append("\n[BEN SAID RECENTLY]")
        for b in ben_lines[-10:]:
            ts_str = datetime.fromtimestamp(b["ts"]).strftime("%H:%M:%S")
            state_parts.append(f"  {ts_str} — {b['text']}")
    messages.append({"role": "user", "content": "\n".join(state_parts)})

    if frames:
        messages.append({"role": "user", "content": _build_vision_content(frames)})

    if audio_b64:
        messages.append({
            "role":    "user",
            "content": [{"type": "input_audio",
                         "input_audio": {"data": audio_b64, "format": "wav"}}],
        })

    messages.append({
        "role":    "user",
        "content": "Describe what you observe in 2-3 sentences.",
    })
    return messages


# ─── LLM call (HF endpoint, SSE) ─────────────────────────

def _normalise_hf_url() -> str:
    base = re.sub(r"/chat/completions$", "", HF_URL.rstrip("/"))
    base = re.sub(r"/v1$", "", base)
    return f"{base}/v1/chat/completions"


def _call_perception_llm(messages: list[dict]) -> str | None:
    url     = _normalise_hf_url()
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload: dict = {
        "model":       HF_MODEL,
        "messages":    messages,
        "stream":      True,
        "max_tokens":  256,
        "temperature": 0.1,
    }
    if _enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
        payload["thinking"]             = {"type": "enabled", "budget_tokens": 512}
    else:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    try:
        resp = requests.post(url, headers=headers, json=payload,
                             stream=True, timeout=(15, 90))
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"LLM request failed: {e}")
        return None

    full_content = ""
    for raw_line in resp.iter_lines():
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line or not line.startswith("data: ") or line.strip() == "data: [DONE]":
            continue
        try:
            chunk = json.loads(line[6:])
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            full_content += delta.get("content", "")
        except Exception:
            continue

    return full_content.strip() or None


# ─── Perception loop ──────────────────────────────────────

def _perceive_loop(interval: float):
    log.info(
        f"loop started — interval={interval}s frames={_n_frames} "
        f"thinking={'on' if _enable_thinking else 'off'} "
        f"audio={'off' if _no_audio else 'on'}"
    )
    while True:
        t0 = time.time()
        try:
            ctx    = _fetch_obs_context()
            stream = ctx.get("stream", {})

            vision_raw = [e for e in ctx.get("vision", []) if e.get("b64")]
            frames     = list(reversed(vision_raw[:_n_frames]))

            audio_b64 = None if _no_audio else _fetch_audio_b64()

            if not frames and not audio_b64:
                log.debug("no frames and no audio — skipping cycle")
                time.sleep(max(0.5, interval - (time.time() - t0)))
                continue

            with _ben_lock:
                ben_snapshot = list(_ben_buffer)

            messages = _build_perception_messages(frames, audio_b64, ben_snapshot, stream)
            desc = _call_perception_llm(messages)

            if desc:
                with _desc_lock:
                    global _description, _desc_ts
                    _description = desc
                    _desc_ts     = time.time()
                log.info(repr(desc[:140]))
            else:
                log.debug("LLM returned empty — keeping previous description")

        except Exception as e:
            log.error(f"cycle error: {e}", exc_info=True)

        elapsed = time.time() - t0
        time.sleep(max(0.1, interval - elapsed))


# ─── Flask ────────────────────────────────────────────────

app = Flask(__name__)
app.logger.setLevel(logging.WARNING)


@app.get("/description")
def get_description():
    with _desc_lock:
        desc = _description
        ts   = _desc_ts
    return jsonify({
        "description": desc,
        "timestamp":   ts,
        "age":         round(time.time() - ts, 1) if ts else None,
    })


@app.post("/ben-spoke")
def ben_spoke():
    d    = request.get_json(force=True, silent=True) or {}
    text = d.get("text", "").strip()
    if text:
        cutoff = time.time() - BEN_BUFFER_WINDOW_S
        with _ben_lock:
            _ben_buffer.append({"ts": d.get("timestamp", time.time()), "text": text})
            # Expire utterances older than BEN_BUFFER_WINDOW_S
            while _ben_buffer and _ben_buffer[0]["ts"] < cutoff:
                _ben_buffer.pop(0)
    return jsonify({"status": "ok"})


# ─── Entry point ──────────────────────────────────────────

def main():
    global _n_frames, _no_audio, _enable_thinking

    parser = argparse.ArgumentParser(description="Rachel Vision Perceiver")
    parser.add_argument(
        "--interval", type=float, default=VISION_PERCEIVE_INTERVAL_S, metavar="S",
        help=f"Perception loop cadence in seconds (default: {VISION_PERCEIVE_INTERVAL_S})",
    )
    parser.add_argument(
        "--frames", type=int, default=3, metavar="N",
        help="Number of vision frames per inference call (default: 3)",
    )
    parser.add_argument(
        "--thinking", action="store_true",
        help="Enable extended reasoning (thinking) for perception calls",
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="Skip audio context from stt-server /audio-context",
    )
    args = parser.parse_args()

    _n_frames        = args.frames
    _no_audio        = args.no_audio
    _enable_thinking = args.thinking

    log.info(f"Starting → http://localhost:{VISION_PERCEIVER_PORT}")
    log.info(f"  HF endpoint: {_normalise_hf_url()}  model={HF_MODEL}")
    log.info(f"  OBS:  {OBS_BRIDGE_URL}/context")
    log.info(f"  STT:  {'disabled' if _no_audio else STT_SERVER_URL + '/audio-context'}")

    threading.Thread(
        target=_perceive_loop,
        args=(args.interval,),
        name="perc-loop",
        daemon=True,
    ).start()

    app.run(host="0.0.0.0", port=VISION_PERCEIVER_PORT,
            threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
