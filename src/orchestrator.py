"""
orchestrator.py — Rachel reaction brain  (port 8052)
─────────────────────────────────────────────────────────
Two reaction paths, selected by config REACTION_MODE:

frames (default, Adaptive Loop)
  WindowAccumulator collects [ben] speech + chat with timestamps. A window
  flushes after WINDOW_S once VAD is idle (hard max WINDOW_MAX_S), early on
  direct chat/events after WINDOW_MIN_S, or — with nothing said — as the
  silence path after SILENCE_THRESHOLD. On flush: GET obs_bridge /window for
  4–10 raw frames → prompt_format.build_reaction_messages → ONE model call →
  tag_parser.parse (Plan) → playback.Sequencer (TTS clips + Warudo beats) → training row.
  While a reaction is in flight or TTS is speaking, input keeps accumulating
  and is merged into the next window (never queued).

text (legacy, A/B + rollback)
  The pre-Adaptive-Loop path below: cached [VISION] description text +
  layered SessionManager messages. Parsing helpers live in legacy_text_path.py.

Concurrency: BoundedSemaphore(2) around model calls — 1 reaction + 1 session
summary (legacy path). obs_bridge's _lock is never held across a model call:
/window snapshots and releases before any processing.

Legacy text-path message array structure (sent on every HF call):
  Layer 0  system   SOUL.md + USER.md (permanent)
  Layer 1  user     "[Earlier this stream: ...]" (compressed history)
  Layer 2  user     <<STREAM_CONTEXT>>...</STREAM_CONTEXT>> (live context)
  Layer 3  user/assistant  Recent ~10 turns (sliding window)
  Layer 4  user     Current input (text or audio)

Auto-commentary is owned here: a self-contained silence timer fires a
timer-triggered response when Rachel has been quiet for SILENCE_THRESHOLD
seconds. No obs_bridge heartbeat, no scene-diffing.

Inbound HTTP (port 8052):
  POST /transcription  {"text": str}                          ← stt-server.py
  POST /audio-input    {"audio_b64": str, "format": str}      ← stt-server.py --direct
  POST /chat-input     {"user": str, "text": str}             ← direct viewer chat (triggers)
  POST /vad            {"speaking": bool}                     ← stt-server.py VAD start/end
  POST /event          {"type": str, "data": dict}            ← obs_bridge twitch events
  GET  /status                                                ← control panel (engine state, no model call)
  GET  /vision-test                                           ← diagnostic

Run:
  python orchestrator.py [--direct] [--text "test message"] [--no-log]
"""

import argparse
import difflib
import json
import logging
import queue
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import requests
from flask import Flask, request, jsonify

import prompt_format
import playback
import tag_parser
from legacy_text_path import (
    _ALL_TAG_RE, _TOOL_STRIP_RE,
    _detect_animation_request, _emotion_to_expression, _execute_tool_call,
    _extract_avatar_cues, _extract_tool_intent_from_reasoning, _normalise_spoken,
    _parse_tool_calls, _strip_emotion_tags, parse_channel, parse_channel_v6,
)
from rachel_tools import TOOLS
from config import (
    REPO_ROOT, ADAPTER_REGISTRY, ROWS_DIR,
    REACTION_MODE, REACTION_TRANSPORT, LEAD_S,
    WINDOW_S, WINDOW_MAX_S, WINDOW_MIN_S, WINDOW_FRAMES_MIN, WINDOW_FRAMES_MAX,
    WINDOW_POLL_S, VAD_STALE_S, ECHO_SIMILARITY, REACTION_HISTORY_PAIRS,
    REACTION_MAX_TOKENS, REACTION_SAMPLING,
    ORCHESTRATOR_PORT,
    OBS_BRIDGE_URL,
    LLM_URL, LLM_TOKEN, LLM_MODEL,
    HF_URL, HF_TOKEN, HF_MODEL,
    SPEAK_URL,
    STT_MUTE_LOCK,
    CHAT_SUMMARY_KEEP, CHAT_WINDOW_LINES,
    WORKSPACE_DIR,
    TRAINING_DATA_DIR,
    SESSION_LOG_DIR,
    SESSION_MAX_TURNS,
    SILENCE_THRESHOLD, SILENCE_POLL,
    BEN_BUFFER_WINDOW_S,
    VISION_STALE_WARN_S, VISION_STALE_DROP_S,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [orch] %(message)s")
log = logging.getLogger(__name__)




# ─── Context state (all writes under _lock) ───────────────

_lock          = threading.Lock()
_chat_buffer   = deque(maxlen=CHAT_WINDOW_LINES * CHAT_SUMMARY_KEEP)
_chat_summaries: list[str] = []
_stream_state  = {}
_pending_events: list[dict] = []
_last_spoke_at = time.time()   # seeded so the silence timer doesn't fire instantly
_last_auto_vision_summaries: deque = deque(maxlen=2)
_last_timer_img: str | None = None   # b64 of the last image sent on a timer turn
_last_fired_ben_texts: list[str] = []

_current_emotion: str = "neutral"
_last_spoken:     str = ""
_recent_spoken: deque = deque(maxlen=6)

# ─── LLM serialisation + input queue ──────────────────────

_llm_lock = threading.Lock()
_input_queue: queue.Queue = queue.Queue(maxsize=10)

# Cap concurrent HF calls to 2 (live reaction + at most one async summary).
# obs_bridge's perceiver holds 1 slot, so worst-case total = 3 < 4 endpoint limit.
_hf_sema = threading.BoundedSemaphore(2)


def _enqueue_input(item: dict) -> None:
    try:
        _input_queue.put_nowait(item)
    except queue.Full:
        log.warning("[queue] full, dropping input")


# ─── Module-level objects (initialised in __main__) ───────

_session        = None   # SessionManager
_logger         = None   # TrainingLogger
_session_logger = None   # SessionLogger
_direct_mode    = False

# Ben's mic utterances accumulated between timer fires
_ben_lock:   threading.Lock = threading.Lock()
_ben_buffer: list[dict]     = []   # [{"ts": float, "text": str}, ...]


# ─── Chat summariser ──────────────────────────────────────

def _summarise_chat(lines: list[str]) -> str:
    if not lines:
        return ""
    blob = " | ".join(lines[-CHAT_WINDOW_LINES:])
    return blob[:300] + ("…" if len(blob) > 300 else "")


# ─── Context fetching ─────────────────────────────────────

def _fetch_context() -> dict:
    try:
        r = requests.get(f"{OBS_BRIDGE_URL}/context", timeout=3)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def _fetch_vision_description() -> str:
    """Return per-slot scene descriptions with independent staleness annotations."""
    try:
        r = requests.get(f"{OBS_BRIDGE_URL}/description", timeout=3)
        if r.status_code != 200:
            return ""
        d = r.json()
        parts = []
        for slot, age_key in (("screen", "screen_age"), ("audio", "audio_age")):
            text = (d.get(slot) or "").strip()
            age  = float(d.get(age_key) or 0)
            if not text:
                continue
            if age >= VISION_STALE_DROP_S:
                log.debug(f"[vision] {slot} {age:.0f}s old — dropping")
                continue
            label = f"[{slot.upper()}]"
            if age >= VISION_STALE_WARN_S:
                label = f"[{slot.upper()} — {age:.0f}s ago, may be stale]"
            parts.append(f"{label} {text}")
        return "\n".join(parts)
    except Exception:
        return ""


def _notify_perceiver_ben_spoke(text: str) -> None:
    """Fire-and-forget: tell vision_perceiver that Ben said something."""
    try:
        requests.post(
            f"{OBS_BRIDGE_URL}/ben-spoke",
            json={"text": text, "timestamp": time.time()},
            timeout=1,
        )
    except Exception:
        pass


def _stream_active(ctx: dict) -> bool:
    return bool(ctx.get("stream", {}).get("active", False))


def _drain_events(events: list[dict]) -> str | None:
    label = {
        "channel.subscribe":             "new sub",
        "channel.subscription.gift":     "gifted subs",
        "channel.cheer":                 "cheer",
        "channel.raid":                  "incoming raid",
        "channel.channel_points_custom_reward_redemption.add": "channel points redeemed",
        "channel.update":                "channel updated",
        "stream_start":                  "stream just started",
        "stream_stop":                   "stream ended",
    }
    for ev in reversed(events):
        desc = label.get(ev.get("type", ""))
        if not desc:
            continue
        d     = ev.get("data", {})
        user  = d.get("user_name") or d.get("from_broadcaster_user_name") or ""
        qty   = d.get("total", d.get("bits", ""))
        parts = [desc]
        if user:
            parts.append(f"from {user}")
        if qty:
            parts.append(f"({qty})")
        return " ".join(parts)
    return None


# ─── Context block builder (Layer 2) ─────────────────────

def _build_context_block(
    ctx: dict,
    is_timer: bool = False,
    vision_description: str = "",
    ben_lines: list[dict] | None = None,
) -> str:
    """
    Build Layer 2a — text-only stream context. Rebuilt every turn.
    vision_description: scene description from vision_perceiver (injected as [VISION]).
    is_timer: when True, [RACHEL RECENTLY COMMENTED] is omitted.
    ben_lines: full _ben_buffer snapshot (all entries, reacted or not) for context.
    """
    stream = ctx.get("stream", {})
    chat   = ctx.get("chat", [])
    events = ctx.get("events", [])

    with _lock:
        for line in chat:
            if line not in _chat_buffer:
                _chat_buffer.append(line)
        summary           = _summarise_chat(list(_chat_buffer))
        last_event_str    = _drain_events(events) or stream.get("last_event", "")
        recent_commentary = list(_last_auto_vision_summaries)

    lines = [
        "[STREAM STATE]",
        f"Game: {stream.get('game') or 'unknown'} | "
        f"Viewers: {stream.get('viewers', 0)} | "
        f"Uptime: {stream.get('duration', '0:00:00')} | "
        f"Scene: {stream.get('scene', '')} | "
        f"FPS: {stream.get('fps', 0)} | "
        f"CPU: {stream.get('cpu', 0)}%",
    ]
    if last_event_str:
        lines.append(f"Last event: {last_event_str}")

    if summary:
        lines += ["\n[RECENT CHAT]", summary]

    if vision_description:
        lines += ["\n[VISION]", vision_description]

    if ben_lines:
        lines.append("\n[BEN SAID RECENTLY]")
        for b in ben_lines[-10:]:
            lines.append(f"  {datetime.fromtimestamp(b['ts']).strftime('%H:%M:%S')} — {b['text']}")

    if recent_commentary and not is_timer:
        lines.append("\n[RACHEL RECENTLY COMMENTED]")
        lines += [f"- {t}" for t in recent_commentary]

    return "\n".join(lines)



# ─── Training snapshot helper ────────────────────────────

def _make_snapshot(ctx: dict, trigger: str) -> dict:
    stream = ctx.get("stream", {})
    chat   = ctx.get("chat", [])
    vision = ctx.get("vision", [])
    return {
        "game":           stream.get("game", ""),
        "viewers":        stream.get("viewers", 0),
        "scene":          stream.get("scene", ""),
        "vision_summary": next((v.get("summary", "") for v in vision if v.get("summary")), ""),
        "chat_summary":   " | ".join(chat[-5:]) if chat else "",
        "trigger":        trigger,
    }


# ─── LLM stream helpers ───────────────────────────────────


def _stream_sse(resp, extract) -> tuple[str, str, int]:
    """
    Generic SSE drain. `extract(chunk) -> (content, reasoning, done)`.
    Returns (full_text, full_reasoning, token_count). Prints spoken text live;
    prints reasoning dimly so the thought is visible in the console.
    """
    full_text = full_reasoning = ""
    token_count = 0
    reasoning_open = False
    for line in resp.iter_lines():
        line = line.decode("utf-8") if isinstance(line, bytes) else line
        if not line or not line.startswith("data: ") or line.strip() == "data: [DONE]":
            continue
        try:
            chunk = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        content, reasoning, done = extract(chunk)

        if reasoning:
            if not reasoning_open:
                print("\n\033[2m[think] ", end="", flush=True)  # dim
                reasoning_open = True
            full_reasoning += reasoning
            print(reasoning, end="", flush=True)

        if content:
            if reasoning_open:
                print("\033[0m", flush=True)  # close dim, newline
                reasoning_open = False
            full_text   += content
            token_count += 1
            display = _ALL_TAG_RE.sub("", content)
            if display:
                print(display, end="", flush=True)

        if done:
            if reasoning_open:
                print("\033[0m", flush=True)
                reasoning_open = False
            log.info(f"[stream] done  tokens={token_count}  "
                     f"chars={len(full_text)}  reasoning={len(full_reasoning)}")
    return full_text, full_reasoning, token_count


def _strip_images(messages: list[dict]) -> list[dict]:
    """Return a copy of messages with all image_url content blocks removed."""
    stripped = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            text_only = [b for b in content if not (isinstance(b, dict) and b.get("type") == "image_url")]
            if not text_only:
                continue  # drop the message entirely if it was image-only
            # Collapse to a plain string only when there is exactly one block and it
            # carries a "text" key — never index ["text"] blindly (input_audio, etc.).
            if len(text_only) == 1 and isinstance(text_only[0], dict) and "text" in text_only[0]:
                new_content = text_only[0]["text"]
            else:
                new_content = text_only
            stripped.append({**m, "content": new_content})
        else:
            stripped.append(m)
    return stripped


def _stream_chat_completions(
    url: str, token: str, model: str, messages: list[dict],
    extra_payload: dict | None = None,
    include_tools: bool = True,
) -> tuple[str, str | None, str | None] | None:
    """POST to /v1/chat/completions (SSE). Used for both local and --direct."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload: dict = {
        "model": model, "messages": messages,
        "stream": True, "max_tokens": 2048, "temperature": 1, "top_p": 0.95,
    }
    if include_tools:
        payload["tools"]       = TOOLS
        payload["tool_choice"] = "auto"
    if extra_payload:
        payload.update(extra_payload)

    try:
        resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=(10, 180))
        if resp.status_code == 400:
            log.warning(f"[llm] 400 Bad Request — body: {resp.text[:400]}")
            # 400 = wire-format rejection. Strip images and retry; the structure
            # error is in the message format, not in the image content itself.
            if _count_images(messages):
                log.warning("[llm] 400 with images — retrying without images")
                payload["messages"] = _strip_images(messages)
                resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=(10, 180))
        resp.raise_for_status()
    except Exception as e:
        log.error(f"LLM error: {e}")
        return None

    def _extract(chunk):
        delta  = chunk.get("choices", [{}])[0].get("delta", {})
        finish = chunk.get("choices", [{}])[0].get("finish_reason")
        return delta.get("content", ""), delta.get("reasoning_content", ""), bool(finish)

    try:
        full_text, full_reasoning, token_count = _stream_sse(resp, _extract)
    except Exception as e:
        log.error(f"LLM stream error: {e}")
        return None

    # tokens=0: model thought but produced no spoken output (thinking consumed budget).
    # NEVER strip the image here — the projector bound it correctly; stripping produces
    # a blind "I see no screenshot" reply. Instead:
    #   • If thinking was enabled (direct/v6): retry once with enable_thinking=false so
    #     the model skips the think block and emits speech directly.
    #   • Otherwise (local): skip the turn — a different truncation cause, don't lie.
    if token_count == 0:
        had_reasoning   = bool(full_reasoning.strip())
        thinking_was_on = payload.get("chat_template_kwargs", {}).get("enable_thinking", False)
        if thinking_was_on:
            log.warning(
                f"[llm] 0 content tokens (reasoning={'yes' if had_reasoning else 'empty'}) — "
                "retrying with enable_thinking=false, images intact"
            )
            retry_payload = {**payload, "chat_template_kwargs": {"enable_thinking": False}}
            retry_payload.pop("thinking", None)  # remove budget cap too
            try:
                resp2 = requests.post(url, headers=headers, json=retry_payload, stream=True, timeout=(10, 180))
                resp2.raise_for_status()
                full_text, full_reasoning, _ = _stream_sse(resp2, _extract)
            except Exception as e:
                log.error(f"[llm] thinking-off retry error: {e}")
                return None
        else:
            log.warning(
                f"[llm] 0 content tokens (reasoning={'yes' if had_reasoning else 'empty'}) — skipping turn"
            )
            return None

    return (full_text, full_reasoning, *_extract_avatar_cues(full_text))


def _count_images(messages: list[dict]) -> int:
    """Count image_url blocks across all messages (diagnostic)."""
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            n += sum(1 for b in c if isinstance(b, dict) and b.get("type") == "image_url")
    return n


def _call_llm(
    messages: list[dict],
    enable_thinking: bool = False,
) -> tuple[str, str | None, str | None, str] | None:
    """
    Returns (full_text, expression, animation, full_reasoning).

    Local (v5): full_text is re-stitched into <|channel>reasoning<channel|>spoken;
                full_reasoning is "" (already merged into full_text for parse_channel).
    Direct (v6): full_text is the raw content token; full_reasoning carries the
                 separate reasoning_content for parse_channel_v6.

    enable_thinking: controls thinking for the HF (direct) path only.
                     Always False for Rachel's reaction calls; vision_perceiver.py
                     makes its own calls with thinking ON independently.
    """
    while messages and messages[-1].get("role") in ("assistant", "model"):
        messages = messages[:-1]

    img_n = _count_images(messages)
    if img_n:
        log.info(f"[llm] sending {img_n} image(s) via /v1/chat/completions")

    if _direct_mode:
        url = HF_URL.rstrip("/") + "/chat/completions"
        if enable_thinking:
            v6_extra = {
                "chat_template_kwargs": {"enable_thinking": True},
                "thinking":             {"type": "enabled", "budget_tokens": 700},
            }
        else:
            v6_extra = {
                "chat_template_kwargs": {"enable_thinking": False},
                "thinking":             {"type": "disabled"},
            }
        _hf_sema.acquire()
        try:
            r = _stream_chat_completions(url, HF_TOKEN, HF_MODEL, messages, extra_payload=v6_extra)
        finally:
            _hf_sema.release()
        if r is None:
            return None
        full_text, full_reasoning, expression, animation = r
        # v6: reasoning_content and content are already separate — do not re-stitch.
        return (full_text, expression, animation, full_reasoning)

    # Normalise local URL to the chat-completions endpoint regardless of how
    # LLM_URL is written (bare host, /v1, or full path).
    base = re.sub(r"/v1/chat/completions$", "", LLM_URL.rstrip("/"))
    base = re.sub(r"/v1$", "", base)
    url  = f"{base}/v1/chat/completions"
    r = _stream_chat_completions(url, LLM_TOKEN, LLM_MODEL, messages)
    if r is None:
        return None
    full_text, full_reasoning, expression, animation = r
    # v5: re-stitch reasoning into the <|channel> envelope that parse_channel expects.
    if full_reasoning.strip():
        full_text = f"<|channel>{full_reasoning.strip()}<channel|>{full_text}"
    return (full_text, expression, animation, "")


# ─── Response relay ───────────────────────────────────────


def _relay(text: str, expression: str | None, animation: str | None):
    global _last_spoke_at
    clean = _ALL_TAG_RE.sub("", text).strip()
    if not clean:
        log.info("[SILENT]")
        return
    log.info(f"→ TTS | expr={expression or '—'} anim={animation or '—'}")
    try:
        requests.post(SPEAK_URL, json={"text": text, "expression": expression, "animation": animation}, timeout=5)
    except Exception as e:
        log.error(f"TTS relay: {e}")
    with _lock:
        _last_spoke_at = time.time()


# ─── Response parser ──────────────────────────────────────

def _parse_response(full_text: str, full_reasoning: str) -> tuple[str, str, str, str]:
    """
    Route to the right channel parser based on what the model actually returned.

    HF endpoint with thinking ON  → reasoning_content holds the channel block;
                                    content holds spoken text. Use parse_channel_v6.
    HF endpoint with thinking OFF → reasoning_content is empty; the model embeds
                                    <|channel>...<channel|> in content as usual.
                                    Fall through to parse_channel so emotion tags,
                                    tool calls, and expression/animation calls are
                                    all extracted from the channel block correctly.
    Local (v5)                    → always parse_channel.
    """
    if _direct_mode and full_reasoning.strip():
        return parse_channel_v6(full_text, full_reasoning)
    return parse_channel(full_text)


# ─── Shared response processing ───────────────────────────

def _process_result(result, messages, ctx, *, is_timer, user_input, source,
                    transcription="", heard="", vision_desc="", context_block=""):
    """
    Shared tail of _handle_input: parse channel, run tool
    calls (+ optional recall follow-up), strip tags, relay TTS, store turn, log.
    Returns True if a response was produced and spoken.
    """
    global _current_emotion, _last_spoken

    full_text, expression, animation, full_reasoning = result
    print()
    spoken, channel, emotion, intensity = _parse_response(full_text, full_reasoning)
    _current_emotion = emotion

    # Tool calls in channel block (expression/animation only; memory handled by server.py)
    for tc in _parse_tool_calls(channel):
        _execute_tool_call(tc)

    channel_stored    = _TOOL_STRIP_RE.sub("", channel).strip()
    spoken, animation = _strip_emotion_tags(spoken, animation)

    # Keyword animation fallback (explicit user requests only)
    if not animation and not is_timer:
        animation = _detect_animation_request(user_input, spoken)

    # Timer dedup: suppress empty or repeated commentary
    if is_timer:
        if not spoken.strip():
            if full_reasoning.strip():
                log.info("[timer] reasoning-only — model thought but produced no spoken output (thinking budget may be too high)")
            else:
                log.info("[timer] empty response — skipping TTS")
            return False
        if spoken.strip().lower() in {s for s in _recent_spoken}:
            log.info("[timer] duplicate response — skipping TTS")
            return False
    elif not spoken.strip() and full_reasoning.strip():
        log.info("[llm] reasoning-only — model produced thinking but no spoken text")
        for tool_name, value in _extract_tool_intent_from_reasoning(full_reasoning):
            key = "expression" if tool_name == "set_expression" else "animation"
            log.info(f"[llm] reasoning-intent tool: {tool_name}({value})")
            _execute_tool_call({"name": tool_name, "arguments": {key: value}})
    _last_spoken = spoken.strip()
    _recent_spoken.append(spoken.strip().lower())

    if is_timer and spoken.strip():
        with _lock:
            _last_auto_vision_summaries.append(spoken.strip())

    _relay(_normalise_spoken(spoken), expression or _emotion_to_expression(emotion), animation)

    _session.add_assistant(
        spoken_text=spoken, channel_text=channel_stored,
        emotion=emotion, intensity=intensity,
        user_input="" if is_timer else user_input,
    )

    if _logger:
        _logger.log(
            messages=messages, raw_output=full_text, spoken_text=spoken,
            channel_text=channel, emotion=emotion, intensity=intensity,
            context_snapshot=_make_snapshot(ctx, source),
            context_block=context_block,
            vision_desc=vision_desc,
            trigger="auto_commentary" if is_timer else "user_input",
            **({"user_text": transcription} if transcription else {}),
        )

    if _session_logger and is_timer:
        _session_logger.log(
            trigger="auto_commentary",
            heard=heard,
            vision=vision_desc,
            thought=channel_stored,
            emotion=emotion,
            intensity=intensity,
            spoke=spoken,
            images=[v["file"] for v in ctx.get("vision", []) if v.get("file")] or None,
            game=ctx.get("stream", {}).get("game", ""),
            scene=ctx.get("stream", {}).get("scene", ""),
        )

    if _session.needs_summary():
        _session.trigger_async_summary(lambda msgs: (lambda r: r[0] if r else None)(_call_llm(msgs, enable_thinking=False)))
    return True


def _acquire_llm(blocking: bool, label: str) -> bool:
    """Acquire _llm_lock honoring mute + drop-if-busy semantics. False = skip."""
    if blocking:
        _llm_lock.acquire()
        return True
    if STT_MUTE_LOCK.exists():
        log.debug(f"mute lock active, dropping {label}")
        return False
    if not _llm_lock.acquire(blocking=False):
        log.debug(f"LLM busy, dropping {label}")
        return False
    return True


# ─── Core pipelines ───────────────────────────────────────

def _handle_input(user_input: str, source: str = "mic", _blocking: bool = False):
    """
    Core reaction pipeline. Timer fires are the only path that calls the LLM.

    source="timer": reads _ben_buffer + vision description, builds context,
                    adds accumulated [ben] turns to history, calls LLM (thinking OFF),
                    then clears the buffered [ben] lines on success.
    source="chat":  immediate per-utterance response (viewer chat stays responsive).
    source="ben":   no longer reaches here — mic utterances are accumulated via
                    recv_transcription() and sent via the timer path.
    """
    is_timer = source == "timer"
    if not _acquire_llm(_blocking, "input"):
        return
    try:
        log.info(f"[{source}] {user_input[:80]}")
        ctx = _fetch_context()

        # Timer novelty check — skip if the scene hasn't changed since last fire.
        # Always fires when Ben has buffered speech so his words never get dropped.
        if is_timer:
            global _last_timer_img, _last_fired_ben_texts
            vision_raw = [e for e in ctx.get("vision", []) if e.get("b64")]
            cur = vision_raw[0]["b64"] if vision_raw else None
            with _ben_lock:
                cur_ben_texts = [b["text"] for b in _ben_buffer if not b.get("reacted", False)]
            has_new_ben = cur_ben_texts != _last_fired_ben_texts
            if not has_new_ben and cur is not None and cur == _last_timer_img:
                log.info("[timer] scene unchanged, no new speech — skipping fire")
                return
            _last_timer_img       = cur
            _last_fired_ben_texts = cur_ben_texts

        # Snapshot [ben] buffer before any async activity.
        # ben_context — full buffer (all entries, for [BEN SAID RECENTLY] awareness).
        # ben_snapshot — unreacted entries only (for session-turn injection + drain).
        with _ben_lock:
            ben_context  = list(_ben_buffer)
            ben_snapshot = [b for b in _ben_buffer if not b.get("reacted", False)]
            snapshot_len = len(_ben_buffer)

        if ben_context:
            for b in ben_context:
                age = time.time() - b["ts"]
                log.info(f"  [ben buf] reacted={b.get('reacted',False)} age={age:.0f}s  {b['text'][:80]}")
        else:
            log.info("  [ben buf] empty")

        vision_desc   = _fetch_vision_description()
        log.info(f"  [vision]  {vision_desc[:120] if vision_desc else '(empty)'}")

        context_block = _build_context_block(ctx, is_timer=is_timer,
                                             vision_description=vision_desc,
                                             ben_lines=ben_context)
        log.info(f"  [ctx] {context_block[:300]}")

        # Add accumulated unreacted [ben] lines to session history as a single user turn
        if is_timer and ben_snapshot:
            ben_text = "\n".join(
                f"[Ben {datetime.fromtimestamp(b['ts']).strftime('%H:%M:%S')}]: {b['text']}"
                for b in ben_snapshot
            )
            _session.add_user(ben_text, source="timer")
            log.info(f"[timer] {len(ben_snapshot)} buffered [ben] line(s) added to history")

        if not is_timer:
            _session.add_user(user_input, source=source)

        # Assemble heard for session log — unreacted Ben speech (what was added to history).
        if is_timer:
            heard = "\n".join(
                f"{datetime.fromtimestamp(b['ts']).strftime('%H:%M:%S')} — {b['text']}"
                for b in ben_snapshot
            )
        else:
            heard = user_input

        messages = _session.build_messages(context_block, vision_frames=None)
        result   = _call_llm(messages, enable_thinking=False)

        if result:
            _process_result(result, messages, ctx,
                            is_timer=is_timer, user_input=user_input, source=source,
                            heard=heard, vision_desc=vision_desc,
                            context_block=context_block)
            if is_timer and snapshot_len > 0:
                with _ben_lock:
                    del _ben_buffer[:snapshot_len]
        elif not is_timer:
            with _session._lock:
                if _session.history and _session.history[-1]["role"] == "user":
                    _session.history.pop()
        else:
            # Timer LLM failed — pop the [ben] user turn we added so history stays clean
            if ben_snapshot:
                with _session._lock:
                    if _session.history and _session.history[-1]["role"] == "user":
                        _session.history.pop()
    except Exception as e:
        log.error(f"_handle_input crashed: {e}", exc_info=True)
    finally:
        _llm_lock.release()


# ─── Silence timer (auto-commentary) ──────────────────────

def _silence_loop():
    """
    Orchestrator-owned auto-commentary. Fires a timer-triggered response when
    Rachel has been silent >= SILENCE_THRESHOLD, the LLM is free, mute is
    inactive, and the stream is active. Silence is the only trigger.
    """
    global _last_spoke_at
    log.info(f"[silence] loop started (threshold={SILENCE_THRESHOLD}s poll={SILENCE_POLL}s)")
    last_skip_log = 0.0
    while True:
        time.sleep(SILENCE_POLL)
        with _lock:
            quiet_for = time.time() - _last_spoke_at
        if quiet_for < SILENCE_THRESHOLD:
            continue

        # Past the threshold — log *why* we skip, at most once every 5 s
        skip = None
        if _llm_lock.locked():
            skip = "LLM busy"
        elif STT_MUTE_LOCK.exists():
            skip = "mute lock"
        elif not _stream_active(_fetch_context()):
            skip = "stream inactive (obs_bridge /context active=false or unreachable)"

        if skip:
            now = time.time()
            if now - last_skip_log >= 5.0:
                log.info(f"[silence] quiet {quiet_for:.0f}s but skipping — {skip}")
                last_skip_log = now
            continue

        log.info(f"[silence] firing commentary after {quiet_for:.0f}s quiet")
        with _lock:
            _last_spoke_at = time.time()   # reset NOW so a failed/empty fire can't re-trigger instantly
        _handle_input("", "timer")


# ─── Input queue worker ───────────────────────────────────

def _queue_worker():
    """Drain _input_queue sequentially; pipelined TTS keeps the next call ready."""
    while True:
        item = _input_queue.get()
        try:
            if item.get("kind") == "text":
                _handle_input(item["text"], item.get("source", "mic"), _blocking=True)
        except Exception as e:
            log.error(f"queue_worker: {e}", exc_info=True)
        finally:
            _input_queue.task_done()


def _start_background_threads():
    if REACTION_MODE == "frames":
        threading.Thread(target=_reaction_loop, name="reaction-loop", daemon=True).start()
        return
    threading.Thread(target=_queue_worker, name="input-queue",   daemon=True).start()
    threading.Thread(target=_silence_loop, name="silence-timer", daemon=True).start()


# ═══ Adaptive Loop: frames reaction path (REACTION_MODE=frames) ═══════════

ROW_SCHEMA_VERSION = 2   # v2: plan + playback_trace replace parsed_segments


@dataclass
class Window:
    window_id: str
    start: float
    end: float
    reason: str
    speech: list[dict] = field(default_factory=list)   # [{"t", "text"}]
    chat:   list[dict] = field(default_factory=list)   # [{"t", "user", "text"}]


def input_kind(speech: list, chat: list) -> str:
    """frames | frames+speech | frames+chat | frames+chat+speech"""
    parts = ["frames"]
    if chat:
        parts.append("chat")
    if speech:
        parts.append("speech")
    return "+".join(parts)


class WindowAccumulator:
    """
    Collects [ben] speech and chat with timestamps and decides when to flush.
    Thread-safe; HTTP handlers add, the reaction loop polls/flushes.

    Flush policy (config):
      busy (reaction in flight / TTS speaking / mute) → never; keep accumulating
      direct chat/event and age ≥ WINDOW_MIN_S        → "direct"
      content and age ≥ WINDOW_MAX_S                  → "hard_max" (even mid-utterance)
      content and age ≥ WINDOW_S and VAD idle         → "window"
      no content: window slides; when quiet ≥ silence_s, VAD idle and stream
      active                                          → "silence" (frames only)
    """

    def __init__(self, window_s=WINDOW_S, max_s=WINDOW_MAX_S, min_s=WINDOW_MIN_S,
                 silence_s=SILENCE_THRESHOLD, clock=time.time):
        self.window_s, self.max_s, self.min_s, self.silence_s = window_s, max_s, min_s, silence_s
        self.clock = clock
        self._lock = threading.Lock()
        self.start = clock()
        self.speech: list[dict] = []
        self.chat: list[dict] = []
        self.direct = False

    def _first_content(self, t: float):
        # First remark into an idle (sliding) window: leave at least min_s for
        # follow-up before a normal flush instead of reacting to every sentence.
        if not (self.speech or self.chat):
            self.start = max(self.start, t - (self.window_s - self.min_s))

    def add_speech(self, text: str, t: float | None = None):
        t = self.clock() if t is None else t
        with self._lock:
            self._first_content(t)
            self.speech.append({"t": t, "text": text})

    def add_chat(self, user: str, text: str, t: float | None = None, direct: bool = False):
        t = self.clock() if t is None else t
        with self._lock:
            self._first_content(t)
            self.chat.append({"t": t, "user": user, "text": text})
            self.direct = self.direct or direct

    def poll(self, *, vad_idle: bool, busy: bool, quiet_for: float, stream_active=lambda: True) -> str | None:
        now = self.clock()
        with self._lock:
            has_content = bool(self.speech or self.chat)
            if not has_content:
                self.start = max(self.start, now - self.window_s)
            age = now - self.start
            if busy:
                return None
            if has_content:
                if self.direct and age >= self.min_s:
                    return "direct"
                if age >= self.max_s:
                    return "hard_max"
                if age >= self.window_s and vad_idle:
                    return "window"
                return None
        if age >= self.window_s and quiet_for >= self.silence_s and vad_idle and stream_active():
            return "silence"
        return None

    def flush(self, reason: str, window_id: str | None = None) -> Window:
        now = self.clock()
        with self._lock:
            w = Window(window_id or uuid.uuid4().hex[:12], self.start, now, reason,
                       sorted(self.speech, key=lambda s: s["t"]), sorted(self.chat, key=lambda c: c["t"]))
            self.start, self.speech, self.chat, self.direct = now, [], [], False
        return w


def _similar(a: str, b: str) -> float:
    norm = lambda s: re.sub(r"\[[^\]]*\]|[^a-z0-9 ]", "", s.lower()).strip()
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


class ReactionEngine:
    """
    Window → frames → prompt → model → parse (Plan) → play (Sequencer) → log.
    All I/O is injected so a recorded window replays deterministically in tests:

      model_fn(turns) -> str | None                   raw model output
      frames_fn(seconds) -> {"window_id", "frames": [{"path", "t"}], "chat": [...]}
      player: .play(plan) -> PlaybackTrace            playback.Sequencer (blocks until the reply ends)
      row_logger: .log_reaction(record) -> row_id     (training/training_logger.py)

    The row is logged after playback so it carries the PlaybackTrace; a
    playback error is caught and the row is still logged.
    """

    def __init__(self, *, system: str, model_fn, frames_fn, player, row_logger=None,
                 clock=time.time,
                 adapter_version_fn=lambda: None, sampling: dict | None = None,
                 history_pairs: int = REACTION_HISTORY_PAIRS,
                 stream_active_fn=lambda: True, mute_fn=lambda: False,
                 silence_s: float = SILENCE_THRESHOLD):
        self.system = system
        self.prompt_version = prompt_format.prompt_version(system)
        self.model_fn, self.frames_fn, self.player, self.row_logger = model_fn, frames_fn, player, row_logger
        self.clock = clock
        self.adapter_version_fn = adapter_version_fn
        self.sampling = sampling or {}
        self.stream_active_fn, self.mute_fn = stream_active_fn, mute_fn
        self.acc = WindowAccumulator(clock=clock, silence_s=silence_s)
        self.history: deque = deque(maxlen=history_pairs)   # {"user", "assistant", "row_id"}
        self._recent_spoken: deque = deque(maxlen=8)
        self._vad_speaking, self._vad_ts = False, 0.0
        self._busy = threading.Event()
        self.last_spoke_at = clock()
        self.last_reaction: dict | None = None   # {id, ts, reason, latency_ms, flags} — for GET /status

    # ── inputs ────────────────────────────────────────────
    def is_echo(self, text: str) -> bool:
        return any(_similar(text, s) >= ECHO_SIMILARITY for s in self._recent_spoken)

    def on_speech(self, text: str, t: float | None = None) -> bool:
        """Buffer Ben's speech. Returns False if dropped as an echo of Rachel's own voice."""
        if self.is_echo(text):
            log.info(f"[ben] dropped echo of Rachel: {text[:60]}")
            return False
        self.acc.add_speech(text, t)
        return True

    def on_chat(self, user: str, text: str, t: float | None = None, direct: bool = True):
        self.acc.add_chat(user, text, t, direct)

    def on_vad(self, speaking: bool, t: float | None = None):
        self._vad_speaking, self._vad_ts = speaking, (self.clock() if t is None else t)

    def vad_idle(self) -> bool:
        return not self._vad_speaking or self.clock() - self._vad_ts > VAD_STALE_S

    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    # ── loop step ─────────────────────────────────────────
    def tick(self) -> dict | None:
        now = self.clock()
        reason = self.acc.poll(
            vad_idle=self.vad_idle(),
            busy=self.busy or self.mute_fn(),
            quiet_for=now - self.last_spoke_at,
            stream_active=self.stream_active_fn,
        )
        if not reason:
            return None
        if reason == "silence":
            self.last_spoke_at = now   # rate-limit the silence path even if the reply is empty
        return self.react(self.acc.flush(reason))

    def react(self, window: Window) -> dict | None:
        self._busy.set()
        try:
            return self._react(window)
        finally:
            self._busy.clear()

    def _react(self, window: Window) -> dict | None:
        seconds = min(max(window.end - window.start, WINDOW_S), 2 * WINDOW_MAX_S)
        fw = self.frames_fn(seconds) or {}
        frames = [f["path"] for f in fw.get("frames", [])]

        # Passive chat from obs_bridge for this window, merged with direct chat (dedup on user+text).
        seen = {(c["user"], c["text"]) for c in window.chat}
        chat = list(window.chat) + [
            c for c in fw.get("chat", [])
            if c.get("t", 0) >= window.start and (c.get("user"), c.get("text")) not in seen
        ]
        chat.sort(key=lambda c: c["t"])

        history = [{"user": h["user"], "assistant": h["assistant"]} for h in self.history]
        turns = prompt_format.build_reaction_messages(
            self.system, history, frames, window.speech, chat, window.start, window.end,
        )
        t0 = time.perf_counter()
        raw = self.model_fn(turns)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        if raw is None:
            log.warning(f"[react] model call failed ({window.reason}, {latency_ms} ms)")
            return None

        plan = tag_parser.parse(raw)
        record = {
            "schema_version":  ROW_SCHEMA_VERSION,
            "ts":              window.end,
            "window_id":       fw.get("window_id") or window.window_id,
            "window_start":    window.start,
            "window_end":      window.end,
            "flush_reason":    window.reason,
            "input_kind":      input_kind(window.speech, chat),
            "frame_paths":     frames,
            "speech":          window.speech,
            "chat":            chat,
            "history_ref":     self.history[-1]["row_id"] if self.history else None,
            "prompt_version":  self.prompt_version,
            "adapter_version": self.adapter_version_fn(),
            "sampling":        self.sampling,
            "latency_ms":      latency_ms,
            "raw_output":      raw,
            "plan":            plan.to_dict(),
            "parse_flags":     plan.flags,
            "playback_trace":  None,
        }
        log.info(f"[react] {window.reason} {record['input_kind']} {len(frames)}f {latency_ms}ms "
                 f"flags={plan.flags} → {raw[:100]!r}")
        if not plan.is_noop:
            try:
                trace = self.player.play(plan)
                record["playback_trace"] = trace.to_dict() if trace is not None else None
            except Exception as e:
                log.error(f"[react] playback failed: {e}")
            self._recent_spoken.append(plan.text)
            self.last_spoke_at = self.clock()

        row_id = None
        if self.row_logger:
            try:
                row_id = self.row_logger.log_reaction(record)
            except Exception as e:
                log.error(f"[react] row log failed: {e}")
        record["id"] = row_id
        self.last_reaction = {"id": row_id, "ts": window.end, "reason": window.reason,
                              "latency_ms": latency_ms, "flags": plan.flags}

        self.history.append({"user": turns[-1].text, "assistant": raw, "row_id": row_id})
        return record


# ── HTTP adapters for the engine ──────────────────────────

def make_player():
    """playback.Sequencer over server.py's /synth, /play-clip and /avatar."""
    return playback.Sequencer(playback.TtsServerBackend(SPEAK_URL.rsplit("/", 1)[0]))


class PrintPlayer:
    """--text one-shot / preflight: print the plan instead of playing it."""

    def play(self, plan):
        for s in plan.steps:
            print(f"  [{type(s).__name__.lower()}] {s.to_dict()}")
        return playback.PlaybackTrace()


def _fetch_window_frames(seconds: float) -> dict:
    try:
        r = requests.get(f"{OBS_BRIDGE_URL}/window", params={
            "seconds": round(seconds, 2), "max_frames": WINDOW_FRAMES_MAX, "min_frames": WINDOW_FRAMES_MIN,
        }, timeout=8)
        return r.json() if r.status_code == 200 else {}
    except Exception as e:
        log.warning(f"[window] obs_bridge /window failed: {e}")
        return {}


def _model_base_url() -> tuple[str, str]:
    """(base_url without /v1..., token) for the reaction model."""
    url, token = (HF_URL, HF_TOKEN) if _direct_mode else (LLM_URL, LLM_TOKEN)
    base = re.sub(r"/(v1/)?(chat/)?completions$", "", url.rstrip("/"))
    return re.sub(r"/v1$", "", base), token


_chat_transport_warned = False


def _transport() -> str:
    """Effective transport: explicit REACTION_TRANSPORT, else chat for --direct (HF has no /completion), completion locally."""
    return REACTION_TRANSPORT or ("chat" if _direct_mode else "completion")


def _reaction_model_call(turns: list) -> str | None:
    """One reaction call, thinking off. Holds one _hf_sema slot for its duration."""
    global _chat_transport_warned
    base, token = _model_base_url()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    with _hf_sema:
        try:
            if _transport() == "completion":
                request = prompt_format.render_inference(turns)
                for attempt in range(2):
                    # Marker is per llama-server process; refetch once if a restart changed it.
                    marker = prompt_format.llamacpp_media_marker(base, refresh=attempt > 0)
                    body = prompt_format.to_llamacpp_completion(request, REPO_ROOT, marker)
                    body.update({
                        "n_predict": REACTION_MAX_TOKENS, "stream": False, "cache_prompt": True,
                        "stop": [prompt_format.EOT], **REACTION_SAMPLING,
                    })
                    lora = _llamacpp_lora_field()
                    if lora is not None:
                        body["lora"] = lora
                    r = requests.post(f"{base}/completion", headers=headers, json=body, timeout=(10, 60))
                    if r.status_code == 400 and attempt == 0 and request["images"]:
                        continue
                    r.raise_for_status()
                    data = r.json()
                    if data.get("stop_type") == "limit":
                        # Ran out of n_predict without <turn|>: a cut-off sentence, or
                        # reasoning that never reached <channel|>. Never speak it.
                        log.warning(f"[react] reply hit n_predict={REACTION_MAX_TOKENS}, dropped: "
                                    f"{data.get('content', '')[:120]!r}")
                        return None
                    return data.get("content", "")
            if not _chat_transport_warned:
                log.warning("[react] REACTION_TRANSPORT=chat — server applies its own template; "
                            "train/inference byte-identity is NOT guaranteed")
                _chat_transport_warned = True
            r = requests.post(f"{base}/v1/chat/completions", headers=headers, json={
                "model": HF_MODEL if _direct_mode else LLM_MODEL,
                "messages": prompt_format.to_chat_messages(turns, REPO_ROOT),
                "max_tokens": REACTION_MAX_TOKENS, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                **REACTION_SAMPLING,
            }, timeout=(10, 60))
            r.raise_for_status()
            return r.json()["choices"][0]["message"].get("content", "")
        except Exception as e:
            log.error(f"[react] model call error: {e}")
            return None


def _read_registry() -> dict:
    try:
        return json.loads(ADAPTER_REGISTRY.read_text(encoding="utf-8"))
    except Exception:
        return {"adapters": [], "current": None}


def _current_adapter_version() -> str | None:
    """adapters/registry.json `current` — read per reaction so promotion/rollback is one pointer change."""
    return _read_registry().get("current")


# llama.cpp LoRA (installed build, Apr 2026): adapters are loaded at startup with
# --lora (optionally --lora-init-without-apply); per-request "lora": [{"id","scale"}]
# and POST /lora-adapters change scales at runtime WITHOUT a restart. Loading a
# NEW adapter file still needs a server restart — so a promotion/rollback between
# adapters that are both loaded is instant; a brand-new nightly needs a restart
# (preflight.py prints the command).
_lora_state: dict = {"key": object(), "field": None, "applied": None}


def _resolve_lora() -> dict:
    """Map registry `current` → loaded llama-server adapter ids; cached per version."""
    version = _current_adapter_version()
    if version == _lora_state["key"]:
        return _lora_state
    field, applied = None, None
    base, _ = _model_base_url()
    try:
        loaded = {re.split(r"[\\/]", a.get("path", ""))[-1]: a["id"]
                  for a in requests.get(f"{base}/lora-adapters", timeout=3).json()}
    except Exception:
        # Server not up yet (or not llama.cpp): don't cache, retry next reaction.
        return {"key": None, "field": None, "applied": None}
    entry = next((a for a in _read_registry().get("adapters", []) if a.get("version") == version), None)
    target = re.split(r"[\\/]", entry["gguf"])[-1] if entry and entry.get("gguf") else None
    if version and target not in loaded:
        log.warning(f"[lora] current adapter {version} ({target}) is not loaded in llama-server — "
                    f"serving base_core; restart llama-server with --lora (see preflight.py)")
    if loaded:
        field = [{"id": i, "scale": 1.0 if name == target else 0.0} for name, i in loaded.items()]
        applied = version if target in loaded else None
    _lora_state.update(key=version, field=field, applied=applied)
    return _lora_state


def _llamacpp_lora_field() -> list | None:
    """Per-request LoRA scales: 1.0 for the current adapter, 0.0 for any other loaded one."""
    return _resolve_lora()["field"]


def _applied_adapter_version() -> str:
    """What actually served this reaction — logged as adapter_version in every row."""
    if _transport() != "completion":
        return f"chat:{HF_MODEL if _direct_mode else LLM_MODEL}"
    return _resolve_lora()["applied"] or "base_core"


_engine: ReactionEngine | None = None


def _make_engine(row_logger=None, player=None) -> ReactionEngine:
    return ReactionEngine(
        system=prompt_format.load_system_prompt(),
        model_fn=_reaction_model_call,
        frames_fn=_fetch_window_frames,
        player=player or make_player(),
        row_logger=row_logger,
        adapter_version_fn=_applied_adapter_version,
        sampling={**REACTION_SAMPLING, "max_tokens": REACTION_MAX_TOKENS, "transport": _transport()},
        stream_active_fn=lambda: _stream_active(_fetch_context()),
        mute_fn=STT_MUTE_LOCK.exists,
    )


def _reaction_loop():
    log.info(f"[react] loop started  window={WINDOW_S}s max={WINDOW_MAX_S}s min={WINDOW_MIN_S}s "
             f"frames={WINDOW_FRAMES_MIN}–{WINDOW_FRAMES_MAX} lead={LEAD_S}s "
             f"transport={_transport()} prompt={_engine.prompt_version}")
    while True:
        try:
            _engine.tick()
        except Exception as e:
            log.error(f"[react] tick crashed: {e}", exc_info=True)
        time.sleep(WINDOW_POLL_S)


# ─── Flask ────────────────────────────────────────────────

app = Flask(__name__)
app.logger.setLevel(logging.WARNING)


def _is_valid_input(text: str) -> bool:
    """Reject STT hallucinations — transcriptions must be mostly ASCII/Latin."""
    if not text.strip():
        return False
    return sum(1 for c in text if ord(c) < 128) / max(len(text), 1) > 0.7


@app.post("/transcription")
def recv_transcription():
    d    = request.get_json(force=True, silent=True) or {}
    text = d.get("text", "").strip()
    if text and _is_valid_input(text) and _engine is not None:
        if _engine.on_speech(text):
            log.info(f"[ben] window += {text[:80]}")
    elif text and _is_valid_input(text):
        cutoff = time.time() - BEN_BUFFER_WINDOW_S
        with _ben_lock:
            _ben_buffer.append({"ts": time.time(), "text": text, "reacted": False})
            while _ben_buffer and _ben_buffer[0]["ts"] < cutoff:
                _ben_buffer.pop(0)
        log.info(f"[ben] buffered: {text[:80]}")
        _notify_perceiver_ben_spoke(text)
    elif text:
        log.info(f"[stt] rejected non-English input: {text[:50]}")
    return jsonify({"status": "ok"})


@app.post("/chat-input")
def recv_chat_input():
    """Viewer chat that should trigger a response (distinct from passive /chat-message)."""
    d    = request.get_json(force=True, silent=True) or {}
    user = d.get("user", "viewer").strip()
    text = d.get("text", "").strip()
    if not text:
        return jsonify({"status": "ok"})
    if not _is_valid_input(text):
        log.info(f"[chat-input] rejected non-Latin input: {text[:50]}")
        return jsonify({"status": "ok"})
    if _engine is not None:
        # Direct input: flush the window early once WINDOW_MIN_S has passed.
        if user.lower() == "ben":
            if _engine.on_speech(text):
                _engine.acc.direct = True
        else:
            _engine.on_chat(user or "viewer", text, direct=True)
        return jsonify({"status": "ok"})
    # Ben speaking via --respond: buffer with reacted=True so the timer doesn't re-fire it
    if user.lower() == "ben":
        cutoff = time.time() - BEN_BUFFER_WINDOW_S
        with _ben_lock:
            _ben_buffer.append({"ts": time.time(), "text": text, "reacted": True})
            while _ben_buffer and _ben_buffer[0]["ts"] < cutoff:
                _ben_buffer.pop(0)
        _notify_perceiver_ben_spoke(text)
        log.info(f"[ben→react] {text[:80]}")
    formatted = f"{user}: {text}" if user and user.lower() != "viewer" else text
    _enqueue_input({"kind": "text", "text": formatted, "source": "chat"})
    return jsonify({"status": "ok"})


@app.post("/audio-input")
def recv_audio_input():
    """stt-server: accumulate transcription text into _ben_buffer. Raw audio blob is ignored."""
    d          = request.get_json(force=True, silent=True) or {}
    transcript = d.get("transcription", "").strip()
    if not transcript and d.get("audio_b64"):
        log.warning("[audio-input] received raw audio with no transcription — stt-server should transcribe locally before posting")
        return jsonify({"status": "ok"})
    if transcript and _is_valid_input(transcript) and _engine is not None:
        _engine.on_speech(transcript)
    elif transcript and _is_valid_input(transcript):
        cutoff = time.time() - BEN_BUFFER_WINDOW_S
        with _ben_lock:
            _ben_buffer.append({"ts": time.time(), "text": transcript, "reacted": False})
            while _ben_buffer and _ben_buffer[0]["ts"] < cutoff:
                _ben_buffer.pop(0)
        log.info(f"[ben/audio] buffered: {transcript[:80]}")
        _notify_perceiver_ben_spoke(transcript)
    return jsonify({"status": "ok"})


@app.post("/vad")
def recv_vad():
    """stt-server VAD start/end — windows don't flush (except hard max) while Ben is mid-utterance."""
    d = request.get_json(force=True, silent=True) or {}
    if _engine is not None:
        _engine.on_vad(bool(d.get("speaking")))
    return jsonify({"status": "ok"})


_EVENT_LABELS = {
    "channel.subscribe":         "new sub",
    "channel.subscription.gift": "gifted subs",
    "channel.cheer":             "cheer",
    "channel.raid":              "incoming raid",
    "channel.channel_points_custom_reward_redemption.add": "channel points redeemed",
}


@app.post("/event")
def recv_event():
    """Twitch event forwarded by obs_bridge — a direct trigger in frames mode."""
    d = request.get_json(force=True, silent=True) or {}
    label = _EVENT_LABELS.get(d.get("type", ""))
    if _engine is not None and label:
        data = d.get("data", {}) or {}
        who  = data.get("user_name") or data.get("from_broadcaster_user_name") or ""
        qty  = data.get("total", data.get("bits", ""))
        text = " ".join(p for p in (label, f"from {who}" if who else "", f"({qty})" if qty else "") if p)
        _engine.on_chat("twitch", text, direct=True)
    return jsonify({"status": "ok"})


@app.get("/status")
def get_status():
    """Cheap engine state for the control panel. Never calls the model."""
    if _engine is None:
        return jsonify({"mode": REACTION_MODE, "engine": False})
    acc, now = _engine.acc, time.time()
    with acc._lock:
        speech_n, chat_n, direct, age = len(acc.speech), len(acc.chat), acc.direct, now - acc.start
    return jsonify({
        "mode": REACTION_MODE, "engine": True, "transport": _transport(),
        "busy": _engine.busy, "muted": STT_MUTE_LOCK.exists(), "vad_idle": _engine.vad_idle(),
        "speech_n": speech_n, "chat_n": chat_n, "direct": direct, "window_age_s": round(age, 1),
        "quiet_for_s": round(now - _engine.last_spoke_at, 1),
        "last_reaction": _engine.last_reaction, "adapter": _lora_state.get("applied") or "base_core",
        "prompt_version": _engine.prompt_version,
    })


@app.get("/vision-test")
def vision_test():
    """
    Diagnostic: grab the latest screenshot, send it to the LLM with a plain
    'describe this image' prompt, and return the raw response.
    GET http://localhost:8052/vision-test to verify the model is seeing images.
    """
    ctx = _fetch_context()
    vision_entries = ctx.get("vision", [])
    if not vision_entries:
        return jsonify({"error": "no screenshots in obs_bridge context"})
    b64 = vision_entries[0].get("b64", "")
    if not b64:
        return jsonify({"error": "screenshot has no b64 data"})

    from session import SessionManager
    test = SessionManager(system_prompt="You are a vision assistant. Describe exactly what you see.", max_pairs=1)
    test.add_user("Describe exactly what you see in this screenshot. List every visible element.", source="ben")
    result = _call_llm(test.build_messages("", vision_frames=[{"b64": b64, "seconds_ago": 0}]))
    if result:
        log.info(f"[vision-test] {result[0][:200]}")
        return jsonify({"response": result[0]})
    return jsonify({"error": "LLM call failed"})


# ─── Entry point ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rachel V5 Orchestrator")
    parser.add_argument("--direct", action="store_true", help="Use HF inference endpoint instead of local llama.cpp")
    parser.add_argument("--text", metavar="TEXT", help="One-shot test: send TEXT, print response, exit")
    parser.add_argument("--no-log", action="store_true", help="Disable training logger")
    parser.add_argument("--mode", choices=("frames", "text"), default=REACTION_MODE,
                        help=f"Reaction path (default from REACTION_MODE: {REACTION_MODE})")
    args = parser.parse_args()

    _direct_mode = args.direct
    REACTION_MODE = args.mode

    # Clear any stale mute lock from a crashed previous run. Normally stt-server
    # owns this, but the orchestrator can run standalone (test/chat-only), so a
    # leftover lock would otherwise block the silence timer forever.
    try:
        if STT_MUTE_LOCK.exists():
            STT_MUTE_LOCK.unlink()
            log.info("Cleared stale STT mute lock at startup")
    except Exception as e:
        log.warning(f"Could not clear stale mute lock: {e}")

    # ── Frames reaction path (Adaptive Loop) ──────────────
    if REACTION_MODE == "frames":
        log.info(f"Reaction mode: frames → {'HF endpoint' if _direct_mode else 'local llama.cpp'}  "
                 f"transport={_transport()}  adapter={_applied_adapter_version()}")
        if args.text:
            # One-shot: treat TEXT as Ben's speech in a window of the latest frames; print, don't speak or log.
            _engine = _make_engine(row_logger=None, player=PrintPlayer())
            _engine.on_speech(args.text)
            record = _engine.react(_engine.acc.flush("oneshot"))
            if record:
                print(f"\n[prompt {record['prompt_version']} | {len(record['frame_paths'])} frames | "
                      f"{record['latency_ms']} ms | flags={record['parse_flags']}]")
                print(f"[raw] {record['raw_output']}")
            else:
                print("[no response — model call failed]")
            raise SystemExit(0)

        _row_logger = None
        if not args.no_log:
            try:
                from training.training_logger import ReactionRowLogger
                _row_logger = ReactionRowLogger(ROWS_DIR)
                log.info(f"Row logger ready → {ROWS_DIR}")
            except Exception as e:
                log.warning(f"Row logger unavailable: {e}")
        _engine = _make_engine(row_logger=_row_logger)
        log.info("Orchestrator starting (frames)...")
        _start_background_threads()
        log.info(f"Listening on port {ORCHESTRATOR_PORT}")
        app.run(host="0.0.0.0", port=ORCHESTRATOR_PORT, threaded=True)
        raise SystemExit(0)

    # ── Load workspace (Layer 0) — legacy text path ───────
    parts = []
    for fname in ("SOUL.md", "USER.md"):
        path = WORKSPACE_DIR / fname
        try:
            parts.append(path.read_text(encoding="utf-8").strip())
        except FileNotFoundError:
            log.warning(f"Workspace file not found: {path}")
    _system_prompt = "\n\n---\n\n".join(parts) + (
        "\n\n---\n\n"
        "OUTPUT FORMAT RULE: Never start your response with an emotion or intensity tag such as "
        "[medium/surprise], [joy/high], <medium/sarcasm>, or any variation of [emotion/intensity] "
        "or [intensity/emotion]. Do not prefix spoken text with bracket tags of any kind. "
        "Begin your response directly with spoken words."
    )
    log.info(f"Workspace loaded: {len(_system_prompt)} chars")

    # ── Session ───────────────────────────────────────────
    from session import SessionManager
    _session = SessionManager(system_prompt=_system_prompt, max_pairs=SESSION_MAX_TURNS)

    # ── Training logger ───────────────────────────────────
    if not args.no_log:
        try:
            from training.training_logger import TrainingLogger
            _logger = TrainingLogger(output_dir=TRAINING_DATA_DIR)
            log.info(f"Training logger ready → {TRAINING_DATA_DIR}")
        except Exception as e:
            log.warning(f"Training logger unavailable: {e}")

    # ── Session logger (fat episodic log for dream.py) ────
    if not args.no_log:
        try:
            from session_logger import SessionLogger
            _session_logger = SessionLogger(output_dir=SESSION_LOG_DIR)
            log.info(f"Session logger ready → {SESSION_LOG_DIR}")
        except Exception as e:
            log.warning(f"Session logger unavailable: {e}")

    # ── One-shot test mode ────────────────────────────────
    if args.text:
        log.info(f"Test mode → {'HF endpoint' if _direct_mode else 'local llama.cpp'}  "
                 f"model={HF_MODEL if _direct_mode else LLM_MODEL}")
        ctx           = _fetch_context()
        vision_desc   = _fetch_vision_description()
        context_block = _build_context_block(ctx, vision_description=vision_desc)
        _session.add_user(args.text, source="ben")
        messages = _session.build_messages(context_block, vision_frames=None)
        print(f"\n[system: {len(_system_prompt)} chars | context: {len(context_block)} chars | messages: {len(messages)}]\n")
        result = _call_llm(messages)
        if result:
            full_text, expression, animation, full_reasoning = result
            spoken, channel, emotion, intensity = _parse_response(full_text, full_reasoning)
            print(f"\n\n[channel: {emotion}/{intensity}]")
            print(f"[spoken: {len(spoken)} chars]")
            print(f"[expression={expression}  animation={animation}]")
        raise SystemExit(0)

    log.info(f"LLM: {'HF endpoint' if _direct_mode else 'local llama.cpp'}  "
             f"url={HF_URL if _direct_mode else LLM_URL}  model={HF_MODEL if _direct_mode else LLM_MODEL}")
    log.info("Orchestrator starting...")
    _start_background_threads()
    log.info(f"Listening on port {ORCHESTRATOR_PORT}")
    app.run(host="0.0.0.0", port=ORCHESTRATOR_PORT, threaded=True)