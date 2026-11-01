"""
legacy_text_path.py — parsing helpers for the legacy text-only reaction path
(REACTION_MODE=text). Moved verbatim out of orchestrator.py so that the new
frames path (tags.py / tag_parser.py) is the only live tag grammar.

These understand the pre-Adaptive-Loop output formats (<|channel> emotion
headers, [emotion] brackets, set_expression/play_animation calls). Kept for
A/B and rollback only — do not use them on the frames reaction path.
"""

import json
import logging
import re

import requests

from config import SPEAK_URL

log = logging.getLogger("orchestrator")


# ─── Regex ────────────────────────────────────────────────

_ALL_TAG_RE    = re.compile(r'\[([^\]]+)\]', re.IGNORECASE)
_EXPR_PAT      = re.compile(r'set_expression[({]\s*(?:expression\s*[:=]\s*)?[\'"]?([a-zA-Z_]+)[\'"]?\s*[)}]')
_ANIM_PAT      = re.compile(r'play_animation[({]\s*(?:animation\s*[:=]\s*)?[\'"]?([a-zA-Z_]+)[\'"]?\s*[)}]')
_TOOL_CALL_RE  = re.compile(r'<\|tool_call>(.*?)<tool_call\|>', re.DOTALL)
_TOOL_STRIP_RE = re.compile(r'<\|tool_call>.*?<tool_call\|>', re.DOTALL)

# Emotion tags the model sometimes drops in spoken text instead of vocal tags.
_EMOTION_TAG_RE = re.compile(
    r'\[(joy|excitement|surprise|anger|sadness|fear|disgust|love|concern|thinking|neutral)\]',
    re.IGNORECASE,
)
_EMOTION_TO_ANIM = {
    "joy": "bounce", "excitement": "dance_1", "surprise": "shrug", "anger": "shrug",
    "sadness": "nod", "fear": "shrug", "disgust": "shrug", "love": "clap",
    "concern": "nod", "thinking": "head_tilt", "neutral": None,
}

_EMOTION_TO_EXPR = {
    "joy": "joy", "sadness": "sadness", "anger": "anger", "fear": "concern",
    "surprise": "surprise", "disgust": "anger", "love": "joy", "sarcasm": "thinking",
    "neutral": "neutral", "excitement": "excitement", "concern": "concern", "thinking": "thinking",
}

_INTENSITIES = {"low", "medium", "high"}


def _strip_emotion_tags(spoken: str, current_animation: str | None) -> tuple[str, str | None]:
    """Remove [emotion] bracket tags from spoken text; derive animation if none set yet."""
    anim = current_animation
    for m in _EMOTION_TAG_RE.finditer(spoken):
        if anim is None:
            anim = _EMOTION_TO_ANIM.get(m.group(1).lower())
    cleaned = re.sub(r"\s{2,}", " ", _EMOTION_TAG_RE.sub("", spoken)).strip()
    return cleaned, anim


def _emotion_to_expression(emotion: str) -> str:
    return _EMOTION_TO_EXPR.get(emotion.lower(), "neutral")


# ─── Special-token escaping ───────────────────────────────

# ─── Tool call helpers ────────────────────────────────────

def _parse_tool_calls(channel_text: str) -> list[dict]:
    """Extract all <|tool_call>...<tool_call|> blocks from channel text."""
    calls = []
    for m in _TOOL_CALL_RE.finditer(channel_text):
        try:
            calls.append(json.loads(m.group(1).strip()))
        except json.JSONDecodeError:
            log.warning(f"Malformed tool call JSON: {m.group(1)[:80]}")
    return calls


def _execute_tool_call(tc: dict) -> str | None:
    """
    Execute one tool call from the channel block.
    Memory tools (write_memory, recall_memory) are handled exclusively by server.py.
    Returns None for all fire-and-forget tools.
    """
    name = tc.get("name", "")
    args = tc.get("arguments", {})

    if name in ("write_memory", "recall_memory"):
        log.debug(f"[tool] {name} ignored — memory is handled by server.py")
        return None

    if name in ("set_expression", "play_animation"):
        key     = "expression" if name == "set_expression" else "animation"
        value   = args.get(key)
        if value:
            try:
                requests.post(SPEAK_URL, json={"text": "", key: value}, timeout=2)
            except Exception as e:
                log.warning(f"{name} failed: {e}")
        return None

    log.warning(f"[tool] unknown tool: {name}")
    return None


# ─── Reasoning-intent tool extraction ────────────────────────────────────────

_REASONING_ANIM_SET = frozenset({
    "wave", "nod", "shrug", "bounce", "dance_1", "dance_2", "clap", "head_tilt",
})
_REASONING_EXPR_SET = frozenset({
    "neutral", "joy", "anger", "sadness", "surprise", "concern", "thinking", "excitement",
})

# Matches explicit call syntax in reasoning: play_animation(animation='wave') or play_animation('wave')
_ANIM_CALL_RE = re.compile(
    r"play_animation\s*[({]\s*(?:animation\s*[:=]\s*)?['\"`]?(\w+)['\"`]?\s*[)}]",
    re.IGNORECASE,
)
_EXPR_CALL_RE = re.compile(
    r"set_expression\s*[({]\s*(?:expression\s*[:=]\s*)?['\"`]?(\w+)['\"`]?\s*[)}]",
    re.IGNORECASE,
)


def _extract_tool_intent_from_reasoning(reasoning: str) -> list[tuple[str, str]]:
    """
    Scan free-form reasoning for play_animation / set_expression intent.
    Two-pass: explicit call syntax first, loose text fallback second.
    Returns [(tool_name, value), ...], at most one per tool type.
    """
    results: list[tuple[str, str]] = []
    low = reasoning.lower()

    # Pass 1 — explicit call syntax: play_animation(animation='wave')
    for m in _ANIM_CALL_RE.finditer(reasoning):
        val = m.group(1).lower()
        if val in _REASONING_ANIM_SET:
            results.append(("play_animation", val))
            break

    # Pass 2 — tool name present somewhere + valid value found anywhere in text
    if not any(t == "play_animation" for t, _ in results) and "play_animation" in low:
        for anim in sorted(_REASONING_ANIM_SET):
            if anim in low:
                results.append(("play_animation", anim))
                break

    for m in _EXPR_CALL_RE.finditer(reasoning):
        val = m.group(1).lower()
        if val in _REASONING_EXPR_SET:
            results.append(("set_expression", val))
            break

    if not any(t == "set_expression" for t, _ in results) and "set_expression" in low:
        for expr in sorted(_REASONING_EXPR_SET):
            if expr in low:
                results.append(("set_expression", expr))
                break

    return results


# ─── Animation keyword detection (fallback when no tool_call emitted) ────────

_ANIMATION_KEYWORDS: dict[str, list[str]] = {
    "wave":      ["wave", "hello", "hi there", "greet", "welcome"],
    "nod":       ["nod", "agree", "yes exactly", "correct", "absolutely"],
    "shrug":     ["shrug", "dunno", "whatever", "idk", "no idea"],
    "bounce":    ["bounce", "excited", "hype", "let's go"],
    "dance_1":   ["dance", "celebrate", "party", "we did it"],
    "dance_2":   ["dance_2"],
    "clap":      ["clap", "applause", "bravo", "well done"],
    "head_tilt": ["confused", "huh", "wait what", "head tilt"],
}


def _detect_animation_request(user_input: str, spoken: str) -> str | None:
    """Keyword fallback when the model didn't emit a <|tool_call> block."""
    text = (user_input + " " + spoken).lower()
    for anim, keywords in _ANIMATION_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return anim
    return None


# ─── V6 channel parser (direct / HF endpoint) ────────────

def parse_channel_v6(content: str, reasoning: str) -> tuple[str, str, str, str]:
    """
    Parse the HF/direct-mode response where the API already separates
    reasoning_content (channel) from content (spoken).

    Looks for [EMOTION: x] / [INTENSITY: y] tags on the first line of
    reasoning, falling back to neutral/medium if absent.
    Returns (spoken_text, channel_text, emotion, intensity).
    """
    emotion   = "neutral"
    intensity = "medium"
    channel   = reasoning.strip()

    if channel:
        first = channel.split("\n")[0]
        em_m = re.search(r'\[EMOTION:\s*(\w+)\]', first, re.IGNORECASE)
        in_m = re.search(r'\[INTENSITY:\s*(low|medium|high)\]', first, re.IGNORECASE)
        if em_m:
            emotion = em_m.group(1).lower()
        if in_m:
            intensity = in_m.group(1).lower()

    return content.strip(), channel, emotion, intensity


# ─── Channel output normalisation ────────────────────────

def _normalize_channel_header(text: str) -> str:
    """Fix malformed nested emotion/intensity tags the model sometimes produces."""
    text = re.sub(
        r'\[EMOTION:\s*\[emotion:\s*\[emotion:\s*(\w+)\].*?\]\s*\[intensity:\s*\w+\]\]',
        r'[EMOTION: \1]', text, flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r'\[EMOTION:\s*\[emotion:\s*(\w+)\]\s*\[intensity:\s*\w+\]\]',
        r'[EMOTION: \1]', text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r'\[INTENSITY:\s*\[intensity:\s*(\w+)\]\]',
        r'[INTENSITY: \1]', text, flags=re.IGNORECASE,
    )
    lines   = text.split('\n')
    cleaned = [lines[0]]
    for line in lines[1:]:
        if re.match(r'^\[EMOTION:\s*\w+\]\s*\[INTENSITY:\s*\w+\]$', line.strip(), re.IGNORECASE):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)


def _split_emotion_intensity(a: str, b: str) -> tuple[str, str]:
    """Given two slash-separated tokens, return (emotion, intensity)."""
    a, b = a.strip().lower(), b.strip().lower()
    if b in _INTENSITIES:
        return a, b
    if a in _INTENSITIES:
        return b, a
    return a, b   # unknown — assume emotion/intensity order


def parse_channel(full_text: str) -> tuple[str, str, str, str]:
    """
    Split Rachel's raw output into (spoken_text, channel_text, emotion, intensity).
    Handles legacy <|channel> tokens, block-bracket, and inline-bracket formats.
    """
    full_text    = _normalize_channel_header(full_text)
    emotion      = "neutral"
    intensity    = "medium"
    channel_text = ""
    spoken_text  = full_text.strip()

    # 1. Legacy <|channel>...<channel|> tokens
    if "<|channel>" in full_text and "<channel|>" in full_text:
        pre, rest    = full_text.split("<|channel>", 1)
        parts        = rest.split("<channel|>", 1)
        channel_text = parts[0].strip()
        spoken_text  = (parts[1].strip() if len(parts) > 1 else "").strip()
        if pre.strip():
            spoken_text = f"{pre.strip()} {spoken_text}".strip()
        first = channel_text.split("\n")[0].strip()
        em_m  = re.search(r'\[EMOTION:\s*(\w+)\]', first, re.IGNORECASE)
        in_m  = re.search(r'\[INTENSITY:\s*(low|medium|high)\]', first, re.IGNORECASE)
        if em_m:
            emotion   = em_m.group(1).lower()
            intensity = in_m.group(1).lower() if in_m else intensity
        elif "/" in first:
            emotion, intensity = _split_emotion_intensity(*first.split("/", 1))
        else:
            emotion = first.lower()
        return spoken_text, channel_text, emotion, intensity

    # 2. Block bracket: [emotion/intensity\n...\n] at start
    block = re.match(r'^\[([a-z]+)/([a-z]+)[^\]]*\]\s*', spoken_text, re.IGNORECASE | re.DOTALL)
    if block:
        emotion, intensity = _split_emotion_intensity(block.group(1), block.group(2))
        return spoken_text[block.end():].strip(), block.group(0).strip(), emotion, intensity

    # 3. Inline bracket: strip all leading [x/y] tags
    inline = re.match(r'^(\[[a-z]+/[a-z]+\]\s*)+', spoken_text, re.IGNORECASE)
    if inline:
        tag = re.search(r'\[([a-z]+)/([a-z]+)\]', inline.group(0), re.IGNORECASE)
        if tag:
            emotion, intensity = _split_emotion_intensity(tag.group(1), tag.group(2))
        channel_text = inline.group(0).strip()
        spoken_text  = spoken_text[inline.end():].strip()

    return spoken_text, channel_text, emotion, intensity


def _extract_avatar_cues(full_text: str) -> tuple[str | None, str | None]:
    """Pull the last set_expression / play_animation literal from raw text."""
    expression = animation = None
    for m in _EXPR_PAT.findall(full_text):
        expression = m
    for m in _ANIM_PAT.findall(full_text):
        animation = m
    return expression, animation


_LOL_RE  = re.compile(r'\blol\b',  re.IGNORECASE)
_LMAO_RE = re.compile(r'\blmao\b', re.IGNORECASE)

def _normalise_spoken(text: str) -> str:
    """Replace laugh shorthands with TTS-friendly laugh tokens."""
    text = _LMAO_RE.sub("[laugh][laugh]", text)
    text = _LOL_RE.sub("[laugh]", text)
    return text
