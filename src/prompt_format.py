"""
prompt_format.py — the ONE module that builds Rachel's exact prompt text for
both training and inference.

Gemma 4 turn format, built explicitly (never via apply_chat_template, which can
strip <channel|> or inject <|think|> depending on version/kwargs):

  <bos><|turn>system\n{system}<turn|>\n
  <|turn>user\n{prior window text}<turn|>\n          ┐ history: text only,
  <|turn>model\n{prior reply, thought stripped}<turn|>\n ┘ no images, no thoughts
  <|turn>user\n<|image|><|image|>…\n{window text}<turn|>\n   ← only turn with images
  <|turn>model\n<|channel>thought\n<channel|>               ← generation prefix

Thinking is OFF. With thinking off, Gemma 4's generation prefix is an empty
thought channel; training rows carry the same prefix so the model never learns
a prompt it won't see at runtime. The completion is the reply + "<turn|>".

Images: `<|image|>` placeholders, placed before the text in the user turn. The
training processor expands each placeholder to the image soft-token span; at
inference the transport swaps the placeholder for the backend's media marker
(see to_llamacpp_completion) at the very last step, so the canonical rendered
string is byte-identical between train and inference.

Zero custom special tokens: only Gemma's own (GEMMA_SPECIAL_TOKENS).
"""

import base64
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import tags

PROMPT_FORMAT_VERSION = "rx1"

# ─── Gemma 4 special tokens (from the base tokenizer_config) ─

BOS         = "<bos>"
SOT         = "<|turn>"
EOT         = "<turn|>"
SOC         = "<|channel>"
EOC         = "<channel|>"
IMAGE_TOKEN = "<|image|>"
THINK_TOKEN = "<|think|>"

GEMMA_SPECIAL_TOKENS = frozenset({
    "<bos>", "<eos>", "<pad>", "<unk>", "<mask>",
    "<|turn>", "<turn|>", "<|channel>", "<channel|>",
    "<|image|>", "<|image>", "<image|>",
    "<|audio|>", "<|audio>", "<audio|>",
    "<|tool_call>", "<tool_call|>", "<|tool>", "<tool|>",
    "<|tool_response>", "<tool_response|>",
    "<|think|>", '<|"|>',
})

EMPTY_THOUGHT     = f"{SOC}thought\n{EOC}"
GENERATION_PREFIX = f"{SOT}model\n{EMPTY_THOUGHT}"

# Markers for Unsloth train_on_responses_only: loss only after the prefix.
INSTRUCTION_PART = f"{SOT}user\n"
RESPONSE_PART    = GENERATION_PREFIX

# llama.cpp mtmd default media marker (tools/mtmd: mtmd_default_marker()).
# Newer llama-server builds randomise it per process (GET /props "media_marker",
# pinnable via LLAMA_MEDIA_MARKER) — see llamacpp_media_marker().
LLAMACPP_MEDIA_MARKER = "<__media__>"

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "persona" / "SYSTEM_PROMPT.md"

# Matches any special-token-looking literal: <|x>, <x|>, <|x|>
_SPECIAL_LIKE_RE = re.compile(r"<\|[^<>\s]{0,32}?>|<[^<>\s|]{1,32}\|>")
_THOUGHT_BLOCK_RE = re.compile(re.escape(SOC) + r".*?" + re.escape(EOC), re.DOTALL)


# ─── Data ─────────────────────────────────────────────────

@dataclass(frozen=True)
class Turn:
    role: str                       # system | user | assistant
    text: str
    images: tuple[str, ...] = field(default_factory=tuple)   # frame paths, current window only


# ─── Text hygiene ─────────────────────────────────────────

def sanitize(text: str) -> str:
    """Remove special-token literals from untrusted text (STT, chat, history)."""
    if not text:
        return ""
    for tok in (*GEMMA_SPECIAL_TOKENS, LLAMACPP_MEDIA_MARKER):
        text = text.replace(tok, "")
    return _SPECIAL_LIKE_RE.sub("", text)


def strip_thought(text: str) -> str:
    """Drop thought-channel blocks (and any stray specials) from an assistant reply."""
    text = _THOUGHT_BLOCK_RE.sub("", text or "")
    if EOC in text:     # opener was in the prompt's generation prefix
        text = text.rsplit(EOC, 1)[1]
    return sanitize(text).strip()


# ─── System prompt ────────────────────────────────────────

def load_system_prompt(path: Path | str | None = None) -> str:
    """SYSTEM_PROMPT.md with the tag grammar from tags.py substituted in."""
    raw = Path(path or SYSTEM_PROMPT_PATH).read_text(encoding="utf-8")
    return raw.replace("{TAG_GRAMMAR}", tags.grammar_summary()).strip()


def prompt_version(system: str) -> str:
    """Format version + hash of the exact system text, e.g. 'rx1+3f9a0c1b'."""
    return f"{PROMPT_FORMAT_VERSION}+{hashlib.sha256(system.encode('utf-8')).hexdigest()[:8]}"


# ─── Window text ──────────────────────────────────────────

def format_window_text(
    speech: list[dict],
    chat: list[dict],
    n_frames: int,
    window_start: float | None = None,
    window_end: float | None = None,
) -> str:
    """
    The text half of a window's user turn.
      speech: [{"t": float, "text": str}]          Ben (STT)
      chat:   [{"t": float, "user": str, "text": str}]  viewers (untrusted)
    Times are rendered relative to window_start, merged in time order.
    """
    t0 = window_start
    if t0 is None:
        ts = [s["t"] for s in speech] + [c["t"] for c in chat]
        t0 = min(ts) if ts else 0.0

    lines: list[str] = []
    if n_frames:
        span = f" over {window_end - t0:.0f}s" if window_end is not None else ""
        lines.append(f"[{n_frames} frame{'s' if n_frames != 1 else ''}{span}, oldest first]")

    events: list[tuple[float, str]] = []
    for s in speech:
        text = sanitize(s.get("text", "")).strip()
        if text:
            events.append((s["t"], f"[ben +{max(0.0, s['t'] - t0):.1f}s] {text}"))
    for c in chat:
        text = sanitize(c.get("text", "")).strip()
        user = sanitize(c.get("user", "viewer")).strip() or "viewer"
        if text:
            events.append((c["t"], f"[chat +{max(0.0, c['t'] - t0):.1f}s] {user}: {text}"))
    events.sort(key=lambda e: e[0])
    lines += [e[1] for e in events]

    if not events:
        lines.append("(nobody said anything)")
    return "\n".join(lines)


# ─── Message assembly ─────────────────────────────────────

def build_reaction_messages(
    system: str,
    history: list[dict],
    frames: list[str],
    speech: list[dict],
    chat: list[dict],
    window_start: float | None = None,
    window_end: float | None = None,
) -> list[Turn]:
    """
    system:  rendered system prompt (load_system_prompt())
    history: [{"user": str, "assistant": str}, ...] prior windows, oldest first.
             "user" is the prior window TEXT (no images); "assistant" is the raw
             prior reply — thoughts are stripped here.
    frames:  current-window frame paths, oldest first.
    """
    turns = [Turn("system", system.strip())]
    for h in history:
        u = sanitize(h.get("user", "")).strip()
        a = strip_thought(h.get("assistant", ""))
        if not u or not a:
            continue
        turns.append(Turn("user", u))
        turns.append(Turn("assistant", a))
    text = format_window_text(speech, chat, len(frames), window_start, window_end)
    turns.append(Turn("user", text, tuple(frames)))
    return turns


# ─── Rendering ────────────────────────────────────────────

def render(turns: list[Turn], add_generation_prompt: bool = True) -> str:
    """
    Exact prompt text. Raises ValueError on structurally invalid input
    (programmer error — model output never reaches this function).
    """
    if not turns or turns[0].role != "system":
        raise ValueError("first turn must be system")
    for i, t in enumerate(turns):
        if t.role not in ("system", "user", "assistant"):
            raise ValueError(f"bad role {t.role!r}")
        if t.role == "system" and i != 0:
            raise ValueError("system turn only allowed first")
        if t.images and (t.role != "user" or i != len(turns) - 1):
            raise ValueError("images only allowed on the final (current window) user turn")

    out = [BOS, f"{SOT}system\n{turns[0].text.strip()}{EOT}\n"]
    for t in turns[1:]:
        if t.role == "assistant":
            out.append(f"{SOT}model\n{strip_thought(t.text)}{EOT}\n")
        else:
            body = IMAGE_TOKEN * len(t.images) + ("\n" if t.images else "") + t.text.strip()
            out.append(f"{SOT}user\n{body}{EOT}\n")
    if add_generation_prompt:
        out.append(GENERATION_PREFIX)
    return "".join(out)


def images_of(turns: list[Turn]) -> list[str]:
    return [p for t in turns for p in t.images]


def render_inference(turns: list[Turn]) -> dict:
    """{"prompt": str, "images": [path]} — the canonical inference request."""
    return {"prompt": render(turns, add_generation_prompt=True), "images": images_of(turns)}


def render_training_example(turns: list[Turn], response: str) -> dict:
    """
    {"prompt", "completion", "images"}. prompt is byte-identical to
    render_inference(turns)["prompt"]; completion = reply + <turn|>.
    """
    reply = strip_thought(response)
    return {
        "prompt":     render(turns, add_generation_prompt=True),
        "completion": f"{reply}{EOT}",
        "images":     images_of(turns),
    }


# ─── Transports ───────────────────────────────────────────

def _b64_file(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


_media_markers: dict[str, str] = {}
_MARKER_RE = re.compile(r"<__media_[A-Za-z0-9]*__>")
_PROBE_PNG = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
              "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


def llamacpp_media_marker(base_url: str, refresh: bool = False) -> str:
    """
    The media marker this llama-server expects, cached per URL. Newer builds
    report it in GET /props; builds that randomise it without reporting it
    (e.g. b8833) reveal it through /apply-template on an image message.
    Old builds use the fixed marker. Call with refresh=True after a failed
    request — a restart picks a new marker.
    """
    base = base_url.rstrip("/")
    if refresh or base not in _media_markers:
        import requests
        try:
            marker = requests.get(f"{base}/props", timeout=5).json().get("media_marker")
            if not marker:
                r = requests.post(f"{base}/apply-template", timeout=5, json={"messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": _PROBE_PNG}}, {"type": "text", "text": "x"}]}]})
                m = _MARKER_RE.search(r.json().get("prompt", "")) if r.ok else None
                marker = m.group(0) if m else LLAMACPP_MEDIA_MARKER
        except Exception:
            return LLAMACPP_MEDIA_MARKER        # server down — don't cache a guess
        _media_markers[base] = marker
    return _media_markers[base]


def to_llamacpp_completion(request: dict, image_root: Path | str | None = None,
                           media_marker: str = LLAMACPP_MEDIA_MARKER) -> dict:
    """
    llama.cpp /completion body. Swaps IMAGE_TOKEN for the server's media marker
    (llamacpp_media_marker()) and attaches images as multimodal_data, in order.
    The leading <bos> is dropped because the server prepends BOS itself (a
    literal one would double it). Token-for-token this matches the canonical prompt.
    """
    root = Path(image_root) if image_root else None
    imgs = [_b64_file(root / p if root and not Path(p).is_absolute() else p) for p in request["images"]]
    prompt = request["prompt"]
    if prompt.startswith(BOS):
        prompt = prompt[len(BOS):]
    prompt = prompt.replace(IMAGE_TOKEN, media_marker)
    if imgs:
        return {"prompt": {"prompt_string": prompt, "multimodal_data": imgs}}
    return {"prompt": prompt}


def to_chat_messages(turns: list[Turn], image_root: Path | str | None = None) -> list[dict]:
    """
    OpenAI-style messages for endpoints that only accept /chat/completions.
    FALLBACK ONLY: the server applies its own template, so byte-identity with
    training is NOT guaranteed. Send chat_template_kwargs={"enable_thinking": False}.
    """
    root = Path(image_root) if image_root else None
    msgs: list[dict] = []
    for t in turns:
        if t.images:
            content = [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                    _b64_file(root / p if root and not Path(p).is_absolute() else p)}}
                for p in t.images
            ]
            content.append({"type": "text", "text": t.text})
            msgs.append({"role": t.role, "content": content})
        else:
            text = strip_thought(t.text) if t.role == "assistant" else t.text
            msgs.append({"role": t.role, "content": text})
    return msgs
