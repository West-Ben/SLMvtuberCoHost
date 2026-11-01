"""
tag_parser.py — model reply → Plan of steps (grammar in tags.py). Pure, never raises.

Steps, in reply order:
  Speak(text)          plain spoken text between tags — one TTS clip
  Vocal(tag)           [sigh] [groan] [chuckle] [gasp] [laugh] [sniff] [cough] [shush] [clear throat] — one TTS clip
  Act(expr?, anim?)    adjacent {expr:X}/{anim:X} tags (whitespace only between) — one group,
                       at most one expr and one anim (a duplicate overrides, flag dup_in_group:<kind>)

An Act binds to the Vocal step(s) that directly follow it; together they are an
action beat. Timing is playback.py's job — the parser only orders steps and
never inserts expression resets.

Degradation: unknown tag → TAG_ALIASES → else dropped (unknown_tag:X); bad
braces stripped (malformed); leaked <|channel>/<|turn>/<|tool_call> markup
stripped (leak); unknown [..] dropped (unknown_vocal:X); more than
MAX_ACT_GROUPS groups → extras dropped (act_cap:N). A plan with no Speak and no
Vocal is a no-op (empty steps, flag empty).
"""

import re
from dataclasses import dataclass, field, replace

import tags
from prompt_format import EOC, GEMMA_SPECIAL_TOKENS, LLAMACPP_MEDIA_MARKER, SOC


@dataclass(frozen=True)
class Speak:
    text: str

    def to_dict(self) -> dict:
        return {"type": "speak", "text": self.text}


@dataclass(frozen=True)
class Vocal:
    tag: str

    def to_dict(self) -> dict:
        return {"type": "vocal", "tag": self.tag}


@dataclass(frozen=True)
class Act:
    expr: str | None = None
    anim: str | None = None

    def to_dict(self) -> dict:
        return {"type": "act", "expr": self.expr, "anim": self.anim}


Step = Speak | Vocal | Act


def step_from_dict(d: dict) -> Step:
    if d["type"] == "speak":
        return Speak(d["text"])
    if d["type"] == "vocal":
        return Vocal(d["tag"])
    return Act(d.get("expr"), d.get("anim"))


@dataclass
class Plan:
    steps: list = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def is_noop(self) -> bool:
        return not any(isinstance(s, (Speak, Vocal)) for s in self.steps)

    @property
    def text(self) -> str:
        """Spoken content in order (vocals as [tag]) — for echo / length / question checks."""
        return " ".join(s.text if isinstance(s, Speak) else f"[{s.tag}]"
                        for s in self.steps if isinstance(s, (Speak, Vocal)))

    @property
    def act_count(self) -> int:
        return sum(isinstance(s, Act) for s in self.steps)

    def to_dict(self) -> dict:
        return {"steps": [s.to_dict() for s in self.steps], "flags": list(self.flags)}

    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        return cls([step_from_dict(s) for s in d.get("steps", [])], list(d.get("flags", [])))


# ─── Leak stripping (Gemma special tokens, not tag grammar) ─

_THOUGHT_RE  = re.compile(re.escape(SOC) + r".*?" + re.escape(EOC), re.DOTALL)
_TOOLCALL_RE = re.compile(r"<\|tool_call>.*?<tool_call\|>", re.DOTALL)
_SPECIAL_RE  = re.compile(r"<\|[^<>\s]{0,32}?>|<[^<>\s|]{1,32}\|>")
_PARTIAL_SPECIAL_RE = re.compile(r"<\||\|>")

# Full {...} span | unclosed "{expr:name" fragment | lone brace | [bracket tag]
_TOKEN_RE = re.compile(
    r"\{[^{}]*\}|\{\s*(?:expr|anim)\s*:\s*[A-Za-z0-9_\-]*|[{}]|\[([^\[\]\n]{1,40})\]",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[^\W_]", re.UNICODE)
_SPACE_PUNCT_RE = re.compile(r"\s+([,.!?;:…])")


def _strip_leaks(raw: str, flags: list[str]) -> str:
    out = _THOUGHT_RE.sub(" ", raw)
    if EOC in out:
        # The prompt already opened the thought channel, so a <channel|> without
        # its <|channel> closes thinking the model did anyway — drop it all.
        out = " " + out.rsplit(EOC, 1)[1]
    out = _TOOLCALL_RE.sub(" ", out)
    for tok in GEMMA_SPECIAL_TOKENS | {LLAMACPP_MEDIA_MARKER}:
        out = out.replace(tok, " ")
    out = _SPECIAL_RE.sub(" ", out)
    out = _PARTIAL_SPECIAL_RE.sub(" ", out)
    if out != raw:
        flags.append("leak")
        out = re.sub(r"^\s*thought\b", " ", out)   # dangling "<|channel>thought"
    return out


def _clean(text: str) -> str:
    text = _PARTIAL_SPECIAL_RE.sub(" ", text)
    text = re.sub(r"[\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return _SPACE_PUNCT_RE.sub(r"\1", text).strip()


def _tokenize(text: str, flags: list[str]) -> list[tuple]:
    """("text", s) | ("act", kind, value) | ("vocal", tag); unknown/malformed tags vanish (flagged)."""
    events, pos = [], 0
    for m in _TOKEN_RE.finditer(text):
        events.append(("text", text[pos:m.start()]))
        pos = m.end()
        span = m.group(0)
        if span.startswith("["):
            name = tags.norm_vocal(m.group(1))
            if name in tags.VOCAL_TAGS:
                events.append(("vocal", name))
            else:
                flags.append(f"unknown_vocal:{name[:24]}")
            continue
        tm = tags.BRACE_TAG_RE.fullmatch(span)
        if tm is None:
            flags.append("malformed")
            continue
        kind, name = tm.group(1).lower(), tm.group(2)
        value = tags.resolve(kind, name)
        if value is None:
            flags.append(f"unknown_tag:{tags.norm_name(name)[:24]}")
            continue
        events.append(("act", kind, value))
    events.append(("text", text[pos:]))
    return events


def _parse(raw: str) -> Plan:
    flags: list[str] = []
    events = _tokenize(_strip_leaks(raw, flags), flags)

    steps: list = []
    buf = ""
    group_open = False      # steps[-1] is an Act and only whitespace has followed it
    dropped_open = False    # inside a group dropped by the cap
    n_groups = capped = 0

    def attach_to_last_speak(punct: str):
        for i in range(len(steps) - 1, -1, -1):
            if isinstance(steps[i], Speak):
                steps[i] = Speak(_clean(steps[i].text + punct))
                return

    def flush() -> bool:
        """Emit buffered text. True if it held anything but whitespace (breaks group adjacency)."""
        nonlocal buf
        s, buf = buf, ""
        if not s.strip():
            return False
        m = _WORD_RE.search(s)
        if m is None:                       # punctuation only: keep it with the preceding text
            attach_to_last_speak(s.strip())
            return True
        lead, rest = s[:m.start()].strip(), s[m.start():]
        if lead:
            attach_to_last_speak(lead)
        text = _clean(rest)
        if text:
            steps.append(Speak(text))
        return True

    for ev in events:
        if ev[0] == "text":
            buf += ev[1]
            continue
        if ev[0] == "vocal":
            flush()
            steps.append(Vocal(ev[1]))
            group_open = dropped_open = False
            continue
        _, kind, value = ev
        same_group = not buf.strip() and (group_open or dropped_open)
        if same_group and dropped_open:
            continue
        if same_group:
            cur = steps[-1]
            if getattr(cur, kind) is not None:
                flags.append(f"dup_in_group:{kind}")
            steps[-1] = replace(cur, **{kind: value})
            continue
        if n_groups >= tags.MAX_ACT_GROUPS:
            # Drop the whole group; don't flush so speech on both sides stays one chunk.
            capped += 1
            dropped_open, group_open = True, False
            continue
        flush()
        steps.append(Act(**{kind: value}))
        n_groups += 1
        group_open, dropped_open = True, False
    flush()

    if capped:
        flags.append(f"act_cap:{capped}")
    steps = [s for s in steps if not isinstance(s, Speak) or _WORD_RE.search(s.text)]
    if not any(isinstance(s, (Speak, Vocal)) for s in steps):
        flags.append("empty")
        return Plan([], flags)
    return Plan(steps, flags)


def parse(raw) -> Plan:
    """Never raises. Worst case: the plain text as one Speak step."""
    if not isinstance(raw, str):
        raw = "" if raw is None else str(raw)
    try:
        return _parse(raw)
    except Exception:
        plain = _clean(_TOKEN_RE.sub(" ", _SPECIAL_RE.sub(" ", raw)))
        if not _WORD_RE.search(plain):
            return Plan([], ["parser_error", "empty"])
        return Plan([Speak(plain)], ["parser_error"])
