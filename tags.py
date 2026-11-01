"""
tags.py — the ONLY definition of Rachel's inline tag grammar.

Imported by the runtime parser (tag_parser.py), the training logger, the
reviewer, the dataset builder, eval_gate and prompt_format (which renders the
grammar into the system prompt). No other module may define tag sets — if you
need a tag list, import it from here.

Grammar
  {expr:X}   facial expression; holds from that point until the next {expr:}
  {anim:X}   one-shot body animation; fires at the start of the following text
  [vocal]    Chatterbox vocal tag; stays inline in the spoken text and is passed
             through to TTS untouched

Pure module: no I/O beyond reading two optional env vars at import.
"""

import hashlib
import os
import re

# ─── Tag vocabularies ─────────────────────────────────────

EXPRESSIONS = frozenset({
    "joy", "happy", "anger", "angry", "relaxed",
    "surprised", "surprise", "concern", "sad", "sadness",
})

ANIMATIONS = frozenset({
    "laugh", "bounce", "think", "done", "agree",
    "greetings", "wave", "head_tilt", "tilt", "shrug",
})

VOCAL_TAGS = frozenset({
    "sigh", "groan", "chuckle", "gasp", "laugh",
    "sniff", "cough", "shush", "clear throat",
})

# Expression used when nothing else is set (and the target of "neutral").
DEFAULT_EXPRESSION = os.getenv("RACHEL_DEFAULT_EXPRESSION", "relaxed")

# Off-grammar names the model is known to emit → canonical name. Resolution
# only accepts an alias if its target is valid for the tag kind it appears in.
TAG_ALIASES: dict[str, str] = {
    "neutral":    os.getenv("RACHEL_NEUTRAL_ALIAS", DEFAULT_EXPRESSION),
    "excited":    "joy",
    "excitement": "joy",
    "mad":        "angry",
    "shocked":    "surprised",
    "worried":    "concern",
    "thinking":   "think",
    "nod":        "agree",
    "hello":      "greetings",
    "hi":         "wave",
}

# ─── Placement rule ───────────────────────────────────────

MAX_ACT_GROUPS = 4          # action groups per reply; extras dropped by the parser
LEADING_ONLY_FLAG_SHARE = 0.40
MIN_CHUNK_WORDS = 3         # speech chunks between beats should be complete phrases
MAX_BEATS_PER_REPLY = 2     # reviewer: more than this reads as twitchy

# Speech before a group must end with one of these for the group to sit on a clause boundary.
CLAUSE_END = ",.!?;:…—–-"

PLACEMENT_RULE = (
    "An action beat is an {anim:X}/{expr:Y} group (adjacent tags, whitespace only "
    "between), optionally followed directly by [vocal] tags that sound with it. "
    "Groups sit at clause boundaries only — the start, the end, or right after a "
    "comma/period/etc. — never mid-phrase. Speech between beats should be complete "
    f"phrases of ~{MIN_CHUNK_WORDS}+ words. At most {MAX_ACT_GROUPS} groups per reply "
    f"(parser cap); the reviewer prefers ≤{MAX_BEATS_PER_REPLY}. Leading-only replies are "
    f"legal, but a batch with more than {int(LEADING_ONLY_FLAG_SHARE * 100)}% leading-only "
    "tagged replies is flagged."
)

# ─── Patterns ─────────────────────────────────────────────

# Well-formed brace tag. Name is permissive so unknown names are captured and
# can be reported; validity is decided by resolve().
BRACE_TAG_RE = re.compile(r"\{\s*(expr|anim)\s*:\s*([A-Za-z0-9_ \-]+?)\s*\}", re.IGNORECASE)

# Any {...} span at all — anything this matches that BRACE_TAG_RE does not is malformed.
ANY_BRACE_RE = re.compile(r"\{[^{}]*\}")

VOCAL_TAG_RE = re.compile(
    r"\[(" + "|".join(re.escape(v) for v in sorted(VOCAL_TAGS, key=len, reverse=True)) + r")\]",
    re.IGNORECASE,
)

_KIND_SETS = {"expr": EXPRESSIONS, "anim": ANIMATIONS}


def norm_name(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def resolve(kind: str, name: str) -> str | None:
    """Canonical tag value for (kind, name), via alias map; None if invalid."""
    valid = _KIND_SETS.get(kind.lower())
    if valid is None:
        return None
    n = norm_name(name)
    if n in valid:
        return n
    alias = TAG_ALIASES.get(n)
    if alias in valid:
        return alias
    return None


def norm_vocal(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def is_vocal(name: str) -> bool:
    return norm_vocal(name) in VOCAL_TAGS


def find_tags(text: str) -> list[tuple[int, int, str, str]]:
    """[(start, end, kind, raw_name)] for every well-formed brace tag, in order."""
    return [(m.start(), m.end(), m.group(1).lower(), m.group(2)) for m in BRACE_TAG_RE.finditer(text)]


def is_leading_only(text: str) -> bool:
    """True if the reply has ≥1 brace tag and every one precedes the first spoken word."""
    tags = find_tags(text)
    if not tags:
        return False
    first_word = re.search(r"[A-Za-z0-9]", BRACE_TAG_RE.sub(lambda m: " " * len(m.group(0)), text))
    if first_word is None:
        return True
    return all(start < first_word.start() for start, _, _, _ in tags)


def grammar_summary() -> str:
    """One-paragraph grammar description rendered into the runtime system prompt."""
    return (
        "{expr:X} sets your face, X in " + ", ".join(sorted(EXPRESSIONS)) + ". "
        "{anim:X} plays a one-shot gesture, X in " + ", ".join(sorted(ANIMATIONS)) + ". "
        "Vocal sounds: " + " ".join(f"[{v}]" for v in sorted(VOCAL_TAGS)) + ". "
        "A beat is {anim}/{expr} side by side, then optionally a [vocal] right after it; "
        "put beats only at the start, the end, or after a comma or period, never mid-phrase, "
        "and keep the speech between beats to full phrases."
    )


def grammar_hash() -> str:
    """Stable short hash of the full grammar; logged in rows and checked by preflight."""
    blob = "|".join([
        ",".join(sorted(EXPRESSIONS)),
        ",".join(sorted(ANIMATIONS)),
        ",".join(sorted(VOCAL_TAGS)),
        ",".join(f"{k}>{v}" for k, v in sorted(TAG_ALIASES.items())),
        DEFAULT_EXPRESSION,
        str(MAX_ACT_GROUPS),
        "beats-v2",   # grammar semantics: action beats bind following vocals
    ])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
