import re
from pathlib import Path

import pytest

import tags

ROOT = Path(__file__).resolve().parent.parent


def test_vocabularies_match_spec():
    assert tags.EXPRESSIONS == {
        "joy", "happy", "anger", "angry", "relaxed",
        "surprised", "surprise", "concern", "sad", "sadness",
    }
    assert tags.ANIMATIONS == {
        "laugh", "bounce", "think", "done", "agree",
        "greetings", "wave", "head_tilt", "tilt", "shrug",
    }
    assert tags.VOCAL_TAGS == {
        "sigh", "groan", "chuckle", "gasp", "laugh",
        "sniff", "cough", "shush", "clear throat",
    }


def test_neutral_aliases_to_relaxed_by_default():
    assert tags.TAG_ALIASES["neutral"] == "relaxed"
    assert tags.resolve("expr", "neutral") == "relaxed"


@pytest.mark.parametrize("kind,name,expected", [
    ("expr", "joy", "joy"),
    ("expr", "JOY", "joy"),
    ("expr", " surprised ", "surprised"),
    ("anim", "head tilt", "head_tilt"),
    ("anim", "head-tilt", "head_tilt"),
    ("anim", "nod", "agree"),
    ("anim", "thinking", "think"),
    ("expr", "thinking", None),      # alias target is an anim, not valid as expr
    ("anim", "neutral", None),       # alias target is an expr, not valid as anim
    ("expr", "dance_1", None),
    ("bogus", "joy", None),
])
def test_resolve(kind, name, expected):
    assert tags.resolve(kind, name) == expected


def test_aliases_point_at_real_tags():
    for src, dst in tags.TAG_ALIASES.items():
        assert dst in tags.EXPRESSIONS or dst in tags.ANIMATIONS, (src, dst)


@pytest.mark.parametrize("text,expected", [
    ("{expr:joy} nice shot", True),
    ("{expr:joy}{anim:bounce} nice shot", True),
    ("nice {expr:joy} shot", False),
    ("{expr:joy} nice {anim:bounce} shot", False),
    ("no tags here", False),
    ("{expr:joy}", True),
    ("[sigh] {expr:sad} again?", False),   # a vocal tag is spoken content, so the expr tag is mid-line
])
def test_is_leading_only(text, expected):
    assert tags.is_leading_only(text) is expected


def test_find_tags_order_and_kinds():
    found = tags.find_tags("a {expr:joy} b {anim:wave} c")
    assert [(k, n) for _, _, k, n in found] == [("expr", "joy"), ("anim", "wave")]


def test_vocal_regex():
    assert tags.VOCAL_TAG_RE.findall("[sigh] ok [clear throat] [nope]") == ["sigh", "clear throat"]


def test_grammar_summary_lists_every_tag():
    s = tags.grammar_summary()
    for name in tags.EXPRESSIONS | tags.ANIMATIONS:
        assert name in s
    for v in tags.VOCAL_TAGS:
        assert f"[{v}]" in s


def test_grammar_hash_stable():
    assert tags.grammar_hash() == tags.grammar_hash()
    assert re.fullmatch(r"[0-9a-f]{12}", tags.grammar_hash())


def test_placement_rule_documented():
    assert "clause boundaries" in tags.PLACEMENT_RULE and "never mid-phrase" in tags.PLACEMENT_RULE
    assert tags.LEADING_ONLY_FLAG_SHARE == 0.40
    assert tags.MAX_ACT_GROUPS == 4


def test_norm_vocal():
    assert tags.norm_vocal("  Clear   Throat ") == "clear throat"
    assert tags.is_vocal("[sigh]".strip("[]"))


# Guardrail: tags.py is the only module that defines tag vocabularies.
_DEFINITION_PATTERNS = [
    re.compile(r"""^\s*\w*(EXPR|ANIM|VOCAL)\w*\s*=\s*(frozenset|set|\{|\[)""", re.M),
]
_PROJECT_PY = [
    p for p in ROOT.rglob("*.py")
    if not any(part in {"bak", "backups", "tests", "__pycache__", ".venv"} for part in p.relative_to(ROOT).parts)
]


@pytest.mark.parametrize("path", _PROJECT_PY, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_other_module_defines_tag_sets(path):
    # Legacy modules with their own (pre-grammar) avatar vocab are tracked here
    # until they are retired; the reaction path must not use them.
    legacy = {"server.py", "legacy_text_path.py", "rachel_tools.py"}
    if path.name == "tags.py" or path.name in legacy:
        pytest.skip("tags.py itself / legacy text path")
    src = path.read_text(encoding="utf-8", errors="ignore")
    for pat in _DEFINITION_PATTERNS:
        assert not pat.search(src), f"{path.name} defines a tag set; import it from tags.py"
