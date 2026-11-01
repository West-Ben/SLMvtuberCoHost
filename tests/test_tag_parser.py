"""Response parser: raw reply → Plan of Speak / Vocal / Act steps (spec: Response Parser + Playback Sequencer)."""

import json
import random

import pytest

import tags
from tag_parser import Act, Plan, Speak, Vocal, parse

EX1 = "Ben you need to use your alt, {anim:shrug}{expr:sad} [sigh] he never listens."
EX2 = "{anim:laugh}[laugh] that was a great move!"


# ─── Worked examples ──────────────────────────────────────

def test_worked_example_1():
    p = parse(EX1)
    assert p.steps == [
        Speak("Ben you need to use your alt,"),
        Act(expr="sad", anim="shrug"),
        Vocal("sigh"),
        Speak("he never listens."),
    ]
    assert p.flags == []


def test_worked_example_2():
    p = parse(EX2)
    assert p.steps == [Act(anim="laugh"), Vocal("laugh"), Speak("that was a great move!")]
    assert p.flags == []


def test_tag_order_inside_group_does_not_matter():
    swapped = EX1.replace("{anim:shrug}{expr:sad}", "{expr:sad}{anim:shrug}")
    assert parse(swapped).steps == parse(EX1).steps


def test_whitespace_between_group_tags_still_one_group():
    assert parse("ok. {expr:sad}   {anim:shrug} [sigh] fine then.").steps == [
        Speak("ok."), Act(expr="sad", anim="shrug"), Vocal("sigh"), Speak("fine then."),
    ]


# ─── Shapes ───────────────────────────────────────────────

@pytest.mark.parametrize("raw,steps", [
    ("we lost that one. [sigh][groan] again.",
     [Speak("we lost that one."), Vocal("sigh"), Vocal("groan"), Speak("again.")]),
    ("[sigh] i guess we're doing this",
     [Vocal("sigh"), Speak("i guess we're doing this")]),
    ("nice one ben. {expr:joy} you earned that one.",
     [Speak("nice one ben."), Act(expr="joy"), Speak("you earned that one.")]),
    ("[laugh]", [Vocal("laugh")]),
    ("that's so dumb. {anim:laugh}", [Speak("that's so dumb."), Act(anim="laugh")]),
    ("that's so dumb. {anim:laugh}[laugh]", [Speak("that's so dumb."), Act(anim="laugh"), Vocal("laugh")]),
    ("{expr:surprised} wait what just happened",
     [Act(expr="surprised"), Speak("wait what just happened")]),
    ("{anim:shrug}{expr:sad}[sigh][groan] whatever you say",
     [Act(expr="sad", anim="shrug"), Vocal("sigh"), Vocal("groan"), Speak("whatever you say")]),
    ("[Clear  Throat] anyway", [Vocal("clear throat"), Speak("anyway")]),
])
def test_shapes(raw, steps):
    assert parse(raw).steps == steps


def test_punctuation_stays_with_its_text():
    # a punctuation-only chunk after a group attaches to the preceding speech
    assert parse("nice {expr:joy}.").steps == [Speak("nice."), Act(expr="joy")]
    assert parse("ok {expr:joy}, you win.").steps == [Speak("ok,"), Act(expr="joy"), Speak("you win.")]


def test_whitespace_is_collapsed():
    assert parse("  so   that   happened  ").steps == [Speak("so that happened")]


def test_punctuation_between_tags_splits_groups():
    p = parse("a b c, {expr:joy}, {anim:wave} d e f")
    assert [type(s) for s in p.steps] == [Speak, Act, Act, Speak]


def test_duplicate_kind_in_group_overrides_and_flags():
    p = parse("well. {expr:joy}{expr:sad} fine then ok")
    assert p.steps[1] == Act(expr="sad")
    assert "dup_in_group:expr" in p.flags


def test_act_groups_capped_at_four():
    raw = " ".join(f"line {i} here. {{expr:joy}}" for i in range(6)) + " end of it"
    p = parse(raw)
    assert sum(isinstance(s, Act) for s in p.steps) == tags.MAX_ACT_GROUPS == 4
    assert "act_cap:2" in p.flags


# ─── Degradation ──────────────────────────────────────────

@pytest.mark.parametrize("raw,steps,flag", [
    ("{expr:neutral} fine.", [Act(expr="relaxed"), Speak("fine.")], None),            # alias
    ("{anim:nod} sure thing ben", [Act(anim="agree"), Speak("sure thing ben")], None),  # alias
    ("{expr:smug} sure buddy", [Speak("sure buddy")], "unknown_tag:smug"),
    ("{anim:dance_1} sure buddy", [Speak("sure buddy")], "unknown_tag:dance_1"),
    ("nice {expr:joy shot here", [Speak("nice shot here")], "malformed"),
    ("nice {joy} shot here", [Speak("nice shot here")], "malformed"),
    ("a {expr:{anim:wave}} b", [Speak("a"), Act(anim="wave"), Speak("b")], "malformed"),
    ("[EMOTION: joy] okay then", [Speak("okay then")], "unknown_vocal:emotion: joy"),
    ("<|channel>thought\nhmm<channel|>rip {expr:sad} her", [Speak("rip"), Act(expr="sad"), Speak("her")], "leak"),
    ("nice<turn|><|turn>user\nhi", [Speak("nice user hi")], "leak"),
    ('ok <|tool_call>call:x{a:<|"|>b<|"|>}<tool_call|> sure', [Speak("ok sure")], "leak"),
])
def test_degrades_without_raising(raw, steps, flag):
    p = parse(raw)
    assert p.steps == steps
    if flag:
        assert flag in p.flags, p.flags


@pytest.mark.parametrize("raw", ["", "   ", "{expr:joy}", "{anim:wave}{expr:sad}", "<|channel>x<channel|>",
                                 ". , !", None, "{expr:joy} . {anim:wave}"])
def test_no_speak_and_no_vocal_is_noop(raw):
    p = parse(raw)
    assert p.is_noop and p.steps == []
    assert "empty" in p.flags


def test_plan_is_json_serializable_and_round_trips():
    p = parse(EX1)
    blob = json.dumps(p.to_dict())
    assert Plan.from_dict(json.loads(blob)) == p
    assert json.loads(blob)["steps"][1] == {"type": "act", "expr": "sad", "anim": "shrug"}


def test_text_property_is_spoken_order():
    assert parse(EX1).text == "Ben you need to use your alt, [sigh] he never listens."


def test_parser_never_inserts_reset_steps():
    p = parse("{expr:sad} ugh fine then.")
    assert p.steps == [Act(expr="sad"), Speak("ugh fine then.")]


# ─── Fuzz ─────────────────────────────────────────────────

_ALPHABET = list("abc xyz{}[]<>|:._-!?,\n\t") + [
    "{expr:", "{anim:", "joy", "wave", "}", "<|channel>", "<channel|>", "<|turn>", "<turn|>",
    "<|tool_call>", "<tool_call|>", "[sigh]", "[", "neutral", "é", "\U0001f600", "\x00",
    "{expr:sad}", "{anim:shrug}", "[laugh]", "[clear throat]",
]


@pytest.mark.parametrize("seed", range(400))
def test_fuzz_never_raises(seed):
    rnd = random.Random(seed)
    raw = "".join(rnd.choice(_ALPHABET) for _ in range(rnd.randint(0, 80)))
    p = parse(raw)
    json.dumps(p.to_dict())
    assert sum(isinstance(s, Act) for s in p.steps) <= tags.MAX_ACT_GROUPS
    for s in p.steps:
        if isinstance(s, Speak):
            assert s.text and s.text == s.text.strip()
            assert not any(c in s.text for c in "{}[]") and "<|" not in s.text and "|>" not in s.text
        elif isinstance(s, Vocal):
            assert s.tag in tags.VOCAL_TAGS
        else:
            assert s.expr or s.anim
            assert s.expr is None or s.expr in tags.EXPRESSIONS
            assert s.anim is None or s.anim in tags.ANIMATIONS
    if not any(isinstance(s, (Speak, Vocal)) for s in p.steps):
        assert p.steps == []


def test_orphan_channel_close_drops_leaked_thinking():
    # prompt ends with "<|channel>thought\n<channel|>"; base model re-thinks and closes again
    p = parse("I need to respond to Ben.\nNo help offered.<channel|>yeah i can hear you. {expr:joy}")
    assert p.steps[0] == Speak("yeah i can hear you.")
    assert "leak" in p.flags
