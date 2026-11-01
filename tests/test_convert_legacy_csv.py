"""training/convert_legacy_csv.py — normalisation rules, relocation, beat cap, rejects, splits, byte identity."""

import csv
import json
import random

import pytest

import prompt_format
import tag_parser
import tags
from training import convert_legacy_csv as cv
from training import reviewer


def norm(raw, policy="drop", ctx=None):
    return cv.normalize_reply(raw, policy, ctx or [])


# ─── 1b rewrites ──────────────────────────────────────────

@pytest.mark.parametrize("raw,reply,edit", [
    ("{expr:neutral} madge is probably just enjoying the quiet", "{expr:relaxed} madge is probably just enjoying the quiet", "expr_neutral->relaxed"),
    ("{expr:excitement} let's go we finally won one", "{expr:joy} let's go we finally won one", "expr_excitement->joy"),
    ("that's the most accurate assessment all day {anim:nod} [chuckle]", "that's the most accurate assessment all day {anim:agree} [chuckle]", "anim_nod->agree"),
    ("nice work everyone, great round {anim:clap}", "nice work everyone, great round", "anim_clap_removed"),
])
def test_explicit_rewrites(raw, reply, edit):
    out, edits, plan = norm(raw)
    assert out == reply and edit in edits and not plan.flags


def test_thinking_policy_drop_and_think():
    out, edits, _ = norm("{expr:thinking} no way we're doing that tonight")
    assert out == "no way we're doing that tonight" and "expr_thinking_dropped" in edits
    out, edits, _ = norm("{expr:thinking} no way we're doing that tonight", policy="think")
    assert out == "{anim:think} no way we're doing that tonight" and "expr_thinking->anim_think" in edits


def test_alias_and_canonical_spelling():
    out, edits, _ = norm("{expr:shocked} [Clear Throat] okay that was unexpected")
    assert out == "{expr:surprised} [clear throat] okay that was unexpected"
    assert "alias:expr_shocked->surprised" in edits and "canonical_vocal" in edits


def test_whitespace_collapsed_and_recorded():
    out, edits, _ = norm("that's the kind of move  I respect")
    assert out == "that's the kind of move I respect" and edits == ["whitespace"]


# ─── rejects ──────────────────────────────────────────────

@pytest.mark.parametrize("raw,reason", [
    ("{anim:dance_1} let's celebrate this one now", "unknown_tag:dance_1"),
    ("{expr:smug} sure thing buddy whatever", "unknown_tag:smug"),
    ("well that happened again {sigh}", "unknown_tag:sigh"),
    ("{warudo} hello there everyone", "unknown_tag:warudo"),
    ("[EMOTION: joy] okay then everyone", "unknown_tag:emotion: joy"),
    ("neutral yeah that's the date i think", "bare_word_tag"),
    ("thinking about it more, no", "bare_word_tag"),
    ("", "empty"),
    ("   ", "empty"),
    ("{expr:thinking}", "empty"),
    ("<|channel>thought\nhmm<channel|>ok sure then", "leak"),
    ("nice one<turn|> more", "leak"),
    ("Antimage forever. {anim:shrug} So bold.", "chunk_too_short"),
    ("want me to help you aim better next time?", "assistant"),
])
def test_reject_reasons(raw, reason):
    with pytest.raises(cv.Reject) as e:
        norm(raw)
    assert e.value.reason == reason


def test_echo_of_chat_rejected():
    with pytest.raises(cv.Reject) as e:
        norm("ben do caustic or riot", ctx=["ben do caustic or riot"])
    assert e.value.reason == "echo"


# ─── relocation + beat cap ────────────────────────────────

def test_mid_phrase_group_moves_to_nearest_boundary():
    raw = ("{expr:surprise} betty grable is a legendary answer, honestly i'm shook {anim:shrug} [sniff] "
           "that's the kind of answer you expect")
    out, edits, plan = norm(raw)
    assert "moved_group" in edits
    assert out == ("{expr:surprise} betty grable is a legendary answer, {anim:shrug} [sniff] honestly i'm shook "
                   "that's the kind of answer you expect")
    assert reviewer.grammar_issue(plan) is None and not plan.flags


def test_relocation_to_end_when_nearer():
    out, edits, _ = norm("Then go to sleep now. Stream will be here {expr:anger} tomorrow.")
    assert out == "Then go to sleep now. Stream will be here tomorrow. {expr:anger}" and "moved_group" in edits


def test_relocation_keeps_bound_vocals_with_group():
    out, _, plan = norm("you really just did {anim:shrug}{expr:sad} [sigh] that again huh")
    steps = plan.steps
    i = next(k for k, s in enumerate(steps) if isinstance(s, tag_parser.Act))
    assert isinstance(steps[i + 1], tag_parser.Vocal) and steps[i + 1].tag == "sigh"


def test_relocation_never_merges_into_another_beat():
    out, _, plan = norm("{anim:wave} hello there my friends {expr:joy} good to see you")
    assert sum(isinstance(s, tag_parser.Act) for s in plan.steps) == 2


def test_beat_cap_drops_later_beats():
    raw = ("first part of this, {expr:joy} second part of it, {anim:shrug} third part of it, "
           "{expr:sad} [sigh] fourth part of it")
    out, edits, plan = norm(raw)
    assert edits.count("dropped_beat") == 1
    assert sum(isinstance(s, tag_parser.Act) for s in plan.steps) == tags.MAX_BEATS_PER_REPLY
    assert "[sigh]" not in out                          # whole beat dropped, incl. its bound vocal


def test_never_adds_words():
    raw = "honestly i'm shook {anim:shrug} that's the kind of answer you expect from him"
    out, _, _ = norm(raw)
    words = lambda s: sorted(w for w in tags.BRACE_TAG_RE.sub(" ", s).split())
    assert words(out) == words(raw)


def test_output_always_parses_without_flags_fuzz():
    rnd = random.Random(0)
    pieces = ["well", "that", "was", "wild,", "honestly.", "{expr:neutral}", "{anim:nod}", "[chuckle]",
              "{expr:joy}", "{anim:shrug}", "[sigh]", "{expr:thinking}", "ben", "again!", "{anim:clap}"]
    for _ in range(2000):
        raw = " ".join(rnd.choice(pieces) for _ in range(rnd.randint(1, 14)))
        try:
            out, _, plan = norm(raw)
        except cv.Reject:
            continue
        p = tag_parser.parse(out)
        assert not p.flags and not p.is_noop and reviewer.grammar_issue(p) is None


# ─── conversion, prompts, splits ──────────────────────────

def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "messages", "createdAt", "updatedAt"])
        w.writeheader()
        for i, msgs in rows:
            w.writerow({"id": i, "messages": json.dumps(json.dumps({"messages": msgs})), "createdAt": "", "updatedAt": ""})


@pytest.fixture
def inputs(tmp_path):
    d = tmp_path / "version7"
    d.mkdir()
    single = []
    for i in range(1, 31):
        single.append((i, [
            {"role": "system", "content": "Rachel — Identity <SOUL.md> <[EMOTION=joy][INTENSITY=high]>"},
            {"role": "system", "content": "<<STREAM_CONTEXT>>\n[RECENT CHAT]\nviewer_2: hi"},
            {"role": "user", "content": f"viewer_{i % 7 + 1}: question number {i} about the match"},
            {"role": "model", "content": f"Answer number {i} is easy. {{expr:joy}} [chuckle] You already knew that {i}."},
        ]))
    _write_csv(d / "Rachel-training-v7-single (2).csv", single)
    convs = []
    exprs = ["neutral", "joy", "anger", "sadness", "concern", "surprise", "excitement"]
    for i in range(1, 121):
        msgs = [{"role": "system", "content": "<<real SOUL.md + USER.md system prompt>>"}]
        for t in range(3):
            msgs.append({"role": "user", "content": f"chat line {i}-{t} about the game"})
            e = exprs[(i + t) % len(exprs)]
            body = f"{{expr:{e}}} reply {i} turn {t} has enough words here" if (i + t) % 5 else f"plain reply {i} turn {t} with words"
            msgs.append({"role": "model", "content": body})
        convs.append((i, msgs))
    _write_csv(d / "Rachel-training-v7 (1).csv", convs)
    _write_csv(d / "Rachel-training-v7-rework (1).csv", convs[:100])
    return d


def test_end_to_end_outputs(inputs):
    out = inputs / "ready"
    meta = cv.run(inputs, out, "drop", 3407, tokenizer_path=inputs / "missing.json")
    for name in ("train.jsonl", "holdout.jsonl", "tags_snapshot.json", "meta.json", "rejects.jsonl", "report.md", "README.md"):
        assert (out / name).exists(), name
    train = [json.loads(l) for l in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    hold = [json.loads(l) for l in (out / "holdout.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {e["session_id"] for e in train}.isdisjoint({e["session_id"] for e in hold})
    assert len(hold) >= cv.HOLDOUT_MIN
    for ex in train + hold:
        assert set(ex) == {"id", "source", "session_id", "weight", "input_kind", "prompt", "completion", "reply",
                           "raw_reply", "edits", "plan", "prompt_version", "grammar_hash"}
        assert ex["prompt"].startswith(prompt_format.BOS) and ex["prompt"].endswith(prompt_format.GENERATION_PREFIX)
        assert prompt_format.IMAGE_TOKEN not in ex["prompt"]
        assert ex["completion"] == ex["reply"] + "<turn|>"
        assert not tag_parser.parse(ex["reply"]).flags
        assert ex["weight"] == (2 if ex["source"] == "single_2" else 1)
        assert ex["input_kind"] == "legacy_text" and ex["grammar_hash"] == tags.grammar_hash()
    # selection constraints (v7_1)
    v7 = [e for e in train + hold if e["source"] == "v7_1"]
    n = len(v7)
    firsts = [cv.first_expr(e) for e in v7]
    assert max(firsts.count(x) for x in set(firsts) - {None}) <= cv.MAX_EXPR_SHARE * n
    assert sum(cv.is_untagged(e) for e in v7) <= cv.MAX_UNTAGGED_SHARE * n
    assert meta["rework_diff"]["common_ids"] == 100 and meta["rework_diff"]["identical_conversations"] == 100


def test_prompt_byte_identical_to_direct_prompt_format_call(inputs):
    out = inputs / "ready"
    cv.run(inputs, out, "drop", 3407, tokenizer_path=inputs / "missing.json")
    rows = [json.loads(l) for l in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    single = next(r for r in rows if r["source"] == "single_2")
    author, text = single["prompt"].rsplit("[chat +0.0s] ", 1)[1].split("<turn|>")[0].split(": ", 1)
    system = prompt_format.load_system_prompt()
    turns = prompt_format.build_reaction_messages(system, [], [], [], [{"t": 0.0, "user": author, "text": text}], 0.0, 0.0)
    assert single["prompt"].encode("utf-8") == prompt_format.render(turns).encode("utf-8")

    multi = next(r for r in rows if r["source"] == "v7_1" and r["id"].endswith("-t3"))
    conv_i = int(multi["session_id"].split("-")[1])
    # rebuild history exactly like the converter: prior user lines + prior normalised replies
    hist = []
    for t in range(2):
        e = ["neutral", "joy", "anger", "sadness", "concern", "surprise", "excitement"][(conv_i + t) % 7]
        raw = f"{{expr:{e}}} reply {conv_i} turn {t} has enough words here" if (conv_i + t) % 5 else f"plain reply {conv_i} turn {t} with words"
        reply, _, _ = cv.normalize_reply(raw, "drop", [f"chat line {conv_i}-{t} about the game"])
        hist.append({"user": f"[chat +0.0s] viewer_1: chat line {conv_i}-{t} about the game", "assistant": reply})
    turns = prompt_format.build_reaction_messages(
        system, hist, [], [], [{"t": 0.0, "user": "viewer_1", "text": f"chat line {conv_i}-2 about the game"}], 0.0, 0.0)
    assert multi["prompt"] == prompt_format.render(turns)


def test_holdout_split_disjoint_and_min_size():
    exs = [{"session_id": f"s{i // 3}", "source": "v7_1" if i % 2 else "single_2"} for i in range(300)]
    train, hold = cv.holdout_split(exs, random.Random(1))
    assert {e["session_id"] for e in train}.isdisjoint({e["session_id"] for e in hold})
    assert len(hold) >= cv.HOLDOUT_MIN and len(train) + len(hold) == 300


def test_discover_ignores_suffixes(tmp_path):
    for name in ("Rachel-training-v7-single (2).csv", "Rachel-training-v7 (1).csv", "Rachel_training_v7_rework.csv"):
        _write_csv(tmp_path / name, [(1, [{"role": "user", "content": "viewer_1: hi"}, {"role": "model", "content": "hey"}])])
    found, notes = cv.discover(tmp_path, repo_root=tmp_path / "nowhere")
    assert set(found) == {"single_2", "v7_1", "rework"}
