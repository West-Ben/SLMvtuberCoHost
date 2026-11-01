"""Phase 4: reviewer Stage A, edit enforcement, outputs, sidecar-only writes."""

import json
import time
from pathlib import Path

import pytest

import prompt_format
from training import reviewer as rv
from training import training_logger as tl

NIGHT_TS = time.mktime((2026, 10, 23, 20, 0, 0, 0, 0, -1))


def rec(i, raw, kind="frames+speech", speech=None, chat=None, frames=True, ref=None):
    ts = NIGHT_TS + i * 20
    return {
        "schema_version": 2, "ts": ts, "window_id": f"w{i}", "window_start": ts - 10, "window_end": ts,
        "flush_reason": "window", "input_kind": kind,
        "frame_paths": [f"imagelog/x/w{i}/f01.jpg"] if frames else [],
        "speech": speech if speech is not None else [{"t": ts - 4, "text": f"ben line {i} about the push"}],
        "chat": chat or [], "history_ref": ref, "prompt_version": "rx1+x", "adapter_version": None,
        "sampling": {}, "latency_ms": 500, "raw_output": raw, "plan": {"steps": [], "flags": []}, "parse_flags": [],
    }


@pytest.fixture
def env(tmp_path, monkeypatch):
    rows = tmp_path / "rows"
    monkeypatch.setattr(rv, "CURATED_DIR", tmp_path / "curated")
    monkeypatch.setattr(rv, "DPO_DIR", tmp_path / "dpo")
    monkeypatch.setattr(rv, "LEDGER_PATH", tmp_path / "persona" / "ledger_pending.jsonl")
    return tmp_path, rows, tl.ReactionRowLogger(rows)


def good_review(**kw):
    r = {"scores": {k: 4 for k in rv.SCORE_KEYS}, "flags": [], "verdict": "accept",
         "edited_output": None, "edit_reason": None, "new_opinion": None}
    r.update(kw)
    return r


# ─── Stage A ──────────────────────────────────────────────

@pytest.mark.parametrize("raw,kw,code", [
    ("hela is fine", {"frames": False}, "no_frames"),
    ("ok", {}, "too_short"),
    ("word " * 70, {}, "too_long"),
    ("want me to pull up a guide for that?", {}, "assistant"),
    ("Great question! hela.", {}, "assistant"),
    ("I see you're playing strange", {}, "assistant"),
    ("why? who? how is that even legal?", {}, "multi_question"),
    ("ben line 3 about the push", {}, "echo"),
])
def test_stage_a_reason_codes(raw, kw, code):
    row = rec(3, raw, **kw)
    assert rv.stage_a(row, []) == code


def test_stage_a_parse_invalid():
    row = rec(1, "<|channel>thought\nhmm<channel|>rip her out there")
    assert rv.stage_a(row, []) == "parse_invalid"


def test_stage_a_repetitive_vs_recent_accepted():
    row = rec(1, "four dps lobbies are a personality test everyone fails")
    assert rv.stage_a(row, ["four dps lobbies are a personality test everyone fails honestly"]) == "repetitive"
    assert rv.stage_a(row, ["jeff is the best part of this game"]) is None


def test_stage_a_passes_good_reply():
    assert rv.stage_a(rec(1, "hela is always cooking. {expr:joy} you know that one."), []) is None
    assert rv.stage_a(rec(1, "{anim:laugh}[laugh] that was a great move!"), []) is None
    assert rv.stage_a(rec(1, "Ben you need to use your alt, {anim:shrug}{expr:sad} [sigh] he never listens."), []) is None
    assert rv.stage_a(rec(1, "that's so dumb {anim:laugh}"), []) is None         # trailing beat = boundary


@pytest.mark.parametrize("raw,code", [
    ("Ben you need {anim:shrug} to use your alt", "tag_mid_phrase"),
    ("we lost. {expr:sad} [sigh] again.", "chunk_too_short"),
    ("[sigh] fine.", "chunk_too_short"),
    ("one two three. {expr:joy} four five six. {anim:wave} seven eight nine. {expr:sad} ten eleven twelve",
     "too_many_beats"),
])
def test_stage_a_beat_grammar(raw, code):
    assert rv.stage_a(rec(1, raw), []) == code


def test_unsplit_short_reply_is_not_chunk_too_short():
    assert rv.stage_a(rec(1, "no. hela."), []) is None


# ─── Edit enforcement ─────────────────────────────────────

def test_light_edit_allowed():
    row = rec(1, "hela is so {expr:joy} cooking right now honestly")     # mid-phrase
    review, why = rv.enforce_edit(row, good_review(verdict="minor_edit",     # moved to the clause boundary
                                                   edited_output="{expr:joy} hela is so cooking right now honestly"))
    assert why is None and review["verdict"] == "minor_edit"


@pytest.mark.parametrize("edited,reason", [
    ("completely different words about a totally unrelated subject entirely", "edit_too_large"),
    ("hela is so cooking right now, want me to give you tips", "edit_assistant"),
    ("hela {expr:smug} is so cooking right now honestly", "edit_parse_flags"),
    ("", "edit_empty"),
    ("hela is so {expr:joy} cooking right now honestly", "edit_tag_mid_phrase"),
])
def test_edit_violations_downgrade_to_reject(edited, reason):
    row = rec(1, "hela is so cooking right now honestly")
    review, why = rv.enforce_edit(row, good_review(verdict="minor_edit", edited_output=edited))
    assert why == reason and review["verdict"] == "reject"
    assert rv.status_for(review) == "rejected"


def test_status_mapping():
    assert rv.status_for(good_review()) == "accepted"
    assert rv.status_for(good_review(flags=["toxic_chat"])) == "excluded"
    assert rv.status_for(good_review(flags=["unsafe"], verdict="reject")) == "excluded"
    assert rv.status_for(good_review(flags=["off_stance"])) == "rejected"


# ─── End-to-end run ───────────────────────────────────────

def test_run_writes_sidecars_outputs_and_never_mutates_rows(env):
    root, rows, lg = env
    ids = [
        lg.log_reaction(rec(0, "hela is always cooking. {expr:joy} you know that one.")),
        lg.log_reaction(rec(1, "strange again. portals aren't a personality ben", kind="frames+chat",
                            chat=[{"t": NIGHT_TS + 15, "user": "v", "text": "strange main"}])),
        lg.log_reaction(rec(2, "want me to help you aim better?")),                       # stage A
        lg.log_reaction(rec(3, "jeff ult again {anim:laugh} every single time", kind="frames")),  # mid-phrase → B fixes
        lg.log_reaction(rec(4, "that push was rough and kind of sad to watch")),           # human bad
    ]
    lg.log_reaction(rec(5, "chat is wild today huh", kind="frames+chat",
                        chat=[{"t": NIGHT_TS + 95, "user": "troll", "text": "slur"}]))     # toxic
    tl.append_mark(ids[4], "bad", rows_dir=rows)
    before = {p.name: p.read_bytes() for p in rows.glob("*.jsonl")}

    def fake_review(row, system, note=""):
        if "chat is wild" in row["raw_output"]:
            return good_review(flags=["toxic_chat"])
        if "jeff" in row["raw_output"]:
            return good_review(verdict="minor_edit", edited_output="jeff ult again. {anim:laugh}[laugh] every single time",
                               edit_reason="tag placement", new_opinion={"topic": "jeff ult", "stance": "always funny"})
        return good_review()

    rep = rv.run(["2026-10-23"], reviewer=fake_review, rows_dir=rows)
    assert {p.name: p.read_bytes() for p in rows.glob("*.jsonl")} == before   # row files untouched

    eff = {r["id"]: r for r in tl.load_effective(rows)}
    assert eff[ids[2]]["status"] == "rejected" and eff[ids[2]]["review"]["reason"] == "assistant"
    assert eff[ids[4]]["status"] == "rejected" and eff[ids[4]]["human"] == "bad"
    assert sum(1 for r in eff.values() if r["status"] == "excluded") == 1
    assert rep["stage_a:assistant"] == 1 and rep["human_bad"] == 1

    curated = [json.loads(l) for l in (root / "curated" / "2026-10-23.jsonl").read_text().splitlines()]
    assert {c["id"] for c in curated} == {ids[0], ids[1], ids[3]}
    for c in curated:
        assert c["prompt"].endswith(prompt_format.GENERATION_PREFIX)
        assert c["completion"].endswith(prompt_format.EOT)
        assert "slur" not in c["prompt"]
    jeff = next(c for c in curated if c["id"] == ids[3])
    assert jeff["edited"] and jeff["completion"] == "jeff ult again. {anim:laugh}[laugh] every single time<turn|>"

    dpo = [json.loads(l) for l in (root / "dpo" / "2026-10-23.jsonl").read_text().splitlines()]
    assert len(dpo) == 1 and dpo[0]["id"] == ids[3]
    assert dpo[0]["rejected"].startswith("jeff ult again {anim:laugh}")
    unpaired = (root / "dpo" / "2026-10-23.unpaired.jsonl").read_text()
    assert ids[4] in unpaired

    ledger = [json.loads(l) for l in (root / "persona" / "ledger_pending.jsonl").read_text().splitlines()]
    assert ledger == [{**ledger[0], "topic": "jeff ult", "stance": "always funny", "status": "pending"}]


def test_human_good_bypasses_stage_b(env):
    root, rows, lg = env
    rid = lg.log_reaction(rec(0, "want me to help?"))   # would fail Stage A
    tl.append_mark(rid, "good", rows_dir=rows)
    called = []
    rv.run(["2026-10-23"], reviewer=lambda *a: called.append(1) or good_review(), rows_dir=rows)
    assert called == []
    curated = (root / "curated" / "2026-10-23.jsonl").read_text()
    assert rid in curated


def test_stratified_cap_balances_input_kinds():
    rows = []
    for i in range(30):
        kind = "frames" if i < 25 else "frames+chat"
        rows.append({"id": f"r{i}", "input_kind": kind, "frame_paths": [],
                     "final_output": f"unique words number {i} " + " ".join(f"w{i}x{j}" for j in range(5)),
                     "review": {"scores": {"a": 4}}})
    picked = rv.stratified_select(rows, cap=10)
    kinds = [r["input_kind"] for r in picked]
    assert len(picked) == 10 and kinds.count("frames+chat") == 5


def test_review_schema_shape():
    s = rv.REVIEW_SCHEMA
    assert set(s["required"]) == {"scores", "flags", "verdict", "edited_output", "edit_reason", "new_opinion"}
    assert set(s["properties"]["scores"]["required"]) == set(rv.SCORE_KEYS)


def test_grammar_issue_accepted_without_fix_is_rejected(env):
    root, rows, lg = env
    rid = lg.log_reaction(rec(0, "Ben you need {anim:shrug} to use your alt"))
    notes = []
    rv.run(["2026-10-23"], reviewer=lambda row, sys_, note="": notes.append(note) or good_review(), rows_dir=rows)
    eff = tl.load_effective(rows)[0]
    assert eff["status"] == "rejected" and eff["review"]["downgrade_reason"] == "tag_mid_phrase"
    assert "tag_mid_phrase" in notes[0]


def test_grammar_issue_rejected_outright_in_stage_a_only(env):
    root, rows, lg = env
    lg.log_reaction(rec(0, "Ben you need {anim:shrug} to use your alt"))
    rep = rv.run(["2026-10-23"], stage_a_only=True, rows_dir=rows)
    assert rep["stage_a:tag_mid_phrase"] == 1
    assert tl.load_effective(rows)[0]["status"] == "rejected"
