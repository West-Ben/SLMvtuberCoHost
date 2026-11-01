"""Phase 5: response-only masking, collator, dataset builder guards, registry, eval gate."""

import json
import time
from pathlib import Path

import pytest

import prompt_format
from training import dataset_builder as db
from training import eval_gate as eg
from training import registry
from training import row_prompt
from training import train_nightly as tn
from training import training_logger as tl

FIX = Path(__file__).parent / "fixtures" / "window_0001"
ROOT = Path(__file__).resolve().parent.parent


# ─── Masking + collator ───────────────────────────────────

def test_responses_only_labels_masks_through_last_marker():
    ids = [1, 2, 9, 9, 3, 9, 9, 4, 5]
    assert tn.responses_only_labels(ids, [9, 9]) == [-100] * 7 + [4, 5]


def test_responses_only_labels_errors():
    with pytest.raises(ValueError):
        tn.responses_only_labels([1, 2, 3], [9])
    with pytest.raises(ValueError):
        tn.responses_only_labels([1, 9], [9])


class CharTok:
    """Fake processor: one token per character; ignores images."""

    def __call__(self, text, add_special_tokens=False, return_tensors=None, images=None):
        import torch
        if isinstance(text, list):
            return {"input_ids": torch.tensor([[ord(c) for c in text[0]]])}
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids):
        return "".join(map(chr, ids))


def test_collator_trains_only_on_completion():
    turns = prompt_format.build_reaction_messages(
        "sys", [{"user": "earlier", "assistant": "old reply"}],
        [f"tests/fixtures/window_0001/f0{i}.jpg" for i in (1, 2)],
        [{"t": 1.0, "text": "hey"}], [], 0.0, 10.0)
    ex = prompt_format.render_training_example(turns, "nice {expr:joy} one")
    col = tn.ReactionCollator(CharTok(), ROOT, max_len=100000)
    enc = col([ex])
    ids, labels = enc["input_ids"][0].tolist(), enc["labels"][0].tolist()
    trained = "".join(chr(t) for t, l in zip(ids, labels) if l != -100)
    assert trained == "nice {expr:joy} one<turn|>"


# ─── Dataset builder ──────────────────────────────────────

def _row(i, day="2026-10-20", hour=20):
    ts = time.mktime(tuple(int(x) for x in day.split("-")) + (hour, i, 0, 0, 0, -1))
    return {"schema_version": 2, "ts": ts, "window_id": f"w{i}", "window_start": ts - 10, "window_end": ts,
            "flush_reason": "window", "input_kind": "frames+speech", "frame_paths": ["imagelog/x/f.jpg"],
            "speech": [{"t": ts - 3, "text": f"line {i}"}], "chat": [], "history_ref": None,
            "prompt_version": "old", "adapter_version": None, "sampling": {}, "latency_ms": 1,
            "raw_output": f"reply number {i}", "plan": {"steps": [], "flags": []}, "parse_flags": []}


@pytest.fixture
def pipe(tmp_path):
    rows, cur = tmp_path / "rows", tmp_path / "curated"
    lg = tl.ReactionRowLogger(rows)
    ids = {}
    for day, hour in (("2026-10-20", 20), ("2026-10-21", 14), ("2026-10-21", 21)):
        for i in range(6):
            rid = lg.log_reaction(_row(i, day, hour))
            ids[rid] = day
            tl.append_review(rid, {"verdict": "accept", "scores": {"persona": 4}}, "accepted", rows_dir=rows)
    by_id = {r["id"]: r for r in tl.load_effective(rows)}
    sessions = row_prompt.session_ids(list(by_id.values()))
    cur.mkdir()
    for day in ("2026-10-20", "2026-10-21"):
        with open(cur / f"{day}.jsonl", "w") as f:
            for rid, d in ids.items():
                if d == day:
                    ex = row_prompt.training_example(by_id[rid], by_id[rid]["raw_output"], by_id)
                    ex["prompt_version"] = "stale"   # forces a rebuild
                    f.write(json.dumps({"id": rid, **{**ex, "prompt_version": "stale"}, "input_kind": "frames+speech",
                                        "session": sessions[rid], "human": None}) + "\n")
    return tmp_path, rows, cur, sessions, list(ids)


def _build(tmp, rows, cur, **kw):
    return db.build(run_id="t", curated_dir=cur, rows_dir=rows, out_root=tmp / "datasets",
                    core_path=tmp / "core.jsonl", eval_path=tmp / "eval.jsonl",
                    registry_path=tmp / "registry.json", **kw)


def test_builder_rebuilds_stale_prompts_and_splits_by_session(pipe):
    tmp, rows, cur, sessions, ids = pipe
    m = _build(tmp, rows, cur)
    train = [json.loads(l) for l in (tmp / "datasets/t/train.jsonl").read_text().splitlines()]
    hold = [json.loads(l) for l in (tmp / "datasets/t/holdout.jsonl").read_text().splitlines()]
    cur_version = prompt_format.prompt_version(prompt_format.load_system_prompt())
    assert all(r["prompt_version"] == cur_version for r in train + hold)
    assert {r["session"] for r in train}.isdisjoint({r["session"] for r in hold})
    assert m["prompt_version"] == cur_version and m["data_range"]["nights"] == ["2026-10-20", "2026-10-21"]


def test_builder_drops_late_human_bad_and_eval_sessions(pipe):
    tmp, rows, cur, sessions, ids = pipe
    tl.append_mark(ids[0], "bad", rows_dir=rows)
    (tmp / "eval.jsonl").write_text(json.dumps({"session": sessions[ids[-1]]}) + "\n")
    _build(tmp, rows, cur)
    out = (tmp / "datasets/t/train.jsonl").read_text() + (tmp / "datasets/t/holdout.jsonl").read_text()
    assert ids[0] not in out
    banned = [i for i in ids if sessions[i] == sessions[ids[-1]]]
    assert banned and not any(i in out for i in banned)


def test_builder_replay_is_about_half(pipe):
    tmp, rows, cur, sessions, ids = pipe
    core = [{"id": f"core{i}", "prompt": "p", "completion": "c<turn|>", "images": [], "session": f"core-s{i}",
             "prompt_version": prompt_format.prompt_version(prompt_format.load_system_prompt())} for i in range(100)]
    (tmp / "core.jsonl").write_text("\n".join(json.dumps(c) for c in core) + "\n")
    m = _build(tmp, rows, cur)
    assert m["counts"]["replay"] == m["counts"]["new"]


def test_builder_only_new_nights_after_current_adapter(pipe):
    tmp, rows, cur, sessions, ids = pipe
    registry.save({"adapters": [{"version": "n1", "status": "promoted", "parent": None,
                                 "data_range": {"nights": ["2026-10-20"], "start": "2026-10-20", "end": "2026-10-20"}}],
                   "current": "n1"}, tmp / "registry.json")
    m = _build(tmp, rows, cur)
    assert m["data_range"]["nights"] == ["2026-10-21"] and m["parent"] == "n1"
    assert m["counts"]["replay"] > 0      # baked night 10-20 becomes replay


# ─── Registry ─────────────────────────────────────────────

def test_registry_promote_and_rollback_are_pointer_moves(tmp_path):
    p = tmp_path / "registry.json"
    registry.save({"adapters": [], "current": None}, p)
    registry.register({"version": "a", "parent": None, "data_range": {}}, p)
    registry.promote("a", p)
    registry.register({"version": "b", "parent": "a", "data_range": {}}, p)
    registry.promote("b", p)
    assert registry.load(p)["current"] == "b"
    assert registry.rollback(p) == "a" and registry.rollback(p) is None
    assert [x["version"] for x in registry.load(p)["adapters"]] == ["a", "b"]


def test_committed_registry_shape():
    reg = json.loads((ROOT / "adapters" / "registry.json").read_text())
    assert set(reg) >= {"adapters", "current"}


# ─── Eval gate ────────────────────────────────────────────

def test_gate_tolerance_and_floors():
    base = {"tag_validity": 0.98, "assistant_rate": 0.01, "judge_persona": 4.0, "core_pass": 1.0, "leading_only": 0.2}
    ok, why = eg.gate(dict(base), base)
    assert ok and not why
    ok, why = eg.gate({**base, "judge_persona": 3.7}, base)
    assert not ok and "judge_persona regressed" in why[0]
    ok, why = eg.gate({**base, "assistant_rate": 0.06}, {**base, "assistant_rate": 0.05})
    assert not ok and any("floor" in w for w in why)


def test_eval_run_promotes_or_rejects(tmp_path, monkeypatch):
    fx = json.loads((FIX / "window.json").read_text())
    item = {"id": "e1", "session": "s", "input_kind": "frames+chat+speech", "history": [],
            "row": {"frame_paths": [f["path"] for f in fx["frames"]], "speech": fx["speech"], "chat": fx["chat"],
                    "window_start": fx["start"], "window_end": fx["end"]}, "reference": "x"}
    (tmp_path / "eval.jsonl").write_text(json.dumps(item) + "\n")
    monkeypatch.setattr(eg, "EVAL_SET_PATH", tmp_path / "eval.jsonl")
    monkeypatch.setattr(eg, "CORE_PROMPTS_PATH", ROOT / "training" / "data" / "core_prompts.jsonl")
    monkeypatch.setattr(eg, "ROOT", tmp_path)
    reg = tmp_path / "registry.json"
    registry.save({"adapters": [], "current": None}, reg)
    registry.register({"version": "good", "parent": None, "data_range": {}}, reg)
    registry.register({"version": "bad", "parent": None, "data_range": {}}, reg)

    def gen(version, turns):
        if version == "bad":
            return "{expr:joy} want me to help you with that? happy to help"
        return "hela {expr:joy} is cooking. you know it."

    judge = lambda row, sys_, note: {"scores": {k: 4 for k in eg.reviewer.SCORE_KEYS}}
    r = eg.run("bad", generate=gen, judge=judge, registry_path=reg)
    assert not r["promote"] and registry.load(reg)["current"] is None
    assert registry.get(registry.load(reg), "bad")["status"] == "rejected"
    r = eg.run("good", generate=gen, judge=judge, registry_path=reg)
    assert r["promote"] and registry.load(reg)["current"] == "good"
    assert (tmp_path / "adapters" / "good" / "eval_report.json").exists()


def test_generator_refuses_hf_endpoint():
    with pytest.raises(SystemExit):
        eg.LlamaCppGenerator("https://xyz.us-east-1.aws.endpoints.huggingface.cloud")
