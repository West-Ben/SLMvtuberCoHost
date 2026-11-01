"""Phase 3: row schema, sidecar marks/reviews, review queue, retention."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pydantic
import pytest

from training import training_logger as tl

ROOT = Path(__file__).resolve().parent.parent


def record(ts=1761300010.0, **kw):
    r = {
        "schema_version": 2, "ts": ts, "window_id": "w1", "window_start": ts - 10, "window_end": ts,
        "flush_reason": "window", "input_kind": "frames+speech",
        "frame_paths": ["imagelog/2026-10-24/w-w1/f01.jpg"],
        "speech": [{"t": ts - 5, "text": "hi"}], "chat": [],
        "history_ref": None, "prompt_version": "rx1+abc", "adapter_version": None,
        "sampling": {"temperature": 1.0}, "latency_ms": 900,
        "raw_output": "hey {expr:joy} you", "plan": {"steps": [], "flags": []}, "parse_flags": [],
    }
    r.update(kw)
    return r


def test_log_reaction_writes_valid_row(tmp_path):
    lg = tl.ReactionRowLogger(tmp_path)
    rid = lg.log_reaction(record())
    rows = list(tl.iter_rows(tmp_path))
    assert len(rows) == 1 and rows[0]["id"] == rid
    assert rows[0]["status"] == "raw" and rows[0]["review"] is None and rows[0]["human"] is None
    assert (tmp_path / f"{tl.row_date(rid)}.jsonl").exists()
    tl.ReactionRow.model_validate(rows[0])


def test_schema_rejects_bad_rows(tmp_path):
    lg = tl.ReactionRowLogger(tmp_path)
    with pytest.raises(pydantic.ValidationError):
        lg.log_reaction(record(input_kind="audio"))
    with pytest.raises(pydantic.ValidationError):
        lg.log_reaction(record(surprise_field=1))


def test_reviews_and_marks_never_touch_row_file(tmp_path):
    lg = tl.ReactionRowLogger(tmp_path)
    rid = lg.log_reaction(record())
    f = tmp_path / f"{tl.row_date(rid)}.jsonl"
    before = f.read_bytes()
    tl.append_review(rid, {"verdict": "reject", "scores": {"persona": 1}}, "rejected", rows_dir=tmp_path)
    tl.append_mark(rid, "good", rows_dir=tmp_path)
    assert f.read_bytes() == before


def test_human_mark_overrides_reviewer(tmp_path):
    lg = tl.ReactionRowLogger(tmp_path)
    rid = lg.log_reaction(record())
    tl.append_review(rid, {"verdict": "reject", "scores": {"persona": 1}, "flags": []}, "rejected", rows_dir=tmp_path)
    assert tl.load_effective(tmp_path)[0]["status"] == "rejected"
    tl.append_mark(rid, "good", rows_dir=tmp_path)
    eff = tl.load_effective(tmp_path)[0]
    assert eff["status"] == "accepted" and eff["human"] == "good"
    tl.append_mark(rid, "fix", "better line", rows_dir=tmp_path)    # latest mark wins
    eff = tl.load_effective(tmp_path)[0]
    assert eff["human"] == "fix" and eff["human_fix"] == "better line" and eff["status"] == "accepted"


def test_unsafe_is_excluded_even_if_marked_good(tmp_path):
    lg = tl.ReactionRowLogger(tmp_path)
    rid = lg.log_reaction(record())
    tl.append_review(rid, {"verdict": "accept", "scores": {}, "flags": ["toxic_chat"]}, "accepted", rows_dir=tmp_path)
    tl.append_mark(rid, "good", rows_dir=tmp_path)
    assert tl.load_effective(tmp_path)[0]["status"] == "excluded"


def test_uncertainty_orders_borderline_first():
    sure_bad = {"review": {"scores": {"a": 1, "b": 1}, "verdict": "reject"}, "parse_flags": []}
    border = {"review": {"scores": {"a": 3, "b": 3}, "verdict": "minor_edit"}, "parse_flags": []}
    sure_good = {"review": {"scores": {"a": 5, "b": 5}, "verdict": "accept"}, "parse_flags": []}
    ranked = sorted([sure_bad, sure_good, border], key=tl.uncertainty, reverse=True)
    assert ranked[0] is border


def test_prune_only_rejected_and_protects_shared_frames(tmp_path):
    root = tmp_path
    rows = root / "rows"
    lg = tl.ReactionRowLogger(rows)
    old = time.time() - 30 * 86400
    shared, rej_only, kept = (f"imagelog/d/w-x/f0{i}.jpg" for i in (1, 2, 3))
    for p in (shared, rej_only, kept):
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_bytes(b"x")
    r_rej = lg.log_reaction(record(ts=old, frame_paths=[shared, rej_only]))
    r_ok = lg.log_reaction(record(ts=old, frame_paths=[shared, kept]))
    tl.append_review(r_rej, {"verdict": "reject"}, "rejected", rows_dir=rows)
    tl.append_review(r_ok, {"verdict": "accept"}, "accepted", rows_dir=rows)

    dry = tl.prune_rejected_frames(7, apply=False, rows_dir=rows, repo_root=root)
    assert [p.name for p in dry] == ["f02.jpg"] and (root / rej_only).exists()
    tl.prune_rejected_frames(7, apply=True, rows_dir=rows, repo_root=root)
    assert not (root / rej_only).exists()
    assert (root / shared).exists() and (root / kept).exists()


def test_prune_respects_age(tmp_path):
    rows = tmp_path / "rows"
    lg = tl.ReactionRowLogger(rows)
    p = "imagelog/d/w-y/f01.jpg"
    (tmp_path / p).parent.mkdir(parents=True); (tmp_path / p).write_bytes(b"x")
    rid = lg.log_reaction(record(ts=time.time() - 3600, frame_paths=[p]))
    tl.append_review(rid, {"verdict": "reject"}, "rejected", rows_dir=rows)
    assert tl.prune_rejected_frames(7, apply=True, rows_dir=rows, repo_root=tmp_path) == []


def test_cli_runs_as_script_and_module():
    for cmd in ([sys.executable, "training/training_logger.py"], [sys.executable, "-m", "training.training_logger"]):
        out = subprocess.run(cmd, cwd=ROOT / "src", capture_output=True, text=True, timeout=60)
        assert out.returncode == 0, out.stderr
        assert "review-queue" in out.stdout
