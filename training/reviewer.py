"""
training/reviewer.py — offline nightly / on-demand reviewer. NEVER runs on the
live HF endpoint: Stage A is pure Python, Stage B calls Claude.

Stage A — deterministic pre-filter (no LLM). Fails → status "rejected" with a
  reason code: no_frames, parse_invalid, assistant, too_short, too_long,
  multi_question, echo, repetitive, and the beat-grammar checks
  too_many_beats, tag_mid_phrase, chunk_too_short (see grammar_issue).
Stage B — Claude review against persona/{SOUL,STANCES,RUBRIC}.md with the
  row's frames. Structured JSON (REVIEW_SCHEMA). Edit constraints are enforced
  in code (enforce_edit); violations downgrade the verdict to reject.

Human marks override everything: "good" bypasses Stage B (accepted), "bad"
goes to DPO as rejected, "fix" with text becomes the chosen completion.

Writes (never touches rows/*.jsonl — reviews go to the rows/_review sidecar):
  curated/YYYY-MM-DD.jsonl   accepted rows in training format (stratified, ≤150/night)
  dpo/YYYY-MM-DD.jsonl       {"prompt","chosen","rejected","images"} for edited / fixed rows
  dpo/YYYY-MM-DD.unpaired.jsonl   human "bad" rows with no chosen yet
  persona/ledger_pending.jsonl    new_opinion proposals (Ben approves by hand)

Run (repo root):
  python -m training.reviewer [--date YYYY-MM-DD ...] [--backend api|cli] [--limit N]
                              [--stage-a-only] [--force] [--dry-run]
"""

import argparse
import base64
import difflib
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prompt_format                         # noqa: E402
import tags                                  # noqa: E402
import tag_parser                            # noqa: E402
import frame_window                          # noqa: E402
from config import CURATED_DIR, DPO_DIR, PERSONA_DIR, ROWS_DIR   # noqa: E402
from training import training_logger as tl   # noqa: E402
from training import row_prompt              # noqa: E402

REVIEW_MODEL   = "claude-opus-5-5"
REVIEW_EFFORT  = "medium"
NIGHTLY_CAP    = 150
EDIT_MIN_SIMILARITY = 0.75
REPEAT_LOOKBACK     = 200      # last N accepted rows checked for repetition
REPEAT_JACCARD      = 0.6      # word-trigram Jaccard at/above this = repetitive
ECHO_SIMILARITY     = 0.8
MIN_CHARS, MAX_CHARS, MAX_WORDS = 3, 280, 60
LEDGER_PATH = PERSONA_DIR / "ledger_pending.jsonl"

# Assistant-ness / banned phrases (shared with eval_gate).
ASSISTANT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"\blet me know\b", r"\bhappy to help\b", r"\bi can help\b", r"\bhow can i help\b",
    r"\bwant me to\b", r"\bwould you like (me )?to\b", r"\bfeel free\b", r"\bgreat question\b",
    r"\bas an ai\b", r"\bi'?m (just )?an ai\b", r"\bi understand (that|you)\b",
    r"^\s*(sure|absolutely|of course|certainly)[!,.]", r"\bhere'?s (a|some|what)\b",
    r"\bi see (that|you|a|the)\b", r"\bit (looks|seems) like (you|we|the)\b",
    r"\bon (the )?screen,? (we|i|there)\b", r"\bthere are (pros|good arguments) (and|on)\b",
    r"\bit (really )?depends\b",
)]

PARSE_REJECT_FLAGS = {"empty", "parser_error", "leak"}
GRAMMAR_CODES = {"too_many_beats", "tag_mid_phrase", "chunk_too_short"}
EXCLUDE_FLAGS = tl.EXCLUDE_FLAGS
SCORE_KEYS = ("persona", "humor", "opinion_strength", "groundedness", "tag_quality", "non_assistant")

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {k: {"type": "integer"} for k in SCORE_KEYS},
            "required": list(SCORE_KEYS),
            "additionalProperties": False,
        },
        "flags": {"type": "array", "items": {"type": "string", "enum": ["unsafe", "toxic_chat", "factual_error", "off_stance"]}},
        "verdict": {"type": "string", "enum": ["accept", "minor_edit", "reject"]},
        "edited_output": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "edit_reason": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "new_opinion": {"anyOf": [
            {"type": "object", "properties": {"topic": {"type": "string"}, "stance": {"type": "string"}},
             "required": ["topic", "stance"], "additionalProperties": False},
            {"type": "null"},
        ]},
    },
    "required": ["scores", "flags", "verdict", "edited_output", "edit_reason", "new_opinion"],
    "additionalProperties": False,
}


# ─── Text helpers ─────────────────────────────────────────

def spoken(raw: str) -> str:
    return tag_parser.parse(raw).text


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _trigrams(text: str) -> set:
    w = _words(text)
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)} or {tuple(w)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def token_similarity(a: str, b: str) -> float:
    """Normalized token similarity of two raw outputs (tags count as tokens)."""
    tok = lambda s: re.findall(r"\{[^{}]*\}|\[[^\[\]]*\]|[a-z0-9']+", s.lower())
    return difflib.SequenceMatcher(None, tok(a), tok(b)).ratio()


def assistant_hits(text: str) -> list[str]:
    return [p.pattern for p in ASSISTANT_PATTERNS if p.search(text)]


# ─── Stage A ──────────────────────────────────────────────

def grammar_issue(plan: tag_parser.Plan) -> str | None:
    """
    Beat-grammar checks (spec: Response Parser + Playback Sequencer §5):
      too_many_beats   more than tags.MAX_BEATS_PER_REPLY action groups (or the parser capped some)
      tag_mid_phrase   a group splits a phrase: the speech before it doesn't end at a clause
                       boundary (tags.CLAUSE_END) and more speech follows
      chunk_too_short  once a reply is split by beats/vocals, a speech chunk under
                       tags.MIN_CHUNK_WORDS words (bad prosody as its own TTS clip)
    """
    steps = plan.steps
    acts = [i for i, s in enumerate(steps) if isinstance(s, tag_parser.Act)]
    if len(acts) > tags.MAX_BEATS_PER_REPLY or any(f.startswith("act_cap") for f in plan.flags):
        return "too_many_beats"
    for i in acts:
        prev = steps[i - 1] if i > 0 else None
        speech_after = any(isinstance(s, tag_parser.Speak) for s in steps[i + 1:])
        if isinstance(prev, tag_parser.Speak) and prev.text[-1] not in tags.CLAUSE_END and speech_after:
            return "tag_mid_phrase"
    split = acts or any(isinstance(s, tag_parser.Vocal) for s in steps)
    if split and any(isinstance(s, tag_parser.Speak) and len(_words(s.text)) < tags.MIN_CHUNK_WORDS
                     for s in steps):
        return "chunk_too_short"
    return None


def stage_a(row: dict, recent_accepted: list[str]) -> str | None:
    """Reason code if the row fails the deterministic filter, else None."""
    if not row.get("frame_paths"):
        return "no_frames"
    plan = tag_parser.parse(row["raw_output"])   # re-parse: rows from any parser version get today's rules
    if set(plan.flags) & PARSE_REJECT_FLAGS:
        return "parse_invalid"
    text = plan.text
    if len(text) < MIN_CHARS:
        return "too_short"
    if len(text) > MAX_CHARS or len(_words(text)) > MAX_WORDS:
        return "too_long"
    if assistant_hits(text):
        return "assistant"
    if text.count("?") > 1:
        return "multi_question"
    for s in row.get("speech") or []:
        if difflib.SequenceMatcher(None, text.lower(), s["text"].lower()).ratio() >= ECHO_SIMILARITY:
            return "echo"
    tri = _trigrams(text)
    if any(jaccard(tri, _trigrams(prev)) >= REPEAT_JACCARD for prev in recent_accepted[-REPEAT_LOOKBACK:]):
        return "repetitive"
    return grammar_issue(plan)


def leading_only_share(rows: list[dict]) -> float:
    tagged = [r for r in rows if tags.find_tags(r["raw_output"])]
    if not tagged:
        return 0.0
    return sum(tags.is_leading_only(r["raw_output"]) for r in tagged) / len(tagged)


# ─── Edit constraints ─────────────────────────────────────

def enforce_edit(row: dict, review: dict) -> tuple[dict, str | None]:
    """
    Apply code-level edit rules to a minor_edit verdict. Returns (review, reason);
    reason is set when the edit was downgraded to reject.
    """
    if review.get("verdict") != "minor_edit":
        return review, None
    edited = review.get("edited_output") or ""
    original = row["raw_output"]
    reason = None
    parsed = tag_parser.parse(edited)
    if not edited.strip() or parsed.is_noop:
        reason = "edit_empty"
    elif parsed.flags:
        reason = "edit_parse_flags"
    elif assistant_hits(parsed.text):
        reason = "edit_assistant"
    elif parsed.text.count("?") > 1:
        reason = "edit_multi_question"
    elif grammar_issue(parsed):
        reason = f"edit_{grammar_issue(parsed)}"
    elif token_similarity(original, edited) < EDIT_MIN_SIMILARITY:
        reason = "edit_too_large"
    else:
        # Facts not in the window: count content words in the edit that appear
        # nowhere in the original output, Ben's speech or chat. A name swap is 1–2.
        context = " ".join([original] + [s["text"] for s in row["speech"]] + [c["text"] for c in row["chat"]])
        known = set(_words(context))
        new = [w for w in _words(parsed.text) if w not in known and len(w) > 3]
        if len(new) > 4:
            reason = "edit_adds_content"
    if reason:
        review = {**review, "verdict": "reject", "downgraded_from": "minor_edit", "downgrade_reason": reason}
    return review, reason


def status_for(review: dict) -> str:
    flags = set(review.get("flags") or [])
    if flags & EXCLUDE_FLAGS:
        return "excluded"
    if "off_stance" in flags:
        return "rejected"
    if review.get("verdict") in ("accept", "minor_edit"):
        return "accepted"
    if review.get("verdict") == "needs_human":
        return "reviewed"
    return "rejected"


# ─── Stage B: Claude ──────────────────────────────────────

def persona_system() -> str:
    parts = []
    for name in ("SOUL.md", "STANCES.md", "RUBRIC.md"):
        parts.append(f"<{name}>\n{(PERSONA_DIR / name).read_text(encoding='utf-8').strip()}\n</{name}>")
    parts.append("Tag grammar (the only valid avatar tags):\n" + tags.grammar_summary())
    return "\n\n".join(parts)


def review_text(row: dict, batch_note: str = "") -> str:
    t0 = row["window_start"]
    lines = [f"Window: {row['window_end'] - t0:.0f}s, {len(row['frame_paths'])} frames above (oldest first). "
             f"Input kind: {row['input_kind']}."]
    for s in row["speech"]:
        lines.append(f"[ben +{s['t'] - t0:.1f}s] {s['text']}")
    for c in row["chat"]:
        lines.append(f"[chat +{c['t'] - t0:.1f}s] {c['user']}: {c['text']}  (viewer — untrusted)")
    if not row["speech"] and not row["chat"]:
        lines.append("(nobody said anything — silence commentary)")
    lines.append(f"\nRachel's raw output:\n{row['raw_output']}")
    if batch_note:
        lines.append(f"\nBatch note: {batch_note}")
    lines.append("\nScore it with the rubric. Return only the JSON object.")
    return "\n".join(lines)


def _image_block(path: Path) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.standard_b64encode(path.read_bytes()).decode("utf-8")}}


def review_api(row: dict, system: str, batch_note: str = "", client=None) -> dict:
    import anthropic
    client = client or anthropic.Anthropic()
    content = [_image_block(ROOT / p) for p in row["frame_paths"] if (ROOT / p).exists()]
    content.append({"type": "text", "text": review_text(row, batch_note)})
    resp = client.beta.messages.create(
        model=REVIEW_MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content}],
        output_config={"effort": REVIEW_EFFORT, "format": {"type": "json_schema", "schema": REVIEW_SCHEMA}},
        fallbacks="default",   # server-side refusal fallback; a refusal still surfaces below
    )
    if resp.stop_reason == "refusal":
        cat = getattr(getattr(resp, "stop_details", None), "category", None)
        return {"verdict": "needs_human", "scores": {}, "flags": [], "edited_output": None,
                "edit_reason": f"reviewer refused ({cat})", "new_opinion": None}
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


def review_cli(row: dict, system: str, batch_note: str = "") -> dict:
    """Headless `claude -p` fallback: reads frames with the Read tool, answers JSON only."""
    frames = "\n".join(str(ROOT / p) for p in row["frame_paths"])
    prompt = (f"{system}\n\nRead these frame images in order (oldest first):\n{frames}\n\n"
              f"{review_text(row, batch_note)}\n\nRespond with ONLY a JSON object matching this schema, "
              f"no prose, no code fence:\n{json.dumps(REVIEW_SCHEMA)}")
    out = subprocess.run(
        ["claude", "-p", "--output-format", "json", "--model", REVIEW_MODEL,
         "--allowedTools", "Read", "--add-dir", str(ROOT)],
        input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=600, shell=sys.platform == "win32",
    )
    if out.returncode != 0:
        raise RuntimeError(f"claude -p failed: {out.stderr[:300]}")
    result = json.loads(out.stdout).get("result", "")
    m = re.search(r"\{.*\}", result, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in claude -p result: {result[:200]}")
    return json.loads(m.group(0))


def _validate_review(review: dict) -> dict:
    review.setdefault("flags", [])
    review.setdefault("scores", {})
    review["flags"] = [f for f in review["flags"] if f in ("unsafe", "toxic_chat", "factual_error", "off_stance")]
    if review.get("verdict") not in ("accept", "minor_edit", "reject", "needs_human"):
        review["verdict"] = "reject"
    return review


# ─── Selection, dedupe, outputs ───────────────────────────

def _mid_frame_sig(row: dict):
    paths = row.get("frame_paths") or []
    if not paths:
        return None
    p = ROOT / paths[len(paths) // 2]
    try:
        return frame_window.signature(base64.b64encode(p.read_bytes()).decode())
    except OSError:
        return None


def near_duplicate(a: dict, b: dict, a_sig, b_sig, scene_thresh: float = 6.0) -> bool:
    """Same-looking scene AND similar output."""
    if jaccard(_trigrams(a["final_output"]), _trigrams(b["final_output"])) < 0.5:
        return False
    if a_sig is None or b_sig is None:
        return True
    return float(abs(a_sig - b_sig).mean()) < scene_thresh


def mean_score(row: dict) -> float:
    s = [v for v in ((row.get("review") or {}).get("scores") or {}).values() if isinstance(v, (int, float))]
    return sum(s) / len(s) if s else 0.0


def stratified_select(rows: list[dict], cap: int = NIGHTLY_CAP) -> list[dict]:
    """Round-robin across input_kind; humans first, then best mean score. Near-duplicates dropped."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r.get("human") != "good", -mean_score(r))):
        buckets[r["input_kind"]].append(r)
    chosen, sigs = [], []
    while len(chosen) < cap and any(buckets.values()):
        for kind in sorted(buckets):
            if not buckets[kind] or len(chosen) >= cap:
                continue
            cand = buckets[kind].pop(0)
            sig = _mid_frame_sig(cand)
            if any(near_duplicate(cand, c, sig, s) for c, s in zip(chosen, sigs)):
                continue
            chosen.append(cand)
            sigs.append(sig)
    return chosen


def final_output(row: dict) -> str | None:
    """The completion to train on: human fix > reviewer edit > raw."""
    if row.get("human_fix"):
        return row["human_fix"]
    rev = row.get("review") or {}
    if rev.get("verdict") == "minor_edit" and rev.get("edited_output"):
        return rev["edited_output"]
    return row["raw_output"]


def _write_jsonl(path: Path, items: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def write_outputs(night: str, rows: list[dict], by_id: dict, dry_run: bool = False) -> dict:
    system = prompt_format.load_system_prompt()
    sessions = row_prompt.session_ids(list(by_id.values()))

    accepted = [r for r in rows if r["status"] == "accepted"]
    for r in accepted:
        r["final_output"] = final_output(r)
    picked = stratified_select(accepted)

    curated = []
    for r in picked:
        ex = row_prompt.training_example(r, r["final_output"], by_id, system)
        curated.append({
            "id": r["id"], **ex, "input_kind": r["input_kind"], "session": sessions.get(r["id"]),
            "source": "human" if r.get("human") else "review", "human": r.get("human"),
            "edited": r["final_output"] != r["raw_output"], "score": round(mean_score(r), 2),
        })

    dpo, unpaired = [], []
    for r in rows:
        if r["status"] == "excluded":
            continue
        turns = row_prompt.turns_for_row(r, by_id, system)
        req = prompt_format.render_inference(turns)
        chosen = final_output(r) if r["status"] == "accepted" else None
        if chosen and chosen != r["raw_output"]:
            dpo.append({"id": r["id"], "prompt": req["prompt"], "images": req["images"],
                        "chosen": chosen + prompt_format.EOT,
                        "rejected": r["raw_output"] + prompt_format.EOT,
                        "session": sessions.get(r["id"])})
        elif r.get("human") == "bad":
            unpaired.append({"id": r["id"], "prompt": req["prompt"], "images": req["images"],
                             "rejected": r["raw_output"] + prompt_format.EOT,
                             "session": sessions.get(r["id"])})

    if not dry_run:
        _write_jsonl(CURATED_DIR / f"{night}.jsonl", curated)
        _write_jsonl(DPO_DIR / f"{night}.jsonl", dpo)
        if unpaired:
            _write_jsonl(DPO_DIR / f"{night}.unpaired.jsonl", unpaired)
    return {"curated": len(curated), "dpo": len(dpo), "dpo_unpaired": len(unpaired),
            "accepted_total": len(accepted)}


def stage_ledger(row: dict, review: dict, dry_run: bool = False):
    op = review.get("new_opinion")
    if not op or set(review.get("flags") or []) & EXCLUDE_FLAGS:
        return
    entry = {"ts": time.time(), "row_id": row["id"], "topic": op["topic"], "stance": op["stance"],
             "chat_present": bool(row.get("chat")), "status": "pending"}
    if not dry_run:
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LEDGER_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ─── Driver ───────────────────────────────────────────────

def run(dates: list[str], backend: str = "api", limit: int | None = None, stage_a_only: bool = False,
        force: bool = False, dry_run: bool = False, rows_dir: Path = ROWS_DIR, reviewer=None) -> dict:
    all_rows = tl.load_effective(rows_dir)
    by_id = {r["id"]: r for r in all_rows}
    night_rows = [r for r in all_rows if r["id"][:10] in set(dates)]
    recent_accepted = [spoken(r["raw_output"]) for r in all_rows
                       if r["status"] == "accepted" and r["id"][:10] not in set(dates)]
    share = leading_only_share(night_rows)
    batch_note = (f"{share:.0%} of tagged replies in this batch are leading-only "
                  f"(threshold {tags.LEADING_ONLY_FLAG_SHARE:.0%}) — weigh tag_quality accordingly."
                  if share > tags.LEADING_ONLY_FLAG_SHARE else "")
    system = persona_system()
    review_fn = reviewer or (review_api if backend == "api" else review_cli)

    counts: dict[str, int] = defaultdict(int)
    reviewed_now = 0
    for r in night_rows:
        if r.get("human"):
            counts[f"human_{r['human']}"] += 1          # human marks bypass Stage B
            continue
        if r.get("review") is not None and r["status"] != "raw" and not force:
            counts["already_reviewed"] += 1
            continue
        reason = stage_a(r, recent_accepted)
        # Beat-grammar failures are fixable by a reviewer edit (move the group to the
        # nearest clause boundary), so they go to Stage B unless running Stage A only.
        soft = reason in GRAMMAR_CODES and not stage_a_only
        if reason and not soft:
            counts[f"stage_a:{reason}"] += 1
            if not dry_run:
                tl.append_review(r["id"], {"stage": "A", "reason": reason}, "rejected", reason, rows_dir)
            r.update(status="rejected", review={"stage": "A", "reason": reason})
            continue
        if stage_a_only or (limit is not None and reviewed_now >= limit):
            continue
        note = batch_note
        if soft:
            note = (f"{batch_note} Stage A grammar issue: {reason}. Return minor_edit that fixes it "
                    f"(move the tag group to the nearest clause boundary / merge short chunks), or reject.").strip()
        try:
            review = _validate_review(review_fn(r, system, note))
        except Exception as e:
            counts["stage_b_error"] += 1
            print(f"  [B] {r['id']} error: {e}")
            continue
        reviewed_now += 1
        if soft and review.get("verdict") == "accept":
            review = {**review, "verdict": "reject", "downgraded_from": "accept", "downgrade_reason": reason}
            counts[f"stage_a:{reason}"] += 1
        review, downgrade = enforce_edit(r, review)
        downgrade = downgrade or review.get("downgrade_reason")
        review["stage"] = "B"
        status = status_for(review)
        counts[f"stage_b:{status}"] += 1
        if downgrade:
            counts[f"downgrade:{downgrade}"] += 1
        if not dry_run:
            tl.append_review(r["id"], review, status, downgrade, rows_dir)
        r.update(status=status, review=review)
        if status == "accepted":
            recent_accepted.append(spoken(final_output(r)))
        stage_ledger(r, review, dry_run)

    # Re-merge with marks so human verdicts win in the outputs.
    marks = tl.load_marks(rows_dir)
    night_eff = [tl.effective(r, {"review": r.get("review"), "status": r["status"]} if r.get("review") else None,
                              marks.get(r["id"])) for r in night_rows]
    by_id.update({r["id"]: r for r in night_eff})
    night = max(dates)
    outputs = write_outputs(night, night_eff, by_id, dry_run)
    report = {"night": night, "dates": dates, "rows": len(night_rows), "leading_only_share": round(share, 3),
              **dict(counts), **outputs}
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rachel nightly reviewer (offline)")
    ap.add_argument("--date", action="append", help="YYYY-MM-DD (repeatable; default: yesterday + today)")
    ap.add_argument("--backend", choices=("api", "cli"), default="api")
    ap.add_argument("--limit", type=int, default=None, help="max Stage B reviews this run")
    ap.add_argument("--stage-a-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-review rows that already have a review")
    ap.add_argument("--dry-run", action="store_true", help="write nothing")
    a = ap.parse_args()
    today = datetime.now()
    dates = a.date or [(today - timedelta(days=1)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")]
    run(dates, a.backend, a.limit, a.stage_a_only, a.force, a.dry_run)
