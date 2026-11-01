"""
training/convert_legacy_csv.py — repair the V7 legacy tagged-reply CSVs into
upload-ready SFT rows in the CURRENT grammar (tags.py) and the exact prompt
format served at inference (prompt_format.py).

Sources (matched by normalised file name, numeric "(N)" suffixes / spaces / underscores ignored):
  single_2  Rachel-training-v7-single*.csv   one viewer line → one reply      weight 2
  v7_1      Rachel-training-v7*.csv          2–5 model turns, one example per model turn   weight 1
  rework    Rachel-training-v7-rework*.csv   NOT trained on — diff count for the report only
  images    Rachel-training-images-V7.csv    not used

Per reply (order matters; every change is recorded in `edits`):
  reject   leak / empty / bare-word pseudo-tag at the start
  rewrite  {expr:neutral}→{expr:relaxed}  {expr:excitement}→{expr:joy}  {anim:nod}→{anim:agree}
           {expr:thinking}: drop (default) | {anim:think} (--thinking-policy think)
           {anim:clap}: removed; other alias spellings → canonical; vocals → canonical spelling
  reject   any other unknown {..} or [..]  (unknown_tag:X)
  repair   tag group mid-phrase → moved to the nearest clause boundary (moved_group)
           more than MAX_BEATS_PER_REPLY beats → later beats dropped (dropped_beat)
  validate tag_parser.parse → non-empty plan with NO flags; reviewer.stage_a (frames check
           bypassed with a sentinel); chunk_too_short etc. reject. Never adds words.

System text always comes from persona/SYSTEM_PROMPT.md through prompt_format; the
CSV system prompts are discarded. Zero frames, input_kind "legacy_text".

Run (from the repo root):
  python -m training.convert_legacy_csv --in path/to/TrainingSets/version7 \
      --out path/to/TrainingSets/version7/ready [--thinking-policy drop|think] [--seed 3407]
"""

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import prompt_format                                        # noqa: E402
import tag_parser                                           # noqa: E402
import tags                                                 # noqa: E402
from config import BASE_CORE_HF, REACTION_HISTORY_PAIRS     # noqa: E402
from training import reviewer                               # noqa: E402

SOURCE_WEIGHT = {"single_2": 2, "v7_1": 1}
INPUT_KIND = "legacy_text"
V7_1_CAP = 450
MAX_EXPR_SHARE = 0.35
MAX_UNTAGGED_SHARE = 0.15
HOLDOUT_SHARE = 0.08
HOLDOUT_MIN = 40
LEGACY_FRAME_SENTINEL = "legacy:no-frames"   # only satisfies reviewer.stage_a's no_frames check
CHAT_T = 0.0                                 # legacy rows have no timestamps: every chat line at +0.0s
DEFAULT_V7_1_AUTHOR = "viewer_1"
UPLOAD_CMD = ("huggingface-cli upload BDubEast/rachel-v7-sft "
              "path/to/TrainingSets/version7/ready . --repo-type dataset --private")

# Explicit legacy rewrites from the brief: (kind, legacy name) → (replacement text, edit label)
EXPLICIT_REWRITES = {
    ("expr", "neutral"):    ("{expr:relaxed}", "expr_neutral->relaxed"),
    ("expr", "excitement"): ("{expr:joy}", "expr_excitement->joy"),
    ("anim", "nod"):        ("{anim:agree}", "anim_nod->agree"),
    ("anim", "clap"):       ("", "anim_clap_removed"),
}
THINKING_POLICY = {
    "drop":  ("", "expr_thinking_dropped"),
    "think": ("{anim:think}", "expr_thinking->anim_think"),
}

# Legacy bare-word pseudo-tags ("neutral yeah that's…"): expression names and the legacy
# names that alias to an expression or to thinking — derived from tags.py, not copied.
BARE_WORDS = frozenset(tags.EXPRESSIONS) | {
    k for k, v in tags.TAG_ALIASES.items() if v in tags.EXPRESSIONS or k == "thinking"
}
_BARE_RE = re.compile(r"^\s*([a-z]+)\s", re.IGNORECASE)
_ANY_BRACE_RE = re.compile(r"\{[^{}]*\}")
_BRACKET_RE = re.compile(r"\[([^\[\]\n]{1,40})\]")
_CANON_TAG_RE = re.compile(r"\{(?:expr|anim):[a-z0-9_]+\}|\[[a-z ]+\]")
_SPACE_PUNCT_RE = re.compile(r"\s+([,.!?;:…])")
_VIEWER_LINE_RE = re.compile(r"^\s*(viewer_\d+)\s*:\s*(.+?)\s*$", re.DOTALL)
_CLAUSE_SPLIT_RE = re.compile("(?<=[" + re.escape(tags.CLAUSE_END) + r"])\s+")


class Reject(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ─── Input loading ────────────────────────────────────────

def norm_name(stem: str) -> str:
    """'01_Rachel-training-v7 (1)' / 'Rachel-training-v7 1' / 'Rachel_training_v7' → 'racheltrainingv7'."""
    s = stem.lower().strip()
    s = re.sub(r"\(\d+\)", "", s)        # export suffix "(1)"
    s = re.sub(r"^\d+[\s_\-]*", "", s)   # numeric prefix
    s = re.sub(r"[\s_]\d+$", "", s)      # trailing " 1"
    return re.sub(r"[\s_\-]+", "", s)


SOURCE_NAMES = {
    "racheltrainingv7single": "single_2",
    "racheltrainingv7": "v7_1",
    "racheltrainingv7rework": "rework",
    "racheltrainingimagesv7": "images",
}
OPTIONAL_NAMES = {"rachelv7singleconformant"}


def unwrap(x):
    while isinstance(x, str):
        x = json.loads(x)
    return x


def read_csv(path: Path) -> list[dict]:
    csv.field_size_limit(10**9)
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def messages_of(row: dict) -> list[dict]:
    m = unwrap(row["messages"])
    return m["messages"] if isinstance(m, dict) else m


def discover(in_dir: Path, repo_root: Path = ROOT) -> tuple[dict[str, Path], list[str]]:
    """{source: path} plus notes about optional / extra files (listed, never included)."""
    found: dict[str, list[Path]] = defaultdict(list)
    notes = []
    for d in (in_dir, repo_root):
        for p in sorted(d.glob("*.csv")):
            key = norm_name(p.stem)
            if key in OPTIONAL_NAMES:
                notes.append(f"optional file present, NOT included: {p.as_posix()}")
            elif key in SOURCE_NAMES and d == in_dir:
                found[SOURCE_NAMES[key]].append(p)
            elif key == "racheltrainingv7single" and d == repo_root:
                notes.append(f"optional file present in repo root, NOT included: {p.as_posix()}")
    chosen = {}
    for src, paths in found.items():
        if src == "single_2" and len(paths) > 1:
            # The ~8,951-row single export shares the name; the primary is the small one.
            paths = sorted(paths, key=lambda p: len(read_csv(p)))
            for extra in paths[1:]:
                notes.append(f"optional file present, NOT included: {extra.as_posix()} ({len(read_csv(extra))} rows)")
        elif len(paths) > 1:
            raise SystemExit(f"ambiguous {src} inputs: {[p.name for p in paths]}")
        chosen[src] = paths[0]
    for src in ("single_2", "v7_1"):
        if src not in chosen:
            raise SystemExit(f"missing required input for {src} in {in_dir.as_posix()}")
    return chosen, notes


# ─── Reply normalisation ──────────────────────────────────

def _collapse(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return _SPACE_PUNCT_RE.sub(r"\1", text)


def rewrite_tags(reply: str, policy: str, edits: list[str]) -> str:
    """Explicit legacy rewrites, alias/canonical spelling, unknown → Reject."""
    def brace(m: re.Match) -> str:
        kind, raw_name = m.group(1).lower(), m.group(2)
        name = tags.norm_name(raw_name)
        if kind == "expr" and name == "thinking":
            repl, label = THINKING_POLICY[policy]
            edits.append(label)
            return repl
        if (kind, name) in EXPLICIT_REWRITES:
            repl, label = EXPLICIT_REWRITES[(kind, name)]
            edits.append(label)
            return repl
        value = tags.resolve(kind, name)
        if value is None:
            raise Reject(f"unknown_tag:{name}")
        canon = f"{{{kind}:{value}}}"
        if value != name:
            edits.append(f"alias:{kind}_{name}->{value}")
        elif m.group(0) != canon:
            edits.append("canonical_tag")
        return canon

    out = tags.BRACE_TAG_RE.sub(brace, reply)
    leftover = _ANY_BRACE_RE.search(tags.BRACE_TAG_RE.sub(" ", out))   # anything that isn't a valid tag
    if leftover:
        raise Reject(f"unknown_tag:{tags.norm_name(leftover.group(0).strip('{}'))[:24] or 'empty_braces'}")
    if "{" in tags.BRACE_TAG_RE.sub(" ", out) or "}" in tags.BRACE_TAG_RE.sub(" ", out):
        raise Reject("malformed")

    def bracket(m: re.Match) -> str:
        name = tags.norm_vocal(m.group(1))
        if name not in tags.VOCAL_TAGS:
            raise Reject(f"unknown_tag:{name[:24]}")
        canon = f"[{name}]"
        if m.group(0) != canon:
            edits.append("canonical_vocal")
        return canon

    return _BRACKET_RE.sub(bracket, out)


def precheck(raw: str) -> None:
    if raw is None or not str(raw).strip():
        raise Reject("empty")
    if any(tok in raw for tok in prompt_format.GEMMA_SPECIAL_TOKENS) or "<|" in raw or "|>" in raw \
            or prompt_format.LLAMACPP_MEDIA_MARKER in raw:
        raise Reject("leak")
    m = _BARE_RE.match(raw)
    if m and not raw.lstrip().startswith(("{", "[")) and m.group(1).lower() in BARE_WORDS:
        raise Reject("bare_word_tag")


# Units: ("text", str) | ("beat", raw_span, n_brace_tags) | ("vocal", raw)

def to_units(reply: str) -> list[tuple]:
    units: list[tuple] = []
    pos = 0
    clean_since = False   # only whitespace since the last tag
    for m in _CANON_TAG_RE.finditer(reply):
        between = reply[pos:m.start()]
        if between.strip():
            for piece in _CLAUSE_SPLIT_RE.split(between.strip()):
                if piece:
                    units.append(("text", piece))
            clean_since = False
        tok = m.group(0)
        if tok.startswith("{"):
            if units and units[-1][0] == "beat" and clean_since and not units[-1][3]:
                kind, span, n, has_vocal = units[-1]
                units[-1] = ("beat", f"{span} {tok}" if between else span + tok, n + 1, False)
            else:
                units.append(("beat", tok, 1, False))
        else:
            if units and units[-1][0] == "beat" and clean_since:
                kind, span, n, _ = units[-1]
                units[-1] = ("beat", f"{span} {tok}" if between else span + tok, n, True)
            else:
                units.append(("vocal", tok))
        clean_since = True
        pos = m.end()
    tail = reply[pos:]
    if tail.strip():
        for piece in _CLAUSE_SPLIT_RE.split(tail.strip()):
            if piece:
                units.append(("text", piece))
    return units


def render_units(units: list[tuple]) -> str:
    return _collapse(" ".join(u[1] for u in units))


def _words_before(units: list[tuple], k: int) -> int:
    return sum(len(u[1].split()) for u in units[:k] if u[0] == "text")


def _mid_beat(units: list[tuple]) -> int | None:
    """Index of the first beat that splits a phrase (mirrors reviewer.grammar_issue)."""
    for i, u in enumerate(units):
        if u[0] != "beat":
            continue
        prev = units[i - 1] if i > 0 else None
        speech_after = any(v[0] == "text" for v in units[i + 1:])
        if prev and prev[0] == "text" and prev[1][-1] not in tags.CLAUSE_END and speech_after:
            return i
    return None


def _valid_slots(units: list[tuple]) -> list[int]:
    """Insert positions k (before units[k]) that are clause boundaries and won't merge with
    another beat or capture an unbound vocal."""
    n = len(units)
    slots = []
    for k in range(n + 1):
        prev = units[k - 1] if k > 0 else None
        nxt = units[k] if k < n else None
        if (prev and prev[0] == "beat") or (nxt and nxt[0] in ("beat", "vocal")):
            continue
        if k == 0 or k == n or prev[0] == "vocal" or (prev[0] == "text" and prev[1][-1] in tags.CLAUSE_END):
            slots.append(k)
    return slots


def relocate_groups(units: list[tuple], edits: list[str]) -> list[tuple]:
    for _ in range(len(units)):
        i = _mid_beat(units)
        if i is None:
            return units
        beat = units[i]
        origin = _words_before(units, i)
        rest = units[:i] + units[i + 1:]
        slots = _valid_slots(rest)
        if not slots:
            raise Reject("tag_mid_phrase")
        # nearest by words; ties go to the earlier boundary (beat before the words it colours)
        k = min(slots, key=lambda s: (abs(_words_before(rest, s) - origin), s))
        units = rest[:k] + [beat] + rest[k:]
        edits.append("moved_group")
    raise Reject("tag_mid_phrase")


def cap_beats(units: list[tuple], edits: list[str]) -> list[tuple]:
    out, seen = [], 0
    for u in units:
        if u[0] == "beat":
            seen += 1
            if seen > tags.MAX_BEATS_PER_REPLY:
                edits.append("dropped_beat")
                continue
        out.append(u)
    return out


def normalize_reply(raw: str, policy: str = "drop", context_texts: list[str] | None = None) -> tuple[str, list[str], tag_parser.Plan]:
    """→ (reply, edits, plan). Raises Reject(reason)."""
    precheck(raw)
    edits: list[str] = []
    if re.search(r"\s{2,}|\s+[,.!?;:…]", raw.strip()):
        edits.append("whitespace")          # extra spaces in the source itself (spoken text unchanged)
    reply = _collapse(rewrite_tags(raw, policy, edits))   # collapse also closes gaps left by removed tags
    if not reply:
        raise Reject("empty")
    units = to_units(reply)
    units = relocate_groups(units, edits)
    units = cap_beats(units, edits)
    reply = render_units(units)

    plan = tag_parser.parse(reply)
    if plan.is_noop:
        raise Reject("empty")
    if plan.flags:
        raise Reject(plan.flags[0])
    issue = reviewer.grammar_issue(plan)
    if issue:
        raise Reject(issue)
    row = {"frame_paths": [LEGACY_FRAME_SENTINEL], "raw_output": reply,
           "speech": [{"t": CHAT_T, "text": t} for t in (context_texts or [])]}
    reason = reviewer.stage_a(row, [])
    if reason:
        raise Reject(reason)
    return reply, edits, plan


def strip_for_history(raw: str) -> str:
    """Fallback history text for a rejected prior reply: spoken words + known vocals only."""
    text = _ANY_BRACE_RE.sub(" ", raw or "")
    text = _BRACKET_RE.sub(lambda m: f" [{tags.norm_vocal(m.group(1))}] " if tags.is_vocal(m.group(1)) else " ", text)
    return _collapse(text)


# ─── Examples ─────────────────────────────────────────────

def chat_entry(author: str, text: str) -> dict:
    return {"t": CHAT_T, "user": author, "text": text.strip()}


def window_text(chat: list[dict]) -> str:
    return prompt_format.format_window_text([], chat, 0, CHAT_T, CHAT_T)


def build_example(*, ex_id: str, source: str, session_id: str, system: str, history: list[dict],
                  chat: list[dict], reply: str, raw_reply: str, edits: list[str], plan) -> dict:
    turns = prompt_format.build_reaction_messages(system, history, [], [], chat, CHAT_T, CHAT_T)
    ex = prompt_format.render_training_example(turns, reply)
    prompt = ex["prompt"]
    # Hard invariants — byte-identical to what inference renders.
    assert prompt == prompt_format.render(turns, add_generation_prompt=True)
    assert prompt.startswith(prompt_format.BOS) and prompt.endswith(prompt_format.GENERATION_PREFIX)
    assert prompt_format.IMAGE_TOKEN not in prompt and not ex["images"]
    assert ex["completion"] == reply + prompt_format.EOT and not ex["completion"].endswith("\n")
    return {
        "id": ex_id, "source": source, "session_id": session_id, "weight": SOURCE_WEIGHT[source],
        "input_kind": INPUT_KIND, "prompt": prompt, "completion": ex["completion"],
        "reply": reply, "raw_reply": raw_reply, "edits": edits, "plan": plan.to_dict(),
        "prompt_version": prompt_format.prompt_version(system), "grammar_hash": tags.grammar_hash(),
    }


def convert_single(rows: list[dict], system: str, policy: str, rejects: list, stats: Counter) -> list[dict]:
    out, seen = [], set()
    for r in rows:
        msgs = messages_of(r)
        users = [m["content"] for m in msgs if m["role"] == "user"]
        models = [m["content"] for m in msgs if m["role"] == "model"]
        stats["read"] += len(models)
        sid = f"single2-{int(r['id']):06d}" if str(r["id"]).isdigit() else f"single2-{r['id']}"
        base = {"source": "single_2", "session_id": sid, "id": f"{sid}-t1"}
        if len(users) != 1 or len(models) != 1:
            rejects.append({**base, "reason": "shape", "raw_reply": models[0] if models else ""})
            stats["rej:shape"] += 1
            continue
        m = _VIEWER_LINE_RE.match(users[0])
        if not m:
            rejects.append({**base, "reason": "user_line_format", "raw_reply": models[0], "user": users[0]})
            stats["rej:user_line_format"] += 1
            continue
        author, text = m.group(1), m.group(2)
        try:
            reply, edits, plan = normalize_reply(models[0], policy, [text])
        except Reject as e:
            rejects.append({**base, "reason": e.reason, "raw_reply": models[0]})
            stats[f"rej:{e.reason}"] += 1
            continue
        if reply in seen:
            rejects.append({**base, "reason": "duplicate", "raw_reply": models[0]})
            stats["rej:duplicate"] += 1
            continue
        seen.add(reply)
        stats["repaired" if any(e != "whitespace" for e in edits) else "clean"] += 1
        out.append(build_example(ex_id=base["id"], source="single_2", session_id=sid, system=system,
                                 history=[], chat=[chat_entry(author, text)], reply=reply,
                                 raw_reply=models[0], edits=edits, plan=plan))
    stats["kept"] = len(out)
    return out


def convert_v7_1(rows: list[dict], system: str, policy: str, rejects: list, stats: Counter) -> list[dict]:
    out, seen = [], set()
    for r in rows:
        msgs = [m for m in messages_of(r) if m["role"] != "system"]
        sid = f"v7_1-{int(r['id']):06d}" if str(r["id"]).isdigit() else f"v7_1-{r['id']}"
        history: list[dict] = []
        pending: list[dict] = []          # chat entries since the last model turn
        turn = 0
        for m in msgs:
            content = m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
            if m["role"] == "user":
                pending.append(chat_entry(DEFAULT_V7_1_AUTHOR, content))
                continue
            if m["role"] != "model":
                continue
            turn += 1
            stats["read"] += 1
            ex_id = f"{sid}-t{turn}"
            base = {"source": "v7_1", "session_id": sid, "id": ex_id}
            chat = pending or []
            pending = []
            try:
                if not chat:
                    raise Reject("no_user_turn")
                reply, edits, plan = normalize_reply(content, policy, [c["text"] for c in chat])
                if reply in seen:
                    raise Reject("duplicate")
            except Reject as e:
                rejects.append({**base, "reason": e.reason, "raw_reply": content})
                stats[f"rej:{e.reason}"] += 1
                hist_text = strip_for_history(content)
                if chat and hist_text:
                    history.append({"user": window_text(chat), "assistant": hist_text})
                continue
            seen.add(reply)
            stats["repaired" if any(e != "whitespace" for e in edits) else "clean"] += 1
            out.append(build_example(ex_id=ex_id, source="v7_1", session_id=sid, system=system,
                                     history=history[-REACTION_HISTORY_PAIRS:], chat=chat, reply=reply,
                                     raw_reply=content, edits=edits, plan=plan))
            history.append({"user": window_text(chat), "assistant": reply})
    stats["kept"] = len(out)
    return out


# ─── Selection + holdout ──────────────────────────────────

def first_expr(ex: dict) -> str | None:
    return next((s["expr"] for s in ex["plan"]["steps"] if s["type"] == "act" and s.get("expr")), None)


def has_vocal(ex: dict) -> bool:
    return any(s["type"] == "vocal" for s in ex["plan"]["steps"])


def has_anim(ex: dict) -> bool:
    return any(s["type"] == "act" and s.get("anim") for s in ex["plan"]["steps"])


def is_untagged(ex: dict) -> bool:
    return not any(s["type"] in ("act", "vocal") for s in ex["plan"]["steps"])


def stratum(ex: dict) -> tuple:
    return (first_expr(ex) or "-", has_vocal(ex), has_anim(ex))


def select_v7_1(examples: list[dict], cap: int, rng: random.Random) -> list[dict]:
    """
    Stratified by (first expr, has_vocal, has_anim), one example per conversation
    first, then extras; no single expr > MAX_EXPR_SHARE, untagged ≤ MAX_UNTAGGED_SHARE
    of the final set.
    """
    pool = list(examples)
    rng.shuffle(pool)
    first_of_session, extras = [], []
    taken = set()
    for ex in pool:
        (extras if ex["session_id"] in taken else first_of_session).append(ex)
        taken.add(ex["session_id"])

    def take(tier: list[dict], chosen: list[dict], counts: Counter, n_untagged: list[int]):
        buckets = defaultdict(list)
        for ex in tier:
            buckets[stratum(ex)].append(ex)
        keys = sorted(buckets, key=str)
        while len(chosen) < cap and any(buckets[k] for k in keys):
            for k in keys:
                if not buckets[k] or len(chosen) >= cap:
                    continue
                ex = buckets[k].pop(0)
                e = first_expr(ex)
                if e and counts[e] + 1 > MAX_EXPR_SHARE * cap:
                    continue
                if is_untagged(ex) and n_untagged[0] + 1 > MAX_UNTAGGED_SHARE * cap:
                    continue
                chosen.append(ex)
                if e:
                    counts[e] += 1
                if is_untagged(ex):
                    n_untagged[0] += 1

    chosen: list[dict] = []
    counts: Counter = Counter()
    n_untagged = [0]
    take(first_of_session, chosen, counts, n_untagged)
    take(extras, chosen, counts, n_untagged)

    # Shares are against the FINAL size; if we came in under cap, trim the over-represented.
    for _ in range(len(chosen)):
        n = len(chosen)
        over = [e for e, c in Counter(first_expr(x) for x in chosen if first_expr(x)).items() if c > MAX_EXPR_SHARE * n]
        untagged_over = sum(is_untagged(x) for x in chosen) > MAX_UNTAGGED_SHARE * n
        if not over and not untagged_over:
            break
        victim_pred = (lambda x: first_expr(x) in over) if over else is_untagged
        # drop an extra (second+ per conversation) first, else any
        idx = next((i for i in range(n - 1, -1, -1) if victim_pred(chosen[i])), None)
        chosen.pop(idx)
    return chosen


def holdout_split(examples: list[dict], rng: random.Random) -> tuple[list[dict], list[dict]]:
    """~HOLDOUT_SHARE of sessions per source (stratified), at least HOLDOUT_MIN examples; disjoint sessions."""
    by_source: dict[str, list[str]] = defaultdict(list)
    size: Counter = Counter()
    for ex in examples:
        size[ex["session_id"]] += 1
        if ex["session_id"] not in by_source[ex["source"]]:
            by_source[ex["source"]].append(ex["session_id"])
    order: dict[str, list[str]] = {}
    hold: set[str] = set()
    for src in sorted(by_source):
        sess = sorted(by_source[src])
        rng.shuffle(sess)
        order[src] = sess
        k = max(1, round(len(sess) * HOLDOUT_SHARE))
        hold.update(sess[:k])
        order[src] = sess[k:]
    srcs = sorted(order)
    i = 0
    while sum(size[s] for s in hold) < HOLDOUT_MIN and any(order.values()):
        src = srcs[i % len(srcs)]
        if order[src]:
            hold.add(order[src].pop(0))
        i += 1
    train = [e for e in examples if e["session_id"] not in hold]
    holdout = [e for e in examples if e["session_id"] in hold]
    assert not ({e["session_id"] for e in train} & {e["session_id"] for e in holdout})
    return train, holdout


# ─── Rework diff ──────────────────────────────────────────

def rework_diff(v7_rows: list[dict], rework_rows: list[dict]) -> dict:
    def turns(row):
        return [(m["role"], m["content"]) for m in messages_of(row) if m["role"] != "system"]
    a = {r["id"]: turns(r) for r in v7_rows}
    b = {r["id"]: turns(r) for r in rework_rows}
    common = sorted(set(a) & set(b), key=lambda x: int(x) if str(x).isdigit() else x)
    same = sum(1 for k in common if a[k] == b[k])
    model_turn_diffs = sum(
        sum(1 for (ra, ca), (rb, cb) in zip(a[k], b[k]) if ra == rb == "model" and ca != cb)
        + abs(len(a[k]) - len(b[k])) // 2 for k in common)
    user_turn_diffs = sum(sum(1 for (ra, ca), (rb, cb) in zip(a[k], b[k]) if ra == rb == "user" and ca != cb)
                          for k in common)
    paired = [(ca, cb) for k in common for (_, ca), (_, cb) in zip(a[k], b[k])]
    identical_turns = sum(1 for ca, cb in paired if ca == cb)
    return {"v7_1_conversations": len(a), "rework_conversations": len(b), "common_ids": len(common),
            "only_in_v7_1": len(set(a) - set(b)), "only_in_rework": len(set(b) - set(a)),
            "identical_conversations": same, "differing_conversations": len(common) - same,
            "differing_model_turns": model_turn_diffs, "differing_user_turns": user_turn_diffs,
            "paired_turns": len(paired), "identical_turns": identical_turns,
            "system_prompt": "rework carries the real SOUL.md text; v7_1 a placeholder (both discarded)"}


# ─── Report ───────────────────────────────────────────────

def _pct(a, b):
    return f"{(100 * a / b):.1f}%" if b else "n/a"


def _token_counter(tokenizer_path: Path):
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tokenizer_path))
        return (lambda s: len(tok.encode(s, add_special_tokens=False).ids)), f"Gemma tokenizer ({tokenizer_path.as_posix()})"
    except Exception as e:
        return (lambda s: round(len(s) / 4)), f"chars/4 estimate (tokenizer unavailable: {e.__class__.__name__})"


def distributions(examples: list[dict], count_tokens) -> dict:
    exprs, anims, vocals, firsts, beats = Counter(), Counter(), Counter(), Counter(), Counter()
    words, toks = [], []
    for ex in examples:
        steps = ex["plan"]["steps"]
        for s in steps:
            if s["type"] == "act":
                if s.get("expr"):
                    exprs[s["expr"]] += 1
                if s.get("anim"):
                    anims[s["anim"]] += 1
            elif s["type"] == "vocal":
                vocals[s["tag"]] += 1
        firsts[first_expr(ex) or "(none)"] += 1
        beats[sum(1 for s in steps if s["type"] == "act")] += 1
        words.append(sum(len(s["text"].split()) for s in steps if s["type"] == "speak"))
        toks.append(count_tokens(ex["prompt"]) + count_tokens(ex["completion"]))
    n = max(len(examples), 1)

    def pctl(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))] if xs else 0
    return {
        "n": len(examples), "expr": dict(exprs.most_common()), "first_expr": dict(firsts.most_common()),
        "anim": dict(anims.most_common()), "vocal": dict(vocals.most_common()),
        "vocal_share": round(sum(has_vocal(e) for e in examples) / n, 4),
        "untagged_share": round(sum(is_untagged(e) for e in examples) / n, 4),
        "beats_per_reply": dict(sorted(beats.items())),
        "words": {"min": min(words, default=0), "median": statistics.median(words) if words else 0,
                  "p95": pctl(words, 0.95), "max": max(words, default=0)},
        "tokens_prompt_plus_completion": {"p50": pctl(toks, 0.5), "p95": pctl(toks, 0.95), "max": max(toks, default=0)},
    }


def write_report(out: Path, ctx: dict):
    L = ["# Rachel V7 legacy → current-grammar SFT set", "",
         f"Generated {ctx['created']} · prompt_version `{ctx['prompt_version']}` · grammar_hash `{ctx['grammar_hash']}` · "
         f"thinking policy `{ctx['policy']}` · seed {ctx['seed']}", ""]
    L += ["## 1. Funnel", "", "| | single_2 | v7_1 |", "|---|---|---|"]
    fs, fv = ctx["funnel"]["single_2"], ctx["funnel"]["v7_1"]
    reasons = sorted({k for f in (fs, fv) for k in f if k.startswith("rej:")})
    L.append(f"| replies read (model turns) | {fs['read']} | {fv['read']} |")
    for k in reasons:
        L.append(f"| rejected · {k[4:]} | {fs.get(k, 0)} | {fv.get(k, 0)} |")
    for label, key in (("passed clean (no edits)", "clean"), ("repaired (≥1 edit)", "repaired"), ("kept", "kept"),
                       ("selected", "selected"), ("train", "train"), ("holdout", "holdout")):
        L.append(f"| {label} | {fs.get(key, 0)} | {fv.get(key, 0)} |")
    L += ["", f"Train sessions: {ctx['train_sessions']} · holdout sessions: {ctx['holdout_sessions']} · overlap: 0", ""]

    L += ["## 2. Distributions (selected = train + holdout)", ""]
    for name, d in ctx["dist"].items():
        L += [f"### {name} (n={d['n']})", "",
              f"- vocal share **{_pct(d['vocal_share'] * d['n'], d['n'])}**, untagged share **{_pct(d['untagged_share'] * d['n'], d['n'])}**",
              f"- first expr: {d['first_expr']}", f"- expr: {d['expr']}", f"- anim: {d['anim']}", f"- vocal: {d['vocal']}",
              f"- beats per reply: {d['beats_per_reply']}",
              f"- spoken words: {d['words']}",
              f"- prompt+completion tokens: {d['tokens_prompt_plus_completion']} ({ctx['token_method']})", ""]

    L += ["## 3. Before / after examples", ""]
    for ex in ctx["examples"]:
        L += [f"**{ex['id']}** — edits: `{', '.join(ex['edits'])}`", "",
              f"- before: `{ex['raw_reply']}`", f"- after:  `{ex['reply']}`", ""]

    d = ctx["rework"]
    L += ["## 4. rework vs v7_1", "",
          f"- conversations: v7_1 {d['v7_1_conversations']}, rework {d['rework_conversations']}, common ids {d['common_ids']} "
          f"(only v7_1: {d['only_in_v7_1']}, only rework: {d['only_in_rework']})",
          f"- identical conversations (ignoring system): **{d['identical_conversations']}**, differing: **{d['differing_conversations']}**",
          f"- differing model turns: {d['differing_model_turns']}, differing user turns: {d['differing_user_turns']}",
          f"- turn-by-turn: {d.get('identical_turns', 0)} of {d.get('paired_turns', 0)} paired turns identical — "
          "**rework is not a light edit of v7_1**: same id space, mostly different conversations (worth checking "
          "before it is ever used).",
          f"- {d['system_prompt']}. rework is not trained on.", ""]

    L += ["## 5. Assumptions and deviations", ""] + [f"- {a}" for a in ctx["assumptions"]] + [""]
    if ctx["notes"]:
        L += ["Files:", ""] + [f"- {n}" for n in ctx["notes"]] + [""]
    L += ["## 6. Upload (not run)", "", "```", UPLOAD_CMD, "```", ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


README = """---
license: other
task_categories: [text-generation]
pretty_name: Rachel V7 legacy SFT (current tag grammar)
---

# rachel-v7-sft

Supervised fine-tuning rows for Rachel (Gemma 4 E4B co-host), converted from the
V7 legacy CSVs into the **current** inline tag grammar (`{expr:X}`, `{anim:X}`,
vocal `[sigh]` …) and the exact prompt string the runtime renders
(`prompt_format.py`, thinking off, zero frames).

Files: `train.jsonl`, `holdout.jsonl` (split by `session_id`, no overlap),
`tags_snapshot.json` (grammar at build time), `meta.json`, `rejects.jsonl`, `report.md`.

Row fields: `id, source, session_id, weight, input_kind ("legacy_text"), prompt,
completion, reply, raw_reply, edits, plan, prompt_version, grammar_hash`.

Training: feed `prompt + completion` as text and compute loss on `completion`
only (it ends with `<turn|>`; `prompt` already ends with the generation prefix
`<|turn>model\\n<|channel>thought\\n<channel|>`). Do not re-template. `weight` is
a sampling weight (single_2 rows = 2). Text-only rows: no `<|image|>` tokens.
"""


# ─── Driver ───────────────────────────────────────────────

def run(in_dir: Path, out_dir: Path, policy: str = "drop", seed: int = 3407,
        tokenizer_path: Path | None = None) -> dict:
    sources, notes = discover(in_dir)
    system = prompt_format.load_system_prompt()
    rejects: list[dict] = []
    funnel = {"single_2": Counter(), "v7_1": Counter()}

    single_rows = read_csv(sources["single_2"])
    v7_rows = read_csv(sources["v7_1"])
    single = convert_single(single_rows, system, policy, rejects, funnel["single_2"])
    v7 = convert_v7_1(v7_rows, system, policy, rejects, funnel["v7_1"])

    rng = random.Random(seed)
    v7_sel = select_v7_1(v7, V7_1_CAP, rng)
    selected = single + v7_sel
    funnel["single_2"]["selected"], funnel["v7_1"]["selected"] = len(single), len(v7_sel)
    train, holdout = holdout_split(selected, rng)
    for split_name, split in (("train", train), ("holdout", holdout)):
        for ex in split:
            funnel[ex["source"]][split_name] += 1

    # Hard gate on every output row.
    for ex in train + holdout:
        p = tag_parser.parse(ex["reply"])
        assert not p.flags and not p.is_noop, (ex["id"], p.flags)
        assert ex["completion"] == ex["reply"] + prompt_format.EOT
    assert not ({e["session_id"] for e in train} & {e["session_id"] for e in holdout})

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("holdout", holdout), ("rejects", rejects)):
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    snapshot = {"expressions": sorted(tags.EXPRESSIONS), "animations": sorted(tags.ANIMATIONS),
                "vocal_tags": sorted(tags.VOCAL_TAGS), "aliases": tags.TAG_ALIASES,
                "default_expression": tags.DEFAULT_EXPRESSION, "max_act_groups": tags.MAX_ACT_GROUPS,
                "min_chunk_words": tags.MIN_CHUNK_WORDS, "max_beats_per_reply": tags.MAX_BEATS_PER_REPLY,
                "clause_end": tags.CLAUSE_END, "placement_rule": tags.PLACEMENT_RULE,
                "grammar_hash": tags.grammar_hash()}
    (out_dir / "tags_snapshot.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    count_tokens, token_method = _token_counter(tokenizer_path or Path(BASE_CORE_HF) / "tokenizer.json")
    dist = {"all": distributions(train + holdout, count_tokens),
            "single_2": distributions([e for e in train + holdout if e["source"] == "single_2"], count_tokens),
            "v7_1": distributions([e for e in train + holdout if e["source"] == "v7_1"], count_tokens)}
    rework = rework_diff(v7_rows, read_csv(sources["rework"])) if "rework" in sources else {
        "v7_1_conversations": len(v7_rows), "rework_conversations": 0, "common_ids": 0, "only_in_v7_1": len(v7_rows),
        "only_in_rework": 0, "identical_conversations": 0, "differing_conversations": 0,
        "differing_model_turns": 0, "differing_user_turns": 0, "system_prompt": "rework file not found"}

    edited = [e for e in train + holdout if any(x != "whitespace" for x in e["edits"])]
    ex_rng = random.Random(seed)
    moved = [e for e in edited if "moved_group" in e["edits"]]
    others = [e for e in edited if "moved_group" not in e["edits"]]
    picks = ex_rng.sample(moved, min(5, len(moved)))
    picks += ex_rng.sample(others, min(10 - len(picks), len(others)))

    assumptions = [
        "A1 · v7_1 user lines have no author, so every one is a chat entry from `viewer_1`.",
        "A2 · Legacy rows have no timestamps or frames: chat lines render at `+0.0s` and the prompt has no "
        "`[N frames …]` header (runtime prompts always have one) — input_kind `legacy_text`.",
        f"A3 · `{{expr:thinking}}` policy `{policy}`" + (" (removed)." if policy == "drop" else " (→ `{anim:think}`)."),
        "single_2: the CSV's second system message (`<<STREAM_CONTEXT>>` with ~10 `[RECENT CHAT]` lines) is "
        "dropped per the brief; only the replied-to `viewer_N:` line is kept, so some replies lose context.",
        "Stage A runs via `reviewer.stage_a` with a sentinel frame path (`legacy:no-frames`) to bypass only the "
        "no_frames check; the replied-to chat text is passed as `speech` so the echo check applies; the "
        "repetition check runs with an empty history (exact duplicates are dropped per source instead).",
        "v7_1 history: a prior reply that was rejected is kept in history with its tags stripped (spoken words + "
        "known vocals); accepted prior replies appear normalised. History capped at REACTION_HISTORY_PAIRS "
        f"({REACTION_HISTORY_PAIRS}).",
        "Group relocation moves a whole beat (tag group + its bound vocals) to the nearest clause boundary by word "
        "distance (ties → earlier); it never lands next to another beat or in front of an unbound vocal. "
        "Beat cap drops whole later beats (group + bound vocals).",
        "Other alias spellings that `tags.resolve` accepts are rewritten to the canonical name (edit `alias:…`); "
        "brace-wrapped vocals like `{sigh}` and `{warudo}` are unknown tags → rejected.",
        "Bare-word pseudo-tag = reply starting with an expression name or an expression alias "
        "(e.g. `neutral yeah …`); derived from tags.py.",
        "Holdout: ≈8% of sessions per source, topped up alternately until ≥40 examples.",
        "images CSV has columns `id, image, text, message, reviewed` (not `messages`) — unused as instructed.",
    ]
    ctx = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "prompt_version": prompt_format.prompt_version(system),
           "grammar_hash": tags.grammar_hash(), "policy": policy, "seed": seed, "funnel": funnel,
           "train_sessions": len({e["session_id"] for e in train}),
           "holdout_sessions": len({e["session_id"] for e in holdout}),
           "dist": dist, "token_method": token_method, "examples": picks, "rework": rework,
           "assumptions": assumptions, "notes": notes}
    write_report(out_dir, ctx)
    (out_dir / "README.md").write_text(README, encoding="utf-8")

    def sha(p: Path):
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    meta = {
        "created": ctx["created"], "seed": seed, "thinking_policy": policy,
        "prompt_version": ctx["prompt_version"], "grammar_hash": ctx["grammar_hash"],
        "input_kind": INPUT_KIND, "inputs": {k: {"path": v.as_posix(), "sha256_16": sha(v)} for k, v in sources.items()},
        "counts": {"train": len(train), "holdout": len(holdout), "rejects": len(rejects),
                   "train_by_source": dict(Counter(e["source"] for e in train)),
                   "holdout_by_source": dict(Counter(e["source"] for e in holdout))},
        "funnel": {k: dict(v) for k, v in funnel.items()},
        "vocal_share": dist["all"]["vocal_share"], "untagged_share": dist["all"]["untagged_share"],
        "caps": {"v7_1": V7_1_CAP, "max_expr_share": MAX_EXPR_SHARE, "max_untagged_share": MAX_UNTAGGED_SHARE,
                 "holdout_share": HOLDOUT_SHARE, "holdout_min": HOLDOUT_MIN},
        "rework_diff": rework, "notes": notes,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta["counts"], indent=2))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Repair V7 legacy CSVs into current-grammar SFT rows")
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--thinking-policy", choices=("drop", "think"), default="drop")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json for token stats (default: BASE_CORE_HF)")
    a = ap.parse_args()
    run(Path(a.in_dir), Path(a.out_dir), a.thinking_policy, a.seed, Path(a.tokenizer) if a.tokenizer else None)
