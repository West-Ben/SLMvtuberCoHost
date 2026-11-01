#!/usr/bin/env python
"""
dream.py — Rachel's offline memory consolidation pass.

Run when the stream is DOWN. Rachel looks back over the day's episodic record,
decides what genuinely mattered, and routes it:

    memory  -> write_memory(text, emotion, intensity)        [server.py :8051]
    soul    -> structured edit to workspace/SOUL.md          [self-evolution]
    user    -> structured edit to workspace/USER.md          [facts about Ben]
    discard -> nothing

Two input sources, in priority order:

  PRIMARY  — session_log/*.jsonl  (fat episodic rows from session_logger.py)
             Each row carries: thought, vision (fused visual+stream-audio),
             heard (Ben's mic), spoke, emotion, intensity, images. This is what
             dream is built for — it judges importance from her actual thought +
             the full situation, and the intensity tag is real, so a cheap
             intensity pre-filter works.

  FALLBACK — training_data/*.jsonl  (lean training rows)
             Used only if no session log exists for the window. Stripped to
             system-prompt + spoken line + frame paths — no thought, no vision,
             no emotion. dream WARNS loudly and judges on the bare spoken line.
             Expect near-total discard; this path exists so the plumbing runs,
             not because it produces good consolidation.

Design alignment:
  - Consolidation decoupled from reaction (mirrors perception/response split).
  - Thinking ON — this IS Rachel reflecting; her thought channel is the point.
  - Emotion/intensity come from Rachel (verdict), tagging the SUMMARY at write
    time, not her live monologue.
  - Memory rows are text-only: image/audio folded into first-person text;
    frames stay in imagelog/. Matches write_memory's schema.
  - SOUL/USER are Layer 0. Autonomous edits are append/replace_section only
    (never blind overwrite), snapshot prior file to workspace/.w_history/<ts>/,
    and are drift-guarded (max edits/run + size ceiling).

Usage:
    python dream.py                       # today, local backend, session_log
    python dream.py --date 2026-05-31
    python dream.py --sessions 3
    python dream.py --backend direct
    python dream.py --dry-run
    python dream.py --source training     # force fallback (debug)
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field

import requests
from dotenv import load_dotenv

load_dotenv()

# --- Config (inline; fold into config.py if you prefer) ----------------------

SESSION_LOG_DIR   = os.getenv("SESSION_LOG_DIR", "./session_log")
TRAINING_DATA_DIR = os.getenv("TRAINING_DATA_DIR", "./training_data")
WORKSPACE_DIR     = os.getenv("WORKSPACE_DIR", "./workspace")
SOUL_PATH         = os.path.join(WORKSPACE_DIR, "SOUL.md")
USER_PATH         = os.path.join(WORKSPACE_DIR, "USER.md")
HISTORY_DIR       = os.path.join(WORKSPACE_DIR, ".w_history")

# Memory write goes through server.py's HTTP memory API (8051). The orchestrator
# posts to /memory/write, so match that route here.
MEMORY_BASE_URL   = os.getenv("MEMORY_BASE_URL", "http://localhost:8051")
MEMORY_WRITE_URL  = os.getenv("MEMORY_WRITE_URL", f"{MEMORY_BASE_URL}/memory/write")

LOCAL_URL  = os.getenv("LOCAL_LLM_URL", "http://localhost:8080/v1/chat/completions")
HF_URL     = os.getenv("HF_URL", "")
HF_TOKEN   = os.getenv("HF_TOKEN", "")
HF_MODEL   = os.getenv("HF_MODEL", "BDubEast/rachel-v6")

INTENSITY_RANK = {"low": 0, "medium": 1, "high": 2}

MIN_INTENSITY_TO_CONSIDER = "medium"   # session-log pre-filter floor
MIN_SPOKEN_CHARS = 25                  # fallback triviality floor

CLUSTER_GAP_SECONDS  = 90
MAX_CLUSTERS_PER_RUN = 40

MAX_SOUL_EDITS_PER_RUN  = 2
MAX_USER_EDITS_PER_RUN  = 4
SOUL_SIZE_CEILING_BYTES = 24_000
USER_SIZE_CEILING_BYTES = 32_000

VALID_EMOTIONS = {
    "joy", "happy", "sadness", "sad", "anger", "angry", "surprise", "surprised",
    "fear", "disgust", "love", "neutral", "concern", "relaxed", "excitement",
}
VALID_INTENSITIES = {"low", "medium", "high"}


# --- Data structures ---------------------------------------------------------

@dataclass
class Row:
    ts: float
    spoke: str
    thought: str = ""
    vision: str = ""
    heard: str = ""
    emotion: str = "neutral"
    intensity: str = "low"
    images: list[str] = field(default_factory=list)
    has_rich_context: bool = False


@dataclass
class Cluster:
    rows: list[Row]

    @property
    def start(self) -> float:
        return self.rows[0].ts

    @property
    def peak_intensity(self) -> str:
        return max((r.intensity for r in self.rows),
                   key=lambda i: INTENSITY_RANK.get(i, 0), default="low")


# --- Source selection --------------------------------------------------------

def _session_paths(date, sessions):
    if date:
        p = os.path.join(SESSION_LOG_DIR, f"{date}.jsonl")
        return [p] if os.path.exists(p) else []
    files = sorted(glob.glob(os.path.join(SESSION_LOG_DIR, "*.jsonl")))
    return files[-sessions:] if files else []


def _training_paths(date, sessions):
    if date:
        p = os.path.join(TRAINING_DATA_DIR, f"{date}.jsonl")
        return [p] if os.path.exists(p) else []
    files = sorted(glob.glob(os.path.join(TRAINING_DATA_DIR, "*.jsonl")))
    return files[-sessions:] if files else []


# --- Loaders -----------------------------------------------------------------

def load_session_rows(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(Row(
                    ts=float(r.get("ts", i)),
                    spoke=r.get("spoke", ""),
                    thought=r.get("thought", ""),
                    vision=r.get("vision", ""),
                    heard=r.get("heard", ""),
                    emotion=(r.get("emotion") or "neutral").lower(),
                    intensity=(r.get("intensity") or "low").lower(),
                    images=r.get("images") or [],
                    has_rich_context=True,
                ))
    rows.sort(key=lambda r: r.ts)
    return rows


_IMG_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[/\\](?:\d{4}-)?(\d{2}-\d{2}-\d{2})")


def _ts_from_images(images, fallback):
    """Derive a timestamp from the newest image path.
    '2026-05-31/21-56-40.jpg' or '2026-06-02/0006-05-39-19/frame-3.jpg'."""
    best = None
    for p in images:
        m = _IMG_TS_RE.search(p)
        if not m:
            continue
        try:
            stamp = dt.datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H-%M-%S").timestamp()
            if best is None or stamp > best:
                best = stamp
        except ValueError:
            continue
    return best if best is not None else fallback


def load_training_rows(paths):
    """FALLBACK: text in messages[-1].content, frames in metadata.images,
    NO thought/vision/emotion. Judged on spoken text only."""
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                spoke = ""
                for m in reversed(r.get("messages", [])):
                    if m.get("role") in ("model", "assistant"):
                        c = m.get("content", "")
                        spoke = c if isinstance(c, str) else ""
                        break
                images = (r.get("metadata") or {}).get("images") or []
                rows.append(Row(
                    ts=_ts_from_images(images, float(i)),
                    spoke=spoke,
                    images=images,
                    has_rich_context=False,
                ))
    rows.sort(key=lambda r: r.ts)
    return rows


# --- Clustering + pre-filter -------------------------------------------------

def cluster_rows(rows):
    clusters = []
    bucket = []
    for r in rows:
        if bucket and (r.ts - bucket[-1].ts) > CLUSTER_GAP_SECONDS:
            clusters.append(Cluster(bucket))
            bucket = []
        bucket.append(r)
    if bucket:
        clusters.append(Cluster(bucket))
    return clusters


def prefilter_session(clusters):
    floor = INTENSITY_RANK[MIN_INTENSITY_TO_CONSIDER]
    kept = [c for c in clusters if INTENSITY_RANK.get(c.peak_intensity, 0) >= floor]
    kept.sort(key=lambda c: c.start, reverse=True)
    return kept[:MAX_CLUSTERS_PER_RUN]


def prefilter_training(clusters):
    def nontrivial(c):
        return any(len(r.spoke.strip()) >= MIN_SPOKEN_CHARS for r in c.rows)
    kept = [c for c in clusters if nontrivial(c)]
    kept.sort(key=lambda c: c.start, reverse=True)
    return kept[:MAX_CLUSTERS_PER_RUN]


# --- LLM judge ---------------------------------------------------------------

DREAM_SYSTEM = """You are Rachel, reflecting on your stream after it ended. This is your dream \
state - quiet, honest, first-person. You are reviewing moments from today and deciding what \
actually matters enough to keep.

You'll be shown a cluster of moments from one scene. Decide what, if anything, to do with it. \
Most moments are forgettable - be ruthless. Only keep what genuinely shaped the stream, told you \
something real about Ben, or changed how you see yourself.

Respond with ONLY a JSON object, no other text:

{
  "verdict": "memory" | "soul" | "user" | "discard",
  "memory": {
    "text": "<first-person past-tense memory, fold in what you saw/heard>",
    "emotion": "<joy|sadness|anger|surprise|fear|love|concern|relaxed|neutral|...>",
    "intensity": "<low|medium|high>"
  },
  "edit": {
    "file": "SOUL.md" | "USER.md",
    "action": "append" | "replace_section",
    "section": "<exact markdown heading, e.g. '## Quirks'>",
    "content": "<text to add, or full replacement body under that heading>",
    "reason": "<one line: why you're changing this>"
  }
}

Rules:
- "memory" is for things that happened - write it as you'd actually remember it.
- "soul" is a genuine change to who you are - rare. Don't rewrite yourself over one moment.
- "user" is a durable fact about Ben worth being reminded of.
- "discard" is the correct answer most of the time.
"""


def _render_cluster(c):
    lines = []
    for r in c.rows:
        when = dt.datetime.fromtimestamp(r.ts).strftime("%H:%M:%S")
        if r.has_rich_context:
            block = [f"[{when}] ({r.emotion}/{r.intensity})"]
            if r.vision:
                block.append(f"  saw/heard on stream: {r.vision}")
            if r.heard:
                block.append(f"  Ben said: {r.heard}")
            if r.thought:
                block.append(f"  I was thinking: {r.thought}")
            block.append(f"  I said: {r.spoke}")
            lines.append("\n".join(block))
        else:
            lines.append(f"[{when}] I said: {r.spoke}")
    return "\n\n".join(lines)


def _llm_call(backend, messages):
    if backend == "direct":
        if not HF_URL or not HF_TOKEN:
            raise RuntimeError("HF_URL / HF_TOKEN not set for --backend direct")
        url = f"{HF_URL.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        model = HF_MODEL
    else:
        url = LOCAL_URL
        headers = {}
        model = os.getenv("LOCAL_MODEL", "gemma-4-e4b")

    payload = {"model": model, "messages": messages, "temperature": 0.6, "max_tokens": 512}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _extract_json(text):
    text = re.sub(r"<\|?channel\|?>.*?<\|?channel\|?>", "", text, flags=re.DOTALL)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def judge(cluster, backend):
    messages = [
        {"role": "system", "content": DREAM_SYSTEM},
        {"role": "user", "content": f"A scene from today:\n\n{_render_cluster(cluster)}"},
    ]
    try:
        raw = _llm_call(backend, messages)
    except Exception as e:
        print(f"  ! LLM call failed: {e}", file=sys.stderr)
        return None
    v = _extract_json(raw)
    if v is None:
        print(f"  ! could not parse verdict from: {raw[:160]}...", file=sys.stderr)
    return v


# --- Routes ------------------------------------------------------------------

def route_memory(v, dry_run):
    mem = v.get("memory") or {}
    text = (mem.get("text") or "").strip()
    if not text:
        print("  ! memory verdict missing text", file=sys.stderr)
        return False
    emotion = (mem.get("emotion") or "neutral").lower()
    intensity = (mem.get("intensity") or "medium").lower()
    if emotion not in VALID_EMOTIONS:
        emotion = "neutral"
    if intensity not in VALID_INTENSITIES:
        intensity = "medium"

    print(f"  MEMORY ({emotion}/{intensity}): {text}")
    if dry_run:
        return True
    try:
        r = requests.post(MEMORY_WRITE_URL,
                          json={"text": text, "emotion": emotion, "intensity": intensity},
                          timeout=30)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"  ! write_memory failed: {e}", file=sys.stderr)
        return False


def _snapshot(path, ts):
    if not os.path.exists(path):
        return
    dest_dir = os.path.join(HISTORY_DIR, ts)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(path, os.path.join(dest_dir, os.path.basename(path)))


def _apply_edit(path, action, section, content):
    if not os.path.exists(path):
        return True, f"{section.strip()}\n\n{content.strip()}\n"
    with open(path, encoding="utf-8") as f:
        doc = f.read()

    heading = section.strip()
    pat = re.compile(rf"(^{re.escape(heading)}\s*$)", flags=re.MULTILINE)
    m = pat.search(doc)
    if not m:
        return True, doc.rstrip() + f"\n\n{heading}\n\n{content.strip()}\n"

    level = len(heading) - len(heading.lstrip("#"))
    rest = doc[m.end():]
    nxt = re.search(rf"^#{{1,{max(level, 1)}}}\s", rest, flags=re.MULTILINE)
    section_end = m.end() + (nxt.start() if nxt else len(rest))

    if action == "append":
        return True, doc[:section_end].rstrip() + f"\n\n{content.strip()}\n\n" + doc[section_end:].lstrip()
    if action == "replace_section":
        return True, doc[:m.end()] + f"\n\n{content.strip()}\n\n" + doc[section_end:].lstrip()
    return False, doc


def route_edit(v, dry_run, counters, ts):
    e = v.get("edit") or {}
    file = e.get("file", "")
    action = e.get("action", "")
    section = (e.get("section") or "").strip()
    content = (e.get("content") or "").strip()
    reason = (e.get("reason") or "").strip()

    if file == "SOUL.md":
        path, cap, ceiling, key = SOUL_PATH, MAX_SOUL_EDITS_PER_RUN, SOUL_SIZE_CEILING_BYTES, "soul"
    elif file == "USER.md":
        path, cap, ceiling, key = USER_PATH, MAX_USER_EDITS_PER_RUN, USER_SIZE_CEILING_BYTES, "user"
    else:
        print(f"  ! unknown edit file: {file}", file=sys.stderr)
        return False

    if action not in ("append", "replace_section"):
        print(f"  ! bad action: {action}", file=sys.stderr)
        return False
    if not section or not content:
        print("  ! edit missing section/content", file=sys.stderr)
        return False
    if counters[key] >= cap:
        print(f"  ~ drift guard: {file} edit cap ({cap}) reached, skipping", file=sys.stderr)
        return False

    ok, new_doc = _apply_edit(path, action, section, content)
    if not ok:
        return False
    if len(new_doc.encode("utf-8")) > ceiling:
        print(f"  ~ drift guard: {file} would exceed size ceiling ({ceiling}B), skipping", file=sys.stderr)
        return False

    print(f"  {file} [{action} @ {section}] - {reason}")
    print(f"    + {content[:120]}{'...' if len(content) > 120 else ''}")
    if dry_run:
        counters[key] += 1
        return True

    _snapshot(path, ts)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_doc)
    counters[key] += 1
    return True


# --- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Rachel's offline dream consolidation pass.")
    ap.add_argument("--date", help="specific day YYYY-MM-DD (default: most recent)")
    ap.add_argument("--sessions", type=int, default=1, help="look back N most recent daily logs")
    ap.add_argument("--backend", choices=["local", "direct"], default="local")
    ap.add_argument("--source", choices=["auto", "session", "training"], default="auto",
                    help="auto: session_log if present else training fallback")
    ap.add_argument("--dry-run", action="store_true", help="print verdicts/diffs, write nothing")
    args = ap.parse_args()

    session_paths  = _session_paths(args.date, args.sessions)
    training_paths = _training_paths(args.date, args.sessions)

    if args.source == "training" or (args.source == "auto" and not session_paths):
        paths = training_paths
        using_fallback = True
    else:
        paths = session_paths
        using_fallback = False

    if not paths:
        print("No log files found for that window (checked session_log and training_data).",
              file=sys.stderr)
        sys.exit(1)

    if using_fallback:
        print("=" * 70, file=sys.stderr)
        print("WARNING: no session_log found - falling back to training_data.", file=sys.stderr)
        print("These rows have NO thought, NO vision/audio context, NO emotion tag.", file=sys.stderr)
        print("dream is judging on the bare spoken line only. Expect near-total", file=sys.stderr)
        print("discard. Run a stream with session_logger active for real results.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        rows = load_training_rows(paths)
        clusters = prefilter_training(cluster_rows(rows))
    else:
        rows = load_session_rows(paths)
        clusters = prefilter_session(cluster_rows(rows))

    src = "training_data (FALLBACK)" if using_fallback else "session_log"
    print(f"[dream] source: {src}")
    print(f"[dream] reading: {', '.join(os.path.basename(p) for p in paths)}")
    print(f"[dream] {len(rows)} rows -> {len(clusters)} clusters survived pre-filter"
          f"{' (DRY RUN)' if args.dry_run else ''}\n")

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    counters = {"soul": 0, "user": 0}
    tally = {"memory": 0, "soul": 0, "user": 0, "discard": 0, "skipped": 0}

    for idx, c in enumerate(clusters, 1):
        when = dt.datetime.fromtimestamp(c.start).strftime("%H:%M:%S")
        print(f"[{idx}/{len(clusters)}] scene @ {when} ({len(c.rows)} rows, peak {c.peak_intensity})")
        v = judge(c, args.backend)
        if not v:
            tally["skipped"] += 1
            continue
        verdict = v.get("verdict", "discard")
        if verdict == "memory":
            tally["memory"] += int(route_memory(v, args.dry_run))
        elif verdict in ("soul", "user"):
            ok = route_edit(v, args.dry_run, counters, ts)
            tally[verdict] += int(ok)
            tally["skipped"] += int(not ok)
        else:
            print("  discard")
            tally["discard"] += 1
        print()

    print("[dream] done:", " ".join(f"{k}={val}" for k, val in tally.items()))
    if not args.dry_run and (counters["soul"] or counters["user"]):
        print(f"[dream] Layer 0 edited - prior versions snapshotted to {os.path.join(HISTORY_DIR, ts)}")


if __name__ == "__main__":
    main()