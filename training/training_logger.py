"""
training/training_logger.py — reaction row logging + quality marking.

Frames path (REACTION_MODE=frames): ReactionRowLogger writes one JSON object
per reaction to rows/YYYY-MM-DD.jsonl (schema: ReactionRow). Rows are
append-only — nothing ever rewrites them. Quality information lives in
sidecars and is merged on read:

  rows/_marks.jsonl              human marks  {"id", "human", "fix_text"?, "ts"}
  rows/_review/YYYY-MM-DD.jsonl  reviewer     {"id", "review", "status", "reason"?, "ts"}

Effective status: human mark > reviewer verdict > "raw". Human marks always
override reviewer verdicts.

CLI (run from the repo root):
  python -m training.training_logger mark <id> good|bad|fix ["fixed text"]
  python -m training.training_logger review-queue [--n 20] [--date YYYY-MM-DD]
  python -m training.training_logger show <id>
  python -m training.training_logger stats [YYYY-MM-DD]
  python -m training.training_logger prune --days N [--apply]   # frames of rejected rows

Legacy text path (REACTION_MODE=text): TrainingLogger, unchanged, still writes
training_data/YYYY-MM-DD.jsonl. CLI: legacy-stats, legacy-build-dataset.
"""

import json
import re
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
ROWS_DIR  = REPO_ROOT / "rows"

SCHEMA_VERSION = 2   # v2: plan + playback_trace (v1 rows carried parsed_segments)
InputKind = Literal["frames", "frames+speech", "frames+chat", "frames+chat+speech"]
Status    = Literal["raw", "reviewed", "accepted", "rejected", "excluded"]
Human     = Literal["good", "bad", "fix"]

_ROW_FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.jsonl$")


# ─── Schema ───────────────────────────────────────────────

class SpeechLine(BaseModel):
    t: float
    text: str


class ChatLine(BaseModel):
    t: float
    user: str
    text: str


class ReactionRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    ts: float
    window_id: str
    input_kind: InputKind
    frame_paths: list[str]
    speech: list[SpeechLine]
    chat: list[ChatLine]
    history_ref: str | None
    prompt_version: str
    adapter_version: str | None
    sampling: dict
    latency_ms: int
    raw_output: str
    plan: dict                          # tag_parser.Plan.to_dict()
    parse_flags: list[str]
    playback_trace: dict | None = None  # playback.PlaybackTrace.to_dict(); None if nothing played
    review: dict | None = None
    human: Human | None = None
    status: Status = "raw"
    # Versioning + exact prompt reconstruction (relative times in the window text)
    schema_version: int = SCHEMA_VERSION
    window_start: float
    window_end: float
    flush_reason: str = ""


def new_row_id(ts: float) -> str:
    dt = datetime.fromtimestamp(ts)
    return f"{dt:%Y-%m-%d}_{dt:%H%M%S}_{uuid.uuid4().hex[:6]}"


def row_date(row_id: str) -> str:
    return row_id[:10]


# ─── Writer (orchestrator) ────────────────────────────────

class ReactionRowLogger:
    """Thread-safe append-only writer used by orchestrator.ReactionEngine."""

    def __init__(self, rows_dir: Path | str = ROWS_DIR):
        self.rows_dir = Path(rows_dir)
        self.rows_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def log_reaction(self, record: dict) -> str:
        data = {k: v for k, v in record.items() if k != "id"}
        data.update(id=new_row_id(record["ts"]), review=None, human=None, status="raw")
        row = ReactionRow.model_validate(data)
        line = row.model_dump_json()
        with self._lock:
            with open(self.rows_dir / f"{row_date(row.id)}.jsonl", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return row.id


# ─── Readers + sidecars ───────────────────────────────────

def _read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def row_files(rows_dir: Path | str = ROWS_DIR, dates: list[str] | None = None) -> list[Path]:
    rows_dir = Path(rows_dir)
    files = sorted(p for p in rows_dir.glob("*.jsonl") if _ROW_FILE_RE.match(p.name))
    if dates:
        files = [p for p in files if p.stem in set(dates)]
    return files


def iter_rows(rows_dir: Path | str = ROWS_DIR, dates: list[str] | None = None) -> Iterator[dict]:
    for p in row_files(rows_dir, dates):
        yield from _read_jsonl(p)


def find_row(row_id: str, rows_dir: Path | str = ROWS_DIR) -> dict | None:
    for r in _read_jsonl(Path(rows_dir) / f"{row_date(row_id)}.jsonl"):
        if r.get("id") == row_id:
            return r
    return None


def marks_path(rows_dir: Path | str = ROWS_DIR) -> Path:
    return Path(rows_dir) / "_marks.jsonl"


def review_path(date: str, rows_dir: Path | str = ROWS_DIR) -> Path:
    return Path(rows_dir) / "_review" / f"{date}.jsonl"


def load_marks(rows_dir: Path | str = ROWS_DIR) -> dict[str, dict]:
    """{id: latest mark}"""
    return {m["id"]: m for m in _read_jsonl(marks_path(rows_dir))}


def load_reviews(rows_dir: Path | str = ROWS_DIR, dates: list[str] | None = None) -> dict[str, dict]:
    """{id: latest review entry}"""
    out: dict[str, dict] = {}
    rdir = Path(rows_dir) / "_review"
    files = sorted(rdir.glob("*.jsonl")) if rdir.exists() else []
    if dates:
        files = [p for p in files if p.stem in set(dates)]
    for p in files:
        for e in _read_jsonl(p):
            out[e["id"]] = e
    return out


def append_mark(row_id: str, human: str, fix_text: str | None = None, rows_dir: Path | str = ROWS_DIR):
    if human not in ("good", "bad", "fix"):
        raise ValueError(f"mark must be good|bad|fix, got {human!r}")
    entry = {"id": row_id, "human": human, "ts": time.time()}
    if fix_text:
        entry["fix_text"] = fix_text
    p = marks_path(rows_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def append_review(row_id: str, review: dict | None, status: str, reason: str | None = None,
                  rows_dir: Path | str = ROWS_DIR):
    """Reviewer writes here — never into the row files."""
    p = review_path(row_date(row_id), rows_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = {"id": row_id, "review": review, "status": status, "ts": time.time()}
    if reason:
        entry["reason"] = reason
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


EXCLUDE_FLAGS = frozenset({"unsafe", "toxic_chat"})


def effective(row: dict, review_entry: dict | None, mark: dict | None) -> dict:
    """Row copy with review/human/status merged. Human mark always wins."""
    r = dict(row)
    if review_entry:
        r["review"] = review_entry.get("review")
        r["status"] = review_entry.get("status", r.get("status", "raw"))
        if review_entry.get("reason"):
            r["review_reason"] = review_entry["reason"]
    flags = set(((r.get("review") or {}).get("flags")) or [])
    if mark:
        r["human"] = mark["human"]
        if mark["human"] == "good":
            r["status"] = "accepted"
        elif mark["human"] == "bad":
            r["status"] = "rejected"
        else:  # fix
            r["status"] = "accepted" if mark.get("fix_text") else "reviewed"
            if mark.get("fix_text"):
                r["human_fix"] = mark["fix_text"]
    # Safety exclusion holds even over a human "good": unsafe/toxic never trains.
    if flags & EXCLUDE_FLAGS:
        r["status"] = "excluded"
    return r


def load_effective(rows_dir: Path | str = ROWS_DIR, dates: list[str] | None = None) -> list[dict]:
    marks, reviews = load_marks(rows_dir), load_reviews(rows_dir, dates)
    return [effective(r, reviews.get(r["id"]), marks.get(r["id"])) for r in iter_rows(rows_dir, dates)]


# ─── Review queue ─────────────────────────────────────────

def uncertainty(row: dict) -> float:
    """
    Higher = Ben should look first. Reviewed rows: closeness of the mean score
    to the 3/5 borderline, plus edits and flags. Unreviewed rows sit mid-queue,
    nudged up by parse flags.
    """
    rev = row.get("review")
    if not rev or not rev.get("scores"):
        return 0.5 + 0.1 * min(len(row.get("parse_flags") or []), 3)
    scores = [v for v in rev["scores"].values() if isinstance(v, (int, float))]
    mean = sum(scores) / len(scores) if scores else 3.0
    u = 1.0 - min(abs(mean - 3.0) / 2.0, 1.0)
    if rev.get("verdict") == "minor_edit":
        u += 0.3
    if rev.get("flags"):
        u += 0.2
    return round(u, 3)


# ─── Retention ────────────────────────────────────────────

def prune_rejected_frames(days: int, apply: bool = False, rows_dir: Path | str = ROWS_DIR,
                          repo_root: Path = REPO_ROOT, now: float | None = None) -> list[Path]:
    """
    Delete frames referenced ONLY by rejected rows older than `days`. A frame
    referenced by any row whose status != rejected is never deleted.
    Dry run unless apply=True. Returns the paths (to be) deleted.
    """
    cutoff = (now or time.time()) - days * 86400
    rows = load_effective(rows_dir)
    protected = {p for r in rows if r["status"] != "rejected" for p in r.get("frame_paths", [])}
    victims: set[str] = set()
    for r in rows:
        if r["status"] == "rejected" and r["ts"] < cutoff:
            victims.update(p for p in r.get("frame_paths", []) if p not in protected)
    out = []
    for rel in sorted(victims):
        path = (repo_root / rel).resolve()
        if not path.is_relative_to((repo_root / "imagelog").resolve()) or not path.exists():
            continue
        out.append(path)
        if apply:
            path.unlink()
            try:
                path.parent.rmdir()   # only succeeds once the window dir is empty
            except OSError:
                pass
    return out


# ─── Legacy text-path logger (REACTION_MODE=text) ─────────

_SCREEN_RE = re.compile(r'\[SCREEN[^\]]*\]\s*(.*?)(?=\n\[|$)', re.DOTALL)
_AUDIO_RE  = re.compile(r'\[AUDIO[^\]]*\]\s*(.*?)(?=\n\[|$)', re.DOTALL)


class TrainingLogger:
    """
    Logs every Rachel interaction as a V6 training example.
    Thread-safe. Writes JSONL to session-dated files.
    """

    def __init__(self, output_dir: str = "./training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._session_date = datetime.now().strftime("%Y-%m-%d")

    def log(
        self,
        messages: list[dict],
        raw_output: str,
        spoken_text: str,
        channel_text: str,
        emotion: str,
        intensity: str,
        context_snapshot: dict,
        trigger: str = "user_input",
        user_text: str = "",
        vision_desc: str = "",
        context_block: str = "",
    ):
        if not raw_output:
            return

        # ── System message: SOUL.md + USER.md + stream context ───────────────
        soul_prompt = ""
        for m in messages:
            if m.get("role") == "system":
                c = m.get("content", "")
                if isinstance(c, str):
                    soul_prompt = c
                break

        ctx_parts: list[str] = []
        game    = context_snapshot.get("game", "").strip()
        scene   = context_snapshot.get("scene", "").strip()
        viewers = context_snapshot.get("viewers")
        if game:    ctx_parts.append(f"Game: {game}")
        if scene:   ctx_parts.append(f"Scene: {scene}")
        if viewers is not None and viewers != 0:
            ctx_parts.append(f"Viewers: {viewers}")
        ctx_str = "\n".join(ctx_parts)

        if soul_prompt and ctx_str:
            system_content = soul_prompt + "\n\n---\n\n" + ctx_str
        elif soul_prompt:
            system_content = soul_prompt
        else:
            system_content = ctx_str or "Stream active"

        # ── User turn: last real input, skip context/summary/audio blocks ─────
        user_content = user_text  # prefer explicit transcription text if provided
        if not user_content:
            for m in reversed(messages):
                if m.get("role") != "user":
                    continue
                content = m.get("content", "")
                if isinstance(content, list):
                    continue  # audio / vision block
                if "<<STREAM_CONTEXT>>" in content:
                    continue  # live context block
                if content.startswith("[Earlier this stream"):
                    continue  # session summary
                user_content = content
                break

        # ── Assemble training example ─────────────────────────────────────────
        train_messages: list[dict] = [
            {"role": "system", "content": system_content},
        ]
        if user_content:
            train_messages.append({"role": "user", "content": user_content})
        train_messages.append({"role": "model", "content": raw_output})

        row: dict = {"messages": train_messages}
        if context_block:
            row["context_block"] = context_block
        if channel_text:
            row["channel_text"] = channel_text
        if emotion and emotion != "neutral":
            row["emotion"] = emotion
        if intensity and intensity != "medium":
            row["intensity"] = intensity
        if vision_desc:
            sm = _SCREEN_RE.search(vision_desc)
            am = _AUDIO_RE.search(vision_desc)
            if sm:
                row["screen_desc"] = sm.group(1).strip()
            if am:
                row["audio_desc"] = am.group(1).strip()

        with self._lock:
            filepath = self.output_dir / f"{self._session_date}.jsonl"
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def session_stats(self) -> dict:
        filepath = self.output_dir / f"{self._session_date}.jsonl"
        if not filepath.exists():
            return {"total": 0, "file": str(filepath)}

        total = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1

        return {"total": total, "file": str(filepath)}


# ─── CLI ──────────────────────────────────────────────────

def _cmd_mark(args: list[str]):
    if len(args) < 2:
        raise SystemExit("usage: mark <id> good|bad|fix [\"fixed text\"]")
    row_id, human = args[0], args[1]
    if find_row(row_id) is None:
        raise SystemExit(f"no row {row_id}")
    fix_text = args[2] if len(args) > 2 else None
    if human == "fix" and not fix_text:
        print("note: 'fix' without text marks the row for editing; it won't train until a fix text is given")
    append_mark(row_id, human, fix_text)
    print(f"marked {row_id} {human}")


def _cmd_review_queue(args: list[str]):
    n = int(args[args.index("--n") + 1]) if "--n" in args else 20
    dates = [args[args.index("--date") + 1]] if "--date" in args else None
    rows = [r for r in load_effective(dates=dates) if r.get("human") is None and r["status"] != "excluded"]
    rows.sort(key=uncertainty, reverse=True)
    for r in rows[:n]:
        out = r["raw_output"].replace("\n", " ")
        verdict = (r.get("review") or {}).get("verdict", "-")
        print(f"{uncertainty(r):.2f}  {r['id']}  {r['status']:<9} {verdict:<10} {r['input_kind']:<18} {out[:90]}")


def _cmd_show(args: list[str]):
    if not args:
        raise SystemExit("usage: show <id>")
    row = find_row(args[0])
    if row is None:
        raise SystemExit(f"no row {args[0]}")
    marks, reviews = load_marks(), load_reviews(dates=[row_date(row["id"])])
    r = effective(row, reviews.get(row["id"]), marks.get(row["id"]))
    t0 = r["window_start"]
    print(f"id        {r['id']}   status={r['status']}  human={r.get('human')}  kind={r['input_kind']}")
    print(f"prompt    {r['prompt_version']}   adapter={r['adapter_version']}   latency={r['latency_ms']}ms")
    print("frames:")
    for p in r["frame_paths"]:
        print(f"  {REPO_ROOT / p}")
    print("speech:")
    for s in r["speech"]:
        print(f"  +{s['t'] - t0:5.1f}s  {s['text']}")
    print("chat:")
    for c in r["chat"]:
        print(f"  +{c['t'] - t0:5.1f}s  {c['user']}: {c['text']}")
    print(f"output:   {r['raw_output']}")
    print(f"flags:    {r['parse_flags']}")
    if r.get("review"):
        print(f"review:   {json.dumps(r['review'], ensure_ascii=False)}")
    if r.get("human_fix"):
        print(f"fix:      {r['human_fix']}")


def _cmd_stats(args: list[str]):
    dates = [args[0]] if args else None
    rows = load_effective(dates=dates)
    by_status: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        by_kind[r["input_kind"]] = by_kind.get(r["input_kind"], 0) + 1
    print(json.dumps({"total": len(rows), "status": by_status, "input_kind": by_kind}, indent=2))


def _cmd_prune(args: list[str]):
    if "--days" not in args:
        raise SystemExit("usage: prune --days N [--apply]")
    days = int(args[args.index("--days") + 1])
    apply = "--apply" in args
    paths = prune_rejected_frames(days, apply=apply)
    print(f"{'deleted' if apply else 'would delete'} {len(paths)} frame(s) of rejected rows older than {days}d")
    if not apply and paths:
        print("re-run with --apply to delete")


def _cmd_legacy_stats(args: list[str]):
    date = args[0] if args else datetime.now().strftime("%Y-%m-%d")
    logger = TrainingLogger()
    logger._session_date = date
    print(json.dumps(logger.session_stats(), indent=2))


def _cmd_legacy_build_dataset(args: list[str]):
    output_path = "rachel-v6-dataset.jsonl"
    if "--output" in args:
        output_path = args[args.index("--output") + 1]

    data_dir = Path("./training_data")
    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for jsonl_file in sorted(data_dir.glob("*.jsonl")):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.write(line + "\n")
                        count += 1
    print(f"Built dataset: {count} examples → {output_path}")


_USAGE = __doc__.split("CLI (run from the repo root):", 1)[1].split("Legacy text path", 1)[0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:" + _USAGE + "  legacy-stats [YYYY-MM-DD] | legacy-build-dataset [--output out.jsonl]")
        sys.exit(0)
    _cmd, _rest = sys.argv[1], sys.argv[2:]
    _commands = {
        "mark": _cmd_mark, "review-queue": _cmd_review_queue, "show": _cmd_show,
        "stats": _cmd_stats, "prune": _cmd_prune,
        "legacy-stats": _cmd_legacy_stats, "legacy-build-dataset": _cmd_legacy_build_dataset,
    }
    if _cmd not in _commands:
        print(f"Unknown command: {_cmd}\nUsage:{_USAGE}")
        sys.exit(1)
    _commands[_cmd](_rest)
