"""
data_api.py — Data Explorer backend.

Read-only views over every on-disk store, plus the few writes the training
loop already exposes through its own helpers: human marks
(training_logger.append_mark) and registry promote/rollback.
Images are served only from inside ALLOWED_IMAGE_ROOTS.
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from . import REPO_ROOT, VTUBE_ROOT

ROWS_DIR = REPO_ROOT / "rows"
CURATED_DIR = REPO_ROOT / "curated"
DPO_DIR = REPO_ROOT / "dpo"
DATASETS_DIR = REPO_ROOT / "datasets"
ADAPTERS_DIR = REPO_ROOT / "adapters"
IMAGELOG_DIR = REPO_ROOT / "imagelog"
TRAININGSETS_DIR = VTUBE_ROOT / "TrainingSets"
MEMORY_DIR = REPO_ROOT / "rachel_memory_db"
SESSION_LOG_DIRS = [REPO_ROOT / "session_log", VTUBE_ROOT / "session_log"]
LEGACY_TRAINING_DIRS = [VTUBE_ROOT / "training_data", REPO_ROOT / "training_data"]
ALLOWED_IMAGE_ROOTS = [IMAGELOG_DIR, TRAININGSETS_DIR]
TTS_BASE = "http://127.0.0.1:8051"

router = APIRouter(prefix="/api/data")

_cache: dict[Path, tuple[float, int, list]] = {}


def read_jsonl(path: Path) -> list[dict]:
    """Whole file, cached on (mtime, size). Bad lines are skipped."""
    st = path.stat()
    hit = _cache.get(path)
    if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
        return hit[2]
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    _cache[path] = (st.st_mtime, st.st_size, rows)
    return rows


def page(items: list, offset: int, limit: int) -> dict:
    return {"total": len(items), "offset": offset, "limit": limit, "items": items[offset:offset + limit]}


def safe_image_path(p: str, base: str = "repo") -> Path:
    """Resolve an image path from a row and refuse anything outside the allowed roots."""
    roots = {"repo": REPO_ROOT, "imagelog": IMAGELOG_DIR, "trainingsets": TRAININGSETS_DIR}
    if base not in roots or not p:
        raise HTTPException(400, "bad image path")
    path = Path(p)
    full = (path if path.is_absolute() else roots[base] / path).resolve()
    if not any(full.is_relative_to(r.resolve()) for r in ALLOWED_IMAGE_ROOTS):
        raise HTTPException(403, "outside allowed image roots")
    if full.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp") or not full.is_file():
        raise HTTPException(404, "no such image")
    return full


@router.get("/img", include_in_schema=False)
def image(p: str, base: str = "repo"):
    return FileResponse(safe_image_path(p, base), headers={"Cache-Control": "max-age=3600"})


def _files(d: Path, pattern: str = "*.jsonl") -> list[str]:
    return sorted((p.name for p in d.glob(pattern)), reverse=True) if d.exists() else []


@router.get("/overview")
def overview():
    from training import training_logger as tl
    return {
        "rows": [p.stem for p in tl.row_files(ROWS_DIR)][::-1],
        "curated": _files(CURATED_DIR),
        "dpo": _files(DPO_DIR),
        "datasets": sorted((p.name for p in DATASETS_DIR.iterdir() if p.is_dir()), reverse=True)
                    if DATASETS_DIR.exists() else [],
        "session_logs": [{"dir": str(d), "files": _files(d)} for d in SESSION_LOG_DIRS if d.exists()],
        "legacy_training": [{"dir": str(d), "files": _files(d)} for d in LEGACY_TRAINING_DIRS if d.exists()],
        "trainingsets": sorted(p.parent.parent.name for p in TRAININGSETS_DIR.glob("*/ready/meta.json"))
                        if TRAININGSETS_DIR.exists() else [],
        "ledger": sum(1 for _ in open(REPO_ROOT / "persona" / "ledger_pending.jsonl", encoding="utf-8"))
                  if (REPO_ROOT / "persona" / "ledger_pending.jsonl").exists() else 0,
    }


# ─── Reaction rows ────────────────────────────────────────

def _light(r: dict) -> dict:
    keep = ("id", "ts", "window_start", "window_end", "flush_reason", "input_kind", "frame_paths", "speech", "chat",
            "adapter_version", "latency_ms", "raw_output", "plan", "parse_flags", "playback_trace", "status", "human",
            "human_fix", "review", "review_reason", "prompt_version")
    return {k: r.get(k) for k in keep}


@router.get("/rows")
def rows(date_: str | None = Query(None, alias="date"), status: str | None = None, kind: str | None = None,
         adapter: str | None = None, flag: str | None = None, q: str | None = None,
         order: str = "desc", offset: int = 0, limit: int = 25):
    from training import training_logger as tl
    items = tl.load_effective(ROWS_DIR, [date_] if date_ else None)
    if status:
        items = [r for r in items if r.get("status") == status]
    if kind:
        items = [r for r in items if r.get("input_kind") == kind]
    if adapter:
        items = [r for r in items if r.get("adapter_version") == adapter]
    if flag:
        items = [r for r in items if flag in (r.get("parse_flags") or [])
                 or flag in ((r.get("review") or {}).get("flags") or [])]
    if q:
        ql = q.lower()
        items = [r for r in items if ql in (r.get("raw_output") or "").lower()
                 or any(ql in s.get("text", "").lower() for s in r.get("speech") or [])
                 or any(ql in c.get("text", "").lower() for c in r.get("chat") or [])]
    if order == "desc":
        items = items[::-1]
    out = page(items, offset, limit)
    out["items"] = [_light(r) for r in out["items"]]
    return out


@router.get("/row/{row_id}")
def row_detail(row_id: str):
    from training import training_logger as tl
    from training import row_prompt
    import prompt_format
    d = tl.row_date(row_id)
    prev = (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    by_id = {r["id"]: r for r in tl.load_effective(ROWS_DIR, [prev, d])}
    row = by_id.get(row_id)
    if row is None:
        raise HTTPException(404, f"no row {row_id}")
    try:
        prompt = prompt_format.render_inference(row_prompt.turns_for_row(row, by_id))["prompt"]
    except Exception as e:
        prompt = f"(could not rebuild prompt: {e})"
    return {**row, "uncertainty": tl.uncertainty(row), "exact_prompt": prompt}


@router.post("/mark")
def mark(body: dict):
    from training import training_logger as tl
    row_id, human, fix = body.get("id"), body.get("human"), (body.get("fix_text") or "").strip() or None
    if not row_id or tl.find_row(row_id, ROWS_DIR) is None:
        raise HTTPException(404, f"no row {row_id}")
    try:
        tl.append_mark(row_id, human, fix, ROWS_DIR)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@router.get("/queue")
def queue(n: int = 30, date_: str | None = Query(None, alias="date")):
    from training import training_logger as tl
    items = [r for r in tl.load_effective(ROWS_DIR, [date_] if date_ else None)
             if r.get("human") is None and r.get("status") != "excluded"]
    items.sort(key=tl.uncertainty, reverse=True)
    return {"total": len(items), "items": [{**_light(r), "uncertainty": tl.uncertainty(r)} for r in items[:n]]}


# ─── Generic JSONL stores ─────────────────────────────────

STORES = {
    "reviews": ROWS_DIR / "_review",
    "curated": CURATED_DIR,
    "dpo": DPO_DIR,
}


def _store_file(store: str, name: str) -> Path:
    if store in STORES:
        d = STORES[store]
    elif store == "session_log":
        d = None
        for cand in SESSION_LOG_DIRS:
            if (cand / name).exists():
                d = cand
                break
    elif store == "legacy_training":
        d = next((c for c in LEGACY_TRAINING_DIRS if (c / name).exists()), None)
    else:
        raise HTTPException(400, f"unknown store {store}")
    if d is None or "/" in name or "\\" in name or not name.endswith(".jsonl"):
        raise HTTPException(404, "no such file")
    p = d / name
    if not p.exists():
        raise HTTPException(404, "no such file")
    return p


@router.get("/store/{store}")
def store_files(store: str):
    if store in STORES:
        return {"files": [{"dir": str(STORES[store]), "files": _files(STORES[store])}]}
    if store == "session_log":
        return {"files": [{"dir": str(d), "files": _files(d)} for d in SESSION_LOG_DIRS if d.exists()]}
    if store == "legacy_training":
        return {"files": [{"dir": str(d), "files": _files(d)} for d in LEGACY_TRAINING_DIRS if d.exists()]}
    raise HTTPException(400, f"unknown store {store}")


@router.get("/store/{store}/{name}")
def store_rows(store: str, name: str, offset: int = 0, limit: int = 25, q: str | None = None):
    items = read_jsonl(_store_file(store, name))
    if q:
        ql = q.lower()
        items = [r for r in items if ql in json.dumps(r, ensure_ascii=False).lower()]
    return page(items, offset, limit)


# ─── Datasets / training sets ─────────────────────────────

@router.get("/datasets")
def datasets():
    out = []
    if DATASETS_DIR.exists():
        for d in sorted(DATASETS_DIR.iterdir(), reverse=True):
            m = d / "manifest.json"
            if m.exists():
                out.append(json.loads(m.read_text(encoding="utf-8")))
    return {"items": out}


@router.get("/datasets/{run_id}/{split}")
def dataset_rows(run_id: str, split: str, offset: int = 0, limit: int = 25):
    if split not in ("train", "holdout") or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise HTTPException(400, "bad split/run")
    p = DATASETS_DIR / run_id / f"{split}.jsonl"
    if not p.exists():
        raise HTTPException(404, "no such dataset")
    return page(read_jsonl(p), offset, limit)


def _trainingset_dir(name: str) -> Path:
    d = (TRAININGSETS_DIR / name / "ready").resolve()
    if not d.is_relative_to(TRAININGSETS_DIR.resolve()) or not d.exists():
        raise HTTPException(404, "no such training set")
    return d


@router.get("/trainingset/{name}")
def trainingset(name: str):
    d = _trainingset_dir(name)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
    report = (d / "report.md").read_text(encoding="utf-8") if (d / "report.md").exists() else ""
    splits = {p.stem: len(read_jsonl(p)) for p in d.glob("*.jsonl")}
    return {"meta": meta, "report": report, "splits": splits}


@router.get("/trainingset/{name}/{split}")
def trainingset_rows(name: str, split: str, offset: int = 0, limit: int = 25, q: str | None = None):
    import tag_parser
    if split not in ("train", "holdout", "rejects"):
        raise HTTPException(400, "bad split")
    items = read_jsonl(_trainingset_dir(name) / f"{split}.jsonl")
    if q:
        ql = q.lower()
        items = [r for r in items if ql in (r.get("reply") or r.get("raw_reply") or "").lower()
                 or ql in (r.get("prompt") or "")[-600:].lower()]
    out = page(items, offset, limit)
    for r in out["items"]:
        if "plan" not in r and r.get("raw_reply"):
            r["plan"] = tag_parser.parse(r["raw_reply"]).to_dict()
    return out


# ─── Adapters / registry ──────────────────────────────────

@router.get("/adapters")
def adapters():
    from training import registry
    reg = registry.load()
    for a in reg["adapters"]:
        rep = ADAPTERS_DIR / a["version"] / "eval_report.json"
        a["eval_report"] = json.loads(rep.read_text(encoding="utf-8")) if rep.exists() else None
        a["gguf_exists"] = bool(a.get("gguf")) and (REPO_ROOT / a["gguf"]).exists()
    return reg


@router.post("/adapters/promote")
def promote(body: dict):
    from training import registry
    try:
        registry.promote(body["version"])
    except KeyError:
        raise HTTPException(404, f"unknown adapter {body.get('version')}")
    return registry.load()


@router.post("/adapters/rollback")
def rollback():
    from training import registry
    registry.rollback()
    return registry.load()


# ─── Memory (ChromaDB) ────────────────────────────────────

_chroma = None


def _tts_up() -> bool:
    try:
        return requests.get(f"{TTS_BASE}/memory/stats", timeout=1).ok
    except requests.RequestException:
        return False


@router.get("/memory")
def memory(offset: int = 0, limit: int = 50, emotion: str | None = None, q: str | None = None):
    """Browse via a direct read of the Chroma store (get, no writes); search via server.py when it's up."""
    out: dict = {"server": _tts_up()}
    if out["server"]:
        out["stats"] = requests.get(f"{TTS_BASE}/memory/stats", timeout=3).json()
        if q:
            out["recall"] = requests.post(f"{TTS_BASE}/memory/recall",
                                          json={"query": q, "n_results": limit}, timeout=30).json()
            return out
    try:
        global _chroma
        if _chroma is None:
            import chromadb
            _chroma = chromadb.PersistentClient(path=str(MEMORY_DIR))
        data = _chroma.get_collection("rachel_autobio").get(include=["documents", "metadatas"])
    except Exception as e:
        out["error"] = f"cannot read {MEMORY_DIR}: {e}"
        return out
    items = [{"id": i, "text": d, **(m or {})} for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])]
    if emotion:
        items = [x for x in items if x.get("emotion") == emotion]
    if q and not out["server"]:
        ql = q.lower()
        items = [x for x in items if ql in (x.get("text") or "").lower()]
    items.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
    out.update(page(items, offset, limit))
    return out


@router.post("/memory/dream")
def memory_dream():
    """server.py POST /memory/dream — mechanical merge/decay/prune (memory.py RachelMemory.dream)."""
    try:
        return requests.post(f"{TTS_BASE}/memory/dream", json={}, timeout=300).json()
    except requests.RequestException as e:
        raise HTTPException(503, f"server.py unreachable: {e}")


# ─── Persona / ledger (read-only) ─────────────────────────

@router.get("/ledger")
def ledger(offset: int = 0, limit: int = 50):
    p = REPO_ROOT / "persona" / "ledger_pending.jsonl"
    return page(read_jsonl(p)[::-1] if p.exists() else [], offset, limit)


@router.get("/today")
def today():
    return {"date": date.today().isoformat()}
