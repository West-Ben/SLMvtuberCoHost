"""
app.py — FastAPI app: service control, jobs, settings, stream status, logs
(WebSocket), data explorer and the Dream pipeline. Binds 127.0.0.1 only.
"""

import asyncio
import json
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import LOG_DIR, REPO_ROOT, SRC_DIR
from . import data_api
from . import settings as settings_mod
from .dream import Dream
from .jobs import PY, JobRunner, palette
from .logbus import LogBus
from .services import build_services, llm_port
from .supervisor import StateStore, Supervisor

STATIC = Path(__file__).resolve().parent / "static"
MUTE_LOCK = Path(tempfile.gettempdir()) / "rachel_speaking.lock"
LOCAL_PORTS = {8050, 8051, 8052, 8053, 8055, 8080, 8081}

bus = LogBus(LOG_DIR)
store = StateStore()
sup = Supervisor(build_services(), bus, store)
jobs = JobRunner(bus)
dream = Dream(sup, jobs, bus)


@asynccontextmanager
async def lifespan(app: FastAPI):
    sup.start_monitor()
    bus.publish("panel", "[panel] control panel started")
    yield
    bus.publish("panel", "[panel] shutting down — stopping owned services")
    await run_in_threadpool(sup.shutdown)


app = FastAPI(title="Rachel Control Panel", lifespan=lifespan)
app.include_router(data_api.router)


# ─── Services ─────────────────────────────────────────────

@app.get("/api/services")
def services():
    return {"services": sup.all_info()}


@app.post("/api/services/{name}/{action}")
async def service_action(name: str, action: str, body: dict | None = None):
    if name not in sup.specs:
        raise HTTPException(404, f"unknown service {name}")
    body = body or {}
    if action == "start":
        return await run_in_threadpool(sup.start, name)
    if action == "stop":
        return await run_in_threadpool(sup.stop, name, bool(body.get("force_external")))
    if action == "restart":
        return await run_in_threadpool(sup.restart, name)
    if action == "flags":
        sup.set_flags(name, body.get("flags") or {})
        return sup.info(name)
    if action == "stack":
        sup.set_in_stack(name, bool(body.get("in_stack")))
        return sup.info(name)
    raise HTTPException(400, f"unknown action {action}")


@app.post("/api/stack/start")
async def stack_start(body: dict | None = None):
    body = body or {}
    if body.get("preflight"):
        job = await run_in_threadpool(jobs.run_sync, "preflight", palette()["preflight"]["argv"])
        if job.state != "succeeded" and not body.get("ignore_preflight"):
            return {"started": False, "preflight": job.view(), "error": "preflight failed — see Console"}
    return {"started": sup.start_stack()}


@app.post("/api/stack/stop")
async def stack_stop(body: dict | None = None):
    await run_in_threadpool(sup.stop_stack, bool((body or {}).get("include_external")))
    return {"services": sup.all_info()}


# ─── Jobs / console palette ───────────────────────────────

@app.get("/api/jobs")
def list_jobs():
    return {"jobs": jobs.views(), "palette": {k: {"help": v["help"], "argv": subprocess.list2cmdline(v["argv"])}
                                             for k, v in palette().items()}}


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: int):
    j = jobs.jobs.get(job_id)
    if not j:
        raise HTTPException(404, "no such job")
    return j.view(with_lines=True)


@app.post("/api/jobs/run")
def run_job(body: dict):
    cmd = palette().get(body.get("name", ""))
    if not cmd:
        raise HTTPException(400, "not a palette command")
    return jobs.run(body["name"], cmd["argv"]).view()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: int):
    return {"cancelled": jobs.cancel(job_id)}


@app.post("/api/request")
async def request_inspector(body: dict):
    """Console → send GET/POST to a local service port. Localhost + known ports only."""
    port, path, method = int(body.get("port", 0)), body.get("path", "/"), (body.get("method") or "GET").upper()
    if port not in LOCAL_PORTS or not path.startswith("/") or method not in ("GET", "POST"):
        raise HTTPException(400, "only GET/POST to local service ports")
    url = f"http://127.0.0.1:{port}{path}"

    def go():
        try:
            payload = json.loads(body["body"]) if body.get("body") else None
            r = requests.request(method, url, json=payload, timeout=float(body.get("timeout", 30)))
            try:
                data = r.json()
            except ValueError:
                data = r.text[:20000]
            return {"status": r.status_code, "elapsed_ms": int(r.elapsed.total_seconds() * 1000), "body": data}
        except (requests.RequestException, json.JSONDecodeError) as e:
            return {"status": None, "error": str(e)}
    return await run_in_threadpool(go)


# ─── Logs ─────────────────────────────────────────────────

@app.get("/api/logs/sources")
def log_sources():
    return {"sources": bus.sources(), "status_lines": bus.status_lines}


@app.get("/api/logs/download")
def log_download(source: str, date: str | None = None):
    p = bus.log_file(source, date)
    if not p.exists():
        raise HTTPException(404, "no log for that day")
    return FileResponse(p, media_type="text/plain", filename=f"{source}-{p.stem}.log")


@app.websocket("/ws")
async def ws(sock: WebSocket):
    """All live events: log lines, status lines, service/job/dream changes. ?source= limits log lines."""
    await sock.accept()
    source = sock.query_params.get("source") or None
    q = bus.subscribe(source)
    try:
        await sock.send_json({"type": "backfill", "entries": bus.backfill(source, 800),
                              "status_lines": bus.status_lines})
        while True:
            msg = await q.get()
            await sock.send_json(msg)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        bus.unsubscribe(q)


# ─── Settings ─────────────────────────────────────────────

@app.get("/api/settings")
def get_settings():
    return {"settings": settings_mod.schema(), "groups": settings_mod.GROUP_ORDER, "env_path": str(settings_mod.ENV_PATH)}


@app.post("/api/settings")
def save_settings(body: dict):
    res = settings_mod.update(body.get("changes") or {})
    if res["changed"]:
        bus.publish("panel", f"[settings] changed {', '.join(res['changed'])}")
        sup.mark_restart_required(res["restart"])
    if any(k in res["changed"] for k in ("LLM_URL",)):
        sup.specs["llama"].port = llm_port()
    return res


@app.get("/api/settings/devices")
async def devices(keyword: str | None = None):
    return await run_in_threadpool(settings_mod.audio_devices, keyword)


@app.post("/api/settings/level-test")
async def level_test(body: dict):
    try:
        return await run_in_threadpool(settings_mod.level_test, int(body["index"]), float(body.get("seconds", 3)))
    except Exception as e:
        raise HTTPException(400, f"could not record: {e}")


@app.get("/api/settings/persona")
def persona():
    return {"files": settings_mod.persona_files()}


@app.get("/api/settings/file-exists")
def file_exists(path: str):
    return {"exists": bool(path) and Path(path).exists()}


# ─── Stream status + manual controls ──────────────────────

def _get_json(url: str, timeout: float = 1.5):
    try:
        r = requests.get(url, timeout=timeout)
        return r.json() if r.ok else None
    except (requests.RequestException, ValueError):
        return None


def _vram():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        used, total, util = [int(x.strip()) for x in out.stdout.strip().splitlines()[0].split(",")]
        return {"used_mb": used, "total_mb": total, "util": util}
    except Exception:
        return None


@app.get("/api/stream/status")
async def stream_status():
    def gather():
        from training import registry
        port = llm_port()
        return {
            "bridge": _get_json("http://127.0.0.1:8055/status"),
            "orchestrator": _get_json("http://127.0.0.1:8052/status"),
            "speaking": MUTE_LOCK.exists(),
            "lora": _get_json(f"http://127.0.0.1:{port}/lora-adapters"),
            "registry_current": registry.load().get("current"),
            "vram": _vram(),
        }
    return await run_in_threadpool(gather)


def _post(url: str, payload: dict, timeout: float = 30):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        try:
            return {"status": r.status_code, "body": r.json()}
        except ValueError:
            return {"status": r.status_code, "body": r.text[:2000]}
    except requests.RequestException as e:
        raise HTTPException(503, str(e))


@app.post("/api/stream/chat")
async def stream_chat(body: dict):
    return await run_in_threadpool(_post, "http://127.0.0.1:8052/chat-input",
                                   {"user": body.get("user") or "ben", "text": body.get("text", "")})


@app.post("/api/stream/speak")
async def stream_speak(body: dict):
    return await run_in_threadpool(_post, "http://127.0.0.1:8051/speak", {"text": body.get("text", "")}, 120)


@app.post("/api/stream/avatar")
async def stream_avatar(body: dict):
    payload = {k: body[k] for k in ("expression", "animation") if body.get(k)}
    return await run_in_threadpool(_post, "http://127.0.0.1:8051/avatar", payload)


@app.post("/api/stream/dry-reaction")
def dry_reaction(body: dict):
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text required")
    argv = [*PY, str(SRC_DIR / "orchestrator.py"), "--text", text, "--no-log"]
    if body.get("direct"):
        argv.append("--direct")
    return jobs.run("dry-reaction", argv).view()


@app.get("/api/stream/tags")
def stream_tags():
    import tags
    return {"expressions": sorted(tags.EXPRESSIONS), "animations": sorted(tags.ANIMATIONS),
            "vocals": sorted(tags.VOCAL_TAGS)}


# ─── Dream ────────────────────────────────────────────────

@app.get("/api/dream")
async def dream_state():
    return await run_in_threadpool(dream.state)


@app.post("/api/dream/run")
def dream_run(body: dict):
    return dream.start(body.get("steps") or [], body.get("params") or {}, bool(body.get("stop_live")))


@app.post("/api/dream/cancel")
def dream_cancel():
    dream.cancel()
    return {"ok": True}


# ─── Static UI ────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health", include_in_schema=False)
def health():
    return PlainTextResponse("ok")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
