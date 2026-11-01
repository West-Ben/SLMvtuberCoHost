"""
jobs.py — one-shot subprocesses (preflight, training steps, CLI helpers).

Each job streams its output to the LogBus as source "job:<name>", keeps its
own lines, and on exit tries to parse the trailing JSON block the training
CLIs print (reviewer report, dataset manifest, eval summary …).
"""

import itertools
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

from . import REPO_ROOT, SRC_DIR, child_env
from .logbus import LineSplitter, LogBus
from .supervisor import kill_tree, spawn_flags

PY = [sys.executable, "-u"]


@dataclass
class Job:
    id: int
    name: str
    argv: list[str]
    state: str = "queued"           # queued · running · succeeded · failed · cancelled
    rc: int | None = None
    started: float | None = None
    ended: float | None = None
    lines: list[str] = field(default_factory=list)
    result: dict | list | None = None
    proc: subprocess.Popen | None = None
    group: str | None = None        # chain id, for the Dream page

    @property
    def source(self) -> str:
        return f"job:{self.name}"

    def view(self, with_lines: bool = False) -> dict:
        d = {"id": self.id, "name": self.name, "argv": subprocess.list2cmdline(self.argv), "state": self.state,
             "rc": self.rc, "started": self.started, "ended": self.ended, "result": self.result,
             "group": self.group, "n_lines": len(self.lines), "source": self.source}
        if with_lines:
            d["lines"] = self.lines[-2000:]
        return d


def trailing_json(lines: list[str]):
    """The last JSON object/array printed at the end of the output, if any."""
    for i in range(len(lines) - 1, max(-1, len(lines) - 400), -1):
        s = lines[i].lstrip()
        if not s.startswith(("{", "[")):
            continue
        try:
            return json.loads("\n".join(lines[i:]).strip())
        except json.JSONDecodeError:
            continue
    return None


class JobRunner:
    def __init__(self, bus: LogBus, keep: int = 100):
        self.bus = bus
        self.jobs: dict[int, Job] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.keep = keep

    def run(self, name: str, argv: list[str], group: str | None = None, on_line=None) -> Job:
        job = Job(next(self._ids), name, argv, group=group)
        with self._lock:
            self.jobs[job.id] = job
            for old in sorted(self.jobs)[:-self.keep]:
                if self.jobs[old].state not in ("queued", "running"):
                    del self.jobs[old]
        threading.Thread(target=self._run, args=(job, on_line), daemon=True, name=f"job:{name}").start()
        return job

    def run_sync(self, name: str, argv: list[str], group: str | None = None, on_line=None) -> Job:
        job = Job(next(self._ids), name, argv, group=group)
        with self._lock:
            self.jobs[job.id] = job
        self._run(job, on_line)
        return job

    def _run(self, job: Job, on_line=None):
        job.state, job.started = "running", time.time()
        self.bus.publish(job.source, f"[job {job.id}] $ {subprocess.list2cmdline(job.argv)}")
        self._emit(job)
        try:
            job.proc = subprocess.Popen(job.argv, cwd=REPO_ROOT, env=child_env(), stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        creationflags=spawn_flags())
        except OSError as e:
            job.state, job.ended = "failed", time.time()
            self.bus.publish(job.source, f"[job {job.id}] spawn failed: {e}", "error")
            self._emit(job)
            return

        def line(text):
            job.lines.append(text)
            self.bus.publish(job.source, text)
            if on_line:
                try:
                    on_line(text)
                except Exception:
                    pass

        split = LineSplitter(line, lambda t: self.bus.set_status_line(job.source, t))
        while True:
            chunk = job.proc.stdout.read1(8192)
            if not chunk:
                break
            split.feed(chunk.decode("utf-8", errors="replace"))
        split.flush()
        job.rc = job.proc.wait()
        job.ended = time.time()
        if job.state != "cancelled":
            job.state = "succeeded" if job.rc == 0 else "failed"
        job.result = trailing_json(job.lines)
        self.bus.publish(job.source, f"[job {job.id}] {job.state} (exit {job.rc}, "
                                     f"{job.ended - job.started:.1f}s)", "info" if job.rc == 0 else "error")
        job.proc = None
        self._emit(job)

    def cancel(self, job_id: int) -> bool:
        job = self.jobs.get(job_id)
        if not job or job.state != "running" or job.proc is None:
            return False
        job.state = "cancelled"
        kill_tree(job.proc.pid)
        return True

    def views(self) -> list[dict]:
        with self._lock:
            return [j.view() for j in sorted(self.jobs.values(), key=lambda j: -j.id)]

    def running(self, group: str | None = None) -> list[Job]:
        return [j for j in self.jobs.values() if j.state == "running" and (group is None or j.group == group)]

    def _emit(self, job: Job):
        self.bus.event({"type": "job", "job": job.view()})


# ─── Console command palette (whitelist — never an arbitrary shell) ─

def palette() -> dict[str, dict]:
    m = lambda mod, *a: [*PY, "-m", mod, *a]
    s = lambda script, *a: [*PY, str(SRC_DIR / script), *a]
    return {
        "preflight":            {"argv": s("preflight.py", "--skip-dry-run"), "help": "start-of-stream checks (no dry reaction)"},
        "preflight-full":       {"argv": s("preflight.py"), "help": "checks + one dry reaction (needs the stack)"},
        "serve-cmd":            {"argv": s("preflight.py", "--serve-cmd"), "help": "print the llama-server command"},
        "pytest":               {"argv": m("pytest", "tests/", "-q", "-p", "no:cacheprovider"), "help": "full test suite"},
        "rows-stats":           {"argv": m("training.training_logger", "stats"), "help": "row counts by status / input_kind"},
        "review-queue":         {"argv": m("training.training_logger", "review-queue", "--n", "20"), "help": "rows Ben should look at first"},
        "prune-dry-run":        {"argv": m("training.training_logger", "prune", "--days", "14"), "help": "frames only rejected rows use (dry run)"},
        "registry-show":        {"argv": m("training.registry", "show"), "help": "adapters/registry.json"},
        "eval-build-set":       {"argv": m("training.eval_gate", "build-set"), "help": "fixed eval set (once)"},
        "memory-stats":         {"argv": s("memory.py", "stats"), "help": "ChromaDB memory stats"},
    }
