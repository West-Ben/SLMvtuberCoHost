"""
dream.py — the nightly training loop as panel jobs ("Dream" page).

  review → dataset → train → eval → (registry: promote / rollback by hand)

Each step is one JobRunner job running the existing CLI unchanged. "Run all"
chains the selected steps and stops at the first failure. The eval step
starts a temporary llama-server on :8081 (serve_argv(port=8081) — includes
every registered candidate GGUF) and always stops it afterwards.
"""

import os
import re
import shutil
import subprocess
import threading
import time
from datetime import date, timedelta

import requests

from . import BASE_ENV, REPO_ROOT, child_env
from .jobs import PY, JobRunner
from .logbus import LogBus
from .settings import env_file_values
from .supervisor import Supervisor, kill_tree, spawn_flags

STEPS = ("review", "dataset", "train", "eval")
GPU_SERVICES = ("llama", "tts")
EVAL_PORT = 8081
_LOSS_RE = re.compile(r"'loss':\s*([0-9.]+)")
_EPOCH_RE = re.compile(r"'epoch':\s*([0-9.]+)")


def step_argv(step: str, p: dict) -> list[str]:
    m = lambda mod, *a: [*PY, "-m", mod, *a]
    if step == "review":
        argv = m("training.reviewer")
        for d in p.get("dates") or []:
            argv += ["--date", d]
        argv += ["--backend", p.get("backend") or "api"]
        if p.get("limit"):
            argv += ["--limit", str(int(p["limit"]))]
        for flag in ("stage_a_only", "force", "dry_run"):
            if p.get(flag):
                argv.append("--" + flag.replace("_", "-"))
        return argv
    if step == "dataset":
        return m("training.dataset_builder", "--k", str(int(p.get("k", 3))), "--seed", str(int(p.get("seed", 0))),
                 "--run-id", p["run_id"])
    if step == "train":
        argv = m("training.train_nightly", "--run-id", p["run_id"], "--version", p["version"],
                 "--lr", str(p.get("lr", 3e-5)), "--epochs", str(p.get("epochs", 1)),
                 "--grad-accum", str(int(p.get("grad_accum", 8))), "--r", str(int(p.get("r", 16))),
                 "--trainer", p.get("trainer") or "sft")
        if p.get("check"):
            argv.append("--check")
        if p.get("export_gguf", True):
            argv.append("--export-gguf")
        return argv
    if step == "eval":
        argv = m("training.eval_gate", "run", p["version"])
        if p.get("no_judge"):
            argv.append("--no-judge")
        if p.get("dry_run"):
            argv.append("--dry-run")
        return argv
    raise ValueError(f"unknown step {step}")


class Dream:
    def __init__(self, sup: Supervisor, jobs: JobRunner, bus: LogBus):
        self.sup, self.jobs, self.bus = sup, jobs, bus
        self.chain: dict | None = None
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()

    # ── preconditions / defaults ──────────────────────────
    def state(self) -> dict:
        from training import registry
        from training import training_logger as tl
        from training.dataset_builder import CURATED_DIR, EVAL_SET_PATH, curated_nights, pick_nights
        reg = registry.load()
        today = date.today()
        version = registry.next_version(reg, today.strftime("%Y%m%d"))
        row_dates = [p.stem for p in tl.row_files()]
        want = [(today - timedelta(days=1)).isoformat(), today.isoformat()]
        nights = curated_nights(CURATED_DIR)
        new, baked = pick_nights(nights, reg, 3) if nights else ([], [])
        env = env_file_values()
        base_hf = env.get("BASE_CORE_HF") or r"F:\LLM\models\gemma-4-e4b"
        return {
            "defaults": {"version": version, "run_id": version,
                         "dates": [d for d in want if d in row_dates] or row_dates[-2:]},
            "row_dates": row_dates,
            "curated_nights": nights, "next_nights": new, "replay_nights": baked,
            "eval_set": EVAL_SET_PATH.exists(),
            "registry": reg,
            "gpu_busy": self.sup.running(GPU_SERVICES),
            "anthropic_key": bool(env.get("ANTHROPIC_API_KEY") or BASE_ENV.get("ANTHROPIC_API_KEY")),
            "claude_cli": bool(shutil.which("claude")),
            "base_core_hf": base_hf, "base_core_hf_exists": os.path.exists(base_hf),
            "base_core_gguf": env.get("BASE_CORE_GGUF", ""),
            "chain": self.chain,
        }

    # ── running ───────────────────────────────────────────
    def start(self, steps: list[str], params: dict, stop_live: bool = False) -> dict:
        if self._thread and self._thread.is_alive():
            return {"error": "a dream run is already in progress"}
        steps = [s for s in STEPS if s in steps]
        if not steps:
            return {"error": "no steps selected"}
        need_gpu = ("train" in steps and not params.get("train", {}).get("check")) or "eval" in steps
        busy = self.sup.running(GPU_SERVICES)
        if "train" in steps and busy and not stop_live:
            return {"error": f"GPU in use by {', '.join(busy)} — stop the live stack first",
                    "needs_stop_live": True}
        self._cancel.clear()
        self.chain = {"id": int(time.time()), "steps": [{"step": s, "state": "pending", "job": None,
                                                         "result": None} for s in steps],
                      "params": params, "loss": [], "state": "running", "started": time.time()}
        self._thread = threading.Thread(target=self._run, args=(steps, params, stop_live and need_gpu), daemon=True)
        self._thread.start()
        return {"ok": True, "chain": self.chain}

    def cancel(self):
        self._cancel.set()
        for j in self.jobs.running(group="dream"):
            self.jobs.cancel(j.id)

    def _emit(self):
        self.bus.event({"type": "dream", "chain": self.chain})

    def _run(self, steps, params, stop_live):
        if stop_live:
            self.bus.publish("job:dream", "[dream] stopping live stack to free the GPU", "warn")
            self.sup.stop_stack(include_external=True)
        for entry in self.chain["steps"]:
            if self._cancel.is_set():
                entry["state"] = "cancelled"
                continue
            entry["state"] = "running"
            self._emit()
            ok = self._run_step(entry, params.get(entry["step"], {}))
            entry["state"] = "succeeded" if ok else ("cancelled" if self._cancel.is_set() else "failed")
            self._emit()
            if not ok:
                for rest in self.chain["steps"]:
                    if rest["state"] == "pending":
                        rest["state"] = "skipped"
                break
        self.chain["state"] = ("succeeded" if all(s["state"] == "succeeded" for s in self.chain["steps"])
                               else "failed")
        self.chain["ended"] = time.time()
        self._emit()

    def _run_step(self, entry: dict, p: dict) -> bool:
        step = entry["step"]
        if step == "eval":
            return self._run_eval(entry, p)
        on_line = self._loss_tap if step == "train" else None
        job = self.jobs.run_sync(f"dream-{step}", step_argv(step, p), group="dream", on_line=on_line)
        entry["job"], entry["result"] = job.id, job.result
        return job.state == "succeeded"

    def _loss_tap(self, line: str):
        m = _LOSS_RE.search(line)
        if m:
            e = _EPOCH_RE.search(line)
            self.chain["loss"].append({"loss": float(m.group(1)), "epoch": float(e.group(1)) if e else None})
            self._emit()

    def _run_eval(self, entry: dict, p: dict) -> bool:
        from training.dataset_builder import EVAL_SET_PATH
        if not EVAL_SET_PATH.exists():
            job = self.jobs.run_sync("dream-eval-build-set", [*PY, "-m", "training.eval_gate", "build-set"],
                                     group="dream")
            if job.state != "succeeded":
                entry["job"] = job.id
                return False
        llama = self.sup.specs["llama"]
        try:
            argv = llama.argv(self.sup.flags("llama"), port=EVAL_PORT)
        except Exception as e:
            self.bus.publish("job:dream-eval", f"[dream] cannot build eval llama-server command: {e}", "error")
            return False
        src = "job:dream-eval-llama"
        self.bus.publish(src, f"[dream] $ {subprocess.list2cmdline(argv)}")
        server = subprocess.Popen(argv, cwd=REPO_ROOT, env=child_env(), stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=spawn_flags())
        threading.Thread(target=self._pump, args=(server, src), daemon=True).start()
        try:
            deadline = time.time() + 300
            while time.time() < deadline and server.poll() is None and not self._cancel.is_set():
                try:
                    if requests.get(f"http://127.0.0.1:{EVAL_PORT}/health", timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                self.bus.publish(src, "[dream] eval llama-server did not become healthy", "error")
                return False
            job = self.jobs.run_sync("dream-eval", step_argv("eval", p), group="dream")
            entry["job"], entry["result"] = job.id, job.result
            return job.state == "succeeded"
        finally:
            self.bus.publish(src, "[dream] stopping eval llama-server")
            kill_tree(server.pid)

    def _pump(self, proc, src):
        for raw in iter(proc.stdout.readline, b""):
            self.bus.publish(src, raw.decode("utf-8", errors="replace"))
