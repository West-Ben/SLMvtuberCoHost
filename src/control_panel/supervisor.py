"""
supervisor.py — spawn, watch, restart and stop the long-running services.

States: stopped · starting · healthy · unhealthy · stopping · crashed · external
  external = the service's port was already bound by a process the panel did
  not start (e.g. a llama-server launched by hand); it is monitored, not owned.

Restart policy: an owned service that exits while it should be running is
restarted with backoff 2, 4, 8 … 60 s; more than MAX_RESTARTS within
RESTART_WINDOW_S marks it crashed and stops retrying.
"""

import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field

import psutil

from . import REPO_ROOT, STATE_PATH, child_env
from .logbus import LineSplitter, LogBus
from .services import STACK_ORDER, ServiceSpec, tcp_open

MAX_RESTARTS = 5
RESTART_WINDOW_S = 300
STOP_GRACE_S = 10
HEALTH_EVERY_S = 2.0


class StateStore:
    """control_panel/state.json — per-service flag values and stack membership."""

    def __init__(self, path=STATE_PATH):
        self.path = path
        self._lock = threading.Lock()
        try:
            self.data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self.data = {}

    def get(self, key: str, default=None):
        with self._lock:
            return self.data.get(key, default)

    def set(self, key: str, value):
        with self._lock:
            self.data[key] = value
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)


@dataclass
class Managed:
    spec: ServiceSpec
    state: str = "stopped"
    proc: subprocess.Popen | None = None
    pid: int | None = None
    started_at: float | None = None
    healthy_since: float | None = None
    want_running: bool = False
    restarts: list = field(default_factory=list)
    consecutive: int = 0
    next_restart_at: float | None = None
    last_line: str = ""
    restart_required: bool = False
    error: str = ""
    exit_code: int | None = None


def find_listener(port: int) -> int | None:
    """PID listening on 127.0.0.1/0.0.0.0:port, if any."""
    try:
        for c in psutil.net_connections(kind="tcp"):
            if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port:
                return c.pid
    except (psutil.AccessDenied, OSError):
        pass
    return None


def spawn_flags() -> int:
    """
    Own process group so Ctrl-Break stops just that child. Children share the
    panel's console (Ctrl-Break needs one); only a console-less panel (pythonw)
    hides theirs with CREATE_NO_WINDOW — then stop falls back to terminate.
    """
    import ctypes
    flags = subprocess.CREATE_NEW_PROCESS_GROUP
    if not ctypes.windll.kernel32.GetConsoleWindow():
        flags |= subprocess.CREATE_NO_WINDOW
    return flags


def kill_tree(pid: int, grace_s: float = STOP_GRACE_S, ctrl_break: bool = True):
    """Ctrl-Break (own process group) → terminate → kill, children included."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = [parent] + parent.children(recursive=True)
    if ctrl_break:
        try:
            os.kill(pid, signal.CTRL_BREAK_EVENT)
        except (OSError, AttributeError):
            pass
        _, alive = psutil.wait_procs([parent], timeout=grace_s / 2)
        if not alive:
            procs = [p for p in procs[1:] if p.is_running()]
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=grace_s / 2)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


class Supervisor:
    def __init__(self, specs: dict[str, ServiceSpec], bus: LogBus, store: StateStore):
        self.specs = specs
        self.bus = bus
        self.store = store
        self.svc: dict[str, Managed] = {n: Managed(s) for n, s in specs.items()}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._stack_thread: threading.Thread | None = None
        self._monitor = threading.Thread(target=self._monitor_loop, name="supervisor", daemon=True)

    def start_monitor(self):
        self._monitor.start()

    # ── flags / stack membership ──────────────────────────
    def flags(self, name: str) -> dict:
        return {**self.specs[name].defaults(), **self.store.get(f"flags:{name}", {})}

    def set_flags(self, name: str, values: dict):
        known = {f.name for f in self.specs[name].flags}
        self.store.set(f"flags:{name}", {k: v for k, v in values.items() if k in known})
        m = self.svc[name]
        if m.state in ("starting", "healthy", "unhealthy"):
            m.restart_required = True
        self._emit(name)

    def in_stack(self, name: str) -> bool:
        return self.store.get(f"stack:{name}", self.specs[name].in_stack)

    def set_in_stack(self, name: str, value: bool):
        self.store.set(f"stack:{name}", bool(value))
        self._emit(name)

    def mark_restart_required(self, names):
        for n in names:
            m = self.svc.get(n)
            if m and m.state in ("starting", "healthy", "unhealthy", "external"):
                m.restart_required = True
                self._emit(n)

    # ── single service ────────────────────────────────────
    def start(self, name: str) -> dict:
        with self._lock:
            m = self.svc[name]
            if m.state in ("starting", "healthy", "unhealthy", "stopping"):
                return self.info(name)
            m.want_running, m.consecutive, m.error = True, 0, ""
            m.restarts.clear()
            if m.spec.port and tcp_open(m.spec.port):
                m.state, m.pid, m.proc = "external", find_listener(m.spec.port), None
                m.started_at = time.time()
                self.bus.publish(name, f"[panel] port {m.spec.port} already in use (pid {m.pid}) — "
                                       f"adopting as external, not starting a second copy", "warn")
                self._emit(name)
                return self.info(name)
            self._spawn(m)
        return self.info(name)

    def _spawn(self, m: Managed):
        name = m.spec.name
        try:
            argv = m.spec.argv(self.flags(name))
        except Exception as e:
            m.state, m.error, m.want_running = "crashed", str(e), False
            self.bus.publish(name, f"[panel] cannot start: {e}", "error")
            self._emit(name)
            return
        self.bus.publish(name, f"[panel] start: {subprocess.list2cmdline(argv)}", "info")
        try:
            proc = subprocess.Popen(
                argv, cwd=REPO_ROOT, env=child_env(), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=spawn_flags(),
            )
        except OSError as e:
            m.state, m.error, m.want_running = "crashed", str(e), False
            self.bus.publish(name, f"[panel] spawn failed: {e}", "error")
            self._emit(name)
            return
        m.proc, m.pid, m.state = proc, proc.pid, "starting"
        m.started_at, m.healthy_since, m.exit_code, m.restart_required = time.time(), None, None, False
        threading.Thread(target=self._reader, args=(name, proc), daemon=True, name=f"read:{name}").start()
        self._emit(name)

    def _reader(self, name: str, proc: subprocess.Popen):
        m = self.svc[name]

        def on_line(text):
            e = self.bus.publish(name, text)
            m.last_line = e["text"][:200]

        split = LineSplitter(on_line, lambda t: self.bus.set_status_line(name, t))
        stream = proc.stdout
        while True:
            chunk = stream.read1(8192) if hasattr(stream, "read1") else stream.read(8192)
            if not chunk:
                break
            split.feed(chunk.decode("utf-8", errors="replace"))
        split.flush()

    def stop(self, name: str, force_external: bool = False) -> dict:
        with self._lock:
            m = self.svc[name]
            m.want_running, m.next_restart_at = False, None
            if m.state == "external":
                if not force_external:
                    return {**self.info(name), "needs_confirm": True}
                pid = m.pid or (find_listener(m.spec.port) if m.spec.port else None)
                m.state = "stopping"
                self._emit(name)
            elif m.proc is not None and m.proc.poll() is None:
                pid = m.proc.pid
                m.state = "stopping"
                self._emit(name)
            else:
                m.state = "stopped" if m.state != "crashed" else m.state
                self._emit(name)
                return self.info(name)
        if pid:
            self.bus.publish(name, f"[panel] stopping pid {pid}", "info")
            kill_tree(pid, ctrl_break=m.proc is not None)
        else:
            self.bus.publish(name, f"[panel] no pid found listening on :{m.spec.port} — nothing stopped", "warn")
        with self._lock:
            m.state, m.proc, m.pid, m.healthy_since = "stopped", None, None, None
            m.restart_required = False
        self._emit(name)
        return self.info(name)

    def restart(self, name: str) -> dict:
        self.stop(name, force_external=True)
        return self.start(name)

    # ── stack ─────────────────────────────────────────────
    def start_stack(self, names=None):
        names = [n for n in (names or STACK_ORDER) if self.in_stack(n)]
        if self._stack_thread and self._stack_thread.is_alive():
            return False
        self._stack_thread = threading.Thread(target=self._start_stack, args=(names,), daemon=True)
        self._stack_thread.start()
        return True

    def _start_stack(self, names):
        self.bus.publish("panel", f"[stack] starting {', '.join(names)}")
        for n in names:
            self.start(n)
            m = self.svc[n]
            deadline = time.time() + m.spec.start_timeout_s
            while time.time() < deadline and m.state == "starting":
                time.sleep(0.5)
            if m.state not in ("healthy", "external"):
                self.bus.publish("panel", f"[stack] {n} is {m.state} after "
                                          f"{m.spec.start_timeout_s:.0f}s — continuing", "warn")
        self.bus.publish("panel", "[stack] start sequence done")

    def stop_stack(self, include_external: bool = False):
        for n in reversed(STACK_ORDER):
            m = self.svc[n]
            if m.state == "external" and not include_external:
                continue
            if m.state not in ("stopped", "crashed"):
                self.stop(n, force_external=include_external)

    def shutdown(self):
        self._stop.set()
        for n in reversed(STACK_ORDER):
            m = self.svc[n]
            if m.proc is not None and m.proc.poll() is None:
                self.stop(n)

    # ── monitor ───────────────────────────────────────────
    def _monitor_loop(self):
        while not self._stop.is_set():
            for name in list(self.svc):
                try:
                    self._check(name)
                except Exception as e:   # never let one probe kill the monitor
                    self.bus.publish("panel", f"[monitor] {name}: {e}", "error")
            self._stop.wait(HEALTH_EVERY_S)

    def _check(self, name: str):
        m = self.svc[name]
        now = time.time()
        prev = m.state
        if m.state == "external":
            if m.spec.port and not tcp_open(m.spec.port):
                m.state, m.pid = "stopped", None
                self.bus.publish(name, "[panel] external process went away", "warn")
        elif (proc := m.proc) is not None and m.state in ("starting", "healthy", "unhealthy"):
            rc = proc.poll()
            if rc is not None:
                with self._lock:
                    if m.proc is not proc or m.state == "stopping":
                        return   # stop() owns this exit
                    m.exit_code, m.proc, m.pid, m.healthy_since = rc, None, None, None
                    self.bus.publish(name, f"[panel] exited with code {rc}", "error" if rc else "warn")
                    self._schedule_restart(m, now)
            else:
                ok = m.spec.health() if m.spec.port else True
                if m.proc is not proc or m.state == "stopping":
                    return
                if ok:
                    if m.healthy_since is None:
                        m.healthy_since = now
                    m.state = "healthy"
                    if now - m.healthy_since > 60:
                        m.consecutive = 0
                elif m.state == "healthy":
                    m.state, m.healthy_since = "unhealthy", None
        elif (m.state == "stopped" and not m.want_running and m.proc is None and m.spec.port
              and tcp_open(m.spec.port)):
            # Something the panel didn't start is serving this port (e.g. a hand-started llama-server).
            m.state, m.pid, m.started_at = "external", find_listener(m.spec.port), now
        elif m.state == "stopped" and m.want_running and m.next_restart_at and now >= m.next_restart_at:
            m.next_restart_at = None
            self.bus.publish(name, f"[panel] restarting (attempt {len(m.restarts)})", "warn")
            with self._lock:
                self._spawn(m)
        if m.state != prev:
            self._emit(name)

    def _schedule_restart(self, m: Managed, now: float):
        if not m.want_running:
            m.state = "stopped"
            return
        m.restarts = [t for t in m.restarts if now - t < RESTART_WINDOW_S] + [now]
        if len(m.restarts) > MAX_RESTARTS:
            m.state, m.want_running = "crashed", False
            m.error = f"{len(m.restarts) - 1} restarts in {RESTART_WINDOW_S // 60} min — giving up"
            self.bus.publish(m.spec.name, f"[panel] {m.error}", "error")
            return
        delay = min(2 ** (m.consecutive + 1), 60)
        m.consecutive += 1
        m.state, m.next_restart_at = "stopped", now + delay
        self.bus.publish(m.spec.name, f"[panel] restart in {delay}s", "warn")

    # ── views ─────────────────────────────────────────────
    def info(self, name: str) -> dict:
        m = self.svc[name]
        s = m.spec
        return {
            "name": name, "title": s.title, "port": s.port, "state": m.state, "pid": m.pid,
            "uptime_s": round(time.time() - m.started_at) if m.started_at and m.state not in ("stopped", "crashed") else None,
            "restarts": len(m.restarts), "restart_required": m.restart_required, "error": m.error,
            "exit_code": m.exit_code, "last_line": m.last_line,
            "status_line": self.bus.status_lines.get(name, ""), "depends": list(s.depends), "gpu": s.gpu,
            "in_stack": self.in_stack(name),
            "next_restart_in": round(m.next_restart_at - time.time()) if m.next_restart_at else None,
            "flags": self.flags(name),
            "flag_schema": [{"name": f.name, "type": f.type, "default": f.default, "help": f.help,
                             "choices": list(f.choices)} for f in s.flags],
        }

    def all_info(self) -> list[dict]:
        return [self.info(n) for n in STACK_ORDER if n in self.svc]

    def running(self, names) -> list[str]:
        return [n for n in names if self.svc[n].state in ("starting", "healthy", "unhealthy", "external")]

    def _emit(self, name: str):
        self.bus.event({"type": "service", "service": self.info(name)})
