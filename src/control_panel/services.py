"""
services.py — the long-running processes the panel supervises.

Flag schemas mirror each script's argparse. Chosen values persist in
control_panel/state.json (see supervisor.StateStore).
"""

import json
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests

from . import REPO_ROOT, SRC_DIR, child_env
from .settings import env_file_values


@dataclass
class Flag:
    name: str                 # "--eq"
    type: str                 # bool | int | float | str | choice
    default: object = None
    help: str = ""
    choices: tuple = ()


@dataclass
class ServiceSpec:
    name: str
    title: str
    port: int | None
    depends: tuple = ()
    flags: list[Flag] = field(default_factory=list)
    gpu: bool = False
    in_stack: bool = True
    start_timeout_s: float = 60.0
    script: str | None = None          # python script under SRC_DIR; None → custom argv()

    def defaults(self) -> dict:
        return {f.name: f.default for f in self.flags}

    def render_flags(self, values: dict) -> list[str]:
        out: list[str] = []
        for f in self.flags:
            v = values.get(f.name, f.default)
            if f.type == "bool":
                if v:
                    out.append(f.name)
            elif v not in (None, ""):
                out += [f.name, str(v)]
        return out

    def argv(self, values: dict) -> list[str]:
        return [sys.executable, "-u", str(SRC_DIR / self.script), *self.render_flags(values)]

    def health(self) -> bool:
        return tcp_open(self.port) if self.port else False


def tcp_open(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def http_ok(url: str, timeout: float = 1.5) -> bool:
    try:
        return requests.get(url, timeout=timeout).status_code < 500
    except requests.RequestException:
        return False


def llm_port() -> int:
    url = env_file_values().get("LLM_URL") or "http://localhost:8080/v1/chat/completions"
    return urlparse(url).port or 8080


# ─── llama-server ─────────────────────────────────────────

class LlamaSpec(ServiceSpec):
    """argv comes from preflight.serve_argv() run in a fresh interpreter, so it sees the current .env + registry."""

    def argv(self, values: dict, port: int | None = None) -> list[str]:
        code = ("import json, preflight; "
                f"print('ARGV=' + json.dumps(preflight.serve_argv({port!r})))")
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=child_env(),
                             capture_output=True, text=True, encoding="utf-8", timeout=120)
        line = next((l for l in out.stdout.splitlines() if l.startswith("ARGV=")), None)
        if line is None:
            raise RuntimeError(f"preflight.serve_argv failed: {(out.stderr or out.stdout).strip()[-400:]}")
        argv = json.loads(line[5:])
        if "<BASE_CORE_GGUF>" in argv:
            raise RuntimeError("BASE_CORE_GGUF is not set — set it on the Settings page")
        ctx = values.get("-c")
        if ctx:
            i = argv.index("-c")
            argv[i + 1] = str(ctx)
        extra = (values.get("extra") or "").split()
        return argv + extra

    def health(self) -> bool:
        return http_ok(f"http://127.0.0.1:{self.port}/health")


class HttpSpec(ServiceSpec):
    def __init__(self, *a, path: str = "/", **kw):
        super().__init__(*a, **kw)
        self.path = path

    def health(self) -> bool:
        return http_ok(f"http://127.0.0.1:{self.port}{self.path}")


class AliveSpec(ServiceSpec):
    """No port: healthy == process alive (decided by the supervisor)."""

    def health(self) -> bool:
        return True


def build_services() -> dict[str, ServiceSpec]:
    specs = [
        LlamaSpec("llama", "llama-server", llm_port(), gpu=True, start_timeout_s=240, flags=[
            Flag("-c", "int", 16384, "context size (overrides serve_cmd's -c)"),
            Flag("extra", "str", "", "extra llama-server args, space separated"),
        ]),
        HttpSpec("tts", "TTS + avatar + memory (server.py)", 8051, script="server.py", path="/memory/stats",
                 depends=("llama",), gpu=True, start_timeout_s=240, flags=[
            Flag("--offload-tts", "str", "", "send TTS to an external endpoint instead of loading Chatterbox"),
        ]),
        HttpSpec("bridge", "OBS bridge (obs_bridge.py)", 8055, script="obs_bridge.py", path="/status", flags=[
            Flag("--test", "bool", False, "force stream.active=true (no OBS stream needed)"),
            Flag("--imagelog-frames", "int", None, "frames saved per imagelog group (default 3)"),
            Flag("--imagelog-interval", "float", None, "seconds between imagelog groups (default 10)"),
            Flag("--perceive-interval", "float", None, "legacy perceiver interval"),
            Flag("--frames-min", "int", None, "min frames per window"),
            Flag("--frames-max", "int", None, "max frames per window"),
            Flag("--loopback-device", "str", "", "WASAPI output device keyword for loopback capture"),
            Flag("--no-loopback", "bool", False, "disable speaker loopback capture"),
        ]),
        HttpSpec("orchestrator", "Orchestrator (orchestrator.py)", 8052, script="orchestrator.py", path="/status",
                 depends=("bridge", "tts", "llama"), flags=[
            Flag("--mode", "choice", "frames", "reaction path", ("frames", "text")),
            Flag("--direct", "bool", False, "use the HF endpoint instead of local llama.cpp"),
            Flag("--no-log", "bool", False, "don't write training rows"),
        ]),
        AliveSpec("twitch", "Twitch listener (twitch_listener.py)", None, script="twitch_listener.py",
                  depends=("bridge",)),
        ServiceSpec("stt", "STT (stt-server.py)", 8053, script="stt-server.py", depends=("orchestrator", "llama"),
                    start_timeout_s=120, flags=[
            Flag("--eq", "bool", True, "voice EQ"),
            Flag("--eq-gain", "float", None, "EQ gain dB"),
            Flag("--gain", "float", 6.0, "input gain dB"),
            Flag("--direct", "bool", False, "send raw audio to the orchestrator for HF transcription"),
            Flag("--offload", "str", "", "external STT endpoint URL"),
            Flag("--respond", "bool", False, "send transcriptions to /chat-input (immediate reply)"),
            Flag("--no-loopback", "bool", False, ""),
            Flag("--loopback-device", "str", "", ""),
        ]),
    ]
    return {s.name: s for s in specs}


STACK_ORDER = ("llama", "tts", "bridge", "orchestrator", "twitch", "stt")
