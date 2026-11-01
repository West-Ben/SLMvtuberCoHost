"""
settings.py — .env editing for the Settings page.

The key list and defaults come from config.py itself (every os.getenv("KEY",
"default") call, found with ast), so there is no second copy of the defaults.
This module only adds grouping, help text, which services need a restart,
and which keys are secrets. Secrets are write-only: never returned in full.
"""

import ast
import re
from pathlib import Path

from dotenv import dotenv_values, set_key, unset_key

from . import ENV_PATH, REPO_ROOT

CONFIG_PATH = REPO_ROOT / "config.py"

# Read by scripts directly (not via config.py) but still set through .env.
EXTRA_KEYS = {
    "EVAL_LLM_URL": "http://localhost:8081",       # training/eval_gate.py
    "ANTHROPIC_API_KEY": "",                       # reviewer Stage B, eval judge
}

_SECRET_RE = re.compile(r"_TOKEN$|SECRET|PASSWORD|API_KEY")

# key prefix / exact key → (group, services that must restart)
_GROUPS: list[tuple[str, str, tuple]] = [
    ("LLM_", "Model & transport", ("orchestrator", "stt", "llama")),
    ("HF_", "Model & transport", ("orchestrator", "stt")),
    ("REACTION_", "Model & transport", ("orchestrator",)),
    ("PERCEIVER_ENABLED", "Vision", ("bridge", "orchestrator")),
    ("VISION_", "Vision", ("bridge", "orchestrator")),
    ("BASE_CORE_", "Model files & training", ("llama",)),
    ("LLAMACPP_DIR", "Model files & training", ("llama",)),
    ("EVAL_LLM_URL", "Model files & training", ()),
    ("ANTHROPIC_API_KEY", "Model files & training", ()),
    ("WINDOW_", "Window policy", ("orchestrator",)),
    ("VAD_STALE_S", "Window policy", ("orchestrator",)),
    ("LEAD_S", "Playback & TTS", ("orchestrator",)),
    ("EXPR_RESET", "Playback & TTS", ("orchestrator",)),
    ("EXPR_RESET_DELAY_S", "Playback & TTS", ("orchestrator",)),
    ("TTS_", "Playback & TTS", ("tts",)),
    ("VOICE_REF_PATH", "Playback & TTS", ("tts",)),
    ("WARUDO_WS_URL", "Playback & TTS", ("tts",)),
    ("CABLE_DEVICE_NAME", "Audio devices", ("bridge", "tts")),
    ("SPEAKER_DEVICE_NAME", "Audio devices", ("bridge", "tts", "stt")),
    ("MIC_KEYWORD", "Audio devices", ("stt",)),
    ("STT_", "Audio devices", ("stt",)),
    ("AUDIO_CONTEXT_SECONDS", "Audio devices", ("stt",)),
    ("BEN_BUFFER_WINDOW_S", "Audio devices", ("orchestrator",)),
    ("OBS_", "OBS", ("bridge",)),
    ("TWITCH_", "Twitch", ("twitch",)),
    ("EMBED_", "Memory", ("tts",)),
    ("MEMORY_DB_PATH", "Memory", ("tts",)),
    ("TRAINING_DATA_DIR", "Legacy text path", ("orchestrator",)),
    ("SESSION_", "Legacy text path", ("orchestrator",)),
]

GROUP_ORDER = ["Model & transport", "Model files & training", "Audio devices", "Playback & TTS",
               "Window policy", "Vision", "OBS", "Twitch", "Memory", "Legacy text path", "Other"]


def env_file_values(path: Path = ENV_PATH) -> dict[str, str]:
    """Current .env contents, read fresh every call (never via os.environ)."""
    return {k: v for k, v in dotenv_values(path).items() if v is not None} if path.exists() else {}


def config_defaults(path: Path = CONFIG_PATH) -> dict[str, dict]:
    """{KEY: {"default": str, "type": "str|int|float|bool", "comment": str}} from os.getenv calls in config.py."""
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    out: dict[str, dict] = {}
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "getenv" and node.args and isinstance(node.args[0], ast.Constant)):
            continue
        key = node.args[0].value
        default = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else ""
        line = lines[node.lineno - 1]
        kind = "str"
        if re.search(rf"int\(os\.getenv\(\"{key}\"", line):
            kind = "int"
        elif re.search(rf"float\(os\.getenv\(\"{key}\"", line):
            kind = "float"
        elif ".lower() in" in line:
            kind = "bool"
        comment = line.split("#", 1)[1].strip() if "#" in line else ""
        out.setdefault(key, {"default": str(default), "type": kind, "comment": comment})
    for key, default in EXTRA_KEYS.items():
        out.setdefault(key, {"default": default, "type": "str", "comment": ""})
    return out


def classify(key: str) -> tuple[str, tuple]:
    for prefix, group, restart in _GROUPS:
        if key == prefix or (prefix.endswith("_") and key.startswith(prefix)):
            return group, restart
    return "Other", ()


def is_secret(key: str) -> bool:
    return bool(_SECRET_RE.search(key))


def schema() -> list[dict]:
    """Every setting, with its current .env value (secrets masked)."""
    env = env_file_values()
    rows = []
    keys = dict(config_defaults())
    for k in env:   # keys only in .env (e.g. OPENCLAW_*) still show up, under Other
        keys.setdefault(k, {"default": "", "type": "str", "comment": "not read by config.py"})
    for key, meta in keys.items():
        group, restart = classify(key)
        secret = is_secret(key)
        val = env.get(key)
        rows.append({
            "key": key, "group": group, "type": meta["type"], "default": meta["default"],
            "comment": meta["comment"], "restart": list(restart), "secret": secret,
            "set": val is not None and val != "",
            "value": None if secret else val,
        })
    rows.sort(key=lambda r: (GROUP_ORDER.index(r["group"]), r["key"]))
    return rows


def validate(key: str, value: str, kind: str) -> str | None:
    if value == "":
        return None
    try:
        if kind == "int":
            int(value)
        elif kind == "float":
            float(value)
        elif kind == "bool" and value.lower() not in ("true", "false", "1", "0", "yes", "no"):
            return "expected true/false"
    except ValueError:
        return f"expected {kind}"
    if "\n" in value:
        return "newlines not allowed"
    return None


def update(changes: dict[str, str | None], path: Path = ENV_PATH) -> dict:
    """
    changes: {KEY: value} — "" or None removes the key (config default applies).
    Returns {"changed": [...], "restart": [services], "errors": {KEY: msg}}.
    """
    known = config_defaults()
    errors, changed, restart = {}, [], set()
    current = env_file_values(path)
    for key, value in changes.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            errors[key] = "bad key"
            continue
        value = "" if value is None else str(value).strip()
        err = validate(key, value, known.get(key, {}).get("type", "str"))
        if err:
            errors[key] = err
            continue
        if current.get(key, "") == value:
            continue
        if value == "":
            if key in current:
                unset_key(str(path), key)
        else:
            path.touch(exist_ok=True)
            set_key(str(path), key, value, quote_mode="never")
        changed.append(key)
        restart.update(classify(key)[1])
    return {"changed": changed, "restart": sorted(restart), "errors": errors}


# ─── Audio devices (Settings → mic picker) ────────────────

def audio_devices(keyword: str | None = None) -> dict:
    import pyaudio
    from audio_devices import list_input_devices, rank_mic_candidates
    from config import DEVICE_RATE
    p = pyaudio.PyAudio()
    try:
        inputs = list_input_devices(p)
        outputs = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get("maxOutputChannels", 0) > 0:
                outputs.append({"index": i, "name": info.get("name", ""),
                                "channels": info["maxOutputChannels"], "rate": int(info["defaultSampleRate"]),
                                "host_api": p.get_host_api_info_by_index(info["hostApi"])["name"]})
    finally:
        p.terminate()
    keyword = keyword if keyword is not None else env_file_values().get("MIC_KEYWORD", "")
    ranked = rank_mic_candidates(inputs, keyword, DEVICE_RATE) if keyword else []
    return {"inputs": inputs, "outputs": outputs, "keyword": keyword,
            "ranked": [{"index": i, "rate": r, "channels": c, "name": n} for i, r, c, n in ranked]}


def level_test(index: int, seconds: float = 3.0) -> dict:
    """Record from one input device at its native rate/channels; RMS + peak per channel."""
    import numpy as np
    import pyaudio
    p = pyaudio.PyAudio()
    try:
        info = p.get_device_info_by_index(index)
        ch, rate = int(info["maxInputChannels"]), int(info["defaultSampleRate"])
        chunk = 1024
        s = p.open(format=pyaudio.paInt16, channels=ch, rate=rate, input=True,
                   input_device_index=index, frames_per_buffer=chunk)
        try:
            n = max(1, int(seconds * rate / chunk))
            raw = b"".join(s.read(chunk, exception_on_overflow=False) for _ in range(n))
        finally:
            s.close()
    finally:
        p.terminate()
    d = np.frombuffer(raw, dtype=np.int16).reshape(-1, ch).astype(np.float32) / 32768
    rms = np.sqrt((d ** 2).mean(axis=0))
    peak = np.abs(d).max(axis=0)
    db = lambda x: round(float(20 * np.log10(max(x, 1e-6))), 1)
    return {"index": index, "name": info.get("name"), "channels": ch, "rate": rate,
            "rms_dbfs": [db(x) for x in rms], "peak_dbfs": [db(x) for x in peak],
            "mono_mix_dbfs": db(float(np.sqrt((d.mean(axis=1) ** 2).mean())))}


# ─── Persona (read-only) ──────────────────────────────────

def persona_files() -> list[dict]:
    d = REPO_ROOT / "persona"
    return [{"name": p.name, "path": str(p), "text": p.read_text(encoding="utf-8", errors="replace")}
            for p in sorted(d.glob("*.md"))]
