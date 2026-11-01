"""
preflight.py — start-of-stream checklist. Exit code 0 = PASS.

  1. versions   prompt_version, grammar hash, current adapter (registry) and
                whether it was trained under the same prompt/grammar
  2. parser     runs tests/test_tags.py, test_tag_parser.py, test_prompt_format.py
  3. serving    llama-server reachable; current adapter's GGUF actually loaded
                (llama.cpp: --lora at startup; runtime scale swap via
                per-request "lora" / POST /lora-adapters; a NEW adapter file
                needs a restart — --serve-cmd prints the command)
  4. services   obs_bridge /window, TTS server (warn only); each vocal tag
                synthesised alone must give a non-empty clip
                (tests/test_tts_smoke.py runs the same check under pytest)
  5. dry run    one real reaction: latest frames → model → parser; printed,
                not spoken, not logged

Run:
  python preflight.py [--direct] [--skip-dry-run] [--serve-cmd]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import requests

import orchestrator as orch
import prompt_format
import tags
from config import (
    ADAPTER_REGISTRY, BASE_CORE_GGUF, BASE_CORE_MMPROJ, LLAMACPP_DIR, LLM_URL,
    LEAD_S, OBS_BRIDGE_URL, REACTION_MODE, REPO_ROOT, SPEAK_URL,
)

PARSER_TESTS = ["tests/test_tags.py", "tests/test_tag_parser.py", "tests/test_playback.py",
                "tests/test_prompt_format.py"]


class Report:
    def __init__(self):
        self.items: list[tuple[str, str, str]] = []

    def add(self, level: str, name: str, detail: str = ""):
        self.items.append((level, name, detail))
        print(f"[{level}] {name}" + (f" - {detail}" if detail else ""))

    @property
    def ok(self) -> bool:
        return not any(level == "FAIL" for level, _, _ in self.items)


def serve_argv(port: int | str | None = None) -> list[str]:
    """llama-server argv: base_core + every registered adapter GGUF, current applied by request scale."""
    import json
    reg = json.loads(ADAPTER_REGISTRY.read_text(encoding="utf-8"))
    ggufs = [a["gguf"] for a in reg["adapters"] if a.get("gguf") and a.get("status") in ("promoted", "candidate")]
    exe = LLAMACPP_DIR / "build" / "bin" / "Release" / "llama-server.exe"
    if port is None:
        port = LLM_URL.split(":")[-1].split("/")[0] if LLM_URL.count(":") >= 2 else "8080"
    argv = [str(exe), "-m", BASE_CORE_GGUF or "<BASE_CORE_GGUF>", "--mmproj", BASE_CORE_MMPROJ]
    for g in ggufs:
        argv += ["--lora", str(REPO_ROOT / g)]
    if ggufs:
        argv.append("--lora-init-without-apply")
    return argv + ["--port", str(port), "-ngl", "99", "-c", "16384"]


def serve_cmd(port: int | str | None = None) -> str:
    return subprocess.list2cmdline(serve_argv(port))


def check_versions(r: Report):
    system = prompt_format.load_system_prompt()
    pv, gh = prompt_format.prompt_version(system), tags.grammar_hash()
    r.add("PASS", "prompt_version", pv)
    r.add("PASS", "grammar_hash", gh)
    reg = orch._read_registry()
    cur = reg.get("current")
    entry = next((a for a in reg.get("adapters", []) if a.get("version") == cur), None)
    if cur is None:
        r.add("WARN", "adapter", "no current adapter — serving base_core")
        return
    if entry is None:
        r.add("FAIL", "adapter", f"registry current={cur} has no entry")
        return
    r.add("PASS", "adapter", f"{cur} (status={entry.get('status')})")
    if entry.get("grammar_hash") and entry["grammar_hash"] != gh:
        r.add("FAIL", "adapter grammar", f"trained with grammar {entry['grammar_hash']}, runtime is {gh}")
    if entry.get("prompt_version") and entry["prompt_version"] != pv:
        r.add("WARN", "adapter prompt", f"trained on {entry['prompt_version']}, runtime is {pv}")
    if entry.get("gguf") and not (REPO_ROOT / entry["gguf"]).exists():
        r.add("FAIL", "adapter gguf", f"missing {entry['gguf']}")


def check_parser_tests(r: Report):
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", *PARSER_TESTS],
                         cwd=REPO_ROOT, capture_output=True, text=True)
    tail = (out.stdout.strip().splitlines() or ["(no output)"])[-1]
    r.add("PASS" if out.returncode == 0 else "FAIL", "parser tests", tail)


def check_serving(r: Report):
    base, _ = orch._model_base_url()
    if orch._transport() != "completion":
        r.add("WARN", "serving", f"transport={orch._transport()} direct={orch._direct_mode} — "
                                 "exact-prompt path not in use; adapter not checked")
        return
    try:
        requests.get(f"{base}/health", timeout=3).raise_for_status()
        r.add("PASS", "llama-server", base)
    except Exception as e:
        r.add("FAIL", "llama-server", f"{base} unreachable ({e.__class__.__name__}) — start it with --serve-cmd")
        return
    state = orch._resolve_lora()
    cur = orch._current_adapter_version()
    if cur and state["applied"] != cur:
        r.add("FAIL", "adapter loaded", f"{cur} not loaded in llama-server — restart with --serve-cmd")
    else:
        r.add("PASS", "adapter loaded", state["applied"] or "base_core")


def check_services(r: Report):
    for name, url in (("obs_bridge /window", f"{OBS_BRIDGE_URL}/window?seconds=5&min_frames=1&max_frames=2"),
                      ("TTS server", SPEAK_URL.rsplit("/", 1)[0] + "/memory/stats")):
        try:
            resp = requests.get(url, timeout=5)
            r.add("PASS" if resp.status_code < 500 else "WARN", name, f"HTTP {resp.status_code}")
        except Exception as e:
            r.add("WARN", name, f"unreachable ({e.__class__.__name__})")


def check_vocal_clips(r: Report):
    """Decision D: each vocal tag synthesised alone must give a non-empty clip (Chatterbox regression guard)."""
    base = SPEAK_URL.rsplit("/", 1)[0]
    bad = []
    for tag in sorted(tags.VOCAL_TAGS):
        try:
            d = requests.post(f"{base}/synth", json={"text": f"[{tag}]"}, timeout=60).json()
            if not d.get("samples"):
                bad.append(tag)
        except Exception:
            r.add("WARN", "vocal clips", "server.py /synth unreachable — skipped")
            return
    r.add("FAIL" if bad else "PASS", "vocal clips", f"empty: {bad}" if bad else f"{len(tags.VOCAL_TAGS)} tags OK")


def dry_reaction(r: Report):
    eng = orch._make_engine(row_logger=None, player=orch.PrintPlayer())
    eng.on_speech("preflight check — say something about this game")
    rec = eng.react(eng.acc.flush("preflight"))
    if rec is None:
        r.add("FAIL", "dry reaction", "model call failed")
        return
    bad = [f for f in rec["parse_flags"] if f.startswith(("leaked_", "parser_error"))]
    level = "FAIL" if "empty" in rec["parse_flags"] or bad else ("WARN" if not rec["frame_paths"] else "PASS")
    r.add(level, "dry reaction", f"{len(rec['frame_paths'])} frames, {rec['latency_ms']} ms, "
                                 f"flags={rec['parse_flags']}, adapter={rec['adapter_version']}: {rec['raw_output'][:80]!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Start-of-stream preflight")
    ap.add_argument("--direct", action="store_true", help="check the HF endpoint path instead of local llama.cpp")
    ap.add_argument("--skip-dry-run", action="store_true")
    ap.add_argument("--serve-cmd", action="store_true", help="print the llama-server command and exit")
    a = ap.parse_args()
    if a.serve_cmd:
        print(serve_cmd())
        return 0
    orch._direct_mode = a.direct
    print(f"Rachel preflight — mode={REACTION_MODE} transport={orch._transport()} lead={LEAD_S}s\n")
    r = Report()
    check_versions(r)
    check_parser_tests(r)
    check_serving(r)
    check_services(r)
    check_vocal_clips(r)
    if not a.skip_dry_run:
        dry_reaction(r)
    print(f"\n{'PASS' if r.ok else 'FAIL'}")
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())
