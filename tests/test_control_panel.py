"""Control panel: log splitting, job output parsing, .env editing, image guard, flags, supervisor."""

import sys
import time

import psutil
import pytest
from fastapi import HTTPException

from control_panel import data_api, settings
from control_panel.dream import step_argv
from control_panel.jobs import trailing_json
from control_panel.logbus import LineSplitter, LogBus
from control_panel.services import Flag, ServiceSpec
from control_panel import supervisor as supmod
from control_panel.supervisor import StateStore, Supervisor


# ─── log splitting ────────────────────────────────────────

def test_carriage_return_meter_becomes_status_line_not_log():
    lines, status = [], []
    sp = LineSplitter(lines.append, status.append)
    sp.feed("boot\n\r\x1b[K level ▁▂ \r\x1b[K level ▃▄")
    sp.feed(" \r\x1b[K[🎤 Ben]: hello there\n")
    assert lines == ["boot", "\x1b[K[🎤 Ben]: hello there"]
    assert status and "level" in status[-1]


def test_split_line_across_chunks():
    lines = []
    sp = LineSplitter(lines.append, lambda t: None)
    sp.feed("hel")
    sp.feed("lo\nwor")
    sp.flush()
    assert lines == ["hello", "wor"]


def test_logbus_levels_and_backfill(tmp_path):
    bus = LogBus(tmp_path)
    bus.publish("orch", "2026 [orch] ERROR boom")
    bus.publish("orch", "\x1b[32mok\x1b[0m")
    back = bus.backfill("orch")
    assert [e["level"] for e in back] == ["error", "info"]
    assert back[1]["text"] == "ok"
    assert bus.log_file("orch").read_text(encoding="utf-8").count("\n") == 2


# ─── job output ───────────────────────────────────────────

def test_trailing_json_multiline_block():
    out = ["loading…", "{", '  "rows": 3,', '  "curated": 2', "}"]
    assert trailing_json(out) == {"rows": 3, "curated": 2}
    assert trailing_json(["no json here"]) is None


# ─── settings / .env ──────────────────────────────────────

def test_env_update_keeps_comments_and_validates(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# my comment\nMIC_KEYWORD=Anker\nOBS_PASSWORD=hunter2\n", encoding="utf-8")
    res = settings.update({"MIC_KEYWORD": "Analogue 1 + 2", "WINDOW_S": "abc", "LEAD_S": "1.5"}, path=env)
    assert res["errors"] == {"WINDOW_S": "expected float"}
    assert set(res["changed"]) == {"MIC_KEYWORD", "LEAD_S"}
    assert "stt" in res["restart"] and "orchestrator" in res["restart"]
    text = env.read_text(encoding="utf-8")
    assert text.startswith("# my comment\n") and "MIC_KEYWORD=Analogue 1 + 2" in text and "LEAD_S=1.5" in text
    settings.update({"LEAD_S": ""}, path=env)
    assert "LEAD_S" not in env.read_text(encoding="utf-8")


def test_schema_never_returns_secret_values(monkeypatch):
    monkeypatch.setattr(settings, "env_file_values", lambda path=None: {"OBS_PASSWORD": "hunter2", "TWITCH_TOKEN": "x"})
    rows = {r["key"]: r for r in settings.schema()}
    assert rows["OBS_PASSWORD"]["secret"] and rows["OBS_PASSWORD"]["set"] and rows["OBS_PASSWORD"]["value"] is None
    assert not rows["REACTION_MAX_TOKENS"]["secret"]
    assert rows["OBS_PORT"]["type"] == "int" and rows["PERCEIVER_ENABLED"]["type"] == "bool"


# ─── image path guard ─────────────────────────────────────

@pytest.mark.parametrize("p,base", [("../config.py", "imagelog"), ("imagelog/../config.py", "repo"),
                                    ("C:/Windows/win.ini", "repo"), ("persona/SOUL.md", "repo")])
def test_image_guard_refuses_outside_roots(p, base):
    with pytest.raises(HTTPException) as e:
        data_api.safe_image_path(p, base)
    assert e.value.status_code in (403, 404)


def test_image_guard_allows_imagelog(tmp_path, monkeypatch):
    img = tmp_path / "imagelog" / "2026-06-06" / "w-1" / "f01.jpg"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"\xff\xd8")
    monkeypatch.setattr(data_api, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(data_api, "IMAGELOG_DIR", tmp_path / "imagelog")
    monkeypatch.setattr(data_api, "ALLOWED_IMAGE_ROOTS", [tmp_path / "imagelog"])
    assert data_api.safe_image_path("imagelog/2026-06-06/w-1/f01.jpg", "repo") == img.resolve()


# ─── flags / dream argv ───────────────────────────────────

def test_render_flags():
    spec = ServiceSpec("x", "x", None, flags=[Flag("--eq", "bool", True), Flag("--gain", "float", 6.0),
                                              Flag("--offload", "str", ""), Flag("--mode", "choice", "frames")])
    assert spec.render_flags({}) == ["--eq", "--gain", "6.0", "--mode", "frames"]
    assert spec.render_flags({"--eq": False, "--gain": None, "--offload": "http://x"}) == \
        ["--offload", "http://x", "--mode", "frames"]


def test_dream_step_argv():
    rv = step_argv("review", {"dates": ["2026-10-27", "2026-10-28"], "backend": "cli", "stage_a_only": True})
    assert rv[-7:] == ["--date", "2026-10-27", "--date", "2026-10-28", "--backend", "cli", "--stage-a-only"]
    tr = step_argv("train", {"run_id": "n1.1", "version": "n1.1", "check": True})
    assert "--check" in tr and "--export-gguf" in tr and tr[tr.index("--run-id") + 1] == "n1.1"
    ev = step_argv("eval", {"version": "n1.1", "no_judge": True})
    assert ev[-3:] == ["run", "n1.1", "--no-judge"]


# ─── supervisor ───────────────────────────────────────────

class ScriptSpec(ServiceSpec):
    def __init__(self, name, code):
        super().__init__(name, name, None)
        self.code = code

    def argv(self, values):
        return [sys.executable, "-u", "-c", self.code]


def make_sup(tmp_path, specs):
    return Supervisor({s.name: s for s in specs}, LogBus(tmp_path / "logs"), StateStore(tmp_path / "state.json"))


def wait_for(cond, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_crashing_service_restarts_then_gives_up(tmp_path, monkeypatch):
    monkeypatch.setattr(supmod, "MAX_RESTARTS", 2)
    sup = make_sup(tmp_path, [ScriptSpec("boom", "print('up'); raise SystemExit(3)")])
    sup.start("boom")
    m = sup.svc["boom"]
    states = []
    for _ in range(4):
        assert wait_for(lambda: m.proc is None or m.proc.poll() is not None)
        sup._check("boom")                 # sees the exit → schedules a restart
        states.append(m.state)
        if m.state == "crashed":
            break
        m.next_restart_at = time.time()    # skip the backoff
        sup._check("boom")                 # respawns
    assert states[-1] == "crashed" and len(m.restarts) == 3 and m.exit_code == 3
    assert any("up" == e["text"] for e in sup.bus.backfill("boom"))


def test_stop_kills_child_tree(tmp_path):
    code = ("import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "print('ready', flush=True); time.sleep(60)")
    sup = make_sup(tmp_path, [ScriptSpec("tree", code)])
    sup.start("tree")
    m = sup.svc["tree"]
    assert wait_for(lambda: any(e["text"] == "ready" for e in sup.bus.backfill("tree")))
    parent = psutil.Process(m.pid)
    assert wait_for(lambda: parent.children())
    kids = parent.children(recursive=True)
    sup.stop("tree")
    assert m.state == "stopped" and not m.want_running
    assert wait_for(lambda: not any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in [parent, *kids]))


def test_flags_persist_and_mark_restart(tmp_path):
    spec = ServiceSpec("svc", "svc", None, flags=[Flag("--gain", "float", 6.0)])
    sup = make_sup(tmp_path, [spec])
    sup.set_flags("svc", {"--gain": 3.0, "--bogus": 1})
    assert StateStore(tmp_path / "state.json").get("flags:svc") == {"--gain": 3.0}
    assert sup.flags("svc") == {"--gain": 3.0}
