"""Phase 2: WindowAccumulator policy, ReactionEngine fixture replay, frame_window locking."""

import base64
import io
import json
import threading
import time
from collections import deque
from pathlib import Path

import pytest
from PIL import Image

import frame_window
import orchestrator as orch
import playback
import prompt_format
import tags

FIXTURE = Path(__file__).parent / "fixtures" / "window_0001" / "window.json"


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def acc(clock, **kw):
    args = dict(window_s=10, max_s=14, min_s=3, silence_s=20, clock=clock)
    args.update(kw)
    return orch.WindowAccumulator(**args)


# ─── WindowAccumulator flush policy ───────────────────────

def poll(a, vad_idle=True, busy=False, quiet_for=0.0, active=True):
    return a.poll(vad_idle=vad_idle, busy=busy, quiet_for=quiet_for, stream_active=lambda: active)


def test_flush_after_window_when_vad_idle():
    c = Clock(); a = acc(c)
    a.add_speech("hi")
    c.advance(9.9); assert poll(a) is None
    c.advance(0.2); assert poll(a) == "window"


def test_no_flush_mid_utterance_until_hard_max():
    c = Clock(); a = acc(c)
    a.add_speech("so then")
    c.advance(12); assert poll(a, vad_idle=False) is None
    c.advance(2.1); assert poll(a, vad_idle=False) == "hard_max"


def test_direct_chat_flushes_early_after_min():
    c = Clock(); a = acc(c)
    c.advance(1); a.add_chat("v", "yo", direct=True)
    assert poll(a) is None
    c.advance(2.1); assert poll(a) == "direct"


def test_passive_chat_does_not_flush_early():
    c = Clock(); a = acc(c)
    a.add_chat("v", "yo", direct=False)
    c.advance(5); assert poll(a) is None


def test_busy_keeps_accumulating_and_merges():
    c = Clock(); a = acc(c)
    a.add_speech("one")
    c.advance(11); assert poll(a, busy=True) is None
    a.add_chat("v", "two")
    c.advance(10); assert poll(a, busy=True) is None
    reason = poll(a)
    assert reason == "hard_max"
    w = a.flush(reason)
    assert [s["text"] for s in w.speech] == ["one"] and [x["text"] for x in w.chat] == ["two"]
    assert w.end - w.start == pytest.approx(21)


def test_frames_only_silence_path_rate_limited_by_threshold():
    c = Clock(); a = acc(c)
    c.advance(30)
    assert poll(a, quiet_for=19) is None
    assert poll(a, quiet_for=21) == "silence"
    assert poll(a, quiet_for=21, active=False) is None
    assert poll(a, quiet_for=21, vad_idle=False) is None


def test_empty_window_slides_and_first_remark_gets_follow_up_time():
    c = Clock(); a = acc(c)
    c.advance(100); poll(a)
    assert c() - a.start == pytest.approx(10)
    a.add_speech("late")                     # start pulled to t - (window - min) = t - 7
    c.advance(2.9); assert poll(a) is None
    c.advance(0.2); assert poll(a) == "window"


def test_flush_resets_window():
    c = Clock(); a = acc(c)
    a.add_speech("x"); c.advance(10.5)
    w = a.flush(poll(a))
    assert w.reason == "window" and a.start == c() and not a.speech and not a.direct


@pytest.mark.parametrize("speech,chat,kind", [
    ([], [], "frames"), ([1], [], "frames+speech"), ([], [1], "frames+chat"), ([1], [1], "frames+chat+speech"),
])
def test_input_kind(speech, chat, kind):
    assert orch.input_kind(speech, chat) == kind


# ─── ReactionEngine fixture replay ────────────────────────

class FakeBackend:
    """Instant TTS/player/Warudo on a fake monotonic clock: every clip lasts 0.5 s."""

    def __init__(self):
        self.t = 0.0
        self.calls = []

    def clock(self):
        return self.t

    def sleep(self, dt):
        self.t += dt

    def synth(self, text):
        return playback.Clip(text=text, duration_s=0.5)

    def play(self, clip):
        self.calls.append((round(self.t, 3), "play", clip.text))
        self.t += clip.duration_s

    def act(self, expr, anim):
        self.calls.append((round(self.t, 3), "act", expr, anim))


class InlinePresynth:
    def __init__(self, texts, synth_fn):
        self.texts, self.synth_fn = texts, synth_fn

    def wait(self, i):
        return self.synth_fn(self.texts[i])


def FakeSpeaker():
    """A real playback.Sequencer over FakeBackend (keeps the old name for the tests below)."""
    be = FakeBackend()
    seq = playback.Sequencer(be, clock=be.clock, sleep=be.sleep, presynth_factory=InlinePresynth,
                             lead_s=1.0, reset_expr="relaxed", reset_delay_s=0.0)
    seq.calls = be.calls
    return seq


class FakeLogger:
    def __init__(self):
        self.rows = []

    def log_reaction(self, record):
        self.rows.append(json.loads(json.dumps(record)))
        return f"row{len(self.rows)}"


def _replay(fx, prompts):
    clock = Clock(fx["start"])
    model_calls = []

    def model_fn(turns):
        text = prompt_format.render(turns)
        prompts.append(text)
        model_calls.append(turns)
        return fx["model_output"]

    def frames_fn(seconds):
        return {"window_id": fx["window_id"], "frames": fx["frames"], "chat": fx["passive_chat"]}

    speaker, logger = FakeSpeaker(), FakeLogger()
    eng = orch.ReactionEngine(
        system="You are Rachel.", model_fn=model_fn, frames_fn=frames_fn, player=speaker,
        row_logger=logger, clock=clock,
        adapter_version_fn=lambda: "test-adapter", sampling={"temperature": 1.0},
    )
    for s in fx["speech"]:
        clock.t = s["t"]; eng.on_speech(s["text"])
    for c in fx["chat"]:
        clock.t = c["t"]; eng.on_chat(c["user"], c["text"], direct=False)
    clock.t = fx["end"]
    record = eng.tick()
    return eng, record, speaker, logger


def test_fixture_window_replays_deterministically():
    fx = json.loads(FIXTURE.read_text())
    p1, p2 = [], []
    _, r1, s1, l1 = _replay(fx, p1)
    _, r2, s2, l2 = _replay(fx, p2)
    assert p1 == p2 and len(p1) == 1
    strip = lambda r: {k: v for k, v in r.items() if k != "latency_ms"}
    assert strip(r1) == strip(r2)
    assert s1.calls == s2.calls


def test_fixture_window_end_to_end_contents():
    fx = json.loads(FIXTURE.read_text())
    prompts = []
    eng, rec, speaker, logger = _replay(fx, prompts)
    prompt = prompts[0]

    # accumulator → prompt
    assert prompt.count(prompt_format.IMAGE_TOKEN) == 5
    assert "[ben +2.1s] okay hela is actually cooking right now" in prompt
    assert "[chat +6.8s] viewer1: strange diff" in prompt
    assert "[chat +8.0s] lurker: jeff ult incoming" in prompt   # passive chat merged
    assert "before window" not in prompt                        # passive chat outside window dropped
    assert prompt.endswith(prompt_format.GENERATION_PREFIX)

    # record → row fields
    assert rec["input_kind"] == "frames+chat+speech"
    assert rec["window_id"] == "fixture0001"
    assert rec["frame_paths"] == [f["path"] for f in fx["frames"]]
    assert rec["adapter_version"] == "test-adapter"
    assert rec["prompt_version"] == prompt_format.prompt_version("You are Rachel.")
    assert rec["raw_output"] == fx["model_output"]
    assert rec["parse_flags"] == []
    assert rec["id"] == "row1" and logger.rows[0]["window_id"] == "fixture0001"

    # parser → plan → sequencer (0.5 s clips, 1.0 s lead) → trace on the row
    assert [s["type"] for s in rec["plan"]["steps"]] == ["speak", "act", "vocal", "speak"]
    assert speaker.calls == [
        (0.0, "play", "hela is always cooking."),
        (0.5, "act", "joy", "shrug"),
        (0.5, "play", "[chuckle]"),
        (1.5, "play", "strange diff is a lie and you know it"),   # max(T + 1.0, vocal end)
        (2.0, "act", "relaxed", None),                             # reset at reply end
    ]
    assert rec["playback_trace"]["complete_t"] == 2.0
    assert logger.rows[0]["playback_trace"] == rec["playback_trace"]

    # history carries text only, no images
    assert eng.history[-1]["assistant"] == fx["model_output"]
    assert prompt_format.IMAGE_TOKEN not in eng.history[-1]["user"]


def test_second_window_uses_history_without_images_and_links_history_ref():
    fx = json.loads(FIXTURE.read_text())
    prompts = []
    eng, rec, _, logger = _replay(fx, prompts)
    eng.clock.advance(5)
    eng.on_chat("v2", "lol", direct=True)
    eng.clock.advance(4)
    rec2 = eng.tick()
    p2 = prompts[-1]
    assert rec2["history_ref"] == "row1"
    assert p2.count(prompt_format.IMAGE_TOKEN) == 5          # only the current window's frames
    assert fx["model_output"] in p2


def test_engine_record_validates_against_row_schema(tmp_path):
    from training.training_logger import ReactionRowLogger, iter_rows
    fx = json.loads(FIXTURE.read_text())
    eng, _, _, _ = _replay(fx, [])
    eng.row_logger = ReactionRowLogger(tmp_path)
    eng.on_speech("again"); eng.clock.advance(11)
    rec = eng.tick()
    rows = list(iter_rows(tmp_path))
    assert len(rows) == 1 and rows[0]["id"] == rec["id"] and rows[0]["history_ref"] == "row1"


def test_model_failure_does_not_speak_or_log():
    sp, lg = FakeSpeaker(), FakeLogger()
    eng = orch.ReactionEngine(system="s", model_fn=lambda t: None, frames_fn=lambda s: {},
                              player=sp, row_logger=lg, clock=Clock())
    eng.on_speech("hi"); eng.clock.advance(11)
    assert eng.tick() is None and sp.calls == [] and lg.rows == []
    assert not eng.busy


def test_garbage_model_output_never_raises():
    sp = FakeSpeaker()
    eng = orch.ReactionEngine(system="s", model_fn=lambda t: "{expr:<|turn>{{{ <channel|>",
                              frames_fn=lambda s: {}, player=sp, clock=Clock())
    eng.on_speech("hi"); eng.clock.advance(11)
    rec = eng.tick()
    assert rec is not None and "empty" in rec["parse_flags"] and sp.calls == []


def test_own_speech_not_fed_back_as_ben():
    fx = json.loads(FIXTURE.read_text())
    eng, _, _, _ = _replay(fx, [])
    assert eng.on_speech("is always cooking strange diff is a lie and you know it") is False
    assert eng.on_speech("wait what was that") is True


def test_busy_engine_does_not_start_new_call():
    c = Clock()
    gate, started = threading.Event(), threading.Event()
    calls = []

    def slow_model(turns):
        calls.append(1); started.set(); gate.wait(5)
        return "ok"

    eng = orch.ReactionEngine(system="s", model_fn=slow_model, frames_fn=lambda s: {},
                              player=FakeSpeaker(), clock=c)
    eng.on_speech("first"); c.advance(11)
    t = threading.Thread(target=eng.tick); t.start()
    assert started.wait(2)
    eng.on_speech("second"); c.advance(30)
    assert eng.tick() is None and len(calls) == 1        # in flight → accumulate, don't call
    gate.set(); t.join(2)
    rec = eng.tick()
    assert [s["text"] for s in rec["speech"]] == ["second"] and len(calls) == 2


# ─── frame_window: selection + lock discipline ────────────

def _jpeg_b64(color, box=None):
    im = Image.new("RGB", (320, 180), color)
    if box:
        im.paste((255, 255, 255), box)
    buf = io.BytesIO(); im.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_static_scene_thins_to_min_frames():
    static = [(float(i), _jpeg_b64((50, 50, 50))) for i in range(10)]
    assert len(frame_window.select_frames(static, 4, 10)) == 4


def test_motion_scene_keeps_more_frames_capped_at_max():
    moving = [(float(i), _jpeg_b64((i * 18,) * 3)) for i in range(14)]
    sel = frame_window.select_frames(moving, 4, 10)
    assert len(sel) == 10
    assert sel[0][0] == 0.0 and sel[-1][0] == 13.0


def test_select_handles_fewer_than_min():
    two = [(0.0, _jpeg_b64((0, 0, 0))), (1.0, _jpeg_b64((0, 0, 0)))]
    assert len(frame_window.select_frames(two, 4, 10)) == 2
    assert frame_window.select_frames([], 4, 10) == []


def test_save_frames_downscales_and_returns_relative_paths(tmp_path):
    imagelog = tmp_path / "imagelog"
    entries = [(1761300000.0 + i, _jpeg_b64((i * 40, 10, 10))) for i in range(4)]
    out = frame_window.save_frames(entries, imagelog, width=160, window_id="abc")
    assert out["window_id"] == "abc"
    for f in out["frames"]:
        assert f["path"].startswith("imagelog/") and "/w-abc/" in f["path"]
        img = Image.open(tmp_path / f["path"])
        assert img.width == 160


def test_window_does_not_hold_lock_during_processing(monkeypatch, tmp_path):
    """Screenshot capture must keep appending while /window downsizes + saves."""
    lock = threading.Lock()
    ring = deque([{"timestamp": time.time(), "b64": _jpeg_b64((i, i, i)), "file": None} for i in range(8)], maxlen=30)
    real = frame_window.downscale_jpeg
    in_processing = threading.Event()

    def slow_downscale(b64, width, quality=80):
        in_processing.set(); time.sleep(0.05)
        return real(b64, width, quality)

    monkeypatch.setattr(frame_window, "downscale_jpeg", slow_downscale)

    captured = []

    def capture():
        in_processing.wait(2)
        for _ in range(5):
            t0 = time.perf_counter()
            with lock:
                ring.append({"timestamp": time.time(), "b64": "x", "file": None})
            captured.append(time.perf_counter() - t0)

    th = threading.Thread(target=capture); th.start()
    entries = frame_window.snapshot(ring, lock, 60, time.time())
    sel = frame_window.select_frames(entries, 4, 10)
    frame_window.save_frames(sel, tmp_path / "imagelog", 64)
    th.join(2)
    assert len(captured) == 5 and max(captured) < 0.02
