"""Playback sequencer timelines (spec section 2 + confirmed decisions A/B/C/E)."""

import pytest

import playback
from playback import Clip, Sequencer
from tag_parser import parse

EX1 = "Ben you need to use your alt, {anim:shrug}{expr:sad} [sigh] he never listens."
EX2 = "{anim:laugh}[laugh] that was a great move!"


class World:
    """Fake monotonic clock + TTS + player + Warudo. Time only moves via sleep/play/synth waits."""

    def __init__(self, durations: dict, ready: dict | None = None, fail: set | None = None):
        self.t = 100.0                     # arbitrary monotonic origin
        self.durations = durations         # clip text → seconds
        self.ready = ready or {}           # clip text → seconds after reply start when synthesis finishes
        self.fail = fail or set()
        self.events: list[tuple] = []      # (t_rel, kind, payload)
        self.t0 = None
        self.timers: list = []

    # clock
    def clock(self):
        return self.t

    def sleep(self, dt):
        assert dt >= 0
        self.t += dt

    def rel(self):
        return round(self.t - self.t0, 6)

    # backend
    def synth(self, text):
        if text in self.fail:
            raise RuntimeError("tts failed")
        return Clip(text=text, duration_s=self.durations[text], handle=text)

    def play(self, clip):
        self.events.append((self.rel(), "play", clip.text))
        self.t += clip.duration_s
        self.events.append((self.rel(), "end", clip.text))

    def act(self, expr, anim):
        self.events.append((self.rel(), "act", (expr, anim)))

    # presynth: deterministic stand-in for the background synthesis thread
    def presynth(self, texts, synth_fn):
        world = self

        class P:
            def wait(self, i):
                text = texts[i]
                due = world.t0 + world.ready.get(text, 0.0)
                if world.t < due:
                    world.t = due
                try:
                    return synth_fn(text)
                except Exception:
                    return None
        world.t0 = world.t if world.t0 is None else world.t0
        return P()

    # timers for the expression reset
    def call_later(self, delay, fn):
        h = {"due": self.t + delay, "fn": fn, "cancelled": False}
        self.timers.append(h)

        class Handle:
            def cancel(self_inner):
                h["cancelled"] = True
        return Handle()

    def run_timers(self, until):
        for h in self.timers:
            if not h["cancelled"] and h["due"] <= until and not h.get("ran"):
                h["ran"] = True
                self.t = max(self.t, h["due"])
                h["fn"]()


def seq(world, **kw):
    args = dict(backend=world, clock=world.clock, sleep=world.sleep, presynth_factory=world.presynth,
                call_later=world.call_later, lead_s=1.0, reset_expr="relaxed", reset_delay_s=0.0)
    args.update(kw)
    return Sequencer(**args)


def timeline(world):
    return [e for e in world.events]


# ─── Worked examples ──────────────────────────────────────

def test_example_1_timeline():
    w = World({"Ben you need to use your alt,": 2.0, "[sigh]": 0.6, "he never listens.": 1.5})
    trace = seq(w).play(parse(EX1))
    assert timeline(w)[:7] == [
        (0.0, "play", "Ben you need to use your alt,"),
        (2.0, "end", "Ben you need to use your alt,"),
        (2.0, "act", ("sad", "shrug")),
        (2.0, "play", "[sigh]"),
        (2.6, "end", "[sigh]"),
        (3.0, "play", "he never listens."),          # max(T + 1.0, end(sigh)) = 3.0
        (4.5, "end", "he never listens."),
    ]
    assert trace.complete_t == pytest.approx(4.5)


def test_example_2_timeline():
    w = World({"[laugh]": 0.8, "that was a great move!": 1.2})
    trace = seq(w).play(parse(EX2))
    assert timeline(w)[:5] == [
        (0.0, "act", (None, "laugh")),
        (0.0, "play", "[laugh]"),
        (0.8, "end", "[laugh]"),
        (1.0, "play", "that was a great move!"),     # max(0 + 1.0, 0.8)
        (2.2, "end", "that was a great move!"),
    ]
    assert trace.complete_t == pytest.approx(2.2)


# ─── Decisions A / B / E ──────────────────────────────────

def test_long_vocal_delays_speech_until_vocal_ends():
    w = World({"Ben you need to use your alt,": 2.0, "[sigh]": 2.0, "he never listens.": 1.5})
    seq(w).play(parse(EX1))
    assert (4.0, "play", "he never listens.") in w.events     # max(2+1, 2+2) = 4.0


def test_action_beat_without_vocal_still_leads_by_one_second():
    w = World({"nice one ben.": 1.0, "you earned that one.": 1.0})
    seq(w).play(parse("nice one ben. {expr:joy} you earned that one."))
    assert (1.0, "act", ("joy", None)) in w.events
    assert (2.0, "play", "you earned that one.") in w.events


def test_vocal_only_runs_play_back_to_back():
    w = World({"we lost that one.": 1.0, "[sigh]": 0.5, "[groan]": 0.7, "again.": 0.4})
    seq(w).play(parse("we lost that one. [sigh][groan] again."))
    plays = [(t, x) for t, k, x in w.events if k == "play"]
    assert plays == [(0.0, "we lost that one."), (1.0, "[sigh]"), (1.5, "[groan]"), (2.2, "again.")]
    assert not any(k == "act" for _, k, _ in w.events)


def test_group_fires_with_first_of_multiple_bound_vocals():
    w = World({"[sigh]": 0.5, "[groan]": 0.5, "whatever you say": 1.0})
    seq(w).play(parse("{anim:shrug}{expr:sad}[sigh][groan] whatever you say"))
    assert w.events[:5] == [(0.0, "act", ("sad", "shrug")), (0.0, "play", "[sigh]"), (0.5, "end", "[sigh]"),
                            (0.5, "play", "[groan]"), (1.0, "end", "[groan]")]
    assert (1.0, "play", "whatever you say") in w.events      # max(0+1.0, 1.0)


def test_trailing_action_fires_at_end_and_completes_immediately():
    w = World({"that's so dumb.": 1.0})
    trace = seq(w).play(parse("that's so dumb. {anim:laugh}"))
    assert (1.0, "act", (None, "laugh")) in w.events
    assert trace.complete_t == pytest.approx(1.0)


def test_trailing_action_with_vocal_completes_after_vocal():
    w = World({"that's so dumb.": 1.0, "[laugh]": 0.9})
    trace = seq(w).play(parse("that's so dumb. {anim:laugh}[laugh]"))
    assert trace.complete_t == pytest.approx(1.9)


def test_leading_speak_plays_first_with_no_action():
    w = World({"hello there you": 1.0})
    seq(w).play(parse("hello there you"))
    assert w.events[0] == (0.0, "play", "hello there you")


# ─── Pre-synthesis, underrun, failure ─────────────────────

def test_underrun_waits_and_timing_follows_actual_times():
    w = World({"first bit here,": 1.0, "second bit here.": 1.0},
              ready={"first bit here,": 0.4, "second bit here.": 3.0})
    trace = seq(w).play(parse("first bit here, {expr:joy} second bit here."))
    plays = [(t, x) for t, k, x in w.events if k == "play"]
    assert plays == [(0.4, "first bit here,"), (3.0, "second bit here.")]
    first, act, second = trace.entries
    # the opening clip's wait is synthesis latency, logged separately from mid-reply underruns
    assert first["scheduled_t"] == 0.0 and first["actual_t"] == pytest.approx(0.4) and first["first_clip_ms"] == 400
    assert "underrun_ms" not in first
    assert act["actual_t"] == pytest.approx(1.4)                   # fired when clip 1 actually ended
    assert second["scheduled_t"] == pytest.approx(2.4) and second["underrun_ms"] == 600


def test_failed_vocal_still_fires_action_on_time():
    w = World({"Ben you need to use your alt,": 2.0, "he never listens.": 1.5}, fail={"[sigh]"})
    trace = seq(w).play(parse(EX1))
    assert (2.0, "act", ("sad", "shrug")) in w.events
    assert (3.0, "play", "he never listens.") in w.events          # T + LEAD_S
    failed = [e for e in trace.entries if e.get("clip_failed")]
    assert len(failed) == 1 and failed[0]["step"] == {"type": "vocal", "tag": "sigh"}
    assert "clip_failed" in trace.flags


def test_failed_speak_is_skipped():
    w = World({"b part here.": 1.0}, fail={"a part here,"})
    seq(w).play(parse("a part here, {expr:joy} b part here."))
    assert w.events[0] == (0.0, "act", ("joy", None))


def test_trace_shape_is_loggable():
    import json
    w = World({"Ben you need to use your alt,": 2.0, "[sigh]": 0.6, "he never listens.": 1.5})
    trace = seq(w).play(parse(EX1))
    d = json.loads(json.dumps(trace.to_dict()))
    assert [e["step"]["type"] for e in d["entries"]] == ["speak", "act", "vocal", "speak"]
    for e in d["entries"]:
        assert {"step", "scheduled_t", "actual_t", "clip_duration"} <= set(e)
    assert d["entries"][1]["clip_duration"] is None


def test_noop_plan_does_nothing():
    w = World({})
    trace = seq(w).play(parse("{expr:joy}"))
    assert w.events == [] and trace.entries == []


# ─── Expression reset (decision C / E) ────────────────────

def test_reset_fires_at_reply_end_when_expression_was_set():
    w = World({"Ben you need to use your alt,": 2.0, "[sigh]": 0.6, "he never listens.": 1.5})
    seq(w).play(parse(EX1))
    w.run_timers(until=w.t)
    assert w.events[-1] == (4.5, "act", ("relaxed", None))


def test_no_reset_when_reply_never_set_expression():
    w = World({"[laugh]": 0.8, "that was a great move!": 1.2})
    seq(w).play(parse(EX2))
    w.run_timers(until=w.t + 10)
    assert not any(k == "act" and x == ("relaxed", None) for _, k, x in w.events)


def test_pending_reset_cancelled_when_new_reply_starts_inside_delay():
    w = World({"ugh fine then.": 1.0, "okay next one now": 1.0})
    s = seq(w, reset_delay_s=2.0)
    s.play(parse("{expr:sad} ugh fine then."))
    w.t += 0.5                                   # new reply 0.5 s later, before the 2 s reset
    w.t0 = None
    s.play(parse("okay next one now"))
    w.run_timers(until=w.t + 10)
    resets = [e for e in w.events if e[1] == "act" and e[2] == ("relaxed", None)]
    assert resets == []


def test_reset_fires_after_delay_when_not_interrupted():
    w = World({"ugh fine then.": 1.0})
    s = seq(w, reset_delay_s=2.0)
    s.play(parse("{expr:sad} ugh fine then."))
    w.run_timers(until=w.t + 2.0)
    assert any(x == ("relaxed", None) for _, k, x in w.events if k == "act")


def test_one_reply_at_a_time():
    w = World({"a b c": 1.0})
    s = seq(w)
    assert s._lock.acquire(blocking=False)
    with pytest.raises(RuntimeError):
        s.play(parse("a b c"))
    s._lock.release()


def test_defaults_come_from_config():
    import config
    assert playback.LEAD_S == config.LEAD_S == 1.0
    assert config.EXPR_RESET == "relaxed" and config.EXPR_RESET_DELAY_S == 0.0
