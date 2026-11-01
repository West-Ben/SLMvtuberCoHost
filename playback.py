"""
playback.py — plays a tag_parser.Plan with spec timing (see CLAUDE.md → Response Parser + Playback Sequencer).

Timing (all measured from ACTUAL times on a monotonic clock):
  1. Clips play strictly in order, one at a time — no overlapping audio.
  2. A leading Speak plays first, with no action.
  3. When the previous clip ends and the next item is an action beat, at that
     instant T the Act (expr + anim together) goes to Warudo and the bound
     Vocal clip(s) start; multiple bound vocals play back to back.
  4. The Speak after an action beat starts at max(T + LEAD_S, end of bound vocals);
     the lead applies to beats with no vocal too. Unbound vocal runs have no gap.
  5. The reply completes when the last clip finishes (a trailing beat with no
     vocal completes the instant it fires).
  A beat that directly follows another beat waits for the first beat's lead.

Pre-synthesis: every clip is submitted for synthesis, in order, the moment
the plan exists; clip 1 plays as soon as it is ready while later clips
synthesise. The opening clip's wait is logged as first_clip_ms; a later clip
not ready when due is waited for (underrun_ms + flag "underrun"); a
beat waits for its first bound vocal so face and sound stay simultaneous. A
clip that fails to synthesise is skipped (clip_failed) and its beat still fires.

Expression reset: after a reply that set an expression, send EXPR_RESET after
EXPR_RESET_DELAY_S; a new reply starting first cancels it. Animations are
one-shot and never reset.

Backend contract (TtsServerBackend talks to server.py):
  synth(text) -> Clip            text is the Speak text or "[tag]"; raises on failure
  play(clip)  -> None            blocks until playback has ended
  act(expr, anim) -> None        one Warudo beat
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import requests

from config import EXPR_RESET, EXPR_RESET_DELAY_S, LEAD_S
from tag_parser import Act, Plan, Speak, Vocal


@dataclass
class Clip:
    text: str
    duration_s: float          # from synthesized audio: samples / sample_rate
    handle: object = None      # backend reference (server clip_id)


@dataclass
class PlaybackTrace:
    entries: list = field(default_factory=list)   # {step, scheduled_t, actual_t, clip_duration, [underrun_ms], [clip_failed]}
    complete_t: float = 0.0
    flags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"entries": self.entries, "complete_t": self.complete_t, "flags": self.flags}


class Presynth:
    """Synthesises clips in order on one worker thread, starting immediately."""

    def __init__(self, texts: list[str], synth_fn):
        ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="presynth")
        self._futures = [ex.submit(synth_fn, t) for t in texts]
        ex.shutdown(wait=False)

    def wait(self, i: int) -> Clip | None:
        try:
            return self._futures[i].result()
        except Exception:
            return None


def clip_text(step) -> str:
    return step.text if isinstance(step, Speak) else f"[{step.tag}]"


def build_items(steps: list) -> list[tuple]:
    """("clip", i) for Speak / unbound Vocal; ("beat", act_i, [bound vocal indices])."""
    items, i = [], 0
    while i < len(steps):
        if isinstance(steps[i], Act):
            j = i + 1
            while j < len(steps) and isinstance(steps[j], Vocal):
                j += 1
            items.append(("beat", i, list(range(i + 1, j))))
            i = j
        else:
            items.append(("clip", i))
            i += 1
    return items


def _default_call_later(delay: float, fn):
    t = threading.Timer(delay, fn)
    t.daemon = True
    t.start()
    return t


class Sequencer:
    def __init__(self, backend, *, clock=time.monotonic, sleep=time.sleep, presynth_factory=Presynth,
                 call_later=_default_call_later, lead_s: float = LEAD_S, reset_expr: str = EXPR_RESET,
                 reset_delay_s: float = EXPR_RESET_DELAY_S):
        self.backend = backend
        self.clock, self.sleep = clock, sleep
        self.presynth_factory, self.call_later = presynth_factory, call_later
        self.lead_s, self.reset_expr, self.reset_delay_s = lead_s, reset_expr, reset_delay_s
        self._lock = threading.Lock()
        self._pending_reset = None

    # ── expression reset ──────────────────────────────────
    def _cancel_reset(self):
        if self._pending_reset is not None:
            self._pending_reset.cancel()
            self._pending_reset = None

    def _reset(self):
        self._pending_reset = None
        self.backend.act(self.reset_expr, None)

    def _schedule_reset(self):
        if self.reset_delay_s <= 0:
            self._reset()
        else:
            self._pending_reset = self.call_later(self.reset_delay_s, self._reset)

    # ── playback ──────────────────────────────────────────
    def play(self, plan: Plan) -> PlaybackTrace:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("a reply is already playing — orchestrator gating should prevent this")
        try:
            self._cancel_reset()
            trace = PlaybackTrace()
            if plan.is_noop:
                return trace
            self._run(plan, trace)
            return trace
        finally:
            self._lock.release()

    def _run(self, plan: Plan, trace: PlaybackTrace):
        steps = plan.steps
        clip_steps = [i for i, s in enumerate(steps) if not isinstance(s, Act)]
        clip_of = {i: n for n, i in enumerate(clip_steps)}
        pre = self.presynth_factory([clip_text(steps[i]) for i in clip_steps], self.backend.synth)
        t0 = self.clock()
        rel = lambda: self.clock() - t0
        free, lead_until = 0.0, 0.0
        expr_set = False

        def wait_until(t: float):
            d = t - rel()
            if d > 0:
                self.sleep(d)

        def entry(i, scheduled, actual, duration):
            e = {"step": steps[i].to_dict(), "scheduled_t": round(scheduled, 4),
                 "actual_t": round(actual, 4), "clip_duration": duration}
            if actual - scheduled > 0.0005:
                wait_ms = round((actual - scheduled) * 1000)
                if not trace.entries:
                    e["first_clip_ms"] = wait_ms     # synthesis latency of the opening clip — expected
                else:
                    e["underrun_ms"] = wait_ms       # a later clip wasn't ready when due
                    if "underrun" not in trace.flags:
                        trace.flags.append("underrun")
            trace.entries.append(e)
            return e

        def play_clip(i: int, scheduled: float):
            nonlocal free
            wait_until(scheduled)
            clip = pre.wait(clip_of[i])
            start = rel()
            if clip is None:
                entry(i, scheduled, start, None)["clip_failed"] = True
                if "clip_failed" not in trace.flags:
                    trace.flags.append("clip_failed")
                free = max(free, start)
                return
            assert start >= free - 1e-6, "clip overlap"
            e = entry(i, scheduled, start, round(clip.duration_s, 4))
            self.backend.play(clip)
            free = rel()
            e["end_t"] = round(free, 4)

        for item in build_items(steps):
            if item[0] == "clip":
                i = item[1]
                play_clip(i, max(free, lead_until))
                if isinstance(steps[i], Speak):
                    lead_until = 0.0
                continue
            _, ai, vocals = item
            act = steps[ai]
            scheduled = max(free, lead_until)
            wait_until(scheduled)
            if vocals:
                pre.wait(clip_of[vocals[0]])     # keep face and sound simultaneous (failure returns at once)
            T = rel()
            self.backend.act(act.expr, act.anim)
            expr_set = expr_set or act.expr is not None
            entry(ai, scheduled, T, None)
            for n, vi in enumerate(vocals):
                play_clip(vi, T if n == 0 else free)
            free = max(free, T)
            lead_until = T + self.lead_s

        trace.complete_t = round(free, 4)
        if expr_set:
            self._schedule_reset()


# ─── server.py backend ────────────────────────────────────

class TtsServerBackend:
    """server.py: POST /synth → clip_id + duration, POST /play-clip (blocks until playback end), POST /avatar."""

    def __init__(self, base_url: str, sleep=time.sleep):
        self.base = base_url.rstrip("/")
        self.sleep = sleep

    def synth(self, text: str) -> Clip:
        r = requests.post(f"{self.base}/synth", json={"text": text}, timeout=60)
        r.raise_for_status()
        d = r.json()
        if not d.get("clip_id") or float(d.get("duration_s") or 0) <= 0:
            raise RuntimeError(f"empty clip for {text!r}")
        return Clip(text=text, duration_s=float(d["duration_s"]), handle=d["clip_id"])

    def play(self, clip: Clip):
        t0 = time.monotonic()
        try:
            r = requests.post(f"{self.base}/play-clip", json={"clip_id": clip.handle},
                              timeout=clip.duration_s + 30)
            r.raise_for_status()
        except Exception:
            # Fallback clock: start + clip duration, so the timeline holds even if playback errored.
            remaining = clip.duration_s - (time.monotonic() - t0)
            if remaining > 0:
                self.sleep(remaining)

    def act(self, expr: str | None, anim: str | None):
        try:
            requests.post(f"{self.base}/avatar", json={"expression": expr, "animation": anim}, timeout=3)
        except Exception:
            pass
