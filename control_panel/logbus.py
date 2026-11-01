"""
logbus.py — every process/job line goes through here.

Per-source ring buffers (for backfill), daily log files under logs/<source>/,
and fan-out to WebSocket subscribers. publish() is thread-safe and is called
from subprocess reader threads; subscribers are asyncio queues on the app loop.
"""

import asyncio
import re
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_ERROR_RE = re.compile(r"\b(ERROR|CRITICAL|Traceback|Exception)\b|🔴|✗")
_WARN_RE = re.compile(r"\b(WARNING|WARN)\b|⚠")
_DEBUG_RE = re.compile(r"\bDEBUG\b")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def level_of(text: str) -> str:
    if _ERROR_RE.search(text):
        return "error"
    if _WARN_RE.search(text):
        return "warn"
    if _DEBUG_RE.search(text):
        return "debug"
    return "info"


def _safe_name(source: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", source)


class LogBus:
    def __init__(self, log_dir: Path, ring_size: int = 3000):
        self.log_dir = Path(log_dir)
        self.ring_size = ring_size
        self._lock = threading.Lock()
        self._rings: dict[str, deque] = {}
        self._all: deque = deque(maxlen=ring_size)
        self._seq = 0
        self._subs: set[tuple[asyncio.AbstractEventLoop, asyncio.Queue, str | None]] = set()
        self.status_lines: dict[str, str] = {}

    # ── write side (any thread) ───────────────────────────
    def publish(self, source: str, text: str, level: str | None = None) -> dict:
        text = strip_ansi(text).rstrip()
        entry = {"ts": time.time(), "source": source, "level": level or level_of(text), "text": text}
        with self._lock:
            self._seq += 1
            entry["seq"] = self._seq
            self._rings.setdefault(source, deque(maxlen=self.ring_size)).append(entry)
            self._all.append(entry)
            subs = list(self._subs)
        self._write_file(source, entry)
        self._fanout(subs, {"type": "log", **entry})
        return entry

    def set_status_line(self, source: str, text: str):
        text = strip_ansi(text).strip()
        if not text or self.status_lines.get(source) == text:
            return
        self.status_lines[source] = text
        with self._lock:
            subs = list(self._subs)
        self._fanout(subs, {"type": "status_line", "source": source, "text": text})

    def event(self, payload: dict):
        """Non-log push (service state changes etc.) to every subscriber."""
        with self._lock:
            subs = list(self._subs)
        self._fanout(subs, payload)

    def _fanout(self, subs, payload: dict):
        for loop, q, want in subs:
            if want and payload.get("type") in ("log", "status_line") and payload.get("source") != want:
                continue
            try:
                loop.call_soon_threadsafe(q.put_nowait, payload)
            except RuntimeError:
                pass   # loop closed

    def _write_file(self, source: str, entry: dict):
        try:
            d = self.log_dir / _safe_name(source)
            d.mkdir(parents=True, exist_ok=True)
            stamp = datetime.fromtimestamp(entry["ts"])
            with open(d / f"{stamp:%Y-%m-%d}.log", "a", encoding="utf-8") as f:
                f.write(f"{stamp:%H:%M:%S} {entry['text']}\n")
        except OSError:
            pass

    # ── read side ─────────────────────────────────────────
    def backfill(self, source: str | None = None, n: int = 500) -> list[dict]:
        with self._lock:
            ring = self._all if source is None else self._rings.get(source, ())
            return list(ring)[-n:]

    def sources(self) -> list[str]:
        with self._lock:
            return sorted(self._rings)

    def log_file(self, source: str, date: str | None = None) -> Path:
        return self.log_dir / _safe_name(source) / f"{date or f'{datetime.now():%Y-%m-%d}'}.log"

    def subscribe(self, source: str | None = None) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=5000)
        with self._lock:
            self._subs.add((asyncio.get_running_loop(), q, source))
        return q

    def unsubscribe(self, q: asyncio.Queue):
        with self._lock:
            self._subs = {s for s in self._subs if s[1] is not q}


class LineSplitter:
    """
    Turns a raw byte stream into log lines + status-line updates.

    A '\\n'-terminated line is a log line; within it, text before the last
    '\\r' was overwritten on a terminal, so only the last segment is kept.
    Text after a '\\r' with no newline yet (stt-server's live level meter)
    is a status-line update, not a log line.
    """

    def __init__(self, on_line, on_status):
        self.on_line, self.on_status = on_line, on_status
        self.buf = ""

    def feed(self, chunk: str):
        self.buf += chunk
        *lines, self.buf = self.buf.split("\n")
        for line in lines:
            segs = [s for s in line.split("\r") if strip_ansi(s).strip()]
            if segs:
                self.on_line(segs[-1])
        if "\r" in self.buf:
            *done, self.buf = self.buf.split("\r")
            segs = [s for s in done if strip_ansi(s).strip()]
            if segs:
                self.on_status(segs[-1])

    def flush(self):
        if strip_ansi(self.buf).strip():
            self.on_line(self.buf)
        self.buf = ""
