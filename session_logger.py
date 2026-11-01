"""
session_logger.py — Fat episodic session log for Rachel's dream pass.

Distinct from training_logger.py:
  - training_logger → lean training rows (system prompt + spoken + frames),
    optimized for the next LoRA. Strips the thought channel and context.
  - session_logger  → FAT episodic rows, optimized for dream.py to reflect on.
    Keeps what dream needs to judge importance: her thought, what she saw/heard,
    what was said, what she said — plus emotion/intensity tagged at write time.

One JSONL file per day: SESSION_LOG_DIR/YYYY-MM-DD.jsonl
One row per logged reaction (timer turns for now — the path that carries full
vision+audio+ben context).

Row schema:
  {
    "ts":        float,      # epoch seconds
    "time":      "HH:MM:SS", # human-readable, same clock as imagelog groups
    "trigger":   str,        # "auto_commentary" (timer) | future: "chat" | "ben"
    "heard":     str,        # Ben's mic speech (STT) that fed this turn
    "vision":    str,        # perceiver's fused visual + stream-audio description
    "thought":   str,        # her channel monologue (tool calls already stripped)
    "emotion":   str,        # parsed from channel header at write time
    "intensity": str,        # low | medium | high
    "spoke":     str,        # what she actually said
    "images":    [str]|null, # imagelog frame paths for this turn
    "game":      str,
    "scene":     str
  }
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime

log = logging.getLogger(__name__)


class SessionLogger:
    def __init__(self, output_dir: str = "./session_log"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._lock = threading.Lock()

    def _path_for_today(self) -> str:
        return os.path.join(self.output_dir, f"{datetime.now().strftime('%Y-%m-%d')}.jsonl")

    def log(
        self,
        *,
        trigger: str,
        heard: str,
        vision: str,
        thought: str,
        emotion: str,
        intensity: str,
        spoke: str,
        images: list[str] | None = None,
        game: str = "",
        scene: str = "",
    ) -> None:
        """Append one fat episodic row. Never raises — logging must not break a turn."""
        now = time.time()
        row = {
            "ts":        round(now, 2),
            "time":      datetime.fromtimestamp(now).strftime("%H:%M:%S"),
            "trigger":   trigger,
            "heard":     heard or "",
            "vision":    vision or "",
            "thought":   thought or "",
            "emotion":   (emotion or "neutral").lower(),
            "intensity": (intensity or "medium").lower(),
            "spoke":     spoke or "",
            "images":    images,
            "game":      game or "",
            "scene":     scene or "",
        }
        try:
            with self._lock:
                with open(self._path_for_today(), "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning(f"[session_log] write failed: {e}")

    # ── Small CLI for inspection (mirrors training_logger ergonomics) ─────────

    def stats(self, date: str | None = None) -> dict:
        path = (os.path.join(self.output_dir, f"{date}.jsonl")
                if date else self._path_for_today())
        if not os.path.exists(path):
            return {"rows": 0, "path": path}
        emo: dict[str, int] = {}
        inten: dict[str, int] = {}
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                emo[r.get("emotion", "?")] = emo.get(r.get("emotion", "?"), 0) + 1
                inten[r.get("intensity", "?")] = inten.get(r.get("intensity", "?"), 0) + 1
        return {"rows": n, "path": path, "emotion": emo, "intensity": inten}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Inspect Rachel's session log.")
    ap.add_argument("command", choices=["stats"], help="stats: per-day row + tag counts")
    ap.add_argument("--date", help="YYYY-MM-DD (default today)")
    ap.add_argument("--dir", default=os.getenv("SESSION_LOG_DIR", "./session_log"))
    a = ap.parse_args()
    s = SessionLogger(output_dir=a.dir)
    if a.command == "stats":
        print(json.dumps(s.stats(a.date), indent=2, ensure_ascii=False))