"""
frame_window.py — window frame selection for obs_bridge GET /window.

Split out of obs_bridge.py (which parses argv and opens audio devices at
import) so it can be unit-tested. Three steps, only the first under _lock:

  snapshot(ring, lock, seconds, now)   copy (ts, b64) refs under the lock, release
  select_frames(entries, min, max)     motion-driven thinning, outside the lock
  save_frames(selected, ...)           downscale + write JPEGs, outside the lock
"""

import base64
import io
import threading
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

MOTION_THRESHOLD = 8.0   # default; obs_bridge passes config.VISION_MOTION_THRESHOLD


def snapshot(ring, lock: threading.Lock, seconds: float, now: float) -> list[tuple[float, str]]:
    """[(ts, b64)] oldest-first for frames in the last `seconds`. Holds lock only to copy."""
    cutoff = now - seconds
    with lock:
        return [(s["timestamp"], s["b64"]) for s in ring if s.get("b64") and s["timestamp"] >= cutoff]


def signature(b64: str):
    """32×32 grayscale array for motion / near-duplicate checks; None if undecodable."""
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L").resize((32, 32))
        return np.asarray(img, dtype=np.float32)
    except Exception:
        return None


def _evenly(items: list, k: int) -> list:
    if k >= len(items):
        return list(items)
    if k <= 1:
        return [items[-1]]
    idx = sorted({round(i * (len(items) - 1) / (k - 1)) for i in range(k)})
    return [items[i] for i in idx]


def select_frames(
    entries: list[tuple[float, str]],
    min_frames: int,
    max_frames: int,
    threshold: float = MOTION_THRESHOLD,
) -> list[tuple[float, str]]:
    """
    Motion-driven thinning, oldest-first in and out. Keeps the oldest and newest
    frame plus every frame that changed > threshold from the last kept frame.
    A static scene thins toward min_frames; never fewer than min(min_frames, len).
    """
    if not entries:
        return []
    sigs = [signature(b) for _, b in entries]
    keep = [0]
    for i in range(1, len(entries)):
        a, b = sigs[keep[-1]], sigs[i]
        if a is None or b is None or float(np.abs(a - b).mean()) > threshold:
            keep.append(i)
    if keep[-1] != len(entries) - 1:
        keep.append(len(entries) - 1)

    if len(keep) < min_frames:
        extra = [i for i in _evenly(list(range(len(entries))), min_frames) if i not in keep]
        keep = sorted(set(keep) | set(extra[: min_frames - len(keep)]))
        if len(keep) < min(min_frames, len(entries)):
            keep = _evenly(list(range(len(entries))), min_frames)
    if len(keep) > max_frames:
        keep = _evenly(keep, max_frames)
    return [entries[i] for i in keep]


def downscale_jpeg(b64: str, width: int, quality: int = 80) -> bytes:
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def save_frames(
    selected: list[tuple[float, str]],
    imagelog_dir: Path,
    width: int,
    window_id: str | None = None,
    now: float | None = None,
) -> dict:
    """
    Downscale and write frames to imagelog/YYYY-MM-DD/w-<window_id>/fNN.jpg.
    Returns {"window_id", "frames": [{"id", "path", "t"}]} where path is
    relative to imagelog_dir's parent (the repo root), posix-style.
    """
    window_id = window_id or uuid.uuid4().hex[:12]
    day = datetime.fromtimestamp(now or (selected[-1][0] if selected else 0)).strftime("%Y-%m-%d")
    rel_dir = Path(imagelog_dir.name) / day / f"w-{window_id}"
    out_dir = imagelog_dir.parent / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, (ts, b64) in enumerate(selected, 1):
        name = f"f{i:02d}.jpg"
        (out_dir / name).write_bytes(downscale_jpeg(b64, width))
        frames.append({"id": f"{window_id}-{i:02d}", "path": (rel_dir / name).as_posix(), "t": ts})
    return {"window_id": window_id, "frames": frames}
