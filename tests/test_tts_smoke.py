"""
Smoke test (decision D): every vocal tag synthesised ALONE yields a non-empty clip.
Catches Chatterbox updates that stop rendering a bare "[sigh]". Needs server.py
running (TTS loaded); skipped otherwise. Also run by preflight.py.
"""

import pytest
import requests

import tags
from config import SPEAK_URL

BASE = SPEAK_URL.rsplit("/", 1)[0]


def _server_up() -> bool:
    try:
        return requests.get(f"{BASE}/memory/stats", timeout=2).status_code < 500
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _server_up(), reason="server.py (TTS) not running")


@pytest.mark.parametrize("tag", sorted(tags.VOCAL_TAGS))
def test_vocal_tag_alone_synthesises_non_empty_clip(tag):
    r = requests.post(f"{BASE}/synth", json={"text": f"[{tag}]"}, timeout=60)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["samples"] > 0 and d["duration_s"] > 0.05, d
