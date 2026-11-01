"""
Start Rachel: opens the control panel at http://127.0.0.1:8060/.

    python start.py [--port 8060] [--no-browser]

Everything else (TTS, OBS bridge, orchestrator, STT, Twitch, llama-server) is
started, watched and restarted from the panel. Code lives in src/.
"""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

os.chdir(ROOT)                    # runtime data (rows/, logs/, imagelog/ …) is relative to the project root
sys.path.insert(0, str(SRC))
for stream in (sys.stdout, sys.stderr):   # Windows code pages can't print the panel's "→"/emoji when redirected
    stream.reconfigure(encoding="utf-8", errors="replace")
runpy.run_module("control_panel", run_name="__main__", alter_sys=True)
