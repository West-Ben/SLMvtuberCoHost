"""Rachel control panel — one local web app to launch, watch, configure and train the pipeline."""

import os
import sys
from pathlib import Path

# Snapshot BEFORE anything imports config.py: config's load_dotenv() copies .env into
# os.environ, and children would inherit those stale values instead of re-reading
# .env after a Settings change. Children are always spawned from this snapshot.
BASE_ENV = dict(os.environ)

SRC_DIR = Path(__file__).resolve().parent.parent       # code
REPO_ROOT = SRC_DIR.parent                              # .env, persona/, runtime data
VTUBE_ROOT = REPO_ROOT.parent
ENV_PATH = REPO_ROOT / ".env"
LOG_DIR = REPO_ROOT / "logs"
STATE_PATH = Path(__file__).resolve().parent / "state.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def child_env(**extra) -> dict:
    # Children run with cwd=REPO_ROOT (data paths are relative to it) and import from src/.
    pythonpath = os.pathsep.join(p for p in (str(SRC_DIR), BASE_ENV.get("PYTHONPATH", "")) if p)
    return {**BASE_ENV, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONPATH": pythonpath, **extra}
