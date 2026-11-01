"""Rachel control panel — one local web app to launch, watch, configure and train the pipeline."""

import os
import sys
from pathlib import Path

# Snapshot BEFORE anything imports config.py: config's load_dotenv() copies .env into
# os.environ, and children would inherit those stale values instead of re-reading
# .env after a Settings change. Children are always spawned from this snapshot.
BASE_ENV = dict(os.environ)

REPO_ROOT = Path(__file__).resolve().parent.parent
VTUBE_ROOT = REPO_ROOT.parent
ENV_PATH = REPO_ROOT / ".env"
LOG_DIR = REPO_ROOT / "logs"
STATE_PATH = Path(__file__).resolve().parent / "state.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def child_env(**extra) -> dict:
    return {**BASE_ENV, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", **extra}
