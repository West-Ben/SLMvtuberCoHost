"""
training/registry.py — adapters/registry.json.

  {"adapters": [{"version", "parent", "data_range", "metrics", "status",
                 "path", "gguf", "created", "prompt_version", "grammar_hash"}],
   "current": "<version>" | null}

status: candidate → promoted | rejected; a promoted adapter that is later
replaced stays "promoted" (history). Promotion and rollback only move the
`current` pointer; the orchestrator reads it per reaction.

CLI:
  python -m training.registry show
  python -m training.registry promote <version>
  python -m training.registry rollback            # current → its parent
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "adapters" / "registry.json"


def load(path: Path = REGISTRY_PATH) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"adapters": [], "current": None}


def save(reg: dict, path: Path = REGISTRY_PATH):
    """Atomic replace so the orchestrator never reads a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def get(reg: dict, version: str | None) -> dict | None:
    return next((a for a in reg["adapters"] if a["version"] == version), None)


def register(entry: dict, path: Path = REGISTRY_PATH) -> dict:
    reg = load(path)
    if get(reg, entry["version"]):
        raise ValueError(f"adapter {entry['version']} already registered")
    entry = {"status": "candidate", "metrics": None, "created": time.time(), **entry}
    reg["adapters"].append(entry)
    save(reg, path)
    return entry


def set_status(version: str, status: str, metrics: dict | None = None, path: Path = REGISTRY_PATH):
    reg = load(path)
    a = get(reg, version)
    if a is None:
        raise KeyError(version)
    a["status"] = status
    if metrics is not None:
        a["metrics"] = metrics
    save(reg, path)


def promote(version: str, path: Path = REGISTRY_PATH):
    reg = load(path)
    a = get(reg, version)
    if a is None:
        raise KeyError(version)
    a["status"] = "promoted"
    a["promoted_at"] = time.time()
    reg["current"] = version
    save(reg, path)


def rollback(path: Path = REGISTRY_PATH) -> str | None:
    """Move `current` to the current adapter's parent (None = base_core only)."""
    reg = load(path)
    cur = get(reg, reg["current"])
    reg["current"] = cur.get("parent") if cur else None
    save(reg, path)
    return reg["current"]


def next_version(reg: dict, day: str) -> str:
    n = sum(1 for a in reg["adapters"] if a["version"].startswith(f"n{day}"))
    return f"n{day}.{n + 1}"


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd in ("-h", "--help"):
        print(__doc__)
    elif cmd == "show":
        print(json.dumps(load(), indent=2))
    elif cmd == "promote" and len(sys.argv) > 2:
        promote(sys.argv[2]); print(f"current → {sys.argv[2]}")
    elif cmd == "rollback":
        print(f"current → {rollback()}")
    else:
        raise SystemExit(__doc__)
