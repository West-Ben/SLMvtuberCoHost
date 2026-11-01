"""
training/dataset_builder.py — assemble one nightly training set.

Mix:
  new      curated/ nights after the current adapter's data_range end
           (falls back to the K most recent nights when nothing is new)
  replay   sampled from the core set ≈ REPLAY_SHARE of the final mix:
           training/data/core.jsonl + curated nights already baked into
           promoted adapters
  human    rows with a human mark are repeated HUMAN_WEIGHT times

Guards:
  - only rows whose CURRENT effective status is accepted (a late human "bad"
    or a reviewer exclusion drops a row that was curated earlier)
  - nothing from eval-set sessions (training/data/eval_set.jsonl), ever
  - rows built for an older prompt_version are rebuilt with the current
    system prompt via prompt_format (train on the prompt we serve)
  - train/holdout split by stream session, never by random frames

Output: datasets/<run_id>/{train,holdout}.jsonl + manifest.json

Run: python -m training.dataset_builder [--k 3] [--seed 0] [--run-id ID]
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prompt_format                                 # noqa: E402
import tags                                          # noqa: E402
from config import CURATED_DIR, DATASETS_DIR, ROWS_DIR, TRAINING_DATA_STATIC   # noqa: E402
from training import registry                        # noqa: E402
from training import row_prompt                      # noqa: E402
from training import training_logger as tl           # noqa: E402

K_NIGHTS        = 3
REPLAY_SHARE    = 0.5
HUMAN_WEIGHT    = 2
HOLDOUT_SHARE   = 0.1
CORE_PATH       = TRAINING_DATA_STATIC / "core.jsonl"
EVAL_SET_PATH   = TRAINING_DATA_STATIC / "eval_set.jsonl"


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def curated_nights(curated_dir: Path = CURATED_DIR) -> list[str]:
    return sorted(p.stem for p in curated_dir.glob("????-??-??.jsonl"))


def eval_sessions(eval_path: Path = EVAL_SET_PATH) -> set[str]:
    return {e["session"] for e in _read(eval_path) if e.get("session")}


def pick_nights(nights: list[str], reg: dict, k: int) -> tuple[list[str], list[str]]:
    """(new nights, nights already in promoted adapters)."""
    promoted = [a for a in reg["adapters"] if a.get("status") == "promoted"]
    baked = sorted({n for a in promoted for n in (a.get("data_range") or {}).get("nights", [])})
    cur = registry.get(reg, reg.get("current"))
    last_end = ((cur or {}).get("data_range") or {}).get("end")
    new = [n for n in nights if last_end is None or n > last_end]
    if not new:
        new = nights[-k:]
    else:
        new = new[-k:] if len(new) > k else new
    return new, [n for n in baked if n not in new]


def refresh(row: dict, eff_by_id: dict, system: str, version: str) -> dict | None:
    """Drop rows no longer accepted; rebuild prompts made for an older prompt_version."""
    eff = eff_by_id.get(row["id"])
    if eff is not None and eff["status"] != "accepted":
        return None
    if row.get("prompt_version") == version:
        return row
    if eff is None:
        return None   # can't rebuild without the source row
    completion = row["completion"][: -len(prompt_format.EOT)] if row["completion"].endswith(prompt_format.EOT) else row["completion"]
    ex = row_prompt.training_example(eff, completion, eff_by_id, system)
    return {**row, **ex}


def split_by_session(rows: list[dict], share: float, rng: random.Random) -> tuple[list[dict], list[dict]]:
    sessions = sorted({r.get("session") or r["id"][:10] for r in rows})
    rng.shuffle(sessions)
    n_hold = max(1, round(len(sessions) * share)) if len(sessions) > 1 else 0
    hold = set(sessions[:n_hold])
    train = [r for r in rows if (r.get("session") or r["id"][:10]) not in hold]
    holdout = [r for r in rows if (r.get("session") or r["id"][:10]) in hold]
    return train, holdout


def build(k: int = K_NIGHTS, seed: int = 0, run_id: str | None = None,
          curated_dir: Path = CURATED_DIR, rows_dir: Path = ROWS_DIR, out_root: Path = DATASETS_DIR,
          core_path: Path = CORE_PATH, eval_path: Path = EVAL_SET_PATH,
          registry_path: Path = registry.REGISTRY_PATH) -> dict:
    rng = random.Random(seed)
    reg = registry.load(registry_path)
    system = prompt_format.load_system_prompt()
    version = prompt_format.prompt_version(system)
    eff_by_id = {r["id"]: r for r in tl.load_effective(rows_dir)}
    banned_sessions = eval_sessions(eval_path)

    nights = curated_nights(curated_dir)
    if not nights:
        raise SystemExit("no curated nights — run the reviewer first")
    new_nights, baked_nights = pick_nights(nights, reg, k)

    def load_nights(ns):
        out = []
        for n in ns:
            for r in _read(curated_dir / f"{n}.jsonl"):
                if r.get("session") in banned_sessions:
                    continue
                r = refresh(r, eff_by_id, system, version)
                if r is not None:
                    out.append(r)
        return out

    new_rows = load_nights(new_nights)
    if not new_rows:
        raise SystemExit("no accepted rows in the selected nights")

    core = [r for r in _read(core_path) if r.get("session") not in banned_sessions] + load_nights(baked_nights)
    core = [r for r in core if r["id"] not in {n["id"] for n in new_rows}]
    want_replay = round(len(new_rows) * REPLAY_SHARE / (1 - REPLAY_SHARE))
    replay = rng.sample(core, min(want_replay, len(core)))
    for r in replay:
        r["mix"] = "replay"
    for r in new_rows:
        r["mix"] = "new"

    train, holdout = split_by_session(new_rows, HOLDOUT_SHARE, rng)
    train = train + replay
    weighted = []
    for r in train:
        weighted.extend([r] * (HUMAN_WEIGHT if r.get("human") else 1))
    rng.shuffle(weighted)

    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", weighted), ("holdout", holdout)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in ("id", "prompt", "completion", "images", "prompt_version",
                                                       "input_kind", "session", "mix", "human") if k in r},
                                   ensure_ascii=False) + "\n")
    manifest = {
        "run_id": run_id, "created": datetime.now().isoformat(timespec="seconds"),
        "prompt_version": version, "grammar_hash": tags.grammar_hash(),
        "data_range": {"nights": new_nights, "start": new_nights[0], "end": new_nights[-1]},
        "parent": reg.get("current"),
        "counts": {"new": len(new_rows), "replay": len(replay), "train_weighted": len(weighted),
                   "holdout": len(holdout), "human": sum(1 for r in new_rows if r.get("human"))},
        "excluded_eval_sessions": len(banned_sessions),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the nightly training set")
    ap.add_argument("--k", type=int, default=K_NIGHTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id")
    a = ap.parse_args()
    build(a.k, a.seed, a.run_id)
