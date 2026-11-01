"""
training/eval_gate.py — promote a candidate adapter only if it holds up.

Fixed eval set (training/data/eval_set.jsonl): ~50 held-out windows across
input kinds, built ONCE from sessions that no promoted adapter trained on.
dataset_builder excludes these sessions forever.

Metrics (candidate and baseline = current adapter or base_core, same run, same judge):
  tag_validity     share of outputs with no unknown/malformed/leaked tag flags
  beat_grammar     share passing reviewer.grammar_issue (beats at clause boundaries, chunks ≥3 words, ≤2 beats)
  leading_only     share of tagged outputs whose tags are all leading
  assistant_rate   share of outputs hitting reviewer.ASSISTANT_PATTERNS
  judge_<score>    Claude rubric means (persona/RUBRIC.md), same as the reviewer
  recognition      share of outputs the judge scores persona ≥ 4 ("recognisably Rachel")
  core_pass        pass rate on training/data/core_prompts.jsonl regression probes

Promote iff no metric regresses beyond TOLERANCE vs baseline AND all FLOORS
hold; otherwise keep the previous adapter. Report → adapters/<version>/eval_report.json.

Generation runs on a LOCAL llama-server (EVAL_LLM_URL, default :8081) started with
  llama-server -m <BASE_CORE_GGUF> --mmproj <mmproj> --lora <current.gguf> --lora <candidate.gguf> \
               --lora-init-without-apply --port 8081
— never the live HF endpoint.

Run:
  python -m training.eval_gate build-set [--n 50] [--force]
  python -m training.eval_gate run <candidate_version> [--no-judge] [--dry-run]
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import requests

SRC = Path(__file__).resolve().parent.parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import prompt_format                              # noqa: E402
import tag_parser                                 # noqa: E402
import tags                                       # noqa: E402
from config import REACTION_MAX_TOKENS, REACTION_SAMPLING, ROWS_DIR, TRAINING_DATA_STATIC   # noqa: E402
from training import registry                     # noqa: E402
from training import reviewer                     # noqa: E402
from training import row_prompt                   # noqa: E402
from training import training_logger as tl        # noqa: E402

EVAL_SET_PATH     = TRAINING_DATA_STATIC / "eval_set.jsonl"
CORE_PROMPTS_PATH = TRAINING_DATA_STATIC / "core_prompts.jsonl"
EVAL_LLM_URL      = os.getenv("EVAL_LLM_URL", "http://localhost:8081")

BAD_TAG_FLAG_RE = re.compile(r"^(unknown_tag|unknown_vocal|malformed|leak|act_cap|dup_in_group)")

# Higher-is-better metrics regress when they drop; lower-is-better when they rise.
HIGHER = {"tag_validity", "beat_grammar", "recognition", "core_pass", *(f"judge_{k}" for k in reviewer.SCORE_KEYS)}
LOWER  = {"leading_only", "assistant_rate"}
TOLERANCE = {"tag_validity": 0.02, "beat_grammar": 0.03, "leading_only": 0.05, "assistant_rate": 0.02, "recognition": 0.05,
             "core_pass": 0.0, **{f"judge_{k}": 0.15 for k in reviewer.SCORE_KEYS}}
FLOORS = {"tag_validity": (">=", 0.95), "assistant_rate": ("<=", 0.05), "leading_only": ("<=", 0.40),
          "judge_persona": (">=", 3.5), "judge_non_assistant": (">=", 3.8), "core_pass": (">=", 0.9)}


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ─── Eval set ─────────────────────────────────────────────

def build_set(n: int = 50, force: bool = False, seed: int = 0, out: Path = EVAL_SET_PATH,
              rows_dir: Path = ROWS_DIR) -> list[dict]:
    if out.exists() and not force:
        raise SystemExit(f"{out} exists — the eval set is fixed. Use --force only if you mean to reset it.")
    rows = [r for r in tl.load_effective(rows_dir) if r["status"] in ("accepted", "reviewed") and r["frame_paths"]]
    by_id = {r["id"]: r for r in tl.load_effective(rows_dir)}
    sessions = row_prompt.session_ids(list(by_id.values()))
    reg = registry.load()
    trained_nights = {n for a in reg["adapters"] if a.get("status") == "promoted"
                      for n in (a.get("data_range") or {}).get("nights", [])}
    pool = [r for r in rows if r["id"][:10] not in trained_nights]
    rng = random.Random(seed)
    by_session = defaultdict(list)
    for r in pool:
        by_session[sessions[r["id"]]].append(r)
    order = sorted(by_session)
    rng.shuffle(order)
    # Take whole sessions until each input kind has its share.
    kinds = ["frames", "frames+speech", "frames+chat", "frames+chat+speech"]
    per_kind = {k: 0 for k in kinds}
    chosen = []
    for s in order:
        if len(chosen) >= n:
            break
        for r in by_session[s]:
            if per_kind[r["input_kind"]] < n // len(kinds) + 1 and len(chosen) < n:
                chosen.append(r)
                per_kind[r["input_kind"]] += 1
    items = [{
        "id": r["id"], "session": sessions[r["id"]], "input_kind": r["input_kind"],
        "row": {k: r[k] for k in ("frame_paths", "speech", "chat", "window_start", "window_end")},
        "history": row_prompt.history_for(r, by_id),
        "reference": r.get("human_fix") or r["raw_output"],
    } for r in chosen]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"eval set: {len(items)} windows from {len({i['session'] for i in items})} sessions  {per_kind}")
    return items


# ─── Generation (local llama-server only) ─────────────────

class LlamaCppGenerator:
    """Exact-prompt generation with a chosen LoRA applied per request."""

    def __init__(self, url: str = EVAL_LLM_URL):
        if "huggingface" in url or "endpoints" in url:
            raise SystemExit("eval_gate must not run on the live HF endpoint")
        self.url = url.rstrip("/")
        self._ids = {Path(a.get("path", "")).name: a["id"] for a in requests.get(f"{self.url}/lora-adapters", timeout=5).json()}

    def lora_for(self, version: str | None) -> list[dict]:
        """Scale 1 for this adapter's GGUF, 0 for every other loaded adapter (None = base_core)."""
        reg = registry.load()
        a = registry.get(reg, version)
        target = Path(a["gguf"]).name if a and a.get("gguf") else None
        if version and target not in self._ids:
            raise SystemExit(f"{version}'s GGUF ({target}) is not loaded in {self.url}; start llama-server with --lora for it")
        return [{"id": i, "scale": 1.0 if name == target else 0.0} for name, i in self._ids.items()]

    def __call__(self, version: str | None, turns: list) -> str:
        body = prompt_format.to_llamacpp_completion(prompt_format.render_inference(turns), ROOT,
                                                    prompt_format.llamacpp_media_marker(self.url))
        body.update({"n_predict": REACTION_MAX_TOKENS, "stream": False, "stop": [prompt_format.EOT],
                     "lora": self.lora_for(version), "seed": 0, **REACTION_SAMPLING})
        r = requests.post(f"{self.url}/completion", json=body, timeout=120)
        r.raise_for_status()
        return r.json().get("content", "")


# ─── Metrics ──────────────────────────────────────────────

def eval_turns(item: dict, system: str) -> list:
    r = item["row"]
    return prompt_format.build_reaction_messages(
        system, item.get("history", []), r["frame_paths"], r["speech"], r["chat"], r["window_start"], r["window_end"],
    )


def core_turns(probe: dict, system: str) -> list:
    t = probe.get("t0", 0.0)
    speech = [{"t": t + 2, "text": probe["ben"]}] if probe.get("ben") else []
    chat = [{"t": t + 3, "user": probe.get("user", "viewer"), "text": probe["chat"]}] if probe.get("chat") else []
    return prompt_format.build_reaction_messages(system, [], [], speech, chat, t, t + 10)


def core_pass(probe: dict, output: str) -> bool:
    text = tag_parser.parse(output).text
    if not text or reviewer.assistant_hits(text):
        return False
    if any(re.search(p, text, re.IGNORECASE) for p in probe.get("must_not", [])):
        return False
    return all(re.search(p, text, re.IGNORECASE) for p in probe.get("must", []))


def text_metrics(outputs: list[str]) -> dict:
    parsed = [tag_parser.parse(o) for o in outputs]
    n = max(len(outputs), 1)
    tagged = [o for o in outputs if tags.find_tags(o)]
    return {
        "tag_validity": round(sum(not any(BAD_TAG_FLAG_RE.match(f) for f in p.flags) for p in parsed) / n, 4),
        "leading_only": round(sum(tags.is_leading_only(o) for o in tagged) / max(len(tagged), 1), 4),
        "assistant_rate": round(sum(bool(reviewer.assistant_hits(p.text)) for p in parsed) / n, 4),
        "beat_grammar": round(sum(reviewer.grammar_issue(p) is None for p in parsed) / n, 4),
    }


def judge_metrics(items: list[dict], outputs: list[str], judge, system_text: str) -> dict:
    scores = defaultdict(list)
    for it, out in zip(items, outputs):
        row = {**it["row"], "id": it["id"], "input_kind": it["input_kind"], "raw_output": out}
        try:
            rev = judge(row, system_text, "")
        except Exception as e:
            print(f"  judge error {it['id']}: {e}")
            continue
        for k in reviewer.SCORE_KEYS:
            if isinstance(rev.get("scores", {}).get(k), (int, float)):
                scores[k].append(rev["scores"][k])
    m = {f"judge_{k}": round(sum(v) / len(v), 3) for k, v in scores.items() if v}
    if scores.get("persona"):
        m["recognition"] = round(sum(s >= 4 for s in scores["persona"]) / len(scores["persona"]), 4)
    return m


def evaluate(version: str | None, items: list[dict], probes: list[dict], generate, judge=None) -> dict:
    system = prompt_format.load_system_prompt()
    outputs = [generate(version, eval_turns(it, system)) for it in items]
    m = text_metrics(outputs)
    if probes:
        m["core_pass"] = round(sum(core_pass(p, generate(version, core_turns(p, system))) for p in probes) / len(probes), 4)
    if judge:
        m.update(judge_metrics(items, outputs, judge, reviewer.persona_system()))
    return {"metrics": m, "samples": [{"id": it["id"], "output": o} for it, o in zip(items[:10], outputs[:10])]}


def gate(candidate: dict, baseline: dict) -> tuple[bool, list[str]]:
    reasons = []
    for k, v in candidate.items():
        b = baseline.get(k)
        if b is None:
            continue
        tol = TOLERANCE.get(k, 0.0)
        if k in HIGHER and v < b - tol:
            reasons.append(f"{k} regressed {b} → {v} (tol {tol})")
        if k in LOWER and v > b + tol:
            reasons.append(f"{k} regressed {b} → {v} (tol {tol})")
    for k, (op, floor) in FLOORS.items():
        v = candidate.get(k)
        if v is None:
            continue
        if (op == ">=" and v < floor) or (op == "<=" and v > floor):
            reasons.append(f"{k}={v} misses floor {op} {floor}")
    return not reasons, reasons


def run(version: str, generate=None, judge=None, use_judge: bool = True, dry_run: bool = False,
        registry_path: Path = registry.REGISTRY_PATH) -> dict:
    reg = registry.load(registry_path)
    cand = registry.get(reg, version)
    if cand is None:
        raise SystemExit(f"unknown adapter {version}")
    items = _read(EVAL_SET_PATH)
    if not items:
        raise SystemExit("no eval set — run `python -m training.eval_gate build-set` first")
    probes = _read(CORE_PROMPTS_PATH)
    generate = generate or LlamaCppGenerator()
    judge = judge or (reviewer.review_api if use_judge else None)
    baseline_version = reg.get("current")

    base_res = evaluate(baseline_version, items, probes, generate, judge)
    cand_res = evaluate(version, items, probes, generate, judge)
    ok, reasons = gate(cand_res["metrics"], base_res["metrics"])
    report = {"candidate": version, "baseline": baseline_version, "promote": ok, "reasons": reasons,
              "candidate_metrics": cand_res["metrics"], "baseline_metrics": base_res["metrics"],
              "candidate_samples": cand_res["samples"], "baseline_samples": base_res["samples"],
              "grammar_hash": tags.grammar_hash(),
              "prompt_version": prompt_format.prompt_version(prompt_format.load_system_prompt())}
    if not dry_run:
        out_dir = ROOT / "adapters" / version
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if ok:
            registry.set_status(version, "candidate", metrics=cand_res["metrics"], path=registry_path)
            registry.promote(version, path=registry_path)
        else:
            registry.set_status(version, "rejected", metrics=cand_res["metrics"], path=registry_path)
    print(json.dumps({k: report[k] for k in ("candidate", "baseline", "promote", "reasons",
                                              "candidate_metrics", "baseline_metrics")}, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Eval gate for nightly adapters")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-set"); b.add_argument("--n", type=int, default=50); b.add_argument("--force", action="store_true")
    r = sub.add_parser("run"); r.add_argument("version"); r.add_argument("--no-judge", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build-set":
        build_set(a.n, a.force)
    else:
        run(a.version, use_judge=not a.no_judge, dry_run=a.dry_run)
