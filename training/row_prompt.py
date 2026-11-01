"""
training/row_prompt.py — rebuild a logged row's prompt with prompt_format.

The runtime engine's history for row N is the chain of prior rows reached via
history_ref (newest first, up to REACTION_HISTORY_PAIRS). Each prior row
contributes user = its window text (format_window_text of its own fields) and
assistant = its raw_output — exactly what ReactionEngine put in its deque — so
a rebuilt prompt matches what the model saw, except that the CURRENT system
prompt is used (we train on the prompt we serve).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prompt_format          # noqa: E402
from config import REACTION_HISTORY_PAIRS   # noqa: E402

SESSION_GAP_S = 30 * 60   # a gap longer than this between rows starts a new stream session


def window_text(row: dict) -> str:
    return prompt_format.format_window_text(
        row["speech"], row["chat"], len(row["frame_paths"]), row["window_start"], row["window_end"],
    )


def history_for(row: dict, by_id: dict[str, dict], pairs: int = REACTION_HISTORY_PAIRS) -> list[dict]:
    chain, ref = [], row.get("history_ref")
    while ref and len(chain) < pairs:
        prev = by_id.get(ref)
        if prev is None:
            break
        chain.append({"user": window_text(prev), "assistant": prev["raw_output"]})
        ref = prev.get("history_ref")
    return list(reversed(chain))


def turns_for_row(row: dict, by_id: dict[str, dict], system: str | None = None) -> list[prompt_format.Turn]:
    system = system if system is not None else prompt_format.load_system_prompt()
    return prompt_format.build_reaction_messages(
        system, history_for(row, by_id), row["frame_paths"], row["speech"], row["chat"],
        row["window_start"], row["window_end"],
    )


def training_example(row: dict, completion: str, by_id: dict[str, dict], system: str | None = None) -> dict:
    system = system if system is not None else prompt_format.load_system_prompt()
    ex = prompt_format.render_training_example(turns_for_row(row, by_id, system), completion)
    ex["prompt_version"] = prompt_format.prompt_version(system)
    return ex


def session_ids(rows: list[dict], gap_s: float = SESSION_GAP_S) -> dict[str, str]:
    """{row_id: session_id}; a session is a run of rows with no gap > gap_s."""
    out, last_ts, current = {}, None, None
    for r in sorted(rows, key=lambda r: r["ts"]):
        if last_ts is None or r["ts"] - last_ts > gap_s:
            current = f"s-{r['id'][:17]}"
        out[r["id"]] = current
        last_ts = r["ts"]
    return out
