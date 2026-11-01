# Rachel VTuber Pipeline — Project Architecture (Adaptive Loop)

## Overview

Rachel is a persistent AI VTuber co-host for Twitch. Live, she reacts to ~10 s
**windows** of the stream: raw frames + what Ben said + what chat said go into
**one** model call, her reply is parsed for inline avatar tags, spoken, and
logged as a training row. Offline, rows are reviewed (deterministic filter +
Claude), curated into a dataset, and a nightly LoRA is trained and gated before
it is promoted. Every piece is versioned and reversible.

| Process | File | Port | Role |
|---------|------|------|------|
| **TTS + avatar + memory** | `server.py` | 8050 (MCP), 8051 (HTTP) | Chatterbox Turbo TTS, Warudo WebSocket, clip API `/synth` + `/play-clip`, `/avatar`, legacy `/speak`, memory API |
| **Capture** | `obs_bridge.py` | 8055 | OBS screenshots → ring buffer, `GET /window`, stream state, chat/events, audio playback + AEC |
| **Orchestrator** | `orchestrator.py` | 8052 | WindowAccumulator + ReactionEngine (frames path) or legacy text path |
| **STT** | `stt-server.py` | — | Mic → Silero VAD → STT → `/transcription`; posts VAD start/end to `/vad` |
| **Twitch listener** | `twitch_listener.py` | — | EventSub → obs_bridge `/twitch-info`, `/chat-message`, `/twitch-event` |
| **Model** | llama-server (local) or HF endpoint | 8080 / `.env HF_URL` | Gemma 4 E4B base_core (+ nightly LoRA) |
| **Control panel** | `control_panel/` (`python -m control_panel`) | 8060 (127.0.0.1) | Supervises all of the above (start order, restart/backoff, adopts already-running ports as `external`), logs → `logs/<source>/DATE.log`, `.env` settings, data explorer, Dream = training pipeline jobs |

**The perceiver is disabled by default** (`PERCEIVER_ENABLED=false`). Its code
(`perceive_loop` in obs_bridge) is kept, but the reaction path **does not use
`[VISION]` description text** — it sends the raw frames.

## Live data flow (REACTION_MODE=frames)

```
stt-server ──/transcription──▶ ┐
           ──/vad (start/end)─▶│  orchestrator.py
viewer chat ──/chat-input────▶ │   WindowAccumulator  (speech + chat, timestamped)
obs_bridge ──/event (twitch)─▶ ┘        │ flush (policy below)
                                        ▼
                     GET obs_bridge /window?seconds=…&min_frames=4&max_frames=10
                       (snapshot ring under _lock → release → thin → downscale → imagelog/DATE/w-<id>/)
                                        ▼
                     prompt_format.build_reaction_messages(system, history, frames, speech, chat)
                                        ▼  BoundedSemaphore(2)
                     ONE model call (thinking OFF, max_tokens ~120, temp 1.0 / top_p 0.95 / top_k 64)
                                        ▼
                     tag_parser.parse → Plan [Speak | Vocal | Act …]
                                        ▼
                     playback.Sequencer.play(plan) → server.py /synth, /play-clip, /avatar → PlaybackTrace
                                        ▼
                     row (plan + parse_flags + playback_trace) → rows/YYYY-MM-DD.jsonl
```

### Window flush policy (config.py)
- `WINDOW_S=10`: flush once elapsed **and** VAD idle (never cut Ben mid-utterance).
- `WINDOW_MAX_S=14`: hard max — flushes even if VAD still says speaking.
- Direct viewer chat / Twitch event: flush early once `WINDOW_MIN_S` (3 s) has passed.
- First remark into an idle window gets ≥ `WINDOW_MIN_S` of follow-up time.
- Nothing said: the window slides; a **frames-only** flush is the silence path,
  rate-limited by `SILENCE_THRESHOLD` and requiring VAD idle + stream active.
- While a reaction is in flight, TTS is speaking, or the mute lock is set: keep
  accumulating, never start a call. Pending input **merges** into the next window.
- Her own speech is never fed back as `[ben]` (echo filter vs. recent output).
- `input_kind` ∈ `frames | frames+speech | frames+chat | frames+chat+speech`.

## Single sources of truth

- **`tags.py`** — the only tag grammar. `{expr:X}` (joy, happy, anger, angry,
  relaxed, surprised, surprise, concern, sad, sadness), `{anim:X}` (laugh,
  bounce, think, done, agree, greetings, wave, head_tilt, tilt, shrug), vocal
  `[sigh] [groan] [chuckle] [gasp] [laugh] [sniff] [cough] [shush] [clear throat]`,
  `TAG_ALIASES` (`neutral → relaxed`), `MAX_ACT_GROUPS=4`, `MIN_CHUNK_WORDS=3`,
  `MAX_BEATS_PER_REPLY=2`, `CLAUSE_END`, `PLACEMENT_RULE`, `grammar_hash()`. `tests/test_tags.py` fails if another module defines a tag set
  (legacy exemptions: `server.py`, `legacy_text_path.py`, `rachel_tools.py`).
- **`prompt_format.py`** — the only prompt builder, for training **and**
  inference. Gemma 4 turns built explicitly (never `apply_chat_template`),
  thinking off ⇒ generation prefix `<|turn>model\n<|channel>thought\n<channel|>`,
  images (`<|image|>`) before text in the current user turn only, history =
  prior window text + prior reply text (no images, no thoughts), untrusted text
  sanitized of special tokens, zero custom special tokens. `prompt_version()` =
  `rx1+<hash of system text>`. Train and inference prompts are asserted
  byte-identical in `tests/test_prompt_format.py`.
- **`persona/`** — `SOUL.md` (identity/register/hard rules), `STANCES.md` (fixed
  opinions), `SYSTEM_PROMPT.md` (short runtime prompt, `{TAG_GRAMMAR}` filled from
  tags.py), `RUBRIC.md` (reviewer + eval judge), `ledger_pending.jsonl` (opinion
  proposals; Ben merges into STANCES.md by hand — never automatic, never from chat).

## Response Parser + Playback Sequencer (spec — supersedes the old SEGMENT_MODE/dispatch design)

### Core model — `tag_parser.py` (pure, never raises)
`parse(raw) → Plan(steps, flags)`, steps in reply order:
- `Speak(text)` — plain text between tags → one TTS clip.
- `Vocal(tag)` — `[sigh] [groan] [chuckle] [gasp] [laugh] [sniff] [cough] [shush] [clear throat]` → one TTS clip.
- `Act(expr?, anim?)` — adjacent `{expr:X}`/`{anim:X}` (whitespace only between);
  max one of each per group, a duplicate overrides (`dup_in_group:<kind>`).

An `Act` **binds** to the `Vocal`(s) directly after it → an **action beat**; an
`Act` with no vocal after it is a beat with no clip. Whitespace-only text makes
no `Speak`; spaces collapse; punctuation stays with its text (a punctuation-only
chunk after a group joins the preceding speech). Unknown tag → `TAG_ALIASES` →
else dropped (`unknown_tag:X`); bad braces → `malformed`; leaked `<|channel>`,
`<|turn>`, `<|tool_call>` → `leak`; unknown `[..]` → `unknown_vocal:X`; >4 groups
→ extras dropped (`act_cap:N`). No `Speak` and no `Vocal` → no-op (`empty`). The
parser never inserts reset steps. `Plan.to_dict()` is stored on the row (`plan`,
`parse_flags`).

### Timing — `playback.py` `Sequencer.play(plan) → PlaybackTrace`
`LEAD_S = 1.0`. Clip durations come from synthesized audio (samples / sample rate).
1. Clips play strictly in order, one at a time — no overlapping audio.
2. A leading `Speak` plays first, with no action.
3. When the previous clip ends at `T` and the next item is a beat: send the
   `Act` (expr + anim together) to Warudo AND start the bound vocal(s) at `T`;
   several adjacent vocals play back to back, the group fires with the first.
4. The `Speak` after a beat starts at `max(T + LEAD_S, end of bound vocals)` —
   a vocal longer than 1 s delays speech until it ends; the lead also applies
   after a beat with no vocal. Unbound vocal runs play back to back, no gap.
5. The reply completes when the last clip finishes.
6. A reply starting with a beat fires at `T = 0` with its vocal.
7. A trailing beat fires at the end of the last clip; the reply completes
   immediately (or after its vocal).
   (Implementation choice: a beat directly after another beat waits for the first one's lead.)

Worked examples (test vectors in `tests/test_tag_parser.py` / `tests/test_playback.py`):
`Ben you need to use your alt, {anim:shrug}{expr:sad} [sigh] he never listens.` —
A plays at 0; at end(A)=T send shrug+sad and play `[sigh]`; "he never
listens." at `max(T+1.0, end(sigh))`. `{anim:laugh}[laugh] that was a great move!` —
laugh+`[laugh]` at 0, speech at `max(1.0, end(laugh))`.

Sequencer contract: all clips are submitted for synthesis in order as soon as the
plan exists (`Presynth`, one worker); clip 1 plays the moment it is ready
(`first_clip_ms`); a later clip not ready when due is waited for (`underrun_ms`,
flag `underrun`) and timing continues from actual times; a beat waits for its
first bound vocal so face and sound stay simultaneous. A clip that fails to
synthesize is skipped (`clip_failed`) and its beat still fires. Monotonic clock;
clip end = return of server.py `/play-clip` (fallback: start + duration). One
reply at a time (asserted). Trace entries `{step, scheduled_t, actual_t,
clip_duration, …}` go on the row as `playback_trace`. Expression reset: after a
reply that set an expression, send `EXPR_RESET` (`relaxed`) after
`EXPR_RESET_DELAY_S` (default 0); a new reply starting first cancels it;
animations never reset. **Note:** server.py's own expression list
(`EXPRESSION_EXAGGERATION`) has no `relaxed` — Warudo's blueprint must map it.

server.py clip API: `POST /synth {"text"}` → `{clip_id, duration_s, samples,
sample_rate}` (text is a Speak chunk or a vocal tag alone, e.g. `"[sigh]"`);
`POST /play-clip {"clip_id"}` → returns when obs_bridge `/play` has finished;
`POST /avatar {"expression"?, "animation"?}` → Warudo (two messages, same instant).
obs_bridge `/play` returns at actual playback end (the old extra
`sleep(dur + 0.25)` that doubled every clip was removed).

### Grammar for training data (reviewer Stage A, eval `beat_grammar`)
- Action groups only at clause boundaries — start, end, or after `, . ! ? ; : …`
  — never mid-phrase (`tag_mid_phrase`).
- Once a reply is split by beats/vocals, each speech chunk is a complete phrase
  of ≥3 words (`chunk_too_short`).
- `{anim}` and `{expr}` side by side when meant as one beat; `[vocal]` right
  after the group to sound with it; ≤2 beats per reply (`too_many_beats`).
- These three go to Stage B with a note (rejected outright under
  `--stage-a-only`); the row is accepted only via a `minor_edit` that fixes them
  (e.g. moving the group to the nearest clause boundary). Edits must pass the
  same checks.
- Smoke test (`tests/test_tts_smoke.py`, also in preflight): every vocal tag
  synthesized alone gives a non-empty clip.

## Concurrency
- Reaction = 1 in-flight call, + 1 legacy session summary → `BoundedSemaphore(2)`.
- **Never hold obs_bridge `_lock` across a model call** — `/window` copies ring
  references under the lock, releases, then thins/downscales/saves.
- Playback, loopback capture and AEC run on their own obs_bridge threads,
  independent of `screenshot_loop` (and of the disabled `perceive_loop`).
- server.py uses `ThreadingHTTPServer` so a blocking `/play-clip` doesn't stall
  `/synth` (pre-synthesis of the next clip), `/avatar` or the memory API.

## Transport and serving
- `REACTION_TRANSPORT`: `completion` = exact prompt via llama.cpp `/completion`
  (`prompt_string` + `multimodal_data`, `<|image|>` → `<__media__>`, leading
  `<bos>` dropped because the server adds it). `chat` = `/v1/chat/completions`
  fallback (server templates it — byte-identity NOT guaranteed). Empty = auto:
  `chat` with `--direct` (HF has no `/completion`), `completion` locally.
- llama.cpp: base_core (`BASE_CORE_GGUF` + mmproj) with every registered adapter
  loaded via `--lora … --lora-init-without-apply`. The installed build (Apr 2026)
  supports per-request `"lora"` scales and `POST /lora-adapters`, so switching
  between **loaded** adapters needs no restart; a **new** adapter file does.
  `python preflight.py --serve-cmd` prints the command.
- The orchestrator reads `adapters/registry.json` `current` per reaction, maps
  it to the loaded adapter by GGUF filename, and logs what was actually applied
  (`adapter_version`: version, `base_core`, or `chat:<model>`) in every row.
- `BASE_CORE_HF` (training) and `BASE_CORE_GGUF` (serving) must be the same weights.

## Training loop (offline — never on the live HF endpoint)

1. **Rows** — `training/training_logger.py` `ReactionRowLogger` → `rows/YYYY-MM-DD.jsonl`,
   pydantic `ReactionRow` (schema_version 2, window bounds, input_kind,
   frame_paths, speech, chat, history_ref, prompt_version, adapter_version,
   sampling, latency_ms, raw_output, plan, parse_flags, playback_trace, review,
   human, status). The row is written after playback so it carries the trace. Rows are **append-only**: human marks → `rows/_marks.jsonl`,
   reviews → `rows/_review/DATE.jsonl`, merged on read. Human marks override
   reviewer verdicts; `unsafe`/`toxic_chat` is always `excluded`.
   CLI: `python -m training.training_logger mark|review-queue|show|stats|prune`.
   Frames referenced by any non-rejected row are never deleted; `prune --days N
   --apply` removes frames used only by rejected rows.
2. **Review** — `python -m training.reviewer` (nightly; default yesterday+today).
   Stage A deterministic (no_frames, parse_invalid, assistant, length,
   multi_question, echo, repetitive, too_many_beats, tag_mid_phrase,
   chunk_too_short; batch leading-only share). Stage B Claude
   (`claude-opus-5-5`, structured JSON, frames attached; `--backend cli` uses
   `claude -p`). Edit rules enforced in code (≥0.75 token similarity, no help
   offers, no new content, valid tags) — violations downgrade to reject.
   Outputs `curated/DATE.jsonl` (≤150/night, balanced by input_kind, near-dupes
   removed), `dpo/DATE.jsonl`, `persona/ledger_pending.jsonl`.
3. **Dataset** — `python -m training.dataset_builder`: new curated nights since
   the current adapter (else K=3 latest) + ~50% replay from core
   (`training/data/core.jsonl` + nights already in promoted adapters) + human
   rows ×2. Re-checks effective status, rebuilds stale `prompt_version` prompts,
   excludes eval-set sessions, splits train/holdout **by session**.
4. **Train** — `python -m training.train_nightly --run-id <id> [--check] [--export-gguf]`:
   Unsloth FastVisionModel QLoRA 4-bit E4B, language layers only, r=16=alpha,
   1 epoch, LR 2e-5…5e-5, batch 1 + grad accum, `use_gradient_checkpointing="unsloth"`,
   loss on the response only (Gemma 4 markers), SFTTrainer or `--trainer hf`.
   Registers a `candidate` in the registry. Same script runs in Colab.
5. **Gate** — `python -m training.eval_gate build-set` (once) / `run <version>`:
   fixed ~50-window eval set + `training/data/core_prompts.jsonl` probes, on a
   local llama-server (`EVAL_LLM_URL`, :8081). Tag validity, leading-only share,
   assistant rate, Claude rubric scores, recognition, core pass. Promotes only if
   nothing regresses beyond tolerance and floors hold; else keeps the previous
   adapter. Report → `adapters/<version>/eval_report.json`.
6. **Registry** — `adapters/registry.json` `{adapters: [...], current}`;
   promote / rollback = one pointer move (`python -m training.registry …`).

## preflight.py (start of stream)
Prompt version, grammar hash, current adapter (and whether it was trained
under the same grammar/prompt), parser test suite, llama-server health +
adapter actually loaded, obs_bridge/TTS reachability, and one dry reaction
(printed, not spoken/logged). Exit 0 = PASS.

## File layout

```
vtuberMCP/
├── persona/            SOUL.md, STANCES.md, SYSTEM_PROMPT.md, RUBRIC.md, ledger_pending.jsonl
├── tags.py             tag grammar (imported everywhere)
├── prompt_format.py    prompt builder (train + inference)
├── tag_parser.py       parse() → Plan of Speak / Vocal / Act steps
├── playback.py         Sequencer: pre-synthesis, beat timing, expression reset → PlaybackTrace
├── orchestrator.py     WindowAccumulator + ReactionEngine; legacy text path
├── frame_window.py     /window snapshot / motion thinning / downscale (pure, tested)
├── legacy_text_path.py parsing helpers for REACTION_MODE=text only
├── preflight.py        serve_argv()/serve_cmd(port) build the llama-server command
├── audio_devices.py    mic ranking shared by stt-server.py and the control panel
├── control_panel/      FastAPI + static UI: supervisor, jobs, settings, data_api, dream (training pipeline)
├── obs_bridge.py  server.py  stt-server.py  twitch_listener.py  session.py  memory.py  config.py
├── training/
│   ├── training_logger.py  reviewer.py  dataset_builder.py  train_nightly.py  eval_gate.py
│   ├── row_prompt.py       rebuild a row's exact prompt (history via history_ref)
│   ├── registry.py         adapters/registry.json helpers
│   └── data/               core_prompts.jsonl, (eval_set.jsonl, core.jsonl)
├── adapters/registry.json  (+ adapters/<version>/ lora, <version>.gguf, eval_report.json)
├── rows/  curated/  dpo/  datasets/  imagelog/
├── tests/              test_tags, test_prompt_format, test_tag_parser, test_playback, test_tts_smoke,
│                       test_orchestrator_window, test_training_logger, test_reviewer,
│                       test_training_pipeline, test_preflight
├── workspace/          legacy SOUL.md/USER.md for the text path
└── backups/pre-adaptive-loop/   pre-change copies of modified files
```

Run tests: `python -m pytest tests/ -q`.

## Config flags (config.py / .env)
`REACTION_MODE` (frames|text), `LEAD_S` (1.0), `EXPR_RESET` (relaxed),
`EXPR_RESET_DELAY_S` (0), `REACTION_TRANSPORT`
(completion|chat|auto), `PERCEIVER_ENABLED` (false), `WINDOW_*`, `VAD_STALE_S`,
`REACTION_MAX_TOKENS`, `REACTION_TEMPERATURE/TOP_P/TOP_K`, `BASE_CORE_HF`,
`BASE_CORE_GGUF`, `BASE_CORE_MMPROJ`, `LLAMACPP_DIR`, `EVAL_LLM_URL`.
Orchestrator CLI: `--mode frames|text`, `--direct`, `--text "..."` (one-shot,
printed not spoken), `--no-log`.

## Open-decision defaults (change only with a comment)
Window 10 s, flush on VAD idle, hard max 14 s, 4–10 frames · `{expr:neutral}` →
`relaxed` · `LEAD_S=1.0`, vocal longer than the lead delays speech, lead after
every beat, reset to `relaxed` at reply end · nightly on the local 4080, QLoRA E4B (12B
deferred) · perceiver disabled, kept in code · opinion ledger = proposals only,
Ben approves · DPO pairs are collected, DPO training only after the SFT loop is stable.

## Guardrails
- Only `tags.py` / `prompt_format.py` define tag / prompt grammar.
- Never hold `_lock` across a model call.
- The reaction path never raises on malformed model output.
- Reviewer, eval and training never touch the live HF endpoint.
- Never retrain from unreviewed rows; viewer chat is untrusted (toxic/unsafe
  excluded; chat never edits persona files or the ledger automatically).
- Marvel Rivals first; other games reuse the loop once it is stable.

## Legacy text path (REACTION_MODE=text)
The previous architecture, kept for A/B and rollback: `SessionManager` layered
messages with the perceiver's cached `[VISION]` description (requires
`PERCEIVER_ENABLED=true`), `workspace/SOUL.md` + `USER.md` system prompt, legacy
`TrainingLogger` → `training_data/`. Its parsing helpers live in `legacy_text_path.py`.
