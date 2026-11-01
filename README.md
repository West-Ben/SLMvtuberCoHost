# Rachel — AI VTuber Co-Host

Snarky AI co-host for Twitch. Reacts to gameplay, chat, and stream events with
genuine, emotion-tagged responses. Runs locally except for the HF inference
endpoint that serves the Rachel V6 model.

> **Adaptive Loop (current):** the reaction path now sends raw frames + speech +
> chat in one call (`REACTION_MODE=frames`), the perceiver is disabled by default
> (`PERCEIVER_ENABLED=false`), and reactions feed a nightly review → adapter loop.
> See `CLAUDE.md` for the current architecture and file layout. The sections
> below describe the legacy text path (`REACTION_MODE=text`), kept for A/B and rollback.

The legacy architecture splits **perception** from **response**: a background perceiver
(inside obs_bridge) turns frames + audio into a plain-text scene description;
Rachel's reaction path consumes only that text, so she responds fast and never
re-processes images or audio per turn.

---

## Stack

| Component | Tech |
|-----------|------|
| LLM | Gemma 4 E4B + Rachel V6 LoRA (emotion-channel fine-tune) via HF Inference Endpoint |
| TTS | Chatterbox Turbo (GPU, voice-cloned) |
| Avatar | Warudo (WebSocket control) |
| STT | Silero VAD + Gemma 4 E4B audio-direct (local) |
| Perception | obs_bridge background loop: N frames + audio → text scene description (thinking off) |
| Memory | ChromaDB + nomic-embed-text (autobiographical, mood-congruent recall) |
| Session | Sliding-window history (20 turns) with summary compression |
| Training data | Live JSONL logger captures every interaction for the next fine-tune |

---

## Architecture (2 core processes + TTS + STT)

```
stt-server.py ──/transcription──▶ orchestrator.py ──/speak──▶ server.py
     │  /audio-context ◀───────────────┐   ▲                      │
     │                                 │   │ GET /description     Warudo
     │                                 │   │ (cached scene text)  VB-Cable
     │                          obs_bridge.py (8055)              speaker
     │                          ├ screenshot/status/event loops
     └──────────────────────────┤ PERCEIVE loop:
                                │   N frames + audio + [ben] + state
                                │   → 1 HF call (thinking off)
                                │   → cached text description
                                └ /context  /description  /ben-spoke
                                       ▲
                          twitch_listener.py ── /twitch-info
```

| Process | File | Port | Role |
|---------|------|------|------|
| TTS + avatar + memory | `server.py` | 8050 (MCP), 8051 (HTTP) | Chatterbox TTS, Warudo, memory API |
| Context + perception | `obs_bridge.py` | 8055 | Screenshots, stream state, **scene perceiver**, chat/events |
| Orchestrator | `orchestrator.py` | 8052 | Session, context assembly, reaction calls, silence timer |
| STT server | `stt-server.py` | 8200 | Mic → VAD → STT → orchestrator; serves rolling audio |
| Twitch listener | `twitch_listener.py` | — | EventSub + viewer polling → obs_bridge |

`vision_perceiver.py` is retired — perception now lives in obs_bridge.

---

## Perception ↔ Response split

**Perceiver (obs_bridge), ~every 2.5 s, thinking OFF:** snapshots a
motion-driven number of recent frames, fetches the matching mic+speaker audio,
adds Ben's recent speech and stream state, and makes **one** HF call that returns
a 2–4 sentence description of what is happening and what was said. Cached.

**Orchestrator reaction, text only:** on a trigger it fetches the cached
description, drops it into the `[VISION]` block, and makes a text-only reaction
call. No images or audio touch the response path.

**Why description-only:** Rachel has her own internal `<|channel>thought`
channel — that voice is the point of the model. The perceiver gives her
situational *text*, not analysis. We never inject an external reasoning trace
into her context, which would flatten her voice and make her react to the
analysis instead of the scene.

### HF concurrency (max 4 calls)
- obs_bridge perceiver: **1** in-flight max (skips a tick if a cycle overruns).
- orchestrator: **BoundedSemaphore(2)** (reaction + at most one summary).
- Worst case 3 concurrent — safely under 4, no cross-process coordination.

---

## Message Array Structure

| Layer | Role | Content | Lifecycle |
|-------|------|---------|-----------|
| 0 | system | SOUL.md + USER.md + output-format rule | Permanent |
| 1 | user | `[Earlier this stream: ...]` | Compressed history, updates on trim |
| 2 | system | `<<STREAM_CONTEXT>>` incl. `[VISION]` cached description | Rebuilt every turn |
| 3 | user/assistant | Recent ~10 turns | Sliding window |
| 4 | user | Current input | What Ben/chat said |

`[VISION]` is the perceiver's cached **text** — reactions never carry raw frames.

---

## Requirements

```
# server.py
torch torchaudio librosa numpy sounddevice chatterbox-tts fastmcp websocket-client httpx pillow

# obs_bridge.py  (now includes perception)
obsws-python flask requests numpy pillow

# orchestrator.py
requests flask openai

# session.py / training_logger.py
(stdlib)

# memory.py
chromadb numpy requests

# twitch_listener.py / twitch-auth.py
requests

# stt-server.py
torch numpy pyaudio soundfile sounddevice silero-vad scipy requests pyrnnoise

# all
python-dotenv
```

| Requirement | Notes |
|-------------|-------|
| NVIDIA GPU (≥ 8 GB) | Chatterbox TTS on CUDA |
| VB-Audio Virtual Cable | TTS → Warudo lip sync |
| Warudo | WebSocket plugin, port 19190 |
| llama-server (Gemma 4 E4B) | Port 8080 — STT (+ optional local reactions) |
| LM Studio embedding server | memory embeddings |
| HF endpoint | Rachel V6, **max 4 concurrent calls** |
| OBS + WebSocket plugin | port 4455 |

---

## Environment Variables

```env
# OBS
OBS_PASSWORD=your_obs_ws_password

# Twitch (run twitch-auth.py to populate tokens)
TWITCH_CLIENT_ID=
TWITCH_CLIENT_SECRET=
TWITCH_TOKEN=
TWITCH_REFRESH_TOKEN=
TWITCH_CHANNEL=squirreldud

# HuggingFace
HF_TOKEN=hf_yourtoken
HF_URL=https://your-endpoint.aws.endpoints.huggingface.cloud/v1
HF_MODEL=BDubEast/rachel-v6

# Overrides (defaults shown)
OPENCLAW_URL=http://127.0.0.1:18789/v1/chat/completions
OBS_BRIDGE_URL=http://localhost:8055
STT_SERVER_URL=http://localhost:8200
WARUDO_WS_URL=ws://localhost:19190
VOICE_REF_PATH=F:\LLM\vtubeModels\rachael_v2.mp3
MIC_KEYWORD=Anker PowerConf C200
EMBED_URL=http://172.17.176.1:1234/v1/embeddings
MEMORY_DB_PATH=./rachel_memory_db
TRAINING_DATA_DIR=./training_data
SESSION_MAX_TURNS=20

# Perception (obs_bridge)
VISION_PERCEIVE_INTERVAL_S=2.5
VISION_FRAMES_MIN=3
VISION_FRAMES_MAX=20
VISION_MOTION_THRESHOLD=8.0
```

---

## Starting the Pipeline

Start each in its own terminal, in order. Wait for the ready line before moving on.

**1. server.py** — TTS + Warudo + memory (~30 s GPU load)
```bat
cd vtuberMCP && python server.py
```
Ready: `Listening on port 8051`

**2. obs_bridge.py** — context + perception (OBS must be running)
```bat
python obs_bridge.py
# offline test (no live audience), custom perception cadence:
python obs_bridge.py --test --perceive-interval 2.5
```
Ready: `OBS bridge listening on port 8055` — then periodic `[scene] (Nf) ...`
perception lines. Screenshot capture must continue uninterrupted alongside them.

**3. orchestrator.py** — reaction brain
```bat
python orchestrator.py
# bypass OpenClaw, hit HF directly:
python orchestrator.py --direct
# one-shot text test (obs_bridge should be running so [VISION] is populated):
python orchestrator.py --direct --no-log --text "what's happening right now"
```
Ready: `Listening on port 8052`

**4. twitch_listener.py** — events (optional)
```bat
python twitch_listener.py
```

**5. stt-server.py** — mic (start last)
```bat
python stt-server.py
python stt-server.py --gain 6            # volume boost
python stt-server.py --eq --gain 6       # Seiren Mini EQ + boost
```
Ready: `Listening…`

**Twitch OAuth** (one-time): fill `TWITCH_CLIENT_ID`/`SECRET` in `.env`, then
`python twitch-auth.py`.

---

## Verifying the perception split

- `curl http://localhost:8055/description` → returns the cached scene text and an
  `age`. Mid-stream `age` should stay under ~2× the perceive interval. If it
  climbs, perception cycles are overrunning — lower `VISION_FRAMES_MAX` or raise
  the interval (fresh-but-shallow beats stale-but-deep for a co-host).
- obs_bridge logs should show perception `[scene]` lines **and** continued
  screenshot/imagelog activity. If capture stalls each cycle, the perception loop
  is holding `_lock` across the HF call — it must snapshot-and-release first.
- A reaction call's message array must contain **zero** image blocks; the
  `[VISION]` block is text only.

---

## Training Data & Dataset Tools

```bat
:: frames path rows (rows/YYYY-MM-DD.jsonl) — see CLAUDE.md for the full loop
python -m training.training_logger stats
python -m training.training_logger mark 2026-10-24_201502_a1b2c3 good
:: legacy text-path logs (training_data/)
python -m training.training_logger legacy-stats
python -m training.training_logger legacy-build-dataset --output rachel-v7.jsonl

# offline: normalize a dataset to canonical Gemma 4 format
python migrate_rachel_dataset.py --in raw.parquet --out clean.parquet --clean --drop-bare-channel

# offline: resharpen logged rows (Gemma 4 format, expression/animation calls, vision frames)
python dataset_regenerator.py --date 2026-05-31 --count 5 --backend local
```

---

## Stopping

`Ctrl+C` in each terminal. Order doesn't matter.