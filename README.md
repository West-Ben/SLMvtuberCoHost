# Rachel — AI VTuber Co-Host

Rachel is an AI co-host for Twitch. Every ~10 seconds she looks at the stream
(raw OBS frames), hears what the streamer said, and reads chat. She answers in
**one** call to a local Gemma 4 vision model. Her reply is spoken with
Chatterbox TTS and drives her avatar in Warudo through inline tags like
`{expr:joy}` and `{anim:laugh}`. Every reaction is logged. A nightly pipeline
reviews the logs and fine-tunes a new LoRA adapter, and the adapter is promoted
only if it passes an evaluation gate.

---

## Project layout

```
start.py            ← run this: opens the control panel (web UI)
requirements.txt    Python dependencies
.env.example        settings template, copy to .env
persona/            who Rachel is: SOUL.md, STANCES.md, SYSTEM_PROMPT.md, RUBRIC.md
workspace/          persona files for the legacy text path
topics.json         conversation starters for quiet moments
src/                all source code
│ ├── control_panel/  web UI: starts/watches services, settings, data explorer, training jobs
│ ├── server.py       TTS + avatar + memory            (ports 8050 MCP, 8051 HTTP)
│ ├── obs_bridge.py   OBS capture, chat/events, audio  (port 8055)
│ ├── orchestrator.py reaction loop                    (port 8052)
│ ├── stt-server.py   mic → voice activity → speech-to-text (port 8053)
│ ├── twitch_listener.py  Twitch EventSub → obs_bridge
│ ├── twitch-auth.py  one-time Twitch login, writes tokens to .env
│ ├── preflight.py    start-of-stream checks
│ ├── tags.py, prompt_format.py, tag_parser.py, playback.py   core grammar/prompt/playback
│ └── training/       nightly review → dataset → train → eval gate → adapter registry
tests/              pytest suite
```

Runtime data is created at the project root on first use: `rows/`, `curated/`,
`dpo/`, `datasets/`, `imagelog/`, `logs/`, `adapters/`, `rachel_memory_db/`.
None of it is committed.

---

## 1. Requirements

### Hardware
- Windows 10/11 (the audio stack uses WASAPI)
- NVIDIA GPU with 12 GB+ VRAM recommended. Chatterbox TTS and the Gemma 4 E4B
  model both run on it. Nightly training was built on an RTX 4080.

### Third-party software (install separately)

| Software | Used for | Notes |
|----------|----------|-------|
| [Python 3.11](https://www.python.org/downloads/) | everything | 3.11 is what the project is tested on |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | runs the model (`llama-server`, port 8080) | build with CUDA; set `LLAMACPP_DIR` |
| Gemma 4 E4B GGUF + its `mmproj` file | the base model Rachel runs on | set `BASE_CORE_GGUF` and `BASE_CORE_MMPROJ` |
| [OBS Studio](https://obsproject.com/) | screen capture + stream state | enable **Tools → WebSocket Server** (port 4455) |
| [Warudo](https://warudo.app/) | the avatar | WebSocket control on port 19190 |
| [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) | routes TTS audio into Warudo for lip sync | |
| An OpenAI-compatible embedding server (e.g. LM Studio) | long-term memory | set `EMBED_URL` / `EMBED_MODEL` |
| A Twitch developer app | chat and events | [dev.twitch.tv/console](https://dev.twitch.tv/console), redirect URL `http://localhost:17563` |

Optional:
- A Hugging Face Inference Endpoint can replace the local model (`HF_URL`,
  `HF_TOKEN`, `HF_MODEL`; enable `--direct` on the orchestrator).
- An Anthropic API key (`ANTHROPIC_API_KEY`) is used by the nightly reviewer and
  the eval judge.
- A voice reference clip (`VOICE_REF_PATH`) clones a voice. Without one,
  Chatterbox's built-in voice is used.

---

## 2. Setup

```powershell
git clone https://github.com/West-Ben/SLMvtuberCoHost.git
cd SLMvtuberCoHost

python -m venv .venv
.venv\Scripts\activate

# PyTorch first, matching your CUDA version (this is CUDA 12.1):
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

copy .env.example .env
```

The training packages (`unsloth`, `trl`, `peft`, `datasets`) are only needed for
the nightly fine-tune. You can comment them out of `requirements.txt` if you
only want to stream.

### Configure `.env`

Open `.env` and fill in at least:

| Key | What to put |
|-----|-------------|
| `LLAMACPP_DIR` | folder of your llama.cpp build |
| `BASE_CORE_GGUF`, `BASE_CORE_MMPROJ` | paths to the Gemma 4 E4B `.gguf` and `mmproj` files |
| `OBS_PASSWORD` | OBS WebSocket password |
| `TWITCH_CLIENT_ID`, `TWITCH_CLIENT_SECRET`, `TWITCH_CHANNEL` | from your Twitch app |
| `EMBED_URL`, `EMBED_MODEL` | your embedding server |
| `MIC_KEYWORD`, `CABLE_DEVICE_NAME`, `SPEAKER_DEVICE_NAME` | part of your mic / virtual cable / speaker device names |

You can also edit everything later in the control panel's **Settings** page.
Secrets there are write-only and never displayed. The page also shows which
services need a restart after a change.

### One-time Twitch login

```powershell
python src\twitch-auth.py
```

This opens a browser and writes `TWITCH_TOKEN` and `TWITCH_REFRESH_TOKEN` into `.env`.

### Check the install

```powershell
python -m pytest tests/ -q
```

---

## 3. Start

Start OBS and Warudo first, then run:

```powershell
python start.py
```

This opens the control panel at **http://127.0.0.1:8060/**. Options:
`--port 8061`, and `--no-browser` to skip opening a browser tab.

From the panel:

1. Click **▶ Go live**. It starts the services in order:
   llama-server → TTS → OBS bridge → orchestrator → Twitch → STT. Each one waits
   for the one before it to report healthy. Crashed services restart with
   backoff, and services already running on their port are adopted as external.
2. In **Console**, run `preflight` for start-of-stream checks: prompt/grammar
   version, adapter, parser tests, and service reachability.
3. Use **■ Stop stack** to stop everything. `Ctrl+C` in the terminal closes the
   panel and every service it started.

Logs for every service are written to `logs/<service>/<date>.log` and shown
live in the panel.

### Running a service by hand (debugging)

Run these from the project root so runtime data lands in the right place:

```powershell
python src\server.py
python src\obs_bridge.py --test          # --test: act as if the stream is live
python src\orchestrator.py --text "what's going on?" --no-log   # one-shot, printed only
python src\preflight.py --serve-cmd      # print the llama-server command line
```

For `-m` modules, put `src` on the path first: `$env:PYTHONPATH="src"`, then
`python -m training.training_logger stats`.

---

## Nightly training loop (offline)

The control panel's **Dream** page runs these steps as jobs. You can also run
them from a shell with `PYTHONPATH=src`:

| Step | Command |
|------|---------|
| Review yesterday's + today's reactions | `python -m training.reviewer` |
| Build the dataset | `python -m training.dataset_builder` |
| Train a LoRA candidate | `python -m training.train_nightly --run-id <id> --export-gguf` |
| Build the fixed eval set (once) | `python -m training.eval_gate build-set` |
| Gate + promote | `python -m training.eval_gate run <version>` |
| Promote / roll back by hand | `python -m training.registry promote <version>` / `rollback` |

Only reviewed rows are used for training. Viewer chat is treated as untrusted,
and changes to `persona/` are never automatic.
