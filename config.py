"""
Shared configuration for the Rachel V5 pipeline.
All processes import from here — change once, applies everywhere.
"""

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# ─── Ports ────────────────────────────────────────────────
ORCHESTRATOR_PORT = 8052    # orchestrator.py HTTP server
OBS_BRIDGE_PORT   = 8055    # obs_bridge.py HTTP server
TTS_SPEAK_PORT    = 8051    # server.py /speak endpoint
MCP_PORT          = 8050    # server.py FastMCP
STT_SERVER_PORT   = 8053    # stt-server.py audio-context endpoint

# ─── Service URLs ─────────────────────────────────────────
ORCHESTRATOR_URL = f"http://localhost:{ORCHESTRATOR_PORT}"
STT_SERVER_URL   = f"http://localhost:{STT_SERVER_PORT}"
OBS_BRIDGE_URL   = f"http://localhost:{OBS_BRIDGE_PORT}"
SPEAK_URL        = f"http://localhost:{TTS_SPEAK_PORT}/speak"

LLM_URL           = os.getenv("LLM_URL",   "http://localhost:8080/v1/chat/completions")
LLM_TOKEN         = os.getenv("LLM_TOKEN",  "local")
LLM_MODEL         = os.getenv("LLM_MODEL",  "")

HF_URL            = os.getenv("HF_URL",   "")
HF_TOKEN          = os.getenv("HF_TOKEN", "")
HF_MODEL          = os.getenv("HF_MODEL", "")

CABLE_DEVICE_NAME   = os.getenv("CABLE_DEVICE_NAME",   "CABLE Input")   # VB-Audio → Warudo lip sync
SPEAKER_DEVICE_NAME = os.getenv("SPEAKER_DEVICE_NAME", "")              # "" = system default output
TTS_DEVICE          = os.getenv("TTS_DEVICE",          "cuda")
TTS_VOLUME          = float(os.getenv("TTS_VOLUME",    "2.0"))           # linear gain applied to TTS output (1.0 = unity)

VISION_URL        = os.getenv("VISION_URL",     "http://localhost:8080/v1/chat/completions")
VISION_MODEL      = os.getenv("VISION_MODEL",   "gemma-4-e4b")

WARUDO_WS_URL     = os.getenv("WARUDO_WS_URL",  "ws://localhost:19190")

# ─── OBS WebSocket ────────────────────────────────────────
OBS_HOST          = os.getenv("OBS_HOST",       "localhost")
OBS_PORT          = int(os.getenv("OBS_PORT",   "4455"))
OBS_PASSWORD      = os.getenv("OBS_PASSWORD",   "")

# ─── Twitch ───────────────────────────────────────────────
TWITCH_CLIENT_ID     = os.getenv("TWITCH_CLIENT_ID",     "")
TWITCH_CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET", "")
TWITCH_NICK          = os.getenv("TWITCH_NICK",          "")
TWITCH_TOKEN         = os.getenv("TWITCH_TOKEN",         "")   # oauth:xxxxx
TWITCH_REFRESH_TOKEN = os.getenv("TWITCH_REFRESH_TOKEN", "")
TWITCH_CHANNEL       = os.getenv("TWITCH_CHANNEL",       "")   # channel name without #

# ─── Microphone ───────────────────────────────────────────
MIC_KEYWORD       = os.getenv("MIC_KEYWORD",    "Razer Seiren Mini")
SAMPLE_RATE       = 16000
CHUNK_SIZE        = 512
DEVICE_RATE       = 48000
DEVICE_CH         = 2

# ─── Orchestrator timers ──────────────────────────────────
COMMENTARY_INTERVAL_S  = 9      # auto-commentary tick
CHAT_FLUSH_INTERVAL_S  = 30     # how often twitch_listener batches chat
TOPIC_IDLE_MIN_S       = 60     # minimum silence before topic injection
TOPIC_IDLE_MAX_S       = 120    # maximum silence before topic injection
VISION_CAPTURE_INTERVAL_S = 1   # screenshot cadence (1 s so a 10 s window can supply up to 10 frames)
SILENCE_THRESHOLD   = 20.0
SILENCE_POLL        = 2.0
VISION_STALE_WARN_S  = float(os.getenv("VISION_STALE_WARN_S",  "20"))
VISION_STALE_DROP_S  = float(os.getenv("VISION_STALE_DROP_S",  "60"))
STT_BUFFER_SILENCE_S = float(os.getenv("STT_BUFFER_SILENCE_S", "2.0"))
AUDIO_CONTEXT_SECONDS        = int(os.getenv("AUDIO_CONTEXT_SECONDS",        "20"))
VISION_PERCEIVE_INTERVAL_S   = float(os.getenv("VISION_PERCEIVE_INTERVAL_S", "2.5"))
BEN_BUFFER_WINDOW_S          = int(os.getenv("BEN_BUFFER_WINDOW_S",          "120"))

# ─── Scene perceiver (embedded in obs_bridge) ─────────────
VISION_FRAMES_MIN        = int(os.getenv("VISION_FRAMES_MIN",        "3"))
VISION_FRAMES_MAX        = int(os.getenv("VISION_FRAMES_MAX",        "20"))
VISION_MOTION_THRESHOLD  = float(os.getenv("VISION_MOTION_THRESHOLD", "8.0"))  # mean abs 32×32 grayscale delta (0-255)

# ─── Orchestrator thresholds ──────────────────────────────
MIN_SPEECH_SAMPLES     = 8000   # < 0.5 s at 16 kHz → skip
VISION_BUFFER_SIZE     = 30     # ring buffer depth (~30s at 1s capture; ≥ 2× WINDOW_MAX_S for merged windows)
CHAT_SUMMARY_KEEP      = 3      # number of rolling summaries to retain
CHAT_WINDOW_LINES      = 30     # raw lines fed into each summary

# ─── Embedding + Memory ───────────────────────────────────
EMBED_URL          = os.getenv("EMBED_URL",          "http://172.17.176.1:1234/v1/embeddings")
EMBED_MODEL        = os.getenv("EMBED_MODEL",        "")
MEMORY_DB_PATH     = os.getenv("MEMORY_DB_PATH",     "./rachel_memory_db")
TRAINING_DATA_DIR  = os.getenv("TRAINING_DATA_DIR",  "./training_data")
SESSION_LOG_DIR    = os.getenv("SESSION_LOG_DIR",    "./session_log")
SESSION_MAX_TURNS  = int(os.getenv("SESSION_MAX_TURNS", "20"))

# ─── Adaptive Loop: reaction path ─────────────────────────
# REACTION_MODE: "frames" = raw frames + speech + chat → one call (default)
#                "text"   = legacy [VISION]-description text path (A/B, rollback)
REACTION_MODE      = os.getenv("REACTION_MODE", "frames")
# Playback sequencer (playback.py) — see "Response Parser + Playback Sequencer" spec in CLAUDE.md
LEAD_S             = float(os.getenv("LEAD_S", "1.0"))               # speech after an action beat waits ≥ this
EXPR_RESET         = os.getenv("EXPR_RESET", "relaxed")             # Warudo expression after a reply that set one
EXPR_RESET_DELAY_S = float(os.getenv("EXPR_RESET_DELAY_S", "0"))   # cancelled if a new reply starts first
# REACTION_TRANSPORT: "completion" = exact prompt via llama.cpp /completion
#                     "chat"       = /chat/completions fallback (server applies its template)
#                     ""           = auto: chat with --direct (HF endpoint), completion locally
REACTION_TRANSPORT = os.getenv("REACTION_TRANSPORT", "")
PERCEIVER_ENABLED  = os.getenv("PERCEIVER_ENABLED", "false").lower() in ("1", "true", "yes")

# Window flush policy (open-decision defaults — change only with a comment)
WINDOW_S          = float(os.getenv("WINDOW_S",     "10"))   # flush once elapsed AND VAD idle
WINDOW_MAX_S      = float(os.getenv("WINDOW_MAX_S", "14"))   # hard max, flush even mid-utterance
WINDOW_MIN_S      = float(os.getenv("WINDOW_MIN_S", "3"))    # earliest flush on direct chat/event
WINDOW_FRAMES_MIN = int(os.getenv("WINDOW_FRAMES_MIN", "4"))
WINDOW_FRAMES_MAX = int(os.getenv("WINDOW_FRAMES_MAX", "10"))
WINDOW_FRAME_WIDTH = int(os.getenv("WINDOW_FRAME_WIDTH", "768"))   # downscale width for model frames
WINDOW_POLL_S     = 0.25
VAD_STALE_S       = float(os.getenv("VAD_STALE_S", "15"))   # no /vad update this long → assume idle
ECHO_SIMILARITY   = 0.80    # drop [ben] text this similar to Rachel's recent speech
REACTION_HISTORY_PAIRS = int(os.getenv("REACTION_HISTORY_PAIRS", "6"))

# Reaction sampling (Gemma 4 recommended)
REACTION_MAX_TOKENS = int(os.getenv("REACTION_MAX_TOKENS", "120"))
REACTION_SAMPLING   = {
    "temperature": float(os.getenv("REACTION_TEMPERATURE", "1.0")),
    "top_p":       float(os.getenv("REACTION_TOP_P",       "0.95")),
    "top_k":       int(os.getenv("REACTION_TOP_K",         "64")),
}

# ─── Paths ────────────────────────────────────────────────
REPO_ROOT       = Path(__file__).parent
PERSONA_DIR     = REPO_ROOT / "persona"
IMAGELOG_DIR    = REPO_ROOT / "imagelog"
ROWS_DIR        = REPO_ROOT / "rows"
CURATED_DIR     = REPO_ROOT / "curated"
DPO_DIR         = REPO_ROOT / "dpo"
ADAPTERS_DIR    = REPO_ROOT / "adapters"
ADAPTER_REGISTRY = ADAPTERS_DIR / "registry.json"
DATASETS_DIR    = REPO_ROOT / "datasets"
TRAINING_DATA_STATIC = REPO_ROOT / "training" / "data"   # core.jsonl (replay core), eval_set.jsonl (fixed)

# ─── Nightly adapter (local 4080, QLoRA E4B; 12B deferred) ─
# base_core: nightly LoRAs are trained on BASE_CORE_HF and served on top of
# BASE_CORE_GGUF — these MUST be the same weights (e.g. both base Gemma 4 E4B,
# or both the merged Rachel V6). A LoRA trained on one and served on the other
# is silently wrong.
BASE_CORE_HF    = os.getenv("BASE_CORE_HF",   r"F:\LLM\models\gemma-4-e4b")
BASE_CORE_GGUF  = os.getenv("BASE_CORE_GGUF", "")
BASE_CORE_MMPROJ = os.getenv("BASE_CORE_MMPROJ", r"F:\LLM\vtubeModels\rachel-v6\mmproj-F32.gguf")
LLAMACPP_DIR    = Path(os.getenv("LLAMACPP_DIR", r"F:\LLM\llama.cpp"))
TOPICS_PATH     = REPO_ROOT / "topics.json"
VOICE_REF_PATH  = Path(os.getenv("VOICE_REF_PATH", r"F:\LLM\vtubeModels\rachael_v2.mp3"))
STT_MUTE_LOCK   = Path(tempfile.gettempdir()) / "rachel_speaking.lock"
WORKSPACE_DIR   = REPO_ROOT / "workspace"
