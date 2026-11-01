"""
Rachel STT Server  (V5 — listen only)
──────────────────────────────────────
Mic → Silero VAD → Gemma 4 E4B (audio-direct STT)
  → POST http://localhost:8052/transcription  (orchestrator.py)

All LLM calls, TTS, and Warudo control are handled by orchestrator.py + server.py.

Install:
    pip install silero-vad pyaudio numpy soundfile requests
Run:
    python stt-server.py
    (orchestrator.py and server.py must already be running)
"""

import os
import site

# ── DLL PATH INJECTION (needed for torch/CUDA on Windows) ─
# Scan every nvidia sub-package's bin/ folder so torchaudio and all
# CUDA libs (cublas, cudnn, cusparse, cusolver, nvJitLink, …) can resolve.
_site_paths = site.getsitepackages()
for _p in _site_paths:
    _nvidia_root = os.path.join(_p, "nvidia")
    if not os.path.isdir(_nvidia_root):
        continue
    for _lib in os.listdir(_nvidia_root):
        _dll = os.path.join(_nvidia_root, _lib, "bin")
        if os.path.isdir(_dll):
            os.add_dll_directory(_dll)
            os.environ["PATH"] = _dll + os.pathsep + os.environ.get("PATH", "")
            print(f"✅ Linked {_lib} DLLs from: {_dll}")

import argparse
import base64
import io
import threading
import time
from collections import deque

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import torch
import pyaudio
from flask import Flask, jsonify
from scipy.signal import butter, sosfilt, iirfilter
from silero_vad import load_silero_vad, VADIterator

from audio_devices import list_input_devices, rank_mic_candidates

from config import (
    MIC_KEYWORD, SAMPLE_RATE, CHUNK_SIZE, DEVICE_RATE, DEVICE_CH,
    MIN_SPEECH_SAMPLES, STT_MUTE_LOCK, ORCHESTRATOR_URL,
    STT_SERVER_PORT, AUDIO_CONTEXT_SECONDS, SPEAKER_DEVICE_NAME,
    STT_BUFFER_SILENCE_S,
    HF_URL, HF_TOKEN, HF_MODEL,
)

STT_URL = "http://localhost:8080/v1/chat/completions"
STT_MODEL = "gemma-4-e4b"

_offload_stt_url: str | None = None


# ─── RNNoise denoising ───────────────────────────────────

_RNNOISE_RATE   = 48_000
_RNNOISE_FRAME  = 480   # 10 ms at 48 kHz

_denoiser = None


def _init_rnnoise():
    global _denoiser
    try:
        from pyrnnoise import RNNoise
        _denoiser = RNNoise(sample_rate=_RNNOISE_RATE)
        print("  🧠 RNNoise ready")
    except Exception as e:
        print(f"  ⚠️  RNNoise unavailable: {e} — running without noise suppression")


def _denoise(audio_16k: np.ndarray) -> np.ndarray:
    """
    Apply RNNoise to a float32 16 kHz mono array.
    Resamples to 48 kHz, processes 480-sample frames, resamples back.
    Returns float32 16 kHz.
    """
    if _denoiser is None:
        return audio_16k

    # 16 kHz → 48 kHz
    n_48k = int(len(audio_16k) * _RNNOISE_RATE / SAMPLE_RATE)
    audio_48k = np.interp(
        np.linspace(0, len(audio_16k), n_48k, endpoint=False),
        np.arange(len(audio_16k)),
        audio_16k,
    )

    # float32 → int16, pad to multiple of _RNNOISE_FRAME
    audio_int16 = (np.clip(audio_48k, -1.0, 1.0) * 32767).astype(np.int16)
    original_len = len(audio_int16)
    pad = (-original_len) % _RNNOISE_FRAME
    if pad:
        audio_int16 = np.pad(audio_int16, (0, pad))

    # Process frame by frame — denoise_chunk expects [channels, samples]
    denoised_frames: list[np.ndarray] = []
    for i in range(0, len(audio_int16), _RNNOISE_FRAME):
        frame = np.expand_dims(audio_int16[i:i + _RNNOISE_FRAME], axis=0)
        for _vad_prob, denoised in _denoiser.denoise_chunk(frame):
            denoised_frames.append(np.squeeze(denoised))

    denoised_48k = np.concatenate(denoised_frames)[:original_len]

    # int16 or float32 output — normalise to float32 either way
    if denoised_48k.dtype == np.int16:
        denoised_f32 = denoised_48k.astype(np.float32) / 32768.0
    else:
        denoised_f32 = denoised_48k.astype(np.float32)

    # 48 kHz → 16 kHz
    return np.interp(
        np.linspace(0, len(denoised_f32), len(audio_16k), endpoint=False),
        np.arange(len(denoised_f32)),
        denoised_f32,
    ).astype(np.float32)


# ─── EQ / audio enhancement ──────────────────────────────

class ExpertStreamerEqualizer:
    """
    Razer Seiren Mini parametric EQ + gain — stateful per-chunk, 16 kHz mono.
    Applied before VAD so pre-roll frames are also processed.

    Chain (cascaded SOS, float32 throughout):
      Band 1  HPF         75 Hz      2nd-order Butterworth   (sub-bass cleanse)
      Band 2  Bell notch  220 Hz     Chebyshev II  -4.5 dB   (acoustic mud)
      Band 3  Bell notch  450 Hz     Chebyshev II  -2.0 dB   (plastic resonance)
      Band 4  Bell notch  3.2 kHz    Chebyshev II  -3.0 dB   (sibilance / fatigue)
      Band 5  Shelf boost 6.5 kHz    parallel HPF  +3.5 dB   (air / presence)
      Gain    linear multiplier applied after EQ, before clip
    """

    def __init__(self, sample_rate: int = 16000, gain_db: float = 6.0):
        self.sample_rate = sample_rate
        self.gain_linear = 10 ** (gain_db / 20)
        self.filters: list  = []
        self.states:  list  = []
        self._compile_chain()

    def _compile_chain(self):
        fs = self.sample_rate

        # Band 1 — HPF 75 Hz
        self._add(butter(N=2, Wn=75, btype="highpass", fs=fs, output="sos"))
        # Band 2 — mud notch 220 Hz
        self._add(iirfilter(N=2, Wn=[190, 255], rp=1, rs=4.5,
                            btype="bandstop", ftype="cheby2", fs=fs, output="sos"))
        # Band 3 — box notch 450 Hz
        self._add(iirfilter(N=2, Wn=[380, 530], rp=1, rs=2.0,
                            btype="bandstop", ftype="cheby2", fs=fs, output="sos"))
        # Band 4 — sibilance notch 3.2 kHz
        self._add(iirfilter(N=2, Wn=[2950, 3450], rp=1, rs=3.0,
                            btype="bandstop", ftype="cheby2", fs=fs, output="sos"))

        # Band 5 — presence shelf 6.5 kHz (parallel path)
        self.sos_shelf   = butter(N=2, Wn=6500, btype="highpass", fs=fs, output="sos")
        self.shelf_state = np.zeros((self.sos_shelf.shape[0], 2), dtype=np.float32)

    def _add(self, sos):
        self.filters.append(sos)
        self.states.append(np.zeros((sos.shape[0], 2), dtype=np.float32))

    def process_chunk(self, audio: np.ndarray) -> np.ndarray:
        out = audio.astype(np.float32)

        for i, sos in enumerate(self.filters):
            out, new_zi = sosfilt(sos, out, zi=self.states[i])
            self.states[i] = new_zi.astype(np.float32)

        highs, new_shelf_zi = sosfilt(self.sos_shelf, out, zi=self.shelf_state)
        self.shelf_state = new_shelf_zi.astype(np.float32)

        # Parallel shelf injection (+3.5 dB ≈ ×0.5 relative to main)
        out = out + highs.astype(np.float32) * 0.5

        # Input gain (volume boost)
        out = out * self.gain_linear

        return np.clip(out, -1.0, 1.0).astype(np.float32)


_eq_processor: ExpertStreamerEqualizer | None = None


def _init_eq(gain_db: float = 6.0):
    global _eq_processor
    try:
        _eq_processor = ExpertStreamerEqualizer(sample_rate=SAMPLE_RATE, gain_db=gain_db)
        print(f"  🎛️  EQ ready (Seiren Mini expert chain, gain={gain_db:+.1f} dB)")
    except Exception as e:
        print(f"  ⚠️  EQ init failed: {e} — running flat")


# ─── Audio playback (loopback) ───────────────────────────

_playback_device: int | None = None


def _init_playback_device():
    global _playback_device
    for i, dev in enumerate(sd.query_devices()):
        name = dev["name"]
        if dev["max_output_channels"] > 0 and "realtek" in name.lower():
            _playback_device = i
            print(f"  🔊 Playback device: [{i}] {name}")
            return
    print("  ⚠️  No Realtek playback device found — loopback disabled")


def _play_audio(audio: np.ndarray):
    if _playback_device is None:
        return
    try:
        sd.play(audio, samplerate=SAMPLE_RATE, device=_playback_device, blocking=False)
    except Exception as e:
        print(f"  ⚠️  Playback error: {e}")


# ─── Audio context ring buffers ──────────────────────────

_RING_CHUNKS: int = int(AUDIO_CONTEXT_SECONDS * SAMPLE_RATE / CHUNK_SIZE) + 5
_mic_ring: deque = deque(maxlen=_RING_CHUNKS)   # float32 16kHz mono chunks from mic
_spk_ring: deque = deque(maxlen=_RING_CHUNKS)   # float32 16kHz mono chunks from speaker loopback
_no_loopback: bool = False                        # set from --no-loopback CLI flag

# ─── STT local buffer (silence-gated flush) ───────────────
_stt_local_buffer:   list[str] = []
_stt_last_segment_ts: float    = 0.0
_stt_buf_lock = threading.Lock()


# Speaker loopback capture removed — obs_bridge.py is the sole loopback reader.
# stt-server's _spk_ring stays empty; /audio-context returns mic-only audio.


# ─── Audio-context Flask endpoint ────────────────────────

_ctx_app = Flask(__name__ + ".ctx")
_ctx_app.logger.setLevel("WARNING")


@_ctx_app.get("/audio-context")
def audio_context():
    mic_chunks = list(_mic_ring)
    spk_chunks = list(_spk_ring)
    if not mic_chunks and not spk_chunks:
        return jsonify({"error": "no audio captured yet"}), 204

    mic = np.concatenate(mic_chunks) if mic_chunks else np.array([], dtype=np.float32)
    spk = np.concatenate(spk_chunks) if spk_chunks else np.array([], dtype=np.float32)

    # Align by zero-padding the shorter array at the front (temporal alignment)
    n = max(len(mic), len(spk))
    if len(mic) < n:
        mic = np.concatenate([np.zeros(n - len(mic), dtype=np.float32), mic])
    if len(spk) < n:
        spk = np.concatenate([np.zeros(n - len(spk), dtype=np.float32), spk])

    mixed = np.clip(mic + spk, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, mixed, SAMPLE_RATE, format="wav")
    buf.seek(0)
    audio_b64 = base64.b64encode(buf.read()).decode()
    return jsonify({"audio_b64": audio_b64, "format": "wav",
                    "seconds": round(len(mixed) / SAMPLE_RATE, 1)})


# ─── Mic discovery ────────────────────────────────────────

def find_mic_candidates(keyword: str, preferred_rate: int = DEVICE_RATE) -> list:
    """
    Return all input devices matching keyword, sorted by preference:
      1. starts-with-"Microphone" + preferred_rate  (clean_match)
      2. preferred_rate                              (preferred_match)
      3. starts-with-"Microphone"                   (plain_match)
      4. any match                                   (first_match)
    Each entry is (index, rate, channels, name).
    No device is opened here — caller tries each in order.
    """
    p = pyaudio.PyAudio()
    print("\n  Available input devices:")
    devices = list_input_devices(p)
    p.terminate()
    for d in devices:
        tag = "◀ KEYWORD MATCH" if keyword.lower() in d["name"].lower() else ""
        print(f"    [{d['index']:2d}] {d['name']}  ch={d['channels']}  sr={d['rate']}  {tag}")

    candidates = rank_mic_candidates(devices, keyword, preferred_rate)
    if not candidates:
        print(f"\n⚠️  No device matched keyword '{keyword}' — will use system default")
        candidates = [(None, preferred_rate, 1, "system default")]
    return candidates


# ─── STT ─────────────────────────────────────────────────

def transcribe(audio_b64: str) -> str | None:
    """Send base64 WAV to STT; return transcription or None."""
    if _offload_stt_url:
        try:
            r = requests.post(_offload_stt_url, json={"audio_b64": audio_b64, "format": "wav"}, timeout=100)
            if r.status_code != 200:
                print(f"   🔴 Offload STT error: {r.status_code}")
                return None
            text = r.json().get("text", "").strip()
            return text if len(text) >= 2 else None
        except Exception as e:
            print(f"   🔴 Offload STT error: {e}")
            return None

    try:
        r = requests.post(STT_URL, json={
            "model": STT_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                    {"type": "text", "text": "Transcribe this audio exactly. Output only the spoken words, nothing else."},
                ],
            }],
            "temperature": 0.1,
            "top_k": 10,
        }, timeout=100)

        if r.status_code != 200:
            print(f"   🔴 STT error: {r.status_code}")
            return None

        text = r.json()["choices"][0]["message"]["content"].strip()
        return text if len(text) >= 2 else None

    except Exception as e:
        print(f"   🔴 STT error: {e}")
        return None


def transcribe_hf(audio_b64: str) -> str | None:
    """Send base64 WAV to HF endpoint for STT; return transcription or None."""
    url = HF_URL.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url + "/chat/completions"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    try:
        r = requests.post(url, headers=headers, json={
            "model": HF_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                    {"type": "text", "text": "Transcribe this audio exactly. Output only the spoken words, nothing else."},
                ],
            }],
            "max_tokens": 200,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "thinking": {"type": "disabled"},
        }, timeout=60)
        if r.status_code != 200:
            print(f"   🔴 HF STT error: {r.status_code} {r.text[:120]}")
            return None
        text = r.json()["choices"][0]["message"]["content"].strip()
        return text if len(text) >= 2 else None
    except Exception as e:
        print(f"   🔴 HF STT error: {e}")
        return None


# ─── STT silence-gated flush thread ──────────────────────

def _flush_stt_buffer():
    """Send accumulated transcriptions to orchestrator after a silence gap."""
    while True:
        time.sleep(0.3)
        with _stt_buf_lock:
            if not _stt_local_buffer:
                continue
            if time.time() - _stt_last_segment_ts < STT_BUFFER_SILENCE_S:
                continue
            combined = " ".join(_stt_local_buffer)
            _stt_local_buffer.clear()
        print(f"\r\033[K[🎤 Ben (flush)]: {combined}  → /transcription")
        send_to_orchestrator(combined, respond=False)


# ─── Orchestrator relay ───────────────────────────────────

def send_to_orchestrator(text: str, respond: bool = False):
    """
    respond=False: send to /transcription (passive buffer).
    respond=True:  send ONLY to /chat-input — the handler buffers with reacted=True
                   and enqueues an immediate reaction. No /transcription post, so
                   the utterance is never picked up again by the silence timer.
    """
    if respond:
        try:
            requests.post(
                f"{ORCHESTRATOR_URL}/chat-input",
                json={"user": "Ben", "text": text},
                timeout=5,
            )
        except Exception as e:
            print(f"   🔴 Orchestrator chat-input relay error: {e}")
    else:
        try:
            requests.post(
                f"{ORCHESTRATOR_URL}/transcription",
                json={"text": text},
                timeout=5,
            )
        except Exception as e:
            print(f"   🔴 Orchestrator relay error: {e}")


def notify_vad(speaking: bool):
    """Fire-and-forget VAD state for the orchestrator's window flush (don't cut Ben mid-utterance)."""
    def _post():
        try:
            requests.post(f"{ORCHESTRATOR_URL}/vad", json={"speaking": speaking, "t": time.time()}, timeout=1)
        except Exception:
            pass
    threading.Thread(target=_post, daemon=True).start()


def send_audio_to_orchestrator(audio_b64: str, transcription: str = "") -> bool:
    """Send raw audio b64 to orchestrator /audio-input (--direct mode).
    Returns True on HTTP 200, False otherwise."""
    try:
        r = requests.post(
            f"{ORCHESTRATOR_URL}/audio-input",
            json={"audio_b64": audio_b64, "format": "wav", "transcription": transcription},
            timeout=5,
        )
        if r.status_code != 200:
            print(f"   🔴 /audio-input HTTP {r.status_code}: {r.text[:120]}")
            return False
        return True
    except Exception as e:
        print(f"   🔴 Orchestrator audio relay error: {e}")
        return False


# ─── Main loop ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rachel STT Server")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Skip local STT — send raw audio to orchestrator /audio-input for HF multimodal transcription",
    )
    parser.add_argument(
        "--eq",
        action="store_true",
        help="Enable Seiren Mini expert EQ chain (HPF + notches + presence shelf + gain)",
    )
    parser.add_argument(
        "--eq-gain",
        type=float,
        default=0.0,
        metavar="DB",
        help="Gain in dB applied inside the EQ chain (default: 0.0).",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=0.0,
        metavar="DB",
        help="Simple volume boost in dB, independent of EQ (default: 0.0, e.g. --gain 6)",
    )
    parser.add_argument(
        "--offload",
        metavar="URL",
        default=None,
        help="Offload STT to external endpoint. POST {audio_b64, format} → {text}. Skips local Gemma STT.",
    )
    parser.add_argument(
        "--no-loopback",
        action="store_true",
        help="Disable WASAPI speaker loopback capture (audio-context will contain mic only).",
    )
    parser.add_argument(
        "--loopback-device",
        metavar="NAME",
        default=None,
        help=(
            "Keyword to select the WASAPI output device for loopback capture "
            "(e.g. 'Realtek', 'Speakers'). "
            f"Defaults to SPEAKER_DEVICE_NAME env var ('{SPEAKER_DEVICE_NAME}'), "
            "then the WASAPI default output."
        ),
    )
    parser.add_argument(
        "--respond",
        action="store_true",
        help=(
            "Send transcriptions to /chat-input (triggers an immediate Rachel response) "
            "instead of /transcription (passive buffer). Use this when you want Rachel to "
            "reply to what you say in real time."
        ),
    )
    args = parser.parse_args()
    direct_mode  = args.direct
    respond_mode = args.respond

    global _offload_stt_url, _no_loopback
    _offload_stt_url = args.offload
    _no_loopback     = args.no_loopback

    # ── Audio-context Flask server ─────────────────────────
    def _run_ctx_server():
        _ctx_app.run(host="0.0.0.0", port=STT_SERVER_PORT, threaded=True, use_reloader=False)
    threading.Thread(target=_run_ctx_server, name="ctx-server", daemon=True).start()
    print(f"  🌐 Audio-context server → http://localhost:{STT_SERVER_PORT}/audio-context")

    print("Loading Silero VAD...")
    vad_model = load_silero_vad()
    vad = VADIterator(
        vad_model,
        threshold=0.3,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=800,
        speech_pad_ms=150,
    )
    print("✅ VAD ready.")

    _init_rnnoise()
    _init_playback_device()
    if args.eq:
        _init_eq(gain_db=args.eq_gain)
    gain_linear = 10 ** (args.gain / 20) if args.gain else 1.0

    # Clear any stale mute lock from a previous crashed run
    if STT_MUTE_LOCK.exists():
        STT_MUTE_LOCK.unlink()
        print(f"  ⚠️  Cleared stale mute lock: {STT_MUTE_LOCK}")

    candidates = find_mic_candidates(MIC_KEYWORD)
    p = pyaudio.PyAudio()
    stream = mic_index = mic_rate = mic_ch = None
    for idx, rate, ch, name in candidates:
        native_chunk = int(round(CHUNK_SIZE * rate / SAMPLE_RATE))
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=ch,
                rate=rate,
                input=True,
                frames_per_buffer=native_chunk,
                input_device_index=idx,
            )
            mic_index, mic_rate, mic_ch = idx, rate, ch
            print(f"\n✅ Mic: [{idx}] {name}  ch={ch}  sr={rate} Hz")
            break
        except Exception as e:
            print(f"  ✗ [{idx}] {name}: {e} — trying next candidate")
    if stream is None:
        print("🔴 Could not open any matching input device.")
        p.terminate()
        return

    native_chunk = int(round(CHUNK_SIZE * mic_rate / SAMPLE_RATE))

    # Speaker loopback is now owned by obs_bridge.py (AEC reference alignment).
    # stt-server no longer opens a loopback device; /audio-context is mic-only.

    threading.Thread(target=_flush_stt_buffer, name="stt-flush", daemon=True).start()

    print(f"\n{'='*55}")
    flags = "".join([
        "  [--direct]"  if direct_mode       else "",
        "  [--offload]" if _offload_stt_url  else "",
        "  [--respond]" if respond_mode      else "",
    ])
    print(f"  Rachel STT  (V5){flags}")
    if direct_mode:
        print(f"  STT:   HF multimodal @ {HF_URL}")
        relay_ep = "/chat-input (respond)" if respond_mode else "/transcription (buffered)"
        print(f"  Relay: {ORCHESTRATOR_URL}{relay_ep}")
    elif _offload_stt_url:
        print(f"  STT:   Offload → {_offload_stt_url}")
        relay_ep = "/chat-input (respond mode)" if respond_mode else "/transcription"
        print(f"  Relay: {ORCHESTRATOR_URL}{relay_ep}")
    else:
        print(f"  STT:   Gemma 4 E4B @ {STT_URL}")
        relay_ep = "/chat-input (respond mode)" if respond_mode else "/transcription"
        print(f"  Relay: {ORCHESTRATOR_URL}{relay_ep}")
    # Show which PortAudio host API is backing this device
    _pa_info = pyaudio.PyAudio()
    _dev_info = _pa_info.get_device_info_by_index(mic_index) if mic_index is not None else {}
    _host_api = _pa_info.get_host_api_info_by_index(_dev_info.get("hostApi", 0))
    _pa_info.terminate()
    print(f"  Mic:   [{mic_index}] ch={mic_ch} sr={mic_rate}")
    print(f"  API:   [{_dev_info.get('hostApi')}] {_host_api.get('name')}  (structVersion={_host_api.get('structVersion')})")
    print(f"  Chunk: native={native_chunk} frames  →  16k={CHUNK_SIZE} frames  (ratio={mic_rate/SAMPLE_RATE:.4f}x)")
    print(f"  VAD:   threshold=0.3  min_silence=800ms")
    print(f"  Spk:   loopback owned by obs_bridge (AEC active)")
    print(f"{'='*55}\n")
    print("🎤 Listening… (speak now — watch level bar)\n")

    speech_chunks: list = []
    in_speech = False
    # Ring buffer of recent chunks — seeded into speech_chunks on VAD start
    # so the first ~640 ms of speech isn't lost to VAD detection lag.
    # 20 chunks × 32 ms/chunk (512 samples @ 16 kHz) = 640 ms pre-roll
    pre_roll: deque = deque(maxlen=20)
    # Speech segments captured while Rachel was talking — sent immediately on unmute
    queued_audio: list = []

    def _encode_and_send(audio: np.ndarray, label: str = "Ben"):
        global _stt_last_segment_ts
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="wav")
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode()
        text = transcribe_hf(audio_b64) if direct_mode else transcribe(audio_b64)
        if text:
            if respond_mode:
                print(f"\r\033[K[🎤 {label}]: {text}  → /chat-input")
                send_to_orchestrator(text, respond=True)
            else:
                with _stt_buf_lock:
                    _stt_local_buffer.append(text)
                    _stt_last_segment_ts = time.time()
                print(f"\r\033[K[🎤 {label} (buffered)]: {text}")
        else:
            print(f"\r\033[K({label}: no speech detected)")

    try:
        while True:
            raw = stream.read(native_chunk, exception_on_overflow=False)

            # Always process audio so VAD runs in both muted and unmuted paths
            chunk_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if mic_ch > 1:
                chunk_np = chunk_np.reshape(-1, mic_ch).mean(axis=1)
            chunk_16k = np.interp(
                np.linspace(0, len(chunk_np), CHUNK_SIZE, endpoint=False),
                np.arange(len(chunk_np)),
                chunk_np,
            ).astype(np.float32)
            if _eq_processor is not None:
                chunk_16k = _eq_processor.process_chunk(chunk_16k)
            if gain_linear != 1.0:
                chunk_16k = np.clip(chunk_16k * gain_linear, -1.0, 1.0)
            chunk_t = torch.from_numpy(chunk_16k)

            _mic_ring.append(chunk_16k)
            pre_roll.append(chunk_16k)
            event = vad(chunk_t, return_seconds=False)
            if event and "start" in event:
                notify_vad(True)
            if event and "end" in event:
                notify_vad(False)

            if STT_MUTE_LOCK.exists():
                # Rachel is speaking — queue user speech for immediate send on unmute
                if event and "start" in event:
                    in_speech     = True
                    speech_chunks = list(pre_roll)
                elif in_speech:
                    speech_chunks.append(chunk_16k)
                if event and "end" in event and speech_chunks:
                    in_speech = False
                    audio = np.concatenate(speech_chunks)
                    speech_chunks = []
                    audio = _denoise(audio)
                    if len(audio) >= MIN_SPEECH_SAMPLES:
                        queued_audio.append(audio)
                        print("\r\033[K[🎤 queued — waiting for Rachel]")
                continue

            # Mute just lifted — drain any speech queued during Rachel's response
            while queued_audio:
                _encode_and_send(queued_audio.pop(0), label="Ben (queued)")

            # Normal speech accumulation
            if event and "start" in event:
                in_speech     = True
                speech_chunks = list(pre_roll)
            elif in_speech:
                speech_chunks.append(chunk_16k)

            if event and "end" in event and speech_chunks:
                in_speech = False

                audio = np.concatenate(speech_chunks)
                speech_chunks = []
                audio = _denoise(audio)

                if len(audio) < MIN_SPEECH_SAMPLES:
                    continue

                _play_audio(audio)
                _encode_and_send(audio, label="Ben")

    except KeyboardInterrupt:
        print("\n🛑 Shutting down.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
