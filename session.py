"""
session.py — Conversation session manager for the Rachel V5 pipeline.

Layer structure sent on every HF endpoint call:
  Layer 0   system        SOUL.md + USER.md — permanent, never changes
  Layer 1   system        <<STREAM_CONTEXT>> — live stream state (rebuilt every turn)
  Layer 2   user          "[Earlier this stream: ...]" — Rachel's voice summary
  Layer 3   user          [images] — vision screenshots with descriptive prefix
  Layer 4   user/asst     Recent conversation history
                            user turns are prefixed: [Ben]: or [Chat]:
                            so Rachel knows who is talking
"""

import re
import threading
import logging

log = logging.getLogger(__name__)

_CHANNEL_CLEAN = re.compile(
    r'(<\|channel>.*?<channel\|>|\[[a-z]+/[a-z]+\][^\]]*)',
    re.DOTALL | re.IGNORECASE,
)


class SessionManager:
    def __init__(self, system_prompt: str, max_pairs: int = 10):
        self.system_prompt  = system_prompt
        self.max_pairs      = max_pairs
        self.history:          list[dict] = []
        self.session_summary:  str        = ""
        self._lock         = threading.Lock()
        self._summary_lock = threading.Lock()

    # ── History mutations ─────────────────────────────────

    def add_user(self, content: str, source: str = "ben"):
        """
        Add a user turn to history.
        source: 'ben'   — Ben speaking via mic (STT)
                'chat'  — Twitch/YouTube chat message via obs_bridge
                'timer' — auto-commentary prompt (no speaker prefix)
        """
        if source == "ben":
            formatted = f"[Ben]: {content}"
        elif source == "chat":
            formatted = f"[Chat]: {content}"
        else:
            formatted = content
        with self._lock:
            self.history.append({"role": "user", "content": formatted})

    def add_assistant(
        self,
        spoken_text: str,
        channel_text: str = "",
        emotion: str = "neutral",
        intensity: str = "medium",
        user_input: str = "",
    ):
        with self._lock:
            if channel_text:
                history_content = f"<|channel> [EMOTION: {emotion}] [INTENSITY: {intensity}]\n{channel_text}\n<channel|>{spoken_text}"
            else:
                history_content = f"<|channel> [EMOTION: {emotion}] [INTENSITY: {intensity}]\n<channel|>{spoken_text}"
            self.history.append({"role": "assistant", "content": history_content})

    def needs_summary(self) -> bool:
        with self._lock:
            return sum(1 for m in self.history if m["role"] == "assistant") > self.max_pairs

    # ── Message assembly ──────────────────────────────────

    @staticmethod
    def _build_vision_content(vision_frames: list[dict]) -> list:
        """
        Build the multipart vision content array with per-frame temporal labels.
        Frames must be in chronological order (oldest first).
        Each frame dict: {"b64": str, "seconds_ago": float, ...}
        """
        n = len(vision_frames)
        parts = []
        parts.append({"type": "text", "text":
            f"[STREAM — {n} frame{'s' if n > 1 else ''}, chronological, oldest to newest]"})
        for i, frame in enumerate(vision_frames):
            sec = round(frame.get("seconds_ago", 0))
            label = f"frame-{i+1}-now" if sec <= 1 else f"frame-{i+1}-{sec}s-ago"
            parts.append({"type": "text", "text": label})
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame['b64']}"}})
        return parts

    def build_messages(
        self,
        context_block: str,
        vision_frames: list[dict] | None = None,
    ) -> list[dict]:
        """
        Assemble the full message array for the HF endpoint.

        Args:
            context_block: Layer 2a text (stream state, chat, memories — no vision).
            vision_frames: Layer 2b — chronological frame dicts [{b64, seconds_ago, ...}],
                           oldest first. Labeled with frame-N-Xs-ago / frame-N-now inline.
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        with self._summary_lock:
            summary = self.session_summary
        if summary:
            messages.append({"role": "user", "content": f"[Earlier this stream: {summary}]"})

        if context_block.strip():
            messages.append({
                "role": "system",
                "content": f"<<STREAM_CONTEXT>>\n{context_block}\n<</STREAM_CONTEXT>>",
            })

        if vision_frames:
            messages.append({"role": "user", "content": self._build_vision_content(vision_frames)})

        with self._lock:
            messages.extend(list(self.history))

        return messages

    def build_messages_with_audio(
        self,
        context_block: str,
        audio_b64: str,
        vision_frames: list[dict] | None = None,
    ) -> list[dict]:
        """Same as build_messages but appends the audio content block as Layer 4."""
        with self._lock:
            history_copy = list(self.history)
        with self._summary_lock:
            summary = self.session_summary

        messages = [{"role": "system", "content": self.system_prompt}]

        if summary:
            messages.append({"role": "user", "content": f"[Earlier this stream: {summary}]"})

        if context_block.strip():
            messages.append({
                "role": "system",
                "content": f"<<STREAM_CONTEXT>>\n{context_block}\n<</STREAM_CONTEXT>>",
            })

        if vision_frames:
            messages.append({"role": "user", "content": self._build_vision_content(vision_frames)})

        messages.extend(history_copy)
        messages.append({
            "role": "user",
            "content": [{"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}],
        })
        return messages

    # ── Async summary ─────────────────────────────────────

    def trigger_async_summary(self, llm_call_fn, on_complete=None):
        """
        Generate a session summary in a background thread (call AFTER TTS plays).

        llm_call_fn(messages: list[dict]) -> str | None
            Must return the raw text from the LLM, or None on failure.
        """
        def _do_summary():
            try:
                # Atomically check count and capture the split point
                with self._lock:
                    pair_count = sum(1 for m in self.history if m["role"] == "assistant")
                    if pair_count <= self.max_pairs:
                        return  # already handled

                    # Walk backwards to find where to split
                    kept = 0
                    split_idx = len(self.history)
                    for i in range(len(self.history) - 1, -1, -1):
                        if self.history[i]["role"] == "assistant":
                            kept += 1
                            if kept >= self.max_pairs:
                                split_idx = i
                                if i > 0 and self.history[i - 1]["role"] == "user":
                                    split_idx = i - 1
                                break

                    old_messages = list(self.history[:split_idx])

                with self._summary_lock:
                    existing_summary = self.session_summary

                # Build summary prompt
                parts: list[str] = []
                if existing_summary:
                    parts.append(f"Previous summary: {existing_summary}")
                parts.append("New conversation to summarize:")
                for msg in old_messages:
                    if msg["role"] == "assistant":
                        role, text = "Rachel", (msg["content"] if isinstance(msg["content"], str) else "[voice input]")
                    elif isinstance(msg["content"], str):
                        text = msg["content"]
                        if text.startswith("<<STREAM_CONTEXT>>") or text.startswith("[Earlier"):
                            continue
                        # Extract speaker label from [Ben]: or [Chat]: prefix
                        _m = re.match(r"^\[(Ben|Chat)\]:\s*(.*)", text, re.DOTALL)
                        if _m:
                            role, text = _m.group(1), _m.group(2)
                        else:
                            role = "Ben"
                    else:
                        continue  # image/audio content — skip
                    parts.append(f"  {role}: {text[:150]}")

                summary_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a concise summarizer. Output only the requested summary as plain text. "
                            "Do not use a thinking channel, emotion tags, expression/animation calls, "
                            "or any special tokens."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Summarize what actually happened in this stream session so far in 2-3 sentences. "
                            "Write in first person as Rachel. Only include things that were actually said or seen — "
                            "do not invent events, games, or moments that are not in the conversation above. "
                            "If very little has happened, keep it short and honest. Keep it under 100 words.\n\n"
                            + "\n".join(parts)
                        ),
                    },
                ]

                summary_text = llm_call_fn(summary_messages)

                if summary_text:
                    summary_text = _CHANNEL_CLEAN.sub("", summary_text).strip().strip('"')

                    with self._summary_lock:
                        self.session_summary = summary_text

                    # Remove only the old messages; new turns added during LLM call are preserved
                    with self._lock:
                        del self.history[:split_idx]

                    log.info(f"[summary] {summary_text[:100]}...")

                if on_complete:
                    on_complete()

            except Exception as e:
                log.error(f"Summary generation failed: {e}", exc_info=True)

        threading.Thread(target=_do_summary, name="summary-gen", daemon=True).start()

    # ── Lifecycle ─────────────────────────────────────────

    def reset(self):
        with self._lock:
            self.history.clear()
        with self._summary_lock:
            self.session_summary = ""

    def turn_count(self) -> int:
        with self._lock:
            return sum(1 for m in self.history if m["role"] == "assistant")
